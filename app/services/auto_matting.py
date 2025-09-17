"""
Auto Background Removal Service
Handles automatic segmentation, matting, and shadow generation for furniture images.
"""

import os
import io
import uuid
import hashlib
from typing import Tuple, Dict, Optional
from flask import Blueprint, request, abort, jsonify, current_app
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import requests
import rembg
import numpy as np
import cv2
from sqlalchemy.orm import Session
from app.database import SessionLocal, Image as ImageModel, Variant
from app.config import S3_BUCKET, S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
import boto3
from botocore.exceptions import ClientError

bp = Blueprint("auto_matting", __name__, url_prefix="/auto-matting")

# Configuration
TIMEOUT = 30
MAX_BYTES = 25 * 1024 * 1024  # 25MB
MAX_SIDE = 2000
CONFIDENCE_THRESHOLD = 0.7  # Below this, flag for manual review

# Initialize models
_YOLO_AVAILABLE = True
_YOLO_ERR = None
try:
    from ultralytics import YOLO
    model_path = "yolov8x-seg.pt"
    yolo_model = YOLO(model_path) if os.path.exists(model_path) else YOLO("yolov8x-seg.pt")
    print("[auto_matting] YOLO model loaded successfully")
except Exception as e:
    _YOLO_AVAILABLE = False
    _YOLO_ERR = e
    print(f"[auto_matting] YOLO unavailable: {e}")

# Initialize rembg sessions
REMBG_SESSION = rembg.new_session()
try:
    REMBG_HUMAN_SESSION = rembg.new_session("isnet-general-human-seg")
except ValueError:
    REMBG_HUMAN_SESSION = REMBG_SESSION  # Fallback to general session

# Initialize S3 client
try:
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name='us-east-1'
    )
    print("[auto_matting] S3 client initialized")
except Exception as e:
    print(f"[auto_matting] S3 client failed: {e}")
    s3_client = None


def _fetch_image(url: str) -> bytes:
    """Fetch image from URL with proper error handling."""
    headers = {"User-Agent": "designstream-auto-matting/1.0", "Accept": "image/*"}
    with requests.get(url, headers=headers, timeout=TIMEOUT, stream=True) as r:
        r.raise_for_status()
        cl = r.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BYTES:
            abort(413, f"image too large: {cl} > {MAX_BYTES}")
        buf, read = io.BytesIO(), 0
        for chunk in r.iter_content(64 * 1024):
            read += len(chunk)
            if read > MAX_BYTES:
                abort(413, f"image too large: streamed > {MAX_BYTES}")
            buf.write(chunk)
        return buf.getvalue()


def _preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image for better segmentation."""
    # Normalize orientation
    image = ImageOps.exif_transpose(image)
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if too large
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    
    return image


def _calculate_confidence_score(original: Image.Image, matted: Image.Image) -> float:
    """Calculate confidence score for the matting result."""
    try:
        # Convert to numpy arrays
        orig_array = np.array(original.convert('RGB'))
        matted_array = np.array(matted.convert('RGBA'))
        
        # Get alpha channel
        alpha = matted_array[:, :, 3]
        
        # Calculate metrics
        total_pixels = alpha.size
        transparent_pixels = np.sum(alpha < 128)
        opaque_pixels = np.sum(alpha > 128)
        
        # Edge quality assessment
        alpha_uint8 = alpha.astype(np.uint8)
        edges = cv2.Canny(alpha_uint8, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        
        # Alpha channel smoothness (less noise = higher confidence)
        alpha_blurred = cv2.GaussianBlur(alpha_uint8, (5, 5), 0)
        alpha_diff = np.abs(alpha_uint8.astype(float) - alpha_blurred.astype(float))
        smoothness = 1.0 - (np.mean(alpha_diff) / 255.0)
        
        # Object size ratio (reasonable furniture should be 10-80% of image)
        object_ratio = opaque_pixels / total_pixels
        size_score = 1.0 - abs(object_ratio - 0.4) / 0.4  # Peak at 40%
        size_score = max(0.0, min(1.0, size_score))
        
        # Combine metrics
        confidence = (
            smoothness * 0.4 +      # Smooth alpha channel
            size_score * 0.3 +      # Reasonable object size
            (1.0 - edge_density) * 0.2 +  # Not too many edges (clean cutout)
            (opaque_pixels / total_pixels) * 0.1  # Some object present
        )
        
        return min(1.0, max(0.0, confidence))
        
    except Exception as e:
        current_app.logger.warning(f"Confidence calculation failed: {e}")
        return 0.5  # Default medium confidence


def _generate_shadow(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """Generate a soft shadow under the object."""
    try:
        # Create shadow from alpha channel
        shadow_alpha = alpha.astype(np.float32) / 255.0
        
        # Apply Gaussian blur for soft shadow
        shadow_blurred = cv2.GaussianBlur(shadow_alpha, (21, 21), 0)
        
        # Offset shadow (slightly down and right)
        h, w = shadow_alpha.shape
        shadow_offset = np.zeros_like(shadow_blurred)
        offset_y, offset_x = 8, 4  # Shadow offset
        
        # Create offset shadow
        shadow_offset[offset_y:, offset_x:] = shadow_blurred[:-offset_y, :-offset_x]
        
        # Darken the shadow
        shadow_offset = shadow_offset * 0.6
        
        # Create RGBA shadow image
        shadow_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        shadow_rgba[:, :, 3] = (shadow_offset * 255).astype(np.uint8)
        
        # Convert to PIL Image
        shadow_image = Image.fromarray(shadow_rgba, 'RGBA')
        
        return shadow_image
        
    except Exception as e:
        current_app.logger.warning(f"Shadow generation failed: {e}")
        # Return transparent image if shadow generation fails
        return Image.new('RGBA', image.size, (0, 0, 0, 0))


def _segment_with_yolo(image: Image.Image) -> Tuple[Image.Image, float]:
    """Segment image using YOLO model."""
    try:
        # Run YOLO prediction
        results = yolo_model.predict(
            source=np.array(image), 
            imgsz=640, 
            conf=0.25, 
            verbose=False
        )[0]
        
        masks = getattr(results, "masks", None)
        if masks is None or masks.data is None:
            raise RuntimeError("No masks detected")
        
        # Get mask data
        mask_data = masks.data
        if hasattr(mask_data, "detach"):
            mask_data = mask_data.detach().cpu().numpy()
        else:
            mask_data = np.asarray(mask_data)
        
        # Union all instance masks
        if mask_data.ndim == 3:
            union_mask = mask_data.max(axis=0)
        elif mask_data.ndim == 2:
            union_mask = mask_data
        else:
            raise RuntimeError(f"Unexpected mask shape: {mask_data.shape}")
        
        # Convert to uint8 and resize
        mask_uint8 = (np.clip(union_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        mask_image = Image.fromarray(mask_uint8, mode="L").resize(image.size, Image.Resampling.NEAREST)
        
        # Apply mask to create RGBA image
        rgba_image = image.copy()
        rgba_image.putalpha(mask_image)
        
        # Calculate confidence based on mask quality
        confidence = _calculate_confidence_score(image, rgba_image)
        
        return rgba_image, confidence
        
    except Exception as e:
        current_app.logger.warning(f"YOLO segmentation failed: {e}")
        raise


def _segment_with_rembg(image: Image.Image, model_type: str = "general") -> Tuple[Image.Image, float]:
    """Segment image using rembg model."""
    try:
        # Choose session based on model type
        if model_type == "human":
            session = REMBG_HUMAN_SESSION
        else:
            session = REMBG_SESSION
        
        # Run rembg
        matted = rembg.remove(
            image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=140,
            alpha_matting_background_threshold=60,
            alpha_matting_erode_size=35,
        )
        
        # Ensure RGBA
        if matted.mode != "RGBA":
            matted = matted.convert("RGBA")
        
        # Calculate confidence
        confidence = _calculate_confidence_score(image, matted)
        
        return matted, confidence
        
    except Exception as e:
        current_app.logger.warning(f"rembg segmentation failed: {e}")
        raise


def _upload_to_s3(image: Image.Image, bucket: str, key: str) -> str:
    """Upload image to S3 and return URL."""
    if not s3_client:
        raise RuntimeError("S3 client not available")
    
    try:
        # Convert image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=img_buffer.getvalue(),
            ContentType='image/png',
            ACL='public-read'
        )
        
        # Return public URL
        return f"{S3_ENDPOINT_URL}/{bucket}/{key}"
        
    except ClientError as e:
        current_app.logger.error(f"S3 upload failed: {e}")
        raise


def _store_matting_result(original_url: str, cutout_url: str, shadow_url: str, 
                         confidence: float, needs_manual: bool) -> Dict:
    """Store matting result in database."""
    db = SessionLocal()
    try:
        # Create image record
        image_record = ImageModel(
            url=original_url,
            cutout_url=cutout_url,
            quality_score=confidence,
            # Store additional metadata in a JSON field if available
        )
        
        db.add(image_record)
        db.commit()
        
        return {
            "id": image_record.id,
            "original_url": original_url,
            "cutout_url": cutout_url,
            "shadow_url": shadow_url,
            "confidence": confidence,
            "needs_manual": needs_manual,
            "created_at": image_record.created_at.isoformat() if hasattr(image_record, 'created_at') else None
        }
        
    except Exception as e:
        db.rollback()
        current_app.logger.error(f"Database storage failed: {e}")
        raise
    finally:
        db.close()


@bp.route("/process", methods=["POST"])
def process_image():
    """
    Process furniture image for automatic background removal.
    
    POST /auto-matting/process
    {
        "image_url": "https://example.com/furniture.jpg",
        "model": "auto|yolo|rembg|human",
        "generate_shadow": true,
        "store_result": true
    }
    
    Returns:
    {
        "success": true,
        "cutout_url": "https://s3.../cutout.png",
        "shadow_url": "https://s3.../shadow.png",
        "confidence": 0.85,
        "needs_manual": false,
        "processing_time": 2.3
    }
    """
    import time
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            abort(400, "JSON data required")
        
        image_url = data.get("image_url")
        model_type = data.get("model", "auto").lower()
        generate_shadow = data.get("generate_shadow", True)
        store_result = data.get("store_result", True)
        
        if not image_url:
            abort(400, "image_url is required")
        
        # Fetch and preprocess image
        current_app.logger.info(f"Processing image: {image_url}")
        img_bytes = _fetch_image(image_url)
        original_image = Image.open(io.BytesIO(img_bytes))
        processed_image = _preprocess_image(original_image)
        
        # Generate unique identifiers
        image_hash = hashlib.md5(img_bytes).hexdigest()
        cutout_key = f"cutouts/{image_hash}_cutout.png"
        shadow_key = f"shadows/{image_hash}_shadow.png"
        
        # Perform segmentation
        confidence = 0.0
        matted_image = None
        
        if model_type == "auto":
            # Try YOLO first, fallback to rembg
            try:
                if _YOLO_AVAILABLE:
                    matted_image, confidence = _segment_with_yolo(processed_image)
                    current_app.logger.info(f"YOLO segmentation completed, confidence: {confidence:.3f}")
                else:
                    raise RuntimeError("YOLO not available")
            except Exception as e:
                current_app.logger.warning(f"YOLO failed, using rembg: {e}")
                matted_image, confidence = _segment_with_rembg(processed_image)
                current_app.logger.info(f"rembg segmentation completed, confidence: {confidence:.3f}")
        
        elif model_type == "yolo":
            if not _YOLO_AVAILABLE:
                abort(400, "YOLO model not available")
            matted_image, confidence = _segment_with_yolo(processed_image)
        
        elif model_type in ["rembg", "human"]:
            matted_image, confidence = _segment_with_rembg(processed_image, model_type)
        
        else:
            abort(400, f"Unknown model type: {model_type}")
        
        # Validate result
        if matted_image is None:
            abort(500, "Segmentation failed")
        
        # Check if result has transparency
        alpha = np.array(matted_image.getchannel("A"))
        if np.sum(alpha > 128) == 0:
            abort(422, "No object detected in image")
        
        # Upload cutout to S3
        cutout_url = _upload_to_s3(matted_image, S3_BUCKET, cutout_key)
        
        # Generate and upload shadow if requested
        shadow_url = None
        if generate_shadow:
            shadow_image = _generate_shadow(processed_image, alpha)
            if shadow_image:
                shadow_url = _upload_to_s3(shadow_image, S3_BUCKET, shadow_key)
        
        # Determine if manual review is needed
        needs_manual = confidence < CONFIDENCE_THRESHOLD
        
        # Store result in database if requested
        result_data = None
        if store_result:
            result_data = _store_matting_result(
                original_url=image_url,
                cutout_url=cutout_url,
                shadow_url=shadow_url,
                confidence=confidence,
                needs_manual=needs_manual
            )
        
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "cutout_url": cutout_url,
            "shadow_url": shadow_url,
            "confidence": round(confidence, 3),
            "needs_manual": needs_manual,
            "processing_time": round(processing_time, 2),
            "model_used": model_type if model_type != "auto" else ("yolo" if _YOLO_AVAILABLE else "rembg"),
            "image_hash": image_hash
        }
        
        if result_data:
            response["result_id"] = result_data["id"]
        
        current_app.logger.info(f"Auto matting completed: confidence={confidence:.3f}, needs_manual={needs_manual}")
        return jsonify(response)
        
    except Exception as e:
        current_app.logger.error(f"Auto matting failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time
        }), 500


@bp.route("/batch-process", methods=["POST"])
def batch_process():
    """
    Process multiple images in batch.
    
    POST /auto-matting/batch-process
    {
        "images": [
            {"image_url": "https://example.com/furniture1.jpg", "model": "auto"},
            {"image_url": "https://example.com/furniture2.jpg", "model": "rembg"}
        ],
        "generate_shadow": true
    }
    """
    import time
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            abort(400, "JSON data required")
        
        images = data.get("images", [])
        if not images:
            abort(400, "images array is required")
        
        if len(images) > 10:  # Limit batch size
            abort(400, "Maximum 10 images per batch")
        
        results = []
        for i, img_data in enumerate(images):
            try:
                # Process each image
                result = process_image()
                if result[1] == 200:  # Success
                    results.append(result[0].get_json())
                else:
                    results.append({
                        "success": False,
                        "error": "Processing failed",
                        "image_url": img_data.get("image_url")
                    })
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e),
                    "image_url": img_data.get("image_url")
                })
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "results": results,
            "total_images": len(images),
            "successful": len([r for r in results if r.get("success")]),
            "processing_time": round(processing_time, 2)
        })
        
    except Exception as e:
        current_app.logger.error(f"Batch processing failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time
        }), 500


@bp.route("/status/<image_hash>", methods=["GET"])
def get_status(image_hash: str):
    """Get processing status for an image."""
    db = SessionLocal()
    try:
        # Find image by hash (you might need to add a hash field to the Image model)
        image_record = db.query(ImageModel).filter(
            ImageModel.url.contains(image_hash)
        ).first()
        
        if not image_record:
            return jsonify({"error": "Image not found"}), 404
        
        return jsonify({
            "image_hash": image_hash,
            "cutout_url": image_record.cutout_url,
            "quality_score": image_record.quality_score,
            "needs_manual": image_record.quality_score < CONFIDENCE_THRESHOLD if image_record.quality_score else None,
            "created_at": image_record.created_at.isoformat() if hasattr(image_record, 'created_at') else None
        })
        
    except Exception as e:
        current_app.logger.error(f"Status check failed: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "yolo_available": _YOLO_AVAILABLE,
        "s3_available": s3_client is not None,
        "rembg_available": True,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }
    
    if _YOLO_ERR:
        status["yolo_error"] = str(_YOLO_ERR)
    
    return jsonify(status)


# Register the blueprint
def register_blueprint(app):
    """Register the auto matting blueprint with the Flask app."""
    app.register_blueprint(bp)
    print("[auto_matting] Auto matting service registered")
