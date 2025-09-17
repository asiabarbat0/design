"""
Matting Studio Service
Handles manual cleanup and review of low-confidence auto matting results.
"""

import os
import io
import uuid
from typing import Dict, List, Optional
from flask import Blueprint, request, abort, jsonify, current_app, render_template
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import requests
import numpy as np
from sqlalchemy.orm import Session
from app.database import SessionLocal, Image as ImageModel
from app.config import S3_BUCKET, S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
import boto3
from botocore.exceptions import ClientError

bp = Blueprint("matting_studio", __name__, url_prefix="/matting-studio")

# Configuration
TIMEOUT = 30
MAX_BYTES = 25 * 1024 * 1024  # 25MB
CONFIDENCE_THRESHOLD = 0.7

# Initialize S3 client
try:
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name='us-east-1'
    )
    print("[matting_studio] S3 client initialized")
except Exception as e:
    print(f"[matting_studio] S3 client failed: {e}")
    s3_client = None


def _fetch_image(url: str) -> bytes:
    """Fetch image from URL with proper error handling."""
    headers = {"User-Agent": "designstream-matting-studio/1.0", "Accept": "image/*"}
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


@bp.route("/", methods=["GET"])
def studio_interface():
    """Serve the Matting Studio interface."""
    return render_template("matting_studio.html")


@bp.route("/queue", methods=["GET"])
def get_review_queue():
    """
    Get list of images that need manual review.
    
    GET /matting-studio/queue?limit=20&offset=0
    """
    try:
        limit = int(request.args.get("limit", 20))
        offset = int(request.args.get("offset", 0))
        
        db = SessionLocal()
        try:
            # Get images that need manual review
            images = db.query(ImageModel).filter(
                ImageModel.quality_score < CONFIDENCE_THRESHOLD
            ).order_by(ImageModel.created_at.desc()).offset(offset).limit(limit).all()
            
            results = []
            for img in images:
                results.append({
                    "id": img.id,
                    "original_url": img.url,
                    "cutout_url": img.cutout_url,
                    "confidence": img.quality_score,
                    "created_at": img.created_at.isoformat() if hasattr(img, 'created_at') else None,
                    "needs_review": True
                })
            
            return jsonify({
                "success": True,
                "images": results,
                "total": len(results),
                "has_more": len(results) == limit
            })
            
        finally:
            db.close()
            
    except Exception as e:
        current_app.logger.error(f"Failed to get review queue: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/process/<int:image_id>", methods=["POST"])
def process_manual_matting(image_id: int):
    """
    Process manual matting adjustments.
    
    POST /matting-studio/process/123
    {
        "brush_strokes": [
            {"type": "add", "x": 100, "y": 100, "radius": 20},
            {"type": "remove", "x": 200, "y": 200, "radius": 15}
        ],
        "refinement_mode": "smooth|sharpen|feather"
    }
    """
    try:
        data = request.get_json()
        if not data:
            abort(400, "JSON data required")
        
        brush_strokes = data.get("brush_strokes", [])
        refinement_mode = data.get("refinement_mode", "smooth")
        
        db = SessionLocal()
        try:
            # Get the image record
            image_record = db.query(ImageModel).filter(ImageModel.id == image_id).first()
            if not image_record:
                return jsonify({"success": False, "error": "Image not found"}), 404
            
            # Fetch original and current cutout images
            original_bytes = _fetch_image(image_record.url)
            original_image = Image.open(io.BytesIO(original_bytes))
            original_image = ImageOps.exif_transpose(original_image).convert("RGB")
            
            if image_record.cutout_url:
                cutout_bytes = _fetch_image(image_record.cutout_url)
                cutout_image = Image.open(io.BytesIO(cutout_bytes))
            else:
                # Create initial cutout if none exists
                cutout_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
            
            # Apply brush strokes
            result_image = _apply_brush_strokes(cutout_image, brush_strokes)
            
            # Apply refinement
            result_image = _apply_refinement(result_image, refinement_mode)
            
            # Generate new shadow
            alpha = np.array(result_image.getchannel("A"))
            shadow_image = _generate_enhanced_shadow(original_image, alpha)
            
            # Upload updated images
            image_hash = str(uuid.uuid4())
            cutout_key = f"manual_cutouts/{image_hash}_cutout.png"
            shadow_key = f"manual_shadows/{image_hash}_shadow.png"
            
            cutout_url = _upload_to_s3(result_image, S3_BUCKET, cutout_key)
            shadow_url = _upload_to_s3(shadow_image, S3_BUCKET, shadow_key)
            
            # Update database record
            image_record.cutout_url = cutout_url
            image_record.quality_score = 0.95  # High confidence for manual work
            
            db.commit()
            
            return jsonify({
                "success": True,
                "cutout_url": cutout_url,
                "shadow_url": shadow_url,
                "confidence": 0.95,
                "needs_manual": False,
                "image_id": image_id
            })
            
        finally:
            db.close()
            
    except Exception as e:
        current_app.logger.error(f"Manual matting failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def _apply_brush_strokes(image: Image.Image, strokes: List[Dict]) -> Image.Image:
    """Apply brush strokes to modify the alpha channel."""
    result = image.copy()
    
    for stroke in strokes:
        stroke_type = stroke.get("type")
        x = int(stroke.get("x", 0))
        y = int(stroke.get("y", 0))
        radius = int(stroke.get("radius", 10))
        
        # Create brush mask
        brush_size = radius * 2
        brush = Image.new("L", (brush_size, brush_size), 0)
        draw = ImageDraw.Draw(brush)
        
        # Create soft circular brush
        draw.ellipse([0, 0, brush_size, brush_size], fill=255)
        brush = brush.filter(ImageFilter.GaussianBlur(radius=radius/4))
        
        # Apply to alpha channel
        alpha = result.getchannel("A")
        alpha_array = np.array(alpha)
        
        # Calculate brush position
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(alpha_array.shape[1], x + radius)
        y2 = min(alpha_array.shape[0], y + radius)
        
        brush_x1 = max(0, radius - x)
        brush_y1 = max(0, radius - y)
        brush_x2 = brush_x1 + (x2 - x1)
        brush_y2 = brush_y1 + (y2 - y1)
        
        brush_cropped = brush.crop((brush_x1, brush_y1, brush_x2, brush_y2))
        brush_array = np.array(brush_cropped)
        
        if stroke_type == "add":
            # Add to alpha (make more opaque)
            alpha_array[y1:y2, x1:x2] = np.maximum(
                alpha_array[y1:y2, x1:x2],
                brush_array
            )
        elif stroke_type == "remove":
            # Remove from alpha (make more transparent)
            alpha_array[y1:y2, x1:x2] = np.minimum(
                alpha_array[y1:y2, x1:x2],
                255 - brush_array
            )
        
        # Update alpha channel
        new_alpha = Image.fromarray(alpha_array, "L")
        result.putalpha(new_alpha)
    
    return result


def _apply_refinement(image: Image.Image, mode: str) -> Image.Image:
    """Apply refinement to the alpha channel."""
    alpha = image.getchannel("A")
    
    if mode == "smooth":
        # Smooth the alpha channel
        alpha_smooth = alpha.filter(ImageFilter.GaussianBlur(radius=1))
        result = image.copy()
        result.putalpha(alpha_smooth)
        return result
    
    elif mode == "sharpen":
        # Sharpen the alpha channel
        alpha_sharp = alpha.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        result = image.copy()
        result.putalpha(alpha_sharp)
        return result
    
    elif mode == "feather":
        # Feather the edges
        alpha_feather = alpha.filter(ImageFilter.GaussianBlur(radius=2))
        result = image.copy()
        result.putalpha(alpha_feather)
        return result
    
    else:
        return image


def _generate_enhanced_shadow(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """Generate enhanced shadow with better quality."""
    try:
        # Create shadow from alpha channel
        shadow_alpha = alpha.astype(np.float32) / 255.0
        
        # Apply multiple blur passes for realistic shadow
        shadow_blurred = cv2.GaussianBlur(shadow_alpha, (15, 15), 0)
        shadow_blurred = cv2.GaussianBlur(shadow_blurred, (25, 25), 0)
        
        # Offset shadow (down and right)
        h, w = shadow_alpha.shape
        shadow_offset = np.zeros_like(shadow_blurred)
        offset_y, offset_x = 12, 6  # Shadow offset
        
        # Create offset shadow
        shadow_offset[offset_y:, offset_x:] = shadow_blurred[:-offset_y, :-offset_x]
        
        # Create gradient shadow (darker in center, lighter at edges)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        gradient = 1.0 - (distance / max_distance) * 0.3
        gradient = np.clip(gradient, 0.7, 1.0)
        
        shadow_offset = shadow_offset * gradient
        
        # Darken the shadow
        shadow_offset = shadow_offset * 0.5
        
        # Create RGBA shadow image
        shadow_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        shadow_rgba[:, :, 3] = (shadow_offset * 255).astype(np.uint8)
        
        # Convert to PIL Image
        shadow_image = Image.fromarray(shadow_rgba, 'RGBA')
        
        return shadow_image
        
    except Exception as e:
        current_app.logger.warning(f"Enhanced shadow generation failed: {e}")
        # Return transparent image if shadow generation fails
        return Image.new('RGBA', image.size, (0, 0, 0, 0))


@bp.route("/approve/<int:image_id>", methods=["POST"])
def approve_image(image_id: int):
    """Approve a manually processed image."""
    try:
        db = SessionLocal()
        try:
            image_record = db.query(ImageModel).filter(ImageModel.id == image_id).first()
            if not image_record:
                return jsonify({"success": False, "error": "Image not found"}), 404
            
            # Mark as approved (high confidence)
            image_record.quality_score = 0.95
            db.commit()
            
            return jsonify({
                "success": True,
                "message": "Image approved",
                "image_id": image_id
            })
            
        finally:
            db.close()
            
    except Exception as e:
        current_app.logger.error(f"Approval failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/reject/<int:image_id>", methods=["POST"])
def reject_image(image_id: int):
    """Reject an image and flag for re-processing."""
    try:
        data = request.get_json()
        reason = data.get("reason", "Quality not acceptable")
        
        db = SessionLocal()
        try:
            image_record = db.query(ImageModel).filter(ImageModel.id == image_id).first()
            if not image_record:
                return jsonify({"success": False, "error": "Image not found"}), 404
            
            # Mark as rejected (very low confidence)
            image_record.quality_score = 0.1
            db.commit()
            
            return jsonify({
                "success": True,
                "message": "Image rejected",
                "reason": reason,
                "image_id": image_id
            })
            
        finally:
            db.close()
            
    except Exception as e:
        current_app.logger.error(f"Rejection failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/stats", methods=["GET"])
def get_stats():
    """Get matting studio statistics."""
    try:
        db = SessionLocal()
        try:
            total_images = db.query(ImageModel).count()
            needs_review = db.query(ImageModel).filter(
                ImageModel.quality_score < CONFIDENCE_THRESHOLD
            ).count()
            approved = db.query(ImageModel).filter(
                ImageModel.quality_score >= 0.9
            ).count()
            
            return jsonify({
                "success": True,
                "stats": {
                    "total_images": total_images,
                    "needs_review": needs_review,
                    "approved": approved,
                    "pending_approval": needs_review,
                    "confidence_threshold": CONFIDENCE_THRESHOLD
                }
            })
            
        finally:
            db.close()
            
    except Exception as e:
        current_app.logger.error(f"Stats retrieval failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# Register the blueprint
def register_blueprint(app):
    """Register the matting studio blueprint with the Flask app."""
    app.register_blueprint(bp)
    print("[matting_studio] Matting Studio service registered")

