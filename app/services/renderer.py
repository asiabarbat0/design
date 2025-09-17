"""
Advanced 2D Compositor and Renderer Service
Handles inpainting, perspective correction, lighting, and progressive rendering
"""

import os
import io
import time
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

import boto3
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import S3_BUCKET, S3_RENDER_BUCKET, S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_REGION

# Configure logging
logger = logging.getLogger(__name__)

# Initialize S3 client
try:
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=S3_REGION
    )
    _S3_AVAILABLE = True
    print("[renderer] S3 client initialized")
except Exception as e:
    _S3_AVAILABLE = False
    print(f"[renderer] S3 unavailable: {e}")

# Create blueprint
renderer_bp = Blueprint('renderer', __name__, url_prefix='/renderer')

class RendererService:
    """Advanced 2D compositor with inpainting and perspective correction"""
    
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.render_times = []
        
    def _get_cache_key(self, room_id: str, item_id: str, size: str = "full") -> str:
        """Generate cache key for rendered image"""
        return f"renders/{room_id}_{item_id}_{size}_{int(time.time() // 3600)}.png"  # Hourly cache
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if rendered image exists in cache"""
        if not _S3_AVAILABLE:
            return None
            
        try:
            s3_client.head_object(Bucket=S3_RENDER_BUCKET, Key=cache_key)
            self.cache_hits += 1
            return f"{S3_ENDPOINT_URL}/{S3_RENDER_BUCKET}/{cache_key}"
        except:
            self.cache_misses += 1
            return None
    
    def _save_to_cache(self, image: np.ndarray, cache_key: str) -> str:
        """Save rendered image to S3 cache"""
        if not _S3_AVAILABLE:
            return None
            
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image.astype(np.uint8))
            else:
                pil_image = Image.fromarray(image.astype(np.uint8))
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG', optimize=True)
            img_buffer.seek(0)
            
            # Upload to S3
            s3_client.put_object(
                Bucket=S3_RENDER_BUCKET,
                Key=cache_key,
                Body=img_buffer.getvalue(),
                ContentType='image/png',
                CacheControl='max-age=3600'  # 1 hour cache
            )
            
            return f"{S3_ENDPOINT_URL}/{S3_RENDER_BUCKET}/{cache_key}"
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
            return None
    
    def _load_image_from_s3(self, bucket: str, key: str) -> Optional[np.ndarray]:
        """Load image from S3"""
        if not _S3_AVAILABLE:
            return None
            
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_data = response['Body'].read()
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(image_data))
            return np.array(image)
        except Exception as e:
            logger.error(f"Failed to load image from S3: {e}")
            return None
    
    def _inpaint_room(self, room_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove original item from room using inpainting"""
        try:
            # Convert to OpenCV format
            room_cv = cv2.cvtColor(room_image, cv2.COLOR_RGB2BGR)
            mask_cv = (mask * 255).astype(np.uint8)
            
            # Use OpenCV's inpainting
            inpainted = cv2.inpaint(room_cv, mask_cv, 3, cv2.INPAINT_TELEA)
            
            # Convert back to RGB
            return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return room_image
    
    def _detect_perspective(self, room_image: np.ndarray) -> np.ndarray:
        """Detect room perspective for proper item placement"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(room_image, cv2.COLOR_RGB2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is not None:
                # Find floor lines (horizontal lines in lower half)
                floor_lines = []
                h, w = room_image.shape[:2]
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if y1 > h * 0.6 and y2 > h * 0.6:  # Lower half of image
                        floor_lines.append(line[0])
                
                if len(floor_lines) > 0:
                    # Calculate average floor angle
                    angles = []
                    for line in floor_lines:
                        angle = np.arctan2(line[3] - line[1], line[2] - line[0])
                        angles.append(angle)
                    
                    avg_angle = np.mean(angles)
                    return np.array([[1, 0, 0], [0, 1, 0], [0, -np.tan(avg_angle), 1]])
            
            # Default perspective (no transformation)
            return np.eye(3)
        except Exception as e:
            logger.error(f"Perspective detection failed: {e}")
            return np.eye(3)
    
    def _apply_perspective(self, item_image: np.ndarray, perspective_matrix: np.ndarray, 
                          target_size: Tuple[int, int]) -> np.ndarray:
        """Apply perspective transformation to item"""
        try:
            h, w = item_image.shape[:2]
            
            # Define source points (item corners)
            src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            
            # Calculate destination points with perspective
            dst_points = []
            for point in src_points:
                # Apply perspective transformation
                x, y = point
                transformed = perspective_matrix @ np.array([x, y, 1])
                if transformed[2] != 0:
                    x_new = transformed[0] / transformed[2]
                    y_new = transformed[1] / transformed[2]
                else:
                    x_new, y_new = x, y
                dst_points.append([x_new, y_new])
            
            dst_points = np.float32(dst_points)
            
            # Get perspective transform matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply transformation
            transformed = cv2.warpPerspective(item_image, M, target_size)
            
            return transformed
        except Exception as e:
            logger.error(f"Perspective transformation failed: {e}")
            return item_image
    
    def _recolor_item(self, item_image: np.ndarray, room_lighting: np.ndarray) -> np.ndarray:
        """Recolor item to match room lighting"""
        try:
            # Convert to PIL for easier color manipulation
            pil_item = Image.fromarray(item_image.astype(np.uint8))
            
            # Calculate room lighting characteristics
            room_mean = np.mean(room_lighting, axis=(0, 1))
            item_mean = np.mean(item_image, axis=(0, 1))
            
            # Calculate color adjustment factors
            color_factors = room_mean / (item_mean + 1e-6)  # Avoid division by zero
            
            # Apply color correction
            r, g, b = pil_item.split()
            r = r.point(lambda x: min(255, int(x * color_factors[0])))
            g = g.point(lambda x: min(255, int(x * color_factors[1])))
            b = b.point(lambda x: min(255, int(x * color_factors[2])))
            
            # Merge channels
            corrected = Image.merge('RGB', (r, g, b))
            
            return np.array(corrected)
        except Exception as e:
            logger.error(f"Recoloring failed: {e}")
            return item_image
    
    def _generate_shadow(self, item_image: np.ndarray, room_image: np.ndarray, 
                        position: Tuple[int, int]) -> np.ndarray:
        """Generate realistic shadow for the item"""
        try:
            # Create shadow mask from item alpha
            if item_image.shape[2] == 4:  # RGBA
                alpha = item_image[:, :, 3]
            else:
                alpha = np.ones(item_image.shape[:2]) * 255
            
            # Create shadow by blurring and darkening
            shadow_mask = cv2.GaussianBlur(alpha, (21, 21), 0)
            shadow_mask = shadow_mask / 255.0
            
            # Create shadow image
            shadow = np.zeros_like(room_image)
            shadow[:, :, 0] = shadow_mask * 0.3  # Dark shadow
            shadow[:, :, 1] = shadow_mask * 0.3
            shadow[:, :, 2] = shadow_mask * 0.3
            
            return shadow
        except Exception as e:
            logger.error(f"Shadow generation failed: {e}")
            return np.zeros_like(room_image)
    
    def _blend_images(self, room_image: np.ndarray, item_image: np.ndarray, 
                     shadow_image: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """Blend item and shadow into room image"""
        try:
            result = room_image.copy()
            h, w = room_image.shape[:2]
            item_h, item_w = item_image.shape[:2]
            
            # Calculate placement bounds
            x1 = max(0, position[0])
            y1 = max(0, position[1])
            x2 = min(w, position[0] + item_w)
            y2 = min(h, position[1] + item_h)
            
            # Calculate item bounds
            item_x1 = max(0, -position[0])
            item_y1 = max(0, -position[1])
            item_x2 = item_x1 + (x2 - x1)
            item_y2 = item_y1 + (y2 - y1)
            
            # Extract regions
            room_region = result[y1:y2, x1:x2]
            item_region = item_image[item_y1:item_y2, item_x1:item_x2]
            shadow_region = shadow_image[y1:y2, x1:x2]
            
            # Get alpha channel
            if item_region.shape[2] == 4:
                alpha = item_region[:, :, 3:4] / 255.0
                item_rgb = item_region[:, :, :3]
            else:
                alpha = np.ones((item_region.shape[0], item_region.shape[1], 1))
                item_rgb = item_region
            
            # Apply shadow first
            shadow_alpha = shadow_region[:, :, 3:4] / 255.0 if shadow_region.shape[2] == 4 else np.ones((shadow_region.shape[0], shadow_region.shape[1], 1)) * 0.5
            room_region = room_region * (1 - shadow_alpha) + shadow_region[:, :, :3] * shadow_alpha
            
            # Blend item
            room_region = room_region * (1 - alpha) + item_rgb * alpha
            
            # Update result
            result[y1:y2, x1:x2] = room_region
            
            return result
        except Exception as e:
            logger.error(f"Image blending failed: {e}")
            return room_image
    
    def render_item(self, room_image: np.ndarray, item_cutout: np.ndarray, 
                   item_shadow: np.ndarray, position: Tuple[int, int], 
                   size: str = "full") -> np.ndarray:
        """Main rendering pipeline"""
        start_time = time.time()
        
        try:
            # Resize for target resolution
            if size == "preview":
                target_width = 960
            else:
                target_width = 1920
            
            # Calculate scale factor
            scale_factor = target_width / room_image.shape[1]
            target_height = int(room_image.shape[0] * scale_factor)
            
            # Resize room image
            room_resized = cv2.resize(room_image, (target_width, target_height))
            
            # Detect perspective
            perspective_matrix = self._detect_perspective(room_resized)
            
            # Resize and apply perspective to item
            item_resized = cv2.resize(item_cutout, (int(item_cutout.shape[1] * scale_factor), 
                                                int(item_cutout.shape[0] * scale_factor)))
            item_transformed = self._apply_perspective(item_resized, perspective_matrix, 
                                                    (target_width, target_height))
            
            # Recolor item to match room lighting
            item_recolored = self._recolor_item(item_transformed, room_resized)
            
            # Generate shadow
            shadow_resized = cv2.resize(item_shadow, (int(item_shadow.shape[1] * scale_factor), 
                                                    int(item_shadow.shape[0] * scale_factor)))
            shadow_transformed = self._apply_perspective(shadow_resized, perspective_matrix, 
                                                       (target_width, target_height))
            
            # Calculate new position
            new_position = (int(position[0] * scale_factor), int(position[1] * scale_factor))
            
            # Generate final shadow
            final_shadow = self._generate_shadow(item_recolored, room_resized, new_position)
            
            # Blend everything together
            result = self._blend_images(room_resized, item_recolored, final_shadow, new_position)
            
            # Log render time
            render_time = time.time() - start_time
            self.render_times.append(render_time)
            logger.info(f"Render completed in {render_time:.2f}s for {size} size")
            
            return result
            
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            return room_image
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rendering statistics"""
        avg_render_time = np.mean(self.render_times) if self.render_times else 0
        total_renders = len(self.render_times)
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "total_renders": total_renders,
            "avg_render_time": avg_render_time,
            "recent_renders": self.render_times[-10:] if self.render_times else []
        }

# Global renderer instance
renderer_service = RendererService()

@renderer_bp.route('/render')
def render_item():
    """Main rendering endpoint"""
    try:
        room_id = request.args.get('roomId')
        item_id = request.args.get('itemId')
        size = request.args.get('size', 'full')  # 'preview' or 'full'
        position_x = int(request.args.get('x', 0))
        position_y = int(request.args.get('y', 0))
        
        if not room_id or not item_id:
            return jsonify({'error': 'roomId and itemId are required'}), 400
        
        if size not in ['preview', 'full']:
            return jsonify({'error': 'size must be preview or full'}), 400
        
        # Check cache first
        cache_key = renderer_service._get_cache_key(room_id, item_id, size)
        cached_url = renderer_service._check_cache(cache_key)
        
        if cached_url:
            return jsonify({
                'success': True,
                'url': cached_url,
                'cached': True,
                'size': size
            })
        
        # Load images from S3
        room_image = renderer_service._load_image_from_s3(S3_BUCKET, f"uploads/{room_id}.jpg")
        item_cutout = renderer_service._load_image_from_s3(S3_BUCKET, f"renders/{item_id}_cutout.png")
        item_shadow = renderer_service._load_image_from_s3(S3_BUCKET, f"renders/{item_id}_shadow.png")
        
        if room_image is None:
            return jsonify({'error': 'Room image not found'}), 404
        if item_cutout is None:
            return jsonify({'error': 'Item cutout not found'}), 404
        if item_shadow is None:
            return jsonify({'error': 'Item shadow not found'}), 404
        
        # Render the item
        rendered = renderer_service.render_item(
            room_image, item_cutout, item_shadow, 
            (position_x, position_y), size
        )
        
        # Save to cache
        cache_url = renderer_service._save_to_cache(rendered, cache_key)
        
        if cache_url:
            return jsonify({
                'success': True,
                'url': cache_url,
                'cached': False,
                'size': size
            })
        else:
            return jsonify({'error': 'Failed to save rendered image'}), 500
            
    except Exception as e:
        logger.error(f"Render endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@renderer_bp.route('/render/batch', methods=['POST'])
def render_batch():
    """Batch rendering endpoint for multiple items"""
    try:
        data = request.get_json()
        if not data or 'renders' not in data:
            return jsonify({'error': 'renders array is required'}), 400
        
        results = []
        for render_request in data['renders']:
            room_id = render_request.get('roomId')
            item_id = render_request.get('itemId')
            size = render_request.get('size', 'full')
            position_x = render_request.get('x', 0)
            position_y = render_request.get('y', 0)
            
            if not room_id or not item_id:
                results.append({'error': 'roomId and itemId are required'})
                continue
            
            # Check cache first
            cache_key = renderer_service._get_cache_key(room_id, item_id, size)
            cached_url = renderer_service._check_cache(cache_key)
            
            if cached_url:
                results.append({
                    'success': True,
                    'url': cached_url,
                    'cached': True,
                    'size': size,
                    'roomId': room_id,
                    'itemId': item_id
                })
                continue
            
            # Load and render
            room_image = renderer_service._load_image_from_s3(S3_BUCKET, f"uploads/{room_id}.jpg")
            item_cutout = renderer_service._load_image_from_s3(S3_BUCKET, f"renders/{item_id}_cutout.png")
            item_shadow = renderer_service._load_image_from_s3(S3_BUCKET, f"renders/{item_id}_shadow.png")
            
            if room_image is None or item_cutout is None or item_shadow is None:
                results.append({'error': 'Required images not found'})
                continue
            
            rendered = renderer_service.render_item(
                room_image, item_cutout, item_shadow, 
                (position_x, position_y), size
            )
            
            cache_url = renderer_service._save_to_cache(rendered, cache_key)
            
            if cache_url:
                results.append({
                    'success': True,
                    'url': cache_url,
                    'cached': False,
                    'size': size,
                    'roomId': room_id,
                    'itemId': item_id
                })
            else:
                results.append({'error': 'Failed to save rendered image'})
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch render endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@renderer_bp.route('/stats')
def get_render_stats():
    """Get rendering statistics and performance metrics"""
    try:
        stats = renderer_service.get_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@renderer_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        's3_available': _S3_AVAILABLE,
        'renderer_ready': True
    })

def register_blueprint(app):
    """Register the renderer blueprint"""
    try:
        app.register_blueprint(renderer_bp)
        print("[renderer] Renderer service registered")
        print("[renderer] enabled")
        return True
    except Exception as e:
        print(f"[renderer] failed to register: {e}")
        return False
