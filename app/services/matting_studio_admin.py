"""
Enhanced Matting Studio Admin Tool
Advanced image editing with brush tools, keyboard shortcuts, and versioned storage
"""

import os
import json
import base64
import io
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import boto3
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from flask import Blueprint, request, jsonify, render_template, current_app
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import S3_BUCKET, S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_REGION
try:
    from app.database import SessionLocal
    _DATABASE_AVAILABLE = True
except Exception as e:
    _DATABASE_AVAILABLE = False
    print(f"[matting_studio_admin] Database unavailable: {e}")

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
    print("[matting_studio_admin] S3 client initialized")
except Exception as e:
    _S3_AVAILABLE = False
    print(f"[matting_studio_admin] S3 unavailable: {e}")

# Create blueprint
matting_studio_admin_bp = Blueprint('matting_studio_admin', __name__, url_prefix='/matting-studio-admin')

# Brush settings
DEFAULT_BRUSH_SIZE = 20
DEFAULT_BRUSH_HARDNESS = 0.8
DEFAULT_EDGE_FEATHER = 2

class MattingStudioEditor:
    """Advanced matting editor with brush tools and versioning"""
    
    def __init__(self):
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.brush_hardness = DEFAULT_BRUSH_HARDNESS
        self.edge_feather = DEFAULT_EDGE_FEATHER
        self.overlay_mode = "checkerboard"  # mask, alpha, checkerboard
        self.keep_shadow = True
        self.history = []
        self.history_index = -1
        
    def create_brush_mask(self, size: int, hardness: float) -> np.ndarray:
        """Create a circular brush mask with hardness falloff"""
        mask = np.zeros((size * 2 + 1, size * 2 + 1), dtype=np.float32)
        center = size
        
        for y in range(size * 2 + 1):
            for x in range(size * 2 + 1):
                distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                if distance <= size:
                    # Hardness falloff
                    normalized_distance = distance / size
                    if normalized_distance <= hardness:
                        mask[y, x] = 1.0
                    else:
                        # Smooth falloff
                        falloff = (1 - normalized_distance) / (1 - hardness)
                        mask[y, x] = max(0, falloff)
        
        return mask
    
    def apply_brush_stroke(self, image: np.ndarray, mask: np.ndarray, 
                          pos: Tuple[int, int], action: str) -> np.ndarray:
        """Apply brush stroke to image mask"""
        brush_mask = self.create_brush_mask(self.brush_size, self.brush_hardness)
        brush_h, brush_w = brush_mask.shape
        
        # Calculate brush bounds
        y1 = max(0, pos[1] - self.brush_size)
        y2 = min(image.shape[0], pos[1] + self.brush_size + 1)
        x1 = max(0, pos[0] - self.brush_size)
        x2 = min(image.shape[1], pos[0] + self.brush_size + 1)
        
        # Calculate brush mask bounds
        brush_y1 = max(0, self.brush_size - pos[1])
        brush_y2 = brush_y1 + (y2 - y1)
        brush_x1 = max(0, self.brush_size - pos[0])
        brush_x2 = brush_x1 + (x2 - x1)
        
        # Extract brush region
        brush_region = brush_mask[brush_y1:brush_y2, brush_x1:brush_x2]
        
        # Apply brush stroke
        if action == "add":
            image[y1:y2, x1:x2] = np.maximum(image[y1:y2, x1:x2], brush_region)
        elif action == "remove":
            image[y1:y2, x1:x2] = np.minimum(image[y1:y2, x1:x2], 1.0 - brush_region)
        
        return image
    
    def feather_edges(self, mask: np.ndarray, feather: int) -> np.ndarray:
        """Apply edge feathering to mask"""
        if feather <= 0:
            return mask
        
        # Convert to uint8 for Gaussian blur
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Apply Gaussian blur for feathering
        blurred = cv2.GaussianBlur(mask_uint8, (feather * 2 + 1, feather * 2 + 1), feather / 3)
        
        # Convert back to float
        return blurred.astype(np.float32) / 255.0
    
    def save_state(self, mask: np.ndarray):
        """Save current state to history for undo/redo"""
        # Remove any states after current index
        self.history = self.history[:self.history_index + 1]
        
        # Add new state
        self.history.append(mask.copy())
        self.history_index += 1
        
        # Limit history size
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_index -= 1
    
    def undo(self) -> Optional[np.ndarray]:
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            return self.history[self.history_index].copy()
        return None
    
    def redo(self) -> Optional[np.ndarray]:
        """Redo last action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            return self.history[self.history_index].copy()
        return None

def _get_image_from_s3(bucket: str, key: str) -> Optional[np.ndarray]:
    """Fetch image from S3 and return as numpy array"""
    if not _S3_AVAILABLE:
        return None
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        
        # Convert to numpy array
        image = Image.open(io.BytesIO(image_data))
        if image.mode == 'RGBA':
            return np.array(image)
        else:
            return np.array(image.convert('RGBA'))
    except Exception as e:
        print(f"Error fetching image from S3: {e}")
        return None

def _save_image_to_s3(image: np.ndarray, bucket: str, key: str, content_type: str = "image/png") -> bool:
    """Save image to S3"""
    if not _S3_AVAILABLE:
        return False
    
    try:
        # Convert numpy array to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=img_buffer.getvalue(),
            ContentType=content_type
        )
        return True
    except Exception as e:
        print(f"Error saving image to S3: {e}")
        return False

def _generate_shadow_from_mask(mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """Generate realistic shadow from mask"""
    # Create shadow by blurring and darkening the mask
    shadow_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # Create shadow image
    shadow = np.zeros_like(original_image)
    shadow[:, :, 3] = shadow_mask * 0.6  # Semi-transparent shadow
    
    return shadow

def _get_review_queue() -> List[Dict[str, Any]]:
    """Get images that need manual review (low confidence)"""
    if not _DATABASE_AVAILABLE:
        return []
    
    try:
        with SessionLocal() as db:
            # Query for low confidence images
            query = text("""
                SELECT id, original_url, cutout_url, shadow_url, confidence_score, 
                       created_at, metadata
                FROM render_sessions 
                WHERE confidence_score < 0.7 
                AND status = 'completed'
                ORDER BY created_at DESC
                LIMIT 50
            """)
            
            result = db.execute(query)
            rows = result.fetchall()
            
            queue = []
            for row in rows:
                queue.append({
                    'id': row.id,
                    'original_url': row.original_url,
                    'cutout_url': row.cutout_url,
                    'shadow_url': row.shadow_url,
                    'confidence_score': float(row.confidence_score) if row.confidence_score else 0.0,
                    'created_at': row.created_at.isoformat() if row.created_at else None,
                    'metadata': json.loads(row.metadata) if row.metadata else {}
                })
            
            return queue
    except Exception as e:
        print(f"Error getting review queue: {e}")
        return []

@matting_studio_admin_bp.route('/')
def matting_studio_admin_home():
    """Main Matting Studio Admin interface"""
    return render_template('matting_studio_admin.html')

@matting_studio_admin_bp.route('/queue')
def get_review_queue():
    """Get images that need manual review"""
    try:
        queue = _get_review_queue()
        return jsonify({
            'success': True,
            'queue': queue,
            'total': len(queue)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@matting_studio_admin_bp.route('/api/matting/<int:matting_id>', methods=['GET'])
def get_matting_data(matting_id: int):
    """Get matting data for editing"""
    if not _DATABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Database not available'}), 500
    
    try:
        with SessionLocal() as db:
            # Get matting session data
            query = text("""
                SELECT id, original_url, cutout_url, shadow_url, confidence_score,
                       created_at, metadata, mask_data
                FROM render_sessions 
                WHERE id = :matting_id
            """)
            
            result = db.execute(query, {'matting_id': matting_id})
            row = result.fetchone()
            
            if not row:
                return jsonify({'success': False, 'error': 'Matting session not found'}), 404
            
            # Load original image
            original_image = _get_image_from_s3(S3_BUCKET, f"uploads/{row.original_url}")
            if original_image is None:
                return jsonify({'success': False, 'error': 'Original image not found'}), 404
            
            # Load existing mask if available
            mask_data = None
            if row.mask_data:
                try:
                    mask_data = json.loads(row.mask_data)
                except:
                    pass
            
            return jsonify({
                'success': True,
                'matting': {
                    'id': row.id,
                    'original_url': row.original_url,
                    'cutout_url': row.cutout_url,
                    'shadow_url': row.shadow_url,
                    'confidence_score': float(row.confidence_score) if row.confidence_score else 0.0,
                    'created_at': row.created_at.isoformat() if row.created_at else None,
                    'metadata': json.loads(row.metadata) if row.metadata else {},
                    'mask_data': mask_data
                },
                'original_image': base64.b64encode(original_image.tobytes()).decode('utf-8'),
                'original_shape': original_image.shape
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@matting_studio_admin_bp.route('/api/matting/<int:matting_id>', methods=['PUT'])
def save_matting_data(matting_id: int):
    """Save edited matting data"""
    if not _DATABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Database not available'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Extract data
        mask_data = data.get('mask_data')
        brush_settings = data.get('brush_settings', {})
        keep_shadow = data.get('keep_shadow', True)
        edited_by = data.get('edited_by', 'admin')
        
        if not mask_data:
            return jsonify({'success': False, 'error': 'Mask data required'}), 400
        
        # Get original matting data
        with SessionLocal() as db:
            query = text("""
                SELECT original_url, cutout_url, shadow_url, metadata
                FROM render_sessions 
                WHERE id = :matting_id
            """)
            
            result = db.execute(query, {'matting_id': matting_id})
            row = result.fetchone()
            
            if not row:
                return jsonify({'success': False, 'error': 'Matting session not found'}), 404
            
            # Load original image
            original_image = _get_image_from_s3(S3_BUCKET, f"uploads/{row.original_url}")
            if original_image is None:
                return jsonify({'success': False, 'error': 'Original image not found'}), 404
            
            # Convert mask data back to numpy array
            mask_array = np.frombuffer(base64.b64decode(mask_data), dtype=np.float32)
            mask_shape = data.get('mask_shape', [original_image.shape[0], original_image.shape[1]])
            mask = mask_array.reshape(mask_shape)
            
            # Apply edge feathering
            feather = brush_settings.get('edge_feather', DEFAULT_EDGE_FEATHER)
            if feather > 0:
                mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), feather / 3)
            
            # Generate new cutout
            cutout = original_image.copy()
            cutout[:, :, 3] = (mask * 255).astype(np.uint8)
            
            # Generate shadow if requested
            shadow = None
            if keep_shadow:
                shadow = _generate_shadow_from_mask(mask, original_image)
            
            # Create versioned filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v{timestamp}"
            
            cutout_key = f"renders/{row.cutout_url}_{version}.png"
            shadow_key = f"renders/{row.shadow_url}_{version}.png" if keep_shadow else None
            
            # Save to S3
            cutout_saved = _save_image_to_s3(cutout, S3_BUCKET, cutout_key)
            shadow_saved = _save_image_to_s3(shadow, S3_BUCKET, shadow_key) if shadow is not None else True
            
            if not cutout_saved or not shadow_saved:
                return jsonify({'success': False, 'error': 'Failed to save images to S3'}), 500
            
            # Update database with new version
            update_query = text("""
                UPDATE render_sessions 
                SET cutout_url = :cutout_url,
                    shadow_url = :shadow_url,
                    mask_data = :mask_data,
                    metadata = :metadata,
                    updated_at = NOW()
                WHERE id = :matting_id
            """)
            
            # Update metadata
            metadata = json.loads(row.metadata) if row.metadata else {}
            metadata.update({
                'edited_by': edited_by,
                'edited_at': datetime.now().isoformat(),
                'version': version,
                'brush_settings': brush_settings,
                'keep_shadow': keep_shadow
            })
            
            db.execute(update_query, {
                'matting_id': matting_id,
                'cutout_url': cutout_key,
                'shadow_url': shadow_key,
                'mask_data': json.dumps(mask_data),
                'metadata': json.dumps(metadata)
            })
            db.commit()
            
            return jsonify({
                'success': True,
                'cutout_url': cutout_key,
                'shadow_url': shadow_key,
                'version': version
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@matting_studio_admin_bp.route('/api/brush/preview', methods=['POST'])
def preview_brush():
    """Generate brush preview for UI"""
    try:
        data = request.get_json()
        size = data.get('size', DEFAULT_BRUSH_SIZE)
        hardness = data.get('hardness', DEFAULT_BRUSH_HARDNESS)
        
        # Create brush mask
        editor = MattingStudioEditor()
        editor.brush_size = size
        editor.brush_hardness = hardness
        
        brush_mask = editor.create_brush_mask(size, hardness)
        
        # Convert to base64 for frontend
        brush_image = (brush_mask * 255).astype(np.uint8)
        brush_pil = Image.fromarray(brush_image)
        
        img_buffer = io.BytesIO()
        brush_pil.save(img_buffer, format='PNG')
        brush_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'brush_preview': f"data:image/png;base64,{brush_b64}",
            'size': size,
            'hardness': hardness
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def register_blueprint(app):
    """Register the Matting Studio Admin blueprint"""
    try:
        app.register_blueprint(matting_studio_admin_bp)
        print("[matting_studio_admin] Matting Studio Admin service registered")
        print("[matting_studio_admin] enabled")
        return True
    except Exception as e:
        print(f"[matting_studio_admin] failed to register: {e}")
        return False

