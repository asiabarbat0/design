#!/usr/bin/env python3
"""
Furniture Replacement Pipeline
==============================

Complete pipeline for furniture replacement:
1. Segment old item to binary mask.png (auto model + optional brush fixes)
2. Inpaint masked region to produce room_clean.png
3. Composite new cutout.png with uniform scale, baseline alignment, and soft shadow

API: POST /render with roomId, itemId, placement params
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import rembg
import hashlib
import time
from flask import Flask, request, jsonify
import boto3
from botocore.exceptions import ClientError

# Configuration
OUTPUT_DIR = "furniture_pipeline_output"
S3_BUCKET = "designstream-uploads"
S3_RENDER_BUCKET = "designstream-renders"

# Initialize Flask app
app = Flask(__name__)

# Initialize S3 client
try:
    s3_client = boto3.client(
        's3',
        endpoint_url=os.getenv('S3_ENDPOINT_URL', 'http://localhost:9000'),
        aws_access_key_id=os.getenv('S3_ACCESS_KEY', 'minioadmin'),
        aws_secret_access_key=os.getenv('S3_SECRET_KEY', 'minioadmin'),
        region_name=os.getenv('S3_REGION', 'us-east-1')
    )
    print("✅ S3 client initialized")
except Exception as e:
    print(f"⚠️ S3 client initialization failed: {e}")
    s3_client = None

def ensure_output_dir():
    """Ensure output directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_image_from_s3(bucket, key, local_path):
    """Download image from S3"""
    try:
        if s3_client:
            s3_client.download_file(bucket, key, local_path)
            return True
        else:
            print(f"⚠️ S3 not available, using local file: {local_path}")
            return os.path.exists(local_path)
    except Exception as e:
        print(f"❌ Failed to download from S3: {e}")
        return False

def upload_image_to_s3(local_path, bucket, key):
    """Upload image to S3"""
    try:
        if s3_client:
            s3_client.upload_file(local_path, bucket, key)
            return f"s3://{bucket}/{key}"
        else:
            print(f"⚠️ S3 not available, using local path: {local_path}")
            return local_path
    except Exception as e:
        print(f"❌ Failed to upload to S3: {e}")
        return local_path

def generate_cache_key(room_id, item_id, params):
    """Generate cache key for render parameters"""
    cache_data = {
        'roomId': room_id,
        'itemId': item_id,
        'params': params
    }
    cache_string = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(cache_string.encode()).hexdigest()

def segment_furniture_to_mask(room_image_path, furniture_type='sofa'):
    """Segment furniture to binary mask using auto model"""
    print(f"🎯 Segmenting {furniture_type} to binary mask...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    # Convert to different color spaces for better detection
    hsv = cv2.cvtColor(room_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(room_img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Color-based detection
    if furniture_type == 'sofa':
        # Light colored furniture
        lower_light = np.array([0, 0, 200])
        upper_light = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_light, upper_light)
    elif furniture_type == 'table':
        # Darker wood furniture
        lower_wood = np.array([10, 50, 50])
        upper_wood = np.array([25, 255, 200])
        mask = cv2.inRange(hsv, lower_wood, upper_wood)
    else:
        # Generic furniture detection
        lower_generic = np.array([0, 0, 150])
        upper_generic = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_generic, upper_generic)
    
    # Method 2: Edge-based refinement
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Combine methods
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges))
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours and keep only the largest
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create clean binary mask
        binary_mask = np.zeros(room_img.shape[:2], np.uint8)
        cv2.fillPoly(binary_mask, [largest_contour], 255)
        
        # Apply minimal smoothing
        binary_mask = cv2.GaussianBlur(binary_mask, (3, 3), 0)
        
        # Save binary mask
        mask_path = os.path.join(OUTPUT_DIR, "mask.png")
        cv2.imwrite(mask_path, binary_mask)
        
        print(f"✅ Binary mask saved: {mask_path}")
        return mask_path, binary_mask
    
    raise ValueError(f"No {furniture_type} detected in room image")

def inpaint_room_clean(room_image_path, mask_path):
    """Inpaint masked region to produce room_clean.png"""
    print(f"🎨 Inpainting room to produce clean version...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    # Use cv2.inpaint with TELEA method (MVP)
    # TODO: Upgrade to LaMa/SDXL inpainting for higher quality
    inpainted = cv2.inpaint(room_img, mask, 3, cv2.INPAINT_TELEA)
    
    # Save clean room
    clean_room_path = os.path.join(OUTPUT_DIR, "room_clean.png")
    cv2.imwrite(clean_room_path, inpainted)
    
    print(f"✅ Clean room saved: {clean_room_path}")
    return clean_room_path

def create_furniture_cutout(furniture_image_path):
    """Create cutout of furniture using rembg"""
    print(f"✂️ Creating furniture cutout...")
    
    # Load the furniture image
    with open(furniture_image_path, 'rb') as f:
        input_data = f.read()
    
    # Use rembg for background removal
    try:
        session = rembg.new_session('u2net')
        output_data = rembg.remove(input_data, session=session)
        
        # Save the cutout
        cutout_path = os.path.join(OUTPUT_DIR, "cutout.png")
        with open(cutout_path, 'wb') as f:
            f.write(output_data)
        
        print(f"✅ Furniture cutout saved: {cutout_path}")
        return cutout_path
        
    except Exception as e:
        print(f"❌ Cutout creation failed: {e}")
        raise

def create_soft_shadow(cutout_path, shadow_size_factor=1.2):
    """Create soft shadow using Gaussian-blurred ellipse"""
    print(f"🌫️ Creating soft shadow...")
    
    # Load the cutout
    cutout_img = Image.open(cutout_path).convert('RGBA')
    
    # Get dimensions
    width, height = cutout_img.size
    
    # Create shadow as a larger, darker version
    shadow_width = int(width * shadow_size_factor)
    shadow_height = int(height * shadow_size_factor)
    
    # Create shadow image
    shadow_img = Image.new('RGBA', (shadow_width, shadow_height), (0, 0, 0, 0))
    
    # Create ellipse for shadow
    draw = ImageDraw.Draw(shadow_img)
    
    # Calculate ellipse position (centered)
    ellipse_x1 = (shadow_width - width) // 2
    ellipse_y1 = (shadow_height - height) // 2
    ellipse_x2 = ellipse_x1 + width
    ellipse_y2 = ellipse_y1 + height
    
    # Draw dark ellipse
    draw.ellipse([ellipse_x1, ellipse_y1, ellipse_x2, ellipse_y2], 
                 fill=(0, 0, 0, 100))  # Semi-transparent black
    
    # Apply Gaussian blur
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=15))
    
    # Save shadow
    shadow_path = os.path.join(OUTPUT_DIR, "shadow.png")
    shadow_img.save(shadow_path)
    
    print(f"✅ Soft shadow saved: {shadow_path}")
    return shadow_path

def composite_furniture(clean_room_path, cutout_path, shadow_path, target_width, anchor_x, baseline_y):
    """Composite furniture with uniform scale, baseline alignment, and soft shadow"""
    print(f"🏠 Compositing furniture with scale {target_width}, anchor {anchor_x}, baseline {baseline_y}...")
    
    # Load images
    room_img = Image.open(clean_room_path).convert('RGBA')
    cutout_img = Image.open(cutout_path).convert('RGBA')
    shadow_img = Image.open(shadow_path).convert('RGBA')
    
    # Calculate uniform scale
    original_width = cutout_img.width
    scale = target_width / original_width
    new_height = int(cutout_img.height * scale)
    
    # Resize furniture and shadow using LANCZOS
    cutout_resized = cutout_img.resize((target_width, new_height), Image.Resampling.LANCZOS)
    shadow_resized = shadow_img.resize((int(shadow_img.width * scale), int(shadow_img.height * scale)), Image.Resampling.LANCZOS)
    
    # Calculate positions
    # anchor_x is the horizontal center of the furniture
    # baseline_y is the bottom edge of the furniture
    furniture_x = anchor_x - (target_width // 2)
    furniture_y = baseline_y - new_height
    
    # Shadow offset (slightly down and right)
    shadow_x = furniture_x - 10
    shadow_y = furniture_y + 5
    
    # Create result canvas
    result_img = room_img.copy()
    
    # First, composite the shadow
    result_img.paste(shadow_resized, (shadow_x, shadow_y), shadow_resized)
    
    # Then, composite the furniture
    result_img.paste(cutout_resized, (furniture_x, furniture_y), cutout_resized)
    
    # Save result
    result_path = os.path.join(OUTPUT_DIR, "render.png")
    result_img.save(result_path)
    
    print(f"✅ Render saved: {result_path}")
    return result_path

def process_furniture_replacement(room_id, item_id, placement_params):
    """Main pipeline for furniture replacement"""
    print(f"🛋️ Processing furniture replacement: room={room_id}, item={item_id}")
    
    ensure_output_dir()
    
    # Step 1: Download room image
    room_key = f"rooms/{room_id}.jpg"
    room_path = os.path.join(OUTPUT_DIR, f"room_{room_id}.jpg")
    if not download_image_from_s3(S3_BUCKET, room_key, room_path):
        raise ValueError(f"Could not download room image: {room_key}")
    
    # Step 2: Download furniture image
    furniture_key = f"furniture/{item_id}.jpg"
    furniture_path = os.path.join(OUTPUT_DIR, f"furniture_{item_id}.jpg")
    if not download_image_from_s3(S3_BUCKET, furniture_key, furniture_path):
        raise ValueError(f"Could not download furniture image: {furniture_key}")
    
    # Step 3: Segment furniture to binary mask
    mask_path, mask = segment_furniture_to_mask(room_path, 'sofa')
    
    # Step 4: Inpaint room to produce clean version
    clean_room_path = inpaint_room_clean(room_path, mask_path)
    
    # Step 5: Create furniture cutout
    cutout_path = create_furniture_cutout(furniture_path)
    
    # Step 6: Create soft shadow
    shadow_path = create_soft_shadow(cutout_path)
    
    # Step 7: Composite furniture
    result_path = composite_furniture(
        clean_room_path, 
        cutout_path, 
        shadow_path,
        placement_params['target_width'],
        placement_params['anchorX'],
        placement_params['baselineY']
    )
    
    # Step 8: Upload results to S3
    clean_room_s3_key = f"renders/{room_id}_{item_id}_clean.png"
    render_s3_key = f"renders/{room_id}_{item_id}_render.png"
    
    clean_room_url = upload_image_to_s3(clean_room_path, S3_RENDER_BUCKET, clean_room_s3_key)
    render_url = upload_image_to_s3(result_path, S3_RENDER_BUCKET, render_s3_key)
    
    return {
        'room_clean_url': clean_room_url,
        'render_url': render_url,
        'cache_key': generate_cache_key(room_id, item_id, placement_params)
    }

# Cache for storing results
render_cache = {}

@app.route('/render', methods=['POST'])
def render_furniture():
    """API endpoint for furniture rendering"""
    try:
        data = request.get_json()
        
        # Validate required parameters
        required_params = ['roomId', 'itemId', 'target_width', 'anchorX', 'baselineY']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'Missing required parameter: {param}'}), 400
        
        room_id = data['roomId']
        item_id = data['itemId']
        placement_params = {
            'target_width': data['target_width'],
            'anchorX': data['anchorX'],
            'baselineY': data['baselineY']
        }
        
        # Check cache first
        cache_key = generate_cache_key(room_id, item_id, placement_params)
        if cache_key in render_cache:
            print(f"✅ Cache hit for {cache_key}")
            return jsonify(render_cache[cache_key])
        
        # Process the furniture replacement
        start_time = time.time()
        result = process_furniture_replacement(room_id, item_id, placement_params)
        processing_time = time.time() - start_time
        
        # Add processing time to result
        result['processing_time'] = processing_time
        
        # Cache the result
        render_cache[cache_key] = result
        
        print(f"✅ Render completed in {processing_time:.2f}s")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Render failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        's3_available': s3_client is not None,
        'cache_size': len(render_cache)
    })

def main():
    """Main function for testing the pipeline"""
    print("🛋️  FURNITURE REPLACEMENT PIPELINE")
    print("=" * 50)
    
    # Test with sample data
    test_params = {
        'roomId': 'test_room',
        'itemId': 'test_item',
        'target_width': 400,
        'anchorX': 500,
        'baselineY': 400
    }
    
    try:
        result = process_furniture_replacement(**test_params)
        print(f"✅ Test completed: {result}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Run as Flask app
    app.run(host='0.0.0.0', port=5004, debug=True)
