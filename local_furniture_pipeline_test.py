#!/usr/bin/env python3
"""
Local Furniture Pipeline Test
=============================

Test the furniture replacement pipeline locally without S3
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import rembg

# Configuration
OUTPUT_DIR = "local_pipeline_test"
ROOM_IMAGE = "static/445.png"
FURNITURE_IMAGE = "https://k-kircher-home.myshopify.com/cdn/shop/files/0787477_sylvie-express-bench-sofa.jpg?v=1752147901&width=1440"

def ensure_output_dir():
    """Ensure output directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_furniture_image():
    """Download furniture image for testing"""
    import requests
    
    print("📥 Downloading furniture image...")
    response = requests.get(FURNITURE_IMAGE)
    response.raise_for_status()
    
    furniture_path = os.path.join(OUTPUT_DIR, "furniture.jpg")
    with open(furniture_path, 'wb') as f:
        f.write(response.content)
    
    print(f"✅ Furniture image downloaded: {furniture_path}")
    return furniture_path

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

def main():
    """Main pipeline test"""
    print("🛋️  LOCAL FURNITURE REPLACEMENT PIPELINE TEST")
    print("=" * 60)
    
    ensure_output_dir()
    
    try:
        # Step 1: Download furniture image
        furniture_path = download_furniture_image()
        
        # Step 2: Segment furniture to binary mask
        mask_path, mask = segment_furniture_to_mask(ROOM_IMAGE, 'sofa')
        
        # Step 3: Inpaint room to produce clean version
        clean_room_path = inpaint_room_clean(ROOM_IMAGE, mask_path)
        
        # Step 4: Create furniture cutout
        cutout_path = create_furniture_cutout(furniture_path)
        
        # Step 5: Create soft shadow
        shadow_path = create_soft_shadow(cutout_path)
        
        # Step 6: Composite furniture with test parameters
        result_path = composite_furniture(
            clean_room_path, 
            cutout_path, 
            shadow_path,
            target_width=400,
            anchor_x=500,
            baseline_y=400
        )
        
        print("\n🎉 PIPELINE TEST COMPLETE!")
        print("=" * 40)
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"🎯 Binary mask: {mask_path}")
        print(f"🏠 Clean room: {clean_room_path}")
        print(f"✂️  Furniture cutout: {cutout_path}")
        print(f"🌫️  Soft shadow: {shadow_path}")
        print(f"🖼️  Final render: {result_path}")
        
        # Open the result
        try:
            result_img = Image.open(result_path)
            result_img.show()
            print("🖼️  Result image opened in default viewer")
        except Exception as e:
            print(f"⚠️  Could not open result image: {e}")
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        raise

if __name__ == "__main__":
    main()
