#!/usr/bin/env python3
"""
Accurate Sofa Replacement
=========================

This version uses manual sofa area definition based on visual inspection
to ensure the new sofa replaces the original sofa exactly.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import rembg
import requests

# Configuration
OUTPUT_DIR = "accurate_sofa_replacement"
ROOM_IMAGE = "static/445.png"
FURNITURE_IMAGE = "https://k-kircher-home.myshopify.com/cdn/shop/files/0787477_sylvie-express-bench-sofa.jpg?v=1752147901&width=1440"

def ensure_output_dir():
    """Ensure output directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_furniture_image():
    """Download furniture image for testing"""
    print("📥 Downloading furniture image...")
    response = requests.get(FURNITURE_IMAGE)
    response.raise_for_status()
    
    furniture_path = os.path.join(OUTPUT_DIR, "furniture.jpg")
    with open(furniture_path, 'wb') as f:
        f.write(response.content)
    
    print(f"✅ Furniture image downloaded: {furniture_path}")
    return furniture_path

def define_sofa_area_manually(room_image_path):
    """Manually define the sofa area based on visual inspection of 445.png"""
    print("🎯 Manually defining sofa area...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    h, w = room_img.shape[:2]
    
    # Based on visual inspection of 445.png:
    # The sofa is in the center-left area of the image
    # It's a light-colored rectangular object
    # Roughly positioned at:
    # - Left edge: about 15% from left
    # - Top edge: about 35% from top
    # - Width: about 50% of image width
    # - Height: about 35% of image height
    
    sofa_x = int(w * 0.15)      # 15% from left
    sofa_y = int(h * 0.35)      # 35% from top
    sofa_w = int(w * 0.50)      # 50% of width
    sofa_h = int(h * 0.35)      # 35% of height
    
    print(f"   Sofa area: x={sofa_x}, y={sofa_y}, w={sofa_w}, h={sofa_h}")
    print(f"   Image size: {w}x{h}")
    
    # Create rectangular mask
    sofa_mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(sofa_mask, (sofa_x, sofa_y), (sofa_x + sofa_w, sofa_y + sofa_h), 255, -1)
    
    # Smooth the mask edges
    sofa_mask = cv2.GaussianBlur(sofa_mask, (15, 15), 0)
    
    # Save mask
    mask_path = os.path.join(OUTPUT_DIR, "manual_sofa_mask.png")
    cv2.imwrite(mask_path, sofa_mask)
    
    print(f"✅ Manual sofa mask saved: {mask_path}")
    return mask_path, sofa_mask, (sofa_x, sofa_y, sofa_w, sofa_h)

def remove_sofa_clean(room_image_path, mask_path):
    """Remove the sofa cleanly from the room"""
    print("🎨 Removing sofa cleanly...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    # Use inpainting to remove the sofa
    inpainted = cv2.inpaint(room_img, mask, 3, cv2.INPAINT_TELEA)
    
    # Apply gentle smoothing
    inpainted = cv2.bilateralFilter(inpainted, 5, 50, 50)
    
    # Save clean room
    clean_room_path = os.path.join(OUTPUT_DIR, "room_without_sofa.png")
    cv2.imwrite(clean_room_path, inpainted)
    
    print(f"✅ Clean room saved: {clean_room_path}")
    return clean_room_path

def create_sofa_cutout(furniture_image_path):
    """Create cutout of the new sofa"""
    print("✂️ Creating sofa cutout...")
    
    # Load the furniture image
    with open(furniture_image_path, 'rb') as f:
        input_data = f.read()
    
    # Use rembg for background removal
    try:
        session = rembg.new_session('u2net')
        output_data = rembg.remove(input_data, session=session)
        
        # Save the cutout
        cutout_path = os.path.join(OUTPUT_DIR, "sofa_cutout.png")
        with open(cutout_path, 'wb') as f:
            f.write(output_data)
        
        print(f"✅ Sofa cutout saved: {cutout_path}")
        return cutout_path
        
    except Exception as e:
        print(f"❌ Cutout creation failed: {e}")
        raise

def create_subtle_shadow(cutout_path, room_size):
    """Create a subtle shadow"""
    print("🌫️ Creating subtle shadow...")
    
    # Load the cutout
    cutout_img = Image.open(cutout_path).convert('RGBA')
    
    # Get dimensions
    width, height = cutout_img.size
    
    # Create a very subtle horizontal shadow
    shadow_width = int(width * 1.1)
    shadow_height = int(height * 0.15)  # Very short shadow
    
    # Create shadow canvas
    shadow_canvas = Image.new('RGBA', (shadow_width, shadow_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow_canvas)
    
    # Draw subtle horizontal ellipse
    draw.ellipse([0, 0, shadow_width, shadow_height], 
                 fill=(0, 0, 0, 30))  # Very subtle
    
    # Apply Gaussian blur
    shadow_canvas = shadow_canvas.filter(ImageFilter.GaussianBlur(radius=10))
    
    # Create full room shadow
    shadow_img = Image.new('RGBA', room_size, (0, 0, 0, 0))
    
    # Position shadow at bottom of sofa area
    shadow_x = (room_size[0] - shadow_width) // 2
    shadow_y = int(room_size[1] * 0.65)  # Near bottom of sofa area
    
    # Paste shadow
    shadow_img.paste(shadow_canvas, (shadow_x, shadow_y), shadow_canvas)
    
    # Save shadow
    shadow_path = os.path.join(OUTPUT_DIR, "subtle_shadow.png")
    shadow_img.save(shadow_path)
    
    print(f"✅ Subtle shadow saved: {shadow_path}")
    return shadow_path

def replace_sofa_exactly(clean_room_path, cutout_path, shadow_path, sofa_bbox):
    """Replace the sofa exactly in the same location"""
    print("🏠 Replacing sofa exactly...")
    
    # Load images
    room_img = Image.open(clean_room_path).convert('RGBA')
    cutout_img = Image.open(cutout_path).convert('RGBA')
    shadow_img = Image.open(shadow_path).convert('RGBA')
    
    # Get sofa bounding box
    x, y, w, h = sofa_bbox
    
    # Calculate scale to fit the sofa space exactly
    target_width = w
    scale = target_width / cutout_img.width
    new_height = int(cutout_img.height * scale)
    
    # Resize furniture to fit the space exactly
    cutout_resized = cutout_img.resize((target_width, new_height), Image.Resampling.LANCZOS)
    
    # Position the new sofa exactly where the old one was
    furniture_x = x
    furniture_y = y + h - new_height  # Align with bottom of original sofa space
    
    # Create result canvas
    result_img = room_img.copy()
    
    # Add subtle shadow first
    result_img.paste(shadow_img, (0, 0), shadow_img)
    
    # Add the new sofa in the exact same position as the old one
    result_img.paste(cutout_resized, (furniture_x, furniture_y), cutout_resized)
    
    # Save result
    result_path = os.path.join(OUTPUT_DIR, "accurate_sofa_replacement.png")
    result_img.save(result_path)
    
    print(f"✅ Accurate sofa replacement saved: {result_path}")
    print(f"   New sofa positioned at: x={furniture_x}, y={furniture_y}")
    print(f"   New sofa size: {target_width}x{new_height}")
    print(f"   Original sofa area: x={x}, y={y}, w={w}, h={h}")
    
    return result_path

def main():
    """Main accurate sofa replacement"""
    print("🛋️  ACCURATE SOFA REPLACEMENT")
    print("=" * 50)
    
    ensure_output_dir()
    
    try:
        # Step 1: Download furniture image
        furniture_path = download_furniture_image()
        
        # Step 2: Manually define sofa area
        mask_path, mask, sofa_bbox = define_sofa_area_manually(ROOM_IMAGE)
        
        # Step 3: Remove sofa cleanly
        clean_room_path = remove_sofa_clean(ROOM_IMAGE, mask_path)
        
        # Step 4: Create sofa cutout
        cutout_path = create_sofa_cutout(furniture_path)
        
        # Step 5: Create subtle shadow
        room_img = Image.open(ROOM_IMAGE)
        shadow_path = create_subtle_shadow(cutout_path, room_img.size)
        
        # Step 6: Replace sofa exactly
        result_path = replace_sofa_exactly(
            clean_room_path, 
            cutout_path, 
            shadow_path,
            sofa_bbox
        )
        
        print("\n🎉 ACCURATE SOFA REPLACEMENT COMPLETE!")
        print("=" * 50)
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"🎯 Sofa mask: {mask_path}")
        print(f"🏠 Clean room: {clean_room_path}")
        print(f"✂️  Sofa cutout: {cutout_path}")
        print(f"🌫️  Subtle shadow: {shadow_path}")
        print(f"🖼️  Final result: {result_path}")
        
        # Open the result
        try:
            result_img = Image.open(result_path)
            result_img.show()
            print("🖼️  Result image opened in default viewer")
        except Exception as e:
            print(f"⚠️  Could not open result image: {e}")
            
    except Exception as e:
        print(f"❌ Accurate sofa replacement failed: {e}")
        raise

if __name__ == "__main__":
    main()
