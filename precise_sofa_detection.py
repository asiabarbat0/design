#!/usr/bin/env python3
"""
Precise Sofa Detection and Replacement
=====================================

This version uses more precise detection to target only the sofa,
not the whole background, and ensures proper replacement.
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
OUTPUT_DIR = "precise_sofa_detection"
ROOM_IMAGE = "new_room.jpg"  # Your actual new room image
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

def detect_sofa_precisely(room_image_path):
    """Detect only the sofa, not the whole background"""
    print("🎯 Detecting sofa precisely (not whole background)...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    h, w = room_img.shape[:2]
    print(f"   Image size: {w}x{h}")
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(room_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(room_img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Look for the specific sofa area (center-bottom of image)
    # The sofa is typically in the center-bottom area of living room images
    center_bottom_y = int(h * 0.3)  # Start from 30% down
    center_bottom_h = int(h * 0.7)  # Cover 70% of height
    
    # Focus on the center-bottom area where sofas typically are
    sofa_region = room_img[center_bottom_y:center_bottom_h, :]
    sofa_hsv = hsv[center_bottom_y:center_bottom_h, :]
    sofa_gray = gray[center_bottom_y:center_bottom_h, :]
    
    # Look for light gray furniture (sofa color)
    lower_gray = np.array([0, 0, 120])
    upper_gray = np.array([180, 50, 220])
    mask_gray = cv2.inRange(sofa_hsv, lower_gray, upper_gray)
    
    # Also look for white throw pillow
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(sofa_hsv, lower_white, upper_white)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask_gray, mask_white)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the sofa region
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour that could be the sofa
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 2000:  # Minimum area for sofa
            # Get bounding box in the sofa region
            x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
            
            # Adjust coordinates back to full image
            x_full = x
            y_full = y + center_bottom_y
            
            # Check if it's roughly sofa-shaped (wider than tall)
            aspect_ratio = w_rect / h_rect
            if aspect_ratio > 1.0:  # Sofa should be wider than tall
                # Create mask for full image
                sofa_mask = np.zeros((h, w), np.uint8)
                cv2.fillPoly(sofa_mask, [largest_contour + [0, center_bottom_y]], 255)
                
                # Smooth the mask
                sofa_mask = cv2.GaussianBlur(sofa_mask, (5, 5), 0)
                
                # Save mask
                mask_path = os.path.join(OUTPUT_DIR, "precise_sofa_mask.png")
                cv2.imwrite(mask_path, sofa_mask)
                
                print(f"✅ Sofa detected precisely: x={x_full}, y={y_full}, w={w_rect}, h={h_rect}")
                print(f"✅ Precise sofa mask saved: {mask_path}")
                return mask_path, sofa_mask, (x_full, y_full, w_rect, h_rect)
    
    # Fallback: Create a more precise manual mask based on typical sofa position
    print("⚠️ Using precise manual sofa area definition")
    
    # Based on typical sofa positioning in living room images
    # Sofa is usually in the center-bottom area
    sofa_x = int(w * 0.15)      # 15% from left
    sofa_y = int(h * 0.4)       # 40% from top (sofa area)
    sofa_w = int(w * 0.7)       # 70% of width
    sofa_h = int(h * 0.4)       # 40% of height
    
    # Create rectangular mask
    sofa_mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(sofa_mask, (sofa_x, sofa_y), (sofa_x + sofa_w, sofa_y + sofa_h), 255, -1)
    
    # Smooth the mask
    sofa_mask = cv2.GaussianBlur(sofa_mask, (10, 10), 0)
    
    # Save mask
    mask_path = os.path.join(OUTPUT_DIR, "manual_precise_sofa_mask.png")
    cv2.imwrite(mask_path, sofa_mask)
    
    print(f"✅ Manual precise sofa area: x={sofa_x}, y={sofa_y}, w={sofa_w}, h={sofa_h}")
    print(f"✅ Manual precise sofa mask saved: {mask_path}")
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
    
    # Apply gentle smoothing only to the inpainted area
    # Create a mask for the inpainted area
    inpainted_area = cv2.bitwise_and(mask, mask)
    
    # Apply bilateral filter only to the inpainted area
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

def calculate_natural_positioning(sofa_bbox, cutout_path):
    """Calculate natural positioning based on sofa area and new sofa size"""
    print("📐 Calculating natural positioning...")
    
    # Get original sofa area
    orig_x, orig_y, orig_w, orig_h = sofa_bbox
    
    # Load new sofa to get its dimensions
    cutout_img = Image.open(cutout_path).convert('RGBA')
    new_w, new_h = cutout_img.size
    
    print(f"   Original sofa area: {orig_w}x{orig_h}")
    print(f"   New sofa size: {new_w}x{new_h}")
    
    # Calculate scale to fit the sofa area naturally
    scale_w = orig_w / new_w
    scale_h = orig_h / new_h
    
    # Use the smaller scale to ensure it fits within the area
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_width = int(new_w * scale)
    new_height = int(new_h * scale)
    
    print(f"   Scale factor: {scale:.2f}")
    print(f"   Scaled size: {new_width}x{new_height}")
    
    # Position the new sofa in the center of the original sofa area
    new_x = orig_x + (orig_w - new_width) // 2
    new_y = orig_y + (orig_h - new_height) // 2
    
    print(f"   New position: x={new_x}, y={new_y}")
    
    return (new_x, new_y, new_width, new_height, scale)

def create_subtle_shadow(cutout_path, room_size, positioning):
    """Create a subtle shadow positioned correctly"""
    print("🌫️ Creating subtle shadow...")
    
    # Load the cutout
    cutout_img = Image.open(cutout_path).convert('RGBA')
    
    # Get positioning info
    new_x, new_y, new_width, new_height, scale = positioning
    
    # Create a very subtle horizontal shadow
    shadow_width = int(new_width * 1.1)
    shadow_height = int(new_height * 0.15)  # Very short shadow
    
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
    
    # Position shadow at the bottom of the new sofa
    shadow_x = new_x + (new_width - shadow_width) // 2
    shadow_y = new_y + new_height - shadow_height + 5  # Slightly below the sofa
    
    # Paste shadow
    shadow_img.paste(shadow_canvas, (shadow_x, shadow_y), shadow_canvas)
    
    # Save shadow
    shadow_path = os.path.join(OUTPUT_DIR, "subtle_shadow.png")
    shadow_img.save(shadow_path)
    
    print(f"✅ Subtle shadow saved: {shadow_path}")
    return shadow_path

def replace_sofa_naturally(clean_room_path, cutout_path, shadow_path, positioning):
    """Replace the sofa naturally in the same area with proper sizing"""
    print("🏠 Replacing sofa naturally...")
    
    # Load images
    room_img = Image.open(clean_room_path).convert('RGBA')
    cutout_img = Image.open(cutout_path).convert('RGBA')
    shadow_img = Image.open(shadow_path).convert('RGBA')
    
    # Get positioning info
    new_x, new_y, new_width, new_height, scale = positioning
    
    # Resize furniture to the calculated size
    cutout_resized = cutout_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create result canvas
    result_img = room_img.copy()
    
    # Add subtle shadow first
    result_img.paste(shadow_img, (0, 0), shadow_img)
    
    # Add the new sofa in the natural position
    result_img.paste(cutout_resized, (new_x, new_y), cutout_resized)
    
    # Save result
    result_path = os.path.join(OUTPUT_DIR, "precise_sofa_replacement.png")
    result_img.save(result_path)
    
    print(f"✅ Precise sofa replacement saved: {result_path}")
    print(f"   New sofa positioned at: x={new_x}, y={new_y}")
    print(f"   New sofa size: {new_width}x{new_height}")
    print(f"   Scale factor: {scale:.2f}")
    
    return result_path

def main():
    """Main precise sofa detection and replacement"""
    print("🛋️  PRECISE SOFA DETECTION AND REPLACEMENT")
    print("=" * 60)
    
    ensure_output_dir()
    
    try:
        # Step 1: Download furniture image
        furniture_path = download_furniture_image()
        
        # Step 2: Detect sofa precisely (not whole background)
        mask_path, mask, sofa_bbox = detect_sofa_precisely(ROOM_IMAGE)
        
        # Step 3: Remove sofa cleanly
        clean_room_path = remove_sofa_clean(ROOM_IMAGE, mask_path)
        
        # Step 4: Create sofa cutout
        cutout_path = create_sofa_cutout(furniture_path)
        
        # Step 5: Calculate natural positioning
        positioning = calculate_natural_positioning(sofa_bbox, cutout_path)
        
        # Step 6: Create subtle shadow
        room_img = Image.open(ROOM_IMAGE)
        shadow_path = create_subtle_shadow(cutout_path, room_img.size, positioning)
        
        # Step 7: Replace sofa naturally
        result_path = replace_sofa_naturally(
            clean_room_path, 
            cutout_path, 
            shadow_path,
            positioning
        )
        
        print("\n🎉 PRECISE SOFA REPLACEMENT COMPLETE!")
        print("=" * 60)
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"🎯 Precise sofa mask: {mask_path}")
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
        print(f"❌ Precise sofa replacement failed: {e}")
        raise

if __name__ == "__main__":
    main()
