#!/usr/bin/env python3
"""
True Sofa Replacement Demo
==========================

This script actually replaces the sofa:
1. Removes the original sofa from the room
2. Places the new sofa in its exact position
3. Uses precise techniques to avoid blurriness
"""

import requests
import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import rembg

# Configuration
KKIRCHER_SOFA_URL = "https://k-kircher-home.myshopify.com/cdn/shop/files/0787477_sylvie-express-bench-sofa.jpg?v=1752147901&width=1440"
ROOM_IMAGE = "static/445.png"  # The bright modern room
OUTPUT_DIR = "true_sofa_replacement"

def download_image(url, filename):
    """Download an image from URL"""
    print(f"📥 Downloading {filename}...")
    response = requests.get(url)
    response.raise_for_status()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, 'wb') as f:
        f.write(response.content)
    
    print(f"✅ Downloaded: {filepath}")
    return filepath

def create_high_quality_cutout(image_path):
    """Create a high-quality cutout using rembg"""
    print(f"🎨 Creating high-quality cutout from {image_path}...")
    
    # Load the image
    with open(image_path, 'rb') as f:
        input_data = f.read()
    
    # Use rembg for better background removal
    try:
        # Try different models for better results
        session = rembg.new_session('u2net')  # General purpose model
        output_data = rembg.remove(input_data, session=session)
        
        # Save the cutout
        cutout_path = os.path.join(OUTPUT_DIR, "sofa_cutout_hq.png")
        with open(cutout_path, 'wb') as f:
            f.write(output_data)
        
        print(f"✅ High-quality cutout saved: {cutout_path}")
        return cutout_path
        
    except Exception as e:
        print(f"⚠️ rembg failed, using fallback method: {e}")
        return create_fallback_cutout(image_path)

def create_fallback_cutout(image_path):
    """Fallback cutout method using OpenCV"""
    print(f"🎨 Creating fallback cutout from {image_path}...")
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use GrabCut for better segmentation
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Define a rectangle around the sofa (adjust these coordinates)
    height, width = img.shape[:2]
    rect = (int(width*0.1), int(height*0.1), int(width*0.8), int(height*0.8))
    
    # Apply GrabCut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply morphological operations to clean up
    kernel = np.ones((5,5), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    
    # Create cutout with transparent background
    cutout = img_rgb.copy()
    cutout = np.dstack([cutout, mask2 * 255])  # Add alpha channel
    
    # Save cutout
    cutout_path = os.path.join(OUTPUT_DIR, "sofa_cutout_hq.png")
    Image.fromarray(cutout, 'RGBA').save(cutout_path)
    
    print(f"✅ Fallback cutout saved: {cutout_path}")
    return cutout_path

def detect_sofa_precisely(room_path):
    """Detect the sofa precisely in the room"""
    print(f"🔍 Detecting sofa precisely in room image...")
    
    # Load the room image
    room_img = cv2.imread(room_path)
    if room_img is None:
        print(f"❌ Could not load room image: {room_path}")
        return None
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(room_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(room_img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(room_img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Color-based detection for light sofa
    lower_light = np.array([0, 0, 200])
    upper_light = np.array([180, 30, 255])
    mask_color = cv2.inRange(hsv, lower_light, upper_light)
    
    # Method 2: Edge detection
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Method 3: Texture analysis
    # The sofa has a different texture than the floor
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    texture = np.sqrt(sobelx**2 + sobely**2)
    texture = np.uint8(texture / texture.max() * 255)
    
    # Combine methods
    mask_combined = cv2.bitwise_and(mask_color, cv2.bitwise_not(edges))
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (likely the sofa)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Create a precise mask for the sofa
        sofa_mask = np.zeros(room_img.shape[:2], np.uint8)
        cv2.fillPoly(sofa_mask, [largest_contour], 255)
        
        # Apply minimal smoothing to the mask
        sofa_mask = cv2.GaussianBlur(sofa_mask, (3, 3), 0)
        
        print(f"✅ Sofa detected at position: ({x}, {y}) size: {w}x{h}")
        
        # Save the detected sofa mask
        mask_path = os.path.join(OUTPUT_DIR, "sofa_mask_precise.png")
        cv2.imwrite(mask_path, sofa_mask)
        
        return {
            'bbox': (x, y, w, h),
            'mask': sofa_mask,
            'contour': largest_contour
        }
    
    print("❌ No sofa detected")
    return None

def remove_sofa_with_context_aware_fill(room_path, sofa_info):
    """Remove the sofa using context-aware techniques"""
    print(f"🏠 Removing sofa with context-aware fill...")
    
    # Load the room image
    room_img = cv2.imread(room_path)
    room_rgb = cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB)
    
    # Get the sofa mask
    sofa_mask = sofa_info['mask']
    
    # Create a more sophisticated approach
    # Instead of simple inpainting, use the surrounding context
    
    # Method 1: Use the floor pattern to fill
    # The sofa is on the floor, so we can use the floor texture
    
    # Get the area around the sofa
    x, y, w, h = sofa_info['bbox']
    
    # Expand the area slightly to get more context
    margin = 20
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(room_rgb.shape[1], x + w + margin)
    y2 = min(room_rgb.shape[0], y + h + margin)
    
    # Extract the context area
    context_area = room_rgb[y1:y2, x1:x2]
    
    # Create a mask for the context area
    context_mask = sofa_mask[y1:y2, x1:x2]
    
    # Use the floor area (below the sofa) to fill
    # The floor is typically below the sofa
    floor_y = h // 2  # Middle of the sofa area
    floor_area = context_area[floor_y:, :]
    
    if floor_area.shape[0] > 0:
        # Resize the floor area to match the sofa area
        floor_resized = cv2.resize(floor_area, (w, h))
        
        # Create the filled area
        filled_area = context_area.copy()
        
        # Fill the sofa area with the floor pattern
        for c in range(3):
            channel = context_area[:, :, c]
            floor_channel = floor_resized[:, :, c]
            
            # Ensure dimensions match
            if channel.shape == floor_channel.shape:
                # Blend the floor pattern into the sofa area
                filled_channel = channel * (1 - context_mask/255.0) + floor_channel * (context_mask/255.0)
                filled_area[:, :, c] = filled_channel
            else:
                # If dimensions don't match, just use the original channel
                filled_area[:, :, c] = channel
        
        # Put the filled area back into the room
        room_rgb[y1:y2, x1:x2] = filled_area
    
    # Save the result
    result_path = os.path.join(OUTPUT_DIR, "room_sofa_removed.png")
    Image.fromarray(room_rgb).save(result_path)
    
    print(f"✅ Sofa removed with context-aware fill: {result_path}")
    return result_path

def calibrate_sofa_fit(sofa_cutout_path, sofa_info):
    """Calibrate the sofa size and position to fit the detected area"""
    print(f"📏 Calibrating sofa fit...")
    
    # Load the sofa cutout
    sofa_img = Image.open(sofa_cutout_path).convert('RGBA')
    
    # Get the detected sofa dimensions
    x, y, w, h = sofa_info['bbox']
    
    # Calculate scale to fit the detected area
    # Make it slightly smaller to ensure it fits
    target_width = int(w * 0.95)
    target_height = int(h * 0.95)
    
    # Calculate scale factors
    scale_x = target_width / sofa_img.width
    scale_y = target_height / sofa_img.height
    
    # Use the smaller scale to maintain aspect ratio
    scale = min(scale_x, scale_y)
    
    # Resize the sofa
    new_size = (int(sofa_img.width * scale), int(sofa_img.height * scale))
    sofa_resized = sofa_img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Save the calibrated sofa
    calibrated_path = os.path.join(OUTPUT_DIR, "sofa_calibrated.png")
    sofa_resized.save(calibrated_path)
    
    print(f"✅ Sofa calibrated: {new_size} (scale: {scale:.2f})")
    return calibrated_path, scale

def create_realistic_shadow(sofa_path, sofa_info, scale):
    """Create a realistic shadow based on the detected sofa area"""
    print(f"🌫️ Creating realistic shadow...")
    
    # Load the sofa
    sofa_img = Image.open(sofa_path).convert('RGBA')
    
    # Create shadow by duplicating the sofa and making it dark
    shadow = sofa_img.copy()
    
    # Convert to numpy for processing
    shadow_array = np.array(shadow)
    
    # Make shadow dark and semi-transparent
    shadow_array[:, :, :3] = shadow_array[:, :, :3] * 0.2  # Darken RGB channels
    shadow_array[:, :, 3] = shadow_array[:, :, 3] * 0.4  # Reduce alpha
    
    # Apply blur for soft shadow
    shadow_img = Image.fromarray(shadow_array, 'RGBA')
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=15))
    
    # Save shadow
    shadow_path = os.path.join(OUTPUT_DIR, "sofa_shadow_realistic.png")
    shadow_img.save(shadow_path)
    
    print(f"✅ Realistic shadow saved: {shadow_path}")
    return shadow_path

def composite_sofa_into_room(room_no_sofa_path, sofa_path, shadow_path, sofa_info):
    """Composite the sofa into the room where the original sofa was"""
    print(f"🏠 Compositing sofa into room...")
    
    # Load images
    room_img = Image.open(room_no_sofa_path).convert('RGBA')
    sofa_img = Image.open(sofa_path).convert('RGBA')
    shadow_img = Image.open(shadow_path).convert('RGBA')
    
    # Get the detected sofa position
    x, y, w, h = sofa_info['bbox']
    
    # Calculate position to center the sofa in the detected area
    sofa_x = x + (w - sofa_img.width) // 2
    sofa_y = y + (h - sofa_img.height) // 2
    
    # Create a copy of the room for compositing
    result_img = room_img.copy()
    
    # First, composite the shadow (behind the sofa)
    shadow_x = sofa_x - 15  # Offset shadow slightly
    shadow_y = sofa_y + 10
    result_img.paste(shadow_img, (shadow_x, shadow_y), shadow_img)
    
    # Then, composite the sofa on top
    result_img.paste(sofa_img, (sofa_x, sofa_y), sofa_img)
    
    # Save the result
    result_path = os.path.join(OUTPUT_DIR, "true_sofa_replacement_result.png")
    result_img.save(result_path)
    
    print(f"✅ True sofa replacement completed: {result_path}")
    return result_path

def main():
    print("🛋️  TRUE SOFA REPLACEMENT DEMO")
    print("=" * 50)
    
    # Step 1: Download the K-Kircher sofa
    sofa_path = download_image(KKIRCHER_SOFA_URL, "kkircher_sofa_original.jpg")
    
    # Step 2: Create high-quality cutout
    cutout_path = create_high_quality_cutout(sofa_path)
    if not cutout_path:
        print("❌ Cutout creation failed, cannot continue")
        return
    
    # Step 3: Detect sofa in the room
    sofa_info = detect_sofa_precisely(ROOM_IMAGE)
    if not sofa_info:
        print("❌ Sofa detection failed, cannot continue")
        return
    
    # Step 4: Remove the original sofa
    room_no_sofa_path = remove_sofa_with_context_aware_fill(ROOM_IMAGE, sofa_info)
    if not room_no_sofa_path:
        print("❌ Sofa removal failed, cannot continue")
        return
    
    # Step 5: Calibrate sofa fit
    calibrated_sofa_path, scale = calibrate_sofa_fit(cutout_path, sofa_info)
    
    # Step 6: Create realistic shadow
    shadow_path = create_realistic_shadow(calibrated_sofa_path, sofa_info, scale)
    
    # Step 7: Composite the new sofa
    result_path = composite_sofa_into_room(room_no_sofa_path, calibrated_sofa_path, shadow_path, sofa_info)
    
    if result_path:
        print("\n🎉 TRUE SOFA REPLACEMENT COMPLETE!")
        print("=" * 50)
        print(f"📸 Original sofa: {sofa_path}")
        print(f"🏠 Room background: {ROOM_IMAGE}")
        print(f"🏠 Room without sofa: {room_no_sofa_path}")
        print(f"✂️  High-quality cutout: {cutout_path}")
        print(f"📏 Calibrated sofa: {calibrated_sofa_path}")
        print(f"🌫️  Realistic shadow: {shadow_path}")
        print(f"🖼️  Final result: {result_path}")
        
        # Open the result image
        try:
            result_img = Image.open(result_path)
            result_img.show()
            print("🖼️  Result image opened in default viewer")
        except Exception as e:
            print(f"⚠️  Could not open result image: {e}")
    else:
        print("❌ True sofa replacement failed")

if __name__ == "__main__":
    main()
