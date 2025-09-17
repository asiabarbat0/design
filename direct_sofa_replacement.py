#!/usr/bin/env python3
"""
Direct Sofa Replacement Demo
============================

This script does direct sofa replacement:
1. Keeps the original room as-is
2. Just replaces the sofa area directly
3. Focuses on getting the sofa to show up properly
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
OUTPUT_DIR = "direct_sofa_replacement"

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

def detect_sofa_simple(room_path):
    """Detect the sofa using a simple approach"""
    print(f"🔍 Detecting sofa in room image...")
    
    # Load the room image
    room_img = cv2.imread(room_path)
    if room_img is None:
        print(f"❌ Could not load room image: {room_path}")
        return None
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(room_img, cv2.COLOR_BGR2HSV)
    
    # Define range for light colored sofa
    lower_light = np.array([0, 0, 200])
    upper_light = np.array([180, 30, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_light, upper_light)
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (likely the sofa)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        print(f"✅ Sofa detected at position: ({x}, {y}) size: {w}x{h}")
        
        return {
            'bbox': (x, y, w, h),
            'contour': largest_contour
        }
    
    print("❌ No sofa detected")
    return None

def calibrate_sofa_fit(sofa_cutout_path, sofa_info):
    """Calibrate the sofa size and position to fit the detected area"""
    print(f"📏 Calibrating sofa fit...")
    
    # Load the sofa cutout
    sofa_img = Image.open(sofa_cutout_path).convert('RGBA')
    
    # Get the detected sofa dimensions
    x, y, w, h = sofa_info['bbox']
    
    # Calculate scale to fit the detected area
    # Make it slightly smaller to ensure it fits
    target_width = int(w * 0.9)
    target_height = int(h * 0.9)
    
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

def create_simple_shadow(sofa_path):
    """Create a simple shadow"""
    print(f"🌫️ Creating simple shadow...")
    
    # Load the sofa
    sofa_img = Image.open(sofa_path).convert('RGBA')
    
    # Create shadow by duplicating the sofa and making it dark
    shadow = sofa_img.copy()
    
    # Convert to numpy for processing
    shadow_array = np.array(shadow)
    
    # Make shadow dark and semi-transparent
    shadow_array[:, :, :3] = shadow_array[:, :, :3] * 0.3  # Darken RGB channels
    shadow_array[:, :, 3] = shadow_array[:, :, 3] * 0.5  # Reduce alpha
    
    # Apply blur for soft shadow
    shadow_img = Image.fromarray(shadow_array, 'RGBA')
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=10))
    
    # Save shadow
    shadow_path = os.path.join(OUTPUT_DIR, "sofa_shadow_simple.png")
    shadow_img.save(shadow_path)
    
    print(f"✅ Simple shadow saved: {shadow_path}")
    return shadow_path

def direct_sofa_replacement(room_path, sofa_path, shadow_path, sofa_info):
    """Directly replace the sofa without changing the background"""
    print(f"🏠 Directly replacing sofa in room...")
    
    # Load the room image
    room_img = Image.open(room_path).convert('RGBA')
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
    shadow_x = sofa_x - 10  # Offset shadow slightly
    shadow_y = sofa_y + 5
    result_img.paste(shadow_img, (shadow_x, shadow_y), shadow_img)
    
    # Then, composite the sofa on top
    result_img.paste(sofa_img, (sofa_x, sofa_y), sofa_img)
    
    # Save the result
    result_path = os.path.join(OUTPUT_DIR, "direct_sofa_replacement_result.png")
    result_img.save(result_path)
    
    print(f"✅ Direct sofa replacement completed: {result_path}")
    return result_path

def main():
    print("🛋️  DIRECT SOFA REPLACEMENT DEMO")
    print("=" * 50)
    
    # Step 1: Download the K-Kircher sofa
    sofa_path = download_image(KKIRCHER_SOFA_URL, "kkircher_sofa_original.jpg")
    
    # Step 2: Create high-quality cutout
    cutout_path = create_high_quality_cutout(sofa_path)
    if not cutout_path:
        print("❌ Cutout creation failed, cannot continue")
        return
    
    # Step 3: Detect sofa in the room
    sofa_info = detect_sofa_simple(ROOM_IMAGE)
    if not sofa_info:
        print("❌ Sofa detection failed, cannot continue")
        return
    
    # Step 4: Calibrate sofa fit
    calibrated_sofa_path, scale = calibrate_sofa_fit(cutout_path, sofa_info)
    
    # Step 5: Create simple shadow
    shadow_path = create_simple_shadow(calibrated_sofa_path)
    
    # Step 6: Direct replacement
    result_path = direct_sofa_replacement(ROOM_IMAGE, calibrated_sofa_path, shadow_path, sofa_info)
    
    if result_path:
        print("\n🎉 DIRECT SOFA REPLACEMENT COMPLETE!")
        print("=" * 50)
        print(f"📸 Original sofa: {sofa_path}")
        print(f"🏠 Room background: {ROOM_IMAGE}")
        print(f"✂️  High-quality cutout: {cutout_path}")
        print(f"📏 Calibrated sofa: {calibrated_sofa_path}")
        print(f"🌫️  Simple shadow: {shadow_path}")
        print(f"🖼️  Final result: {result_path}")
        
        # Open the result image
        try:
            result_img = Image.open(result_path)
            result_img.show()
            print("🖼️  Result image opened in default viewer")
        except Exception as e:
            print(f"⚠️  Could not open result image: {e}")
    else:
        print("❌ Direct sofa replacement failed")

if __name__ == "__main__":
    main()
