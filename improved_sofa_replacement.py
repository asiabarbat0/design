#!/usr/bin/env python3
"""
Improved Sofa Replacement Demo
==============================

This script improves the sofa replacement by:
1. Using better background removal (rembg library)
2. Actually removing the original sofa from the room
3. Better positioning and scaling
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
OUTPUT_DIR = "improved_sofa_replacement"

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

def remove_original_sofa(room_path):
    """Remove the original sofa from the room using inpainting"""
    print(f"🏠 Removing original sofa from room...")
    
    # Load the room image
    room_img = cv2.imread(room_path)
    if room_img is None:
        print(f"❌ Could not load room image: {room_path}")
        return None
    
    # Convert to RGB
    room_rgb = cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB)
    
    # Create a mask for the sofa area (you may need to adjust these coordinates)
    height, width = room_rgb.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    
    # Define the sofa area (adjust these coordinates based on the 445.png sofa position)
    sofa_x1, sofa_y1 = int(width * 0.2), int(height * 0.3)
    sofa_x2, sofa_y2 = int(width * 0.8), int(height * 0.9)
    
    # Create rectangular mask for sofa area
    cv2.rectangle(mask, (sofa_x1, sofa_y1), (sofa_x2, sofa_y2), 255, -1)
    
    # Apply Gaussian blur to soften edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # Use inpainting to remove the sofa
    inpainted = cv2.inpaint(room_rgb, mask, 3, cv2.INPAINT_TELEA)
    
    # Save the room without sofa
    room_no_sofa_path = os.path.join(OUTPUT_DIR, "room_no_sofa.png")
    Image.fromarray(inpainted).save(room_no_sofa_path)
    
    print(f"✅ Room without sofa saved: {room_no_sofa_path}")
    return room_no_sofa_path

def create_realistic_shadow(cutout_path, room_dimensions):
    """Create a more realistic shadow"""
    print(f"🌫️ Creating realistic shadow...")
    
    # Load cutout
    cutout_img = Image.open(cutout_path).convert('RGBA')
    
    # Create shadow by duplicating the cutout and making it dark
    shadow = cutout_img.copy()
    
    # Convert to numpy for processing
    shadow_array = np.array(shadow)
    
    # Make shadow dark and semi-transparent
    shadow_array[:, :, :3] = shadow_array[:, :, :3] * 0.2  # Darken RGB channels
    shadow_array[:, :, 3] = shadow_array[:, :, 3] * 0.4  # Reduce alpha
    
    # Apply blur for soft shadow
    shadow_img = Image.fromarray(shadow_array, 'RGBA')
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=10))
    
    # Save shadow
    shadow_path = os.path.join(OUTPUT_DIR, "sofa_shadow_realistic.png")
    shadow_img.save(shadow_path)
    
    print(f"✅ Realistic shadow saved: {shadow_path}")
    return shadow_path

def composite_sofa_into_room(room_no_sofa_path, cutout_path, shadow_path):
    """Composite the sofa into the room with proper positioning"""
    print(f"🏠 Compositing sofa into room...")
    
    # Load images
    room_img = Image.open(room_no_sofa_path).convert('RGBA')
    cutout_img = Image.open(cutout_path).convert('RGBA')
    shadow_img = Image.open(shadow_path).convert('RGBA')
    
    # Get room dimensions
    room_width, room_height = room_img.size
    
    # Scale the sofa to fit the room properly
    # Calculate scale based on room width (sofa should be about 60% of room width)
    target_width = int(room_width * 0.6)
    scale = target_width / cutout_img.width
    
    new_size = (int(cutout_img.width * scale), int(cutout_img.height * scale))
    cutout_img = cutout_img.resize(new_size, Image.Resampling.LANCZOS)
    shadow_img = shadow_img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Position the sofa in the center-bottom area where the original sofa was
    x = int((room_width - cutout_img.width) / 2)
    y = int(room_height - cutout_img.height - 50)  # 50px from bottom
    
    # Create a copy of the room for compositing
    result_img = room_img.copy()
    
    # First, composite the shadow (behind the sofa)
    shadow_x = x - 20  # Offset shadow slightly
    shadow_y = y + 10
    result_img.paste(shadow_img, (shadow_x, shadow_y), shadow_img)
    
    # Then, composite the sofa on top
    result_img.paste(cutout_img, (x, y), cutout_img)
    
    # Save the result
    result_path = os.path.join(OUTPUT_DIR, "improved_sofa_replacement_result.png")
    result_img.save(result_path)
    
    print(f"✅ Improved sofa replacement completed: {result_path}")
    return result_path

def main():
    print("🛋️  IMPROVED SOFA REPLACEMENT DEMO")
    print("=" * 50)
    
    # Step 1: Download the K-Kircher sofa
    sofa_path = download_image(KKIRCHER_SOFA_URL, "kkircher_sofa_original.jpg")
    
    # Step 2: Create high-quality cutout
    cutout_path = create_high_quality_cutout(sofa_path)
    if not cutout_path:
        print("❌ Cutout creation failed, cannot continue")
        return
    
    # Step 3: Remove original sofa from room
    room_no_sofa_path = remove_original_sofa(ROOM_IMAGE)
    if not room_no_sofa_path:
        print("❌ Failed to remove original sofa, cannot continue")
        return
    
    # Step 4: Create realistic shadow
    room_img = Image.open(ROOM_IMAGE)
    shadow_path = create_realistic_shadow(cutout_path, room_img.size)
    
    # Step 5: Composite into the room
    result_path = composite_sofa_into_room(room_no_sofa_path, cutout_path, shadow_path)
    
    if result_path:
        print("\n🎉 IMPROVED SOFA REPLACEMENT COMPLETE!")
        print("=" * 50)
        print(f"📸 Original sofa: {sofa_path}")
        print(f"🏠 Room background: {ROOM_IMAGE}")
        print(f"🏠 Room without sofa: {room_no_sofa_path}")
        print(f"✂️  High-quality cutout: {cutout_path}")
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
        print("❌ Improved sofa replacement failed")

if __name__ == "__main__":
    main()
