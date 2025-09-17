#!/usr/bin/env python3
"""
Local Sofa Replacement Demo
===========================

This script demonstrates sofa replacement using local file processing:
1. Download the K-Kircher Home sofa image
2. Use auto matting to create cutout (local processing)
3. Use OpenCV to manually composite the sofa into the 445.png room
"""

import requests
import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

# Configuration
KKIRCHER_SOFA_URL = "https://k-kircher-home.myshopify.com/cdn/shop/files/0787477_sylvie-express-bench-sofa.jpg?v=1752147901&width=1440"
ROOM_IMAGE = "static/445.png"  # The bright modern room
OUTPUT_DIR = "sofa_replacement_output"

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

def create_sofa_cutout(image_path):
    """Create a cutout of the sofa using simple background removal"""
    print(f"🎨 Creating sofa cutout from {image_path}...")
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None, None
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Simple background removal using color thresholding
    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for white/light background (adjust these values)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    # Create mask for background
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Invert mask to get foreground (sofa)
    mask_inv = cv2.bitwise_not(mask)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
    
    # Create cutout with transparent background
    cutout = img_rgb.copy()
    cutout = np.dstack([cutout, mask_inv])  # Add alpha channel
    
    # Create shadow (simplified)
    shadow = np.zeros_like(img_rgb)
    shadow[:, :, :] = [50, 50, 50]  # Dark gray shadow
    shadow = np.dstack([shadow, (mask_inv * 0.3).astype(np.uint8)])  # Semi-transparent shadow
    
    # Save cutout
    cutout_path = os.path.join(OUTPUT_DIR, "sofa_cutout.png")
    Image.fromarray(cutout, 'RGBA').save(cutout_path)
    
    # Save shadow
    shadow_path = os.path.join(OUTPUT_DIR, "sofa_shadow.png")
    Image.fromarray(shadow, 'RGBA').save(shadow_path)
    
    print(f"✅ Cutout saved: {cutout_path}")
    print(f"✅ Shadow saved: {shadow_path}")
    
    return cutout_path, shadow_path

def composite_sofa_into_room(room_path, cutout_path, shadow_path, x=400, y=200):
    """Composite the sofa cutout into the room image"""
    print(f"🏠 Compositing sofa into room...")
    
    # Load room image
    room_img = Image.open(room_path).convert('RGBA')
    cutout_img = Image.open(cutout_path).convert('RGBA')
    shadow_img = Image.open(shadow_path).convert('RGBA')
    
    # Resize cutout to fit better (adjust scale as needed)
    scale = 0.8
    new_size = (int(cutout_img.width * scale), int(cutout_img.height * scale))
    cutout_img = cutout_img.resize(new_size, Image.Resampling.LANCZOS)
    shadow_img = shadow_img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Create a copy of the room for compositing
    result_img = room_img.copy()
    
    # First, composite the shadow (behind the sofa)
    shadow_x = x - 20  # Offset shadow slightly
    shadow_y = y + 10
    result_img.paste(shadow_img, (shadow_x, shadow_y), shadow_img)
    
    # Then, composite the sofa on top
    result_img.paste(cutout_img, (x, y), cutout_img)
    
    # Save the result
    result_path = os.path.join(OUTPUT_DIR, "sofa_replacement_result.png")
    result_img.save(result_path)
    
    print(f"✅ Sofa replacement completed: {result_path}")
    return result_path

def main():
    print("🛋️  LOCAL SOFA REPLACEMENT DEMO")
    print("=" * 50)
    
    # Step 1: Download the K-Kircher sofa
    sofa_path = download_image(KKIRCHER_SOFA_URL, "kkircher_sofa_original.jpg")
    
    # Step 2: Create cutout and shadow
    cutout_path, shadow_path = create_sofa_cutout(sofa_path)
    if not cutout_path or not shadow_path:
        print("❌ Cutout creation failed, cannot continue")
        return
    
    # Step 3: Composite into the room
    result_path = composite_sofa_into_room(ROOM_IMAGE, cutout_path, shadow_path)
    
    if result_path:
        print("\n🎉 SOFA REPLACEMENT COMPLETE!")
        print("=" * 50)
        print(f"📸 Original sofa: {sofa_path}")
        print(f"🏠 Room background: {ROOM_IMAGE}")
        print(f"✂️  Sofa cutout: {cutout_path}")
        print(f"🌫️  Sofa shadow: {shadow_path}")
        print(f"🖼️  Final result: {result_path}")
        
        # Open the result image
        try:
            result_img = Image.open(result_path)
            result_img.show()
            print("🖼️  Result image opened in default viewer")
        except Exception as e:
            print(f"⚠️  Could not open result image: {e}")
    else:
        print("❌ Sofa replacement failed")

if __name__ == "__main__":
    main()
