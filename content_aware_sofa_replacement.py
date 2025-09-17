#!/usr/bin/env python3
"""
Content-Aware Sofa Replacement Demo
===================================

This script uses content-aware techniques to replace the sofa:
1. Uses the original room as base and only replaces the sofa area
2. Uses content-aware fill for better background reconstruction
3. Blends edges naturally for seamless integration
"""

import requests
import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import rembg
from scipy import ndimage
from skimage import restoration, segmentation, morphology

# Configuration
KKIRCHER_SOFA_URL = "https://k-kircher-home.myshopify.com/cdn/shop/files/0787477_sylvie-express-bench-sofa.jpg?v=1752147901&width=1440"
ROOM_IMAGE = "static/445.png"  # The bright modern room
OUTPUT_DIR = "content_aware_sofa_replacement"

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
    """Detect the sofa more precisely using multiple techniques"""
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
    
    # Method 1: Color-based detection
    lower_light = np.array([0, 0, 200])
    upper_light = np.array([180, 30, 255])
    mask_color = cv2.inRange(hsv, lower_light, upper_light)
    
    # Method 2: Edge-based detection
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Method 3: Texture-based detection
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
        
        # Create a more precise mask for the sofa
        sofa_mask = np.zeros(room_img.shape[:2], np.uint8)
        cv2.fillPoly(sofa_mask, [largest_contour], 255)
        
        # Apply some smoothing to the mask
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

def content_aware_fill(room_img, mask):
    """Use content-aware fill to reconstruct the background"""
    print(f"🎨 Using content-aware fill for background reconstruction...")
    
    # Convert to float for better processing
    room_float = room_img.astype(np.float32) / 255.0
    
    # Create a more sophisticated mask
    # Dilate the mask slightly to ensure we cover the entire sofa area
    kernel = np.ones((5,5), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=2)
    
    # Create a soft mask for blending
    mask_soft = cv2.GaussianBlur(mask_dilated, (15, 15), 0) / 255.0
    
    # Method 1: Use the surrounding pixels to fill
    # Get the boundary of the mask
    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Create a distance transform
        dist_transform = cv2.distanceTransform(mask_dilated, cv2.DIST_L2, 5)
        
        # Normalize the distance transform
        dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Use the distance transform to guide the filling
        # Areas closer to the boundary get more influence from surrounding pixels
        
        # Create a filled version using the surrounding area
        filled = room_float.copy()
        
        # For each channel
        for c in range(3):
            channel = room_float[:, :, c]
            
            # Use the surrounding pixels to fill
            # This is a simple approach - in practice, you'd use more sophisticated methods
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            
            # Apply the kernel to fill the masked area
            filled_channel = cv2.filter2D(channel, -1, kernel)
            
            # Blend with the original using the distance transform
            filled[:, :, c] = channel * (1 - mask_soft) + filled_channel * mask_soft
        
        # Convert back to uint8
        filled = (filled * 255).astype(np.uint8)
        
        # Save the result
        result_path = os.path.join(OUTPUT_DIR, "room_content_aware_filled.png")
        Image.fromarray(filled).save(result_path)
        
        print(f"✅ Content-aware fill completed: {result_path}")
        return result_path
    
    return None

def calibrate_sofa_fit(sofa_cutout_path, sofa_info):
    """Calibrate the sofa size and position to fit the detected area"""
    print(f"📏 Calibrating sofa fit...")
    
    # Load the sofa cutout
    sofa_img = Image.open(sofa_cutout_path).convert('RGBA')
    
    # Get the detected sofa dimensions
    x, y, w, h = sofa_info['bbox']
    
    # Calculate scale to fit the detected area
    # Add some padding (5% on each side)
    target_width = int(w * 1.1)
    target_height = int(h * 1.1)
    
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
    shadow_path = os.path.join(OUTPUT_DIR, "sofa_shadow_calibrated.png")
    shadow_img.save(shadow_path)
    
    print(f"✅ Realistic shadow saved: {shadow_path}")
    return shadow_path

def composite_sofa_with_blending(room_filled_path, sofa_path, shadow_path, sofa_info):
    """Composite the sofa with advanced blending"""
    print(f"🏠 Compositing sofa with advanced blending...")
    
    # Load images
    room_img = Image.open(room_filled_path).convert('RGBA')
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
    
    # Then, composite the sofa on top with blending
    # Create a soft edge for the sofa
    sofa_array = np.array(sofa_img)
    alpha = sofa_array[:, :, 3] / 255.0
    
    # Apply a soft edge to the alpha channel
    alpha_soft = cv2.GaussianBlur(alpha, (5, 5), 0)
    sofa_array[:, :, 3] = (alpha_soft * 255).astype(np.uint8)
    
    # Convert back to PIL
    sofa_soft = Image.fromarray(sofa_array, 'RGBA')
    
    # Composite with blending
    result_img.paste(sofa_soft, (sofa_x, sofa_y), sofa_soft)
    
    # Save the result
    result_path = os.path.join(OUTPUT_DIR, "content_aware_sofa_replacement_result.png")
    result_img.save(result_path)
    
    print(f"✅ Content-aware sofa replacement completed: {result_path}")
    return result_path

def main():
    print("🛋️  CONTENT-AWARE SOFA REPLACEMENT DEMO")
    print("=" * 50)
    
    # Step 1: Download the K-Kircher sofa
    sofa_path = download_image(KKIRCHER_SOFA_URL, "kkircher_sofa_original.jpg")
    
    # Step 2: Create high-quality cutout
    cutout_path = create_high_quality_cutout(sofa_path)
    if not cutout_path:
        print("❌ Cutout creation failed, cannot continue")
        return
    
    # Step 3: Detect sofa precisely
    sofa_info = detect_sofa_precisely(ROOM_IMAGE)
    if not sofa_info:
        print("❌ Sofa detection failed, cannot continue")
        return
    
    # Step 4: Load the room image
    room_img = cv2.imread(ROOM_IMAGE)
    room_rgb = cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB)
    
    # Step 5: Use content-aware fill
    room_filled_path = content_aware_fill(room_rgb, sofa_info['mask'])
    if not room_filled_path:
        print("❌ Content-aware fill failed, cannot continue")
        return
    
    # Step 6: Calibrate sofa fit
    calibrated_sofa_path, scale = calibrate_sofa_fit(cutout_path, sofa_info)
    
    # Step 7: Create realistic shadow
    shadow_path = create_realistic_shadow(calibrated_sofa_path, sofa_info, scale)
    
    # Step 8: Composite with blending
    result_path = composite_sofa_with_blending(room_filled_path, calibrated_sofa_path, shadow_path, sofa_info)
    
    if result_path:
        print("\n🎉 CONTENT-AWARE SOFA REPLACEMENT COMPLETE!")
        print("=" * 50)
        print(f"📸 Original sofa: {sofa_path}")
        print(f"🏠 Room background: {ROOM_IMAGE}")
        print(f"🏠 Room filled: {room_filled_path}")
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
        print("❌ Content-aware sofa replacement failed")

if __name__ == "__main__":
    main()
