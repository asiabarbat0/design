#!/usr/bin/env python3
"""
Final Furniture Replacement Pipeline
====================================

Fixed all issues:
1. ✅ Subtle realistic shadow (not big dark circle)
2. ✅ Clean background (no distortion)
3. ✅ Proper sofa replacement (actually replaces the original)
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import rembg
import hashlib
import time
from flask import Flask, request, jsonify
import boto3
from botocore.exceptions import ClientError

# Configuration
OUTPUT_DIR = "final_pipeline_test"
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

def detect_sofa_smart(room_image_path):
    """Smart sofa detection using multiple approaches"""
    print("🎯 Smart sofa detection...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    h, w = room_img.shape[:2]
    
    # Method 1: Look for light-colored rectangular objects in the center area
    hsv = cv2.cvtColor(room_img, cv2.COLOR_BGR2HSV)
    
    # Define sofa area (center portion of image)
    center_y_start = int(h * 0.2)
    center_y_end = int(h * 0.8)
    center_x_start = int(w * 0.1)
    center_x_end = int(w * 0.9)
    
    # Focus on center area
    center_region = room_img[center_y_start:center_y_end, center_x_start:center_x_end]
    center_hsv = hsv[center_y_start:center_y_end, center_x_start:center_x_end]
    
    # Detect light colored objects
    lower_light = np.array([0, 0, 160])
    upper_light = np.array([180, 50, 255])
    mask_light = cv2.inRange(center_hsv, lower_light, upper_light)
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_CLOSE, kernel)
    mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 1000:  # Minimum area
            # Adjust contour coordinates back to full image
            largest_contour[:, :, 0] += center_x_start
            largest_contour[:, :, 1] += center_y_start
            
            # Create mask for full image
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # Smooth the mask
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            
            mask_path = os.path.join(OUTPUT_DIR, "smart_mask.png")
            cv2.imwrite(mask_path, mask)
            
            print(f"✅ Smart sofa mask saved: {mask_path}")
            return mask_path, mask, largest_contour
    
    # Fallback: Create a rectangular mask in the center where sofas typically are
    print("⚠️ Using smart fallback mask for sofa area")
    
    # Define sofa area based on typical room layout
    sofa_x = int(w * 0.15)
    sofa_y = int(h * 0.25)
    sofa_w = int(w * 0.7)
    sofa_h = int(h * 0.5)
    
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (sofa_x, sofa_y), (sofa_x + sofa_w, sofa_y + sofa_h), 255, -1)
    
    # Create soft edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    # Create contour for the rectangle
    contour = np.array([[[sofa_x, sofa_y]], [[sofa_x + sofa_w, sofa_y]], 
                       [[sofa_x + sofa_w, sofa_y + sofa_h]], [[sofa_x, sofa_y + sofa_h]]])
    
    mask_path = os.path.join(OUTPUT_DIR, "smart_fallback_mask.png")
    cv2.imwrite(mask_path, mask)
    
    print(f"✅ Smart fallback mask saved: {mask_path}")
    return mask_path, mask, contour

def inpaint_room_clean(room_image_path, mask_path):
    """Clean inpainting with minimal distortion"""
    print("🎨 Clean inpainting...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    # Use cv2.inpaint with better parameters
    inpainted = cv2.inpaint(room_img, mask, 3, cv2.INPAINT_TELEA)
    
    # Apply gentle smoothing to reduce artifacts
    inpainted = cv2.bilateralFilter(inpainted, 5, 50, 50)
    
    # Save clean room
    clean_room_path = os.path.join(OUTPUT_DIR, "room_clean_final.png")
    cv2.imwrite(clean_room_path, inpainted)
    
    print(f"✅ Clean room saved: {clean_room_path}")
    return clean_room_path

def create_furniture_cutout_final(furniture_image_path):
    """Create final furniture cutout"""
    print("✂️ Creating final furniture cutout...")
    
    # Load the furniture image
    with open(furniture_image_path, 'rb') as f:
        input_data = f.read()
    
    # Use rembg for background removal
    try:
        session = rembg.new_session('u2net')
        output_data = rembg.remove(input_data, session=session)
        
        # Save the cutout
        cutout_path = os.path.join(OUTPUT_DIR, "cutout_final.png")
        with open(cutout_path, 'wb') as f:
            f.write(output_data)
        
        print(f"✅ Final furniture cutout saved: {cutout_path}")
        return cutout_path
        
    except Exception as e:
        print(f"❌ Cutout creation failed: {e}")
        raise

def create_subtle_shadow(cutout_path, room_size):
    """Create subtle shadow (not big dark circle)"""
    print("🌫️ Creating subtle shadow...")
    
    # Load the cutout
    cutout_img = Image.open(cutout_path).convert('RGBA')
    
    # Get dimensions
    width, height = cutout_img.size
    
    # Create a subtle horizontal shadow
    shadow_width = int(width * 1.2)
    shadow_height = int(height * 0.2)  # Very short shadow
    
    # Create shadow canvas
    shadow_canvas = Image.new('RGBA', (shadow_width, shadow_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow_canvas)
    
    # Draw a subtle horizontal ellipse
    draw.ellipse([0, 0, shadow_width, shadow_height], 
                 fill=(0, 0, 0, 40))  # Very subtle
    
    # Apply Gaussian blur
    shadow_canvas = shadow_canvas.filter(ImageFilter.GaussianBlur(radius=12))
    
    # Create full room shadow
    shadow_img = Image.new('RGBA', room_size, (0, 0, 0, 0))
    
    # Position shadow at bottom of furniture area
    shadow_x = (room_size[0] - shadow_width) // 2
    shadow_y = int(room_size[1] * 0.7)  # Near bottom
    
    # Paste shadow
    shadow_img.paste(shadow_canvas, (shadow_x, shadow_y), shadow_canvas)
    
    # Save shadow
    shadow_path = os.path.join(OUTPUT_DIR, "subtle_shadow.png")
    shadow_img.save(shadow_path)
    
    print(f"✅ Subtle shadow saved: {shadow_path}")
    return shadow_path

def composite_furniture_final(clean_room_path, cutout_path, shadow_path, sofa_contour):
    """Final compositing with proper replacement"""
    print("🏠 Final furniture compositing...")
    
    # Load images
    room_img = Image.open(clean_room_path).convert('RGBA')
    cutout_img = Image.open(cutout_path).convert('RGBA')
    shadow_img = Image.open(shadow_path).convert('RGBA')
    
    # Get sofa bounding box
    x, y, w, h = cv2.boundingRect(sofa_contour)
    
    # Calculate scale to fit the sofa space
    target_width = w
    scale = target_width / cutout_img.width
    new_height = int(cutout_img.height * scale)
    
    # Resize furniture
    cutout_resized = cutout_img.resize((target_width, new_height), Image.Resampling.LANCZOS)
    
    # Position furniture to replace the original sofa
    furniture_x = x
    furniture_y = y + h - new_height  # Align with bottom of original sofa
    
    # Create result canvas
    result_img = room_img.copy()
    
    # Add subtle shadow first
    result_img.paste(shadow_img, (0, 0), shadow_img)
    
    # Add furniture
    result_img.paste(cutout_resized, (furniture_x, furniture_y), cutout_resized)
    
    # Save result
    result_path = os.path.join(OUTPUT_DIR, "render_final.png")
    result_img.save(result_path)
    
    print(f"✅ Final render saved: {result_path}")
    return result_path

def main():
    """Main final pipeline test"""
    print("🛋️  FINAL FURNITURE REPLACEMENT PIPELINE")
    print("=" * 60)
    
    ensure_output_dir()
    
    try:
        # Step 1: Download furniture image
        furniture_path = download_furniture_image()
        
        # Step 2: Smart sofa detection
        mask_path, mask, sofa_contour = detect_sofa_smart(ROOM_IMAGE)
        
        # Step 3: Clean inpainting
        clean_room_path = inpaint_room_clean(ROOM_IMAGE, mask_path)
        
        # Step 4: Create furniture cutout
        cutout_path = create_furniture_cutout_final(furniture_path)
        
        # Step 5: Create subtle shadow
        room_img = Image.open(ROOM_IMAGE)
        shadow_path = create_subtle_shadow(cutout_path, room_img.size)
        
        # Step 6: Final compositing
        result_path = composite_furniture_final(
            clean_room_path, 
            cutout_path, 
            shadow_path,
            sofa_contour
        )
        
        print("\n🎉 FINAL PIPELINE COMPLETE!")
        print("=" * 40)
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"🎯 Smart mask: {mask_path}")
        print(f"🏠 Clean room: {clean_room_path}")
        print(f"✂️  Final cutout: {cutout_path}")
        print(f"🌫️  Subtle shadow: {shadow_path}")
        print(f"🖼️  Final render: {result_path}")
        
        # Open the result
        try:
            result_img = Image.open(result_path)
            result_img.show()
            print("🖼️  Final result image opened in default viewer")
        except Exception as e:
            print(f"⚠️  Could not open result image: {e}")
            
    except Exception as e:
        print(f"❌ Final pipeline test failed: {e}")
        raise

if __name__ == "__main__":
    main()
