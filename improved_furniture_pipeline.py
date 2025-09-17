#!/usr/bin/env python3
"""
Improved Furniture Replacement Pipeline
======================================

Fixed version that addresses:
1. Big dark circle shadow - make it more subtle and realistic
2. Background distortion - better inpainting and blending
3. Cutout not replacing couch - ensure proper positioning and scaling
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
OUTPUT_DIR = "improved_pipeline_test"
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

def detect_sofa_precisely(room_image_path):
    """Detect sofa more precisely using multiple methods"""
    print("🎯 Detecting sofa precisely...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(room_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(room_img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Light colored furniture detection (more permissive)
    lower_light = np.array([0, 0, 150])
    upper_light = np.array([180, 60, 255])
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    
    # Method 2: Edge detection
    edges = cv2.Canny(gray, 20, 80)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Method 3: Combine methods
    combined_mask = cv2.bitwise_or(mask_light, edges)
    
    # Clean up the combined mask
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour that could be a sofa
    best_contour = None
    best_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Lower minimum area
            # Check if it's roughly rectangular (sofa-like)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 1.0 < aspect_ratio < 5.0:  # More permissive aspect ratio
                if area > best_area:
                    best_area = area
                    best_contour = contour
    
    if best_contour is not None:
        # Create precise mask
        mask = np.zeros(room_img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [best_contour], 255)
        
        # Smooth the mask edges
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Save mask
        mask_path = os.path.join(OUTPUT_DIR, "precise_mask.png")
        cv2.imwrite(mask_path, mask)
        
        print(f"✅ Precise sofa mask saved: {mask_path}")
        return mask_path, mask, best_contour
    else:
        # Fallback: create a simple rectangular mask in the center
        print("⚠️ No sofa detected, using fallback rectangular mask")
        h, w = room_img.shape[:2]
        x, y = int(w * 0.2), int(h * 0.3)
        w_rect, h_rect = int(w * 0.6), int(h * 0.4)
        
        mask = np.zeros(room_img.shape[:2], np.uint8)
        cv2.rectangle(mask, (x, y), (x + w_rect, y + h_rect), 255, -1)
        
        # Create a simple contour for the rectangle
        best_contour = np.array([[[x, y]], [[x + w_rect, y]], [[x + w_rect, y + h_rect]], [[x, y + h_rect]]])
        
        mask_path = os.path.join(OUTPUT_DIR, "fallback_mask.png")
        cv2.imwrite(mask_path, mask)
        
        print(f"✅ Fallback mask saved: {mask_path}")
        return mask_path, mask, best_contour

def inpaint_room_advanced(room_image_path, mask_path):
    """Advanced inpainting with better quality"""
    print("🎨 Advanced inpainting for clean room...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    # Method 1: Use cv2.inpaint with better parameters
    inpainted = cv2.inpaint(room_img, mask, 5, cv2.INPAINT_TELEA)
    
    # Method 2: Additional smoothing to reduce distortion
    # Apply bilateral filter to smooth the inpainted area
    inpainted = cv2.bilateralFilter(inpainted, 9, 75, 75)
    
    # Method 3: Blend with original image at edges
    # Create a soft edge mask
    edge_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    edge_mask = edge_mask.astype(np.float32) / 255.0
    
    # Blend inpainted and original
    room_float = room_img.astype(np.float32)
    inpainted_float = inpainted.astype(np.float32)
    
    # Create 3-channel edge mask
    edge_mask_3d = np.stack([edge_mask, edge_mask, edge_mask], axis=2)
    
    # Blend
    blended = room_float * (1 - edge_mask_3d) + inpainted_float * edge_mask_3d
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Save clean room
    clean_room_path = os.path.join(OUTPUT_DIR, "room_clean_advanced.png")
    cv2.imwrite(clean_room_path, blended)
    
    print(f"✅ Advanced clean room saved: {clean_room_path}")
    return clean_room_path

def create_furniture_cutout_improved(furniture_image_path):
    """Create improved furniture cutout"""
    print("✂️ Creating improved furniture cutout...")
    
    # Load the furniture image
    with open(furniture_image_path, 'rb') as f:
        input_data = f.read()
    
    # Use rembg for background removal
    try:
        session = rembg.new_session('u2net')
        output_data = rembg.remove(input_data, session=session)
        
        # Save the cutout
        cutout_path = os.path.join(OUTPUT_DIR, "cutout_improved.png")
        with open(cutout_path, 'wb') as f:
            f.write(output_data)
        
        # Post-process the cutout for better quality
        cutout_img = Image.open(cutout_path).convert('RGBA')
        
        # Enhance the image
        enhancer = ImageEnhance.Contrast(cutout_img)
        cutout_img = enhancer.enhance(1.1)
        
        # Save improved cutout
        cutout_img.save(cutout_path)
        
        print(f"✅ Improved furniture cutout saved: {cutout_path}")
        return cutout_path
        
    except Exception as e:
        print(f"❌ Cutout creation failed: {e}")
        raise

def create_realistic_shadow(cutout_path, room_dimensions):
    """Create realistic shadow instead of big dark circle"""
    print("🌫️ Creating realistic shadow...")
    
    # Load the cutout
    cutout_img = Image.open(cutout_path).convert('RGBA')
    
    # Get dimensions
    width, height = cutout_img.size
    
    # Create shadow that follows the furniture shape
    shadow_img = Image.new('RGBA', room_dimensions, (0, 0, 0, 0))
    
    # Create a more realistic shadow shape
    # Use the furniture silhouette but make it more shadow-like
    shadow_width = int(width * 1.1)
    shadow_height = int(height * 0.3)  # Much shorter shadow
    
    # Create shadow as a horizontal ellipse
    shadow_canvas = Image.new('RGBA', (shadow_width, shadow_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow_canvas)
    
    # Draw a horizontal ellipse (realistic shadow shape)
    draw.ellipse([0, 0, shadow_width, shadow_height], 
                 fill=(0, 0, 0, 60))  # Much more subtle
    
    # Apply Gaussian blur for softness
    shadow_canvas = shadow_canvas.filter(ImageFilter.GaussianBlur(radius=8))
    
    # Position shadow at the bottom of the furniture
    shadow_x = (room_dimensions[0] - shadow_width) // 2
    shadow_y = room_dimensions[1] - shadow_height - 50  # Near bottom
    
    # Paste shadow onto room canvas
    shadow_img.paste(shadow_canvas, (shadow_x, shadow_y), shadow_canvas)
    
    # Save shadow
    shadow_path = os.path.join(OUTPUT_DIR, "realistic_shadow.png")
    shadow_img.save(shadow_path)
    
    print(f"✅ Realistic shadow saved: {shadow_path}")
    return shadow_path

def composite_furniture_improved(clean_room_path, cutout_path, shadow_path, sofa_contour):
    """Improved compositing with proper positioning"""
    print("🏠 Improved furniture compositing...")
    
    # Load images
    room_img = Image.open(clean_room_path).convert('RGBA')
    cutout_img = Image.open(cutout_path).convert('RGBA')
    shadow_img = Image.open(shadow_path).convert('RGBA')
    
    # Get sofa bounding box for proper positioning
    x, y, w, h = cv2.boundingRect(sofa_contour)
    
    # Calculate scale to fit the sofa space
    target_width = w
    scale = target_width / cutout_img.width
    new_height = int(cutout_img.height * scale)
    
    # Resize furniture using LANCZOS
    cutout_resized = cutout_img.resize((target_width, new_height), Image.Resampling.LANCZOS)
    
    # Position furniture to replace the original sofa
    furniture_x = x
    furniture_y = y + h - new_height  # Align with bottom of original sofa
    
    # Create result canvas
    result_img = room_img.copy()
    
    # First, composite the shadow (subtle)
    result_img.paste(shadow_img, (0, 0), shadow_img)
    
    # Then, composite the furniture
    result_img.paste(cutout_resized, (furniture_x, furniture_y), cutout_resized)
    
    # Save result
    result_path = os.path.join(OUTPUT_DIR, "render_improved.png")
    result_img.save(result_path)
    
    print(f"✅ Improved render saved: {result_path}")
    return result_path

def main():
    """Main improved pipeline test"""
    print("🛋️  IMPROVED FURNITURE REPLACEMENT PIPELINE")
    print("=" * 60)
    
    ensure_output_dir()
    
    try:
        # Step 1: Download furniture image
        furniture_path = download_furniture_image()
        
        # Step 2: Detect sofa precisely
        mask_path, mask, sofa_contour = detect_sofa_precisely(ROOM_IMAGE)
        
        # Step 3: Advanced inpainting
        clean_room_path = inpaint_room_advanced(ROOM_IMAGE, mask_path)
        
        # Step 4: Create improved furniture cutout
        cutout_path = create_furniture_cutout_improved(furniture_path)
        
        # Step 5: Create realistic shadow
        room_img = Image.open(ROOM_IMAGE)
        shadow_path = create_realistic_shadow(cutout_path, room_img.size)
        
        # Step 6: Improved compositing
        result_path = composite_furniture_improved(
            clean_room_path, 
            cutout_path, 
            shadow_path,
            sofa_contour
        )
        
        print("\n🎉 IMPROVED PIPELINE COMPLETE!")
        print("=" * 40)
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"🎯 Precise mask: {mask_path}")
        print(f"🏠 Clean room: {clean_room_path}")
        print(f"✂️  Improved cutout: {cutout_path}")
        print(f"🌫️  Realistic shadow: {shadow_path}")
        print(f"🖼️  Final render: {result_path}")
        
        # Open the result
        try:
            result_img = Image.open(result_path)
            result_img.show()
            print("🖼️  Improved result image opened in default viewer")
        except Exception as e:
            print(f"⚠️  Could not open result image: {e}")
            
    except Exception as e:
        print(f"❌ Improved pipeline test failed: {e}")
        raise

if __name__ == "__main__":
    main()
