#!/usr/bin/env python3
"""
Manual Sofa Replacement
=======================

This version uses manual masking to precisely target only the sofa
and ensure proper replacement without background blur.
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
OUTPUT_DIR = "manual_sofa_replacement"
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

def create_manual_sofa_mask(room_image_path):
    """Create a manual mask for the sofa area only"""
    print("🎯 Creating manual sofa mask...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    h, w = room_img.shape[:2]
    print(f"   Image size: {w}x{h}")
    
    # Create a precise manual mask for the sofa
    # Based on the image description, the sofa is in the center-bottom area
    # and is light gray with a white throw pillow
    
    # Define the sofa area more precisely
    # The sofa takes up roughly the center portion of the image
    sofa_x = int(w * 0.2)       # 20% from left
    sofa_y = int(h * 0.4)       # 40% from top (sofa starts here)
    sofa_w = int(w * 0.6)       # 60% of width
    sofa_h = int(h * 0.35)      # 35% of height
    
    print(f"   Manual sofa area: x={sofa_x}, y={sofa_y}, w={sofa_w}, h={sofa_h}")
    
    # Create a more precise mask using multiple rectangles to match sofa shape
    mask = np.zeros((h, w), np.uint8)
    
    # Main sofa body (rectangular)
    cv2.rectangle(mask, (sofa_x, sofa_y), (sofa_x + sofa_w, sofa_y + sofa_h), 255, -1)
    
    # Add some rounded corners to make it more sofa-like
    # This creates a more natural sofa shape
    corner_size = 20
    cv2.rectangle(mask, (sofa_x + corner_size, sofa_y), (sofa_x + sofa_w - corner_size, sofa_y + sofa_h), 255, -1)
    
    # Smooth the mask edges for better blending
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    # Save mask
    mask_path = os.path.join(OUTPUT_DIR, "manual_sofa_mask.png")
    cv2.imwrite(mask_path, mask)
    
    print(f"✅ Manual sofa mask saved: {mask_path}")
    return mask_path, mask, (sofa_x, sofa_y, sofa_w, sofa_h)

def remove_sofa_with_context_fill(room_image_path, mask_path):
    """Remove sofa using context-aware fill to avoid blur"""
    print("🎨 Removing sofa with context-aware fill...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    # Convert to different color spaces for better analysis
    hsv = cv2.cvtColor(room_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(room_img, cv2.COLOR_BGR2LAB)
    
    # Create a more sophisticated inpainting approach
    # First, try to use the surrounding context to fill the sofa area
    
    # Method 1: Use cv2.inpaint with better parameters
    inpainted = cv2.inpaint(room_img, mask, 5, cv2.INPAINT_TELEA)
    
    # Method 2: Apply additional smoothing only to the inpainted area
    # Create a mask for the inpainted area
    inpainted_area = mask > 0
    
    # Apply gentle smoothing to reduce artifacts
    inpainted = cv2.bilateralFilter(inpainted, 3, 30, 30)
    
    # Method 3: Blend with original image at edges to reduce artifacts
    # Create a soft edge mask
    edge_mask = cv2.GaussianBlur(mask, (21, 21), 0)
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
    clean_room_path = os.path.join(OUTPUT_DIR, "room_without_sofa.png")
    cv2.imwrite(clean_room_path, blended)
    
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

def calculate_sofa_positioning(sofa_bbox, cutout_path):
    """Calculate positioning for the new sofa"""
    print("📐 Calculating sofa positioning...")
    
    # Get original sofa area
    orig_x, orig_y, orig_w, orig_h = sofa_bbox
    
    # Load new sofa to get its dimensions
    cutout_img = Image.open(cutout_path).convert('RGBA')
    new_w, new_h = cutout_img.size
    
    print(f"   Original sofa area: {orig_w}x{orig_h}")
    print(f"   New sofa size: {new_w}x{new_h}")
    
    # Calculate scale to fit the sofa area
    scale_w = orig_w / new_w
    scale_h = orig_h / new_h
    
    # Use the smaller scale to ensure it fits
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

def create_realistic_shadow(cutout_path, room_size, positioning):
    """Create a realistic shadow"""
    print("🌫️ Creating realistic shadow...")
    
    # Load the cutout
    cutout_img = Image.open(cutout_path).convert('RGBA')
    
    # Get positioning info
    new_x, new_y, new_width, new_height, scale = positioning
    
    # Create a realistic shadow
    shadow_width = int(new_width * 1.2)
    shadow_height = int(new_height * 0.2)  # Short shadow
    
    # Create shadow canvas
    shadow_canvas = Image.new('RGBA', (shadow_width, shadow_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow_canvas)
    
    # Draw a more realistic shadow shape
    draw.ellipse([0, 0, shadow_width, shadow_height], 
                 fill=(0, 0, 0, 40))  # Subtle shadow
    
    # Apply Gaussian blur
    shadow_canvas = shadow_canvas.filter(ImageFilter.GaussianBlur(radius=8))
    
    # Create full room shadow
    shadow_img = Image.new('RGBA', room_size, (0, 0, 0, 0))
    
    # Position shadow at the bottom of the new sofa
    shadow_x = new_x + (new_width - shadow_width) // 2
    shadow_y = new_y + new_height - shadow_height + 3  # Slightly below the sofa
    
    # Paste shadow
    shadow_img.paste(shadow_canvas, (shadow_x, shadow_y), shadow_canvas)
    
    # Save shadow
    shadow_path = os.path.join(OUTPUT_DIR, "realistic_shadow.png")
    shadow_img.save(shadow_path)
    
    print(f"✅ Realistic shadow saved: {shadow_path}")
    return shadow_path

def replace_sofa_properly(clean_room_path, cutout_path, shadow_path, positioning):
    """Replace the sofa properly in the exact location"""
    print("🏠 Replacing sofa properly...")
    
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
    
    # Add shadow first
    result_img.paste(shadow_img, (0, 0), shadow_img)
    
    # Add the new sofa in the exact position where the old one was
    result_img.paste(cutout_resized, (new_x, new_y), cutout_resized)
    
    # Save result
    result_path = os.path.join(OUTPUT_DIR, "manual_sofa_replacement.png")
    result_img.save(result_path)
    
    print(f"✅ Manual sofa replacement saved: {result_path}")
    print(f"   New sofa positioned at: x={new_x}, y={new_y}")
    print(f"   New sofa size: {new_width}x{new_height}")
    print(f"   Scale factor: {scale:.2f}")
    
    return result_path

def main():
    """Main manual sofa replacement"""
    print("🛋️  MANUAL SOFA REPLACEMENT")
    print("=" * 50)
    
    ensure_output_dir()
    
    try:
        # Step 1: Download furniture image
        furniture_path = download_furniture_image()
        
        # Step 2: Create manual sofa mask
        mask_path, mask, sofa_bbox = create_manual_sofa_mask(ROOM_IMAGE)
        
        # Step 3: Remove sofa with context-aware fill
        clean_room_path = remove_sofa_with_context_fill(ROOM_IMAGE, mask_path)
        
        # Step 4: Create sofa cutout
        cutout_path = create_sofa_cutout(furniture_path)
        
        # Step 5: Calculate positioning
        positioning = calculate_sofa_positioning(sofa_bbox, cutout_path)
        
        # Step 6: Create realistic shadow
        room_img = Image.open(ROOM_IMAGE)
        shadow_path = create_realistic_shadow(cutout_path, room_img.size, positioning)
        
        # Step 7: Replace sofa properly
        result_path = replace_sofa_properly(
            clean_room_path, 
            cutout_path, 
            shadow_path,
            positioning
        )
        
        print("\n🎉 MANUAL SOFA REPLACEMENT COMPLETE!")
        print("=" * 50)
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"🎯 Manual sofa mask: {mask_path}")
        print(f"🏠 Clean room: {clean_room_path}")
        print(f"✂️  Sofa cutout: {cutout_path}")
        print(f"🌫️  Realistic shadow: {shadow_path}")
        print(f"🖼️  Final result: {result_path}")
        
        # Open the result
        try:
            result_img = Image.open(result_path)
            result_img.show()
            print("🖼️  Result image opened in default viewer")
        except Exception as e:
            print(f"⚠️  Could not open result image: {e}")
            
    except Exception as e:
        print(f"❌ Manual sofa replacement failed: {e}")
        raise

if __name__ == "__main__":
    main()
