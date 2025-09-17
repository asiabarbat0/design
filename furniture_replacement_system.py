#!/usr/bin/env python3
"""
Furniture Replacement System
============================

This system:
1. Detects furniture in a room
2. Removes it completely
3. Finds products that fit the space
4. Replaces with properly sized furniture
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
OUTPUT_DIR = "furniture_replacement_system"

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
        session = rembg.new_session('u2net')
        output_data = rembg.remove(input_data, session=session)
        
        # Save the cutout
        cutout_path = os.path.join(OUTPUT_DIR, "furniture_cutout.png")
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
    
    # Define a rectangle around the furniture
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
    cutout_path = os.path.join(OUTPUT_DIR, "furniture_cutout.png")
    Image.fromarray(cutout, 'RGBA').save(cutout_path)
    
    print(f"✅ Fallback cutout saved: {cutout_path}")
    return cutout_path

def detect_furniture_in_room(room_path):
    """Detect all furniture in the room"""
    print(f"🔍 Detecting furniture in room...")
    
    # Load the room image
    room_img = cv2.imread(room_path)
    if room_img is None:
        print(f"❌ Could not load room image: {room_path}")
        return None
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(room_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(room_img, cv2.COLOR_BGR2GRAY)
    
    # Detect different types of furniture
    furniture_items = []
    
    # 1. Detect sofa (light colored, large)
    lower_light = np.array([0, 0, 200])
    upper_light = np.array([180, 30, 255])
    mask_sofa = cv2.inRange(hsv, lower_light, upper_light)
    
    # Clean up sofa mask
    kernel = np.ones((5,5), np.uint8)
    mask_sofa = cv2.morphologyEx(mask_sofa, cv2.MORPH_CLOSE, kernel)
    mask_sofa = cv2.morphologyEx(mask_sofa, cv2.MORPH_OPEN, kernel)
    
    # Find sofa contours
    sofa_contours, _ = cv2.findContours(mask_sofa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if sofa_contours:
        largest_sofa = max(sofa_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_sofa)
        
        # Create precise mask
        sofa_mask = np.zeros(room_img.shape[:2], np.uint8)
        cv2.fillPoly(sofa_mask, [largest_sofa], 255)
        
        furniture_items.append({
            'type': 'sofa',
            'bbox': (x, y, w, h),
            'mask': sofa_mask,
            'contour': largest_sofa,
            'area': cv2.contourArea(largest_sofa)
        })
        
        print(f"✅ Sofa detected: ({x}, {y}) size: {w}x{h} area: {cv2.contourArea(largest_sofa):.0f}")
    
    # 2. Detect table (darker wood)
    lower_wood = np.array([10, 50, 50])
    upper_wood = np.array([25, 255, 200])
    mask_table = cv2.inRange(hsv, lower_wood, upper_wood)
    
    # Clean up table mask
    mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_CLOSE, kernel)
    mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_OPEN, kernel)
    
    # Find table contours
    table_contours, _ = cv2.findContours(mask_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if table_contours:
        largest_table = max(table_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_table)
        
        furniture_items.append({
            'type': 'table',
            'bbox': (x, y, w, h),
            'contour': largest_table,
            'area': cv2.contourArea(largest_table)
        })
        
        print(f"✅ Table detected: ({x}, {y}) size: {w}x{h} area: {cv2.contourArea(largest_table):.0f}")
    
    return furniture_items

def remove_furniture_completely(room_path, furniture_item):
    """Remove furniture completely from the room"""
    print(f"🏠 Removing {furniture_item['type']} completely from room...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load the room image
    room_img = cv2.imread(room_path)
    room_rgb = cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB)
    
    # Get the furniture mask
    furniture_mask = furniture_item['mask']
    x, y, w, h = furniture_item['bbox']
    
    # Method 1: Use the surrounding floor/wall pattern
    # Get a larger context area around the furniture
    margin = 50
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(room_rgb.shape[1], x + w + margin)
    y2 = min(room_rgb.shape[0], y + h + margin)
    
    # Extract context area
    context_area = room_rgb[y1:y2, x1:x2]
    context_mask = furniture_mask[y1:y2, x1:x2]
    
    # Use the floor area (below the furniture) to fill
    # The floor is typically below the furniture
    floor_y = h // 2
    floor_area = context_area[floor_y:, :]
    
    if floor_area.shape[0] > 0:
        # Resize the floor area to match the furniture area
        floor_resized = cv2.resize(floor_area, (w, h))
        
        # Create the filled area
        filled_area = context_area.copy()
        
        # Fill the furniture area with the floor pattern
        for c in range(3):
            channel = context_area[:, :, c]
            floor_channel = floor_resized[:, :, c]
            
            # Ensure dimensions match
            if channel.shape == floor_channel.shape:
                # Blend the floor pattern into the furniture area
                filled_channel = channel * (1 - context_mask/255.0) + floor_channel * (context_mask/255.0)
                filled_area[:, :, c] = filled_channel
            else:
                # If dimensions don't match, just use the original channel
                filled_area[:, :, c] = channel
        
        # Put the filled area back into the room
        room_rgb[y1:y2, x1:x2] = filled_area
    
    # Save the result
    result_path = os.path.join(OUTPUT_DIR, f"room_without_{furniture_item['type']}.png")
    Image.fromarray(room_rgb).save(result_path)
    
    print(f"✅ {furniture_item['type'].title()} removed: {result_path}")
    return result_path

def find_products_that_fit(furniture_item, available_products):
    """Find products that fit in the furniture space"""
    print(f"🔍 Finding products that fit in {furniture_item['type']} space...")
    
    x, y, w, h = furniture_item['bbox']
    target_area = w * h
    
    # For now, we'll use the K-Kircher sofa as our available product
    # In a real system, this would search a product database
    available_products = [
        {
            'name': 'K-Kircher Sylvie Express Bench Sofa',
            'url': KKIRCHER_SOFA_URL,
            'type': 'sofa',
            'original_size': (1440, 1440),  # Original image size
            'aspect_ratio': 1.0
        }
    ]
    
    # Find products that match the furniture type
    matching_products = [p for p in available_products if p['type'] == furniture_item['type']]
    
    if not matching_products:
        print(f"❌ No products found for {furniture_item['type']}")
        return None
    
    # Calculate the best fit
    best_product = None
    best_fit_score = 0
    
    for product in matching_products:
        # Calculate how well the product fits
        product_w, product_h = product['original_size']
        product_aspect = product_w / product_h
        target_aspect = w / h
        
        # Calculate fit score based on aspect ratio and size
        aspect_score = 1.0 - abs(product_aspect - target_aspect) / max(product_aspect, target_aspect)
        size_score = min(1.0, target_area / (product_w * product_h))
        
        fit_score = (aspect_score + size_score) / 2
        
        if fit_score > best_fit_score:
            best_fit_score = fit_score
            best_product = product
    
    print(f"✅ Best fit product: {best_product['name']} (fit score: {best_fit_score:.2f})")
    return best_product

def calibrate_product_to_fit(product, furniture_item):
    """Calibrate the product to fit the furniture space"""
    print(f"📏 Calibrating {product['name']} to fit space...")
    
    # Get the furniture dimensions
    x, y, w, h = furniture_item['bbox']
    
    # Calculate the scale to fit the space
    # Make it slightly smaller to ensure it fits
    target_width = int(w * 0.9)
    target_height = int(h * 0.9)
    
    # Calculate scale factors
    scale_x = target_width / product['original_size'][0]
    scale_y = target_height / product['original_size'][1]
    
    # Use the smaller scale to maintain aspect ratio
    scale = min(scale_x, scale_y)
    
    # Calculate new size
    new_width = int(product['original_size'][0] * scale)
    new_height = int(product['original_size'][1] * scale)
    
    print(f"✅ Product calibrated: {new_width}x{new_height} (scale: {scale:.2f})")
    
    return {
        'scale': scale,
        'new_size': (new_width, new_height),
        'position': (x + (w - new_width) // 2, y + (h - new_height) // 2)
    }

def create_product_cutout(product_url):
    """Create a cutout of the product"""
    print(f"🎨 Creating cutout of {product_url}...")
    
    # Download the product image
    response = requests.get(product_url)
    response.raise_for_status()
    
    # Save temporarily
    temp_path = os.path.join(OUTPUT_DIR, "temp_product.jpg")
    with open(temp_path, 'wb') as f:
        f.write(response.content)
    
    # Create cutout
    cutout_path = create_high_quality_cutout(temp_path)
    
    # Clean up temp file
    os.remove(temp_path)
    
    return cutout_path

def composite_product_into_room(room_no_furniture_path, product_cutout_path, calibration_info, furniture_item):
    """Composite the product into the room"""
    print(f"🏠 Compositing product into room...")
    
    # Load images
    room_img = Image.open(room_no_furniture_path).convert('RGBA')
    product_img = Image.open(product_cutout_path).convert('RGBA')
    
    # Resize the product to fit
    new_width, new_height = calibration_info['new_size']
    product_resized = product_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Get position
    x, y = calibration_info['position']
    
    # Create a copy of the room for compositing
    result_img = room_img.copy()
    
    # Composite the product
    result_img.paste(product_resized, (x, y), product_resized)
    
    # Save the result
    result_path = os.path.join(OUTPUT_DIR, "furniture_replacement_result.png")
    result_img.save(result_path)
    
    print(f"✅ Product composited: {result_path}")
    return result_path

def main():
    print("🛋️  FURNITURE REPLACEMENT SYSTEM")
    print("=" * 50)
    
    # Step 1: Detect furniture in the room
    furniture_items = detect_furniture_in_room(ROOM_IMAGE)
    if not furniture_items:
        print("❌ No furniture detected, cannot continue")
        return
    
    # Step 2: Find the sofa (target sofas specifically)
    sofa_items = [item for item in furniture_items if item['type'] == 'sofa']
    if not sofa_items:
        print("❌ No sofa detected, cannot continue")
        return
    
    largest_furniture = max(sofa_items, key=lambda x: x['area'])
    print(f"🎯 Targeting sofa: {largest_furniture['type']} (area: {largest_furniture['area']:.0f})")
    
    # Step 3: Find products that fit
    best_product = find_products_that_fit(largest_furniture, [])
    if not best_product:
        print("❌ No suitable products found, cannot continue")
        return
    
    # Step 4: Remove the original furniture
    room_no_furniture_path = remove_furniture_completely(ROOM_IMAGE, largest_furniture)
    if not room_no_furniture_path:
        print("❌ Furniture removal failed, cannot continue")
        return
    
    # Step 5: Create product cutout
    product_cutout_path = create_product_cutout(best_product['url'])
    if not product_cutout_path:
        print("❌ Product cutout creation failed, cannot continue")
        return
    
    # Step 6: Calibrate product to fit
    calibration_info = calibrate_product_to_fit(best_product, largest_furniture)
    
    # Step 7: Composite the product
    result_path = composite_product_into_room(room_no_furniture_path, product_cutout_path, calibration_info, largest_furniture)
    
    if result_path:
        print("\n🎉 FURNITURE REPLACEMENT COMPLETE!")
        print("=" * 50)
        print(f"🏠 Original room: {ROOM_IMAGE}")
        print(f"🛋️  Furniture removed: {largest_furniture['type']}")
        print(f"🛋️  Room without furniture: {room_no_furniture_path}")
        print(f"🛋️  Product: {best_product['name']}")
        print(f"🛋️  Product cutout: {product_cutout_path}")
        print(f"🖼️  Final result: {result_path}")
        
        # Open the result image
        try:
            result_img = Image.open(result_path)
            result_img.show()
            print("🖼️  Result image opened in default viewer")
        except Exception as e:
            print(f"⚠️  Could not open result image: {e}")
    else:
        print("❌ Furniture replacement failed")

if __name__ == "__main__":
    main()
