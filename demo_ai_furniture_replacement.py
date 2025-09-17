#!/usr/bin/env python3
"""
Demo AI Furniture Replacement
============================

Demonstrate the AI furniture replacement service with various prompts
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import requests

# Configuration
OUTPUT_DIR = "ai_furniture_demo"
ROOM_IMAGE = "new_room.jpg"  # Your actual new room image

def ensure_output_dir():
    """Ensure output directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_furniture_smart(room_image_path, furniture_type="couch"):
    """Smart furniture detection"""
    print(f"🎯 Detecting {furniture_type} smartly...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    h, w = room_img.shape[:2]
    print(f"   Image size: {w}x{h}")
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(room_img, cv2.COLOR_BGR2HSV)
    
    # Look for furniture in the center-bottom area
    center_bottom_y = int(h * 0.3)
    center_bottom_h = int(h * 0.7)
    
    furniture_region = room_img[center_bottom_y:center_bottom_h, :]
    furniture_hsv = hsv[center_bottom_y:center_bottom_h, :]
    
    # Detect furniture based on type
    if furniture_type == "couch" or furniture_type == "sofa":
        # Look for light colored furniture
        lower_light = np.array([0, 0, 120])
        upper_light = np.array([180, 50, 220])
        mask_light = cv2.inRange(furniture_hsv, lower_light, upper_light)
        
        # Also look for white throw pillows
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(furniture_hsv, lower_white, upper_white)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_light, mask_white)
    
    else:
        # Generic furniture detection
        lower_generic = np.array([0, 0, 100])
        upper_generic = np.array([180, 50, 200])
        combined_mask = cv2.inRange(furniture_hsv, lower_generic, upper_generic)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 2000:
            # Get bounding box
            x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
            
            # Adjust coordinates back to full image
            x_full = x
            y_full = y + center_bottom_y
            
            # Create mask for full image
            furniture_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(furniture_mask, [largest_contour + [0, center_bottom_y]], 255)
            
            # Smooth the mask
            furniture_mask = cv2.GaussianBlur(furniture_mask, (7, 7), 0)
            
            # Save mask
            mask_path = os.path.join(OUTPUT_DIR, f"{furniture_type}_mask.png")
            cv2.imwrite(mask_path, furniture_mask)
            
            print(f"✅ {furniture_type} detected: x={x_full}, y={y_full}, w={w_rect}, h={h_rect}")
            return mask_path, furniture_mask, (x_full, y_full, w_rect, h_rect)
    
    # Fallback: Create manual mask
    print(f"⚠️ Using manual {furniture_type} area definition")
    
    furniture_x = int(w * 0.15)
    furniture_y = int(h * 0.4)
    furniture_w = int(w * 0.7)
    furniture_h = int(h * 0.35)
    
    # Create rectangular mask
    furniture_mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(furniture_mask, (furniture_x, furniture_y), 
                  (furniture_x + furniture_w, furniture_y + furniture_h), 255, -1)
    
    # Smooth the mask
    furniture_mask = cv2.GaussianBlur(furniture_mask, (15, 15), 0)
    
    # Save mask
    mask_path = os.path.join(OUTPUT_DIR, f"manual_{furniture_type}_mask.png")
    cv2.imwrite(mask_path, furniture_mask)
    
    print(f"✅ Manual {furniture_type} area: x={furniture_x}, y={furniture_y}, w={furniture_w}, h={furniture_h}")
    return mask_path, furniture_mask, (furniture_x, furniture_y, furniture_w, furniture_h)

def generate_ai_furniture(room_image_path, mask_path, furniture_prompt, output_size=(1920, 1920)):
    """Generate new furniture using AI inpainting simulation"""
    print(f"🤖 Generating {furniture_prompt} using AI inpainting...")
    
    # Load the room image
    room_img = cv2.imread(room_image_path)
    if room_img is None:
        raise ValueError(f"Could not load room image: {room_image_path}")
    
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    # Use cv2.inpaint as MVP fallback
    print("   Using cv2.inpaint as MVP fallback...")
    inpainted = cv2.inpaint(room_img, mask, 5, cv2.INPAINT_TELEA)
    
    # Apply color adjustment to simulate the new furniture
    mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
    
    # Adjust the inpainted area to match the furniture prompt
    if "white" in furniture_prompt.lower():
        # Make the inpainted area white-ish
        white_color = np.array([255, 255, 255])
        inpainted = inpainted * (1 - mask_3d) + white_color * mask_3d
    elif "black" in furniture_prompt.lower():
        # Make the inpainted area black-ish
        black_color = np.array([0, 0, 0])
        inpainted = inpainted * (1 - mask_3d) + black_color * mask_3d
    elif "brown" in furniture_prompt.lower():
        # Make the inpainted area brown-ish
        brown_color = np.array([139, 69, 19])
        inpainted = inpainted * (1 - mask_3d) + brown_color * mask_3d
    elif "blue" in furniture_prompt.lower():
        # Make the inpainted area blue-ish
        blue_color = np.array([0, 0, 255])
        inpainted = inpainted * (1 - mask_3d) + blue_color * mask_3d
    elif "red" in furniture_prompt.lower():
        # Make the inpainted area red-ish
        red_color = np.array([255, 0, 0])
        inpainted = inpainted * (1 - mask_3d) + red_color * mask_3d
    elif "green" in furniture_prompt.lower():
        # Make the inpainted area green-ish
        green_color = np.array([0, 255, 0])
        inpainted = inpainted * (1 - mask_3d) + green_color * mask_3d
    
    # Apply gentle smoothing (ensure correct data type)
    inpainted = inpainted.astype(np.uint8)
    inpainted = cv2.bilateralFilter(inpainted, 5, 50, 50)
    
    # Resize to target size
    if output_size != (room_img.shape[1], room_img.shape[0]):
        inpainted = cv2.resize(inpainted, output_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Save the result
    result_path = os.path.join(OUTPUT_DIR, f"ai_generated_{furniture_prompt.replace(' ', '_')}.png")
    cv2.imwrite(result_path, inpainted)
    
    print(f"✅ AI generated furniture saved: {result_path}")
    return result_path

def create_preview_image(full_image_path, preview_size=(960, 960)):
    """Create a quick preview image"""
    print(f"📱 Creating {preview_size[0]}px preview...")
    
    # Load the full image
    full_img = cv2.imread(full_image_path)
    if full_img is None:
        raise ValueError(f"Could not load full image: {full_image_path}")
    
    # Resize to preview size
    preview_img = cv2.resize(full_img, preview_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Save preview
    preview_path = os.path.join(OUTPUT_DIR, f"preview_{preview_size[0]}px_{os.path.basename(full_image_path)}")
    cv2.imwrite(preview_path, preview_img)
    
    print(f"✅ Preview saved: {preview_path}")
    return preview_path

def demo_furniture_replacements():
    """Demo various furniture replacements"""
    print("🤖 AI FURNITURE REPLACEMENT DEMO")
    print("=" * 50)
    
    ensure_output_dir()
    
    # Test cases
    test_cases = [
        ("white couch", "couch"),
        ("black leather sofa", "sofa"),
        ("brown wooden chair", "chair"),
        ("blue modern sofa", "sofa"),
        ("red velvet couch", "couch"),
        ("green accent chair", "chair")
    ]
    
    results = []
    
    for furniture_prompt, furniture_type in test_cases:
        print(f"\n🛋️ Testing: {furniture_prompt}")
        print("-" * 40)
        
        try:
            # Step 1: Detect furniture
            mask_path, mask, furniture_bbox = detect_furniture_smart(ROOM_IMAGE, furniture_type)
            
            # Step 2: Generate AI furniture
            full_result_path = generate_ai_furniture(ROOM_IMAGE, mask_path, furniture_prompt, (1920, 1920))
            
            # Step 3: Create preview
            preview_path = create_preview_image(full_result_path, (960, 960))
            
            results.append({
                'furniture_prompt': furniture_prompt,
                'furniture_type': furniture_type,
                'full_path': full_result_path,
                'preview_path': preview_path,
                'mask_path': mask_path
            })
            
            print(f"✅ {furniture_prompt} completed successfully!")
            
        except Exception as e:
            print(f"❌ {furniture_prompt} failed: {e}")
    
    return results

def main():
    """Main demo function"""
    print("🤖 AI FURNITURE REPLACEMENT DEMO")
    print("=" * 50)
    
    try:
        results = demo_furniture_replacements()
        
        print("\n🎉 DEMO COMPLETE!")
        print("=" * 30)
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"🛋️ Generated {len(results)} furniture replacements")
        
        for result in results:
            print(f"   • {result['furniture_prompt']} ({result['furniture_type']})")
            print(f"     Full: {result['full_path']}")
            print(f"     Preview: {result['preview_path']}")
        
        # Open the first result
        if results:
            try:
                first_result = results[0]
                result_img = Image.open(first_result['full_path'])
                result_img.show()
                print(f"\n🖼️  First result opened in default viewer: {first_result['furniture_prompt']}")
            except Exception as e:
                print(f"⚠️  Could not open result image: {e}")
                
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
