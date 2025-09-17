#!/usr/bin/env python3
"""
Sofa Replacement Demo
=====================

This script demonstrates how to:
1. Download the K-Kircher Home sofa image
2. Use auto matting to create a cutout
3. Render the cutout into the 445.png room, replacing the existing sofa
"""

import requests
import os
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5003"
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

def auto_matting(image_path):
    """Use the auto matting service to create cutout and shadow"""
    print(f"🎨 Processing {image_path} with auto matting...")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/auto-matting/process", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Auto matting completed!")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
        print(f"   Cutout URL: {result.get('cutout_url', 'N/A')}")
        print(f"   Shadow URL: {result.get('shadow_url', 'N/A')}")
        return result
    else:
        print(f"❌ Auto matting failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def render_sofa_replacement(room_image, cutout_url, shadow_url, x=400, y=200):
    """Render the sofa replacement in the room"""
    print(f"🏠 Rendering sofa replacement in room...")
    
    # First, let's upload the room image to S3 for the renderer
    room_upload_url = f"{BASE_URL}/renderer/upload-room"
    with open(room_image, 'rb') as f:
        files = {'room_image': f}
        data = {'room_id': 'sofa_replacement_room'}
        response = requests.post(room_upload_url, files=files, data=data)
    
    if response.status_code != 200:
        print(f"❌ Failed to upload room image: {response.text}")
        return None
    
    # Now render the sofa replacement
    render_url = f"{BASE_URL}/renderer/render"
    params = {
        'roomId': 'sofa_replacement_room',
        'itemId': 'kkircher_sofa',
        'size': 'full',
        'x': x,
        'y': y,
        'cutoutUrl': cutout_url,
        'shadowUrl': shadow_url
    }
    
    response = requests.get(render_url, params=params)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Rendering completed!")
        print(f"   Preview URL: {result.get('preview_url', 'N/A')}")
        print(f"   Full URL: {result.get('full_url', 'N/A')}")
        return result
    else:
        print(f"❌ Rendering failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def main():
    print("🛋️  SOFA REPLACEMENT DEMO")
    print("=" * 50)
    
    # Step 1: Download the K-Kircher sofa
    sofa_path = download_image(KKIRCHER_SOFA_URL, "kkircher_sofa_original.jpg")
    
    # Step 2: Auto matting to create cutout
    matting_result = auto_matting(sofa_path)
    if not matting_result:
        print("❌ Auto matting failed, cannot continue")
        return
    
    # Step 3: Render the sofa replacement
    render_result = render_sofa_replacement(
        ROOM_IMAGE, 
        matting_result.get('cutout_url'),
        matting_result.get('shadow_url'),
        x=400,  # Position where the original sofa is
        y=200
    )
    
    if render_result:
        print("\n🎉 SOFA REPLACEMENT COMPLETE!")
        print("=" * 50)
        print(f"📸 Original sofa: {sofa_path}")
        print(f"🏠 Room background: {ROOM_IMAGE}")
        print(f"🖼️  Final result: {render_result.get('full_url', 'Check S3')}")
        print(f"👀 Preview: {render_result.get('preview_url', 'Check S3')}")
        
        # Save the results to a file
        results = {
            'original_sofa': sofa_path,
            'room_image': ROOM_IMAGE,
            'matting_result': matting_result,
            'render_result': render_result
        }
        
        results_file = os.path.join(OUTPUT_DIR, "sofa_replacement_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved to: {results_file}")
    else:
        print("❌ Sofa replacement failed")

if __name__ == "__main__":
    main()
