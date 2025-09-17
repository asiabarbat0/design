#!/usr/bin/env python3
"""
Simple Image Viewer
View the downloaded images and test furniture items
"""

import os
from PIL import Image
import webbrowser
import tempfile

def view_image(image_path, title="Image"):
    """Display an image using the default system viewer"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        # Open image to get info
        img = Image.open(image_path)
        print(f"📸 {title}")
        print(f"   File: {image_path}")
        print(f"   Size: {img.size[0]}x{img.size[1]}")
        print(f"   Format: {img.format}")
        print(f"   Mode: {img.mode}")
        print(f"   File size: {os.path.getsize(image_path)} bytes")
        
        # Open with default system viewer
        webbrowser.open(f"file://{os.path.abspath(image_path)}")
        print(f"   ✅ Opened in default viewer")
        return True
        
    except Exception as e:
        print(f"❌ Error viewing {image_path}: {e}")
        return False

def main():
    """View all the test images"""
    print("🖼️  Image Viewer")
    print("=" * 50)
    
    # List of images to view
    images = [
        ("test_room.jpg", "K-Kircher Home Living Room"),
        ("test_item_cutout.png", "Test Chair (Cutout)"),
        ("test_item_shadow.png", "Test Chair (Shadow)"),
        ("demo_chair_cutout.png", "Demo Chair (Cutout)"),
        ("demo_chair_shadow.png", "Demo Chair (Shadow)"),
        ("demo_table_cutout.png", "Demo Table (Cutout)"),
        ("demo_table_shadow.png", "Demo Table (Shadow)")
    ]
    
    print("Opening images in your default viewer...")
    print()
    
    for image_file, title in images:
        view_image(image_file, title)
        print()
    
    print("🎉 All images opened!")
    print("\n💡 Tips:")
    print("- The living room image is from K-Kircher Home")
    print("- The cutout images have transparent backgrounds")
    print("- The shadow images are semi-transparent")
    print("- You can use these for testing the renderer service")

if __name__ == "__main__":
    main()
