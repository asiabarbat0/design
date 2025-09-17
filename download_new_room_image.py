#!/usr/bin/env python3
"""
Download New Room Image
=======================

Download the new living room image that was provided by the user.
"""

import requests
from PIL import Image
import os

def download_new_room_image():
    """Download the new living room image"""
    print("📥 Downloading new living room image...")
    
    # The user provided a new living room image, but I need to save it locally
    # Since I can't directly access the image from the description, I'll create a placeholder
    # and ask the user to provide the actual image file
    
    print("⚠️  I need the actual image file to proceed.")
    print("Please save the new living room image as 'new_room.jpg' in the current directory.")
    print("Then I can use it for the sofa replacement test.")
    
    # Check if the image exists
    if os.path.exists("new_room.jpg"):
        print("✅ New room image found: new_room.jpg")
        return "new_room.jpg"
    else:
        print("❌ New room image not found. Please save it as 'new_room.jpg'")
        return None

if __name__ == "__main__":
    download_new_room_image()
