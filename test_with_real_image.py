#!/usr/bin/env python3
"""
Test Renderer Service with Real Living Room Image
Uses the K-Kircher Home living room image for testing
"""

import requests
import json
import time
import numpy as np
from PIL import Image
import io
import os

# Configuration
BASE_URL = "http://localhost:5003"
ROOM_IMAGE_URL = "https://k-kircher-home.myshopify.com/cdn/shop/files/0787477_sylvie-express-bench-sofa.jpg?v=1752147901&width=1440"

def download_room_image():
    """Download the living room image"""
    print("📥 Downloading living room image...")
    try:
        response = requests.get(ROOM_IMAGE_URL, timeout=30)
        response.raise_for_status()
        
        # Save the image
        with open("test_room.jpg", "wb") as f:
            f.write(response.content)
        
        print(f"✅ Room image downloaded: {len(response.content)} bytes")
        
        # Load and display image info
        image = Image.open(io.BytesIO(response.content))
        print(f"   Dimensions: {image.size[0]}x{image.size[1]}")
        print(f"   Format: {image.format}")
        print(f"   Mode: {image.mode}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to download room image: {e}")
        return False

def create_test_item():
    """Create a test furniture item to place in the room"""
    print("\n🪑 Creating test furniture item...")
    
    # Create a simple chair-like item
    item = Image.new('RGBA', (200, 300), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(item)
    
    # Draw a modern chair
    # Chair back
    draw.rectangle([50, 0, 150, 100], fill=(139, 69, 19, 255))  # Brown back
    
    # Chair seat
    draw.rectangle([30, 100, 170, 150], fill=(160, 82, 45, 255))  # Darker brown seat
    
    # Chair legs
    draw.rectangle([40, 150, 60, 300], fill=(101, 67, 33, 255))  # Dark brown leg
    draw.rectangle([140, 150, 160, 300], fill=(101, 67, 33, 255))  # Dark brown leg
    
    # Save item
    item.save("test_item_cutout.png", "PNG")
    print("✅ Test item created: test_item_cutout.png")
    
    # Create shadow
    shadow = Image.new('RGBA', (200, 300), color=(0, 0, 0, 0))
    draw_shadow = ImageDraw.Draw(shadow)
    
    # Create soft shadow
    shadow_color = (0, 0, 0, 100)  # Semi-transparent black
    draw_shadow.ellipse([20, 250, 180, 280], fill=shadow_color)
    draw_shadow.ellipse([30, 240, 170, 270], fill=shadow_color)
    
    shadow.save("test_item_shadow.png", "PNG")
    print("✅ Test shadow created: test_item_shadow.png")
    
    return True

def upload_to_s3_mock():
    """Mock S3 upload - in real implementation, this would upload to S3"""
    print("\n📤 Mock S3 Upload (in real implementation, upload to S3)")
    print("   Room image: uploads/test_room.jpg")
    print("   Item cutout: renders/test_item_cutout.png")
    print("   Item shadow: renders/test_item_shadow.png")
    return True

def test_renderer_with_real_image():
    """Test the renderer service with the real living room image"""
    print("🎨 Testing Renderer with Real Living Room Image")
    print("=" * 60)
    print(f"🌐 Base URL: {BASE_URL}")
    print(f"🖼️  Room Image: {ROOM_IMAGE_URL}")
    print()

    # Step 1: Download room image
    if not download_room_image():
        return False

    # Step 2: Create test item
    if not create_test_item():
        return False

    # Step 3: Mock S3 upload
    upload_to_s3_mock()

    # Step 4: Test renderer health
    print("\n🏥 Testing Renderer Health")
    print("=" * 30)
    try:
        response = requests.get(f"{BASE_URL}/renderer/health")
        data = response.json()
        
        if data['status'] == 'healthy':
            print("✅ Renderer service is healthy")
            print(f"   S3 available: {data['s3_available']}")
        else:
            print("❌ Renderer service is not healthy")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

    # Step 5: Test render with real image (will fail without S3)
    print("\n🎯 Testing Render with Real Image")
    print("=" * 30)
    
    # Test preview render
    print("Testing preview render...")
    try:
        response = requests.get(f"{BASE_URL}/renderer/render?roomId=test_room&itemId=test_item&size=preview&x=200&y=300")
        data = response.json()
        
        if data.get('success'):
            print("✅ Preview render successful")
            print(f"   URL: {data['url']}")
            print(f"   Cached: {data['cached']}")
        else:
            print(f"⚠️  Preview render failed: {data.get('error', 'Unknown error')}")
            print("   This is expected without S3 setup")
    except Exception as e:
        print(f"❌ Preview render test failed: {e}")

    # Test full render
    print("\nTesting full render...")
    try:
        response = requests.get(f"{BASE_URL}/renderer/render?roomId=test_room&itemId=test_item&size=full&x=200&y=300")
        data = response.json()
        
        if data.get('success'):
            print("✅ Full render successful")
            print(f"   URL: {data['url']}")
            print(f"   Cached: {data['cached']}")
        else:
            print(f"⚠️  Full render failed: {data.get('error', 'Unknown error')}")
            print("   This is expected without S3 setup")
    except Exception as e:
        print(f"❌ Full render test failed: {e}")

    # Step 6: Test different positions
    print("\n📍 Testing Different Positions")
    print("=" * 30)
    
    positions = [
        (100, 200, "Left side"),
        (400, 250, "Center"),
        (700, 300, "Right side"),
        (300, 400, "Front")
    ]
    
    for x, y, description in positions:
        print(f"Testing position: {description} ({x}, {y})")
        try:
            response = requests.get(f"{BASE_URL}/renderer/render?roomId=test_room&itemId=test_item&size=preview&x={x}&y={y}")
            data = response.json()
            
            if data.get('success'):
                print(f"   ✅ {description}: Success")
            else:
                print(f"   ⚠️  {description}: {data.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   ❌ {description}: {e}")

    # Step 7: Performance test
    print("\n⚡ Performance Test")
    print("=" * 30)
    try:
        start_time = time.time()
        
        # Test multiple renders
        for i in range(3):
            response = requests.get(f"{BASE_URL}/renderer/render?roomId=test_room&itemId=test_item&size=preview&x={100+i*50}&y={200+i*50}")
            # Don't check success since we don't have S3
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Performance test completed")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg time per request: {total_time/3:.2f}s")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

    # Step 8: Get final stats
    print("\n📊 Final Statistics")
    print("=" * 30)
    try:
        response = requests.get(f"{BASE_URL}/renderer/stats")
        data = response.json()
        
        if data['success']:
            stats = data['stats']
            print("✅ Final stats retrieved")
            print(f"   Cache hits: {stats['cache_hits']}")
            print(f"   Cache misses: {stats['cache_misses']}")
            print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
            print(f"   Total renders: {stats['total_renders']}")
            print(f"   Avg render time: {stats['avg_render_time']:.2f}s")
        else:
            print(f"❌ Failed to get stats: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Stats test failed: {e}")

    print("\n🎉 Test completed!")
    print("\n📝 Next steps:")
    print("1. Set up S3/MinIO for image storage")
    print("2. Upload the test images to S3")
    print("3. Test with real rendering pipeline")
    print("4. Optimize for production use")

    return True

if __name__ == "__main__":
    # Import PIL ImageDraw here to avoid issues
    from PIL import ImageDraw
    
    test_renderer_with_real_image()
