#!/usr/bin/env python3
"""
Demo script for Renderer Service
Generates sample data and tests the full rendering pipeline
"""

import requests
import json
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# Configuration
BASE_URL = "http://localhost:5003"

def create_sample_room(width=1920, height=1080):
    """Create a sample room image"""
    # Create a simple room background
    room = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(room)
    
    # Draw floor
    floor_color = (200, 180, 160)
    draw.rectangle([0, height//2, width, height], fill=floor_color)
    
    # Draw walls
    wall_color = (220, 220, 220)
    draw.rectangle([0, 0, width, height//2], fill=wall_color)
    
    # Draw some furniture outlines
    draw.rectangle([100, height//2 + 50, 300, height//2 + 200], outline=(100, 100, 100), width=3)
    draw.rectangle([400, height//2 + 100, 600, height//2 + 250], outline=(100, 100, 100), width=3)
    
    # Add some text
    try:
        font = ImageFont.load_default()
        draw.text((50, 50), "Sample Room", fill=(0, 0, 0), font=font)
    except:
        pass
    
    return np.array(room)

def create_sample_item(width=200, height=300):
    """Create a sample furniture item"""
    # Create a simple chair-like item
    item = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(item)
    
    # Draw chair back
    draw.rectangle([50, 0, 150, 100], fill=(139, 69, 19, 255))  # Brown chair back
    
    # Draw chair seat
    draw.rectangle([30, 100, 170, 150], fill=(160, 82, 45, 255))  # Darker brown seat
    
    # Draw chair legs
    draw.rectangle([40, 150, 60, 300], fill=(101, 67, 33, 255))  # Dark brown leg
    draw.rectangle([140, 150, 160, 300], fill=(101, 67, 33, 255))  # Dark brown leg
    
    return np.array(item)

def create_sample_shadow(width=200, height=300):
    """Create a sample shadow for the item"""
    shadow = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    
    # Create a soft shadow
    shadow_color = (0, 0, 0, 100)  # Semi-transparent black
    
    # Draw shadow shapes
    draw.ellipse([20, 250, 180, 280], fill=shadow_color)
    draw.ellipse([30, 240, 170, 270], fill=shadow_color)
    
    return np.array(shadow)

def upload_image_to_s3(image_array, bucket, key):
    """Upload image to S3 (mock function)"""
    # In a real implementation, this would upload to S3
    # For demo purposes, we'll just return a mock URL
    return f"https://s3.amazonaws.com/{bucket}/{key}"

def test_full_pipeline():
    """Test the complete rendering pipeline"""
    print("🎨 Renderer Demo - Full Pipeline Test")
    print("=" * 60)
    print(f"🌐 Base URL: {BASE_URL}")
    print()

    # Test 1: Health check
    print("🏥 Checking Renderer Health")
    print("=" * 30)
    try:
        response = requests.get(f"{BASE_URL}/renderer/health")
        data = response.json()
        
        if data['status'] == 'healthy':
            print("✅ Renderer service is healthy")
        else:
            print("❌ Renderer service is not healthy")
            return
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return

    # Test 2: Generate sample data
    print("\n🖼️  Generating Sample Data")
    print("=" * 30)
    
    # Create sample room
    room_image = create_sample_room(1920, 1080)
    print("✅ Sample room created (1920x1080)")
    
    # Create sample item
    item_image = create_sample_item(200, 300)
    print("✅ Sample item created (200x300)")
    
    # Create sample shadow
    shadow_image = create_sample_shadow(200, 300)
    print("✅ Sample shadow created (200x300)")
    
    # In a real implementation, these would be uploaded to S3
    print("📤 Sample data ready for upload to S3")

    # Test 3: Test render endpoints
    print("\n🎯 Testing Render Endpoints")
    print("=" * 30)
    
    # Test preview render
    print("Testing preview render...")
    try:
        response = requests.get(f"{BASE_URL}/renderer/render?roomId=demo_room&itemId=demo_item&size=preview&x=100&y=200")
        data = response.json()
        
        if data.get('success'):
            print("✅ Preview render successful")
            print(f"   URL: {data['url']}")
            print(f"   Cached: {data['cached']}")
        else:
            print(f"⚠️  Preview render failed: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Preview render test failed: {e}")
    
    # Test full render
    print("\nTesting full render...")
    try:
        response = requests.get(f"{BASE_URL}/renderer/render?roomId=demo_room&itemId=demo_item&size=full&x=100&y=200")
        data = response.json()
        
        if data.get('success'):
            print("✅ Full render successful")
            print(f"   URL: {data['url']}")
            print(f"   Cached: {data['cached']}")
        else:
            print(f"⚠️  Full render failed: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Full render test failed: {e}")

    # Test 4: Batch render test
    print("\n🔄 Testing Batch Render")
    print("=" * 30)
    try:
        batch_data = {
            "renders": [
                {
                    "roomId": "demo_room_1",
                    "itemId": "demo_item_1",
                    "size": "preview",
                    "x": 150,
                    "y": 250
                },
                {
                    "roomId": "demo_room_2",
                    "itemId": "demo_item_2",
                    "size": "full",
                    "x": 300,
                    "y": 400
                }
            ]
        }
        
        response = requests.post(f"{BASE_URL}/renderer/render/batch", json=batch_data)
        data = response.json()
        
        if data['success']:
            print("✅ Batch render successful")
            print(f"   Total renders: {data['total']}")
            for i, result in enumerate(data['results']):
                if result.get('success'):
                    print(f"   Render {i+1}: {result['roomId']}/{result['itemId']} - {result['size']}")
                else:
                    print(f"   Render {i+1}: Error - {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Batch render failed: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Batch render test failed: {e}")

    # Test 5: Performance monitoring
    print("\n⚡ Performance Monitoring")
    print("=" * 30)
    try:
        response = requests.get(f"{BASE_URL}/renderer/stats")
        data = response.json()
        
        if data['success']:
            stats = data['stats']
            print("✅ Performance stats retrieved")
            print(f"   Cache hits: {stats['cache_hits']}")
            print(f"   Cache misses: {stats['cache_misses']}")
            print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
            print(f"   Total renders: {stats['total_renders']}")
            print(f"   Avg render time: {stats['avg_render_time']:.2f}s")
            
            if stats['recent_renders']:
                print(f"   Recent render times: {[f'{t:.2f}s' for t in stats['recent_renders']]}")
        else:
            print(f"❌ Failed to get stats: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")

    print("\n🎉 Demo completed!")
    print("\n📝 Next steps:")
    print("1. Upload real room and item images to S3")
    print("2. Test with actual product catalogs")
    print("3. Monitor render performance and optimize")
    print("4. Integrate with the main application")

if __name__ == "__main__":
    test_full_pipeline()
