#!/usr/bin/env python3
"""
Test script for Matting Studio Admin functionality
"""

import requests
import json
import time
import base64
import numpy as np
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5003"
TEST_IMAGE_URL = "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=400&h=300&fit=crop"

def test_matting_studio_admin():
    """Test the Matting Studio Admin service"""
    print("🎨 Matting Studio Admin Test Suite")
    print("=" * 60)
    print(f"🌐 Base URL: {BASE_URL}")
    print()

    # Test 1: Check if server is running
    print("🏥 Testing Server Health")
    print("=" * 30)
    try:
        response = requests.get(f"{BASE_URL}/matting-studio-admin/", timeout=5)
        if response.status_code == 200:
            print("✅ Matting Studio Admin is accessible")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Server is not running: {e}")
        return

    # Test 2: Test review queue
    print("\n📋 Testing Review Queue")
    print("=" * 30)
    try:
        response = requests.get(f"{BASE_URL}/matting-studio-admin/queue")
        data = response.json()
        
        if data['success']:
            print(f"✅ Review queue loaded: {data['total']} items")
            if data['queue']:
                print(f"   First item: ID {data['queue'][0]['id']}, Confidence: {data['queue'][0]['confidence_score']:.2f}")
            else:
                print("   No items in queue")
        else:
            print(f"❌ Failed to load queue: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error testing queue: {e}")

    # Test 3: Test brush preview
    print("\n🖌️  Testing Brush Preview")
    print("=" * 30)
    try:
        response = requests.post(f"{BASE_URL}/matting-studio-admin/api/brush/preview", 
                               json={"size": 30, "hardness": 0.7})
        data = response.json()
        
        if data['success']:
            print(f"✅ Brush preview generated: {data['size']}px, {data['hardness']:.1f} hardness")
        else:
            print(f"❌ Failed to generate brush preview: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error testing brush preview: {e}")

    # Test 4: Test matting data retrieval (if queue has items)
    print("\n📸 Testing Matting Data Retrieval")
    print("=" * 30)
    try:
        # First get the queue to find an item
        queue_response = requests.get(f"{BASE_URL}/matting-studio-admin/queue")
        queue_data = queue_response.json()
        
        if queue_data['success'] and queue_data['queue']:
            matting_id = queue_data['queue'][0]['id']
            print(f"   Testing with matting ID: {matting_id}")
            
            response = requests.get(f"{BASE_URL}/matting-studio-admin/api/matting/{matting_id}")
            data = response.json()
            
            if data['success']:
                print(f"✅ Matting data retrieved successfully")
                print(f"   Original shape: {data['original_shape']}")
                print(f"   Confidence: {data['matting']['confidence_score']:.2f}")
                print(f"   Has mask data: {'Yes' if data['matting']['mask_data'] else 'No'}")
            else:
                print(f"❌ Failed to retrieve matting data: {data.get('error', 'Unknown error')}")
        else:
            print("   No items in queue to test with")
    except Exception as e:
        print(f"❌ Error testing matting data retrieval: {e}")

    # Test 5: Test matting data save (mock data)
    print("\n💾 Testing Matting Data Save")
    print("=" * 30)
    try:
        # Create a simple test mask
        test_mask = np.random.rand(100, 100).astype(np.float32)
        mask_base64 = base64.b64encode(test_mask.tobytes()).decode('utf-8')
        
        # Try to save to a test matting ID (this might fail if ID doesn't exist)
        test_data = {
            "mask_data": mask_base64,
            "mask_shape": [100, 100],
            "brush_settings": {
                "size": 20,
                "hardness": 0.8,
                "edge_feather": 2
            },
            "keep_shadow": True,
            "edited_by": "test_admin"
        }
        
        # Use a test ID (this will likely fail, but we can test the endpoint)
        response = requests.put(f"{BASE_URL}/matting-studio-admin/api/matting/999999", 
                              json=test_data)
        data = response.json()
        
        if data['success']:
            print("✅ Matting data saved successfully")
        else:
            print(f"⚠️  Save failed (expected for test ID): {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error testing matting data save: {e}")

    print("\n🎉 Test suite completed!")
    print("\n📝 Next steps:")
    print("1. Visit http://localhost:5003/matting-studio-admin/ for the admin interface")
    print("2. Use the review queue to select images for editing")
    print("3. Test the brush tools and keyboard shortcuts")
    print("4. Save edited matting data")

if __name__ == "__main__":
    test_matting_studio_admin()
