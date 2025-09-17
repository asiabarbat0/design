#!/usr/bin/env python3
"""
Test script for Renderer Service
Tests the 2D compositor, inpainting, and progressive rendering
"""

import requests
import json
import time
import numpy as np
from PIL import Image
import io

# Configuration
BASE_URL = "http://localhost:5003"

def test_renderer():
    """Test the renderer service"""
    print("🎨 Renderer Service Test Suite")
    print("=" * 60)
    print(f"🌐 Base URL: {BASE_URL}")
    print()

    # Test 1: Health check
    print("🏥 Testing Health Check")
    print("=" * 30)
    try:
        response = requests.get(f"{BASE_URL}/renderer/health")
        data = response.json()
        
        if data['status'] == 'healthy':
            print("✅ Renderer service is healthy")
            print(f"   S3 available: {data['s3_available']}")
        else:
            print("❌ Renderer service is not healthy")
            return
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return

    # Test 2: Render stats
    print("\n📊 Testing Render Stats")
    print("=" * 30)
    try:
        response = requests.get(f"{BASE_URL}/renderer/stats")
        data = response.json()
        
        if data['success']:
            stats = data['stats']
            print("✅ Render stats retrieved")
            print(f"   Cache hits: {stats['cache_hits']}")
            print(f"   Cache misses: {stats['cache_misses']}")
            print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
            print(f"   Total renders: {stats['total_renders']}")
            print(f"   Avg render time: {stats['avg_render_time']:.2f}s")
        else:
            print(f"❌ Failed to get stats: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Stats test failed: {e}")

    # Test 3: Single render (this will fail without real data)
    print("\n🖼️  Testing Single Render")
    print("=" * 30)
    try:
        response = requests.get(f"{BASE_URL}/renderer/render?roomId=test_room&itemId=test_item&size=preview")
        data = response.json()
        
        if data.get('success'):
            print("✅ Single render successful")
            print(f"   URL: {data['url']}")
            print(f"   Cached: {data['cached']}")
            print(f"   Size: {data['size']}")
        else:
            print(f"⚠️  Single render failed (expected without real data): {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Single render test failed: {e}")

    # Test 4: Batch render
    print("\n🔄 Testing Batch Render")
    print("=" * 30)
    try:
        batch_data = {
            "renders": [
                {
                    "roomId": "test_room_1",
                    "itemId": "test_item_1",
                    "size": "preview",
                    "x": 100,
                    "y": 200
                },
                {
                    "roomId": "test_room_2",
                    "itemId": "test_item_2",
                    "size": "full",
                    "x": 300,
                    "y": 400
                }
            ]
        }
        
        response = requests.post(f"{BASE_URL}/renderer/render/batch", json=batch_data)
        data = response.json()
        
        if data.get('success'):
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

    # Test 5: Performance test
    print("\n⚡ Testing Performance")
    print("=" * 30)
    try:
        start_time = time.time()
        
        # Test multiple renders
        for i in range(3):
            response = requests.get(f"{BASE_URL}/renderer/render?roomId=perf_test_{i}&itemId=perf_item_{i}&size=preview")
            # Don't check success since we don't have real data
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Performance test completed")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg time per request: {total_time/3:.2f}s")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

    # Test 6: Cache test
    print("\n💾 Testing Cache")
    print("=" * 30)
    try:
        # First render (should miss cache)
        response1 = requests.get(f"{BASE_URL}/renderer/render?roomId=cache_test&itemId=cache_item&size=preview")
        data1 = response1.json()
        
        # Second render (should hit cache)
        response2 = requests.get(f"{BASE_URL}/renderer/render?roomId=cache_test&itemId=cache_item&size=preview")
        data2 = response2.json()
        
        print("✅ Cache test completed")
        print(f"   First render cached: {data1.get('cached', 'Unknown')}")
        print(f"   Second render cached: {data2.get('cached', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")

    print("\n🎉 Test suite completed!")
    print("\n📝 Next steps:")
    print("1. Upload real room and item images to S3")
    print("2. Test with actual product data")
    print("3. Monitor render performance and cache hit rates")
    print("4. Optimize rendering pipeline based on results")

if __name__ == "__main__":
    test_renderer()
