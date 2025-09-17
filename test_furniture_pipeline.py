#!/usr/bin/env python3
"""
Test Furniture Replacement Pipeline
===================================

Test script for the furniture replacement pipeline
"""

import requests
import json
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:5004"
TEST_ROOM_IMAGE = "static/445.png"
TEST_FURNITURE_IMAGE = "https://k-kircher-home.myshopify.com/cdn/shop/files/0787477_sylvie-express-bench-sofa.jpg?v=1752147901&width=1440"

def upload_test_images():
    """Upload test images to S3 (or use local files for testing)"""
    print("📤 Uploading test images...")
    
    # For testing, we'll use local files
    # In production, these would be uploaded to S3
    print("✅ Test images ready (using local files)")

def test_render_api():
    """Test the render API endpoint"""
    print("🧪 Testing render API...")
    
    # Test data
    test_data = {
        "roomId": "test_room",
        "itemId": "test_item", 
        "target_width": 400,
        "anchorX": 500,
        "baselineY": 400
    }
    
    try:
        # Make API request
        response = requests.post(f"{API_BASE_URL}/render", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Render API test successful!")
            print(f"   Room clean URL: {result.get('room_clean_url', 'N/A')}")
            print(f"   Render URL: {result.get('render_url', 'N/A')}")
            print(f"   Processing time: {result.get('processing_time', 'N/A')}s")
            print(f"   Cache key: {result.get('cache_key', 'N/A')}")
            return True
        else:
            print(f"❌ Render API test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Render API test error: {e}")
        return False

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check successful!")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   S3 Available: {result.get('s3_available', 'N/A')}")
            print(f"   Cache Size: {result.get('cache_size', 'N/A')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_cache_functionality():
    """Test cache functionality"""
    print("💾 Testing cache functionality...")
    
    # Test data
    test_data = {
        "roomId": "test_room",
        "itemId": "test_item",
        "target_width": 400,
        "anchorX": 500,
        "baselineY": 400
    }
    
    try:
        # First request (should process)
        print("   First request (should process)...")
        response1 = requests.post(f"{API_BASE_URL}/render", json=test_data)
        time1 = response1.json().get('processing_time', 0)
        
        # Second request (should be cached)
        print("   Second request (should be cached)...")
        response2 = requests.post(f"{API_BASE_URL}/render", json=test_data)
        time2 = response2.json().get('processing_time', 0)
        
        if time2 < time1:
            print("✅ Cache functionality working!")
            print(f"   First request: {time1:.2f}s")
            print(f"   Second request: {time2:.2f}s")
            return True
        else:
            print("⚠️ Cache might not be working properly")
            return False
            
    except Exception as e:
        print(f"❌ Cache test error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 FURNITURE REPLACEMENT PIPELINE TEST")
    print("=" * 50)
    
    # Upload test images
    upload_test_images()
    
    # Test health check
    health_ok = test_health_check()
    
    if not health_ok:
        print("❌ Health check failed, stopping tests")
        return
    
    # Test render API
    render_ok = test_render_api()
    
    if not render_ok:
        print("❌ Render API test failed, stopping tests")
        return
    
    # Test cache functionality
    cache_ok = test_cache_functionality()
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Render API: {'✅ PASS' if render_ok else '❌ FAIL'}")
    print(f"Cache: {'✅ PASS' if cache_ok else '❌ FAIL'}")
    
    if all([health_ok, render_ok, cache_ok]):
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️ Some tests failed")

if __name__ == "__main__":
    main()
