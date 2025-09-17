#!/usr/bin/env python3
"""
Test AI Furniture Replacement Service
====================================

Test script for the AI furniture replacement service
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:5006"

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

def test_furniture_replacement(furniture_prompt, furniture_type="couch"):
    """Test furniture replacement with different prompts"""
    print(f"🛋️ Testing furniture replacement: '{furniture_prompt}'")
    
    # Test data
    test_data = {
        "roomId": "test_room",
        "furniturePrompt": furniture_prompt,
        "furnitureType": furniture_type
    }
    
    try:
        # Make API request
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/replace-furniture", json=test_data)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Furniture replacement successful!")
            print(f"   Furniture Prompt: {result.get('furniture_prompt', 'N/A')}")
            print(f"   Furniture Type: {result.get('furniture_type', 'N/A')}")
            print(f"   Preview URL: {result.get('preview_url', 'N/A')}")
            print(f"   Full URL: {result.get('full_url', 'N/A')}")
            print(f"   Processing Time: {result.get('processing_time', 'N/A')}s")
            print(f"   Total Time: {processing_time:.2f}s")
            return True
        else:
            print(f"❌ Furniture replacement failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Furniture replacement error: {e}")
        return False

def test_cache_functionality():
    """Test cache functionality"""
    print("💾 Testing cache functionality...")
    
    # Test data
    test_data = {
        "roomId": "test_room",
        "furniturePrompt": "white couch",
        "furnitureType": "couch"
    }
    
    try:
        # First request (should process)
        print("   First request (should process)...")
        start_time = time.time()
        response1 = requests.post(f"{API_BASE_URL}/replace-furniture", json=test_data)
        time1 = time.time() - start_time
        
        if response1.status_code != 200:
            print(f"❌ First request failed: {response1.status_code}")
            return False
        
        # Second request (should be cached)
        print("   Second request (should be cached)...")
        start_time = time.time()
        response2 = requests.post(f"{API_BASE_URL}/replace-furniture", json=test_data)
        time2 = time.time() - start_time
        
        if response2.status_code != 200:
            print(f"❌ Second request failed: {response2.status_code}")
            return False
        
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

def test_multiple_furniture_types():
    """Test different furniture types"""
    print("🪑 Testing multiple furniture types...")
    
    test_cases = [
        ("white couch", "couch"),
        ("black leather sofa", "sofa"),
        ("brown wooden chair", "chair"),
        ("glass coffee table", "table"),
        ("modern white desk", "desk")
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for furniture_prompt, furniture_type in test_cases:
        print(f"\n   Testing: {furniture_prompt} ({furniture_type})")
        if test_furniture_replacement(furniture_prompt, furniture_type):
            success_count += 1
    
    print(f"\n📊 Multiple furniture types test: {success_count}/{total_count} successful")
    return success_count == total_count

def main():
    """Main test function"""
    print("🤖 AI FURNITURE REPLACEMENT SERVICE TEST")
    print("=" * 60)
    
    # Test health check
    health_ok = test_health_check()
    
    if not health_ok:
        print("❌ Health check failed, stopping tests")
        return
    
    # Test basic furniture replacement
    basic_ok = test_furniture_replacement("white couch", "couch")
    
    if not basic_ok:
        print("❌ Basic furniture replacement failed, stopping tests")
        return
    
    # Test cache functionality
    cache_ok = test_cache_functionality()
    
    # Test multiple furniture types
    multiple_ok = test_multiple_furniture_types()
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Basic Replacement: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"Cache: {'✅ PASS' if cache_ok else '❌ FAIL'}")
    print(f"Multiple Types: {'✅ PASS' if multiple_ok else '❌ FAIL'}")
    
    if all([health_ok, basic_ok, cache_ok, multiple_ok]):
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️ Some tests failed")

if __name__ == "__main__":
    main()
