#!/usr/bin/env python3
"""
Test script for the Auto Matting Service
Demonstrates the automatic background removal functionality.
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5003"
TEST_IMAGE_URL = "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=800&h=600&fit=crop"  # Furniture image

def test_auto_matting():
    """Test the auto matting service with a sample image."""
    print("🧪 Testing Auto Matting Service")
    print("=" * 50)
    
    # Test data
    test_data = {
        "image_url": TEST_IMAGE_URL,
        "model": "auto",  # Try YOLO first, fallback to rembg
        "generate_shadow": True,
        "store_result": True
    }
    
    print(f"📸 Processing image: {TEST_IMAGE_URL}")
    print(f"🔧 Model: {test_data['model']}")
    print(f"🌫️  Generate shadow: {test_data['generate_shadow']}")
    print()
    
    try:
        # Make request to auto matting service
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/auto-matting/process",
            json=test_data,
            timeout=60
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Auto Matting Successful!")
            print(f"⏱️  Processing time: {result.get('processing_time', processing_time):.2f}s")
            print(f"🎯 Confidence: {result.get('confidence', 0):.3f}")
            print(f"🤖 Model used: {result.get('model_used', 'unknown')}")
            print(f"🔍 Needs manual review: {result.get('needs_manual', False)}")
            print()
            
            if result.get('cutout_url'):
                print(f"✂️  Cutout URL: {result['cutout_url']}")
            
            if result.get('shadow_url'):
                print(f"🌫️  Shadow URL: {result['shadow_url']}")
            
            if result.get('result_id'):
                print(f"🆔 Result ID: {result['result_id']}")
            
            print()
            
            # Test confidence levels
            confidence = result.get('confidence', 0)
            if confidence >= 0.8:
                print("🟢 High confidence - Ready for production")
            elif confidence >= 0.6:
                print("🟡 Medium confidence - May need review")
            else:
                print("🔴 Low confidence - Requires manual cleanup")
            
        else:
            print(f"❌ Auto Matting Failed!")
            print(f"Status Code: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Response: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_batch_processing():
    """Test batch processing with multiple images."""
    print("\n🔄 Testing Batch Processing")
    print("=" * 50)
    
    # Test with multiple images
    batch_data = {
        "images": [
            {
                "image_url": "https://images.unsplash.com/photo-1586023492125-27b2c04ef369?w=400&h=300&fit=crop",
                "model": "auto"
            },
            {
                "image_url": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=400&h=300&fit=crop",
                "model": "rembg"
            }
        ],
        "generate_shadow": True
    }
    
    print(f"📸 Processing {len(batch_data['images'])} images...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/auto-matting/batch-process",
            json=batch_data,
            timeout=120
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Batch Processing Successful!")
            print(f"⏱️  Total processing time: {result.get('processing_time', processing_time):.2f}s")
            print(f"📊 Successful: {result.get('successful', 0)}/{result.get('total_images', 0)}")
            print()
            
            for i, img_result in enumerate(result.get('results', [])):
                print(f"Image {i+1}:")
                if img_result.get('success'):
                    print(f"  ✅ Confidence: {img_result.get('confidence', 0):.3f}")
                    print(f"  🔍 Needs manual: {img_result.get('needs_manual', False)}")
                else:
                    print(f"  ❌ Error: {img_result.get('error', 'Unknown error')}")
                print()
        
        else:
            print(f"❌ Batch Processing Failed!")
            print(f"Status Code: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Response: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_health_check():
    """Test the health check endpoint."""
    print("\n🏥 Testing Health Check")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/auto-matting/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            
            print("✅ Auto Matting Service is healthy!")
            print(f"🤖 YOLO available: {health_data.get('yolo_available', False)}")
            print(f"🗄️  S3 available: {health_data.get('s3_available', False)}")
            print(f"✂️  rembg available: {health_data.get('rembg_available', False)}")
            print(f"🎯 Confidence threshold: {health_data.get('confidence_threshold', 0.7)}")
            
            if health_data.get('yolo_error'):
                print(f"⚠️  YOLO error: {health_data['yolo_error']}")
        
        else:
            print(f"❌ Health check failed! Status: {response.status_code}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_matting_studio():
    """Test the Matting Studio interface."""
    print("\n🎨 Testing Matting Studio")
    print("=" * 50)
    
    try:
        # Test queue endpoint
        response = requests.get(f"{BASE_URL}/matting-studio/queue", timeout=10)
        
        if response.status_code == 200:
            queue_data = response.json()
            
            print("✅ Matting Studio is accessible!")
            print(f"📋 Queue items: {len(queue_data.get('images', []))}")
            
            if queue_data.get('images'):
                print("\nImages needing review:")
                for img in queue_data['images'][:3]:  # Show first 3
                    print(f"  - ID: {img['id']}, Confidence: {img['confidence']:.3f}")
            else:
                print("  No images need review")
        
        else:
            print(f"❌ Matting Studio queue failed! Status: {response.status_code}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Matting Studio request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def main():
    """Run all tests."""
    print("🚀 Auto Matting Service Test Suite")
    print("=" * 60)
    print(f"🌐 Base URL: {BASE_URL}")
    print(f"📸 Test Image: {TEST_IMAGE_URL}")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/auto-matting/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding properly")
            return
    except:
        print("❌ Server is not running. Please start the application first:")
        print("   make dev")
        print("   or")
        print("   python run.py")
        return
    
    # Run tests
    test_health_check()
    test_auto_matting()
    test_batch_processing()
    test_matting_studio()
    
    print("\n🎉 Test suite completed!")
    print("\n📝 Next steps:")
    print("1. Check the generated cutout and shadow images")
    print("2. Visit http://localhost:5002/matting-studio/ for manual review")
    print("3. Test with your own furniture images")


if __name__ == "__main__":
    main()
