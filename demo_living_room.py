#!/usr/bin/env python3
"""
Living Room Renderer Demo
Demonstrates how the renderer would work with the K-Kircher Home living room
"""

import requests
import json
import time
from PIL import Image, ImageDraw
import io

# Configuration
BASE_URL = "http://localhost:5003"

def create_demo_furniture():
    """Create demo furniture items for the living room"""
    print("🪑 Creating Demo Furniture Items")
    print("=" * 40)
    
    # Create a modern chair
    chair = Image.new('RGBA', (150, 200), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(chair)
    
    # Chair design
    draw.rectangle([30, 0, 120, 80], fill=(139, 69, 19, 255))  # Back
    draw.rectangle([20, 80, 130, 120], fill=(160, 82, 45, 255))  # Seat
    draw.rectangle([25, 120, 40, 200], fill=(101, 67, 33, 255))  # Leg 1
    draw.rectangle([110, 120, 125, 200], fill=(101, 67, 33, 255))  # Leg 2
    
    chair.save("demo_chair_cutout.png", "PNG")
    print("✅ Modern chair created")
    
    # Create chair shadow
    chair_shadow = Image.new('RGBA', (150, 200), color=(0, 0, 0, 0))
    draw_shadow = ImageDraw.Draw(chair_shadow)
    draw_shadow.ellipse([10, 180, 140, 200], fill=(0, 0, 0, 80))
    chair_shadow.save("demo_chair_shadow.png", "PNG")
    print("✅ Chair shadow created")
    
    # Create a side table
    table = Image.new('RGBA', (100, 120), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(table)
    
    # Table design
    draw.rectangle([10, 0, 90, 20], fill=(139, 69, 19, 255))  # Top
    draw.rectangle([15, 20, 25, 120], fill=(101, 67, 33, 255))  # Leg 1
    draw.rectangle([75, 20, 85, 120], fill=(101, 67, 33, 255))  # Leg 2
    
    table.save("demo_table_cutout.png", "PNG")
    print("✅ Side table created")
    
    # Create table shadow
    table_shadow = Image.new('RGBA', (100, 120), color=(0, 0, 0, 0))
    draw_shadow = ImageDraw.Draw(table_shadow)
    draw_shadow.ellipse([5, 100, 95, 120], fill=(0, 0, 0, 60))
    table_shadow.save("demo_table_shadow.png", "PNG")
    print("✅ Table shadow created")
    
    return True

def demo_renderer_capabilities():
    """Demonstrate renderer capabilities"""
    print("🎨 Living Room Renderer Demo")
    print("=" * 60)
    print("Using K-Kircher Home living room image")
    print("https://k-kircher-home.myshopify.com/cdn/shop/files/0787477_sylvie-express-bench-sofa.jpg")
    print()
    
    # Step 1: Health check
    print("🏥 Checking Renderer Health")
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

    # Step 2: Create demo furniture
    create_demo_furniture()

    # Step 3: Demonstrate different scenarios
    print("\n🎯 Rendering Scenarios")
    print("=" * 30)
    
    scenarios = [
        {
            "name": "Add Chair to Left Side",
            "roomId": "test_room",
            "itemId": "demo_chair",
            "x": 100,
            "y": 300,
            "description": "Place a modern chair to the left of the sofa"
        },
        {
            "name": "Add Side Table",
            "roomId": "test_room", 
            "itemId": "demo_table",
            "x": 600,
            "y": 250,
            "description": "Add a side table next to the sofa"
        },
        {
            "name": "Chair Near Window",
            "roomId": "test_room",
            "itemId": "demo_chair", 
            "x": 400,
            "y": 150,
            "description": "Place chair near the window for reading"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   Position: ({scenario['x']}, {scenario['y']})")
        
        # Test preview render
        try:
            start_time = time.time()
            response = requests.get(f"{BASE_URL}/renderer/render?roomId={scenario['roomId']}&itemId={scenario['itemId']}&size=preview&x={scenario['x']}&y={scenario['y']}")
            render_time = time.time() - start_time
            
            data = response.json()
            
            if data.get('success'):
                print(f"   ✅ Preview render successful ({render_time:.2f}s)")
                print(f"   URL: {data['url']}")
            else:
                print(f"   ⚠️  Preview render failed: {data.get('error', 'Unknown error')}")
                print("   (Expected without S3 setup)")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Step 4: Show renderer features
    print("\n✨ Renderer Features")
    print("=" * 30)
    print("🎨 2D Compositing: Advanced image blending with alpha channels")
    print("🖼️  Inpainting: Removes original items using OpenCV")
    print("📐 Perspective Correction: Automatically detects room perspective")
    print("💡 Lighting Matching: Recolors items to match room lighting")
    print("🌫️  Shadow Generation: Creates realistic shadows")
    print("⚡ Progressive Rendering: 960px preview + 1920px full")
    print("💾 S3 Caching: Intelligent caching with cache hit monitoring")
    print("📊 Performance Monitoring: Real-time render time tracking")

    # Step 5: Show API endpoints
    print("\n🔌 API Endpoints")
    print("=" * 30)
    print("GET  /renderer/health          - Health check")
    print("GET  /renderer/render          - Single item render")
    print("POST /renderer/render/batch    - Batch rendering")
    print("GET  /renderer/stats           - Performance statistics")

    # Step 6: Show usage examples
    print("\n💻 Usage Examples")
    print("=" * 30)
    print("Single render:")
    print("  GET /renderer/render?roomId=room_123&itemId=chair_456&size=preview&x=100&y=200")
    print()
    print("Batch render:")
    print("  POST /renderer/render/batch")
    print("  {")
    print('    "renders": [')
    print('      {"roomId": "room_123", "itemId": "chair_456", "size": "preview", "x": 100, "y": 200},')
    print('      {"roomId": "room_123", "itemId": "table_789", "size": "full", "x": 300, "y": 400}')
    print('    ]')
    print("  }")

    print("\n🎉 Demo completed!")
    print("\n📝 Next steps:")
    print("1. Set up S3/MinIO for image storage")
    print("2. Upload the living room image and furniture items to S3")
    print("3. Test with real rendering pipeline")
    print("4. Integrate with the main application UI")

    return True

if __name__ == "__main__":
    demo_renderer_capabilities()
