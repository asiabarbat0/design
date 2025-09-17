#!/usr/bin/env python3
"""
Furniture Replacement API Server
================================

Simple API server for furniture replacement that can be integrated
into the main Flask application
"""

from flask import Flask, request, jsonify
import os
import json
import hashlib
import time
from furniture_replacement_pipeline import (
    segment_furniture_to_mask,
    inpaint_room_clean,
    create_furniture_cutout,
    create_soft_shadow,
    composite_furniture,
    generate_cache_key
)

# Initialize Flask app
app = Flask(__name__)

# Cache for storing results
render_cache = {}

@app.route('/render', methods=['POST'])
def render_furniture():
    """API endpoint for furniture rendering"""
    try:
        data = request.get_json()
        
        # Validate required parameters
        required_params = ['roomId', 'itemId', 'target_width', 'anchorX', 'baselineY']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'Missing required parameter: {param}'}), 400
        
        room_id = data['roomId']
        item_id = data['itemId']
        placement_params = {
            'target_width': data['target_width'],
            'anchorX': data['anchorX'],
            'baselineY': data['baselineY']
        }
        
        # Check cache first
        cache_key = generate_cache_key(room_id, item_id, placement_params)
        if cache_key in render_cache:
            print(f"✅ Cache hit for {cache_key}")
            return jsonify(render_cache[cache_key])
        
        # For this demo, we'll use local files
        # In production, these would be downloaded from S3
        room_path = f"static/{room_id}.jpg"
        furniture_path = f"static/{item_id}.jpg"
        
        if not os.path.exists(room_path):
            return jsonify({'error': f'Room image not found: {room_path}'}), 404
        
        if not os.path.exists(furniture_path):
            return jsonify({'error': f'Furniture image not found: {furniture_path}'}), 404
        
        # Process the furniture replacement
        start_time = time.time()
        
        # Step 1: Segment furniture to binary mask
        mask_path, mask = segment_furniture_to_mask(room_path, 'sofa')
        
        # Step 2: Inpaint room to produce clean version
        clean_room_path = inpaint_room_clean(room_path, mask_path)
        
        # Step 3: Create furniture cutout
        cutout_path = create_furniture_cutout(furniture_path)
        
        # Step 4: Create soft shadow
        shadow_path = create_soft_shadow(cutout_path)
        
        # Step 5: Composite furniture
        result_path = composite_furniture(
            clean_room_path, 
            cutout_path, 
            shadow_path,
            placement_params['target_width'],
            placement_params['anchorX'],
            placement_params['baselineY']
        )
        
        processing_time = time.time() - start_time
        
        # Create result
        result = {
            'room_clean_url': f'/static/renders/{room_id}_{item_id}_clean.png',
            'render_url': f'/static/renders/{room_id}_{item_id}_render.png',
            'cache_key': cache_key,
            'processing_time': processing_time
        }
        
        # Cache the result
        render_cache[cache_key] = result
        
        print(f"✅ Render completed in {processing_time:.2f}s")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Render failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'cache_size': len(render_cache)
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=True)
