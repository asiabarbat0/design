# Furniture Replacement Pipeline

A complete furniture replacement system that segments old furniture, inpaints the room, and composites new furniture with proper scaling and shadows.

## Pipeline Overview

The system implements a 3-step pipeline:

1. **Segmentation**: Auto-detect furniture and create binary mask.png
2. **Inpainting**: Remove furniture using cv2.inpaint() to produce room_clean.png  
3. **Compositing**: Add new furniture with uniform scale, baseline alignment, and soft shadow

## API Endpoints

### POST /render

Renders furniture replacement with caching.

**Request Body:**
```json
{
  "roomId": "room_123",
  "itemId": "sofa_456", 
  "target_width": 400,
  "anchorX": 500,
  "baselineY": 400
}
```

**Response:**
```json
{
  "room_clean_url": "s3://bucket/renders/room_123_sofa_456_clean.png",
  "render_url": "s3://bucket/renders/room_123_sofa_456_render.png",
  "cache_key": "abc123...",
  "processing_time": 2.34
}
```

**Parameters:**
- `roomId`: Unique identifier for the room image
- `itemId`: Unique identifier for the furniture item
- `target_width`: Desired width of the furniture in pixels
- `anchorX`: Horizontal center position of the furniture
- `baselineY`: Bottom edge position of the furniture

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "s3_available": true,
  "cache_size": 42
}
```

## Technical Implementation

### 1. Segmentation (Auto Model + Optional Brush Fixes)

```python
def segment_furniture_to_mask(room_image_path, furniture_type='sofa'):
    # Color-based detection (HSV)
    # Edge-based refinement (Canny)
    # Contour analysis (largest contour)
    # Binary mask creation
    # Optional: Manual brush fixes (future enhancement)
```

**Output:** `mask.png` - Binary mask of the furniture to remove

### 2. Inpainting (MVP: cv2.inpaint, Future: LaMa/SDXL)

```python
def inpaint_room_clean(room_image_path, mask_path):
    # Load room image and mask
    # Apply cv2.inpaint with TELEA method
    # TODO: Upgrade to LaMa/SDXL for higher quality
    # Save clean room
```

**Output:** `room_clean.png` - Room with furniture removed

### 3. Compositing (Pillow with LANCZOS + Gaussian Shadow)

```python
def composite_furniture(clean_room_path, cutout_path, shadow_path, 
                       target_width, anchor_x, baseline_y):
    # Calculate uniform scale
    # Resize with Image.LANCZOS
    # Position with baseline alignment
    # Add Gaussian-blurred ellipse shadow
    # Composite with alpha blending
```

**Output:** `render.png` - Final furniture replacement

## File Structure

```
furniture_pipeline_output/
├── mask.png              # Binary mask of old furniture
├── room_clean.png        # Room with furniture removed
├── cutout.png           # New furniture with transparent background
├── shadow.png           # Soft shadow for new furniture
└── render.png           # Final composite result
```

## Caching Strategy

- **Cache Key**: MD5 hash of `(roomId, itemId, params)`
- **Storage**: In-memory dictionary (production: Redis)
- **TTL**: No expiration (production: configurable)
- **Benefits**: Instant retries, reduced processing costs

## S3 Integration

### Bucket Structure
```
designstream-uploads/
├── rooms/
│   └── {roomId}.jpg
└── furniture/
    └── {itemId}.jpg

designstream-renders/
└── renders/
    ├── {roomId}_{itemId}_clean.png
    └── {roomId}_{itemId}_render.png
```

### Environment Variables
```bash
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_REGION=us-east-1
```

## Usage Examples

### Python Client
```python
import requests

# Render furniture replacement
response = requests.post('http://localhost:5004/render', json={
    'roomId': 'living_room_1',
    'itemId': 'modern_sofa_2',
    'target_width': 500,
    'anchorX': 600,
    'baselineY': 450
})

result = response.json()
print(f"Render URL: {result['render_url']}")
```

### cURL
```bash
curl -X POST http://localhost:5004/render \
  -H "Content-Type: application/json" \
  -d '{
    "roomId": "living_room_1",
    "itemId": "modern_sofa_2", 
    "target_width": 500,
    "anchorX": 600,
    "baselineY": 450
  }'
```

## Performance

### Processing Times (Local)
- **Segmentation**: ~0.5s
- **Inpainting**: ~1.0s  
- **Cutout Creation**: ~2.0s
- **Shadow Generation**: ~0.2s
- **Compositing**: ~0.3s
- **Total**: ~4.0s (first run), ~0.1s (cached)

### Optimization Opportunities
1. **GPU Acceleration**: Use CUDA for cv2.inpaint
2. **Model Caching**: Keep rembg session in memory
3. **Async Processing**: Background job queue
4. **CDN Integration**: CloudFront for S3 assets

## Future Enhancements

### High-Quality Inpainting
- **LaMa**: Large Mask Inpainting for better results
- **SDXL**: Stable Diffusion XL for photorealistic inpainting
- **Custom Models**: Fine-tuned for furniture removal

### Advanced Features
- **Brush Tools**: Manual mask refinement UI
- **Multiple Furniture**: Batch processing
- **Real-time Preview**: WebSocket updates
- **A/B Testing**: Multiple render variants

### Production Considerations
- **Error Handling**: Retry logic, fallback methods
- **Monitoring**: Metrics, alerting, logging
- **Scaling**: Horizontal scaling, load balancing
- **Security**: Authentication, rate limiting

## Testing

### Local Test
```bash
python local_furniture_pipeline_test.py
```

### API Test
```bash
python test_furniture_pipeline.py
```

### Integration Test
```bash
# Start API server
python furniture_api_server.py

# Test endpoints
curl http://localhost:5005/health
curl -X POST http://localhost:5005/render -d @test_request.json
```

## Dependencies

```python
# Core
opencv-python>=4.8.0
Pillow>=10.0.0
rembg>=2.0.50
numpy>=1.24.0

# API
Flask>=2.3.0
requests>=2.31.0

# Storage
boto3>=1.28.0

# Optional (Future)
torch>=2.0.0
diffusers>=0.21.0
```

## License

MIT License - See LICENSE file for details.
