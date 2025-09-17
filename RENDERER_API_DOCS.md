# Renderer Service API Documentation

## Overview

The Renderer Service is a server-side 2D compositor that handles inpainting, perspective correction, lighting, and progressive rendering for furniture placement in room images.

## Features

- **2D Compositing**: Advanced image blending with proper alpha channel handling
- **Inpainting**: Removes original items from room images using OpenCV inpainting
- **Perspective Correction**: Automatically detects and applies room perspective
- **Progressive Rendering**: 960px preview (~1.5s) + 1920px full resolution
- **S3 Caching**: Intelligent caching with cache hit monitoring
- **Performance Monitoring**: Real-time render time and cache statistics
- **Batch Processing**: Multiple renders in a single request

## API Endpoints

### 1. Health Check

**GET** `/renderer/health`

Check if the renderer service is healthy and S3 is available.

**Response:**
```json
{
  "status": "healthy",
  "s3_available": true,
  "renderer_ready": true
}
```

### 2. Single Render

**GET** `/renderer/render`

Render a single item into a room image.

**Parameters:**
- `roomId` (string, required): Room image identifier
- `itemId` (string, required): Item identifier
- `size` (string, optional): "preview" (960px) or "full" (1920px), default: "full"
- `x` (int, optional): X position for item placement, default: 0
- `y` (int, optional): Y position for item placement, default: 0

**Example:**
```
GET /renderer/render?roomId=room_123&itemId=chair_456&size=preview&x=100&y=200
```

**Response:**
```json
{
  "success": true,
  "url": "https://s3.amazonaws.com/bucket/renders/room_123_chair_456_preview_1234567890.png",
  "cached": false,
  "size": "preview"
}
```

**Error Response:**
```json
{
  "error": "Room image not found"
}
```

### 3. Batch Render

**POST** `/renderer/render/batch`

Render multiple items in a single request.

**Request Body:**
```json
{
  "renders": [
    {
      "roomId": "room_123",
      "itemId": "chair_456",
      "size": "preview",
      "x": 100,
      "y": 200
    },
    {
      "roomId": "room_456",
      "itemId": "table_789",
      "size": "full",
      "x": 300,
      "y": 400
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "success": true,
      "url": "https://s3.amazonaws.com/bucket/renders/room_123_chair_456_preview_1234567890.png",
      "cached": false,
      "size": "preview",
      "roomId": "room_123",
      "itemId": "chair_456"
    },
    {
      "success": true,
      "url": "https://s3.amazonaws.com/bucket/renders/room_456_table_789_full_1234567890.png",
      "cached": true,
      "size": "full",
      "roomId": "room_456",
      "itemId": "table_789"
    }
  ],
  "total": 2
}
```

### 4. Render Statistics

**GET** `/renderer/stats`

Get performance metrics and cache statistics.

**Response:**
```json
{
  "success": true,
  "stats": {
    "cache_hits": 150,
    "cache_misses": 50,
    "cache_hit_rate": 0.75,
    "total_renders": 200,
    "avg_render_time": 1.2,
    "recent_renders": [1.1, 1.3, 1.0, 1.2, 1.4]
  }
}
```

## Rendering Pipeline

### 1. Image Loading
- Loads room image from S3: `uploads/{roomId}.jpg`
- Loads item cutout from S3: `renders/{itemId}_cutout.png`
- Loads item shadow from S3: `renders/{itemId}_shadow.png`

### 2. Preprocessing
- Resizes images to target resolution (960px or 1920px)
- Detects room perspective using edge detection
- Applies perspective transformation to item

### 3. Inpainting
- Removes original item from room using OpenCV inpainting
- Uses TELEA algorithm for high-quality results

### 4. Item Processing
- Applies perspective transformation to item
- Recolors item to match room lighting
- Generates realistic shadow

### 5. Compositing
- Blends shadow into room first
- Blends item with proper alpha channel handling
- Applies final color correction

### 6. Caching
- Saves result to S3 with hourly cache keys
- Returns cached URL for future requests

## Performance Targets

- **Preview Render**: ~1.5 seconds (960px)
- **Full Render**: ~3-5 seconds (1920px)
- **Cache Hit Rate**: >70% for production
- **Concurrent Renders**: Supports multiple simultaneous requests

## Error Handling

The service gracefully handles:
- Missing images (returns 404 with descriptive error)
- S3 connectivity issues (falls back to local processing)
- Invalid parameters (returns 400 with validation errors)
- Processing failures (returns 500 with error details)

## Monitoring

The service tracks:
- Render times for performance optimization
- Cache hit/miss rates for efficiency monitoring
- Error rates for reliability tracking
- Recent render times for trend analysis

## S3 Storage Structure

```
s3://bucket/
├── uploads/
│   ├── room_123.jpg          # Room images
│   └── room_456.jpg
├── renders/
│   ├── item_123_cutout.png   # Item cutouts
│   ├── item_123_shadow.png   # Item shadows
│   ├── room_123_item_456_preview_1234567890.png  # Cached renders
│   └── room_123_item_456_full_1234567890.png
```

## Usage Examples

### JavaScript/TypeScript
```javascript
// Single render
const response = await fetch('/renderer/render?roomId=room_123&itemId=chair_456&size=preview');
const data = await response.json();

if (data.success) {
  console.log('Render URL:', data.url);
  console.log('Cached:', data.cached);
}

// Batch render
const batchResponse = await fetch('/renderer/render/batch', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    renders: [
      { roomId: 'room_123', itemId: 'chair_456', size: 'preview', x: 100, y: 200 },
      { roomId: 'room_123', itemId: 'table_789', size: 'full', x: 300, y: 400 }
    ]
  })
});
```

### Python
```python
import requests

# Single render
response = requests.get('http://localhost:5003/renderer/render', params={
    'roomId': 'room_123',
    'itemId': 'chair_456',
    'size': 'preview',
    'x': 100,
    'y': 200
})

data = response.json()
if data['success']:
    print(f"Render URL: {data['url']}")

# Batch render
batch_data = {
    "renders": [
        {"roomId": "room_123", "itemId": "chair_456", "size": "preview", "x": 100, "y": 200},
        {"roomId": "room_123", "itemId": "table_789", "size": "full", "x": 300, "y": 400}
    ]
}

response = requests.post('http://localhost:5003/renderer/render/batch', json=batch_data)
data = response.json()
```

## Testing

Run the test suite:
```bash
python test_renderer.py
```

Run the demo with sample data:
```bash
python demo_renderer.py
```

## Dependencies

- OpenCV for image processing and inpainting
- PIL/Pillow for image manipulation
- NumPy for array operations
- Boto3 for S3 integration
- Flask for API endpoints
