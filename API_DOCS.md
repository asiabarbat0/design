# Auto Matting Service API Documentation

## Overview

The Auto Matting Service provides automatic background removal for furniture images using AI-powered segmentation and matting. It generates cutout images with transparent backgrounds and soft shadows, with confidence scoring to flag images that need manual review.

## Endpoints

### 1. Process Single Image

**POST** `/auto-matting/process`

Process a single furniture image for automatic background removal.

#### Request Body
```json
{
  "image_url": "https://example.com/furniture.jpg",
  "model": "auto|yolo|rembg|human",
  "generate_shadow": true,
  "store_result": true
}
```

#### Parameters
- `image_url` (string, required): URL of the furniture image to process
- `model` (string, optional): AI model to use
  - `auto`: Try YOLO first, fallback to rembg (default)
  - `yolo`: Use YOLO segmentation model
  - `rembg`: Use rembg general model
  - `human`: Use rembg human-specific model
- `generate_shadow` (boolean, optional): Generate shadow image (default: true)
- `store_result` (boolean, optional): Store result in database (default: true)

#### Response
```json
{
  "success": true,
  "cutout_url": "https://s3.../cutout.png",
  "shadow_url": "https://s3.../shadow.png",
  "confidence": 0.85,
  "needs_manual": false,
  "processing_time": 2.3,
  "model_used": "yolo",
  "image_hash": "abc123...",
  "result_id": 123
}
```

#### Response Fields
- `success` (boolean): Whether processing was successful
- `cutout_url` (string): URL of the cutout image with transparent background
- `shadow_url` (string): URL of the shadow image (if generated)
- `confidence` (float): Confidence score (0.0-1.0)
- `needs_manual` (boolean): Whether image needs manual review
- `processing_time` (float): Processing time in seconds
- `model_used` (string): AI model that was actually used
- `image_hash` (string): Unique hash of the processed image
- `result_id` (integer): Database ID of the stored result

### 2. Batch Processing

**POST** `/auto-matting/batch-process`

Process multiple images in batch.

#### Request Body
```json
{
  "images": [
    {
      "image_url": "https://example.com/furniture1.jpg",
      "model": "auto"
    },
    {
      "image_url": "https://example.com/furniture2.jpg",
      "model": "rembg"
    }
  ],
  "generate_shadow": true
}
```

#### Response
```json
{
  "success": true,
  "results": [
    {
      "success": true,
      "cutout_url": "https://s3.../cutout1.png",
      "shadow_url": "https://s3.../shadow1.png",
      "confidence": 0.85,
      "needs_manual": false
    },
    {
      "success": false,
      "error": "Processing failed"
    }
  ],
  "total_images": 2,
  "successful": 1,
  "processing_time": 4.2
}
```

### 3. Get Processing Status

**GET** `/auto-matting/status/{image_hash}`

Get processing status for a specific image.

#### Response
```json
{
  "image_hash": "abc123...",
  "cutout_url": "https://s3.../cutout.png",
  "quality_score": 0.85,
  "needs_manual": false,
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 4. Health Check

**GET** `/auto-matting/health`

Check service health and availability.

#### Response
```json
{
  "status": "healthy",
  "yolo_available": true,
  "s3_available": true,
  "rembg_available": true,
  "confidence_threshold": 0.7
}
```

## Matting Studio Endpoints

### 1. Studio Interface

**GET** `/matting-studio/`

Serve the Matting Studio web interface for manual review.

### 2. Get Review Queue

**GET** `/matting-studio/queue?limit=20&offset=0`

Get list of images that need manual review.

#### Response
```json
{
  "success": true,
  "images": [
    {
      "id": 123,
      "original_url": "https://example.com/furniture.jpg",
      "cutout_url": "https://s3.../cutout.png",
      "confidence": 0.45,
      "created_at": "2024-01-15T10:30:00Z",
      "needs_review": true
    }
  ],
  "total": 1,
  "has_more": false
}
```

### 3. Process Manual Matting

**POST** `/matting-studio/process/{image_id}`

Process manual matting adjustments.

#### Request Body
```json
{
  "brush_strokes": [
    {
      "type": "add",
      "x": 100,
      "y": 100,
      "radius": 20
    },
    {
      "type": "remove",
      "x": 200,
      "y": 200,
      "radius": 15
    }
  ],
  "refinement_mode": "smooth|sharpen|feather"
}
```

#### Response
```json
{
  "success": true,
  "cutout_url": "https://s3.../manual_cutout.png",
  "shadow_url": "https://s3.../manual_shadow.png",
  "confidence": 0.95,
  "needs_manual": false,
  "image_id": 123
}
```

### 4. Approve Image

**POST** `/matting-studio/approve/{image_id}`

Approve a manually processed image.

#### Response
```json
{
  "success": true,
  "message": "Image approved",
  "image_id": 123
}
```

### 5. Reject Image

**POST** `/matting-studio/reject/{image_id}`

Reject an image and flag for re-processing.

#### Request Body
```json
{
  "reason": "Quality not acceptable"
}
```

#### Response
```json
{
  "success": true,
  "message": "Image rejected",
  "reason": "Quality not acceptable",
  "image_id": 123
}
```

### 6. Get Statistics

**GET** `/matting-studio/stats`

Get matting studio statistics.

#### Response
```json
{
  "success": true,
  "stats": {
    "total_images": 150,
    "needs_review": 12,
    "approved": 135,
    "pending_approval": 12,
    "confidence_threshold": 0.7
  }
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "success": false,
  "error": "Error message description",
  "processing_time": 1.2
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `413 Payload Too Large`: Image file too large (>25MB)
- `415 Unsupported Media Type`: Unsupported image format
- `422 Unprocessable Entity`: Processing failed (e.g., no object detected)
- `500 Internal Server Error`: Server error

## Confidence Scoring

The service uses a sophisticated confidence scoring system that evaluates:

- **Alpha Channel Smoothness** (40%): Quality of the segmentation edges
- **Object Size Ratio** (30%): Whether the detected object is reasonably sized
- **Edge Density** (20%): Clean cutout without excessive edge artifacts
- **Object Presence** (10%): Confirmation that an object was detected

### Confidence Levels

- **0.8-1.0**: High confidence - Ready for production
- **0.6-0.8**: Medium confidence - May need review
- **0.0-0.6**: Low confidence - Requires manual cleanup

## S3 Storage

All processed images are stored in S3 with the following structure:

```
s3://designstream-uploads/
├── cutouts/
│   └── {image_hash}_cutout.png
├── shadows/
│   └── {image_hash}_shadow.png
├── manual_cutouts/
│   └── {uuid}_cutout.png
└── manual_shadows/
    └── {uuid}_shadow.png
```

## Usage Examples

### Python Example

```python
import requests

# Process a single image
response = requests.post('http://localhost:5002/auto-matting/process', json={
    'image_url': 'https://example.com/furniture.jpg',
    'model': 'auto',
    'generate_shadow': True
})

result = response.json()
if result['success']:
    print(f"Cutout: {result['cutout_url']}")
    print(f"Shadow: {result['shadow_url']}")
    print(f"Confidence: {result['confidence']}")
```

### cURL Example

```bash
curl -X POST http://localhost:5002/auto-matting/process \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/furniture.jpg",
    "model": "auto",
    "generate_shadow": true
  }'
```

## Testing

Use the provided test script to verify the service:

```bash
python test_auto_matting.py
```

This will test all endpoints and demonstrate the complete workflow.

