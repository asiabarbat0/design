# Model Files Download

This repository requires some large model files that are not included due to GitHub's file size limits.

## Required Model Files

### YOLO Models
Download these files and place them in the project root:

1. **yolov8n-seg.pt** (6.2 MB)
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
   ```

2. **yolov8x-seg.pt** (137.4 MB)
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt
   ```

### Alternative Download Methods

#### Using curl:
```bash
curl -L -o yolov8n-seg.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
curl -L -o yolov8x-seg.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt
```

#### Using Python:
```python
import urllib.request

# Download YOLOv8n-seg
urllib.request.urlretrieve(
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt",
    "yolov8n-seg.pt"
)

# Download YOLOv8x-seg
urllib.request.urlretrieve(
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt",
    "yolov8x-seg.pt"
)
```

## Model Usage

- **yolov8n-seg.pt**: Lightweight model for fast inference
- **yolov8x-seg.pt**: High-accuracy model for precise segmentation

The application will automatically use the available models. If neither is present, it will fall back to basic computer vision techniques.

## File Structure After Download

```
designstreamaigrok/
├── yolov8n-seg.pt      # 6.2 MB
├── yolov8x-seg.pt      # 137.4 MB
├── app/
├── migrations/
└── ...
```

## Notes

- These models are used for furniture detection and segmentation
- The application will work without them but with reduced accuracy
- Models are automatically downloaded on first run if not present (in some configurations)
