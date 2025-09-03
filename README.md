# Automatic Number Plate Recognition (ANPR) System

This is a comprehensive license plate detection and recognition system built with Python with a PyQt5 frontend. It provides both video processing capabilities and live stream monitoring for automatic number plate recognition using advanced computer vision models.

## Features

- **Video Processing**: Process pre-recorded video files for license plate detection and OCR
- **Live Stream Monitoring**: Real-time ANPR from RTSP camera streams
- **GUI Interface**: User-friendly PyQt5 interface for easy operation
- **Batch Processing**: Process multiple videos with configurable parameters
- **S3 Integration**: Download and process videos from S3-compatible storage
- **Export Capabilities**: Export detection results to CSV format with duplicate removal
- **Advanced OCR**: Multiple OCR engines for accurate text extraction (EasyOCR has the best performance out the box, Tesseract as backup - with fine-tuning performance would be significantly better than EasyOCR)

## Computer Vision Models

### License Plate Detection
- **Primary Model**: Custom YOLOv8 license plate detection model (`license-plate-recognition.pt`, downloadable from: https://huggingface.co/yasirfaizahmed/license-plate-object-detection/blob/main/best.pt)
- **Fallback Model**: YOLOv8n general object detection model (`yolov8n.pt`, will auto download if not present locally - **the previous plate detection fine-tuned model is recommended for best results**)
- **Detection Strategy**: Uses vehicle detection (cars, trucks, buses, motorcycles) as regions of interest for license plates

### OCR (Optical Character Recognition)
- **Primary Engine**: EasyOCR for robust text extraction
- **Backup Engine**: Tesseract OCR (available in `ocr_tesseract_backup.py`, **requires fine-tuning**)
- **GPU Support**: Optional GPU acceleration for faster processing (will auto enable if GPU is available)

## Installation

### Prerequisites
- Python 3.8 or higher (built on 3.13)
- OpenCV-compatible system
- CUDA (optional, for GPU acceleration)

### Clone the Repository
```bash
gh repo clone Jacob-Haynes/cv-anpr
cd watchkeeper
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Additional Requirements
- **Tesseract OCR** (if using backup OCR engine):
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt-get install tesseract-ocr`
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Model Files
The required model files should be placed in `cv/models/`:
- `license-plate-recognition.pt` - Custom YOLOv8 license plate detection model from: https://huggingface.co/yasirfaizahmed/license-plate-object-detection/blob/main/best.pt
- `yolov8n.pt` - YOLOv8n general object detection model, will auto download on first run if custom license plate model is not found

## Usage

### Quick Start
Run the main GUI application:
```bash
python main.py
```

### GUI Features

#### Video Processing
1. **Load Videos**: Import video files from local storage or S3 (configure S3 in `.env`, see `.envexample`)
2. **Configure Processing**: Set detection confidence, OCR thresholds, frame skip rates
3. **Process Videos**: Run analysis on selected video with progress tracking
4. **Review Results**: View detected plates, OCR results, and confidence scores within UI with auto video seeking.
5. **Export Data**: Save results to CSV with customizable fields

#### Live Stream Monitoring
1. **Configure Stream**: Enter RTSP URL in `.env` for live camera feed
2. **Real-time Detection**: Monitor live license plate detections
3. **Save Results**: Automatically save detected plates and OCR results and view list within UI.

### Configuration Options

#### Detection Parameters
- **Detection Confidence**: Minimum confidence threshold for plate detection (0.0-1.0)
- **OCR Confidence**: Minimum confidence threshold for text recognition (0-100)
- **Frame Skip**: Process every Nth frame to improve performance
- **Crop Padding**: Additional pixels around detected plates for better OCR

#### Output Options
- **Save Cropped Plates**: Extract and save individual license plate images
- **OCR Processing**: Enable/disable text extraction
- **Result Formats**: JSON and CSV export options

## Project Structure

```
watchkeeper/
├── main.py                     # Main application entry point
├── requirements.txt            # Python dependencies
├── cv/                         # Computer vision modules
│   ├── plate_detection.py      # YOLOv8 license plate detection
│   ├── ocr.py                  # EasyOCR text extraction
│   ├── ocr_tesseract_backup.py # Tesseract OCR fallback
│   └── models/                 # Pre-trained model files
├── frontend/                   # PyQt5 GUI components
│   ├── gui.py                  # Main window interface
│   ├── live_stream_widget.py   # Live stream monitoring
│   └── [other GUI dialogs]     # Configuration and progress dialogs
├── video_processing/           # Video processing pipeline
│   ├── video_pipeline.py       # Main processing logic
│   └── data_processing.py      # Result data handling
├── live_stream/               # Real-time processing
│   └── live_video.py          # RTSP stream handling
├── videos/                    # Video management
│   ├── get_s3_videos.py       # S3 video downloading
│   └── local_videos/          # Local video storage
└── output/                    # Processing results
    ├── [video_name]/          # Per-video results
    │   ├── results.json       # Detection metadata
    │   ├── cropped_plates/    # Extracted plate images
    │   └── ocr_output/        # OCR visualization
    └── live_stream/           # Live stream results
```

## Output Format

### JSON Results
Each processed video generates a `results.json` file containing:
```json
{
  "video_info": {
    "filename": "video.mp4",
    "total_frames": 1500,
    "fps": 30.0,
    "duration": 50.0
  },
  "detections": [
    {
      "frame_number": 100,
      "timestamp": 3.33,
      "plate_id": 0,
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.85,
      "ocr_text": "ABC123",
      "ocr_confidence": 87.5,
      "cropped_image_path": "cropped_plates/plate_f000100_t3.33_p0_conf0.85.jpg"
    }
  ]
}
```

### CSV Export
Customizable CSV export with fields including:
- Frame number and timestamp
- License plate text and confidence
- Bounding box coordinates
- Detection confidence scores
- File paths to cropped images

## Configuration

### Environment Variables
Create a `.env` file for S3 configuration:
```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=your_bucket_name
AWS_S3_ENDPOINT_URL=your_s3_endpoint
```

### Performance Tuning
- **GPU Acceleration**: Enable GPU support in OCR settings for faster processing
- **Frame Skipping**: Increase frame skip for faster processing of long videos
- **Confidence Thresholds**: Adjust to balance accuracy vs. detection quantity
- **Resolution**: Lower input resolution can improve processing speed

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model files are in `cv/models/` directory
   - Check file permissions and paths

2. **OCR Performance**
   - Install GPU-compatible versions for acceleration
   - Adjust confidence thresholds for better results
   - Ensure adequate lighting in video/stream

3. **Live Stream Issues**
   - Verify RTSP URL format and accessibility
   - Check network connectivity and permissions

4. **Memory Issues**
   - Increase frame skip rate
   - Close unnecessary applications

