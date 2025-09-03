"""
Video Processing Pipeline for License Plate Detection and OCR

This module provides a comprehensive video processing pipeline that combines license plate
detection using YOLOv8 and text extraction using EasyOCR. It processes video files frame
by frame, extracts license plate regions, performs OCR, and generates structured output.

Classes:
    VideoProcessor: Main pipeline for processing videos and extracting license plate data

Example:
    processor = VideoProcessor(
        detection_confidence=0.6,
        ocr_confidence_threshold=60.0,
        frame_skip=100
    )
    results = processor.process_video("video.mp4", "results.json")
"""

import cv2
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np

# Import our custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cv.plate_detection import LicensePlateDetector
from cv.ocr import LicensePlateOCR


class VideoProcessorConfig:
    """Configuration constants for the VideoProcessor."""

    DEFAULT_CROP_PADDING = 10
    PROGRESS_UPDATE_INTERVAL = 100
    TEMP_FRAME_PREFIX = "temp_frame_"
    CROPPED_PLATE_PREFIX = "plate_f"
    OCR_OUTPUT_PREFIX = "ocr_f"

    # Output directory names
    CROPPED_PLATES_DIR = "cropped_plates"
    OCR_OUTPUT_DIR = "ocr_output"


class VideoProcessor:
    """
    Video Processing Pipeline for License Plate Detection and OCR.

    This class provides a complete pipeline for processing video files to detect license
    plates and extract text information. It handles frame extraction, detection, OCR,
    and result storage with configurable parameters for performance optimization.

    Attributes:
        detector: License plate detection model
        ocr: OCR text extraction system
        detection_confidence: Minimum confidence for plate detection
        ocr_confidence_threshold: Minimum confidence for OCR validation
        frame_skip: Number of frames to skip between processing
        output_base: Base directory for output files
        results: List of processing results
    """

    def __init__(
        self,
        detection_confidence: float = 0.5,
        ocr_confidence_threshold: float = 50.0,
        frame_skip: int = 30,
        output_base_dir: str = "output",
    ) -> None:
        """
        Initialize the video processor with configuration parameters.

        Args:
            detection_confidence: Minimum confidence threshold for license plate detection (0.0-1.0)
            ocr_confidence_threshold: Minimum confidence threshold for OCR text recognition
            frame_skip: Process every Nth frame (higher values = faster processing)
            output_base_dir: Base directory for saving output files
        """
        self._validate_initialization_params(
            detection_confidence, ocr_confidence_threshold, frame_skip
        )

        # Initialize components
        self.detector = LicensePlateDetector(use_license_plate_model=True)
        self.ocr = LicensePlateOCR()
        print("Using EasyOCR for text recognition")

        # Store configuration
        self.detection_confidence = detection_confidence
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.frame_skip = frame_skip

        # Set up output directories
        self.output_base = Path(output_base_dir)
        self.cropped_plates_dir = (
            self.output_base / VideoProcessorConfig.CROPPED_PLATES_DIR
        )
        self.ocr_output_dir = self.output_base / VideoProcessorConfig.OCR_OUTPUT_DIR

        self._setup_output_directories()
        self.results: List[Dict[str, Any]] = []

    def _validate_initialization_params(
        self,
        detection_confidence: float,
        ocr_confidence_threshold: float,
        frame_skip: int,
    ) -> None:
        """Validate initialization parameters."""
        if not 0.0 <= detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be between 0.0 and 1.0")
        if ocr_confidence_threshold < 0:
            raise ValueError("ocr_confidence_threshold must be >= 0")
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")

    def _setup_output_directories(self) -> None:
        """Create necessary output directories."""
        self.cropped_plates_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_output_dir.mkdir(parents=True, exist_ok=True)

    def process_video(
        self, video_path: str, output_json: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a video file and extract license plate information.

        Args:
            video_path: Path to the input video file
            output_json: Optional filename for JSON output results

        Returns:
            List of detection results with OCR data

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video file cannot be opened
        """
        self._validate_video_path(video_path)

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            video_info = self._extract_video_metadata(cap, video_path)
            self._display_video_info(video_info)

            self.results = []
            self._process_all_frames(cap, video_info)

        finally:
            cap.release()

        self._display_processing_summary()

        if output_json:
            self._save_results_to_json(output_json)

        return self.results

    def _validate_video_path(self, video_path: str) -> None:
        """Validate that the video file exists and is accessible."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

    def _extract_video_metadata(
        self, cap: cv2.VideoCapture, video_path: str
    ) -> Dict[str, Any]:
        """Extract comprehensive video information."""
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        return {
            "name": Path(video_path).name,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
        }

    def _display_video_info(self, video_info: Dict[str, Any]) -> None:
        """Display video information and processing parameters."""
        print(f"Processing video: {video_info['name']}")
        print(
            f"FPS: {video_info['fps']:.2f}, "
            f"Total frames: {video_info['total_frames']}, "
            f"Duration: {video_info['duration']:.2f}s"
        )
        print(f"Processing every {self.frame_skip} frame(s)")

    def _process_all_frames(
        self, cap: cv2.VideoCapture, video_info: Dict[str, Any]
    ) -> None:
        """Process all frames in the video according to frame_skip setting."""
        frame_number = 0
        processed_frames = 0
        fps = video_info["fps"]
        total_frames = video_info["total_frames"]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % self.frame_skip == 0:
                timestamp = frame_number / fps if fps > 0 else frame_number
                frame_results = self._process_single_frame(
                    frame, frame_number, timestamp
                )
                self.results.extend(frame_results)

                processed_frames += 1
                if (
                    processed_frames % VideoProcessorConfig.PROGRESS_UPDATE_INTERVAL
                    == 0
                ):
                    print(
                        f"Processed {processed_frames} frames "
                        f"(frame {frame_number}/{total_frames})"
                    )

            frame_number += 1

    def _process_single_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> List[Dict[str, Any]]:
        """
        Process a single frame to detect plates and perform OCR.

        Args:
            frame: Video frame as numpy array
            frame_number: Current frame number
            timestamp: Timestamp in seconds

        Returns:
            List of detection results for this frame
        """
        temp_frame_path = (
            self.output_base
            / f"{VideoProcessorConfig.TEMP_FRAME_PREFIX}{frame_number}.jpg"
        )
        frame_results = []

        try:
            # Save frame temporarily for detection
            cv2.imwrite(str(temp_frame_path), frame)

            # Detect license plates
            detections = self.detector.detect_license_plates(
                str(temp_frame_path), confidence=self.detection_confidence
            )

            # Process each detected plate
            for plate_idx, detection in enumerate(detections):
                plate_result = self._process_detected_plate(
                    frame, detection, frame_number, timestamp, plate_idx
                )
                frame_results.append(plate_result)

        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
        finally:
            # Clean up temporary file
            if temp_frame_path.exists():
                temp_frame_path.unlink()

        return frame_results

    def _process_detected_plate(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        frame_number: int,
        timestamp: float,
        plate_idx: int,
    ) -> Dict[str, Any]:
        """
        Process a single detected license plate through the complete pipeline.

        Args:
            frame: Original video frame
            detection: Detection result with bbox and confidence
            frame_number: Current frame number
            timestamp: Frame timestamp in seconds
            plate_idx: Index of this plate in the current frame

        Returns:
            Complete result dictionary for this detection
        """
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = bbox

        # Extract and save license plate region
        cropped_plate = self._crop_plate_region(frame, x1, y1, x2, y2)
        crop_path = self._save_cropped_plate_image(
            cropped_plate, frame_number, timestamp, plate_idx, confidence
        )

        # Perform OCR on cropped plate
        ocr_result = self.ocr.extract_text(cropped_plate)

        # Create structured result
        result = self._create_detection_result(
            frame_number, timestamp, confidence, bbox, crop_path, ocr_result
        )

        # Save individual OCR result
        self._save_individual_ocr_result(result, frame_number, timestamp, plate_idx)

        return result

    def _crop_plate_region(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """
        Crop the license plate region with padding to improve OCR accuracy.

        Args:
            frame: Source video frame
            x1, y1, x2, y2: Bounding box coordinates

        Returns:
            Cropped plate image with padding
        """
        h, w = frame.shape[:2]
        padding = VideoProcessorConfig.DEFAULT_CROP_PADDING

        # Apply padding while staying within frame boundaries
        y1_crop = max(0, y1 - padding)
        y2_crop = min(h, y2 + padding)
        x1_crop = max(0, x1 - padding)
        x2_crop = min(w, x2 + padding)

        return frame[y1_crop:y2_crop, x1_crop:x2_crop]

    def _save_cropped_plate_image(
        self,
        cropped_plate: np.ndarray,
        frame_number: int,
        timestamp: float,
        plate_idx: int,
        confidence: float,
    ) -> Path:
        """Save the cropped plate image with descriptive filename."""
        timestamp_str = f"{timestamp:.3f}".replace(".", "_")
        filename = (
            f"{VideoProcessorConfig.CROPPED_PLATE_PREFIX}{frame_number:06d}_"
            f"t{timestamp_str}_p{plate_idx}_conf{confidence:.2f}.jpg"
        )
        file_path = self.cropped_plates_dir / filename
        cv2.imwrite(str(file_path), cropped_plate)
        return file_path

    def _create_detection_result(
        self,
        frame_number: int,
        timestamp: float,
        confidence: float,
        bbox: List[int],
        crop_path: Path,
        ocr_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a comprehensive result dictionary for a detection.

        Args:
            frame_number: Frame number where detection occurred
            timestamp: Timestamp in seconds
            confidence: Detection confidence score
            bbox: Bounding box coordinates
            crop_path: Path to saved cropped image
            ocr_result: OCR extraction results

        Returns:
            Structured result dictionary with all detection and OCR data
        """
        # Ensure all values are JSON serializable
        ocr_text = str(ocr_result.get("text", ""))
        ocr_raw_text = str(ocr_result.get("raw_text", ""))
        ocr_confidence = float(ocr_result.get("confidence", 0.0))
        ocr_valid = bool(ocr_result.get("valid", False))

        # Apply confidence threshold to determine final validity
        ocr_meets_threshold = ocr_confidence >= self.ocr_confidence_threshold
        final_ocr_valid = ocr_valid and ocr_meets_threshold

        # Use cleaned text as VRN if valid, otherwise use raw text if available, or empty string
        vrn = ""
        if final_ocr_valid and ocr_text:
            vrn = ocr_text
        elif ocr_raw_text and ocr_confidence >= self.ocr_confidence_threshold:
            # Fallback: use raw text cleaned up if main text is empty but raw text exists
            vrn = "".join(
                c for c in ocr_raw_text.upper().replace(" ", "") if c.isalnum()
            )

        return {
            "frame_number": int(frame_number),
            "timestamp": round(float(timestamp), 3),
            "detection_confidence": round(float(confidence), 3),
            "bbox": [int(x) for x in bbox],
            "cropped_plate_path": str(crop_path),
            "vrn": vrn,
            "raw_ocr_text": ocr_raw_text,
            "ocr_confidence": ocr_confidence,
            "ocr_valid": final_ocr_valid,
            "processing_timestamp": datetime.now().isoformat(),
        }

    def _save_individual_ocr_result(
        self,
        result: Dict[str, Any],
        frame_number: int,
        timestamp: float,
        plate_idx: int,
    ) -> None:
        """Save individual OCR result to a JSON file."""
        timestamp_str = f"{timestamp:.3f}".replace(".", "_")
        filename = f"{VideoProcessorConfig.OCR_OUTPUT_PREFIX}{frame_number:06d}_t{timestamp_str}_p{plate_idx}.json"
        file_path = self.ocr_output_dir / filename

        with open(file_path, "w") as f:
            json.dump(result, f, indent=2)

    def _display_processing_summary(self) -> None:
        """Display a summary of processing results."""
        valid_ocr_count = len([r for r in self.results if r["ocr_valid"]])
        total_detections = len(self.results)

        print("Video processing complete!")
        print(f"Total plates detected: {total_detections}")
        print(f"Valid OCR results: {valid_ocr_count}")

        if total_detections > 0:
            success_rate = (valid_ocr_count / total_detections) * 100
            print(f"OCR success rate: {success_rate:.1f}%")

    def _save_results_to_json(self, output_filename: str) -> None:
        """Save all results to a comprehensive JSON file."""
        output_path = self.output_base / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary_data = self._create_comprehensive_summary()

        with open(output_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Results saved to: {output_path}")

    def _create_comprehensive_summary(self) -> Dict[str, Any]:
        """Create a comprehensive summary dictionary for JSON output."""
        stats = self._calculate_processing_statistics()

        return {
            "processing_summary": {
                "total_detections": stats["total_detections"],
                "valid_ocr_results": stats["valid_ocr_results"],
                "unique_vrns_detected": stats["unique_vrns_detected"],
                "processing_date": datetime.now().isoformat(),
                "configuration": {
                    "detection_confidence_threshold": self.detection_confidence,
                    "ocr_confidence_threshold": self.ocr_confidence_threshold,
                    "frame_skip": self.frame_skip,
                },
            },
            "results": self.results,
        }

    def _calculate_processing_statistics(self) -> Dict[str, int]:
        """Calculate comprehensive statistics from processing results."""
        total_detections = len(self.results)
        valid_ocr_results = len([r for r in self.results if r["ocr_valid"]])
        unique_vrns = len(
            set(r["vrn"] for r in self.results if r["ocr_valid"] and r["vrn"])
        )

        return {
            "total_detections": total_detections,
            "valid_ocr_results": valid_ocr_results,
            "unique_vrns_detected": unique_vrns,
        }
