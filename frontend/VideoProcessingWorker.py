from PyQt5.QtCore import QThread, pyqtSignal

from video_processing.video_pipeline import VideoProcessor


class VideoProcessingWorker(QThread):
    """Worker thread for video processing to prevent UI blocking"""

    progress_updated = pyqtSignal(int, str)  # progress percentage, status message
    processing_finished = pyqtSignal(str, bool)  # output_path, success
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, video_path, output_dir, config):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.config = config
        self.processor = None

    def run(self):
        try:
            self.progress_updated.emit(0, "Initializing video processor...")

            # Create processor with config
            self.processor = VideoProcessor(
                detection_confidence=self.config["detection_confidence"],
                ocr_confidence_threshold=self.config["ocr_confidence_threshold"],
                frame_skip=self.config["frame_skip"],
                output_base_dir=self.output_dir,
            )

            self.progress_updated.emit(5, "Opening video file...")

            # Process the video with progress callback
            results = self._process_video_with_progress()

            self.progress_updated.emit(100, "Processing completed successfully!")
            self.processing_finished.emit(self.output_dir, True)

        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            self.error_occurred.emit(error_msg)
            self.processing_finished.emit(self.output_dir, False)

    def _process_video_with_progress(self):
        """Process video with detailed progress tracking"""
        import cv2
        from pathlib import Path

        # Open video to get metadata
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        try:
            # Get video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_name = Path(self.video_path).name

            self.progress_updated.emit(
                8, f"Video: {video_name} ({total_frames} frames)"
            )

            frame_number = 0
            processed_frames = 0
            results = []

            self.progress_updated.emit(10, "Starting frame processing...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frames according to frame_skip
                if frame_number % self.config["frame_skip"] == 0:
                    timestamp = frame_number / fps if fps > 0 else frame_number

                    # Process this frame using the original processor logic
                    frame_results = self._process_frame_with_processor(
                        frame, frame_number, timestamp
                    )
                    results.extend(frame_results)
                    processed_frames += 1

                frame_number += 1

                # Update progress based on actual frame progress
                if frame_number % 50 == 0 or frame_number == total_frames:
                    progress_percent = min(
                        90, int((frame_number / total_frames) * 80) + 10
                    )
                    frames_processed_text = f"Frame {frame_number}/{total_frames}"
                    if processed_frames > 0:
                        frames_processed_text += f" ({processed_frames} analyzed)"

                    self.progress_updated.emit(progress_percent, frames_processed_text)

        finally:
            cap.release()

        # Save results
        self.progress_updated.emit(95, "Saving results...")
        self._save_results(results)

        return results

    def _process_frame_with_processor(self, frame, frame_number, timestamp):
        """Process a single frame using the video processor logic"""
        import cv2
        import numpy as np
        from pathlib import Path

        temp_frame_path = Path(self.output_dir) / f"temp_frame_{frame_number}.jpg"
        frame_results = []

        try:
            # Save frame temporarily for detection
            cv2.imwrite(str(temp_frame_path), frame)

            # Detect license plates
            detections = self.processor.detector.detect_license_plates(
                str(temp_frame_path), confidence=self.config["detection_confidence"]
            )

            # Process each detected plate
            for plate_idx, detection in enumerate(detections):
                plate_result = self._process_detected_plate(
                    frame, detection, frame_number, timestamp, plate_idx
                )
                if plate_result:
                    frame_results.append(plate_result)

        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
        finally:
            # Clean up temporary file
            if temp_frame_path.exists():
                temp_frame_path.unlink()

        return frame_results

    def _process_detected_plate(
        self, frame, detection, frame_number, timestamp, plate_idx
    ):
        """Process a detected license plate"""
        import cv2
        from pathlib import Path

        bbox = detection["bbox"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = bbox

        # Extract license plate region with padding
        padding = 10
        h, w = frame.shape[:2]
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        cropped_plate = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        if cropped_plate.size == 0:
            return None

        # Save cropped plate
        plate_filename = f"plate_f{frame_number:06d}_t{timestamp:.3f}_p{plate_idx}_conf{confidence:.2f}.jpg"
        cropped_plates_dir = Path(self.output_dir) / "cropped_plates"
        cropped_plates_dir.mkdir(exist_ok=True)
        plate_path = cropped_plates_dir / plate_filename

        cv2.imwrite(str(plate_path), cropped_plate)

        # Perform OCR
        ocr_result = self.processor.ocr.extract_text(str(plate_path))

        # Extract VRN using improved logic
        ocr_text = str(ocr_result.get("text", ""))
        ocr_raw_text = str(ocr_result.get("raw_text", ""))
        ocr_confidence = float(ocr_result.get("confidence", 0.0))
        ocr_confidence_threshold = self.config.get("ocr_confidence_threshold", 50.0)

        # Improved VRN extraction logic
        vrn = ""

        # Primary: Use cleaned OCR text if it meets confidence threshold
        if ocr_text and ocr_confidence >= ocr_confidence_threshold:
            vrn = ocr_text
        # Secondary: Use cleaned raw text if main text is empty but raw text exists and meets threshold
        elif ocr_raw_text and ocr_confidence >= ocr_confidence_threshold:
            # Clean the raw text: remove spaces, convert to uppercase, keep only alphanumeric
            cleaned_raw = "".join(
                c for c in ocr_raw_text.upper().replace(" ", "") if c.isalnum()
            )
            if len(cleaned_raw) >= 2:  # At least 2 characters for a potential VRN
                vrn = cleaned_raw
        # Tertiary: If confidence is lower but text exists, still try to extract if it looks reasonable
        elif (
            ocr_text or ocr_raw_text
        ) and ocr_confidence >= 30.0:  # Lower threshold for fallback
            text_to_clean = ocr_text if ocr_text else ocr_raw_text
            cleaned_fallback = "".join(
                c for c in text_to_clean.upper().replace(" ", "") if c.isalnum()
            )
            if (
                len(cleaned_fallback) >= 3
            ):  # Slightly higher requirement for lower confidence
                vrn = cleaned_fallback

        # Create result entry
        result = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "bbox": [x1, y1, x2, y2],
            "detection_confidence": confidence,
            "cropped_plate_path": str(plate_path.relative_to(Path(self.output_dir))),
            "vrn": vrn,
            "raw_ocr_text": ocr_raw_text,
            "ocr_confidence": ocr_confidence,
        }

        return result

    def _save_results(self, results):
        """Save results to JSON file"""
        import json
        from pathlib import Path
        from datetime import datetime

        output_data = {
            "video_path": self.video_path,
            "processing_timestamp": datetime.now().isoformat(),
            "configuration": self.config,
            "total_detections": len(results),
            "results": results,
        }

        results_path = Path(self.output_dir) / "results.json"
        with open(results_path, "w") as f:
            json.dump(output_data, f, indent=2)
