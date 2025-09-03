"""
License Plate Detection Video Processor GUI

A PyQt5-based application for processing videos with license plate detection
and OCR capabilities, featuring video playback with detection overlays.
"""

import sys
import os
import json
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QLineEdit,
    QGroupBox,
    QFormLayout,
    QSlider,
    QDialog,
    QDialogButtonBox,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)
from PyQt5.QtCore import Qt, QUrl, QTimer, QThread, pyqtSignal, QRectF, QSizeF
from PyQt5.QtGui import QPixmap, QColor, QPainter, QPen, QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem

from video_processing.video_pipeline import VideoProcessor
from video_processing.data_processing import DataProcessor

# Constants
LOCAL_VIDEOS_DIR = "videos/local_videos"
OUTPUT_DIR = "output"
RESULTS_JSON = "results.json"

# Styling constants
PROCESS_BUTTON_STYLE = """
    QPushButton {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 8px;
        border: none;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
    QPushButton:pressed {
        background-color: #3d8b40;
    }
"""

DETECTION_INFO_STYLE = """
    QLabel {
        background-color: rgba(0, 0, 0, 180);
        color: white;
        padding: 8px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 12px;
    }
"""


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
            cleaned_raw = ''.join(c for c in ocr_raw_text.upper().replace(' ', '') if c.isalnum())
            if len(cleaned_raw) >= 2:  # At least 2 characters for a potential VRN
                vrn = cleaned_raw
        # Tertiary: If confidence is lower but text exists, still try to extract if it looks reasonable
        elif (ocr_text or ocr_raw_text) and ocr_confidence >= 30.0:  # Lower threshold for fallback
            text_to_clean = ocr_text if ocr_text else ocr_raw_text
            cleaned_fallback = ''.join(c for c in text_to_clean.upper().replace(' ', '') if c.isalnum())
            if len(cleaned_fallback) >= 3:  # Slightly higher requirement for lower confidence
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


class VideoProcessingConfigDialog(QDialog):
    """Dialog for configuring video processing parameters"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Processing Configuration")
        self.setModal(True)
        self.resize(400, 300)

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Configuration form
        form_layout = QFormLayout()

        # Detection confidence
        self.detection_confidence = QDoubleSpinBox()
        self.detection_confidence.setRange(0.1, 1.0)
        self.detection_confidence.setValue(0.5)
        self.detection_confidence.setSingleStep(0.1)
        self.detection_confidence.setDecimals(1)
        form_layout.addRow("Detection Confidence:", self.detection_confidence)

        # OCR confidence threshold
        self.ocr_confidence = QDoubleSpinBox()
        self.ocr_confidence.setRange(0.0, 100.0)
        self.ocr_confidence.setValue(50.0)
        self.ocr_confidence.setSingleStep(5.0)
        self.ocr_confidence.setDecimals(1)
        form_layout.addRow("OCR Confidence Threshold:", self.ocr_confidence)

        # Frame skip
        self.frame_skip = QSpinBox()
        self.frame_skip.setRange(1, 1000)
        self.frame_skip.setValue(30)
        form_layout.addRow("Frame Skip (process every Nth frame):", self.frame_skip)

        layout.addLayout(form_layout)

        # Add description
        description = QLabel(
            "Configuration Help:\n"
            "• Detection Confidence: Higher values = fewer but more accurate detections\n"
            "• OCR Confidence: Minimum confidence for text recognition\n"
            "• Frame Skip: Higher values = faster processing but may miss plates"
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(description)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_config(self):
        """Return the configuration as a dictionary"""
        return {
            "detection_confidence": self.detection_confidence.value(),
            "ocr_confidence_threshold": self.ocr_confidence.value(),
            "frame_skip": self.frame_skip.value(),
        }


class VideoProcessingProgressDialog(QDialog):
    """Dialog showing video processing progress"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Video...")
        self.setModal(True)
        self.resize(500, 200)

        # Prevent closing during processing
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Status label
        self.status_label = QLabel("Initializing...")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)
        layout.addWidget(self.log_output)

        # Cancel button (initially hidden)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.hide()
        layout.addWidget(self.cancel_button)

        self.processing_complete = False

    def update_progress(self, percentage, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        self.log_output.append(f"{percentage}%: {message}")

        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def show_error(self, error_message):
        """Show error message"""
        self.status_label.setText("Error occurred!")
        self.log_output.append(f"ERROR: {error_message}")
        self.cancel_button.setText("Close")
        self.cancel_button.show()
        self.processing_complete = True

    def processing_finished(self, success):
        """Handle processing completion"""
        if success:
            self.status_label.setText("Processing completed successfully!")
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText("Processing failed!")

        self.cancel_button.setText("Close")
        self.cancel_button.show()
        self.processing_complete = True


class ExportCSVConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export to CSV Options")
        self.setModal(True)
        self.resize(300, 150)
        layout = QVBoxLayout()
        self.setLayout(layout)
        form_layout = QFormLayout()
        self.dedup_window = QDoubleSpinBox()
        self.dedup_window.setRange(0.0, 60.0)
        self.dedup_window.setValue(5.0)
        self.dedup_window.setSingleStep(0.5)
        form_layout.addRow("Dedup Window (seconds):", self.dedup_window)
        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setRange(0.0, 1.0)
        self.min_confidence.setValue(0.3)
        self.min_confidence.setSingleStep(0.05)
        form_layout.addRow("Min Confidence:", self.min_confidence)
        layout.addLayout(form_layout)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_config(self):
        return {
            "dedup_window": self.dedup_window.value(),
            "min_confidence": self.min_confidence.value(),
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("License Plate Detection Video Processor")
        self.resize(1400, 900)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create horizontal layout for main content
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Left side - main content
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_widget.setLayout(self.left_layout)
        self.main_layout.addWidget(self.left_widget, 2)  # Takes 2/3 of space

        # Right side - video sidebar
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_widget.setLayout(self.right_layout)
        self.main_layout.addWidget(self.right_widget, 1)  # Takes 1/3 of space

        # Left side content
        self.video_list = QListWidget()
        self.video_list_label = QLabel("Available Videos:")
        self.left_layout.addWidget(self.video_list_label)
        self.left_layout.addWidget(self.video_list)

        self.play_button = QPushButton("Play Selected Video")
        self.left_layout.addWidget(self.play_button)
        self.play_button.clicked.connect(self.play_selected_video)

        # Upload button
        self.upload_button = QPushButton("Upload Video")
        self.left_layout.addWidget(self.upload_button)
        self.upload_button.clicked.connect(self.upload_video)

        # Add Download from S3 button next to Upload Video
        self.download_s3_button = QPushButton("Download from S3")
        self.left_layout.addWidget(self.download_s3_button)
        self.download_s3_button.clicked.connect(self.download_from_s3)

        # Process Video button
        self.process_video_button = QPushButton("Process Video")
        self.process_video_button.setStyleSheet(PROCESS_BUTTON_STYLE)
        self.left_layout.addWidget(self.process_video_button)
        self.process_video_button.clicked.connect(self.process_selected_video)

        # Results explorer
        self.results_group = QGroupBox("Processed Results Explorer")
        self.results_layout = QVBoxLayout()
        self.results_group.setLayout(self.results_layout)
        self.left_layout.addWidget(self.results_group)

        self.processed_video_list = QListWidget()
        self.results_layout.addWidget(QLabel("Processed Videos:"))
        self.results_layout.addWidget(self.processed_video_list)
        self.processed_video_list.itemSelectionChanged.connect(
            self.load_results_for_selected_video
        )

        self.search_layout = QFormLayout()
        self.vrn_search = QLineEdit()
        self.raw_ocr_search = QLineEdit()  # Add search field for raw OCR text
        self.timestamp_search = QLineEdit()
        self.search_layout.addRow("Search VRN:", self.vrn_search)
        self.search_layout.addRow(
            "Search Raw OCR:", self.raw_ocr_search
        )  # Add raw OCR search
        self.search_layout.addRow("Timestamp Range (start,end):", self.timestamp_search)
        self.results_layout.addLayout(self.search_layout)
        self.search_button = QPushButton("Search")
        self.results_layout.addWidget(self.search_button)
        self.search_button.clicked.connect(self.search_results)

        # Make results table larger
        self.results_table = QTableWidget()
        self.results_table.setMinimumHeight(400)
        self.results_layout.addWidget(self.results_table)
        self.results_table.cellClicked.connect(self.on_result_cell_clicked)
        self.results_table.itemChanged.connect(self.on_result_item_changed)

        # Export to CSV button
        self.export_csv_button = QPushButton("Export to CSV")
        self.results_layout.addWidget(self.export_csv_button)
        self.export_csv_button.clicked.connect(self.export_results_to_csv)

        # Store current results data for editing
        self.current_results_data = []
        self.current_video_dir = None

        # Right side - Video sidebar
        self.video_filename_label = QLabel("No video loaded.")
        self.right_layout.addWidget(self.video_filename_label)

        # --- REPLACEMENT START: Replace QVideoWidget and overlay with QGraphicsView ---
        # Replace QVideoWidget and the overlay with QGraphicsView
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setMinimumSize(640, 480)  # Double the minimum size

        # Set better scaling and view properties
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setFrameStyle(0)  # Remove frame border

        # Make the graphics view expand more aggressively
        from PyQt5.QtWidgets import QSizePolicy

        self.graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.right_layout.addWidget(self.graphics_view)

        # Create the video item which will be part of the scene
        self.video_item = QGraphicsVideoItem()
        self.graphics_scene.addItem(self.video_item)

        # Set up media player
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        # CRITICAL: Set the video output to the QGraphicsVideoItem
        self.media_player.setVideoOutput(self.video_item)

        # Keep a list of bounding box items to easily remove them
        self.bbox_items = []
        # --- REPLACEMENT END ---

        # Detection info display
        self.detection_info_label = QLabel()
        self.detection_info_label.setStyleSheet(DETECTION_INFO_STYLE)
        self.detection_info_label.setAlignment(Qt.AlignCenter)
        self.detection_info_label.hide()
        self.right_layout.addWidget(self.detection_info_label)

        # Initialize variables
        self.overlay_data = []
        self.video_native_size = (1280, 720)

        # Status label for video playback
        self.video_status_label = QLabel()
        self.right_layout.addWidget(self.video_status_label)

        # Add stretch to push video controls to top
        self.right_layout.addStretch()

        # Video controls
        self.video_controls_widget = QWidget()
        self.video_controls_layout = QHBoxLayout()
        self.video_controls_widget.setLayout(self.video_controls_layout)

        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.video_controls_layout.addWidget(self.play_pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        self.video_controls_layout.addWidget(self.stop_button)

        # Add position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.sliderPressed.connect(self.slider_pressed)
        self.position_slider.sliderReleased.connect(self.slider_released)

        # Add timestamp labels
        self.current_time_label = QLabel("00:00")
        self.current_time_label.setMinimumWidth(50)
        self.total_time_label = QLabel("00:00")
        self.total_time_label.setMinimumWidth(50)

        # Add to layout: current time, slider, total time
        self.video_controls_layout.addWidget(self.current_time_label)
        self.video_controls_layout.addWidget(self.position_slider)
        self.video_controls_layout.addWidget(self.total_time_label)

        self.right_layout.addWidget(self.video_controls_widget)

        self.media_player.mediaStatusChanged.connect(self.handle_media_status)
        self.media_player.error.connect(self.handle_media_error)
        self.media_player.positionChanged.connect(self.on_video_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.media_player.stateChanged.connect(
            self.on_state_changed
        )  # Add state change handler

        self.load_video_list()
        self.load_processed_video_list()

        self.detections = []  # List of dicts from results.json
        self.overlay_timer = QTimer(self)
        self.overlay_timer.setSingleShot(True)
        self.overlay_timer.timeout.connect(self.hide_overlay)

    def format_time(self, seconds):
        """Format time in seconds to MM:SS or HH:MM:SS format"""
        if seconds < 0:
            return "00:00"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def parse_time(self, time_str):
        """Parse time string (MM:SS or HH:MM:SS) back to seconds"""
        try:
            parts = time_str.split(':')
            if len(parts) == 2:  # MM:SS format
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 3:  # HH:MM:SS format
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            else:
                return 0.0
        except (ValueError, AttributeError):
            return 0.0

    def load_video_list(self):
        self.video_list.clear()
        video_dir = Path(LOCAL_VIDEOS_DIR)
        for file in video_dir.glob("*.mp4"):
            self.video_list.addItem(str(file.name))

    def play_selected_video(self):
        selected = self.video_list.currentItem()
        if selected:
            video_path = Path(LOCAL_VIDEOS_DIR) / selected.text()
            video_path = video_path.resolve()
            if not video_path.is_file():
                self.video_status_label.setText(f"Video file not found: {video_path}")
                self.video_filename_label.setText("No video loaded.")
                return

            url = QUrl.fromLocalFile(str(video_path))
            self.media_player.setMedia(QMediaContent(url))

            # Better video fitting with proper sizing
            def fit_video_to_view():
                if not self.video_item.nativeSize().isEmpty():
                    # Get the graphics view size
                    view_size = self.graphics_view.size()
                    print(
                        f"DEBUG: Graphics view size: {view_size.width()}x{view_size.height()}"
                    )

                    # Set the video item to a much larger size
                    target_width = min(
                        view_size.width() - 20, 800
                    )  # Leave some margin, max 800px
                    target_height = int(
                        target_width / (16 / 9)
                    )  # Maintain 16:9 aspect ratio

                    print(f"DEBUG: Target video size: {target_width}x{target_height}")

                    # Set the video item size directly
                    self.video_item.setSize(QSizeF(target_width, target_height))

                    # Update scene rect to match the video item
                    video_rect = self.video_item.boundingRect()
                    self.graphics_scene.setSceneRect(video_rect)

                    # Fit the view to show the entire video
                    self.graphics_view.fitInView(video_rect, Qt.KeepAspectRatio)

                    print(
                        f"DEBUG: Final video rect: {video_rect.width()}x{video_rect.height()}"
                    )
                else:
                    # Try again if video not ready
                    QTimer.singleShot(200, fit_video_to_view)

            # Use multiple timers to ensure video is properly fitted
            QTimer.singleShot(100, fit_video_to_view)
            QTimer.singleShot(500, fit_video_to_view)
            QTimer.singleShot(1000, fit_video_to_view)  # Extra attempt

            self.media_player.play()
            self.video_status_label.setText(f"Playing: {video_path.name}")
            self.video_filename_label.setText(f"Current video: {video_path.name}")
        else:
            self.video_status_label.setText("No video selected.")
            self.video_filename_label.setText("No video loaded.")

    def handle_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.video_status_label.setText("Playback finished.")
            self.play_pause_button.setText("Play")
        elif status == QMediaPlayer.LoadedMedia:
            self.video_status_label.setText("Video loaded - ready to play.")
        elif status == QMediaPlayer.InvalidMedia:
            self.video_status_label.setText("Invalid video file or unsupported format.")
        elif status == QMediaPlayer.NoMedia:
            self.video_status_label.setText("No media loaded.")
        elif status == QMediaPlayer.BufferingMedia:
            self.video_status_label.setText("Buffering video...")
        elif status == QMediaPlayer.StalledMedia:
            # Don't show stalled message immediately, it's often temporary
            if self.media_player.state() == QMediaPlayer.PlayingState:
                current_video = self.video_filename_label.text()
                if "Current video:" in current_video:
                    video_name = current_video.replace("Current video: ", "")
                    self.video_status_label.setText(f"Playing: {video_name}")
                else:
                    self.video_status_label.setText("Playing video...")
            else:
                self.video_status_label.setText("Video temporarily stalled.")
        elif status == QMediaPlayer.LoadingMedia:
            self.video_status_label.setText("Loading video...")
        # else: keep previous status

    def handle_media_error(self, error):
        if error != QMediaPlayer.NoError:
            self.video_status_label.setText(
                f"Playback error: {self.media_player.errorString()}"
            )

    def upload_video(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Videos (*.mp4 *.avi *.mov)")
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            for file_path in selected_files:
                dest_path = Path(LOCAL_VIDEOS_DIR) / Path(file_path).name
                os.makedirs(LOCAL_VIDEOS_DIR, exist_ok=True)
                with open(file_path, "rb") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
            self.load_video_list()

    def download_from_s3(self):
        """Show dialog to select and download video from S3 bucket."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout, QLabel, QProgressBar, QMessageBox
        from PyQt5.QtCore import QObject, pyqtSignal
        from videos.get_s3_videos import S3VideoDownloader
        import threading
        import traceback
        from pathlib import Path

        class S3DownloadWorker(QObject):
            progress = pyqtSignal(str, float, int, int)  # filename, percent, downloaded, total
            finished = pyqtSignal(bool, str)  # success, error message

            def __init__(self, downloader, video_key, local_path):
                super().__init__()
                self.downloader = downloader
                self.video_key = video_key
                self.local_path = local_path
                self.cancelled = False

            def start(self):
                def run():
                    try:
                        def callback(filename, percent, downloaded, total):
                            self.progress.emit(filename, percent, downloaded, total)
                            if self.cancelled:
                                raise Exception("Download cancelled")
                        self.downloader.download_video(self.video_key, self.local_path, callback)
                        if not self.cancelled:
                            self.finished.emit(True, "")
                    except Exception as e:
                        self.finished.emit(False, str(e))
                threading.Thread(target=run, daemon=True).start()

            def cancel(self):
                self.cancelled = True

        class S3VideoSelectDialog(QDialog):
            def __init__(self, parent=None, video_list=None):
                super().__init__(parent)
                self.setWindowTitle("Select S3 Video to Download")
                self.resize(500, 400)
                layout = QVBoxLayout()
                self.setLayout(layout)
                self.list_widget = QListWidget()
                if video_list:
                    self.list_widget.addItems(video_list)
                layout.addWidget(QLabel("Available S3 Videos:"))
                layout.addWidget(self.list_widget)
                button_layout = QHBoxLayout()
                self.ok_button = QPushButton("Download")
                self.cancel_button = QPushButton("Cancel")
                button_layout.addWidget(self.ok_button)
                button_layout.addWidget(self.cancel_button)
                layout.addLayout(button_layout)
                self.ok_button.clicked.connect(self.accept)
                self.cancel_button.clicked.connect(self.reject)
            def get_selected_video(self):
                item = self.list_widget.currentItem()
                return item.text() if item else None
        # Step 1: List S3 videos
        try:
            downloader = S3VideoDownloader()
            s3_videos = downloader.list_videos()
        except Exception as e:
            QMessageBox.warning(self, "S3 Error", f"Failed to list S3 videos:\n{e}")
            return
        # Step 2: Show selection dialog
        dialog = S3VideoSelectDialog(self, s3_videos)
        if dialog.exec_() != QDialog.Accepted:
            return
        selected_video = dialog.get_selected_video()
        if not selected_video:
            return
        # Step 3: Download with progress bar (thread-safe)
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle(f"Downloading {selected_video}")
        progress_dialog.resize(400, 150)
        vlayout = QVBoxLayout()
        progress_dialog.setLayout(vlayout)
        status_label = QLabel(f"Downloading: {selected_video}")
        vlayout.addWidget(status_label)
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        vlayout.addWidget(progress_bar)
        cancel_button = QPushButton("Cancel")
        vlayout.addWidget(cancel_button)
        # Worker setup
        local_path = str(Path(LOCAL_VIDEOS_DIR) / Path(selected_video).name)
        worker = S3DownloadWorker(downloader, selected_video, local_path)
        def on_progress(filename, percent, downloaded, total):
            progress_bar.setValue(int(percent))
            status_label.setText(f"{filename}: {percent:.1f}% ({downloaded//1024}KB/{total//1024}KB)")
        def on_finished(success, error):
            if success:
                progress_dialog.accept()
            else:
                QMessageBox.warning(self, "Download Error", f"Failed to download:\n{error}")
                progress_dialog.reject()
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        cancel_button.clicked.connect(lambda: (worker.cancel(), progress_dialog.reject()))
        worker.start()
        progress_dialog.exec_()
        # Step 4: Refresh video list
        self.load_video_list()

    def load_processed_video_list(self):
        self.processed_video_list.clear()
        output_dir = Path(OUTPUT_DIR)
        if output_dir.exists():
            for subdir in output_dir.iterdir():
                if subdir.is_dir() and (subdir / RESULTS_JSON).exists():
                    self.processed_video_list.addItem(str(subdir.name))

    def load_results_for_selected_video(self):
        selected = self.processed_video_list.currentItem()
        if not selected:
            return
        video_dir = Path(OUTPUT_DIR) / selected.text()
        results_path = video_dir / RESULTS_JSON
        if not results_path.exists():
            return
        with open(results_path, "r") as f:
            data = json.load(f)
        results = data.get("results", [])
        self.detections = results
        # Try to infer video size from first bbox
        if results and "bbox" in results[0]:
            max_x = max([b["bbox"][2] for b in results if "bbox" in b])
            max_y = max([b["bbox"][3] for b in results if "bbox" in b])
            self.video_native_size = (max_x, max_y)
        self.display_results(results, video_dir)

    def search_results(self):
        selected = self.processed_video_list.currentItem()
        if not selected:
            return
        video_dir = Path(OUTPUT_DIR) / selected.text()
        results_path = video_dir / RESULTS_JSON
        if not results_path.exists():
            return
        with open(results_path, "r") as f:
            data = json.load(f)
        results = data.get("results", [])
        vrn_query = self.vrn_search.text().strip().upper()
        raw_ocr_query = self.raw_ocr_search.text().strip().upper()  # Get raw OCR query
        ts_query = self.timestamp_search.text().strip()
        ts_start, ts_end = None, None
        if ts_query:
            try:
                ts_start, ts_end = map(float, ts_query.split(","))
            except Exception:
                pass
        filtered = []
        for entry in results:
            vrn = entry.get("vrn", "").upper()
            raw_ocr_text = entry.get("raw_ocr_text", "").upper()  # Get raw OCR text
            timestamp = entry.get("timestamp", 0.0)
            if vrn_query and vrn_query not in vrn:
                continue
            if raw_ocr_query and raw_ocr_query not in raw_ocr_text:
                continue  # Filter out based on raw OCR text
            if ts_start is not None and ts_end is not None:
                if not (ts_start <= timestamp <= ts_end):
                    continue
            filtered.append(entry)
        self.display_results(filtered, video_dir)

    def display_results(self, results, video_dir):
        # Store current results data and video directory for editing
        self.current_results_data = results.copy()
        self.current_video_dir = video_dir

        self.results_table.clear()
        self.results_table.setRowCount(len(results))
        self.results_table.setColumnCount(5)  # Increased from 4 to 5 columns
        self.results_table.setHorizontalHeaderLabels(
            [
                "VRN",
                "Raw OCR Text",
                "Timestamp",
                "Image",
                "Confidence",
            ]  # Added "Raw OCR Text" column
        )

        # Temporarily disconnect itemChanged to avoid triggering during setup
        try:
            self.results_table.itemChanged.disconnect()
        except:
            pass

        for i, entry in enumerate(results):
            vrn = entry.get("vrn", "")
            raw_ocr_text = entry.get("raw_ocr_text", "")  # Get raw OCR text
            timestamp = entry.get("timestamp", 0.0)
            image = entry.get("cropped_plate_path", "")
            detection_conf = entry.get("detection_confidence", 0.0)

            # Format timestamp using the proper time format and confidence to 2 decimal places
            timestamp_formatted = self.format_time(timestamp) if timestamp else "00:00"
            conf_formatted = f"{float(detection_conf):.2f}" if detection_conf else "0.00"

            # VRN column (editable)
            vrn_item = QTableWidgetItem(str(vrn))
            vrn_item.setFlags(vrn_item.flags() | Qt.ItemIsEditable)
            self.results_table.setItem(i, 0, vrn_item)

            # Raw OCR Text column (read-only)
            raw_ocr_item = QTableWidgetItem(str(raw_ocr_text))
            raw_ocr_item.setFlags(raw_ocr_item.flags() & ~Qt.ItemIsEditable)
            self.results_table.setItem(i, 1, raw_ocr_item)

            # Timestamp column (read-only)
            timestamp_item = QTableWidgetItem(timestamp_formatted)
            timestamp_item.setFlags(timestamp_item.flags() & ~Qt.ItemIsEditable)
            self.results_table.setItem(i, 2, timestamp_item)

            # Display image thumbnail in table (read-only)
            image_path = image
            if not Path(image_path).is_file():
                rel_path = video_dir / image_path
                if rel_path.is_file():
                    image_path = str(rel_path)
            pixmap = QPixmap(image_path) if Path(image_path).is_file() else QPixmap()
            if not pixmap.isNull():
                thumbnail = pixmap.scaled(100, 40, Qt.KeepAspectRatio)
                image_item = QTableWidgetItem()
                image_item.setData(Qt.DecorationRole, thumbnail)
                image_item.setFlags(image_item.flags() & ~Qt.ItemIsEditable)
                self.results_table.setItem(i, 3, image_item)  # Shifted to column 3
            else:
                image_item = QTableWidgetItem(image)
                image_item.setFlags(image_item.flags() & ~Qt.ItemIsEditable)
                self.results_table.setItem(i, 3, image_item)  # Shifted to column 3

            # Confidence column (read-only)
            conf_item = QTableWidgetItem(conf_formatted)
            conf_item.setFlags(conf_item.flags() & ~Qt.ItemIsEditable)
            self.results_table.setItem(i, 4, conf_item)  # Formatted confidence

        # Reconnect itemChanged signal after setup is complete
        self.results_table.itemChanged.connect(self.on_result_item_changed)

    def on_result_item_changed(self, item):
        """Handle when a table item is changed (only VRN column is editable)"""
        if not item or item.column() != 0:  # Only handle VRN column (column 0)
            return

        row = item.row()
        new_vrn = item.text().strip()

        # Update the in-memory data
        if row < len(self.current_results_data):
            old_vrn = self.current_results_data[row].get("vrn", "")
            self.current_results_data[row]["vrn"] = new_vrn

            # Also update the detections list used for video overlays
            if row < len(self.detections):
                self.detections[row]["vrn"] = new_vrn

            # Save the updated results to file
            self.save_updated_results()

            print(f"VRN updated for row {row}: '{old_vrn}' -> '{new_vrn}'")

    def save_updated_results(self):
        """Save the current results data back to the JSON file"""
        if not self.current_video_dir or not self.current_results_data:
            return

        results_path = self.current_video_dir / RESULTS_JSON

        try:
            # Read the original file to preserve other metadata
            with open(results_path, "r") as f:
                data = json.load(f)

            # Update the results section with our modified data
            data["results"] = self.current_results_data

            # Write back to file
            with open(results_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"Results saved to {results_path}")

        except Exception as e:
            print(f"Error saving results: {e}")
            # Show error dialog to user for better UX
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Save Error", f"Failed to save results: {e}")

    def on_result_cell_clicked(self, row, col):
        """Handle clicks on result table cells, seeking video for timestamp clicks."""
        # Only respond to timestamp column clicks (column 2)
        if col != 2:
            return

        timestamp_item = self.results_table.item(row, 2)
        if not timestamp_item:
            return

        # Store the clicked row index so we can show the exact detection
        self.selected_detection_row = row

        # Use the exact timestamp from the detection data instead of parsing the formatted time
        # This avoids rounding errors from the formatted display
        if row < len(self.current_results_data):
            exact_timestamp = self.current_results_data[row].get("timestamp", 0.0)
            print(f"Using exact timestamp {exact_timestamp:.3f}s from detection data for row {row}")
        else:
            # Fallback to parsing the formatted timestamp
            exact_timestamp = self.parse_time(timestamp_item.text())
            print(f"Fallback: parsed timestamp {exact_timestamp:.3f}s from formatted display")

        self._seek_to_timestamp(exact_timestamp)

    def _seek_to_timestamp(self, timestamp):
        """Seek video to a specific timestamp."""
        selected_video = self.processed_video_list.currentItem()
        if not selected_video:
            return

        video_name = selected_video.text()
        video_path = Path(LOCAL_VIDEOS_DIR) / (video_name + ".mp4")
        video_path = video_path.resolve()

        if not video_path.is_file():
            print(f"Video file not found: {video_path}")
            return

        url = QUrl.fromLocalFile(str(video_path))
        self._setup_media_for_seeking(url, timestamp)

    def _setup_media_for_seeking(self, url, timestamp):
        """Setup media player for seeking to a specific timestamp."""
        # Clear any existing connections to avoid conflicts
        self._disconnect_media_status_safely()

        # Reconnect the main media status handler
        self.media_player.mediaStatusChanged.connect(self.handle_media_status)

        # Set the media
        self.media_player.setMedia(QMediaContent(url))

        # Store timestamp for later use
        self.pending_timestamp = timestamp
        self.is_seeking_to_timestamp = True

        # Connect a one-time handler for when media is loaded
        self.media_player.mediaStatusChanged.connect(self._handle_media_loaded_for_seek)

        self.video_filename_label.setText(f"Current video: {url.fileName()}")

    def _disconnect_media_status_safely(self):
        """Safely disconnect media status change signals."""
        try:
            self.media_player.mediaStatusChanged.disconnect()
        except TypeError:
            # No connections exist, which is fine
            pass

    def _handle_media_loaded_for_seek(self, status):
        """Handle media loaded event when seeking to timestamp."""
        if (status == QMediaPlayer.LoadedMedia and
            hasattr(self, "is_seeking_to_timestamp") and
            self.is_seeking_to_timestamp):

            # Disconnect this handler
            self._disconnect_media_status_safely()

            # Reconnect the main handler
            self.media_player.mediaStatusChanged.connect(self.handle_media_status)

            # Execute the seek after a brief delay
            QTimer.singleShot(100, self.execute_timestamp_seek)


    def execute_timestamp_seek(self):
        """Execute the timestamp seeking with video fitting and pause/resume functionality"""
        if not self._validate_seeking_state():
            return

        timestamp = self.pending_timestamp
        self._cleanup_seeking_state()

        # Start video playback
        self._start_video_for_seeking(timestamp)

        # Setup the fitting and seeking process
        QTimer.singleShot(100, lambda: self._fit_and_seek(timestamp))

    def _validate_seeking_state(self):
        """Validate that we have the necessary state for seeking."""
        return (hasattr(self, "pending_timestamp") and
                hasattr(self, "is_seeking_to_timestamp") and
                self.is_seeking_to_timestamp)

    def _cleanup_seeking_state(self):
        """Clean up seeking state variables."""
        self.is_seeking_to_timestamp = False
        if hasattr(self, "pending_timestamp"):
            delattr(self, "pending_timestamp")

    def _start_video_for_seeking(self, timestamp):
        """Start video playback for seeking to timestamp."""
        print(f"Starting video playback for timestamp {timestamp:.3f}s")
        self.media_player.play()
        self.play_pause_button.setText("Pause")

    def _fit_and_seek(self, timestamp, retry_count=0):
        """Fit video to view and seek to timestamp with retry logic."""
        MAX_RETRIES = 15
        RETRY_DELAY_MS = 300

        if retry_count > MAX_RETRIES:
            print(f"Max retries ({MAX_RETRIES}) reached, giving up")
            return

        if not self._is_media_ready_for_seeking():
            print(f"Media not ready yet, retry {retry_count + 1}")
            QTimer.singleShot(RETRY_DELAY_MS,
                            lambda: self._fit_and_seek(timestamp, retry_count + 1))
            return

        if not self._is_video_size_available():
            print(f"Video native size empty, retrying... (attempt {retry_count + 1})")
            QTimer.singleShot(RETRY_DELAY_MS,
                            lambda: self._fit_and_seek(timestamp, retry_count + 1))
            return

        # Media is ready, proceed with fitting and seeking
        self._setup_video_display()
        self._execute_timestamp_seek_operation(timestamp)

    def _is_media_ready_for_seeking(self):
        """Check if media player is ready for seeking operations."""
        media_status = self.media_player.mediaStatus()
        player_state = self.media_player.state()

        ready_statuses = [QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia]
        ready_states = [QMediaPlayer.PlayingState, QMediaPlayer.PausedState]

        return media_status in ready_statuses and player_state in ready_states

    def _is_video_size_available(self):
        """Check if video native size is available."""
        native_size = self.video_item.nativeSize()
        return not native_size.isEmpty()

    def _setup_video_display(self):
        """Setup video display size and fitting."""
        native_size = self.video_item.nativeSize()
        print(f"Video is ready, native size: {native_size.width()}x{native_size.height()}")

        # Calculate target video size
        view_size = self.graphics_view.size()
        target_width = min(view_size.width() - 20, 800)  # Leave margin, max 800px
        target_height = int(target_width / (16 / 9))  # Maintain 16:9 aspect ratio

        # Configure video item
        self.video_item.setSize(QSizeF(target_width, target_height))

        # Update scene and view
        video_rect = self.video_item.boundingRect()
        self.graphics_scene.setSceneRect(video_rect)
        self.graphics_view.fitInView(video_rect, Qt.KeepAspectRatio)

    def _execute_timestamp_seek_operation(self, timestamp):
        """Execute the actual seek operation to timestamp."""
        # Calculate timing
        start_time = max(0.0, timestamp - 1.0)  # 1 second before target
        start_time_ms = int(start_time * 1000)

        print(f"Starting playback at {start_time:.3f}s, target timestamp {timestamp:.3f}s")

        # Seek and play
        self.media_player.setPosition(start_time_ms)
        self.media_player.play()
        self.play_pause_button.setText("Pause")

        # Setup monitoring for target timestamp
        self._setup_timestamp_monitoring(timestamp)

    def _setup_timestamp_monitoring(self, timestamp):
        """Setup monitoring to pause at target timestamp."""
        self.seeking_target_timestamp = timestamp
        self.seeking_paused = False

        # Create and start position monitor timer
        self.position_monitor_timer = QTimer()
        self.position_monitor_timer.setSingleShot(True)
        self.position_monitor_timer.timeout.connect(self._monitor_position)
        self.position_monitor_timer.start(200)  # Start monitoring after 200ms

    def _monitor_position(self):
        """Monitor video position and pause at target timestamp."""
        if not hasattr(self, "seeking_target_timestamp") or getattr(self, "seeking_paused", False):
            return

        current_position_ms = self.media_player.position()
        current_time = current_position_ms / 1000.0

        # Check if we've reached the target timestamp
        if current_time >= self.seeking_target_timestamp:
            self._handle_target_timestamp_reached()
        else:
            # Continue monitoring
            self.position_monitor_timer.start(50)  # Check every 50ms

    def _handle_target_timestamp_reached(self):
        """Handle when target timestamp is reached during seeking."""
        print(f"Reached target timestamp {self.seeking_target_timestamp:.3f}s, pausing")

        # Pause video
        self.media_player.pause()
        self.play_pause_button.setText("Play")
        self.seeking_paused = True

        # Show detection overlay after short delay to allow frame update
        QTimer.singleShot(150, lambda: self.show_detection_at_timestamp(self.seeking_target_timestamp, duration=5000))

        # Schedule resume after 3 seconds
        QTimer.singleShot(3000, self._resume_playback_after_seek)

    def _resume_playback_after_seek(self):
        """Resume playback after pausing at target timestamp."""
        print("Resuming video after 3 seconds")

        self.media_player.play()
        self.play_pause_button.setText("Pause")

        # Clean up seeking variables
        self._cleanup_seeking_variables()

    def _cleanup_seeking_variables(self):
        """Clean up all seeking-related instance variables."""
        variables_to_clean = [
            "seeking_target_timestamp",
            "seeking_paused",
            "position_monitor_timer"
        ]

        for var_name in variables_to_clean:
            if hasattr(self, var_name):
                attr = getattr(self, var_name)
                if hasattr(attr, 'stop'):  # For timers
                    attr.stop()
                delattr(self, var_name)

    def on_video_position_changed(self, position):
        """Handle video position changes to update UI and show automatic detections"""
        # Update the slider position (without triggering sliderMoved signal)
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)

        # Update timestamp labels
        current_time = position / 1000.0  # Convert to seconds
        total_time = self.media_player.duration() / 1000.0  # Convert to seconds
        self.current_time_label.setText(self.format_time(current_time))
        self.total_time_label.setText(self.format_time(total_time))

        # Don't show automatic detections if we're currently showing a selected timestamp detection
        if hasattr(self, "showing_selected_detection") and self.showing_selected_detection:
            return

        # Find detections within a wider time window to catch overlapping/close detections
        current_time = position / 1000.0
        boxes_to_show = []
        for det in self.detections:
            ts = det.get("timestamp", 0.0)
            # Increased time window from 0.1 to 0.3 seconds to catch more overlapping detections
            if abs(ts - current_time) < 0.3:
                boxes_to_show.append(
                    {
                        "bbox": det.get("bbox", [0, 0, 0, 0]),
                        "vrn": det.get("vrn", ""),
                        "color": QColor(255, 0, 0, 200),  # Red for automatic detection
                    }
                )

        if boxes_to_show:
            # Show VRN in the detection info label and bounding boxes
            vrns = [box["vrn"] for box in boxes_to_show if box["vrn"]]
            if vrns:
                self.detection_info_label.setText(f"Detected ({len(boxes_to_show)}): {', '.join(vrns)}")
                self.detection_info_label.show()
            # Update overlay with new boxes
            self.update_bounding_boxes(boxes_to_show)

            # Stop the overlay timer to prevent premature clearing
            if self.overlay_timer.isActive():
                self.overlay_timer.stop()

        else:
            # Only clear if no detections AND we're not in a "grace period" after showing detections
            # This prevents flickering when detections are close together
            if not hasattr(self, '_last_detection_time'):
                self._last_detection_time = 0

            # If we just showed detections recently, don't clear immediately
            time_since_last_detection = current_time - self._last_detection_time
            if time_since_last_detection > 1:  # 1 second grace period
                self.detection_info_label.hide()
                self.clear_bounding_boxes()

        # Update the last detection time when we have detections
        if boxes_to_show:
            self._last_detection_time = current_time

    def show_detection_at_timestamp(self, timestamp, duration=None):
        """Show bounding box overlay for detection at specific timestamp"""
        # Set flag to prevent automatic detections from interfering
        self.showing_selected_detection = True

        boxes_to_show = []

        # If we have a specific row selected, prioritize showing that detection
        if hasattr(self, 'selected_detection_row') and self.selected_detection_row is not None:
            # Show the specific detection from the clicked row
            if (self.selected_detection_row < len(self.current_results_data) and
                self.selected_detection_row < len(self.detections)):

                # Use the detection from current_results_data (which matches table display)
                selected_detection = self.current_results_data[self.selected_detection_row]
                det_timestamp = selected_detection.get("timestamp", 0.0)

                # Only show if timestamps are close (within 0.5 seconds for more tolerance)
                if abs(det_timestamp - timestamp) < 0.5:
                    boxes_to_show.append({
                        "bbox": selected_detection.get("bbox", [0, 0, 0, 0]),
                        "vrn": selected_detection.get("vrn", ""),
                        "color": QColor(0, 255, 0, 200),  # Green for selected detection
                    })
                    print(f"Showing specific detection from row {self.selected_detection_row}: VRN={selected_detection.get('vrn', '')}")
                else:
                    print(f"Selected detection timestamp {det_timestamp:.3f}s doesn't match target {timestamp:.3f}s")

            # Clear the selection after use
            self.selected_detection_row = None

        # If no specific selection found, fall back to timestamp matching
        if not boxes_to_show:
            print(f"No specific selection, searching for detections near timestamp {timestamp:.3f}s")
            for det in self.detections:
                ts = det.get("timestamp", 0.0)
                if abs(ts - timestamp) < 0.1:  # Within 0.1 seconds
                    boxes_to_show.append(
                        {
                            "bbox": det.get("bbox", [0, 0, 0, 0]),
                            "vrn": det.get("vrn", ""),
                            "color": QColor(0, 255, 0, 200),  # Green for selected detection
                        }
                    )

        if boxes_to_show:
            vrns = [box["vrn"] for box in boxes_to_show if box["vrn"]]
            self.detection_info_label.setText(f"Selected: {', '.join(vrns)}")
            self.detection_info_label.show()

            # Add a small delay before showing overlay to ensure video frame has updated
            QTimer.singleShot(250, lambda: self._show_overlay_with_delay(boxes_to_show, duration))
        else:
            # Keep the label visible but clear the text when no detection
            self.detection_info_label.setText("No detection at this timestamp")
            self.detection_info_label.show()
            self.clear_bounding_boxes()

    def _show_overlay_with_delay(self, boxes_to_show, duration):
        """Show the overlay after a small delay to ensure video frame is updated"""
        # Update overlay with new boxes
        self.update_bounding_boxes(boxes_to_show)
        # Use timer if duration is specified, otherwise keep visible
        if duration:
            self.overlay_timer.start(duration)

    def hide_overlay(self):
        self.detection_info_label.hide()
        self.clear_bounding_boxes()
        # Clear the flag so automatic detections can show again
        self.showing_selected_detection = False

    def toggle_play_pause(self):
        """Toggle play and pause for the video"""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_pause_button.setText("Play")
        else:
            self.media_player.play()
            self.play_pause_button.setText("Pause")

    def stop_video(self):
        """Stop the video and reset position"""
        self.media_player.stop()
        self.position_slider.setValue(0)
        self.play_pause_button.setText("Play")
        # Clear any visible overlays and detection info when stopping
        self.hide_overlay()
        self.clear_bounding_boxes()
        # Reset any seeking-related flags
        if hasattr(self, 'showing_selected_detection'):
            self.showing_selected_detection = False

    def set_position(self, position):
        """Set the video position based on slider movement"""
        print(f"DEBUG: Setting position to {position}ms")
        self.media_player.setPosition(position)

    def on_duration_changed(self, duration):
        """Update the slider range when the video duration changes"""
        self.position_slider.setRange(0, duration)

    def slider_pressed(self):
        """Handle slider press event"""
        print("DEBUG: Slider pressed")
        self.user_seeking = True

    def slider_released(self):
        """Handle slider release event to seek video"""
        print("DEBUG: Slider released")
        position = self.position_slider.value()
        print(f"DEBUG: Seeking to position {position}ms")
        self.media_player.setPosition(position)
        self.user_seeking = False

    def on_state_changed(self, state):
        """Handle media player state changes"""
        if state == QMediaPlayer.StoppedState:
            self.play_pause_button.setText("Play")
        elif state == QMediaPlayer.PlayingState:
            self.play_pause_button.setText("Pause")
        elif state == QMediaPlayer.PausedState:
            self.play_pause_button.setText("Play")

    def process_selected_video(self):
        """Process the selected video with license plate detection"""
        selected = self.video_list.currentItem()
        if not selected:
            self.video_status_label.setText("No video selected for processing.")
            return

        video_name = selected.text()
        video_path = Path(LOCAL_VIDEOS_DIR) / video_name

        if not video_path.is_file():
            self.video_status_label.setText(f"Video file not found: {video_path}")
            return

        # Show configuration dialog
        config_dialog = VideoProcessingConfigDialog(self)
        if config_dialog.exec_() != QDialog.Accepted:
            return

        config = config_dialog.get_config()

        # Create output directory
        video_stem = video_path.stem
        output_dir = Path(OUTPUT_DIR) / video_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Show progress dialog
        self.progress_dialog = VideoProcessingProgressDialog(self)
        self.progress_dialog.cancel_button.clicked.connect(self.cancel_processing)

        # Create and start worker thread
        self.processing_worker = VideoProcessingWorker(
            video_path=str(video_path), output_dir=str(output_dir), config=config
        )

        # Connect signals
        self.processing_worker.progress_updated.connect(
            self.progress_dialog.update_progress
        )
        self.processing_worker.error_occurred.connect(self.progress_dialog.show_error)
        self.processing_worker.processing_finished.connect(self.on_processing_finished)

        # Start processing
        self.processing_worker.start()
        self.progress_dialog.show()

        self.video_status_label.setText(f"Processing video: {video_name}")

    def cancel_processing(self):
        """Cancel the video processing"""
        try:
            if hasattr(self, "processing_worker") and self.processing_worker is not None:
                if self.processing_worker.isRunning():
                    self.processing_worker.terminate()
                    self.processing_worker.wait()
        except RuntimeError:
            # Worker object has been deleted by Qt, which is fine
            pass
        except AttributeError:
            # Worker object doesn't exist, which is also fine
            pass

        if hasattr(self, "progress_dialog"):
            self.progress_dialog.close()
        self.video_status_label.setText("Processing cancelled.")

    def on_processing_finished(self, output_dir, success):
        """Handle processing completion"""
        self.progress_dialog.processing_finished(success)

        if success:
            self.video_status_label.setText("Video processing completed successfully!")
            # Refresh the processed video list
            self.load_processed_video_list()
        else:
            self.video_status_label.setText("Video processing failed.")

        # Clean up worker
        if hasattr(self, "processing_worker"):
            self.processing_worker.deleteLater()

    def update_bounding_boxes(self, boxes, video_size=None):
        """Update the bounding boxes displayed on the video"""
        # Clear existing boxes
        self.clear_bounding_boxes()

        if not boxes or len(boxes) == 0:
            return

        # Get the video item's current size and position for scaling
        video_rect = self.video_item.boundingRect()
        if video_rect.width() == 0 or video_rect.height() == 0:
            # Video not loaded yet, try again later
            QTimer.singleShot(
                100, lambda: self.update_bounding_boxes(boxes, video_size)
            )
            return

        # Get the original video size
        native_size = self.video_item.nativeSize()
        if native_size.isEmpty():
            # Use the stored video size if available
            if hasattr(self, "video_native_size") and self.video_native_size:
                video_width, video_height = self.video_native_size
            else:
                # Fallback to default size
                video_width, video_height = 1920, 1080
        else:
            video_width = native_size.width()
            video_height = native_size.height()

        # Calculate the actual video display area accounting for letterboxing
        video_aspect = video_width / video_height
        item_aspect = video_rect.width() / video_rect.height()

        if video_aspect > item_aspect:
            # Video is wider than the container - letterboxing (black bars top/bottom)
            actual_video_width = video_rect.width()
            actual_video_height = video_rect.width() / video_aspect
            x_offset = 0
            y_offset = (video_rect.height() - actual_video_height) / 2
        else:
            # Video is taller than the container - pillarboxing (black bars left/right)
            actual_video_width = video_rect.height() * video_aspect
            actual_video_height = video_rect.height()
            x_offset = (video_rect.width() - actual_video_width) / 2
            y_offset = 0

        # Calculate scaling factors based on actual video display area
        scale_x = actual_video_width / video_width
        scale_y = actual_video_height / video_height

        # Get video item position offset
        video_pos = self.video_item.pos()

        # Update the scene with new boxes
        for i, box_info in enumerate(boxes):
            bbox = box_info.get("bbox", [0, 0, 0, 0])
            vrn = box_info.get("vrn", "")
            color = box_info.get("color", QColor(255, 0, 0, 255))

            # bbox format: [x1, y1, x2, y2] in original video coordinates
            x1, y1, x2, y2 = bbox

            # Scale coordinates and add proper offsets for letterboxing/pillarboxing
            scaled_x1 = x1 * scale_x + video_pos.x() + x_offset
            scaled_y1 = y1 * scale_y + video_pos.y() + y_offset
            scaled_x2 = x2 * scale_x + video_pos.x() + x_offset
            scaled_y2 = y2 * scale_y + video_pos.y() + y_offset


            scaled_width = scaled_x2 - scaled_x1
            scaled_height = scaled_y2 - scaled_y1

            # Add a new rectangle item for the bounding box
            rect_item = QGraphicsRectItem(
                QRectF(scaled_x1, scaled_y1, scaled_width, scaled_height)
            )
            pen = QPen(color)
            pen.setWidth(
                max(2, int(6 * min(scale_x, scale_y)))
            )  # Thicker lines: increased from 3 to 6
            rect_item.setPen(pen)
            self.graphics_scene.addItem(rect_item)
            self.bbox_items.append(rect_item)

            # Draw VRN text if available
            if vrn:
                text_item = QGraphicsSimpleTextItem(vrn)
                # Position text above the box, accounting for scaling and offset
                text_y = scaled_y1 - 40 * min(scale_x, scale_y)  # More space above box
                text_item.setPos(scaled_x1, text_y)
                text_item.setBrush(QColor(255, 255, 255))  # White text

                # Larger font size based on video scaling
                font = QFont()
                font.setPointSize(
                    max(12, int(24 * min(scale_x, scale_y)))
                )  # Larger text: increased from 16 to 24
                font.setBold(True)
                text_item.setFont(font)

                self.graphics_scene.addItem(text_item)
                self.bbox_items.append(text_item)

    def clear_bounding_boxes(self):
        """Clear all bounding boxes from the video"""
        for item in self.bbox_items:
            self.graphics_scene.removeItem(item)
        self.bbox_items.clear()

    def export_results_to_csv(self):
        """Export the current processed results to CSV using DataProcessor."""
        from PyQt5.QtWidgets import QMessageBox
        selected = self.processed_video_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Export Error", "No processed video selected.")
            return
        video_dir = Path(OUTPUT_DIR) / selected.text()
        results_path = video_dir / RESULTS_JSON
        if not results_path.exists():
            QMessageBox.warning(self, "Export Error", f"Results file not found: {results_path}")
            return
        # Show dialog for dedup_window and min_confidence
        dialog = ExportCSVConfigDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        config = dialog.get_config()
        try:
            processor = DataProcessor(
                dedup_window_seconds=config["dedup_window"],
                min_confidence=config["min_confidence"],
                output_dir=str(video_dir)
            )
            df = processor.process_video_results(str(results_path))
            csv_path = processor.save_to_csv()
            QMessageBox.information(self, "Export Successful", f"CSV exported to:\n{csv_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export CSV:\n{e}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
