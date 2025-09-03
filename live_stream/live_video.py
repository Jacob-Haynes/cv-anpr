import sys
import cv2
import os
import tempfile
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from dotenv import load_dotenv
from cv.plate_detection import LicensePlateDetector
from cv.ocr import LicensePlateOCR
import json

# Load environment variables from .env file
load_dotenv()

class VideoThread(QThread):
    """
    A QThread subclass for capturing video frames from an RTSP stream.
    """
    change_pixmap_signal = pyqtSignal(QPixmap)

    def __init__(self, rtsp_url):
        super().__init__()
        self._run_flag = True
        self.rtsp_url = rtsp_url
        # Use the correct absolute model path for license plate recognition
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cv", "models", "license-plate-recognition.pt")
        self.detector = LicensePlateDetector(model_path=model_path)
        self.ocr = LicensePlateOCR()
        self.last_detection_time = 0
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "output/live_stream/cropped_plates"
        )
        self.results_json_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "output/live_stream/results.json"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print(f"Error: Could not open video stream at {self.rtsp_url}")
            return
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.change_pixmap_signal.emit(pixmap)

                # Plate detection every second
                now = datetime.now().timestamp()
                if now - self.last_detection_time >= 1:
                    self.last_detection_time = now
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        temp_path = tmp.name
                        cv2.imwrite(temp_path, cv_img)
                    try:
                        detections = self.detector.detect_license_plates(temp_path, confidence=0.5)
                        for i, detection in enumerate(detections):
                            x1, y1, x2, y2 = detection["bbox"]
                            cropped = cv_img[y1:y2, x1:x2]
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                            filename = f"plate_{timestamp}_{i}.jpg"
                            save_path = os.path.join(self.output_dir, filename)
                            cv2.imwrite(save_path, cropped)
                            # Run OCR
                            ocr_result = self.ocr.extract_text_with_enhancements(cropped)
                            # Build result entry
                            result_entry = {
                                "timestamp": timestamp,
                                "bbox": [x1, y1, x2, y2],
                                "detection_confidence": detection["confidence"],
                                "cropped_plate_path": f"cropped_plates/{filename}",
                                "vrn": ocr_result.get("text", ""),
                                "raw_ocr_text": ocr_result.get("raw_text", ""),
                                "ocr_confidence": ocr_result.get("confidence", 0.0)
                            }
                            # Append to results.json
                            if os.path.exists(self.results_json_path):
                                with open(self.results_json_path, "r") as f:
                                    data = json.load(f)
                            else:
                                data = {"results": []}
                            data["results"].append(result_entry)
                            with open(self.results_json_path, "w") as f:
                                json.dump(data, f, indent=2)
                    finally:
                        os.remove(temp_path)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 RTSP Video Stream")
        self.disply_width = 640
        self.display_height = 480
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.disply_width, self.display_height)
        self.image_label.setStyleSheet("border: 2px solid black; background-color: #333;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Connecting to stream...")
        self.setFixedSize(self.disply_width, self.display_height)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        central_widget.setLayout(vbox)
        # Get RTSP URL from environment variable
        rtsp_url = os.getenv("RTSP_URL")
        if not rtsp_url:
            self.image_label.setText("Error: RTSP_URL not set in .env file.")
            return
        self.thread = VideoThread(rtsp_url)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def closeEvent(self, event):
        if hasattr(self, 'thread'):
            self.thread.stop()
        event.accept()

    @pyqtSlot(QPixmap)
    def update_image(self, pixmap):
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
