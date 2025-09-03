from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QLabel,
    QDialogButtonBox,
)


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
