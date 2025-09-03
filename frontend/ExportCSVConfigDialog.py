from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QDoubleSpinBox,
    QDialogButtonBox,
)


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
