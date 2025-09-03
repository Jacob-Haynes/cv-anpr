from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QTextEdit,
    QPushButton,
)
from PyQt5.QtCore import Qt


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
