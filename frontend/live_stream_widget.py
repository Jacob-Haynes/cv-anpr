from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap
from live_stream.live_video import VideoThread

class LiveStreamWidget(QWidget):
    def __init__(self, rtsp_url, width=640, height=480, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Stream")
        self.disply_width = width
        self.display_height = height

        # QLabel to display video frames
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.setStyleSheet("border: 2px solid black; background-color: #333;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Connecting to stream...")

        # Layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)

        # Video thread
        self.thread = VideoThread(rtsp_url)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(QPixmap)
    def update_image(self, p):
        scaled_pixmap = p.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
