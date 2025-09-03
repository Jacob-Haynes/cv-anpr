"""
S3 Video Downloader Dialog

A PyQt5-based dialog for downloading videos from S3 bucket with progress tracking.
"""

import threading
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QLabel,
    QProgressBar,
    QMessageBox,
)
from PyQt5.QtCore import QObject, pyqtSignal

from videos.get_s3_videos import S3VideoDownloader


class S3DownloadWorker(QObject):
    """Worker class for downloading S3 videos in a separate thread."""

    progress = pyqtSignal(str, float, int, int)  # filename, percent, downloaded, total
    finished = pyqtSignal(bool, str)  # success, error message

    def __init__(self, downloader, video_key, local_path):
        super().__init__()
        self.downloader = downloader
        self.video_key = video_key
        self.local_path = local_path
        self.cancelled = False

    def start(self):
        """Start the download in a separate thread."""
        def run():
            try:
                def callback(filename, percent, downloaded, total):
                    self.progress.emit(filename, percent, downloaded, total)
                    if self.cancelled:
                        raise Exception("Download cancelled")

                self.downloader.download_video(
                    self.video_key, self.local_path, callback
                )
                if not self.cancelled:
                    self.finished.emit(True, "")
            except Exception as e:
                self.finished.emit(False, str(e))

        threading.Thread(target=run, daemon=True).start()

    def cancel(self):
        """Cancel the download."""
        self.cancelled = True


class S3VideoSelectDialog(QDialog):
    """Dialog for selecting a video from S3 bucket."""

    def __init__(self, parent=None, video_list=None):
        super().__init__(parent)
        self.setWindowTitle("Select S3 Video to Download")
        self.resize(500, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Video list widget
        self.list_widget = QListWidget()
        if video_list:
            self.list_widget.addItems(video_list)

        layout.addWidget(QLabel("Available S3 Videos:"))
        layout.addWidget(self.list_widget)

        # Button layout
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Download")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Connect signals
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_selected_video(self):
        """Get the currently selected video."""
        item = self.list_widget.currentItem()
        return item.text() if item else None


class S3DownloadProgressDialog(QDialog):
    """Dialog showing download progress."""

    def __init__(self, parent, video_name):
        super().__init__(parent)
        self.setWindowTitle(f"Downloading {video_name}")
        self.resize(400, 150)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Status label
        self.status_label = QLabel(f"Downloading: {video_name}")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        layout.addWidget(self.cancel_button)

    def update_progress(self, filename, percent, downloaded, total):
        """Update the progress display."""
        self.progress_bar.setValue(int(percent))
        self.status_label.setText(
            f"{filename}: {percent:.1f}% ({downloaded//1024}KB/{total//1024}KB)"
        )


class S3VideoDownloaderDialog:
    """Main class for handling S3 video downloads with GUI."""

    def __init__(self, parent, local_videos_dir):
        self.parent = parent
        self.local_videos_dir = local_videos_dir
        self.worker = None
        self.progress_dialog = None

    def download_from_s3(self, on_complete_callback=None):
        """
        Show dialog to select and download video from S3 bucket.

        Args:
            on_complete_callback: Optional callback function to call when download completes
        """
        # Step 1: List S3 videos
        try:
            downloader = S3VideoDownloader()
            s3_videos = downloader.list_videos()
        except Exception as e:
            QMessageBox.warning(self.parent, "S3 Error", f"Failed to list S3 videos:\n{e}")
            return

        # Step 2: Show selection dialog
        dialog = S3VideoSelectDialog(self.parent, s3_videos)
        if dialog.exec_() != QDialog.Accepted:
            return

        selected_video = dialog.get_selected_video()
        if not selected_video:
            return

        # Step 3: Download with progress bar
        self._start_download(downloader, selected_video, on_complete_callback)

    def _start_download(self, downloader, selected_video, on_complete_callback):
        """Start the download process with progress tracking."""
        # Create progress dialog
        self.progress_dialog = S3DownloadProgressDialog(self.parent, selected_video)

        # Setup worker
        local_path = str(Path(self.local_videos_dir) / Path(selected_video).name)
        self.worker = S3DownloadWorker(downloader, selected_video, local_path)

        # Connect signals
        self.worker.progress.connect(self.progress_dialog.update_progress)
        self.worker.finished.connect(
            lambda success, error: self._on_download_finished(
                success, error, on_complete_callback
            )
        )
        self.progress_dialog.cancel_button.clicked.connect(self._cancel_download)

        # Start download and show progress
        self.worker.start()
        self.progress_dialog.exec_()

    def _on_download_finished(self, success, error, on_complete_callback):
        """Handle download completion."""
        if success:
            self.progress_dialog.accept()
            if on_complete_callback:
                on_complete_callback()
        else:
            QMessageBox.warning(
                self.parent, "Download Error", f"Failed to download:\n{error}"
            )
            self.progress_dialog.reject()

    def _cancel_download(self):
        """Cancel the current download."""
        if self.worker:
            self.worker.cancel()
        if self.progress_dialog:
            self.progress_dialog.reject()
