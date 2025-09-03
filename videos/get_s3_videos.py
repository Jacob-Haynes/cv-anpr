import boto3
import os
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv
from urllib.parse import urlparse
import logging
from botocore.config import Config
import warnings

# Suppress SSL warnings when verification is disabled
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class S3VideoDownloader:
    """Download videos from S3-compatible storage."""

    def __init__(self):
        """Initialize S3 client with credentials from environment variables."""
        self._load_config()
        self._init_s3_client()

    def _load_config(self):
        """Load and validate configuration from environment variables."""
        self.bucket_url = os.getenv("VIDEO_BUCKET")
        self.access_key = os.getenv("KEY")
        self.secret_key = os.getenv("SECRET_KEY")

        if not all([self.bucket_url, self.access_key, self.secret_key]):
            raise ValueError(
                "Missing required environment variables: VIDEO_BUCKET, KEY, SECRET_KEY"
            )

        # Parse bucket URL
        parsed_url = urlparse(self.bucket_url)
        self.endpoint_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.bucket_name = parsed_url.path.strip("/")

    def _init_s3_client(self):
        """Initialize S3 client with automatic SSL fallback."""
        config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"}, max_pool_connections=50
        )

        # Try with SSL verification first, then without
        for verify_ssl in [True, False]:
            try:
                self.s3_client = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name="auto",
                    verify=verify_ssl,
                    config=config,
                )

                # Test connection
                self.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)

                if not verify_ssl:
                    logger.warning("Connected with SSL verification disabled")

                logger.info(f"Connected to bucket: {self.bucket_name}")
                return

            except Exception as e:
                if verify_ssl and ("SSL" in str(e) or "certificate" in str(e).lower()):
                    continue
                raise

        raise ConnectionError("Failed to connect to S3 storage")

    def list_videos(self, prefix: str = "") -> List[str]:
        """List all video files in the bucket."""
        video_extensions = {".mp4"}

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix
            )

            videos = [
                obj["Key"]
                for obj in response.get("Contents", [])
                if any(obj["Key"].lower().endswith(ext) for ext in video_extensions)
            ]

            logger.info(f"Found {len(videos)} video files")
            return videos

        except Exception as e:
            logger.error(f"Error listing videos: {e}")
            raise

    def download_video(
        self, video_key: str, local_path: Optional[str] = None, progress_callback=None
    ) -> str:
        """Download a specific video from S3."""
        if local_path is None:
            local_path = Path(__file__).parent / Path(video_key).name
        else:
            local_path = Path(local_path)

        # Create directory if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Downloading {video_key}...")

            # Add progress tracking if callback provided
            if progress_callback:
                # Get file size first
                try:
                    response = self.s3_client.head_object(
                        Bucket=self.bucket_name, Key=video_key
                    )
                    file_size = response["ContentLength"]

                    class ProgressTracker:
                        def __init__(self, filename, size, callback):
                            self.filename = filename
                            self.size = size
                            self.callback = callback
                            self.downloaded = 0

                        def __call__(self, bytes_transferred):
                            self.downloaded += bytes_transferred
                            percent = (self.downloaded / self.size) * 100
                            self.callback(
                                self.filename, percent, self.downloaded, self.size
                            )

                    tracker = ProgressTracker(video_key, file_size, progress_callback)

                    self.s3_client.download_file(
                        Bucket=self.bucket_name,
                        Key=video_key,
                        Filename=str(local_path),
                        Callback=tracker,
                    )
                except Exception:
                    # Fallback without progress if getting file size fails
                    self.s3_client.download_file(
                        Bucket=self.bucket_name, Key=video_key, Filename=str(local_path)
                    )
            else:
                self.s3_client.download_file(
                    Bucket=self.bucket_name, Key=video_key, Filename=str(local_path)
                )

            print(f"Downloaded to: {local_path}")
            return str(local_path)

        except Exception as e:
            print(f"Download failed: {e}")
            raise

    def download_all_videos(
        self, local_directory: Optional[str] = None, overwrite: bool = False
    ) -> List[str]:
        """Download all videos from the bucket."""
        if local_directory is None:
            local_directory = Path(__file__).parent / "downloaded"
        else:
            local_directory = Path(local_directory)

        local_directory.mkdir(parents=True, exist_ok=True)

        videos = self.list_videos()
        downloaded = []

        for video_key in videos:
            local_path = local_directory / Path(video_key).name

            if local_path.exists() and not overwrite:
                logger.info(f"Skipping {video_key} - already exists")
                downloaded.append(str(local_path))
                continue

            try:
                path = self.download_video(video_key, str(local_path))
                downloaded.append(path)
            except Exception as e:
                logger.error(f"Failed to download {video_key}: {e}")

        logger.info(f"Downloaded {len(downloaded)} videos to {local_directory}")
        return downloaded

    def get_video_info(self, video_key: str) -> Dict:
        """Get metadata for a video file."""
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name, Key=video_key
            )

            return {
                "key": video_key,
                "size_mb": round(response["ContentLength"] / (1024 * 1024), 2),
                "last_modified": response["LastModified"],
                "content_type": response.get("ContentType", "unknown"),
            }

        except Exception as e:
            logger.error(f"Error getting info for {video_key}: {e}")
            raise


# Convenience functions
def download_video_from_s3(video_key: str, local_path: Optional[str] = None) -> str:
    """Download a single video from S3."""
    return S3VideoDownloader().download_video(video_key, local_path)


def list_available_videos() -> List[str]:
    """List all available videos in S3."""
    return S3VideoDownloader().list_videos()


def main():
    """Command line interface."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python get_s3_videos.py list")
        print("  python get_s3_videos.py download <video_key> [local_path]")
        print("  python get_s3_videos.py download_all [local_directory]")
        print("  python get_s3_videos.py info <video_key>")
        return

    try:
        downloader = S3VideoDownloader()
        command = sys.argv[1].lower()

        if command == "list":
            videos = downloader.list_videos()
            print(f"\nFound {len(videos)} videos:")
            for video in videos:
                print(f"  • {video}")

        elif command == "download" and len(sys.argv) >= 3:
            video_key = sys.argv[2]
            local_path = sys.argv[3] if len(sys.argv) > 3 else None
            path = downloader.download_video(video_key, local_path)
            print(f"✓ Downloaded to: {path}")

        elif command == "download_all":
            local_dir = sys.argv[2] if len(sys.argv) > 2 else None
            paths = downloader.download_all_videos(local_dir)
            print(f"✓ Downloaded {len(paths)} videos")

        elif command == "info" and len(sys.argv) >= 3:
            video_key = sys.argv[2]
            info = downloader.get_video_info(video_key)
            print(f"\nVideo: {info['key']}")
            print(f"Size: {info['size_mb']} MB")
            print(f"Modified: {info['last_modified']}")
            print(f"Type: {info['content_type']}")

        else:
            print("Invalid command or missing arguments")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
