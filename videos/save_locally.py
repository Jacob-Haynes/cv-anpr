#!/usr/bin/env python3
"""
Interactive script to download videos from S3 storage locally for testing.
Downloads videos to the local_videos subdirectory.
"""

import sys
from pathlib import Path

# Import S3 downloader from same directory
from get_s3_videos import S3VideoDownloader


def format_bytes(bytes_val):
    """Convert bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def show_progress(filename, percent, downloaded, total):
    """Display download progress bar."""
    bar_length = 40
    filled = int(bar_length * percent / 100)
    bar = "█" * filled + "░" * (bar_length - filled)

    downloaded_str = format_bytes(downloaded)
    total_str = format_bytes(total)

    print(
        f"\r{filename}: [{bar}] {percent:.1f}% ({downloaded_str}/{total_str})",
        end="",
        flush=True,
    )


def select_video():
    """List videos and get user selection."""
    print("Connecting to S3 storage...")
    downloader = S3VideoDownloader()

    print("Fetching available videos...")
    videos = downloader.list_videos()

    if not videos:
        print("No videos found.")
        return None, None

    # Show video list
    print(f"\nFound {len(videos)} videos:")
    print("-" * 50)
    for i, video in enumerate(videos, 1):
        print(f"{i:2d}. {video}")
    print("-" * 50)

    # Get user selection
    while True:
        selection = input("\nEnter video name (or 'q' to quit): ").strip()

        if selection.lower() == "q":
            return None, None

        if selection in videos:
            return selection, downloader

        # Try partial matching
        matches = [v for v in videos if selection.lower() in v.lower()]
        if len(matches) == 1:
            print(f"Found: {matches[0]}")
            return matches[0], downloader
        elif len(matches) > 1:
            print("Multiple matches:")
            for match in matches:
                print(f"  - {match}")
            print("Be more specific.")
        else:
            print(f"'{selection}' not found. Try again.")


def download_video(video_name, downloader):
    """Download selected video with progress."""
    local_dir = Path(__file__).parent / "local_videos"
    local_dir.mkdir(exist_ok=True)
    local_path = local_dir / video_name

    print(f"Downloading {video_name}...")

    try:
        downloaded_path = downloader.download_video(
            video_name, str(local_path), show_progress
        )
        print()  # New line after progress bar
        print(f"✓ Downloaded to: {downloaded_path}")

        # Show file size
        size_mb = Path(downloaded_path).stat().st_size / (1024 * 1024)
        print(f"✓ Size: {size_mb:.1f} MB")

        return downloaded_path

    except Exception as e:
        print(f"\nError: {e}")
        return None


def main():
    """Main function."""
    print("=== Video Download Tool ===")

    # Select video
    video_name, downloader = select_video()
    if not video_name:
        print("No video selected.")
        return

    # Check if exists
    local_path = Path(__file__).parent / "local_videos" / video_name
    if local_path.exists():
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"\n{video_name} already exists ({size_mb:.1f} MB)")
        if input("Download again? (y/N): ").lower() != "y":
            print("Skipped.")
            return

    # Download
    result = download_video(video_name, downloader)
    if result:
        print(f"\n✓ {video_name} ready for testing!")
    else:
        print(f"\n✗ Download failed")


if __name__ == "__main__":
    main()
