"""
Data Processing Module for License Plate Detection Results

This module provides comprehensive data processing capabilities for license plate detection
results, including de-duplication, validation, statistical analysis, and output formatting.
It handles both successful and failed OCR results for complete analysis coverage.

Classes:
    DataProcessor: Main class for processing and analyzing license plate detection data

Example:
    processor = DataProcessor(
        dedup_window_seconds=5.0,
        min_confidence=0.6,
        output_dir="results"
    )
    df = processor.process_video_results("results.json")
    summary = processor.get_summary()
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections import Counter


class DataProcessorConfig:
    """Configuration constants for the DataProcessor."""

    # Default values
    DEFAULT_DEDUP_WINDOW = 5.0
    DEFAULT_MIN_CONFIDENCE = 0.5
    DEFAULT_OUTPUT_DIR = "output"

    # VRN validation
    MIN_VRN_LENGTH = 3
    OCR_FAILED_PLACEHOLDER = "[OCR_FAILED]"

    # File naming
    CSV_FILENAME_PREFIX = "license_plate_detections_"
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    TIME_DISPLAY_FORMAT = "%H:%M:%S.%f"

    # DataFrame columns
    REQUIRED_COLUMNS = [
        "timestamp",
        "vrn",
        "frame_number",
        "detection_confidence",
        "ocr_confidence",
        "bbox",
        "cropped_plate_path",
        "recorded_at",
    ]


class DataProcessor:
    """
    Data Processing Pipeline for License Plate Detection Results.

    This class handles the processing, validation, de-duplication, and analysis of license
    plate detection results from video processing pipelines. It provides comprehensive
    data management with configurable parameters for different use cases.

    Attributes:
        dedup_window: Time window for de-duplication in seconds
        min_confidence: Minimum confidence threshold for valid detections
        output_dir: Directory for saving output files
        results: List of processed detection results
        recent_vrns: Dictionary tracking recent VRN detections for de-duplication
    """

    def __init__(
        self,
        dedup_window_seconds: float = DataProcessorConfig.DEFAULT_DEDUP_WINDOW,
        min_confidence: float = DataProcessorConfig.DEFAULT_MIN_CONFIDENCE,
        output_dir: str = DataProcessorConfig.DEFAULT_OUTPUT_DIR,
    ) -> None:
        """
        Initialize the data processor with configuration parameters.

        Args:
            dedup_window_seconds: Time window for de-duplication (seconds)
            min_confidence: Minimum confidence threshold for valid detections
            output_dir: Directory for saving output files

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_initialization_params(dedup_window_seconds, min_confidence)

        self.dedup_window = dedup_window_seconds
        self.min_confidence = min_confidence
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage
        self.results: List[Dict[str, Any]] = []
        self.recent_vrns: Dict[str, float] = {}  # {vrn: last_timestamp}

    def _validate_initialization_params(
        self, dedup_window: float, min_confidence: float
    ) -> None:
        """Validate initialization parameters."""
        if dedup_window < 0:
            raise ValueError("dedup_window_seconds must be >= 0")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

    def add_detection(
        self,
        frame_number: int,
        timestamp: float,
        vrn: str,
        detection_confidence: float,
        ocr_confidence: float,
        bbox: List[int],
        cropped_plate_path: Optional[str] = None,
    ) -> bool:
        """
        Add a detection result with comprehensive validation and de-duplication.

        Args:
            frame_number: Frame number where detection occurred
            timestamp: Timestamp in seconds
            vrn: Vehicle registration number (license plate text)
            detection_confidence: Confidence score for plate detection
            ocr_confidence: Confidence score for OCR text recognition
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            cropped_plate_path: Optional path to cropped plate image

        Returns:
            bool: True if detection was added successfully, False if rejected
        """
        # Validate VRN
        if not self._is_valid_vrn(vrn):
            return False

        # Validate confidence threshold
        if detection_confidence < self.min_confidence:
            return False

        # Check for duplicates (skip for failed OCR to capture all failures)
        if vrn != DataProcessorConfig.OCR_FAILED_PLACEHOLDER and self._is_duplicate(
            vrn, timestamp
        ):
            return False

        # Create structured result
        result = self._create_detection_result(
            frame_number,
            timestamp,
            vrn,
            detection_confidence,
            ocr_confidence,
            bbox,
            cropped_plate_path,
        )

        self.results.append(result)

        # Update de-duplication tracking (only for successful OCR)
        if vrn != DataProcessorConfig.OCR_FAILED_PLACEHOLDER:
            self.recent_vrns[vrn] = timestamp

        return True

    def _is_valid_vrn(self, vrn: str) -> bool:
        """Validate VRN format and content."""
        if not vrn:
            return False

        # Allow OCR_FAILED placeholder for analysis
        if vrn == DataProcessorConfig.OCR_FAILED_PLACEHOLDER:
            return True

        # Validate minimum length for actual VRNs
        return len(vrn) >= DataProcessorConfig.MIN_VRN_LENGTH

    def _is_duplicate(self, vrn: str, timestamp: float) -> bool:
        """
        Check if VRN was recently detected within the de-duplication window.

        Args:
            vrn: Vehicle registration number
            timestamp: Current detection timestamp

        Returns:
            bool: True if this is a duplicate detection
        """
        if vrn not in self.recent_vrns:
            return False

        last_seen = self.recent_vrns[vrn]
        time_diff = timestamp - last_seen
        return time_diff < self.dedup_window

    def _create_detection_result(
        self,
        frame_number: int,
        timestamp: float,
        vrn: str,
        detection_confidence: float,
        ocr_confidence: float,
        bbox: List[int],
        cropped_plate_path: Optional[str],
    ) -> Dict[str, Any]:
        """Create a structured detection result dictionary."""
        return {
            "timestamp": round(timestamp, 3),
            "vrn": vrn,
            "frame_number": int(frame_number),
            "detection_confidence": round(detection_confidence, 3),
            "ocr_confidence": round(ocr_confidence, 3),
            "bbox": [int(x) for x in bbox],
            "cropped_plate_path": cropped_plate_path,
            "recorded_at": datetime.now().isoformat(),
        }

    def process_video_results(self, results_json_path: str) -> pd.DataFrame:
        """
        Process results from video pipeline JSON output with comprehensive analysis.

        Args:
            results_json_path: Path to JSON results file from video pipeline

        Returns:
            DataFrame with processed and de-duplicated results

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If JSON format is invalid
        """
        self._validate_json_path(results_json_path)

        # Load and validate JSON data
        data = self._load_json_results(results_json_path)

        # Reset processor state
        self._reset_processor_state()

        # Process all detections
        self._process_detection_results(data.get("results", []))

        return self.to_dataframe()

    def _validate_json_path(self, json_path: str) -> None:
        """Validate JSON file path exists."""
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Results JSON file not found: {json_path}")

    def _load_json_results(self, json_path: str) -> Dict[str, Any]:
        """Load and validate JSON results file."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {json_path}: {e}")

    def _reset_processor_state(self) -> None:
        """Reset processor state for new data processing."""
        self.results = []
        self.recent_vrns = {}

    def _process_detection_results(
        self, detection_results: List[Dict[str, Any]]
    ) -> None:
        """Process list of detection results from video pipeline."""
        for result in detection_results:
            self._process_single_detection(result)

    def _process_single_detection(self, result: Dict[str, Any]) -> None:
        """Process a single detection result."""
        # Determine VRN based on OCR validity
        vrn = self._extract_vrn_from_result(result)

        # Add detection to processor
        self.add_detection(
            frame_number=result["frame_number"],
            timestamp=result["timestamp"],
            vrn=vrn,
            detection_confidence=result["detection_confidence"],
            ocr_confidence=result["ocr_confidence"],
            bbox=result["bbox"],
            cropped_plate_path=result.get("cropped_plate_path"),
        )

    def _extract_vrn_from_result(self, result: Dict[str, Any]) -> str:
        """Extract VRN from detection result, handling failed OCR cases."""
        # Check if ocr_valid field exists (for newer format)
        if "ocr_valid" in result:
            if result.get("ocr_valid", False):
                return result.get("vrn", "")
            else:
                return DataProcessorConfig.OCR_FAILED_PLACEHOLDER

        # Fallback for older format - determine validity from ocr_confidence and raw_ocr_text
        ocr_confidence = result.get("ocr_confidence", 0.0)
        raw_ocr_text = result.get("raw_ocr_text", "").strip()

        # Use the same confidence threshold as the video processor
        if ocr_confidence >= 50 and raw_ocr_text:
            # Clean and return the raw OCR text as VRN
            return self._clean_ocr_text(raw_ocr_text)
        else:
            return DataProcessorConfig.OCR_FAILED_PLACEHOLDER

    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR text to extract potential VRN."""
        if not text:
            return ""

        # Remove common OCR artifacts and clean the text
        cleaned = text.strip().upper()

        # Remove spaces and common OCR noise
        cleaned = ''.join(c for c in cleaned if c.isalnum())

        # Basic UK license plate pattern validation
        # UK plates are typically 7 characters: 2 letters, 2 numbers, 3 letters
        # But we'll be more flexible to handle variations
        if 3 <= len(cleaned) <= 8:  # Reasonable length for a license plate
            return cleaned

        return cleaned[:8] if len(cleaned) > 8 else cleaned  # Truncate if too long

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a comprehensive pandas DataFrame with additional analysis columns.

        Returns:
            DataFrame with processed results and computed columns
        """
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # Add computed columns for analysis
        df = self._add_computed_columns(df)

        # Sort by timestamp for chronological analysis
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def _add_computed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed columns for enhanced analysis."""
        # Add formatted time display
        df["time_formatted"] = (
            pd.to_datetime(df["timestamp"], unit="s")
            .dt.strftime(DataProcessorConfig.TIME_DISPLAY_FORMAT)
            .str[:-3]  # Remove last 3 digits for millisecond precision
        )

        # Add cleaned VRN for analysis
        df["vrn_clean"] = df["vrn"].str.replace(" ", "").str.upper()

        # Add success indicator
        df["ocr_successful"] = df["vrn"] != DataProcessorConfig.OCR_FAILED_PLACEHOLDER

        return df

    def save_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Save processed results to CSV file with automatic filename generation.

        Args:
            filename: Optional custom filename for CSV output

        Returns:
            str: Path to saved CSV file
        """
        if filename is None:
            filename = self._generate_csv_filename()

        csv_path = self.output_dir / filename
        df = self.to_dataframe()

        if df.empty:
            print("No valid detections to save.")
            return str(csv_path)

        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} detections to: {csv_path}")

        return str(csv_path)

    def _generate_csv_filename(self) -> str:
        """Generate timestamp-based filename for CSV output."""
        timestamp = datetime.now().strftime(DataProcessorConfig.TIMESTAMP_FORMAT)
        return f"{DataProcessorConfig.CSV_FILENAME_PREFIX}{timestamp}.csv"

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics of processed data.

        Returns:
            Dictionary containing detailed analysis summary
        """
        df = self.to_dataframe()

        if df.empty:
            return {"message": "No data to summarize"}

        return self._calculate_comprehensive_summary(df)

    def _calculate_comprehensive_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics."""
        # Basic statistics
        total_detections = len(df)
        successful_ocr = len(df[df["ocr_successful"]])
        failed_ocr = total_detections - successful_ocr

        # VRN analysis (excluding failed OCR)
        successful_df = df[df["ocr_successful"]]
        unique_vrns = (
            successful_df["vrn_clean"].nunique() if not successful_df.empty else 0
        )

        # Confidence statistics
        avg_detection_conf = df["detection_confidence"].mean()
        avg_ocr_conf = df["ocr_confidence"].mean()

        # Time analysis
        time_span = df["timestamp"].max() - df["timestamp"].min()

        # Most common VRNs (excluding failed OCR)
        if not successful_df.empty:
            vrn_counts = successful_df["vrn_clean"].value_counts().head(10)
            most_common_vrns = vrn_counts.to_dict()
        else:
            most_common_vrns = {}

        return {
            "total_detections": total_detections,
            "successful_ocr": successful_ocr,
            "failed_ocr": failed_ocr,
            "ocr_success_rate": (
                round((successful_ocr / total_detections) * 100, 1)
                if total_detections > 0
                else 0.0
            ),
            "unique_vrns": unique_vrns,
            "time_span_seconds": round(time_span, 2),
            "avg_detection_confidence": round(avg_detection_conf, 3),
            "avg_ocr_confidence": round(avg_ocr_conf, 3),
            "deduplication_window": self.dedup_window,
            "most_common_vrns": most_common_vrns,
        }
