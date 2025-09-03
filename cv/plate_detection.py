import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
from typing import List, Tuple, Optional
import requests
import zipfile


class LicensePlateDetector:
    """
    License Plate Detection using YOLOv8

    This class provides functionality to detect license plates in images,
    draw bounding boxes around them, and crop the detected plates.
    """

    def __init__(
        self, model_path: Optional[str] = None, use_license_plate_model: bool = True
    ):
        """
        Initialize the license plate detector.

        Args:
            model_path: Path to a custom YOLO model. If None, uses pre-trained model.
            use_license_plate_model: If True, attempts to use a license plate specific model.
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.is_custom_model = True
        elif use_license_plate_model:
            # Try to use a license plate specific model
            try:
                # This will use license plate specific model if available
                self.model = YOLO(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "cv",
                        "models",
                        "license-plate-recognition.pt",
                    )
                )
                self.is_custom_model = True
            except:
                # Fall back to general object detection
                self.model = YOLO(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "cv",
                        "models",
                        "yolov8n.pt",
                    )
                )
                self.is_custom_model = False
        else:
            # download YOLOv8n model
            self.model = YOLO("yolov8n.pt")
            self.is_custom_model = False

        # Set up class filtering based on model type
        if self.is_custom_model:
            # For custom license plate models, we'll use all detections
            self.target_classes = None
        else:
            # For general COCO models, focus on vehicles which may contain plates
            self.target_classes = ["car", "truck", "bus", "motorcycle"]

    def _filter_detections_for_plates(
        self, detections: List[dict], image_shape: Tuple[int, int]
    ) -> List[dict]:
        """
        Filter detections to focus on likely license plate regions.

        Args:
            detections: Raw detections from YOLO
            image_shape: Shape of the input image (height, width)

        Returns:
            Filtered detections more likely to contain license plates
        """
        if self.is_custom_model:
            # Custom model should already be trained for license plates
            return detections

        filtered = []
        image_height, image_width = image_shape[:2]

        for detection in detections:
            if self.target_classes and detection["class"] not in self.target_classes:
                continue

            x1, y1, x2, y2 = detection["bbox"]
            width = x2 - x1
            height = y2 - y1

            # Filter based on aspect ratio and size
            aspect_ratio = width / height if height > 0 else 0
            area = width * height
            relative_area = area / (image_width * image_height)

            # License plates are typically rectangular with certain aspect ratios
            # and occupy a certain portion of the vehicle
            if (
                0.1 <= aspect_ratio <= 10.0  # Reasonable aspect ratio
                and relative_area >= 0.001  # Not too small
                and relative_area <= 0.5
            ):  # Not too large
                filtered.append(detection)

        return filtered

    def detect_license_plates(
        self, image_path: str, confidence: float = 0.5
    ) -> List[dict]:
        """
        Detect license plates in an image.

        Args:
            image_path: Path to the input image
            confidence: Minimum confidence threshold for detections

        Returns:
            List of dictionaries containing detection information
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image to get shape
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Run YOLO detection
        results = self.model(image_path, conf=confidence)

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # Get class name
                    class_name = self.model.names[cls]

                    detection = {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": float(conf),
                        "class": class_name,
                        "class_id": cls,
                    }
                    detections.append(detection)

        # Filter detections for license plates
        filtered_detections = self._filter_detections_for_plates(
            detections, image.shape
        )

        return filtered_detections

    def detect_and_visualize(
        self,
        image_path: str,
        output_path: str = None,
        confidence: float = 0.5,
        save_crops: bool = True,
    ) -> str:
        """
        Detect license plates and draw bounding boxes on the image.

        Args:
            image_path: Path to input image
            output_path: Path to save the annotated image
            confidence: Minimum confidence threshold
            save_crops: Whether to save cropped license plate regions

        Returns:
            Path to the saved annotated image
        """
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Get detections
        detections = self.detect_license_plates(image_path, confidence)

        # Create output directory if it doesn't exist
        if output_path is None:
            base_path = Path(image_path).parent
            output_path = base_path / f"detected_{Path(image_path).name}"

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Draw bounding boxes
        annotated_image = image.copy()
        crop_count = 0

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]
            class_name = detection["class"]

            # Use different colors for different confidence levels
            if conf >= 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif conf >= 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence

            thickness = 2
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)

            # Add label with confidence
            label = f"Plate Candidate: {conf:.2f}"
            if not self.is_custom_model:
                label = f"{class_name}: {conf:.2f}"

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )

            # Save cropped region if requested
            if save_crops:
                # Add some padding around the detected region
                padding = 10
                y1_crop = max(0, y1 - padding)
                y2_crop = min(image.shape[0], y2 + padding)
                x1_crop = max(0, x1 - padding)
                x2_crop = min(image.shape[1], x2 + padding)

                cropped = image[y1_crop:y2_crop, x1_crop:x2_crop]
                if cropped.size > 0:
                    crop_filename = f"license_plate_{crop_count}_conf_{conf:.2f}.jpg"
                    crop_path = output_dir / crop_filename
                    cv2.imwrite(str(crop_path), cropped)
                    crop_count += 1
                    print(f"Saved cropped license plate: {crop_path}")

        # Save annotated image
        cv2.imwrite(str(output_path), annotated_image)
        print(f"Saved annotated image: {output_path}")
        print(f"Found {len(detections)} potential license plate regions")

        return str(output_path)

    def crop_license_plates(
        self,
        image_path: str,
        output_dir: str = None,
        confidence: float = 0.5,
        add_padding: bool = True,
    ) -> List[str]:
        """
        Detect and crop license plate regions from an image.

        Args:
            image_path: Path to input image
            output_dir: Directory to save cropped images
            confidence: Minimum confidence threshold
            add_padding: Whether to add padding around cropped regions

        Returns:
            List of paths to saved cropped images
        """
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Get detections
        detections = self.detect_license_plates(image_path, confidence)

        # Create output directory
        if output_dir is None:
            output_dir = Path(image_path).parent / "cropped_license_plates"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Crop and save each detection
        saved_crops = []
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]
            class_name = detection["class"]

            # Add padding if requested
            if add_padding:
                padding = 10
                y1 = max(0, y1 - padding)
                y2 = min(image.shape[0], y2 + padding)
                x1 = max(0, x1 - padding)
                x2 = min(image.shape[1], x2 + padding)

            # Crop the region
            cropped = image[y1:y2, x1:x2]

            if cropped.size > 0:
                # Create filename with timestamp for uniqueness
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                crop_filename = f"license_plate_{timestamp}_{i}_conf_{conf:.2f}.jpg"
                crop_path = output_dir / crop_filename

                # Save cropped image
                cv2.imwrite(str(crop_path), cropped)
                saved_crops.append(str(crop_path))
                print(f"Saved cropped license plate: {crop_path}")

        return saved_crops


def detect_plates_in_image(
    image_path: str,
    output_path: str = None,
    confidence: float = 0.5,
    save_crops: bool = True,
    use_license_plate_model: bool = True,
) -> str:
    """
    Convenience function to detect license plates in a single image.

    Args:
        image_path: Path to the input image
        output_path: Path to save the annotated image
        confidence: Minimum confidence threshold for detections
        save_crops: Whether to save cropped license plate regions
        use_license_plate_model: Whether to try using a license plate specific model

    Returns:
        Path to the saved annotated image
    """
    detector = LicensePlateDetector(use_license_plate_model=use_license_plate_model)
    return detector.detect_and_visualize(
        image_path, output_path, confidence, save_crops
    )


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python plate_detection.py <image_path> [output_path] [confidence]"
        )
        print("Example: python plate_detection.py input.jpg output.jpg 0.5")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    try:
        result_path = detect_plates_in_image(image_path, output_path, confidence)
        print(f"Detection complete! Results saved to: {result_path}")
    except Exception as e:
        print(f"Error: {e}")
