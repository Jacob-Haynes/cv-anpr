import cv2
import numpy as np
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import os
import json
import ssl
import urllib.request

# Fix SSL certificate issues for EasyOCR
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available. Install with: pip install easyocr")


class LicensePlateOCR:
    """Main OCR system using EasyOCR for license plate text extraction."""

    def __init__(
        self, languages: List[str] = ["en"], gpu: bool = False
    ):  # Default to CPU to avoid issues
        """Initialize EasyOCR reader."""
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR is not installed. Please install with: pip install easyocr"
            )

        try:
            print("Initializing EasyOCR (this may take a moment on first use)...")
            # Force CPU usage initially to avoid GPU issues
            self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
            self.gpu_enabled = gpu
            print(f"EasyOCR initialized successfully (GPU: {gpu})")
        except Exception as e:
            print(f"Warning: Failed to initialize EasyOCR with GPU={gpu}: {e}")
            try:
                print("Trying CPU fallback...")
                self.reader = easyocr.Reader(languages, gpu=False, verbose=False)
                self.gpu_enabled = False
                print("EasyOCR initialized successfully with CPU")
            except Exception as e2:
                raise RuntimeError(f"Failed to initialize EasyOCR: {e2}")

        # Character whitelist for license plates
        self.allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    def enhance_image_for_ocr(self, image: np.ndarray) -> List[np.ndarray]:
        """Create enhanced versions optimized for OCR."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        enhanced_images = []

        # 1. Aggressive upscaling - EasyOCR works better with larger images
        height, width = gray.shape
        min_height = 120
        min_width = 400

        if height < min_height or width < min_width:
            scale_h = max(1, min_height / height)
            scale_w = max(1, min_width / width)
            scale = max(
                scale_h, scale_w, 6
            )  # At least 6x upscale for very small images

            # Use INTER_CUBIC for better quality
            upscaled = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        else:
            upscaled = gray.copy()

        # 2. Multiple enhancement strategies

        # Original upscaled
        enhanced_images.append(upscaled)

        # Denoising + sharpening
        denoised = cv2.fastNlMeansDenoising(upscaled)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
        enhanced_images.append(sharpened)

        # OTSU threshold
        _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced_images.append(otsu)

        # Inverted OTSU (for dark plates with light text)
        otsu_inv = cv2.bitwise_not(otsu)
        enhanced_images.append(otsu_inv)

        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
        enhanced_images.append(adaptive)

        # Contrast enhancement with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(sharpened)
        enhanced_images.append(contrast_enhanced)

        # Convert to 3-channel for EasyOCR (it expects color images)
        enhanced_3ch = []
        for img in enhanced_images:
            if len(img.shape) == 2:
                img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_3ch = img
            enhanced_3ch.append(img_3ch)

        return enhanced_3ch

    def clean_ocr_text(self, text: str) -> str:
        """Clean and format OCR text output."""
        if not text:
            return ""

        # Remove all non-alphanumeric characters and convert to uppercase
        cleaned = re.sub(r"[^A-Z0-9]", "", text.strip().upper())

        return cleaned

    def extract_text_with_enhancements(
        self, image: np.ndarray, debug: bool = False
    ) -> Dict:
        """Extract text using OCR with multiple enhancement strategies."""
        enhanced_images = self.enhance_image_for_ocr(image)

        if debug:
            debug_dir = Path("debug_ocr")
            debug_dir.mkdir(exist_ok=True)
            for i, img in enumerate(enhanced_images):
                cv2.imwrite(str(debug_dir / f"enhanced_ocr_{i+1}.jpg"), img)

        best_result = {"text": "", "raw_text": "", "confidence": 0.0, "valid": False}
        all_results = []

        # Try OCR on each enhanced image
        for img_idx, enhanced_img in enumerate(enhanced_images):
            try:
                # EasyOCR returns [(bbox, text, confidence), ...]
                results = self.reader.readtext(
                    enhanced_img,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    width_ths=0.7,  # Threshold for text width
                    height_ths=0.7,  # Threshold for text height
                    paragraph=False,  # Don't group text into paragraphs
                    detail=1,  # Return detailed results with coordinates and confidence
                )

                if not results:
                    continue

                # Process all detected text regions
                combined_text = ""
                total_confidence = 0
                valid_detections = 0

                for bbox, text, confidence in results:
                    if (
                        confidence > 0.3
                    ):  # Only consider reasonably confident detections
                        cleaned_text = self.clean_ocr_text(text)
                        if cleaned_text:
                            combined_text += cleaned_text
                            total_confidence += (
                                confidence * 100
                            )  # EasyOCR returns 0-1, convert to 0-100
                            valid_detections += 1

                avg_confidence = (
                    total_confidence / valid_detections if valid_detections > 0 else 0
                )
                final_cleaned = self.clean_ocr_text(combined_text)

                # Create result - simplified validation (just check minimum length)
                raw_texts = [text for _, text, _ in results]
                result = {
                    "text": final_cleaned,
                    "raw_text": " ".join(raw_texts),
                    "confidence": round(avg_confidence, 2),
                    "valid": len(final_cleaned) >= 3,  # Simple length check only
                    "method": f"ocr_img_{img_idx+1}",
                    "detections_count": len(results),
                    "valid_detections": valid_detections,
                }

                all_results.append(result)

                # Score this result
                score = self.calculate_ocr_score(result)
                best_score = self.calculate_ocr_score(best_result)

                if score > best_score:
                    best_result = result

            except Exception as e:
                if debug:
                    print(f"OCR error on image {img_idx+1}: {e}")
                continue

        # Fallback: try with more permissive settings if no good result
        if not best_result["valid"] and enhanced_images:
            try:
                # Try with the first enhanced image and relaxed settings
                results = self.reader.readtext(
                    enhanced_images[0], width_ths=0.5, height_ths=0.5, paragraph=False
                )

                if results:
                    combined_text = "".join([text for _, text, _ in results])
                    cleaned = self.clean_ocr_text(combined_text)
                    if len(cleaned) >= 2:
                        avg_conf = (
                            sum([conf for _, _, conf in results]) / len(results) * 100
                        )
                        best_result = {
                            "text": cleaned,
                            "raw_text": combined_text,
                            "confidence": round(avg_conf, 2),
                            "valid": len(cleaned) >= 3,  # Simple length check only
                            "method": "ocr_fallback",
                        }
            except:
                pass

        # Add debugging info
        if debug:
            best_result["all_attempts"] = len(all_results)
            best_result["gpu_enabled"] = self.gpu_enabled

        return best_result

    def calculate_ocr_score(self, result: Dict) -> float:
        """Calculate quality score for OCR result."""
        if not result.get("text"):
            return 0.0

        score = 0.0

        # Base confidence score (higher weight for EasyOCR as it's generally more reliable)
        score += result.get("confidence", 0) * 0.5

        # Length bonus (license plates are typically 5-7 characters)
        text_len = len(result["text"])
        if 5 <= text_len <= 7:
            score += 35
        elif 4 <= text_len <= 8:
            score += 25
        elif text_len >= 3:
            score += 15

        # Validity bonus
        if result.get("valid", False):
            score += 30

        # Detection count bonus (more detections often means better segmentation)
        detections = result.get("detections_count", 0)
        score += min(detections * 3, 15)

        # Valid detections bonus
        valid_detections = result.get("valid_detections", 0)
        score += valid_detections * 5

        return score

    def extract_text(self, image_input, debug: bool = False) -> Dict:
        """Main extraction method - compatible with existing interface."""
        # Handle both file paths and numpy arrays
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image: {image_input}")
        else:
            image = image_input

        return self.extract_text_with_enhancements(image, debug)

    def process_directory(
        self, directory: str, output_file: Optional[str] = None
    ) -> List[Dict]:
        """Process all images in a directory using OCR."""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [f for f in dir_path.iterdir() if f.suffix.lower() in extensions]

        results = []
        for image_file in image_files:
            print(f"Processing {image_file.name} with OCR...")

            try:
                result = self.extract_text(str(image_file))
                result.update(
                    {"filename": image_file.name, "filepath": str(image_file)}
                )
                print(
                    f"  Text: '{result['text']}' (Confidence: {result['confidence']}%)"
                )
            except Exception as e:
                print(f"  Error: {e}")
                result = {
                    "filename": image_file.name,
                    "filepath": str(image_file),
                    "text": "",
                    "error": str(e),
                    "valid": False,
                }

            results.append(result)

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"OCR results saved to: {output_file}")

        return results


# Convenience functions
def extract_plate_text(image_path: str, debug: bool = False) -> str:
    """Extract text from a license plate image."""
    return LicensePlateOCR().extract_text(image_path, debug)["text"]


def process_plate_directory(directory: str, output_file: str = None) -> List[Dict]:
    """Process all license plate images in a directory."""
    return LicensePlateOCR().process_directory(directory, output_file)


def main():
    """Command line interface for EasyOCR."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ocr2.py <image_path> [--debug]")
        print("  python ocr2.py batch <directory> [output_file]")
        return

    try:
        ocr = LicensePlateOCR()
        print(f"EasyOCR initialized (GPU: {ocr.gpu_enabled})")

        if sys.argv[1] == "batch":
            if len(sys.argv) < 3:
                print("Error: Directory required for batch mode")
                return

            directory = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            results = ocr.process_directory(directory, output_file)

            # Summary
            valid = [r for r in results if r.get("valid", False)]
            print(f"\n--- EasyOCR Summary ---")
            print(
                f"Total: {len(results)}, Success: {len(valid)}, Failed: {len(results) - len(valid)}"
            )
            if valid:
                avg_conf = sum(r["confidence"] for r in valid) / len(valid)
                print(f"Average confidence: {avg_conf:.1f}%")

        else:
            # Single image
            image_path = sys.argv[1]
            debug = "--debug" in sys.argv
            result = ocr.extract_text(image_path, debug)

            print(f"\n--- EasyOCR Results ---")
            print(f"File: {image_path}")
            print(f"Text: '{result['text']}'")
            print(f"Raw: '{result['raw_text']}'")
            print(f"Confidence: {result['confidence']}%")
            print(f"Valid: {result['valid']}")
            print(f"Method: {result.get('method', 'N/A')}")

            if "error" in result:
                print(f"Error: {result['error']}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
