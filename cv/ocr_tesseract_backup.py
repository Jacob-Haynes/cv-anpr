import cv2
import numpy as np
import pytesseract
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import os
import json

"""
THIS MODULE WAS REPLACED WITH EASYOCR IMPLEMENTATION
Teseract was attempted first but results were poor without fine-tuning the model so attempted easyOCR which had good success out of the box.
This is kept as a backup incase I have time to work on the fine-tuning later.
"""


class LicensePlateOCR:
    """Enhanced OCR system for extracting text from license plate images."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Multiple OCR configurations for different scenarios
        self.configs = {
            "license_plate": r"--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "single_line": r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "single_word": r"--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "auto": r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "digits": r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
            "letters": r"--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        }

    def enhance_image_quality(self, image: np.ndarray) -> List[np.ndarray]:
        """Create multiple enhanced versions of the image for better OCR."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        enhanced_images = []

        # 1. Aggressive upscaling for very small images
        height, width = gray.shape
        min_height = 80
        min_width = 300

        if height < min_height or width < min_width:
            scale_h = max(1, min_height / height)
            scale_w = max(1, min_width / width)
            scale = max(scale_h, scale_w, 4)  # At least 4x upscale

            # Use INTER_CUBIC for better quality on upscaling
            upscaled = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        else:
            upscaled = gray.copy()

        # 2. Denoising and sharpening
        denoised = cv2.fastNlMeansDenoising(upscaled)

        # Sharpening kernel
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

        # 3. Multiple threshold approaches
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        enhanced_images.append(adaptive)

        # OTSU threshold
        _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced_images.append(otsu)

        # Inverted OTSU (for dark text on light background)
        otsu_inv = cv2.bitwise_not(otsu)
        enhanced_images.append(otsu_inv)

        # 4. Morphological operations for cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # Clean up adaptive threshold
        adaptive_clean = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        adaptive_clean = cv2.morphologyEx(adaptive_clean, cv2.MORPH_OPEN, kernel)
        adaptive_clean = cv2.medianBlur(adaptive_clean, 3)
        enhanced_images.append(adaptive_clean)

        # 5. Edge-based enhancement
        edges = cv2.Canny(sharpened, 50, 150)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        edge_enhanced = cv2.bitwise_or(otsu, dilated_edges)
        enhanced_images.append(edge_enhanced)

        # 6. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(sharpened)
        _, contrast_thresh = cv2.threshold(
            contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        enhanced_images.append(contrast_thresh)

        return enhanced_images

    def clean_text(self, text: str) -> str:
        """Clean and format OCR text with better rules."""
        if not text:
            return ""

        # Remove all non-alphanumeric characters
        cleaned = re.sub(r"[^A-Z0-9]", "", text.strip().upper())

        # Common OCR corrections
        corrections = {
            "O": "0",
            "I": "1",
            "S": "5",
            "B": "8",
            "G": "6",
            "D": "0",
            "Z": "2",
            "T": "7",
            "L": "1",
        }

        # Apply corrections only if the result makes sense
        corrected = cleaned
        for old, new in corrections.items():
            if old in corrected and len(corrected) > 2:
                test_corrected = corrected.replace(old, new)
                # Only apply if it creates a reasonable license plate pattern
                if self.is_reasonable_plate_pattern(test_corrected):
                    corrected = test_corrected

        return corrected

    def is_reasonable_plate_pattern(self, text: str) -> bool:
        """Check if text follows reasonable license plate patterns."""
        if len(text) < 3 or len(text) > 8:
            return False

        # Check for reasonable mix of letters and numbers
        letters = sum(1 for c in text if c.isalpha())
        numbers = sum(1 for c in text if c.isdigit())

        # Must have at least one of each, or be all numbers/letters in reasonable ranges
        if letters == 0 and (numbers < 3 or numbers > 6):
            return False
        if numbers == 0 and (letters < 2 or letters > 4):
            return False

        return True

    def extract_text_multiple_methods(
        self, image: np.ndarray, debug: bool = False
    ) -> Dict:
        """Try multiple OCR methods and return the best result."""
        enhanced_images = self.enhance_image_quality(image)

        if debug:
            debug_dir = Path("debug_ocr")
            debug_dir.mkdir(exist_ok=True)
            for i, img in enumerate(enhanced_images):
                cv2.imwrite(str(debug_dir / f"enhanced_{i+1}.jpg"), img)

        best_result = {"text": "", "raw_text": "", "confidence": 0.0, "valid": False}
        all_results = []

        # Try each enhanced image with each configuration
        for img_idx, enhanced_img in enumerate(enhanced_images):
            for config_name, config in self.configs.items():
                try:
                    # Get detailed OCR data
                    data = pytesseract.image_to_data(
                        enhanced_img, config=config, output_type=pytesseract.Output.DICT
                    )
                    raw_text = pytesseract.image_to_string(enhanced_img, config=config)

                    # Calculate average confidence for text regions
                    confidences = [int(c) for c in data["conf"] if int(c) > 0]
                    avg_confidence = (
                        sum(confidences) / len(confidences) if confidences else 0
                    )

                    # Get text from high-confidence regions only
                    high_conf_text = []
                    for i, conf in enumerate(data["conf"]):
                        if int(conf) > 30 and data["text"][i].strip():
                            high_conf_text.append(data["text"][i].strip())

                    combined_text = "".join(high_conf_text)
                    cleaned_text = self.clean_text(
                        combined_text if combined_text else raw_text
                    )

                    result = {
                        "text": cleaned_text,
                        "raw_text": raw_text.strip(),
                        "confidence": round(avg_confidence, 2),
                        "valid": len(cleaned_text) >= 3
                        and self.is_reasonable_plate_pattern(cleaned_text),
                        "method": f"img_{img_idx+1}_{config_name}",
                        "high_conf_regions": len(high_conf_text),
                    }

                    all_results.append(result)

                    # Update best result based on multiple criteria
                    score = self.calculate_result_score(result)
                    best_score = self.calculate_result_score(best_result)

                    if score > best_score:
                        best_result = result

                except Exception as e:
                    continue

        # If no good result found, try a simplified approach
        if not best_result["valid"] and enhanced_images:
            try:
                # Use the first enhanced image with basic config
                simple_text = pytesseract.image_to_string(
                    enhanced_images[0], config=self.configs["license_plate"]
                )
                cleaned = self.clean_text(simple_text)
                if len(cleaned) >= 2:
                    best_result = {
                        "text": cleaned,
                        "raw_text": simple_text.strip(),
                        "confidence": 30.0,  # Default confidence
                        "valid": len(cleaned) >= 3,
                        "method": "fallback",
                    }
            except:
                pass

        # Add debugging info
        if debug:
            best_result["all_attempts"] = len(all_results)
            best_result["debug_dir"] = str(debug_dir)

        return best_result

    def calculate_result_score(self, result: Dict) -> float:
        """Calculate a score for OCR result quality."""
        if not result.get("text"):
            return 0.0

        score = 0.0

        # Base confidence score
        score += result.get("confidence", 0) * 0.4

        # Length bonus (ideal length is 5-7 characters)
        text_len = len(result["text"])
        if 5 <= text_len <= 7:
            score += 30
        elif 4 <= text_len <= 8:
            score += 20
        elif text_len >= 3:
            score += 10

        # Pattern validation bonus
        if result.get("valid", False):
            score += 25

        # High confidence regions bonus
        high_conf_regions = result.get("high_conf_regions", 0)
        score += min(high_conf_regions * 5, 20)

        # Reasonable character mix bonus
        if self.is_reasonable_plate_pattern(result["text"]):
            score += 15

        return score

    def extract_text(self, image_input, debug: bool = False) -> Dict:
        """Extract text from image file path or numpy array with enhanced methods."""
        # Handle both file paths and numpy arrays
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image: {image_input}")
        else:
            image = image_input

        return self.extract_text_multiple_methods(image, debug)

    def process_directory(
        self, directory: str, output_file: Optional[str] = None
    ) -> List[Dict]:
        """Process all images in a directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [f for f in dir_path.iterdir() if f.suffix.lower() in extensions]

        results = []
        for image_file in image_files:
            print(f"Processing {image_file.name}...")

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
            print(f"Results saved to: {output_file}")

        return results


# Convenience functions
def extract_plate_text(image_path: str, debug: bool = False) -> str:
    """Extract text from a license plate image."""
    return LicensePlateOCR().extract_text(image_path, debug)["text"]


def process_plate_directory(directory: str, output_file: str = None) -> List[Dict]:
    """Process all license plate images in a directory."""
    return LicensePlateOCR().process_directory(directory, output_file)


def main():
    """Command line interface."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ocr.py <image_path> [--debug]")
        print("  python ocr.py batch <directory> [output_file]")
        return

    try:
        ocr = LicensePlateOCR()

        if sys.argv[1] == "batch":
            if len(sys.argv) < 3:
                print("Error: Directory required for batch mode")
                return

            directory = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            results = ocr.process_directory(directory, output_file)

            # Summary
            valid = [r for r in results if r.get("valid", False)]
            print(f"\n--- Summary ---")
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

            print(f"\n--- OCR Results ---")
            print(f"File: {image_path}")
            print(f"Text: '{result['text']}'")
            print(f"Raw: '{result['raw_text']}'")
            print(f"Confidence: {result['confidence']}%")
            print(f"Valid: {result['valid']}")

            if "error" in result:
                print(f"Error: {result['error']}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
