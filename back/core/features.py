"""Runtime feature switches shared by startup, HTTP and knowledge ingestion."""
import os


def image_features_enabled() -> bool:
    """Keep original behavior by default; text-only deployments explicitly opt out."""
    return os.getenv("IMAGE_FEATURES_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}


def local_ocr_enabled() -> bool:
    """API-only deployments retain image chat without loading local OCR."""
    return image_features_enabled() and os.getenv("LOCAL_OCR_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
