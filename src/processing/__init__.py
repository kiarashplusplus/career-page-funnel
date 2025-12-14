"""
Processing pipeline for job data.

This module provides:
- Deduplication: Hash-based and TF-IDF similarity detection
- Classification: Experience level classification (entry, mid, senior, etc.)
- Normalization: Title, location, and salary normalization
"""

from .dedup import (
    DuplicateDetector,
    DuplicateMatch,
    DuplicateType,
    compute_content_hash,
    compute_description_hash,
    normalize_url,
)
from .classifier import (
    ExperienceLevel,
    JobType,
    RemoteType,
    JobClassifier,
    ClassificationResult,
)
from .normalizer import (
    JobNormalizer,
    NormalizedSalary,
    NormalizedLocation,
    SalaryPeriod,
)

__all__ = [
    # Deduplication
    "DuplicateDetector",
    "DuplicateMatch",
    "DuplicateType",
    "compute_content_hash",
    "compute_description_hash",
    "normalize_url",
    # Classification
    "ExperienceLevel",
    "JobType",
    "RemoteType",
    "JobClassifier",
    "ClassificationResult",
    # Normalization
    "JobNormalizer",
    "NormalizedSalary",
    "NormalizedLocation",
    "SalaryPeriod",
]
