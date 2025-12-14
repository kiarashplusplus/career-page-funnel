"""
Job deduplication using hash-based and TF-IDF similarity detection.

This module provides multiple deduplication strategies:
1. Hash-based: Fast exact matching using content hashes
2. URL-based: Matching by normalized URL
3. TF-IDF similarity: Catch near-duplicates with similar descriptions

Based on patterns from:
- ai-job-scraper: SHA256 content hashing
- JobFunnel: TF-IDF cosine similarity
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    np = None
    TfidfVectorizer = None
    cosine_similarity = None

logger = logging.getLogger(__name__)


class DuplicateType(str, Enum):
    """Type of duplicate match detected."""
    HASH = "hash"           # Exact content hash match
    URL = "url"             # Same URL (normalized)
    EXTERNAL_ID = "external_id"  # Same external ID from source
    TFIDF = "tfidf"         # High TF-IDF similarity


@dataclass
class DuplicateMatch:
    """Represents a detected duplicate."""
    original_id: int
    duplicate_id: Optional[int]  # None if not yet in DB
    match_type: DuplicateType
    similarity_score: float = 1.0  # 1.0 for exact matches
    
    def __repr__(self) -> str:
        return (
            f"DuplicateMatch(original={self.original_id}, "
            f"duplicate={self.duplicate_id}, type={self.match_type.value}, "
            f"similarity={self.similarity_score:.2f})"
        )


def compute_content_hash(title: str, company: str, location: Optional[str] = None) -> str:
    """
    Compute a SHA256 hash for deduplication based on normalized job metadata.
    
    Args:
        title: Job title
        company: Company name
        location: Job location (optional)
        
    Returns:
        64-character hex string hash
    """
    # Normalize inputs
    title_norm = _normalize_for_hash(title)
    company_norm = _normalize_for_hash(company)
    location_norm = _normalize_for_hash(location or "")
    
    content = f"{title_norm}|{company_norm}|{location_norm}"
    return hashlib.sha256(content.encode()).hexdigest()[:64]


def compute_description_hash(description: str) -> str:
    """
    Compute a hash of the job description for content-based deduplication.
    
    Args:
        description: Full job description text
        
    Returns:
        64-character hex string hash
    """
    # Normalize: lowercase, remove extra whitespace, strip HTML
    text = _strip_html(description or "")
    text = _normalize_for_hash(text)
    
    return hashlib.sha256(text.encode()).hexdigest()[:64]


def _normalize_for_hash(text: str) -> str:
    """Normalize text for consistent hashing."""
    if not text:
        return ""
    # Lowercase, strip, collapse whitespace
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Remove common punctuation that varies
    text = re.sub(r'[,.\-_/\\|]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return ""
    # Simple HTML tag removal
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    return text


def normalize_url(url: str) -> str:
    """
    Normalize a URL for comparison.
    
    Removes:
    - Tracking parameters (utm_*, ref, etc.)
    - Fragment identifiers
    - Trailing slashes
    - www. prefix
    """
    if not url:
        return ""
    
    try:
        parsed = urlparse(url.lower().strip())
        
        # Remove www. prefix
        netloc = parsed.netloc
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        
        # Remove tracking parameters
        tracking_params = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 
                          'utm_content', 'ref', 'source', 'fbclid', 'gclid'}
        if parsed.query:
            params = parsed.query.split('&')
            filtered_params = [p for p in params 
                             if not any(p.startswith(t + '=') for t in tracking_params)]
            query = '&'.join(filtered_params)
        else:
            query = ''
        
        # Remove trailing slash from path
        path = parsed.path.rstrip('/')
        
        # Reconstruct URL without fragment
        normalized = urlunparse((
            parsed.scheme,
            netloc,
            path,
            parsed.params,
            query,
            ''  # No fragment
        ))
        
        return normalized
    except Exception:
        return url.lower().strip()


@dataclass
class DuplicateDetector:
    """
    Detect duplicate jobs using multiple strategies.
    
    Strategies:
    1. Hash-based: Fast O(1) lookup using content hash
    2. URL-based: Match by normalized URL
    3. External ID: Match by source-specific ID
    4. TF-IDF: Similarity matching for near-duplicates (optional)
    
    Example:
        detector = DuplicateDetector()
        
        # Add existing jobs to the detector
        for job in existing_jobs:
            detector.add_job(
                job_id=job.id,
                content_hash=job.content_hash,
                url=job.url,
                external_id=job.external_id,
                description=job.description,
            )
        
        # Check if a new job is a duplicate
        match = detector.find_duplicate(
            content_hash=new_job.content_hash,
            url=new_job.url,
            external_id=new_job.external_id,
            description=new_job.description,
        )
        
        if match:
            print(f"Duplicate found: {match}")
    """
    
    # Configuration
    tfidf_threshold: float = 0.75  # Minimum similarity for TF-IDF match
    min_corpus_size: int = 25      # Minimum jobs before TF-IDF is useful
    enable_tfidf: bool = True      # Enable TF-IDF similarity matching
    
    # Internal state
    _hash_index: Dict[str, int] = field(default_factory=dict)
    _url_index: Dict[str, int] = field(default_factory=dict)
    _external_id_index: Dict[Tuple[int, str], int] = field(default_factory=dict)
    _descriptions: Dict[int, str] = field(default_factory=dict)
    _job_ids: List[int] = field(default_factory=list)
    _vectorizer: Optional[object] = field(default=None)
    _tfidf_matrix: Optional[object] = field(default=None)
    _tfidf_dirty: bool = field(default=True)
    
    def __post_init__(self):
        """Initialize TF-IDF vectorizer if sklearn is available."""
        if self.enable_tfidf and SKLEARN_AVAILABLE:
            self._vectorizer = TfidfVectorizer(
                strip_accents='unicode',
                lowercase=True,
                analyzer='word',
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2),
            )
        elif self.enable_tfidf and not SKLEARN_AVAILABLE:
            logger.warning(
                "TF-IDF deduplication disabled: scikit-learn not installed. "
                "Install with: pip install scikit-learn"
            )
            self.enable_tfidf = False
    
    def add_job(
        self,
        job_id: int,
        content_hash: str,
        url: Optional[str] = None,
        external_id: Optional[str] = None,
        source_id: Optional[int] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Add a job to the deduplication index.
        
        Args:
            job_id: Database ID of the job
            content_hash: Pre-computed content hash
            url: Job URL (will be normalized)
            external_id: External ID from source system
            source_id: Source ID (required if external_id provided)
            description: Job description for TF-IDF matching
        """
        # Index by content hash
        if content_hash:
            self._hash_index[content_hash] = job_id
        
        # Index by normalized URL
        if url:
            norm_url = normalize_url(url)
            self._url_index[norm_url] = job_id
        
        # Index by external ID (scoped to source)
        if external_id and source_id is not None:
            self._external_id_index[(source_id, external_id)] = job_id
        
        # Store description for TF-IDF
        if description and self.enable_tfidf:
            clean_desc = _strip_html(description)
            if len(clean_desc) > 50:  # Only index substantial descriptions
                self._descriptions[job_id] = clean_desc
                self._job_ids.append(job_id)
                self._tfidf_dirty = True
    
    def find_duplicate(
        self,
        content_hash: str,
        url: Optional[str] = None,
        external_id: Optional[str] = None,
        source_id: Optional[int] = None,
        description: Optional[str] = None,
    ) -> Optional[DuplicateMatch]:
        """
        Check if a job is a duplicate of an existing job.
        
        Checks in order of speed/accuracy:
        1. Content hash (fastest, exact match)
        2. External ID (fast, source-specific)
        3. URL (fast, may miss reposts)
        4. TF-IDF similarity (slowest, catches near-duplicates)
        
        Args:
            content_hash: Pre-computed content hash
            url: Job URL
            external_id: External ID from source system
            source_id: Source ID (required if external_id provided)
            description: Job description for TF-IDF matching
            
        Returns:
            DuplicateMatch if duplicate found, None otherwise
        """
        # Check content hash
        if content_hash and content_hash in self._hash_index:
            return DuplicateMatch(
                original_id=self._hash_index[content_hash],
                duplicate_id=None,
                match_type=DuplicateType.HASH,
                similarity_score=1.0,
            )
        
        # Check external ID (scoped to source)
        if external_id and source_id is not None:
            key = (source_id, external_id)
            if key in self._external_id_index:
                return DuplicateMatch(
                    original_id=self._external_id_index[key],
                    duplicate_id=None,
                    match_type=DuplicateType.EXTERNAL_ID,
                    similarity_score=1.0,
                )
        
        # Check URL
        if url:
            norm_url = normalize_url(url)
            if norm_url in self._url_index:
                return DuplicateMatch(
                    original_id=self._url_index[norm_url],
                    duplicate_id=None,
                    match_type=DuplicateType.URL,
                    similarity_score=1.0,
                )
        
        # Check TF-IDF similarity
        if description and self.enable_tfidf:
            match = self._check_tfidf_similarity(description)
            if match:
                return match
        
        return None
    
    def _check_tfidf_similarity(self, description: str) -> Optional[DuplicateMatch]:
        """Check if description is similar to existing jobs using TF-IDF."""
        if not SKLEARN_AVAILABLE or not self.enable_tfidf:
            return None
        
        # Need minimum corpus size for meaningful similarity
        if len(self._descriptions) < self.min_corpus_size:
            return None
        
        # Rebuild TF-IDF matrix if needed
        if self._tfidf_dirty:
            self._rebuild_tfidf_matrix()
        
        if self._tfidf_matrix is None:
            return None
        
        # Vectorize the new description
        clean_desc = _strip_html(description)
        if len(clean_desc) < 50:
            return None
        
        try:
            new_vector = self._vectorizer.transform([clean_desc])
            
            # Compute similarity with all existing descriptions
            similarities = cosine_similarity(new_vector, self._tfidf_matrix)[0]
            
            # Find the most similar job
            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]
            
            if max_similarity >= self.tfidf_threshold:
                original_id = self._job_ids[max_idx]
                return DuplicateMatch(
                    original_id=original_id,
                    duplicate_id=None,
                    match_type=DuplicateType.TFIDF,
                    similarity_score=float(max_similarity),
                )
        except Exception as e:
            logger.warning(f"TF-IDF similarity check failed: {e}")
        
        return None
    
    def _rebuild_tfidf_matrix(self) -> None:
        """Rebuild the TF-IDF matrix from stored descriptions."""
        if not self._descriptions or not SKLEARN_AVAILABLE:
            return
        
        try:
            # Get descriptions in order of job_ids
            descriptions = [self._descriptions[jid] for jid in self._job_ids 
                          if jid in self._descriptions]
            
            if len(descriptions) >= self.min_corpus_size:
                self._tfidf_matrix = self._vectorizer.fit_transform(descriptions)
                self._tfidf_dirty = False
                logger.debug(f"Rebuilt TF-IDF matrix with {len(descriptions)} documents")
        except Exception as e:
            logger.warning(f"Failed to rebuild TF-IDF matrix: {e}")
            self._tfidf_matrix = None
    
    def find_all_duplicates(
        self,
        jobs: List[dict],
        content_hash_key: str = "content_hash",
        url_key: str = "url",
        description_key: str = "description",
    ) -> Tuple[List[dict], List[DuplicateMatch]]:
        """
        Process a batch of jobs, separating unique from duplicates.
        
        Args:
            jobs: List of job dictionaries
            content_hash_key: Key for content hash in job dict
            url_key: Key for URL in job dict
            description_key: Key for description in job dict
            
        Returns:
            Tuple of (unique_jobs, duplicate_matches)
        """
        unique_jobs = []
        duplicates = []
        
        for job in jobs:
            match = self.find_duplicate(
                content_hash=job.get(content_hash_key, ""),
                url=job.get(url_key),
                description=job.get(description_key),
            )
            
            if match:
                duplicates.append(match)
            else:
                unique_jobs.append(job)
        
        logger.info(
            f"Processed {len(jobs)} jobs: {len(unique_jobs)} unique, "
            f"{len(duplicates)} duplicates"
        )
        
        return unique_jobs, duplicates
    
    @property
    def stats(self) -> dict:
        """Get statistics about the deduplication index."""
        return {
            "hash_index_size": len(self._hash_index),
            "url_index_size": len(self._url_index),
            "external_id_index_size": len(self._external_id_index),
            "tfidf_corpus_size": len(self._descriptions),
            "tfidf_enabled": self.enable_tfidf and SKLEARN_AVAILABLE,
            "tfidf_threshold": self.tfidf_threshold,
        }
    
    def clear(self) -> None:
        """Clear all indexes."""
        self._hash_index.clear()
        self._url_index.clear()
        self._external_id_index.clear()
        self._descriptions.clear()
        self._job_ids.clear()
        self._tfidf_matrix = None
        self._tfidf_dirty = True


# Convenience function for CLI/testing
if __name__ == "__main__":
    # Demo usage
    detector = DuplicateDetector(tfidf_threshold=0.75)
    
    # Add some sample jobs
    jobs = [
        {
            "id": 1,
            "title": "Software Engineer",
            "company": "Acme Inc",
            "location": "San Francisco, CA",
            "url": "https://acme.com/jobs/123",
            "description": "We are looking for a software engineer to join our team. "
                          "You will work on backend systems using Python and PostgreSQL.",
        },
        {
            "id": 2,
            "title": "Backend Developer",
            "company": "Acme Inc",
            "location": "San Francisco",
            "url": "https://acme.com/jobs/456",
            "description": "Join our engineering team as a backend developer. "
                          "Work with Python and PostgreSQL on exciting projects.",
        },
    ]
    
    for job in jobs:
        content_hash = compute_content_hash(job["title"], job["company"], job["location"])
        detector.add_job(
            job_id=job["id"],
            content_hash=content_hash,
            url=job["url"],
            description=job["description"],
        )
    
    # Check for duplicate
    new_job = {
        "title": "Software Engineer",
        "company": "Acme Inc.",  # Slight variation
        "location": "San Francisco, CA",
        "url": "https://acme.com/jobs/123?utm_source=linkedin",
        "description": "Looking for a software engineer for backend work with Python.",
    }
    
    content_hash = compute_content_hash(new_job["title"], new_job["company"], new_job["location"])
    match = detector.find_duplicate(
        content_hash=content_hash,
        url=new_job["url"],
        description=new_job["description"],
    )
    
    if match:
        print(f"✅ Duplicate detected: {match}")
    else:
        print("❌ No duplicate found")
    
    print(f"\n📊 Detector stats: {detector.stats}")
