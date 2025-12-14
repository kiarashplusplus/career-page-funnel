"""
Job classification for experience level and job type.

This module classifies jobs based on:
1. Experience level: Entry, Mid, Senior, Lead, Executive
2. Job type: Full-time, Part-time, Contract, Internship
3. Remote type: Remote, Hybrid, On-site

Based on patterns from:
- job-scraper: is_entry_level() keyword and years-of-experience regex
- JobFunnel: keyword-based filtering
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ExperienceLevel(str, Enum):
    """Experience level classification."""
    INTERN = "intern"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    STAFF = "staff"
    PRINCIPAL = "principal"
    EXECUTIVE = "executive"
    UNKNOWN = "unknown"


class JobType(str, Enum):
    """Job type classification."""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"
    UNKNOWN = "unknown"


class RemoteType(str, Enum):
    """Remote work classification."""
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of job classification."""
    experience_level: ExperienceLevel
    job_type: JobType
    remote_type: RemoteType
    is_entry_level_friendly: bool
    years_required: Optional[int]  # Minimum years from description
    confidence: float  # 0-1 confidence score
    signals: List[str]  # Keywords/patterns that contributed to classification
    
    def __repr__(self) -> str:
        return (
            f"ClassificationResult("
            f"level={self.experience_level.value}, "
            f"type={self.job_type.value}, "
            f"remote={self.remote_type.value}, "
            f"entry_friendly={self.is_entry_level_friendly}, "
            f"years={self.years_required}, "
            f"confidence={self.confidence:.2f})"
        )


# Keyword sets for classification
ENTRY_LEVEL_KEYWORDS = {
    # Titles indicating entry level
    'junior', 'jr', 'jr.', 'entry', 'entry-level', 'associate', 'graduate',
    'new grad', 'new-grad', 'recent grad', 'college grad', 'university grad',
    'apprentice', 'trainee', 'early career', 'early-career',
    # Entry-level signals
    'no experience required', 'no experience needed', 'will train',
    '0-1 years', '0-2 years', '1-2 years', 'up to 2 years',
    'entry level', 'level i', 'level 1', 'l1', 'i ', 'ii '
}

MID_LEVEL_KEYWORDS = {
    # Mid-level titles
    'mid', 'mid-level', 'intermediate', 'level ii', 'level 2', 'level iii',
    'level 3', 'l2', 'l3', 'ii', 'iii',
    # Experience indicators
    '2-5 years', '3-5 years', '2-4 years', '3-6 years', '4-6 years',
    '2+ years', '3+ years', '4+ years',
}

SENIOR_KEYWORDS = {
    # Senior titles
    'senior', 'sr', 'sr.', 'experienced', 'level iv', 'level 4', 'l4',
    'iv', 'advanced',
    # Experience indicators
    '5+ years', '5-7 years', '5-8 years', '6+ years', '7+ years',
    '5-10 years', '8+ years',
}

LEAD_KEYWORDS = {
    # Lead/management titles
    'lead', 'tech lead', 'team lead', 'manager', 'engineering manager',
    'development manager', 'level v', 'level 5', 'l5',
    # Experience indicators
    '8+ years', '10+ years', '7-10 years', '8-12 years',
}

STAFF_KEYWORDS = {
    # Staff/Principal titles
    'staff', 'staff engineer', 'level vi', 'level 6', 'l6',
    '10+ years', '12+ years',
}

PRINCIPAL_KEYWORDS = {
    # Principal titles
    'principal', 'principal engineer', 'level vii', 'level 7', 'l7',
    'distinguished', 'fellow',
}

EXECUTIVE_KEYWORDS = {
    # Executive titles
    'director', 'vp', 'vice president', 'cto', 'cio', 'chief',
    'head of engineering', 'head of', 'executive',
    '15+ years', '20+ years',
}

INTERN_KEYWORDS = {
    'intern', 'internship', 'co-op', 'coop', 'summer', 'fall intern',
    'spring intern', 'winter intern',
}

# Job type keywords
FULL_TIME_KEYWORDS = {'full-time', 'full time', 'fulltime', 'permanent', 'fte'}
PART_TIME_KEYWORDS = {'part-time', 'part time', 'parttime', 'half-time'}
CONTRACT_KEYWORDS = {'contract', 'contractor', 'consulting', 'freelance', 'c2c', 'c2h', 'corp-to-corp'}
INTERNSHIP_KEYWORDS = {'internship', 'intern', 'co-op', 'coop'}
TEMPORARY_KEYWORDS = {'temporary', 'temp', 'seasonal'}

# Remote keywords
REMOTE_KEYWORDS = {'remote', 'work from home', 'wfh', 'telecommute', 'distributed', 'anywhere'}
HYBRID_KEYWORDS = {'hybrid', 'flexible', 'partial remote', 'some remote'}
ONSITE_KEYWORDS = {'onsite', 'on-site', 'in-office', 'in office', 'office-based', 'on premises'}

# Years of experience regex patterns
# Match patterns like "5+ years", "3-5 years", "minimum 5 years", etc.
YEARS_PATTERNS = [
    r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
    r'(\d+)\s*-\s*\d+\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
    r'minimum\s+(?:of\s+)?(\d+)\s*(?:years?|yrs?)',
    r'at\s+least\s+(\d+)\s*(?:years?|yrs?)',
    r'(\d+)\s*(?:years?|yrs?)\s+(?:minimum|min)',
    r'experience:\s*(\d+)\+?\s*(?:years?|yrs?)',
]


class JobClassifier:
    """
    Classify jobs by experience level, job type, and remote status.
    
    Uses keyword matching and years-of-experience extraction to determine
    appropriate classification for each job posting.
    
    Example:
        classifier = JobClassifier()
        
        result = classifier.classify(
            title="Senior Software Engineer",
            description="5+ years of experience required...",
            location="Remote, USA",
        )
        
        print(result.experience_level)  # ExperienceLevel.SENIOR
        print(result.is_entry_level_friendly)  # False
    """
    
    def __init__(
        self,
        max_entry_level_years: int = 2,
        strict_entry_level: bool = False,
    ):
        """
        Initialize the classifier.
        
        Args:
            max_entry_level_years: Maximum years of experience for entry-level
            strict_entry_level: If True, require explicit entry-level signals
        """
        self.max_entry_level_years = max_entry_level_years
        self.strict_entry_level = strict_entry_level
        
        # Compile years patterns
        self._years_patterns = [re.compile(p, re.IGNORECASE) for p in YEARS_PATTERNS]
    
    def classify(
        self,
        title: str,
        description: Optional[str] = None,
        location: Optional[str] = None,
        job_type_hint: Optional[str] = None,
    ) -> ClassificationResult:
        """
        Classify a job posting.
        
        Args:
            title: Job title
            description: Full job description (optional but recommended)
            location: Job location (optional)
            job_type_hint: Explicit job type from ATS (optional)
            
        Returns:
            ClassificationResult with all classifications
        """
        signals: List[str] = []
        
        # Combine text for analysis
        title_lower = title.lower() if title else ""
        desc_lower = description.lower() if description else ""
        loc_lower = location.lower() if location else ""
        
        # Full text for keyword matching
        full_text = f"{title_lower} {desc_lower}"
        
        # Extract years of experience
        years_required = self._extract_years(full_text)
        if years_required:
            signals.append(f"years_required={years_required}")
        
        # Classify experience level
        experience_level, exp_signals = self._classify_experience(
            title_lower, desc_lower, years_required
        )
        signals.extend(exp_signals)
        
        # Classify job type
        job_type, type_signals = self._classify_job_type(
            title_lower, desc_lower, job_type_hint
        )
        signals.extend(type_signals)
        
        # Classify remote status
        remote_type, remote_signals = self._classify_remote(
            title_lower, desc_lower, loc_lower
        )
        signals.extend(remote_signals)
        
        # Determine if entry-level friendly
        is_entry_level_friendly = self._is_entry_level_friendly(
            experience_level, years_required, signals
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(signals, description is not None)
        
        return ClassificationResult(
            experience_level=experience_level,
            job_type=job_type,
            remote_type=remote_type,
            is_entry_level_friendly=is_entry_level_friendly,
            years_required=years_required,
            confidence=confidence,
            signals=signals,
        )
    
    def _extract_years(self, text: str) -> Optional[int]:
        """Extract minimum years of experience from text."""
        years_found: List[int] = []
        
        for pattern in self._years_patterns:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    years = int(match)
                    if 0 <= years <= 30:  # Sanity check
                        years_found.append(years)
                except (ValueError, TypeError):
                    continue
        
        # Return minimum years found (most restrictive for filtering)
        return min(years_found) if years_found else None
    
    def _classify_experience(
        self,
        title: str,
        description: str,
        years_required: Optional[int],
    ) -> Tuple[ExperienceLevel, List[str]]:
        """Classify experience level based on keywords and years."""
        signals: List[str] = []
        full_text = f"{title} {description}"
        
        # Check title first (stronger signal)
        title_level = self._check_keywords_for_level(title, signals, prefix="title")
        if title_level != ExperienceLevel.UNKNOWN:
            return title_level, signals
        
        # Check description keywords
        desc_level = self._check_keywords_for_level(full_text, signals, prefix="desc")
        if desc_level != ExperienceLevel.UNKNOWN:
            return desc_level, signals
        
        # Fall back to years-based classification
        if years_required is not None:
            if years_required <= self.max_entry_level_years:
                signals.append(f"years_based=entry({years_required})")
                return ExperienceLevel.ENTRY, signals
            elif years_required <= 4:
                signals.append(f"years_based=mid({years_required})")
                return ExperienceLevel.MID, signals
            elif years_required <= 7:
                signals.append(f"years_based=senior({years_required})")
                return ExperienceLevel.SENIOR, signals
            elif years_required <= 10:
                signals.append(f"years_based=lead({years_required})")
                return ExperienceLevel.LEAD, signals
            else:
                signals.append(f"years_based=staff+({years_required})")
                return ExperienceLevel.STAFF, signals
        
        return ExperienceLevel.UNKNOWN, signals
    
    def _check_keywords_for_level(
        self,
        text: str,
        signals: List[str],
        prefix: str = "",
    ) -> ExperienceLevel:
        """Check text for experience level keywords."""
        # Order matters: check more specific levels first
        
        # Intern (most specific)
        for kw in INTERN_KEYWORDS:
            if kw in text:
                signals.append(f"{prefix}_keyword={kw}")
                return ExperienceLevel.INTERN
        
        # Executive
        for kw in EXECUTIVE_KEYWORDS:
            if kw in text:
                signals.append(f"{prefix}_keyword={kw}")
                return ExperienceLevel.EXECUTIVE
        
        # Principal
        for kw in PRINCIPAL_KEYWORDS:
            if kw in text:
                signals.append(f"{prefix}_keyword={kw}")
                return ExperienceLevel.PRINCIPAL
        
        # Staff
        for kw in STAFF_KEYWORDS:
            if kw in text:
                signals.append(f"{prefix}_keyword={kw}")
                return ExperienceLevel.STAFF
        
        # Lead
        for kw in LEAD_KEYWORDS:
            if kw in text:
                signals.append(f"{prefix}_keyword={kw}")
                return ExperienceLevel.LEAD
        
        # Senior
        for kw in SENIOR_KEYWORDS:
            if kw in text:
                signals.append(f"{prefix}_keyword={kw}")
                return ExperienceLevel.SENIOR
        
        # Mid-level
        for kw in MID_LEVEL_KEYWORDS:
            if kw in text:
                signals.append(f"{prefix}_keyword={kw}")
                return ExperienceLevel.MID
        
        # Entry-level
        for kw in ENTRY_LEVEL_KEYWORDS:
            if kw in text:
                signals.append(f"{prefix}_keyword={kw}")
                return ExperienceLevel.ENTRY
        
        return ExperienceLevel.UNKNOWN
    
    def _classify_job_type(
        self,
        title: str,
        description: str,
        job_type_hint: Optional[str],
    ) -> Tuple[JobType, List[str]]:
        """Classify job type."""
        signals: List[str] = []
        full_text = f"{title} {description}"
        
        # Use hint if provided
        if job_type_hint:
            hint_lower = job_type_hint.lower()
            if any(kw in hint_lower for kw in INTERNSHIP_KEYWORDS):
                signals.append(f"hint={job_type_hint}")
                return JobType.INTERNSHIP, signals
            if any(kw in hint_lower for kw in CONTRACT_KEYWORDS):
                signals.append(f"hint={job_type_hint}")
                return JobType.CONTRACT, signals
            if any(kw in hint_lower for kw in PART_TIME_KEYWORDS):
                signals.append(f"hint={job_type_hint}")
                return JobType.PART_TIME, signals
            if any(kw in hint_lower for kw in FULL_TIME_KEYWORDS):
                signals.append(f"hint={job_type_hint}")
                return JobType.FULL_TIME, signals
        
        # Check keywords (order: internship, contract, part-time, full-time)
        for kw in INTERNSHIP_KEYWORDS:
            if kw in full_text:
                signals.append(f"keyword={kw}")
                return JobType.INTERNSHIP, signals
        
        for kw in CONTRACT_KEYWORDS:
            if kw in full_text:
                signals.append(f"keyword={kw}")
                return JobType.CONTRACT, signals
        
        for kw in TEMPORARY_KEYWORDS:
            if kw in full_text:
                signals.append(f"keyword={kw}")
                return JobType.TEMPORARY, signals
        
        for kw in PART_TIME_KEYWORDS:
            if kw in full_text:
                signals.append(f"keyword={kw}")
                return JobType.PART_TIME, signals
        
        for kw in FULL_TIME_KEYWORDS:
            if kw in full_text:
                signals.append(f"keyword={kw}")
                return JobType.FULL_TIME, signals
        
        # Default to unknown (don't assume full-time)
        return JobType.UNKNOWN, signals
    
    def _classify_remote(
        self,
        title: str,
        description: str,
        location: str,
    ) -> Tuple[RemoteType, List[str]]:
        """Classify remote work status."""
        signals: List[str] = []
        full_text = f"{title} {description} {location}"
        
        # Check for hybrid (often specified explicitly)
        for kw in HYBRID_KEYWORDS:
            if kw in full_text:
                signals.append(f"remote_keyword={kw}")
                return RemoteType.HYBRID, signals
        
        # Check for remote
        for kw in REMOTE_KEYWORDS:
            if kw in full_text:
                signals.append(f"remote_keyword={kw}")
                return RemoteType.REMOTE, signals
        
        # Check for on-site
        for kw in ONSITE_KEYWORDS:
            if kw in full_text:
                signals.append(f"remote_keyword={kw}")
                return RemoteType.ONSITE, signals
        
        return RemoteType.UNKNOWN, signals
    
    def _is_entry_level_friendly(
        self,
        level: ExperienceLevel,
        years_required: Optional[int],
        signals: List[str],
    ) -> bool:
        """Determine if job is entry-level friendly."""
        # Definitely entry-level
        if level in (ExperienceLevel.INTERN, ExperienceLevel.ENTRY):
            return True
        
        # Definitely not entry-level
        if level in (ExperienceLevel.SENIOR, ExperienceLevel.LEAD, 
                     ExperienceLevel.STAFF, ExperienceLevel.PRINCIPAL,
                     ExperienceLevel.EXECUTIVE):
            return False
        
        # For mid-level or unknown, check years
        if years_required is not None:
            if years_required <= self.max_entry_level_years:
                return True
            return False
        
        # Unknown level, no years specified
        if self.strict_entry_level:
            return False  # Require explicit entry-level signals
        
        # Permissive: unknown might be entry-level
        return level == ExperienceLevel.UNKNOWN
    
    def _calculate_confidence(
        self,
        signals: List[str],
        has_description: bool,
    ) -> float:
        """Calculate confidence score for classification."""
        base_confidence = 0.5 if has_description else 0.3
        
        # More signals = more confidence
        signal_boost = min(len(signals) * 0.1, 0.4)
        
        # Title-based signals are stronger
        title_signals = sum(1 for s in signals if 'title' in s)
        title_boost = min(title_signals * 0.1, 0.2)
        
        return min(base_confidence + signal_boost + title_boost, 1.0)
    
    def is_entry_level(
        self,
        title: str,
        description: Optional[str] = None,
    ) -> bool:
        """
        Quick check if a job is entry-level friendly.
        
        This is a convenience method for filtering job lists.
        
        Args:
            title: Job title
            description: Job description (optional)
            
        Returns:
            True if the job appears to be entry-level friendly
        """
        result = self.classify(title, description)
        return result.is_entry_level_friendly
    
    def filter_entry_level(
        self,
        jobs: List[dict],
        title_key: str = "title",
        description_key: str = "description",
    ) -> List[dict]:
        """
        Filter a list of jobs to only entry-level friendly ones.
        
        Args:
            jobs: List of job dictionaries
            title_key: Key for title in job dict
            description_key: Key for description in job dict
            
        Returns:
            List of entry-level friendly jobs
        """
        entry_level_jobs = []
        
        for job in jobs:
            title = job.get(title_key, "")
            description = job.get(description_key)
            
            if self.is_entry_level(title, description):
                entry_level_jobs.append(job)
        
        logger.info(
            f"Filtered {len(jobs)} jobs to {len(entry_level_jobs)} entry-level friendly"
        )
        
        return entry_level_jobs


# Convenience function for CLI/testing
if __name__ == "__main__":
    classifier = JobClassifier(max_entry_level_years=2)
    
    test_cases = [
        {
            "title": "Junior Software Engineer",
            "description": "Great opportunity for new graduates. No experience required.",
        },
        {
            "title": "Software Engineer",
            "description": "Looking for engineers with 5+ years of experience in Python.",
        },
        {
            "title": "Senior Backend Developer",
            "description": "We need a senior engineer with 7-10 years of experience.",
        },
        {
            "title": "Staff Engineer",
            "description": "Lead technical initiatives across multiple teams. 10+ years required.",
        },
        {
            "title": "Software Engineering Intern",
            "description": "Summer 2024 internship opportunity.",
        },
        {
            "title": "Full Stack Developer",
            "description": "Remote position. Work from anywhere. Hybrid option available.",
            "location": "Remote, USA",
        },
    ]
    
    print("🏷️ Job Classification Demo\n")
    
    for i, job in enumerate(test_cases, 1):
        result = classifier.classify(
            title=job["title"],
            description=job.get("description"),
            location=job.get("location"),
        )
        
        entry_emoji = "✅" if result.is_entry_level_friendly else "❌"
        
        print(f"{i}. {job['title']}")
        print(f"   Level: {result.experience_level.value}")
        print(f"   Type: {result.job_type.value}")
        print(f"   Remote: {result.remote_type.value}")
        print(f"   Entry-Level Friendly: {entry_emoji}")
        print(f"   Years Required: {result.years_required}")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   Signals: {result.signals}")
        print()
