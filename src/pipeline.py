"""
Integrated job processing pipeline.

This module orchestrates the entire flow:
1. Scraping from ATS sources
2. Deduplication against existing jobs
3. Classification (experience level, job type)
4. Normalization (titles, locations, salaries)
5. Storage in database

Usage:
    from src.pipeline import JobPipeline
    
    pipeline = JobPipeline(db_session)
    result = await pipeline.run_scraper("greenhouse", "stripe")
    print(f"Processed {result.total} jobs, {result.new_jobs} new")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Type

from .database import JobRepository, SourceRepository
from .models.job import Job, JobCreate
from .models.source import ComplianceStatus, Source, SourceCreate
from .processing import (
    DuplicateDetector,
    JobClassifier,
    JobNormalizer,
    compute_content_hash,
)
from .scrapers import BaseScraper, ScraperResult
from .scrapers.greenhouse import GreenhouseScraper
from .scrapers.lever import LeverScraper
from .scrapers.amazon_jobs import AmazonJobsScraper

logger = logging.getLogger(__name__)


# Registry of available scrapers with their class and URL patterns
SCRAPER_REGISTRY: Dict[str, dict] = {
    "greenhouse": {
        "class": GreenhouseScraper,
        "url_template": "https://boards-api.greenhouse.io/v1/boards/{company}",
    },
    "lever": {
        "class": LeverScraper,
        "url_template": "https://api.lever.co/v0/postings/{company}",
    },
    "amazon_jobs": {
        "class": AmazonJobsScraper,
        "url_template": "https://www.amazon.jobs/en/search",
        "is_direct": True,  # No company param needed
    },
}


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    
    source_name: str
    company: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Counts
    total_scraped: int = 0
    duplicates_skipped: int = 0
    new_jobs: int = 0
    updated_jobs: int = 0
    errors: List[str] = field(default_factory=list)
    
    # Classification stats
    entry_level_count: int = 0
    mid_level_count: int = 0
    senior_count: int = 0
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    def complete(self) -> "PipelineStats":
        self.completed_at = datetime.utcnow()
        return self
    
    def to_dict(self) -> dict:
        return {
            "source": self.source_name,
            "company": self.company,
            "duration_seconds": self.duration_seconds,
            "total_scraped": self.total_scraped,
            "duplicates_skipped": self.duplicates_skipped,
            "new_jobs": self.new_jobs,
            "updated_jobs": self.updated_jobs,
            "entry_level_count": self.entry_level_count,
            "mid_level_count": self.mid_level_count,
            "senior_count": self.senior_count,
            "errors": self.errors,
            "success": self.success,
        }


@dataclass 
class BatchStats:
    """Statistics from processing multiple sources."""
    
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    source_stats: List[PipelineStats] = field(default_factory=list)
    
    @property
    def total_scraped(self) -> int:
        return sum(s.total_scraped for s in self.source_stats)
    
    @property
    def total_new(self) -> int:
        return sum(s.new_jobs for s in self.source_stats)
    
    @property
    def total_updated(self) -> int:
        return sum(s.updated_jobs for s in self.source_stats)
    
    @property
    def total_errors(self) -> int:
        return sum(len(s.errors) for s in self.source_stats)
    
    @property
    def success_rate(self) -> float:
        if not self.source_stats:
            return 0.0
        successful = sum(1 for s in self.source_stats if s.success)
        return successful / len(self.source_stats)
    
    def complete(self) -> "BatchStats":
        self.completed_at = datetime.utcnow()
        return self


class JobPipeline:
    """
    Orchestrates the complete job processing pipeline.
    
    Handles:
    - Source registration and compliance checking
    - Scraper instantiation and execution
    - Job deduplication
    - Classification and normalization
    - Database storage
    
    Example:
        from sqlalchemy.orm import Session
        from src.pipeline import JobPipeline
        
        with Session(engine) as db:
            pipeline = JobPipeline(db)
            
            # Process a single company
            stats = await pipeline.run_scraper("greenhouse", "stripe")
            
            # Process multiple companies
            batch = await pipeline.run_batch([
                ("greenhouse", "stripe"),
                ("greenhouse", "airbnb"),
                ("lever", "spotify"),
            ])
    """
    
    def __init__(
        self,
        db,
        enable_tfidf: bool = True,
        tfidf_threshold: float = 0.75,
        normalize_titles: bool = True,
        classify_jobs: bool = True,
    ):
        """
        Initialize the pipeline.
        
        Args:
            db: SQLAlchemy session
            enable_tfidf: Enable TF-IDF similarity detection
            tfidf_threshold: Similarity threshold for TF-IDF duplicates
            normalize_titles: Normalize job titles
            classify_jobs: Classify experience levels
        """
        self.db = db
        self.job_repo = JobRepository(db)
        self.source_repo = SourceRepository(db)
        
        # Processing components
        self.dedup = DuplicateDetector(
            enable_tfidf=enable_tfidf,
            tfidf_threshold=tfidf_threshold,
        )
        self.classifier = JobClassifier()
        self.normalizer = JobNormalizer() if normalize_titles else None
        self.classify_jobs = classify_jobs
        
        # Load existing jobs into dedup index
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Load existing jobs into deduplication index."""
        if self._initialized:
            return
        
        logger.info("Initializing deduplication index from database...")
        
        # Load all active jobs
        jobs = self.job_repo.search(is_active=True, limit=100000)
        
        for job in jobs:
            self.dedup.add_job(
                job_id=job.id,
                content_hash=job.content_hash,
                url=job.url,
                external_id=job.external_id,
                source_id=job.source_id,
                description=job.description,
            )
        
        logger.info(f"Loaded {len(jobs)} jobs into deduplication index")
        self._initialized = True
    
    def get_or_create_source(
        self,
        scraper_type: str,
        company: str,
        base_url: Optional[str] = None,
    ) -> Source:
        """
        Get or create a source record for a company/scraper combination.
        
        Args:
            scraper_type: Type of scraper (greenhouse, lever, etc.)
            company: Company identifier
            base_url: Optional base URL override
            
        Returns:
            Source record
        """
        source_name = f"{company}_{scraper_type}"
        
        # Check if source exists
        source = self.source_repo.get_by_name(source_name)
        if source:
            return source
        
        # Determine base URL
        if base_url is None:
            base_url = self._get_default_base_url(scraper_type, company)
        
        # Create new source with conditional compliance (needs ToS review)
        new_source = SourceCreate(
            name=source_name,
            base_url=base_url,
            scraper_type=scraper_type,
            compliance_status=ComplianceStatus.CONDITIONAL,
            is_active=True,
            rate_limit_requests=10,
            rate_limit_period=60,
        )
        
        source = self.source_repo.create(new_source)
        logger.info(f"Created new source: {source_name}")
        
        return source
    
    def _get_default_base_url(self, scraper_type: str, company: str) -> str:
        """Get default API URL for a scraper type."""
        if scraper_type in SCRAPER_REGISTRY:
            return SCRAPER_REGISTRY[scraper_type]["url_template"].format(company=company)
        return f"https://{company}.com/careers"
    
    def _get_scraper(self, scraper_type: str, source: Source, company: str) -> BaseScraper:
        """Instantiate the appropriate scraper."""
        if scraper_type not in SCRAPER_REGISTRY:
            raise ValueError(f"Unknown scraper type: {scraper_type}. Available: {list(SCRAPER_REGISTRY.keys())}")
        
        scraper_class = SCRAPER_REGISTRY[scraper_type]["class"]
        scraper_config = SCRAPER_REGISTRY[scraper_type]
        
        # Different scrapers have different constructors
        if scraper_type == "greenhouse":
            return scraper_class(source, board_token=company)
        elif scraper_type == "lever":
            return scraper_class(source, company_slug=company)
        elif scraper_type == "amazon_jobs":
            # Direct scraper - company can be category filter
            return scraper_class(source, category=company if company != "amazon" else None)
        else:
            # Generic case - just pass source
            return scraper_class(source)
    
    def _process_job(self, job: JobCreate, stats: PipelineStats) -> Optional[JobCreate]:
        """
        Process a single job through normalization and classification.
        
        Returns:
            Processed JobCreate or None if duplicate
        """
        # Check for duplicates
        match = self.dedup.find_duplicate(
            content_hash=job.content_hash,
            url=job.url,
            external_id=job.external_id,
            source_id=job.source_id,
            description=job.description,
        )
        
        if match:
            stats.duplicates_skipped += 1
            logger.debug(f"Duplicate detected: {job.title} at {job.company} ({match.match_type.value})")
            return None
        
        # Normalize title
        if self.normalizer:
            job.title = self.normalizer.normalize_title(job.title)
            
            # Normalize location
            if job.location:
                loc = self.normalizer.normalize_location(job.location)
                job.location = str(loc)
                if loc.is_remote and not job.remote_type:
                    job.remote_type = "remote"
            
            # Normalize salary
            if job.salary_min or job.salary_max:
                salary = self.normalizer.normalize_salary(
                    job.salary_min, job.salary_max,
                    period="yearly",
                    currency=job.salary_currency or "USD"
                )
                job.salary_min = salary.min_yearly
                job.salary_max = salary.max_yearly
                job.salary_currency = "USD"
        
        # Classify experience level
        if self.classify_jobs:
            result = self.classifier.classify(
                title=job.title,
                description=job.description,
                location=job.location,
            )
            
            if not job.experience_level:
                job.experience_level = result.experience_level.value
            
            if not job.job_type and result.job_type.value != "unknown":
                job.job_type = result.job_type.value
            
            if not job.remote_type and result.remote_type.value != "unknown":
                job.remote_type = result.remote_type.value
            
            # Track classification stats
            if result.experience_level.value in ("entry", "intern"):
                stats.entry_level_count += 1
            elif result.experience_level.value == "mid":
                stats.mid_level_count += 1
            elif result.experience_level.value in ("senior", "lead", "staff", "principal"):
                stats.senior_count += 1
        
        return job
    
    async def run_scraper(
        self,
        scraper_type: str,
        company: str,
        base_url: Optional[str] = None,
    ) -> PipelineStats:
        """
        Run a scraper and process results through the pipeline.
        
        Args:
            scraper_type: Type of scraper (greenhouse, lever, ashby)
            company: Company identifier
            base_url: Optional base URL override
            
        Returns:
            PipelineStats with processing results
        """
        stats = PipelineStats(source_name=scraper_type, company=company)
        
        try:
            # Ensure dedup index is loaded
            self._ensure_initialized()
            
            # Get or create source
            source = self.get_or_create_source(scraper_type, company, base_url)
            
            # Check compliance
            if source.compliance_status == ComplianceStatus.PROHIBITED:
                stats.errors.append(f"Source {source.name} is PROHIBITED")
                return stats.complete()
            
            # Create and run scraper
            scraper = self._get_scraper(scraper_type, source, company)
            scrape_result = await scraper.scrape()
            
            stats.total_scraped = scrape_result.job_count
            stats.errors.extend(scrape_result.errors)
            
            if not scrape_result.success:
                return stats.complete()
            
            # Process each job
            for job in scrape_result.jobs:
                processed = self._process_job(job, stats)
                
                if processed:
                    # Store in database
                    stored_job, is_new = self.job_repo.upsert(processed)
                    
                    if is_new:
                        stats.new_jobs += 1
                        # Add to dedup index
                        self.dedup.add_job(
                            job_id=stored_job.id,
                            content_hash=stored_job.content_hash,
                            url=stored_job.url,
                            external_id=stored_job.external_id,
                            source_id=stored_job.source_id,
                            description=stored_job.description,
                        )
                    else:
                        stats.updated_jobs += 1
            
            logger.info(
                f"Pipeline complete for {company} ({scraper_type}): "
                f"{stats.new_jobs} new, {stats.updated_jobs} updated, "
                f"{stats.duplicates_skipped} duplicates"
            )
            
        except Exception as e:
            error_msg = f"Pipeline error for {company}: {str(e)}"
            logger.exception(error_msg)
            stats.errors.append(error_msg)
        
        return stats.complete()
    
    async def run_batch(
        self,
        sources: List[tuple],
        concurrent: int = 3,
    ) -> BatchStats:
        """
        Run multiple scrapers in parallel.
        
        Args:
            sources: List of (scraper_type, company) tuples
            concurrent: Max concurrent scrapers
            
        Returns:
            BatchStats with aggregated results
        """
        batch_stats = BatchStats()
        
        semaphore = asyncio.Semaphore(concurrent)
        
        async def run_with_limit(scraper_type: str, company: str):
            async with semaphore:
                return await self.run_scraper(scraper_type, company)
        
        # Create tasks
        tasks = [
            run_with_limit(scraper_type, company)
            for scraper_type, company in sources
        ]
        
        # Run all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                error_stats = PipelineStats(source_name="unknown", company="unknown")
                error_stats.errors.append(str(result))
                batch_stats.source_stats.append(error_stats.complete())
            else:
                batch_stats.source_stats.append(result)
        
        return batch_stats.complete()
    
    def get_stats(self) -> dict:
        """Get current pipeline statistics."""
        job_stats = self.job_repo.get_stats()
        dedup_stats = self.dedup.stats
        
        return {
            "database": job_stats,
            "deduplication": dedup_stats,
            "scrapers_available": list(SCRAPER_REGISTRY.keys()),
        }


# CLI entry point
async def main():
    """Demo/test the pipeline."""
    import sys
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    
    # Use SQLite for demo
    engine = create_engine("sqlite:///demo_jobs.db")
    
    # Create tables (simplified for demo)
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                base_url TEXT NOT NULL,
                scraper_type TEXT NOT NULL,
                compliance_status TEXT DEFAULT 'conditional',
                tos_url TEXT,
                tos_notes TEXT,
                rate_limit_requests INTEGER DEFAULT 10,
                rate_limit_period INTEGER DEFAULT 60,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                external_id TEXT,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                location TEXT,
                description TEXT,
                url TEXT NOT NULL,
                salary_min INTEGER,
                salary_max INTEGER,
                salary_currency TEXT,
                job_type TEXT,
                experience_level TEXT,
                remote_type TEXT,
                content_hash TEXT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES sources(id)
            )
        """))
        conn.commit()
    
    with Session(engine) as db:
        pipeline = JobPipeline(db, enable_tfidf=False)
        
        # Test with some companies
        test_sources = [
            ("greenhouse", "stripe"),
            ("lever", "spotify"),
        ]
        
        print("🚀 Running pipeline demo...\n")
        
        for scraper_type, company in test_sources:
            print(f"Processing {company} ({scraper_type})...")
            stats = await pipeline.run_scraper(scraper_type, company)
            
            if stats.success:
                print(f"  ✅ {stats.total_scraped} scraped, {stats.new_jobs} new, {stats.duplicates_skipped} duplicates")
                print(f"  📊 Entry: {stats.entry_level_count}, Mid: {stats.mid_level_count}, Senior: {stats.senior_count}")
            else:
                print(f"  ❌ Errors: {stats.errors}")
            print()
        
        # Show overall stats
        print("\n📈 Pipeline Stats:")
        print(pipeline.get_stats())


if __name__ == "__main__":
    asyncio.run(main())
