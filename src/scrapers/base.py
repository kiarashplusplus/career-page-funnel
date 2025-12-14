"""Base scraper class with compliance checks and rate limiting."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Optional

import aiohttp
from requests_ratelimiter import LimiterSession

from ..models.job import JobCreate
from ..models.source import ComplianceStatus, Source

logger = logging.getLogger(__name__)


@dataclass
class ScraperResult:
    """Result from a scraping operation."""
    
    jobs: list[JobCreate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    @property
    def job_count(self) -> int:
        return len(self.jobs)
    
    def complete(self) -> "ScraperResult":
        self.completed_at = datetime.utcnow()
        return self


class BaseScraper(ABC):
    """
    Base class for all job scrapers.
    
    Enforces compliance checks before scraping and provides rate limiting.
    All scrapers must implement the _scrape method.
    """
    
    SCRAPER_TYPE: str = "base"
    
    def __init__(self, source: Source):
        """
        Initialize scraper with source configuration.
        
        Args:
            source: Source configuration including compliance status and rate limits
        """
        self.source = source
        self._session: Optional[LimiterSession] = None
        self._async_session: Optional[aiohttp.ClientSession] = None
    
    @property
    def session(self) -> LimiterSession:
        """Get or create a rate-limited requests session."""
        if self._session is None:
            self._session = LimiterSession(
                per_second=self.source.rate_limit_requests / self.source.rate_limit_period
            )
        return self._session
    
    async def get_async_session(self) -> aiohttp.ClientSession:
        """Get or create an async HTTP session."""
        if self._async_session is None or self._async_session.closed:
            self._async_session = aiohttp.ClientSession(
                headers={"User-Agent": "CareerPageFunnel/1.0 (Compliance Bot)"}
            )
        return self._async_session
    
    def check_compliance(self) -> tuple[bool, str]:
        """
        Check if source is compliant for scraping.
        
        Returns:
            Tuple of (is_compliant, reason)
        """
        if self.source.compliance_status == ComplianceStatus.PROHIBITED:
            return False, f"Source '{self.source.name}' is marked as PROHIBITED. Scraping is not allowed."
        
        if self.source.compliance_status == ComplianceStatus.PENDING_REVIEW:
            return False, f"Source '{self.source.name}' is PENDING_REVIEW. Complete ToS review before scraping."
        
        if not self.source.is_active:
            return False, f"Source '{self.source.name}' is not active."
        
        return True, "Source is compliant for scraping."
    
    async def scrape(self) -> ScraperResult:
        """
        Main entry point for scraping. Checks compliance before proceeding.
        
        Returns:
            ScraperResult with jobs and any errors
        """
        result = ScraperResult()
        
        # Check compliance first
        is_compliant, reason = self.check_compliance()
        if not is_compliant:
            logger.warning(f"Compliance check failed: {reason}")
            result.errors.append(reason)
            return result.complete()
        
        logger.info(f"Starting scrape for {self.source.name} ({self.SCRAPER_TYPE})")
        
        try:
            async for job in self._scrape():
                result.jobs.append(job)
        except Exception as e:
            error_msg = f"Error scraping {self.source.name}: {str(e)}"
            logger.exception(error_msg)
            result.errors.append(error_msg)
        finally:
            await self.close()
        
        logger.info(f"Completed scrape for {self.source.name}: {result.job_count} jobs, {len(result.errors)} errors")
        return result.complete()
    
    @abstractmethod
    async def _scrape(self) -> AsyncIterator[JobCreate]:
        """
        Implement the actual scraping logic.
        
        Yields:
            JobCreate objects for each job found
        """
        pass
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()
        if self._session:
            self._session.close()
