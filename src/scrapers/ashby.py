"""
Ashby ATS scraper - uses public Posting API.

Ashby provides a public API specifically designed for job board integrations:
https://api.ashbyhq.com/posting-api/job-board/{company}

Documentation: https://developers.ashbyhq.com/docs/posting-api-overview
"""

import logging
from typing import AsyncIterator, Optional
from datetime import datetime

from ..models.job import JobCreate
from ..models.source import Source
from .base import BaseScraper

logger = logging.getLogger(__name__)


class AshbyCompanyNotFoundError(Exception):
    """Raised when a company is not found on Ashby."""
    pass


class AshbyScraper(BaseScraper):
    """
    Scraper for Ashby job boards.
    
    Ashby's Posting API is explicitly designed for job aggregation use cases.
    Many companies have migrated from Greenhouse/Lever to Ashby.
    
    Example companies using Ashby:
    - notion
    - ramp
    - mercury
    - linear
    - openai
    - plaid
    """
    
    SCRAPER_TYPE = "ashby"
    API_BASE = "https://api.ashbyhq.com/posting-api/job-board"
    
    def __init__(self, source: Source, company_slug: str):
        """
        Initialize Ashby scraper.
        
        Args:
            source: Source configuration
            company_slug: The company's Ashby identifier (e.g., 'notion', 'ramp')
        """
        super().__init__(source)
        self.company_slug = company_slug.lower()
    
    async def _scrape(self) -> AsyncIterator[JobCreate]:
        """
        Scrape jobs from Ashby Posting API.
        
        Yields:
            JobCreate objects for each job
        """
        session = await self.get_async_session()
        
        # Ashby returns all jobs in a single request (no pagination needed)
        url = f"{self.API_BASE}/{self.company_slug}"
        
        async with session.get(url) as response:
            if response.status == 404:
                raise AshbyCompanyNotFoundError(
                    f"Company '{self.company_slug}' not found on Ashby. "
                    f"Verify at: https://jobs.ashbyhq.com/{self.company_slug}"
                )
            
            if response.status != 200:
                raise Exception(f"Failed to fetch jobs: HTTP {response.status}")
            
            data = await response.json()
        
        jobs = data.get("jobs", [])
        logger.info(f"Found {len(jobs)} jobs for {self.company_slug}")
        
        for job_data in jobs:
            try:
                job = self._parse_job(job_data)
                if job:
                    yield job
            except Exception as e:
                logger.warning(f"Failed to parse job {job_data.get('id')}: {e}")
    
    def _parse_job(self, data: dict) -> Optional[JobCreate]:
        """
        Parse an Ashby job into a JobCreate model.
        
        Args:
            data: Raw job data from Ashby API
            
        Returns:
            JobCreate model or None if parsing fails
        """
        # Required fields
        external_id = str(data.get("id", ""))
        title = data.get("title", "").strip()
        
        if not external_id or not title:
            return None
        
        # Location handling
        location = self._parse_location(data)
        
        # Build the job URL - Ashby provides jobUrl or we construct it
        url = (
            data.get("jobUrl") or 
            data.get("applyUrl") or 
            f"https://jobs.ashbyhq.com/{self.company_slug}/{external_id}"
        )
        
        # Extract description
        description = data.get("descriptionPlain") or data.get("descriptionHtml") or ""
        
        # Parse remote status
        is_remote = data.get("isRemote")
        if is_remote is None and location:
            is_remote = "remote" in location.lower()
        
        # Determine remote type
        remote_type = None
        if is_remote:
            remote_type = "remote"
        elif data.get("isOnsite"):
            remote_type = "onsite"
        
        # Parse employment type
        job_type = self._parse_employment_type(data.get("employmentType"))
        
        # Parse posted date
        posted_at = None
        if data.get("publishedAt"):
            try:
                posted_at = datetime.fromisoformat(
                    data["publishedAt"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass
        
        # Department/team info
        department = data.get("department") or data.get("team") or data.get("departmentName")
        
        return JobCreate(
            source_id=self.source.id,  # Required field from base class
            title=title,
            company=self.source.name,  # Use source name for consistency
            url=url,
            location=location,
            description=description,
            external_id=external_id,
            job_type=job_type,
            remote_type=remote_type,
        )
    
    def _parse_location(self, data: dict) -> Optional[str]:
        """Extract location from various Ashby location formats."""
        # Direct location string
        if data.get("location"):
            return data["location"]
        
        # Structured address
        if data.get("address"):
            addr = data["address"]
            if isinstance(addr, dict):
                parts = [
                    addr.get("addressLocality"),
                    addr.get("addressRegion"),
                    addr.get("addressCountry")
                ]
                return ", ".join(p for p in parts if p)
        
        # Location object with name
        if data.get("locationName"):
            return data["locationName"]
        
        # Multiple locations
        if data.get("locations"):
            locs = data["locations"]
            if isinstance(locs, list) and locs:
                if isinstance(locs[0], str):
                    return locs[0]
                elif isinstance(locs[0], dict):
                    return locs[0].get("name") or locs[0].get("location")
        
        return None
    
    def _parse_employment_type(self, emp_type: Optional[str]) -> Optional[str]:
        """Convert Ashby employment type to standard format."""
        if not emp_type:
            return None
        
        mapping = {
            "FullTime": "full-time",
            "Full-Time": "full-time",
            "Full Time": "full-time",
            "PartTime": "part-time",
            "Part-Time": "part-time",
            "Part Time": "part-time",
            "Contract": "contract",
            "Contractor": "contract",
            "Intern": "internship",
            "Internship": "internship",
            "Temporary": "temporary",
            "Temp": "temporary",
        }
        return mapping.get(emp_type, emp_type.lower().replace(" ", "-"))
