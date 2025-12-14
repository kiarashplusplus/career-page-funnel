"""Greenhouse ATS scraper - uses public job board API."""

import asyncio
import logging
from typing import AsyncIterator, Optional

from ..models.job import JobCreate
from ..models.source import Source
from .base import BaseScraper

logger = logging.getLogger(__name__)


class GreenhouseScraper(BaseScraper):
    """
    Scraper for Greenhouse job boards.
    
    Greenhouse provides a public JSON API for job boards:
    https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs
    
    This is the recommended compliant way to access job data.
    """
    
    SCRAPER_TYPE = "greenhouse"
    API_BASE = "https://boards-api.greenhouse.io/v1/boards"
    
    def __init__(self, source: Source, board_token: str):
        """
        Initialize Greenhouse scraper.
        
        Args:
            source: Source configuration
            board_token: The company's Greenhouse board token (e.g., 'netflix')
        """
        super().__init__(source)
        self.board_token = board_token
    
    async def _scrape(self) -> AsyncIterator[JobCreate]:
        """
        Scrape jobs from Greenhouse board API.
        
        Yields:
            JobCreate objects for each job
        """
        session = await self.get_async_session()
        
        # Get list of jobs
        jobs_url = f"{self.API_BASE}/{self.board_token}/jobs"
        
        async with session.get(jobs_url, params={"content": "true"}) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch jobs: HTTP {response.status}")
            
            data = await response.json()
        
        jobs = data.get("jobs", [])
        logger.info(f"Found {len(jobs)} jobs for {self.board_token}")
        
        for job_data in jobs:
            try:
                job = self._parse_job(job_data)
                if job:
                    yield job
            except Exception as e:
                logger.warning(f"Failed to parse job {job_data.get('id')}: {e}")
    
    def _parse_job(self, data: dict) -> Optional[JobCreate]:
        """
        Parse a Greenhouse job into a JobCreate model.
        
        Args:
            data: Raw job data from Greenhouse API
            
        Returns:
            JobCreate model or None if parsing fails
        """
        # Required fields
        external_id = str(data.get("id", ""))
        title = data.get("title", "").strip()
        
        if not external_id or not title:
            return None
        
        # Location handling
        location = None
        if data.get("location"):
            location = data["location"].get("name", "")
        
        # Build the job URL
        url = data.get("absolute_url", f"https://boards.greenhouse.io/{self.board_token}/jobs/{external_id}")
        
        # Extract description (HTML content)
        description = data.get("content", "")
        
        # Greenhouse metadata fields
        metadata = data.get("metadata", [])
        job_type = None
        experience_level = None
        
        for meta in metadata:
            name = meta.get("name", "").lower()
            value = meta.get("value")
            if name in ("employment type", "job type", "type"):
                job_type = value
            elif name in ("experience level", "seniority", "level"):
                experience_level = value
        
        return JobCreate(
            source_id=self.source.id,
            external_id=external_id,
            title=title,
            company=self.source.name,
            location=location,
            description=description,
            url=url,
            job_type=job_type,
            experience_level=experience_level,
        )


async def scrape_greenhouse_company(board_token: str) -> list[dict]:
    """
    Quick utility to scrape a Greenhouse board without full source setup.
    
    Args:
        board_token: The company's board token (e.g., 'netflix', 'airbnb')
        
    Returns:
        List of raw job dictionaries
    """
    import aiohttp
    
    url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params={"content": "true"}) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch: HTTP {response.status}")
            data = await response.json()
            return data.get("jobs", [])


# CLI support
if __name__ == "__main__":
    import sys
    
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python -m src.scrapers.greenhouse <board_token>")
            print("Example: python -m src.scrapers.greenhouse netflix")
            sys.exit(1)
        
        board_token = sys.argv[1]
        jobs = await scrape_greenhouse_company(board_token)
        
        print(f"\nFound {len(jobs)} jobs for {board_token}:\n")
        for job in jobs[:10]:  # Show first 10
            print(f"  - {job['title']}")
            if job.get("location"):
                print(f"    Location: {job['location']['name']}")
            print(f"    URL: {job.get('absolute_url', 'N/A')}")
            print()
        
        if len(jobs) > 10:
            print(f"  ... and {len(jobs) - 10} more")
    
    asyncio.run(main())
