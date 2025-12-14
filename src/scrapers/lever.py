"""Lever ATS scraper - uses public job posting API."""

import asyncio
import logging
from typing import AsyncIterator, Optional

from ..models.job import JobCreate
from ..models.source import Source
from .base import BaseScraper

logger = logging.getLogger(__name__)


class LeverScraper(BaseScraper):
    """
    Scraper for Lever job boards.
    
    Lever provides a public JSON API for job postings:
    https://api.lever.co/v0/postings/{company}
    
    This is the recommended compliant way to access job data.
    """
    
    SCRAPER_TYPE = "lever"
    API_BASE = "https://api.lever.co/v0/postings"
    
    def __init__(self, source: Source, company_slug: str):
        """
        Initialize Lever scraper.
        
        Args:
            source: Source configuration
            company_slug: The company's Lever slug (e.g., 'figma')
        """
        super().__init__(source)
        self.company_slug = company_slug
    
    async def _scrape(self) -> AsyncIterator[JobCreate]:
        """
        Scrape jobs from Lever postings API.
        
        Yields:
            JobCreate objects for each job
        """
        session = await self.get_async_session()
        
        # Lever API supports pagination
        skip = 0
        limit = 100  # Lever's max per request
        
        while True:
            url = f"{self.API_BASE}/{self.company_slug}"
            params = {"mode": "json", "skip": skip, "limit": limit}
            
            async with session.get(url, params=params) as response:
                if response.status == 404:
                    raise Exception(f"Company '{self.company_slug}' not found on Lever")
                if response.status != 200:
                    raise Exception(f"Failed to fetch jobs: HTTP {response.status}")
                
                jobs = await response.json()
            
            if not jobs:
                break
            
            logger.info(f"Fetched {len(jobs)} jobs for {self.company_slug} (offset {skip})")
            
            for job_data in jobs:
                try:
                    job = self._parse_job(job_data)
                    if job:
                        yield job
                except Exception as e:
                    logger.warning(f"Failed to parse job {job_data.get('id')}: {e}")
            
            # Check if we got a full page (more might exist)
            if len(jobs) < limit:
                break
            
            skip += limit
            
            # Rate limiting between pages
            await asyncio.sleep(0.5)
    
    def _parse_job(self, data: dict) -> Optional[JobCreate]:
        """
        Parse a Lever job into a JobCreate model.
        
        Args:
            data: Raw job data from Lever API
            
        Returns:
            JobCreate model or None if parsing fails
        """
        # Required fields
        external_id = data.get("id", "")
        title = data.get("text", "").strip()
        
        if not external_id or not title:
            return None
        
        # Location from categories
        categories = data.get("categories", {})
        location = categories.get("location", "")
        
        # Team/department info
        team = categories.get("team", "")
        commitment = categories.get("commitment", "")  # Full-time, Part-time, etc.
        
        # Job URL
        url = data.get("hostedUrl") or data.get("applyUrl", "")
        
        # Description - Lever provides sections
        description_parts = []
        
        # Opening description
        if data.get("descriptionPlain"):
            description_parts.append(data["descriptionPlain"])
        
        # Additional lists (requirements, responsibilities, etc.)
        for lst in data.get("lists", []):
            list_title = lst.get("text", "")
            list_content = lst.get("content", "")
            if list_title and list_content:
                description_parts.append(f"\n{list_title}:\n{list_content}")
        
        description = "\n".join(description_parts) if description_parts else None
        
        # Map commitment to job_type
        job_type = None
        if commitment:
            commitment_lower = commitment.lower()
            if "full" in commitment_lower:
                job_type = "full-time"
            elif "part" in commitment_lower:
                job_type = "part-time"
            elif "contract" in commitment_lower:
                job_type = "contract"
            elif "intern" in commitment_lower:
                job_type = "internship"
        
        return JobCreate(
            source_id=self.source.id,
            external_id=external_id,
            title=title,
            company=self.source.name,
            location=location if location else None,
            description=description,
            url=url,
            job_type=job_type,
        )


async def scrape_lever_company(company_slug: str) -> list[dict]:
    """
    Quick utility to scrape a Lever company without full source setup.
    
    Args:
        company_slug: The company's Lever slug (e.g., 'figma')
        
    Returns:
        List of raw job dictionaries
    """
    import aiohttp
    
    url = f"https://api.lever.co/v0/postings/{company_slug}"
    
    async with aiohttp.ClientSession() as session:
        all_jobs = []
        skip = 0
        limit = 100
        
        while True:
            async with session.get(url, params={"mode": "json", "skip": skip, "limit": limit}) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch: HTTP {response.status}")
                jobs = await response.json()
            
            if not jobs:
                break
            
            all_jobs.extend(jobs)
            
            if len(jobs) < limit:
                break
            
            skip += limit
            await asyncio.sleep(0.5)
        
        return all_jobs


# CLI support
if __name__ == "__main__":
    import sys
    
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python -m src.scrapers.lever <company_slug>")
            print("Example: python -m src.scrapers.lever figma")
            sys.exit(1)
        
        company_slug = sys.argv[1]
        jobs = await scrape_lever_company(company_slug)
        
        print(f"\nFound {len(jobs)} jobs for {company_slug}:\n")
        for job in jobs[:10]:  # Show first 10
            print(f"  - {job['text']}")
            categories = job.get("categories", {})
            if categories.get("location"):
                print(f"    Location: {categories['location']}")
            if categories.get("team"):
                print(f"    Team: {categories['team']}")
            print(f"    URL: {job.get('hostedUrl', 'N/A')}")
            print()
        
        if len(jobs) > 10:
            print(f"  ... and {len(jobs) - 10} more")
    
    asyncio.run(main())
