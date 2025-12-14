"""
Amazon Jobs direct scraper.

Uses the public JSON API at amazon.jobs which provides job data.

COMPLIANCE NOTES:
- robots.txt (checked Dec 2024): User-agent: * allows all except /internal/* paths
- No authentication required for public search API
- Returns structured JSON data designed for public consumption
- Rate limiting: Conservative default (10 req/60s) to be a good citizen
"""

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator, Optional

import aiohttp

from ..models.job import JobCreate
from ..models.source import Source
from .base import BaseScraper

logger = logging.getLogger(__name__)


class AmazonJobsScraper(BaseScraper):
    """
    Scraper for Amazon Jobs (amazon.jobs).
    
    Uses the public JSON API endpoint:
    https://www.amazon.jobs/en/search.json
    
    This API is publicly accessible and returns structured job data.
    The robots.txt allows scraping of job pages (only /internal is blocked).
    """
    
    SCRAPER_TYPE = "amazon_jobs"
    API_BASE = "https://www.amazon.jobs"
    SEARCH_ENDPOINT = "/en/search.json"
    
    # Job categories and filters available on Amazon Jobs
    JOB_CATEGORIES = [
        "software-development",
        "machine-learning-science", 
        "data-science",
        "solutions-architect",
        "project-program-product-management-technical",
        "systems-quality-security-engineering",
        "database-administration",
        "cloud-infrastructure-architecture",
    ]
    
    def __init__(
        self,
        source: Source,
        category: Optional[str] = None,
        location: Optional[str] = None,
        page_size: int = 100,
        max_jobs: int = 5000,
    ):
        """
        Initialize Amazon Jobs scraper.
        
        Args:
            source: Source configuration
            category: Optional job category filter (e.g., 'software-development')
            location: Optional location filter
            page_size: Number of results per page (max 100)
            max_jobs: Maximum number of jobs to fetch
        """
        super().__init__(source)
        self.category = category
        self.location = location
        self.page_size = min(page_size, 100)  # API max is 100
        self.max_jobs = max_jobs
    
    async def _scrape(self) -> AsyncIterator[JobCreate]:
        """
        Scrape jobs from Amazon Jobs API.
        
        Yields:
            JobCreate objects for each job
        """
        session = await self.get_async_session()
        
        offset = 0
        total_fetched = 0
        total_available = None
        
        while total_fetched < self.max_jobs:
            # Build API params
            params = {
                "offset": offset,
                "result_limit": self.page_size,
            }
            
            if self.category:
                params["category[]"] = self.category
            if self.location:
                params["location[]"] = self.location
            
            url = f"{self.API_BASE}{self.SEARCH_ENDPOINT}"
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch jobs: HTTP {response.status}")
                        break
                    
                    data = await response.json()
                    
                    if data.get("error"):
                        logger.error(f"API error: {data.get('error')}")
                        break
                    
                    # Get total count on first request
                    if total_available is None:
                        total_available = data.get("hits", 0)
                        logger.info(f"Amazon Jobs: Found {total_available} total jobs")
                    
                    jobs = data.get("jobs", [])
                    if not jobs:
                        logger.info("No more jobs to fetch")
                        break
                    
                    for job_data in jobs:
                        try:
                            job = self._parse_job(job_data)
                            if job:
                                yield job
                                total_fetched += 1
                                
                                if total_fetched >= self.max_jobs:
                                    logger.info(f"Reached max_jobs limit: {self.max_jobs}")
                                    return
                        except Exception as e:
                            logger.warning(f"Failed to parse job {job_data.get('id_icims')}: {e}")
                    
                    offset += len(jobs)
                    
                    # Small delay between requests to be respectful
                    await asyncio.sleep(0.5)
                    
            except aiohttp.ClientError as e:
                logger.error(f"Network error: {e}")
                break
        
        logger.info(f"Amazon Jobs: Scraped {total_fetched} jobs")
    
    def _parse_job(self, data: dict) -> Optional[JobCreate]:
        """
        Parse an Amazon Jobs listing into a JobCreate model.
        
        Args:
            data: Raw job data from Amazon Jobs API
            
        Returns:
            JobCreate model or None if parsing fails
        """
        # Required fields
        external_id = data.get("id_icims") or data.get("id", "")
        title = (data.get("title") or "").strip()
        
        if not external_id or not title:
            return None
        
        # Location handling - Amazon provides detailed location data
        location = data.get("location") or ""
        normalized_location = data.get("normalized_location") or location
        city = data.get("city") or ""
        
        # Use most specific location available
        if normalized_location:
            location = normalized_location
        elif city:
            location = f"{city}, {data.get('country_code', '')}"
        
        # URL
        job_path = data.get("job_path", f"/en/jobs/{external_id}")
        url = f"{self.API_BASE}{job_path}"
        
        # Description
        description = data.get("description") or ""
        description_short = data.get("description_short") or ""
        
        # Combine short and full description
        if description_short and description:
            full_description = f"{description_short}\n\n{description}"
        else:
            full_description = description or description_short
        
        # Add qualifications if available
        basic_quals = data.get("basic_qualifications") or ""
        preferred_quals = data.get("preferred_qualifications") or ""
        
        if basic_quals:
            full_description += f"\n\n## Basic Qualifications\n{basic_quals}"
        if preferred_quals:
            full_description += f"\n\n## Preferred Qualifications\n{preferred_quals}"
        
        # Job type
        job_schedule = data.get("job_schedule_type") or ""  # full-time, part-time
        job_type = job_schedule if job_schedule else None
        
        # Experience level (derived from category/title)
        experience_level = self._derive_experience_level(title, data)
        
        # Remote type from location info
        remote_type = self._derive_remote_type(data)
        
        return JobCreate(
            source_id=self.source.id,
            external_id=str(external_id),
            title=title,
            company="Amazon",  # Always Amazon for this scraper
            location=location,
            description=full_description,
            url=url,
            job_type=job_type,
            experience_level=experience_level,
            remote_type=remote_type,
        )
    
    def _derive_experience_level(self, title: str, data: dict) -> Optional[str]:
        """Derive experience level from job title and category."""
        title_lower = title.lower()
        
        # Check for explicit levels
        if "intern" in title_lower or data.get("is_intern"):
            return "intern"
        if "entry" in title_lower or "new grad" in title_lower or data.get("university_job"):
            return "entry"
        if "junior" in title_lower or "jr" in title_lower:
            return "entry"
        if "senior" in title_lower or "sr" in title_lower:
            return "senior"
        if "principal" in title_lower or "staff" in title_lower:
            return "lead"
        if "director" in title_lower or "vp" in title_lower:
            return "executive"
        if data.get("is_manager"):
            return "senior"
        
        return None  # Unknown - let classifier handle it
    
    def _derive_remote_type(self, data: dict) -> Optional[str]:
        """Derive remote type from location data."""
        locations = data.get("locations", [])
        
        for loc in locations:
            if isinstance(loc, str):
                try:
                    import json
                    loc_data = json.loads(loc)
                except (json.JSONDecodeError, TypeError):
                    continue
            else:
                loc_data = loc
            
            if isinstance(loc_data, dict):
                loc_type = loc_data.get("type", "").upper()
                if loc_type == "REMOTE":
                    return "remote"
                elif loc_type == "HYBRID":
                    return "hybrid"
                elif loc_type == "ONSITE":
                    return "onsite"
        
        return None


async def scrape_amazon_jobs(
    category: Optional[str] = None,
    location: Optional[str] = None,
    max_jobs: int = 100,
) -> list[dict]:
    """
    Quick utility to scrape Amazon Jobs without full source setup.
    
    Args:
        category: Optional job category filter
        location: Optional location filter
        max_jobs: Maximum number of jobs to fetch
        
    Returns:
        List of raw job dictionaries
    """
    url = f"{AmazonJobsScraper.API_BASE}{AmazonJobsScraper.SEARCH_ENDPOINT}"
    
    params = {
        "offset": 0,
        "result_limit": min(max_jobs, 100),
    }
    
    if category:
        params["category[]"] = category
    if location:
        params["location[]"] = location
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch: HTTP {response.status}")
            data = await response.json()
            return data.get("jobs", [])


# CLI support
if __name__ == "__main__":
    import sys
    
    async def main():
        print("Amazon Jobs Scraper")
        print("=" * 50)
        
        category = sys.argv[1] if len(sys.argv) > 1 else None
        max_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        
        print(f"Category: {category or 'All'}")
        print(f"Max jobs: {max_jobs}")
        print()
        
        jobs = await scrape_amazon_jobs(category=category, max_jobs=max_jobs)
        
        print(f"Found {len(jobs)} jobs:")
        for job in jobs:
            print(f"  - {job.get('title')[:60]}... ({job.get('normalized_location', 'Unknown location')})")
    
    asyncio.run(main())
