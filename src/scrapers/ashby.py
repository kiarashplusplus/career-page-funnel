"""
Ashby job board scraper.

Uses Ashby's public Posting API: https://api.ashbyhq.com/posting-api/job-board/{company}
"""
import asyncio
from datetime import datetime
from typing import AsyncIterator

import aiohttp

from src.models.job import Job
from src.models.source import ComplianceStatus
from src.scrapers.base import BaseScraper


class AshbyScraper(BaseScraper):
    """
    Scraper for Ashby-hosted job boards.
    
    Example companies using Ashby:
    - notion
    - anthropic
    - ramp
    - mercury
    - figma
    """
    
    SOURCE_NAME = "Ashby"
    BASE_URL = "https://api.ashbyhq.com"
    COMPLIANCE_STATUS = ComplianceStatus.APPROVED
    
    def __init__(self, company_slug: str):
        """
        Initialize with company slug (e.g., 'notion', 'anthropic').
        
        Args:
            company_slug: The company identifier used in the Ashby URL
        """
        super().__init__(rate_limit=1.0)  # 1 request per second
        self.company_slug = company_slug.lower()
    
    async def _fetch_jobs_page(self, session: aiohttp.ClientSession) -> dict:
        """Fetch all jobs from Ashby API (returns all at once, no pagination)."""
        url = f"{self.BASE_URL}/posting-api/job-board/{self.company_slug}"
        
        async with session.get(url) as response:
            if response.status == 404:
                return {"jobs": [], "error": f"Company '{self.company_slug}' not found on Ashby"}
            response.raise_for_status()
            return await response.json()
    
    async def scrape(self) -> AsyncIterator[Job]:
        """
        Scrape all jobs from a company's Ashby board.
        
        Yields:
            Job: Parsed job objects
        """
        async with aiohttp.ClientSession() as session:
            data = await self._fetch_jobs_page(session)
            
            if "error" in data:
                print(f"Warning: {data['error']}")
                return
            
            jobs = data.get("jobs", [])
            
            for job_data in jobs:
                try:
                    job = self._parse_job(job_data)
                    yield job
                except Exception as e:
                    print(f"Error parsing job {job_data.get('id', 'unknown')}: {e}")
                    continue
    
    def _parse_job(self, data: dict) -> Job:
        """Parse Ashby API response into Job model."""
        # Extract location from nested structure
        location = None
        if data.get("location"):
            location = data["location"]
        elif data.get("address"):
            addr = data["address"]
            if isinstance(addr, dict):
                parts = [
                    addr.get("addressLocality"),
                    addr.get("addressRegion"),
                    addr.get("addressCountry")
                ]
                location = ", ".join(p for p in parts if p)
        
        # Parse remote status
        is_remote = data.get("isRemote")
        if is_remote is None and location:
            is_remote = "remote" in location.lower()
        
        # Parse posted date
        posted_at = None
        if data.get("publishedAt"):
            try:
                posted_at = datetime.fromisoformat(
                    data["publishedAt"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass
        
        return Job(
            title=data["title"],
            company=self.company_slug,  # Will be normalized later
            url=data.get("jobUrl") or data.get("applyUrl") or f"https://jobs.ashbyhq.com/{self.company_slug}/{data['id']}",
            location=location,
            description=data.get("descriptionPlain") or data.get("descriptionHtml"),
            posted_at=posted_at,
            external_id=data.get("id"),
            source=self.SOURCE_NAME,
            is_remote=is_remote,
            job_type=self._parse_employment_type(data.get("employmentType")),
            department=data.get("department") or data.get("team"),
        )
    
    def _parse_employment_type(self, emp_type: str | None) -> str | None:
        """Convert Ashby employment type to our standard format."""
        if not emp_type:
            return None
        
        mapping = {
            "FullTime": "full-time",
            "PartTime": "part-time",
            "Contract": "contract",
            "Intern": "internship",
            "Internship": "internship",
            "Temporary": "temporary",
        }
        return mapping.get(emp_type, emp_type.lower())


async def main():
    """Test the Ashby scraper."""
    import sys
    
    company = sys.argv[1] if len(sys.argv) > 1 else "notion"
    
    print(f"\n🔍 Scraping jobs from {company}'s Ashby board...")
    print("=" * 60)
    
    scraper = AshbyScraper(company)
    jobs = []
    
    async for job in scraper.scrape():
        jobs.append(job)
    
    print(f"\n✅ Found {len(jobs)} jobs\n")
    
    # Show sample jobs
    for job in jobs[:5]:
        print(f"📌 {job.title}")
        print(f"   📍 {job.location or 'Location not specified'}")
        if job.department:
            print(f"   🏢 {job.department}")
        if job.is_remote:
            print(f"   🌐 Remote")
        print(f"   🔗 {job.url}")
        print()
    
    if len(jobs) > 5:
        print(f"... and {len(jobs) - 5} more jobs")
    
    return jobs


if __name__ == "__main__":
    asyncio.run(main())
