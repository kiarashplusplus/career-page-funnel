# career-page-funnel

A **legally redistributable** database of job postings, aggregated from compliant sources.

---

## 🎯 Project Goal

Build a live job database that can be **publicly shared and redistributed** without violating Terms of Service or copyright law.

> **Key Constraint:** Most job boards (LinkedIn, Indeed, Glassdoor) explicitly prohibit scraping and redistribution. This project focuses exclusively on sources where redistribution is permitted.

---

## ⚖️ Compliance-First Architecture

### The Problem with Traditional Job Scrapers

Most open-source job scrapers (including popular libraries like JobSpy) scrape from sources that **explicitly prohibit** automated access and redistribution:

| Source | Scraping Allowed? | Redistribution Allowed? | Legal Risk |
|--------|-------------------|------------------------|------------|
| **LinkedIn** | ❌ NO | ❌ NO | 🔴 HIGH - Actively litigates ([User Agreement §8.2](https://www.linkedin.com/legal/user-agreement)) |
| **Indeed** | ❌ NO | ❌ NO | 🔴 HIGH - Blocks IPs, requires Publisher API ([ToS §2](https://www.indeed.com/legal)) |
| **Glassdoor** | ❌ NO | ❌ NO | 🔴 HIGH - Aggressive bot detection ([ToS](https://www.glassdoor.com/about/terms.htm)) |
| **ZipRecruiter** | ⚠️ API Only | ⚠️ With approval | 🟡 MEDIUM - [Partner program](https://www.ziprecruiter.com/partner) available |
| **Google Jobs** | ⚠️ Aggregator | ⚠️ Check original source | 🟡 MEDIUM - Data comes from other sources |

### Our Approach: Compliant Sources Only

This project scrapes **only** from sources where job data is intentionally made public for distribution:

| Source Type | Examples | Why It's Compliant |
|-------------|----------|-------------------|
| **Direct Company Career Pages** | amazon.jobs, careers.google.com | Companies publish jobs to attract applicants |
| **ATS Public Job Boards** | Greenhouse, Lever, Ashby, Workday | Designed for public access; companies pay to post |
| **Open Source Job Lists** | SimplifyJobs/New-Grad-Positions | Community-contributed under open licenses |
| **Official Partner APIs** | Indeed Publisher API, LinkedIn Jobs API | Explicitly permitted with approval |
| **Press Releases** | Company announcements | Public information |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ✅ COMPLIANT SOURCES ONLY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Direct Career Pages        ATS Public Boards         Curated Lists         │
│  ──────────────────        ─────────────────         ─────────────         │
│  • amazon.jobs              • Greenhouse (*.greenhouse.io)                  │
│  • careers.google.com       • Lever (*.lever.co)       • SimplifyJobs       │
│  • careers.microsoft.com    • Ashby (*.ashbyhq.com)    • New-Grad-Positions │
│  • meta.com/careers         • Workday (*.myworkdayjobs.com)                 │
│  • apple.com/careers        • BambooHR, Jobvite, iCIMS                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROCESSING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Source Validation      2. Deduplication          3. Enrichment          │
│  ───────────────────      ──────────────────        ────────────            │
│  • Verify source is        • URL-based (fast)        • Normalize titles     │
│    in approved registry    • TF-IDF similarity       • Extract salary       │
│  • Check robots.txt        • Content hashing         • Classify level       │
│  • Log compliance status                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REDISTRIBUTABLE DATABASE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  SQLite / PostgreSQL                                                        │
│  • All jobs tagged with source compliance status                            │
│  • Attribution preserved for conditional sources                            │
│  • FTS5 full-text search enabled                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DISTRIBUTION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Public API          Notifications           Data Exports                   │
│  ──────────         ─────────────           ────────────                   │
│  • REST/GraphQL      • Discord webhooks      • CSV/JSON dumps              │
│  • Webhooks          • Email alerts          • Database snapshots          │
│  • RSS feeds         • Slack integration     • API access                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Source Registry

Every data source must be registered and reviewed before scraping:

### ✅ Approved Sources (Safe to Redistribute)

| Source | Type | API Endpoint | Status | Notes |
|--------|------|--------------|--------|-------|
| **Greenhouse** | ATS | `boards-api.greenhouse.io/v1/boards/{company}/jobs` | ✅ TESTED | Public API, works |
| **Lever** | ATS | `api.lever.co/v0/postings/{company}` | ✅ WORKING | API works; many companies migrated away |
| **Ashby** | ATS | `api.ashbyhq.com/posting-api/job-board/{company}` | ✅ TESTED | Public Posting API |
| **SimplifyJobs** | Curated | GitHub repositories | ✅ | MIT license |

### ⚠️ Conditional Sources (Require Registration/Consent)

| Source | API Endpoint | Requirement | Status |
|--------|--------------|-------------|--------|
| **SmartRecruiters** | `api.smartrecruiters.com/v1/companies/{id}/postings` | Developer registration | NEEDS REGISTRATION |
| **JazzHR/Resumator** | `api.resumatorapi.com/v1/jobs` | Customer API key | NEEDS CUSTOMER CONSENT |
| **BambooHR** | `api.bamboohr.com/api/gateway.php/{company}/v1/` | Customer API key | NEEDS CUSTOMER CONSENT |

### ❌ Prohibited Sources (Do Not Use)

| Source | Reason | Legal Risk |
|--------|--------|------------|
| **LinkedIn** | ToS §8.2 explicitly prohibits scraping/redistribution | 🔴 HIGH - Active litigation history |
| **Indeed** | ToS §2 prohibits automated access | 🔴 HIGH - Aggressive blocking |
| **Glassdoor** | ToS prohibits scraping and redistribution | 🔴 HIGH - Bot detection |
| **iCIMS** | ToS explicitly prohibits robots/spiders | 🔴 HIGH - No public API |
| **Workday** | No public API, requires partner network | 🔴 MEDIUM - Partner-only |
| **ZipRecruiter** | Partner-only, no public API | 🟡 MEDIUM - Apply for partnership |

### 📅 ToS Review Log

| Date | Platform | Reviewed By | Finding | Action |
|------|----------|-------------|---------|--------|
| 2025-01 | Greenhouse | Agent | Public API at boards-api.greenhouse.io | ✅ APPROVED |
| 2025-01 | Lever | Agent | Public API at api.lever.co | ✅ APPROVED (API may have changed) |
| 2025-01 | Ashby | Agent | Public Posting API for job board integrations | ✅ APPROVED |
| 2025-01 | SmartRecruiters | Agent | Developer registration required | ⚠️ CONDITIONAL |
| 2025-01 | iCIMS | Agent | ToS explicitly prohibits automated access | ❌ PROHIBITED |
| 2025-01 | Workday | Agent | No public API, partner network only | ❌ PROHIBITED |
| 2025-01 | LinkedIn | Agent | §8.2 prohibits automated data collection | ❌ PROHIBITED |
| 2025-01 | Indeed | Agent | §2 prohibits automated access | ❌ PROHIBITED |

---

## 🔧 Implementation

### Database Schema

```sql
-- Source registry for compliance tracking
CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,                    -- "Greenhouse", "Amazon Jobs", etc.
    base_url TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,             -- "ats", "direct", "api", "curated"
    
    -- Compliance
    compliance_status TEXT NOT NULL,       -- "approved", "conditional", "prohibited"
    tos_url TEXT,
    robots_txt_allows BOOLEAN,
    reviewed_at TIMESTAMP,
    
    -- Attribution (for conditional sources)
    requires_attribution BOOLEAN DEFAULT FALSE,
    attribution_text TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Jobs with compliance tracking
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    
    -- Job data
    external_id TEXT,                      -- ID from source system
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    description TEXT,
    url TEXT NOT NULL,
    posted_at TIMESTAMP,
    
    -- Salary (if available)
    salary_min INTEGER,
    salary_max INTEGER,
    salary_currency TEXT DEFAULT 'USD',
    
    -- Classification
    experience_level TEXT,                 -- "entry", "mid", "senior", "lead"
    job_type TEXT,                         -- "full-time", "part-time", "contract"
    is_remote BOOLEAN,
    
    -- Deduplication
    content_hash TEXT NOT NULL,            -- SHA256 of normalized content
    
    -- Metadata
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    UNIQUE(source_id, external_id)
);

-- Full-text search
CREATE VIRTUAL TABLE jobs_fts USING fts5(
    title, company, location, description,
    content='jobs',
    content_rowid='id'
);

-- Index for fast filtering
CREATE INDEX idx_jobs_company ON jobs(company);
CREATE INDEX idx_jobs_level ON jobs(experience_level);
CREATE INDEX idx_jobs_posted ON jobs(posted_at DESC);
CREATE INDEX idx_jobs_active ON jobs(is_active) WHERE is_active = TRUE;
```

### Scraper Base Class

```python
"""
Base scraper with built-in compliance checking.
Only scrapes from pre-approved sources.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import hashlib
from urllib.robotparser import RobotFileParser
from requests_ratelimiter import LimiterSession


@dataclass
class ScrapedJob:
    title: str
    company: str
    url: str
    location: str | None = None
    description: str | None = None
    posted_at: datetime | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    external_id: str | None = None
    
    @property
    def content_hash(self) -> str:
        """Generate hash for deduplication."""
        content = f"{self.title}|{self.company}|{self.url}".lower()
        return hashlib.sha256(content.encode()).hexdigest()


class ComplianceError(Exception):
    """Raised when attempting to scrape non-compliant source."""
    pass


class BaseScraper(ABC):
    """
    Abstract base class for compliant job scrapers.
    Includes rate limiting and robots.txt checking.
    """
    
    # Must be set by subclass
    SOURCE_NAME: str
    BASE_URL: str
    COMPLIANCE_STATUS: str  # "approved" or "conditional"
    
    def __init__(self, requests_per_second: float = 0.5):
        self.session = LimiterSession(per_second=requests_per_second)
        self.session.headers.update({
            "User-Agent": "CareerPageFunnel/1.0 (job-aggregator; +https://github.com/yourrepo)"
        })
        self._verify_compliance()
    
    def _verify_compliance(self):
        """Check robots.txt before scraping."""
        rp = RobotFileParser()
        rp.set_url(f"{self.BASE_URL}/robots.txt")
        try:
            rp.read()
            if not rp.can_fetch("*", self.BASE_URL):
                raise ComplianceError(
                    f"robots.txt disallows scraping {self.BASE_URL}"
                )
        except Exception as e:
            # If robots.txt is unavailable, proceed with caution
            print(f"Warning: Could not read robots.txt for {self.BASE_URL}: {e}")
    
    @abstractmethod
    def scrape(self) -> list[ScrapedJob]:
        """Scrape jobs from this source. Implemented by subclass."""
        pass
    
    def fetch(self, url: str, **kwargs) -> str:
        """Rate-limited HTTP GET."""
        response = self.session.get(url, **kwargs)
        response.raise_for_status()
        return response.text
```

### Example: Greenhouse Scraper

```python
"""
Scraper for Greenhouse-hosted job boards.
Greenhouse boards are public and designed for distribution.
"""
import json
from .base import BaseScraper, ScrapedJob


class GreenhouseScraper(BaseScraper):
    """
    Scrapes jobs from Greenhouse ATS boards.
    
    Greenhouse provides a public JSON API for job boards:
    https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs
    """
    
    SOURCE_NAME = "Greenhouse"
    BASE_URL = "https://boards-api.greenhouse.io"
    COMPLIANCE_STATUS = "approved"  # Public API, designed for access
    
    def __init__(self, board_token: str):
        """
        Args:
            board_token: Company's Greenhouse board identifier
                        (e.g., "netflix" for Netflix's board)
        """
        super().__init__()
        self.board_token = board_token
        self.api_url = f"{self.BASE_URL}/v1/boards/{board_token}/jobs"
    
    def scrape(self) -> list[ScrapedJob]:
        """Fetch all jobs from the Greenhouse board."""
        response = self.fetch(f"{self.api_url}?content=true")
        data = json.loads(response)
        
        jobs = []
        for job in data.get("jobs", []):
            jobs.append(ScrapedJob(
                external_id=str(job["id"]),
                title=job["title"],
                company=data.get("name", self.board_token),
                url=job["absolute_url"],
                location=job.get("location", {}).get("name"),
                description=job.get("content"),  # HTML content
                posted_at=job.get("updated_at"),
            ))
        
        return jobs


# Usage
if __name__ == "__main__":
    # Scrape Netflix's Greenhouse board
    scraper = GreenhouseScraper("netflix")
    jobs = scraper.scrape()
    print(f"Found {len(jobs)} jobs at Netflix")
```

### Example: Lever Scraper

```python
"""
Scraper for Lever-hosted job boards.
Lever provides public JSON endpoints for job listings.
"""
import json
from .base import BaseScraper, ScrapedJob


class LeverScraper(BaseScraper):
    """
    Scrapes jobs from Lever ATS boards.
    
    Lever provides job data at:
    https://api.lever.co/v0/postings/{company}
    """
    
    SOURCE_NAME = "Lever"
    BASE_URL = "https://api.lever.co"
    COMPLIANCE_STATUS = "approved"
    
    def __init__(self, company_slug: str):
        super().__init__()
        self.company_slug = company_slug
        self.api_url = f"{self.BASE_URL}/v0/postings/{company_slug}"
    
    def scrape(self) -> list[ScrapedJob]:
        response = self.fetch(self.api_url)
        data = json.loads(response)
        
        jobs = []
        for job in data:
            # Extract salary if available
            salary_info = job.get("salaryRange", {})
            
            jobs.append(ScrapedJob(
                external_id=job["id"],
                title=job["text"],
                company=self.company_slug.replace("-", " ").title(),
                url=job["hostedUrl"],
                location=job.get("categories", {}).get("location"),
                description=job.get("descriptionPlain"),
                salary_min=salary_info.get("min"),
                salary_max=salary_info.get("max"),
            ))
        
        return jobs
```

---

## 📊 Compliant Company List

Companies using **public ATS boards** we can safely scrape:

### Greenhouse Companies (1000+ companies)
```
Netflix, Airbnb, Stripe, Discord, Figma, Notion, 
Coinbase, DoorDash, Instacart, Robinhood, Plaid,
Square, Shopify, Twitch, Pinterest, Snap, Lyft...
```

### Lever Companies (Active on Lever)
```
Plaid (76 jobs), Spotify (123 jobs), Palantir (230 jobs)...
(Note: Many companies like Figma, Anthropic, OpenAI, Coinbase 
 have migrated away from Lever to other ATS platforms)
```

### Direct Career Pages (Major Tech)
```
Amazon (amazon.jobs), Google (careers.google.com),
Microsoft (careers.microsoft.com), Meta (meta.com/careers),
Apple (jobs.apple.com), Netflix, Salesforce...
```

---

## 🚫 What This Project Does NOT Do

To maintain legal compliance, this project **explicitly avoids**:

1. ❌ Scraping LinkedIn job listings
2. ❌ Scraping Indeed job listings  
3. ❌ Scraping Glassdoor job listings
4. ❌ Scraping any site that prohibits automated access in ToS
5. ❌ Redistributing data from prohibited sources
6. ❌ Bypassing anti-bot measures or CAPTCHAs
7. ❌ Using residential proxies to evade detection
8. ❌ Impersonating human users

---

## 📚 Reference Repositories

Research from these open-source projects informed our architecture:

| Repository | Key Learnings | Compliance Status |
|------------|---------------|-------------------|
| [ai-job-scraper](references/scrapers/ai-job-scraper) | SQLite optimization, Pydantic models, database schema | Uses JobSpy (non-compliant for redistribution) |
| [job-scraper](references/scrapers/job-scraper) | Company scraper patterns, deduplication, notifications | Mixed - company scrapers are compliant |
| [JobSpy](references/scrapers/JobSpy) | Multi-board API design | ⚠️ For personal use only |
| [JobFunnel](references/scrapers/JobFunnel) | TF-IDF deduplication, filtering | Archived - anti-bot measures broke it |
| [New-Grad-Positions](references/job-lists/New-Grad-Positions) | Curated job list format | ✅ Open source, community contributed |
| [readytotouch](references/platforms/readytotouch) | Go platform architecture, PostgreSQL | ✅ Compliant approach |

### Code Patterns Adopted

| Pattern | Source | Description |
|---------|--------|-------------|
| Rate-limited sessions | job-scraper | `LimiterSession(per_second=0.5)` |
| Content-hash deduplication | ai-job-scraper | SHA256 of normalized job content |
| TF-IDF similarity | JobFunnel | Catch near-duplicate job descriptions |
| Entry-level filtering | job-scraper | Keyword + years-of-experience regex |
| SQLite pragmas | ai-job-scraper | WAL mode, 64MB cache, mmap |
| Pydantic models | ai-job-scraper | Type-safe job data validation |

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **Language** | Python 3.11+ | Ecosystem, async support |
| **HTTP Client** | requests + requests-ratelimiter | Built-in rate limiting |
| **Database** | SQLite (dev) / PostgreSQL (prod) | FTS5 search, proven at scale |
| **ORM** | SQLModel | Pydantic + SQLAlchemy combined |
| **Validation** | Pydantic v2 | Fast, type-safe data models |
| **Deduplication** | hashlib + scikit-learn | Fast hashing + TF-IDF similarity |
| **Notifications** | aiohttp | Async Discord/Slack webhooks |
| **Config** | pydantic-settings | Type-safe .env loading |

---

## 📁 Project Structure

```
career-page-funnel/
├── src/
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseScraper with compliance
│   │   ├── greenhouse.py        # Greenhouse ATS scraper
│   │   ├── lever.py             # Lever ATS scraper
│   │   ├── workday.py           # Workday ATS scraper
│   │   └── direct/              # Direct career page scrapers
│   │       ├── amazon.py
│   │       ├── google.py
│   │       └── microsoft.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── job.py               # Job Pydantic model
│   │   └── source.py            # Source registry model
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py        # DB connection with optimizations
│   │   └── repositories.py      # CRUD operations
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── dedup.py             # Deduplication logic
│   │   ├── classifier.py        # Experience level classification
│   │   └── normalizer.py        # Title/location normalization
│   ├── notifications/
│   │   ├── __init__.py
│   │   ├── discord.py
│   │   └── email.py
│   └── config.py
├── references/                   # Cloned reference repos
│   ├── scrapers/
│   ├── job-lists/
│   └── platforms/
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

---

## ⚠️ Legal Disclaimer

This project is designed for **legally compliant** job data aggregation. However:

1. **This is not legal advice.** Consult a lawyer before commercial use.
2. **Terms of Service change.** Re-verify compliance periodically.
3. **Enforcement varies by jurisdiction.** Laws differ globally.
4. **robots.txt is not legally binding** but should be respected as best practice.
5. **Even compliant scraping can be blocked.** Respect rate limits and server resources.

### Key Legal Cases

| Case | Ruling | Relevance |
|------|--------|-----------|
| **hiQ Labs v. LinkedIn (2022)** | hiQ could scrape public LinkedIn data | Only applies to publicly accessible pages; doesn't permit redistribution or ToS violation |
| **Meta v. BrandTotal (2022)** | Meta won; scraping violated CFAA | ToS violations can lead to legal liability |
| **Clearview AI (ongoing)** | Multiple fines/lawsuits | Aggregating data without consent carries risk |

---

## � Docker Setup

The project is fully dockerized with PostgreSQL, Redis, and optional reference implementations.

### Quick Start

```bash
# Clone the repository
git clone https://github.com/kiarashplusplus/career-page-funnel.git
cd career-page-funnel

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Start core services (PostgreSQL + Redis + main app)
docker compose up -d

# View logs
docker compose logs -f app
```

### Service Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     docker-compose.yml                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CORE SERVICES (always running)                                 │
│  ───────────────────────────────                                │
│  • db (PostgreSQL 16)     - Primary job database                │
│  • redis (Redis 7)        - Caching & job queues                │
│  • app                    - Main application                    │
│                                                                 │
│  REFERENCE IMPLEMENTATIONS (via profiles)                       │
│  ─────────────────────────────────────────                      │
│  • ai-job-scraper         - Streamlit UI, SQLite, AI features   │
│  • job-scraper            - Company scrapers + notifications    │
│  • jobspy                 - Multi-board library (personal use)  │
│  • readytotouch           - Go platform reference               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Running Reference Implementations

Reference implementations are included as profiles. They're vendored (`.git` removed) so you can modify them for compliance:

```bash
# Run AI Job Scraper (Streamlit UI on port 8501)
docker compose --profile ai-scraper up ai-job-scraper

# Run Job Scraper (company scrapers with Discord/email alerts)
docker compose --profile job-scraper up job-scraper

# Run JobSpy container (for personal research only - NOT for redistribution)
docker compose --profile jobspy up jobspy

# Run ReadyToTouch platform (Go-based)
docker compose --profile readytotouch up readytotouch

# Run ALL reference implementations
docker compose --profile all-refs up
```

### Available Services

| Service | Port | Profile | Description |
|---------|------|---------|-------------|
| `db` | 5432 | (core) | PostgreSQL 16 with job schema |
| `redis` | 6379 | (core) | Redis 7 for caching |
| `app` | 8000 | (core) | Main application |
| `ai-job-scraper` | 8501 | `ai-scraper` | Streamlit UI + local AI |
| `job-scraper` | - | `job-scraper` | Company scrapers |
| `jobspy` | - | `jobspy` | JobSpy library shell |
| `readytotouch` | 8080 | `readytotouch` | Go platform |

### Database Access

```bash
# Connect to PostgreSQL
docker compose exec db psql -U cpf -d jobs

# View redistributable jobs only
docker compose exec db psql -U cpf -d jobs -c "SELECT * FROM redistributable_jobs LIMIT 10;"

# Check registered sources
docker compose exec db psql -U cpf -d jobs -c "SELECT name, compliance_status FROM sources;"
```

### Development Workflow

```bash
# Rebuild after code changes
docker compose build app

# Restart specific service
docker compose restart app

# View all logs
docker compose logs -f

# Stop everything
docker compose down

# Stop and remove volumes (fresh start)
docker compose down -v
```

---

## 🚀 Getting Started (Without Docker)

```bash
# Clone the repository
git clone https://github.com/kiarashplusplus/career-page-funnel.git
cd career-page-funnel

# Install dependencies
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your configuration
```

### CLI Pipeline

The integrated pipeline handles scraping → deduplication → classification → storage:

```bash
# Scrape a single company (ATS scrapers)
python -m src.cli scrape greenhouse stripe
python -m src.cli scrape lever spotify

# Scrape Amazon Jobs (direct scraper - by category)
python -m src.cli scrape amazon_jobs software-development
python -m src.cli scrape amazon_jobs machine-learning-science

# Batch scrape multiple companies
python -m src.cli batch greenhouse:stripe,airbnb lever:spotify,plaid

# Show pipeline statistics
python -m src.cli stats

# Search for jobs
python -m src.cli search -q "software engineer" -l "remote"
python -m src.cli search -c "Stripe" --limit 50

# Export jobs to files
python -m src.cli export -f csv -o exports/all_jobs.csv
python -m src.cli export -f json --level entry -o exports/entry_level.json
python -m src.cli export -f jsonl --company Amazon -o exports/amazon.jsonl
python -m src.cli export --summary  # Show export summary without exporting
```

### Available Scrapers

| Scraper | Type | Usage | Notes |
|---------|------|-------|-------|
| `greenhouse` | ATS | `scrape greenhouse {board_token}` | 10k+ companies use Greenhouse |
| `lever` | ATS | `scrape lever {company_slug}` | Common for tech startups |
| `amazon_jobs` | Direct | `scrape amazon_jobs {category}` | Categories: software-development, machine-learning-science, etc. |

### Export Formats

Export jobs to CSV, JSON, or JSONL for analysis, sharing, or integration:

```bash
# Export all jobs to CSV
python -m src.cli export -f csv -o exports/jobs.csv

# Export with filters
python -m src.cli export -f json --level entry -o exports/entry_level.json
python -m src.cli export -f csv --company "Amazon" --location "Seattle"

# Include job descriptions (larger files)
python -m src.cli export -f json --include-description -o exports/jobs_full.json

# Compact JSON (no formatting)
python -m src.cli export -f json --compact -o exports/jobs.json

# JSONL format (one job per line, great for streaming)
python -m src.cli export -f jsonl -o exports/jobs.jsonl

# View export summary without exporting
python -m src.cli export --summary
```

| Format | Use Case | File Size |
|--------|----------|-----------|
| `csv` | Spreadsheets, data analysis | Smallest |
| `json` | APIs, web apps | Medium (with metadata) |
| `jsonl` | Streaming, data pipelines | Medium (no metadata) |

### Amazon Jobs Categories

Use these as the "company" argument for amazon_jobs:
- `software-development`
- `machine-learning-science`
- `data-science`
- `solutions-architect`
- `project-program-product-management-technical`
- `systems-quality-security-engineering`

### Pipeline Features

| Feature | Description |
|---------|-------------|
| **Deduplication** | Hash-based (SHA256) + optional TF-IDF similarity detection |
| **Classification** | Auto-detects experience level (entry/mid/senior), job type, remote status |
| **Normalization** | Standardizes titles (Sr. → Senior), locations (SF → San Francisco), salaries |
| **Compliance** | Only scrapes from registered, approved sources |

### Python API

```python
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.pipeline import JobPipeline

engine = create_engine("sqlite:///jobs.db")

async def main():
    with Session(engine) as db:
        pipeline = JobPipeline(db)
        
        # Scrape single company
        stats = await pipeline.run_scraper("greenhouse", "stripe")
        print(f"Found {stats.new_jobs} new jobs")
        
        # Batch scrape
        batch = await pipeline.run_batch([
            ("greenhouse", "stripe"),
            ("greenhouse", "airbnb"),
            ("lever", "spotify"),
        ])
        print(f"Total: {batch.total_new} new jobs")

asyncio.run(main())
```

### Processing Components

```python
from src.processing import (
    DuplicateDetector,  # Hash + TF-IDF deduplication
    JobClassifier,       # Experience level classification
    JobNormalizer,       # Title/location/salary normalization
)

# Classify a job
classifier = JobClassifier()
result = classifier.classify("Senior Software Engineer", "5+ years required")
print(result.experience_level)      # ExperienceLevel.SENIOR
print(result.is_entry_level_friendly)  # False

# Normalize data
normalizer = JobNormalizer()
print(normalizer.normalize_title("Sr. SWE"))  # "Senior Software Engineer"
print(normalizer.normalize_location("SF, CA"))  # "San Francisco, CA"
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

**Note:** This license applies to the code in this repository. Job data scraped from external sources remains subject to those sources' terms of service.

