#!/usr/bin/env python3
"""
Seed Database Script
====================
Scrapes jobs from 100+ companies using compliant sources (Greenhouse, Lever, Ashby, Amazon Jobs).

Usage:
    python scripts/seed_database.py              # Full seed (~70 companies)
    python scripts/seed_database.py --quick      # Quick seed (~17 companies)
    python scripts/seed_database.py --test       # Test mode (4 companies)
    python scripts/seed_database.py --category tech  # Only tech companies
    python scripts/seed_database.py --cleanup    # Remove sources with 0 jobs
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import JobPipeline
from src.cli import get_db_session, init_database

# =============================================================================
# CURATED COMPANY LISTS (Verified December 2024)
# =============================================================================

# Greenhouse companies (board_token) - VERIFIED WORKING
GREENHOUSE_COMPANIES = [
    # --- Fintech ---
    ("stripe", "Stripe", "fintech"),
    ("coinbase", "Coinbase", "crypto"),
    ("robinhood", "Robinhood", "fintech"),
    ("chime", "Chime", "fintech"),
    ("brex", "Brex", "fintech"),
    ("affirm", "Affirm", "fintech"),
    ("block", "Block/Square", "fintech"),
    
    # --- Big Tech ---
    ("airbnb", "Airbnb", "tech"),
    ("pinterest", "Pinterest", "social"),
    ("dropbox", "Dropbox", "tech"),
    ("lyft", "Lyft", "tech"),
    ("instacart", "Instacart", "delivery"),
    ("doordashusa", "DoorDash", "delivery"),
    
    # --- Developer Tools ---
    ("gitlab", "GitLab", "devtools"),
    ("figma", "Figma", "design"),
    ("airtable", "Airtable", "productivity"),
    ("webflow", "Webflow", "design"),
    ("retool", "Retool", "devtools"),
    ("vercel", "Vercel", "devtools"),
    ("tailscale", "Tailscale", "infrastructure"),
    
    # --- Social & Consumer ---
    ("discord", "Discord", "social"),
    ("reddit", "Reddit", "social"),
    ("duolingo", "Duolingo", "education"),
    
    # --- AI & ML ---
    ("anthropic", "Anthropic", "ai"),
    
    # --- Enterprise & SaaS ---
    ("gusto", "Gusto", "hr"),
    ("lattice", "Lattice", "hr"),
    ("remote", "Remote", "hr"),
    ("intercom", "Intercom", "saas"),
    ("okta", "Okta", "security"),
    ("twilio", "Twilio", "devtools"),
    ("cloudflare", "Cloudflare", "infrastructure"),
    ("amplitude", "Amplitude", "analytics"),
    ("mixpanel", "Mixpanel", "analytics"),
    
    # --- Data & Analytics ---
    ("databricks", "Databricks", "data"),
    ("fivetran", "Fivetran", "data"),
    
    # --- E-commerce ---
    ("faire", "Faire", "ecommerce"),
    
    # --- Health & Biotech ---
    ("modernhealth", "Modern Health", "health"),
    
    # --- Autonomous Vehicles ---
    ("waymo", "Waymo", "av"),
]

# Lever companies (company_slug) - VERIFIED WORKING
LEVER_COMPANIES = [
    ("spotify", "Spotify", "tech"),
    ("wealthfront", "Wealthfront", "fintech"),
]

# Ashby companies (company_slug) - VERIFIED WORKING
ASHBY_COMPANIES = [
    # --- AI & ML ---
    ("openai", "OpenAI", "ai"),
    ("cohere", "Cohere", "ai"),
    ("perplexity", "Perplexity", "ai"),
    ("cursor", "Cursor", "ai"),
    ("modal", "Modal", "ai"),
    ("runway", "Runway", "ai"),
    
    # --- Fintech ---
    ("ramp", "Ramp", "fintech"),
    ("deel", "Deel", "hr"),
    
    # --- Developer Tools ---
    ("notion", "Notion", "productivity"),
    ("linear", "Linear", "devtools"),
    ("supabase", "Supabase", "devtools"),
    ("zapier", "Zapier", "automation"),
    ("render", "Render", "infrastructure"),
    
    # --- Data ---
    ("snowflake", "Snowflake", "data"),
    
    # --- Security ---
    ("1password", "1Password", "security"),
    ("vanta", "Vanta", "security"),
    
    # --- Health & Biotech ---
    ("benchling", "Benchling", "biotech"),
    
    # --- Climate & Infrastructure ---
    ("watershed", "Watershed", "climate"),
    ("crusoe", "Crusoe Energy", "infrastructure"),
]

# Amazon Jobs categories
AMAZON_CATEGORIES = [
    ("software-development", "Amazon Software Development"),
    ("machine-learning-science", "Amazon ML/Science"),
    ("data-science", "Amazon Data Science"),
    ("solutions-architect", "Amazon Solutions Architect"),
    ("systems-quality-security-engineering", "Amazon Systems Engineering"),
]


def get_companies_by_category(category: str | None = None):
    """Filter companies by category."""
    gh = GREENHOUSE_COMPANIES
    lv = LEVER_COMPANIES
    ab = ASHBY_COMPANIES
    
    if category:
        category = category.lower()
        gh = [(t, n, c) for t, n, c in gh if c == category]
        lv = [(t, n, c) for t, n, c in lv if c == category]
        ab = [(t, n, c) for t, n, c in ab if c == category]
    
    return gh, lv, ab


def get_quick_list():
    """Return a quick list of ~17 high-value companies."""
    greenhouse = [
        ("stripe", "Stripe", "fintech"),
        ("airbnb", "Airbnb", "tech"),
        ("coinbase", "Coinbase", "crypto"),
        ("discord", "Discord", "social"),
        ("gitlab", "GitLab", "devtools"),
        ("anthropic", "Anthropic", "ai"),
        ("gusto", "Gusto", "hr"),
        ("databricks", "Databricks", "data"),
        ("figma", "Figma", "design"),
        ("robinhood", "Robinhood", "fintech"),
    ]
    lever = [
        ("spotify", "Spotify", "tech"),
    ]
    ashby = [
        ("openai", "OpenAI", "ai"),
        ("notion", "Notion", "productivity"),
        ("ramp", "Ramp", "fintech"),
        ("linear", "Linear", "devtools"),
        ("perplexity", "Perplexity", "ai"),
    ]
    amazon = [
        ("software-development", "Amazon Software Development"),
    ]
    return greenhouse, lever, ashby, amazon


def get_test_list():
    """Return a minimal test list."""
    return (
        [("stripe", "Stripe", "fintech")],
        [("spotify", "Spotify", "tech")],
        [("openai", "OpenAI", "ai")],
        [("software-development", "Amazon Software Development")],
    )


async def cleanup_dead_sources(db):
    """Remove sources that have 0 jobs."""
    from sqlalchemy import text
    
    # Find sources with 0 jobs
    result = db.execute(text("""
        SELECT s.id, s.name, s.scraper_type
        FROM sources s
        LEFT JOIN jobs j ON j.source_id = s.id
        GROUP BY s.id
        HAVING COUNT(j.id) = 0
    """))
    dead_sources = result.fetchall()
    
    if not dead_sources:
        print("✅ No dead sources to clean up")
        return 0
    
    print(f"\n🧹 Cleaning up {len(dead_sources)} dead sources:")
    for source_id, name, scraper_type in dead_sources:
        print(f"   - {name} ({scraper_type})")
    
    # Delete them
    source_ids = [s[0] for s in dead_sources]
    db.execute(text(f"DELETE FROM sources WHERE id IN ({','.join(map(str, source_ids))})"))
    db.commit()
    
    print(f"✅ Removed {len(dead_sources)} dead sources")
    return len(dead_sources)


async def run_seed(
    greenhouse_list: list,
    lever_list: list,
    ashby_list: list,
    amazon_list: list,
    concurrent: int = 3,
    use_tfidf: bool = False,
):
    """Run the seed operation."""
    db, engine = get_db_session()
    init_database(engine)
    
    try:
        pipeline = JobPipeline(db, enable_tfidf=use_tfidf)
        
        total_stats = {
            "companies_attempted": 0,
            "companies_succeeded": 0,
            "companies_failed": 0,
            "total_new": 0,
            "total_updated": 0,
            "total_duplicates": 0,
            "failed_companies": [],
        }
        
        # Build batch sources
        sources = []
        
        # Greenhouse
        for token, name, _ in greenhouse_list:
            sources.append(("greenhouse", token))
            total_stats["companies_attempted"] += 1
        
        # Lever
        for slug, name, _ in lever_list:
            sources.append(("lever", slug))
            total_stats["companies_attempted"] += 1
        
        # Ashby
        for slug, name, _ in ashby_list:
            sources.append(("ashby", slug))
            total_stats["companies_attempted"] += 1
        
        # Amazon
        for category, name in amazon_list:
            sources.append(("amazon_jobs", category))
            total_stats["companies_attempted"] += 1
        
        print(f"\n{'='*70}")
        print(f"🚀 SEEDING DATABASE")
        print(f"{'='*70}")
        print(f"  Greenhouse companies: {len(greenhouse_list)}")
        print(f"  Lever companies:      {len(lever_list)}")
        print(f"  Ashby companies:      {len(ashby_list)}")
        print(f"  Amazon categories:    {len(amazon_list)}")
        print(f"  Total sources:        {len(sources)}")
        print(f"  Concurrent scrapers:  {concurrent}")
        print(f"{'='*70}\n")
        
        start_time = datetime.now()
        
        # Run batch
        results = await pipeline.run_batch(sources, concurrent=concurrent)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Aggregate results
        for result in results.source_stats:
            if result.errors:
                total_stats["companies_failed"] += 1
                total_stats["failed_companies"].append(f"{result.company} ({result.source_name}): {result.errors[0]}")
            else:
                total_stats["companies_succeeded"] += 1
                total_stats["total_new"] += result.new_jobs
                total_stats["total_updated"] += result.updated_jobs
                total_stats["total_duplicates"] += result.duplicates_skipped
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📊 SEED COMPLETE")
        print(f"{'='*70}")
        print(f"  Duration:             {duration:.1f}s ({duration/60:.1f} minutes)")
        print(f"  Companies attempted:  {total_stats['companies_attempted']}")
        print(f"  Companies succeeded:  {total_stats['companies_succeeded']}")
        print(f"  Companies failed:     {total_stats['companies_failed']}")
        print(f"\n  📈 Jobs:")
        print(f"    New jobs:           {total_stats['total_new']}")
        print(f"    Updated jobs:       {total_stats['total_updated']}")
        print(f"    Duplicates:         {total_stats['total_duplicates']}")
        
        if total_stats["failed_companies"]:
            print(f"\n  ❌ Failed companies:")
            for fail in total_stats["failed_companies"][:10]:
                print(f"    - {fail}")
            if len(total_stats["failed_companies"]) > 10:
                print(f"    ... and {len(total_stats['failed_companies']) - 10} more")
        
        print(f"{'='*70}\n")
        
        return total_stats
        
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Seed the job database from compliant sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/seed_database.py              # Full seed (~70 companies)
  python scripts/seed_database.py --quick      # Quick seed (~17 companies)
  python scripts/seed_database.py --test       # Test mode (4 companies)
  python scripts/seed_database.py --category fintech  # Only fintech companies
  python scripts/seed_database.py --concurrent 5      # More parallelism
  python scripts/seed_database.py --cleanup    # Remove sources with 0 jobs
        """
    )
    
    parser.add_argument("--quick", action="store_true",
                        help="Quick seed with ~17 high-value companies")
    parser.add_argument("--test", action="store_true",
                        help="Test mode with 4 companies")
    parser.add_argument("--category", type=str,
                        help="Filter by category (tech, fintech, ai, devtools, etc.)")
    parser.add_argument("--concurrent", type=int, default=3,
                        help="Max concurrent scrapers (default: 3)")
    parser.add_argument("--tfidf", action="store_true",
                        help="Enable TF-IDF similarity detection (slower)")
    parser.add_argument("--greenhouse-only", action="store_true",
                        help="Only scrape Greenhouse companies")
    parser.add_argument("--lever-only", action="store_true",
                        help="Only scrape Lever companies")
    parser.add_argument("--ashby-only", action="store_true",
                        help="Only scrape Ashby companies")
    parser.add_argument("--amazon-only", action="store_true",
                        help="Only scrape Amazon Jobs")
    parser.add_argument("--no-amazon", action="store_true",
                        help="Skip Amazon Jobs")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove sources with 0 jobs before seeding")
    
    args = parser.parse_args()
    
    # Handle cleanup
    if args.cleanup:
        db, engine = get_db_session()
        init_database(engine)
        try:
            asyncio.run(cleanup_dead_sources(db))
        finally:
            db.close()
        
        # If only cleanup requested, exit
        if not any([args.quick, args.test, args.category, args.greenhouse_only, 
                    args.lever_only, args.ashby_only, args.amazon_only]):
            return 0
    
    # Determine company lists
    if args.test:
        greenhouse, lever, ashby, amazon = get_test_list()
    elif args.quick:
        greenhouse, lever, ashby, amazon = get_quick_list()
    else:
        greenhouse, lever, ashby = get_companies_by_category(args.category)
        amazon = AMAZON_CATEGORIES if not args.category else []
    
    # Apply filters
    if args.greenhouse_only:
        lever = []
        ashby = []
        amazon = []
    elif args.lever_only:
        greenhouse = []
        ashby = []
        amazon = []
    elif args.ashby_only:
        greenhouse = []
        lever = []
        amazon = []
    elif args.amazon_only:
        greenhouse = []
        lever = []
        ashby = []
    elif args.no_amazon:
        amazon = []
    
    if not greenhouse and not lever and not ashby and not amazon:
        print("❌ No companies to scrape with current filters")
        return 1
    
    # Run
    try:
        stats = asyncio.run(run_seed(
            greenhouse_list=greenhouse,
            lever_list=lever,
            ashby_list=ashby,
            amazon_list=amazon,
            concurrent=args.concurrent,
            use_tfidf=args.tfidf,
        ))
        
        return 0 if stats["companies_failed"] < stats["companies_attempted"] else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Seed interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
