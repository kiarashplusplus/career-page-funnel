#!/usr/bin/env python3
"""
Seed Database Script
====================
Scrapes jobs from 100+ companies using compliant sources (Greenhouse, Lever, Amazon Jobs).

Usage:
    python scripts/seed_database.py              # Full seed (~100 companies)
    python scripts/seed_database.py --quick      # Quick seed (~20 companies)
    python scripts/seed_database.py --test       # Test mode (3 companies)
    python scripts/seed_database.py --category tech  # Only tech companies
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
# CURATED COMPANY LISTS (Verified Board Tokens)
# =============================================================================

# Greenhouse companies (board_token)
# Format: (board_token, company_name, category)
GREENHOUSE_COMPANIES = [
    # --- FAANG & Big Tech ---
    ("stripe", "Stripe", "fintech"),
    ("airbnb", "Airbnb", "tech"),
    ("square", "Square/Block", "fintech"),
    ("coinbase", "Coinbase", "crypto"),
    ("robinhood", "Robinhood", "fintech"),
    ("plaid", "Plaid", "fintech"),
    ("chime", "Chime", "fintech"),
    ("brex", "Brex", "fintech"),
    ("ramp", "Ramp", "fintech"),
    
    # --- Developer Tools ---
    ("gitlab", "GitLab", "devtools"),
    ("notion", "Notion", "productivity"),
    ("figma", "Figma", "design"),
    ("airtable", "Airtable", "productivity"),
    ("webflow", "Webflow", "design"),
    ("miro", "Miro", "productivity"),
    ("loom", "Loom", "productivity"),
    ("zapier", "Zapier", "automation"),
    ("retool", "Retool", "devtools"),
    ("vercel", "Vercel", "devtools"),
    ("supabase", "Supabase", "devtools"),
    ("planetscale", "PlanetScale", "devtools"),
    ("railway", "Railway", "devtools"),
    
    # --- Social & Consumer ---
    ("discord", "Discord", "social"),
    ("reddit", "Reddit", "social"),
    ("pinterestearly", "Pinterest", "social"),
    ("snap", "Snap", "social"),
    ("bumble", "Bumble", "social"),
    ("duolingo", "Duolingo", "education"),
    
    # --- AI & ML ---
    ("openai", "OpenAI", "ai"),
    ("anthropic", "Anthropic", "ai"),
    ("huggingface", "Hugging Face", "ai"),
    ("stability", "Stability AI", "ai"),
    ("cohere", "Cohere", "ai"),
    ("scale", "Scale AI", "ai"),
    ("anyscale", "Anyscale", "ai"),
    ("weights", "Weights & Biases", "ai"),
    
    # --- Enterprise & B2B ---
    ("gusto", "Gusto", "hr"),
    ("rippling", "Rippling", "hr"),
    ("lattice", "Lattice", "hr"),
    ("deel", "Deel", "hr"),
    ("remote", "Remote", "hr"),
    ("ashbyhq", "Ashby", "hr"),
    ("lever", "Lever", "hr"),
    
    # --- Security ---
    ("snaborofficial", "Snyk", "security"),
    ("1password", "1Password", "security"),
    ("crowdstrike", "CrowdStrike", "security"),
    ("lacework", "Lacework", "security"),
    ("waborofficial", "Wiz", "security"),
    
    # --- Data & Analytics ---
    ("databricks", "Databricks", "data"),
    ("snowflake", "Snowflake", "data"),
    ("dbt", "dbt Labs", "data"),
    ("fivetran", "Fivetran", "data"),
    ("airbyte", "Airbyte", "data"),
    ("hex", "Hex", "data"),
    ("census", "Census", "data"),
    
    # --- Infrastructure ---
    ("hashicorp", "HashiCorp", "infrastructure"),
    ("tailscale", "Tailscale", "infrastructure"),
    ("fly", "Fly.io", "infrastructure"),
    ("render", "Render", "infrastructure"),
    
    # --- E-commerce ---
    ("shopify", "Shopify", "ecommerce"),
    ("faire", "Faire", "ecommerce"),
    ("bolt", "Bolt", "ecommerce"),
    ("affirm", "Affirm", "fintech"),
    ("klarna", "Klarna", "fintech"),
    
    # --- Health & Biotech ---
    ("tempus", "Tempus", "health"),
    ("color", "Color Health", "health"),
    ("ro", "Ro", "health"),
    ("hims", "Hims & Hers", "health"),
    ("modernhealth", "Modern Health", "health"),
]

# Lever companies (company_slug)
# Format: (company_slug, company_name, category)
LEVER_COMPANIES = [
    # --- Tech ---
    ("spotify", "Spotify", "tech"),
    ("netflix", "Netflix", "tech"),
    ("cloudflare", "Cloudflare", "infrastructure"),
    ("twilio", "Twilio", "devtools"),
    
    # --- Fintech ---
    ("nerdwallet", "NerdWallet", "fintech"),
    ("wealthfront", "Wealthfront", "fintech"),
    ("betterment", "Betterment", "fintech"),
    
    # --- E-commerce & Delivery ---
    ("instacart", "Instacart", "delivery"),
    ("doordash", "DoorDash", "delivery"),
    ("postmates", "Postmates", "delivery"),
    
    # --- SaaS ---
    ("amplitude", "Amplitude", "analytics"),
    ("mixpanel", "Mixpanel", "analytics"),
    ("segment", "Segment", "data"),
    ("intercom", "Intercom", "saas"),
    ("zendesk", "Zendesk", "saas"),
    
    # --- AI/ML ---
    ("huggingface", "Hugging Face", "ai"),
    ("replicate", "Replicate", "ai"),
    ("modal", "Modal", "ai"),
    
    # --- Security ---
    ("okta", "Okta", "security"),
    ("auth0", "Auth0", "security"),
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
    
    if category:
        category = category.lower()
        gh = [(t, n, c) for t, n, c in gh if c == category]
        lv = [(t, n, c) for t, n, c in lv if c == category]
    
    return gh, lv


def get_quick_list():
    """Return a quick list of ~20 high-value companies."""
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
        ("chime", "Chime", "fintech"),
        ("airtable", "Airtable", "productivity"),
    ]
    lever = [
        ("spotify", "Spotify", "tech"),
        ("netflix", "Netflix", "tech"),
    ]
    amazon = [
        ("software-development", "Amazon Software Development"),
    ]
    return greenhouse, lever, amazon


def get_test_list():
    """Return a minimal test list."""
    return (
        [("stripe", "Stripe", "fintech")],
        [("spotify", "Spotify", "tech")],
        [("software-development", "Amazon Software Development")],
    )


async def run_seed(
    greenhouse_list: list,
    lever_list: list,
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
        
        # Amazon
        for category, name in amazon_list:
            sources.append(("amazon_jobs", category))
            total_stats["companies_attempted"] += 1
        
        print(f"\n{'='*70}")
        print(f"🚀 SEEDING DATABASE")
        print(f"{'='*70}")
        print(f"  Greenhouse companies: {len(greenhouse_list)}")
        print(f"  Lever companies:      {len(lever_list)}")
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
  python scripts/seed_database.py              # Full seed (~100 companies)
  python scripts/seed_database.py --quick      # Quick seed (~20 companies)
  python scripts/seed_database.py --test       # Test mode (3 companies)
  python scripts/seed_database.py --category fintech  # Only fintech companies
  python scripts/seed_database.py --concurrent 5      # More parallelism
        """
    )
    
    parser.add_argument("--quick", action="store_true",
                        help="Quick seed with ~20 high-value companies")
    parser.add_argument("--test", action="store_true",
                        help="Test mode with 3 companies")
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
    parser.add_argument("--amazon-only", action="store_true",
                        help="Only scrape Amazon Jobs")
    parser.add_argument("--no-amazon", action="store_true",
                        help="Skip Amazon Jobs")
    
    args = parser.parse_args()
    
    # Determine company lists
    if args.test:
        greenhouse, lever, amazon = get_test_list()
    elif args.quick:
        greenhouse, lever, amazon = get_quick_list()
    else:
        greenhouse, lever = get_companies_by_category(args.category)
        amazon = AMAZON_CATEGORIES if not args.category else []
    
    # Apply filters
    if args.greenhouse_only:
        lever = []
        amazon = []
    elif args.lever_only:
        greenhouse = []
        amazon = []
    elif args.amazon_only:
        greenhouse = []
        lever = []
    elif args.no_amazon:
        amazon = []
    
    if not greenhouse and not lever and not amazon:
        print("❌ No companies to scrape with current filters")
        return 1
    
    # Run
    try:
        stats = asyncio.run(run_seed(
            greenhouse_list=greenhouse,
            lever_list=lever,
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
