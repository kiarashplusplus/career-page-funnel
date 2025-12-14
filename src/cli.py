#!/usr/bin/env python3
"""
Career Page Funnel - CLI Interface

Run scrapers and process jobs through the pipeline.

Usage:
    python -m src.cli scrape greenhouse stripe
    python -m src.cli scrape lever spotify
    python -m src.cli batch greenhouse:stripe,airbnb lever:spotify
    python -m src.cli stats
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_db_session():
    """Get a database session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    
    # Use environment variable or default to SQLite
    import os
    db_url = os.environ.get("DATABASE_URL", "sqlite:///jobs.db")
    
    engine = create_engine(db_url)
    return Session(engine), engine


def init_database(engine):
    """Initialize database tables if they don't exist."""
    from sqlalchemy import text
    
    with engine.connect() as conn:
        # Check if tables exist (PostgreSQL/SQLite compatible)
        try:
            conn.execute(text("SELECT 1 FROM sources LIMIT 1"))
            conn.execute(text("SELECT 1 FROM jobs LIMIT 1"))
            return  # Tables exist
        except Exception:
            pass
        
        # Create tables
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
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_jobs_hash ON jobs(content_hash)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source_id)"))
        conn.commit()
        logger.info("Database tables created")


async def cmd_scrape(args):
    """Run a single scraper."""
    from src.pipeline import JobPipeline
    
    db, engine = get_db_session()
    init_database(engine)
    
    try:
        pipeline = JobPipeline(db, enable_tfidf=not args.no_tfidf)
        stats = await pipeline.run_scraper(args.scraper_type, args.company)
        
        print(f"\n{'='*60}")
        print(f"📊 Scrape Results: {args.company} ({args.scraper_type})")
        print(f"{'='*60}")
        print(f"  Total scraped:     {stats.total_scraped}")
        print(f"  New jobs:          {stats.new_jobs}")
        print(f"  Updated jobs:      {stats.updated_jobs}")
        print(f"  Duplicates:        {stats.duplicates_skipped}")
        print(f"  Duration:          {stats.duration_seconds:.1f}s")
        print()
        print(f"  📈 Classification:")
        print(f"    Entry-level:     {stats.entry_level_count}")
        print(f"    Mid-level:       {stats.mid_level_count}")
        print(f"    Senior+:         {stats.senior_count}")
        
        if stats.errors:
            print(f"\n  ⚠️  Errors:")
            for error in stats.errors:
                print(f"    - {error}")
        
        print()
        return 0 if stats.success else 1
        
    finally:
        db.close()


async def cmd_batch(args):
    """Run multiple scrapers in batch."""
    from src.pipeline import JobPipeline
    
    db, engine = get_db_session()
    init_database(engine)
    
    try:
        pipeline = JobPipeline(db, enable_tfidf=not args.no_tfidf)
        
        # Parse source specs: "greenhouse:stripe,airbnb lever:spotify"
        sources = []
        for spec in args.sources:
            if ":" in spec:
                scraper_type, companies = spec.split(":", 1)
                for company in companies.split(","):
                    sources.append((scraper_type.strip(), company.strip()))
            else:
                # Assume it's just a company for the default scraper
                sources.append(("greenhouse", spec.strip()))
        
        print(f"\n🚀 Running batch scrape for {len(sources)} sources...")
        print(f"   Sources: {sources}\n")
        
        batch_stats = await pipeline.run_batch(sources, concurrent=args.concurrent)
        
        print(f"\n{'='*60}")
        print(f"📊 Batch Results")
        print(f"{'='*60}")
        
        for stats in batch_stats.source_stats:
            status = "✅" if stats.success else "❌"
            print(f"  {status} {stats.company} ({stats.source_name}): "
                  f"{stats.new_jobs} new, {stats.updated_jobs} updated")
        
        print(f"\n  📈 Totals:")
        print(f"    Total scraped:   {batch_stats.total_scraped}")
        print(f"    New jobs:        {batch_stats.total_new}")
        print(f"    Updated jobs:    {batch_stats.total_updated}")
        print(f"    Success rate:    {batch_stats.success_rate:.0%}")
        print()
        
        return 0 if batch_stats.success_rate == 1.0 else 1
        
    finally:
        db.close()


async def cmd_stats(args):
    """Show pipeline statistics."""
    from src.pipeline import JobPipeline
    
    db, engine = get_db_session()
    init_database(engine)
    
    try:
        pipeline = JobPipeline(db, enable_tfidf=False)  # No need for TF-IDF for stats
        stats = pipeline.get_stats()
        
        print(f"\n{'='*60}")
        print(f"📊 Pipeline Statistics")
        print(f"{'='*60}")
        
        if "database" in stats:
            db_stats = stats["database"]
            print(f"\n  📁 Database:")
            print(f"    Total jobs:      {db_stats.get('total_jobs', 'N/A')}")
            print(f"    Active jobs:     {db_stats.get('active_jobs', 'N/A')}")
            print(f"    Companies:       {db_stats.get('unique_companies', 'N/A')}")
            print(f"    Sources:         {db_stats.get('sources_with_jobs', 'N/A')}")
        
        if "deduplication" in stats:
            dedup_stats = stats["deduplication"]
            print(f"\n  🔍 Deduplication Index:")
            print(f"    Hash index:      {dedup_stats.get('hash_index_size', 0)}")
            print(f"    URL index:       {dedup_stats.get('url_index_size', 0)}")
            print(f"    TF-IDF enabled:  {dedup_stats.get('tfidf_enabled', False)}")
        
        print(f"\n  🔧 Available Scrapers:")
        for scraper in stats.get("scrapers_available", []):
            print(f"    - {scraper}")
        
        print()
        return 0
        
    finally:
        db.close()


async def cmd_search(args):
    """Search for jobs."""
    from src.database import JobRepository
    
    db, engine = get_db_session()
    init_database(engine)
    
    try:
        repo = JobRepository(db)
        jobs = repo.search(
            query=args.query,
            company=args.company,
            location=args.location,
            limit=args.limit,
        )
        
        print(f"\n🔍 Found {len(jobs)} jobs")
        print(f"{'='*60}\n")
        
        for job in jobs:
            level_emoji = {
                "entry": "🟢", "intern": "🟢",
                "mid": "🟡",
                "senior": "🔴", "lead": "🔴", "staff": "🔴",
            }.get(job.experience_level, "⚪")
            
            print(f"{level_emoji} {job.title}")
            print(f"   🏢 {job.company}")
            print(f"   📍 {job.location or 'Location not specified'}")
            if job.experience_level:
                print(f"   📊 {job.experience_level}")
            print(f"   🔗 {job.url}")
            print()
        
        return 0
        
    finally:
        db.close()


async def cmd_export(args):
    """Export jobs to CSV/JSON."""
    from src.export import JobExporter, ExportOptions
    from pathlib import Path
    
    db, engine = get_db_session()
    init_database(engine)
    
    try:
        exporter = JobExporter(db)
        
        # Build export options
        options = ExportOptions(
            company=args.company,
            location=args.location,
            source_type=args.source,
            experience_level=args.level,
            active_only=not args.include_inactive,
            redistributable_only=not args.all_sources,
            include_description=args.include_description,
            include_source_info=True,
            limit=args.limit,
        )
        
        # Show summary if requested
        if args.summary:
            summary = exporter.get_export_summary(options)
            print(f"\n📊 Export Summary")
            print(f"{'='*60}")
            print(f"  Total jobs: {summary['total_jobs']}")
            print(f"\n  Top Companies:")
            for item in summary['top_companies'][:5]:
                print(f"    - {item['company']}: {item['count']} jobs")
            print(f"\n  By Experience Level:")
            for level, count in summary['by_experience_level'].items():
                print(f"    - {level}: {count}")
            print()
            return 0
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = args.format
            output_path = Path(f"exports/jobs_{timestamp}.{ext}")
        
        # Ensure exports directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📤 Exporting jobs...")
        print(f"   Format: {args.format.upper()}")
        print(f"   Output: {output_path}")
        
        # Run export
        if args.format == "csv":
            result = exporter.export_csv(options, output_path)
        elif args.format == "jsonl":
            result = exporter.export_jsonl(options, output_path)
        else:  # json
            result = exporter.export_json(options, output_path, pretty=not args.compact)
        
        if result.success:
            print(f"\n✅ Export complete!")
            print(f"   Jobs exported: {result.job_count}")
            print(f"   File: {result.file_path}")
            print(f"   Duration: {result.duration_seconds:.2f}s")
        else:
            print(f"\n❌ Export failed:")
            for error in result.errors:
                print(f"   - {error}")
            return 1
        
        return 0
        
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Career Page Funnel - Job Aggregation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scrape greenhouse stripe     Scrape Stripe's Greenhouse board
  %(prog)s scrape lever spotify         Scrape Spotify's Lever postings
  %(prog)s batch greenhouse:stripe,airbnb lever:spotify
                                        Batch scrape multiple companies
  %(prog)s stats                        Show pipeline statistics
  %(prog)s search -q "software engineer" -l "remote"
                                        Search for jobs
  %(prog)s export -f csv -o jobs.csv    Export all jobs to CSV
  %(prog)s export -f json --level entry Export entry-level jobs to JSON
  %(prog)s export --summary             Show export summary
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Run a single scraper")
    scrape_parser.add_argument("scraper_type", choices=["greenhouse", "lever", "ashby", "amazon_jobs"],
                               help="Type of scraper to use")
    scrape_parser.add_argument("company", help="Company identifier/slug (or category for amazon_jobs)")
    scrape_parser.add_argument("--no-tfidf", action="store_true",
                               help="Disable TF-IDF similarity detection")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run multiple scrapers")
    batch_parser.add_argument("sources", nargs="+",
                              help="Source specs: 'scraper:company1,company2 ...'")
    batch_parser.add_argument("-c", "--concurrent", type=int, default=3,
                              help="Max concurrent scrapers (default: 3)")
    batch_parser.add_argument("--no-tfidf", action="store_true",
                              help="Disable TF-IDF similarity detection")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show pipeline statistics")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for jobs")
    search_parser.add_argument("-q", "--query", help="Search query")
    search_parser.add_argument("-c", "--company", help="Filter by company")
    search_parser.add_argument("-l", "--location", help="Filter by location")
    search_parser.add_argument("-n", "--limit", type=int, default=20,
                               help="Max results (default: 20)")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export jobs to CSV/JSON")
    export_parser.add_argument("-f", "--format", choices=["csv", "json", "jsonl"],
                               default="csv", help="Export format (default: csv)")
    export_parser.add_argument("-o", "--output", help="Output file path")
    export_parser.add_argument("-c", "--company", help="Filter by company name")
    export_parser.add_argument("-l", "--location", help="Filter by location")
    export_parser.add_argument("-s", "--source", 
                               choices=["greenhouse", "lever", "ashby", "amazon_jobs"],
                               help="Filter by source type")
    export_parser.add_argument("--level", 
                               choices=["intern", "entry", "mid", "senior", "lead", "staff"],
                               help="Filter by experience level")
    export_parser.add_argument("-n", "--limit", type=int, help="Max jobs to export")
    export_parser.add_argument("--include-description", action="store_true",
                               help="Include job descriptions in export")
    export_parser.add_argument("--include-inactive", action="store_true",
                               help="Include inactive/closed jobs")
    export_parser.add_argument("--all-sources", action="store_true",
                               help="Include non-redistributable sources")
    export_parser.add_argument("--compact", action="store_true",
                               help="Compact JSON output (no indentation)")
    export_parser.add_argument("--summary", action="store_true",
                               help="Show export summary without exporting")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Run the appropriate command
    if args.command == "scrape":
        return asyncio.run(cmd_scrape(args))
    elif args.command == "batch":
        return asyncio.run(cmd_batch(args))
    elif args.command == "stats":
        return asyncio.run(cmd_stats(args))
    elif args.command == "search":
        return asyncio.run(cmd_search(args))
    elif args.command == "export":
        return asyncio.run(cmd_export(args))
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
