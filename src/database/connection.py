"""Database connection management."""

import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# Get database URL from environment - default to SQLite for easy development
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///jobs.db"  # SQLite by default for easy development
)

# Lazy initialization
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        if DATABASE_URL.startswith("sqlite"):
            _engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        else:
            _engine = create_engine(DATABASE_URL)
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def get_engine():
    """Get the SQLAlchemy engine."""
    return engine


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.
    
    Usage:
        with get_db() as db:
            db.execute(...)
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database schema.
    
    For PostgreSQL, this is handled by init-db.sql in Docker.
    This function is for SQLite development or manual initialization.
    """
    engine = get_engine()
    
    with engine.connect() as conn:
        # Check if tables exist (SQLite compatible)
        try:
            conn.execute(text("SELECT 1 FROM sources LIMIT 1"))
            return  # Tables already exist
        except Exception:
            pass  # Tables don't exist, create them
        
        # Create tables for SQLite
        if DATABASE_URL.startswith("sqlite"):
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
                    is_active INTEGER DEFAULT 1,
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
                    is_active INTEGER DEFAULT 1,
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
            print("SQLite database initialized")
        else:
            # For PostgreSQL, use init-db.sql
            init_sql_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "docker", "init-db.sql"
            )
            
            if os.path.exists(init_sql_path):
                with open(init_sql_path) as f:
                    sql = f.read()
                conn.execute(text(sql))
                conn.commit()
                print("Database initialized from init-db.sql")
            else:
                print("Warning: init-db.sql not found. Database may not be properly initialized.")


# CLI support
if __name__ == "__main__":
    init_db()
    print("Database initialization complete.")
