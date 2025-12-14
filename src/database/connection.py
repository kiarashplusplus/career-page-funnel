"""Database connection management."""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# Get database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://cpf:cpf_dev_password@localhost:5432/jobs"
)

# For SQLite fallback in development
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
    from sqlalchemy import MetaData
    
    # Create tables if they don't exist
    metadata = MetaData()
    
    with engine.connect() as conn:
        # Check if tables exist
        result = conn.execute(text(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'sources')"
        ))
        exists = result.scalar()
        
        if not exists:
            # Read and execute init-db.sql
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
