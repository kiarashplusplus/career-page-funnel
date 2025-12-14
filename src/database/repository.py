"""Database repository classes for jobs and sources."""

from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models.job import Job, JobCreate, JobUpdate
from ..models.source import ComplianceStatus, Source, SourceCreate, SourceUpdate


class SourceRepository:
    """Repository for source CRUD operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, source_id: int) -> Optional[Source]:
        """Get a source by ID."""
        result = self.db.execute(
            text("SELECT * FROM sources WHERE id = :id"),
            {"id": source_id}
        )
        row = result.fetchone()
        return Source(**row._asdict()) if row else None
    
    def get_by_name(self, name: str) -> Optional[Source]:
        """Get a source by name."""
        result = self.db.execute(
            text("SELECT * FROM sources WHERE name = :name"),
            {"name": name}
        )
        row = result.fetchone()
        return Source(**row._asdict()) if row else None
    
    def get_all(self, include_inactive: bool = False) -> list[Source]:
        """Get all sources."""
        query = "SELECT * FROM sources"
        if not include_inactive:
            query += " WHERE is_active = true"
        query += " ORDER BY name"
        
        result = self.db.execute(text(query))
        return [Source(**row._asdict()) for row in result.fetchall()]
    
    def get_approved(self) -> list[Source]:
        """Get all approved sources for scraping."""
        result = self.db.execute(
            text("""
                SELECT * FROM sources 
                WHERE compliance_status IN ('approved', 'conditional')
                AND is_active = true
                ORDER BY name
            """)
        )
        return [Source(**row._asdict()) for row in result.fetchall()]
    
    def create(self, source: SourceCreate) -> Source:
        """Create a new source."""
        result = self.db.execute(
            text("""
                INSERT INTO sources (name, base_url, scraper_type, compliance_status, 
                                     tos_url, tos_notes, rate_limit_requests, rate_limit_period, is_active)
                VALUES (:name, :base_url, :scraper_type, :compliance_status,
                        :tos_url, :tos_notes, :rate_limit_requests, :rate_limit_period, :is_active)
                RETURNING *
            """),
            source.model_dump()
        )
        row = result.fetchone()
        self.db.commit()
        return Source(**row._asdict())
    
    def update(self, source_id: int, source: SourceUpdate) -> Optional[Source]:
        """Update an existing source."""
        updates = source.model_dump(exclude_unset=True)
        if not updates:
            return self.get_by_id(source_id)
        
        updates["id"] = source_id
        updates["updated_at"] = datetime.utcnow()
        
        set_clause = ", ".join(f"{k} = :{k}" for k in updates if k != "id")
        
        result = self.db.execute(
            text(f"UPDATE sources SET {set_clause} WHERE id = :id RETURNING *"),
            updates
        )
        row = result.fetchone()
        self.db.commit()
        return Source(**row._asdict()) if row else None


class JobRepository:
    """Repository for job CRUD operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, job_id: int) -> Optional[Job]:
        """Get a job by ID."""
        result = self.db.execute(
            text("SELECT * FROM jobs WHERE id = :id"),
            {"id": job_id}
        )
        row = result.fetchone()
        return Job(**row._asdict()) if row else None
    
    def get_by_hash(self, content_hash: str) -> Optional[Job]:
        """Get a job by content hash (for deduplication)."""
        result = self.db.execute(
            text("SELECT * FROM jobs WHERE content_hash = :hash"),
            {"hash": content_hash}
        )
        row = result.fetchone()
        return Job(**row._asdict()) if row else None
    
    def get_by_external_id(self, source_id: int, external_id: str) -> Optional[Job]:
        """Get a job by source and external ID."""
        result = self.db.execute(
            text("SELECT * FROM jobs WHERE source_id = :source_id AND external_id = :external_id"),
            {"source_id": source_id, "external_id": external_id}
        )
        row = result.fetchone()
        return Job(**row._asdict()) if row else None
    
    def search(
        self,
        query: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        source_id: Optional[int] = None,
        is_active: bool = True,
        redistributable_only: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> list[Job]:
        """
        Search for jobs with various filters.
        
        Args:
            query: Full-text search query
            company: Filter by company name
            location: Filter by location (partial match)
            source_id: Filter by source
            is_active: Only active jobs
            redistributable_only: Only jobs from approved sources
            limit: Max results
            offset: Pagination offset
            
        Returns:
            List of matching jobs
        """
        conditions = []
        params = {"limit": limit, "offset": offset}
        
        if is_active:
            conditions.append("j.is_active = true")
        
        if redistributable_only:
            conditions.append("s.compliance_status IN ('approved', 'conditional')")
        
        if company:
            conditions.append("j.company ILIKE :company")
            params["company"] = f"%{company}%"
        
        if location:
            conditions.append("j.location ILIKE :location")
            params["location"] = f"%{location}%"
        
        if source_id:
            conditions.append("j.source_id = :source_id")
            params["source_id"] = source_id
        
        if query:
            conditions.append("j.search_vector @@ plainto_tsquery('english', :query)")
            params["query"] = query
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"""
            SELECT j.* FROM jobs j
            JOIN sources s ON j.source_id = s.id
            WHERE {where_clause}
            ORDER BY j.last_seen_at DESC
            LIMIT :limit OFFSET :offset
        """
        
        result = self.db.execute(text(sql), params)
        return [Job(**row._asdict()) for row in result.fetchall()]
    
    def upsert(self, job: JobCreate) -> tuple[Job, bool]:
        """
        Insert or update a job.
        
        Uses content_hash for deduplication. If job exists, updates last_seen_at.
        
        Args:
            job: Job to insert or update
            
        Returns:
            Tuple of (job, is_new)
        """
        now = datetime.utcnow()
        
        # Check if job exists by content hash
        existing = self.get_by_hash(job.content_hash)
        
        if existing:
            # Update last_seen_at
            result = self.db.execute(
                text("""
                    UPDATE jobs 
                    SET last_seen_at = :now, is_active = true, updated_at = :now
                    WHERE id = :id
                    RETURNING *
                """),
                {"now": now, "id": existing.id}
            )
            row = result.fetchone()
            self.db.commit()
            return Job(**row._asdict()), False
        
        # Insert new job
        job_data = job.model_dump()
        job_data["first_seen_at"] = now
        job_data["last_seen_at"] = now
        
        result = self.db.execute(
            text("""
                INSERT INTO jobs (source_id, external_id, title, company, location, description,
                                 url, salary_min, salary_max, salary_currency, job_type, 
                                 experience_level, remote_type, content_hash, first_seen_at, last_seen_at)
                VALUES (:source_id, :external_id, :title, :company, :location, :description,
                        :url, :salary_min, :salary_max, :salary_currency, :job_type,
                        :experience_level, :remote_type, :content_hash, :first_seen_at, :last_seen_at)
                RETURNING *
            """),
            job_data
        )
        row = result.fetchone()
        self.db.commit()
        return Job(**row._asdict()), True
    
    def mark_inactive(self, source_id: int, seen_before: datetime) -> int:
        """
        Mark jobs as inactive if not seen since a given time.
        
        Args:
            source_id: Source to update
            seen_before: Mark inactive if last_seen_at < this time
            
        Returns:
            Number of jobs marked inactive
        """
        result = self.db.execute(
            text("""
                UPDATE jobs 
                SET is_active = false, updated_at = :now
                WHERE source_id = :source_id 
                AND last_seen_at < :seen_before
                AND is_active = true
            """),
            {"source_id": source_id, "seen_before": seen_before, "now": datetime.utcnow()}
        )
        self.db.commit()
        return result.rowcount
    
    def get_stats(self) -> dict:
        """Get job statistics."""
        result = self.db.execute(text("""
            SELECT 
                COUNT(*) as total_jobs,
                COUNT(*) FILTER (WHERE is_active) as active_jobs,
                COUNT(DISTINCT company) as unique_companies,
                COUNT(DISTINCT source_id) as sources_with_jobs
            FROM jobs
        """))
        row = result.fetchone()
        return row._asdict()
