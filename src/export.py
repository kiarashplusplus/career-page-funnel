"""
Data export module for job database.

Supports exporting jobs to CSV and JSON formats with filtering options.
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Iterator, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from .models.job import Job

logger = logging.getLogger(__name__)


@dataclass
class ExportOptions:
    """Options for data export."""
    
    # Filters
    company: Optional[str] = None
    location: Optional[str] = None
    source_type: Optional[str] = None  # greenhouse, lever, amazon_jobs
    experience_level: Optional[str] = None  # entry, mid, senior
    job_type: Optional[str] = None  # full-time, part-time
    remote_type: Optional[str] = None  # remote, hybrid, onsite
    active_only: bool = True
    redistributable_only: bool = True
    
    # Date filters
    posted_after: Optional[datetime] = None
    posted_before: Optional[datetime] = None
    
    # Output options
    include_description: bool = False  # Descriptions can be large
    include_source_info: bool = True
    limit: Optional[int] = None


@dataclass
class ExportResult:
    """Result from an export operation."""
    
    format: str
    job_count: int
    file_path: Optional[Path] = None
    data: Optional[str] = None  # For in-memory exports
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    errors: list[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    def complete(self) -> "ExportResult":
        self.completed_at = datetime.utcnow()
        return self


# CSV columns for export
CSV_COLUMNS = [
    "id",
    "title",
    "company",
    "location",
    "url",
    "job_type",
    "experience_level",
    "remote_type",
    "salary_min",
    "salary_max",
    "salary_currency",
    "first_seen_at",
    "last_seen_at",
    "is_active",
]

CSV_COLUMNS_WITH_SOURCE = [
    "source_name",
    "source_type",
    "compliance_status",
]

CSV_COLUMNS_WITH_DESCRIPTION = [
    "description",
]


class JobExporter:
    """Export jobs from database to various formats."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def _build_query(self, options: ExportOptions) -> tuple[str, dict]:
        """Build SQL query from export options."""
        conditions = []
        params: dict[str, Any] = {}
        
        # Active filter
        if options.active_only:
            conditions.append("j.is_active = true")
        
        # Redistributable filter
        if options.redistributable_only:
            conditions.append("s.compliance_status IN ('approved', 'conditional')")
        
        # Company filter
        if options.company:
            conditions.append("j.company ILIKE :company")
            params["company"] = f"%{options.company}%"
        
        # Location filter
        if options.location:
            conditions.append("j.location ILIKE :location")
            params["location"] = f"%{options.location}%"
        
        # Source type filter
        if options.source_type:
            conditions.append("s.scraper_type = :source_type")
            params["source_type"] = options.source_type
        
        # Experience level filter
        if options.experience_level:
            conditions.append("j.experience_level = :experience_level")
            params["experience_level"] = options.experience_level
        
        # Job type filter
        if options.job_type:
            conditions.append("j.job_type = :job_type")
            params["job_type"] = options.job_type
        
        # Remote type filter
        if options.remote_type:
            conditions.append("j.remote_type = :remote_type")
            params["remote_type"] = options.remote_type
        
        # Date filters
        if options.posted_after:
            conditions.append("j.first_seen_at >= :posted_after")
            params["posted_after"] = options.posted_after
        
        if options.posted_before:
            conditions.append("j.first_seen_at <= :posted_before")
            params["posted_before"] = options.posted_before
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Build SELECT columns
        select_cols = """
            j.id, j.title, j.company, j.location, j.url,
            j.job_type, j.experience_level, j.remote_type,
            j.salary_min, j.salary_max, j.salary_currency,
            j.first_seen_at, j.last_seen_at, j.is_active,
            j.external_id, j.content_hash
        """
        
        if options.include_source_info:
            select_cols += """,
                s.name as source_name,
                s.scraper_type as source_type,
                s.compliance_status
            """
        
        if options.include_description:
            select_cols += ", j.description"
        
        sql = f"""
            SELECT {select_cols}
            FROM jobs j
            JOIN sources s ON j.source_id = s.id
            WHERE {where_clause}
            ORDER BY j.company, j.first_seen_at DESC
        """
        
        if options.limit:
            sql += f" LIMIT {options.limit}"
        
        return sql, params
    
    def _fetch_jobs(self, options: ExportOptions) -> Iterator[dict]:
        """Fetch jobs from database as dictionaries."""
        sql, params = self._build_query(options)
        
        result = self.db.execute(text(sql), params)
        
        for row in result:
            yield row._asdict()
    
    def count_jobs(self, options: ExportOptions) -> int:
        """Count jobs matching export options."""
        sql, params = self._build_query(options)
        
        # Wrap in COUNT query
        count_sql = f"SELECT COUNT(*) FROM ({sql}) as subq"
        
        result = self.db.execute(text(count_sql), params)
        return result.scalar() or 0
    
    def export_csv(
        self,
        options: ExportOptions,
        output_path: Optional[Path] = None,
    ) -> ExportResult:
        """
        Export jobs to CSV format.
        
        Args:
            options: Export options
            output_path: Path to write CSV file. If None, returns data in result.
            
        Returns:
            ExportResult with file path or data
        """
        result = ExportResult(format="csv", job_count=0)
        
        try:
            # Build column list
            columns = CSV_COLUMNS.copy()
            if options.include_source_info:
                columns.extend(CSV_COLUMNS_WITH_SOURCE)
            if options.include_description:
                columns.extend(CSV_COLUMNS_WITH_DESCRIPTION)
            
            # Create output
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                file = open(output_path, 'w', newline='', encoding='utf-8')
            else:
                file = StringIO()
            
            try:
                writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
                writer.writeheader()
                
                for job_dict in self._fetch_jobs(options):
                    # Format datetime fields (only if not already strings)
                    for dt_field in ['first_seen_at', 'last_seen_at']:
                        val = job_dict.get(dt_field)
                        if val and hasattr(val, 'isoformat'):
                            job_dict[dt_field] = val.isoformat()
                    
                    writer.writerow(job_dict)
                    result.job_count += 1
                
                if output_path:
                    result.file_path = output_path
                    logger.info(f"Exported {result.job_count} jobs to {output_path}")
                else:
                    result.data = file.getvalue()
                    
            finally:
                file.close()
                
        except Exception as e:
            error_msg = f"CSV export failed: {str(e)}"
            logger.exception(error_msg)
            result.errors.append(error_msg)
        
        return result.complete()
    
    def export_json(
        self,
        options: ExportOptions,
        output_path: Optional[Path] = None,
        pretty: bool = True,
    ) -> ExportResult:
        """
        Export jobs to JSON format.
        
        Args:
            options: Export options
            output_path: Path to write JSON file. If None, returns data in result.
            pretty: Pretty-print JSON output
            
        Returns:
            ExportResult with file path or data
        """
        result = ExportResult(format="json", job_count=0)
        
        try:
            jobs = []
            
            for job_dict in self._fetch_jobs(options):
                # Convert datetime fields to ISO format strings
                for key, value in job_dict.items():
                    if isinstance(value, datetime):
                        job_dict[key] = value.isoformat()
                
                jobs.append(job_dict)
                result.job_count += 1
            
            # Build export structure
            export_data = {
                "metadata": {
                    "exported_at": datetime.utcnow().isoformat(),
                    "job_count": result.job_count,
                    "filters": {
                        "company": options.company,
                        "location": options.location,
                        "source_type": options.source_type,
                        "experience_level": options.experience_level,
                        "active_only": options.active_only,
                        "redistributable_only": options.redistributable_only,
                    },
                    "source": "career-page-funnel",
                    "license": "Data from compliant sources only - see individual source compliance status",
                },
                "jobs": jobs,
            }
            
            # Serialize
            indent = 2 if pretty else None
            json_str = json.dumps(export_data, indent=indent, ensure_ascii=False)
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json_str, encoding='utf-8')
                result.file_path = output_path
                logger.info(f"Exported {result.job_count} jobs to {output_path}")
            else:
                result.data = json_str
                
        except Exception as e:
            error_msg = f"JSON export failed: {str(e)}"
            logger.exception(error_msg)
            result.errors.append(error_msg)
        
        return result.complete()
    
    def export_jsonl(
        self,
        options: ExportOptions,
        output_path: Optional[Path] = None,
    ) -> ExportResult:
        """
        Export jobs to JSON Lines format (one JSON object per line).
        
        Good for streaming large datasets.
        
        Args:
            options: Export options
            output_path: Path to write JSONL file. If None, returns data in result.
            
        Returns:
            ExportResult with file path or data
        """
        result = ExportResult(format="jsonl", job_count=0)
        
        try:
            lines = []
            
            for job_dict in self._fetch_jobs(options):
                # Convert datetime fields to ISO format strings
                for key, value in job_dict.items():
                    if isinstance(value, datetime):
                        job_dict[key] = value.isoformat()
                
                lines.append(json.dumps(job_dict, ensure_ascii=False))
                result.job_count += 1
            
            jsonl_str = "\n".join(lines)
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(jsonl_str, encoding='utf-8')
                result.file_path = output_path
                logger.info(f"Exported {result.job_count} jobs to {output_path}")
            else:
                result.data = jsonl_str
                
        except Exception as e:
            error_msg = f"JSONL export failed: {str(e)}"
            logger.exception(error_msg)
            result.errors.append(error_msg)
        
        return result.complete()
    
    def get_export_summary(self, options: ExportOptions) -> dict:
        """Get a summary of what would be exported with given options."""
        count = self.count_jobs(options)
        
        # Get breakdown by company
        sql, params = self._build_query(options)
        company_sql = f"""
            SELECT company, COUNT(*) as count 
            FROM ({sql}) as subq 
            GROUP BY company 
            ORDER BY count DESC 
            LIMIT 10
        """
        
        result = self.db.execute(text(company_sql), params)
        top_companies = [{"company": row[0], "count": row[1]} for row in result]
        
        # Get breakdown by experience level
        level_sql = f"""
            SELECT experience_level, COUNT(*) as count 
            FROM ({sql}) as subq 
            GROUP BY experience_level 
            ORDER BY count DESC
        """
        
        result = self.db.execute(text(level_sql), params)
        by_level = {row[0] or "unknown": row[1] for row in result}
        
        return {
            "total_jobs": count,
            "top_companies": top_companies,
            "by_experience_level": by_level,
            "filters_applied": {
                k: v for k, v in options.__dict__.items() 
                if v is not None and v is not True and v is not False
            },
        }


def export_jobs_to_csv(
    db: Session,
    output_path: str,
    company: Optional[str] = None,
    include_description: bool = False,
) -> ExportResult:
    """
    Convenience function to export jobs to CSV.
    
    Args:
        db: Database session
        output_path: Path to write CSV file
        company: Optional company filter
        include_description: Include job descriptions
        
    Returns:
        ExportResult
    """
    exporter = JobExporter(db)
    options = ExportOptions(
        company=company,
        include_description=include_description,
    )
    return exporter.export_csv(options, Path(output_path))


def export_jobs_to_json(
    db: Session,
    output_path: str,
    company: Optional[str] = None,
    include_description: bool = False,
) -> ExportResult:
    """
    Convenience function to export jobs to JSON.
    
    Args:
        db: Database session
        output_path: Path to write JSON file
        company: Optional company filter
        include_description: Include job descriptions
        
    Returns:
        ExportResult
    """
    exporter = JobExporter(db)
    options = ExportOptions(
        company=company,
        include_description=include_description,
    )
    return exporter.export_json(options, Path(output_path))
