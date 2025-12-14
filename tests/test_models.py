"""Basic tests for the career page funnel."""

import pytest
from src.models.job import JobCreate, Job
from src.models.source import Source, ComplianceStatus


class TestJobModel:
    """Tests for the Job model."""
    
    def test_job_create_generates_content_hash(self):
        """Test that JobCreate generates a content hash."""
        job = JobCreate(
            source_id=1,
            title="Software Engineer",
            company="Test Corp",
            location="San Francisco, CA",
            url="https://example.com/jobs/1",
        )
        
        assert job.content_hash is not None
        assert len(job.content_hash) == 64
    
    def test_same_job_generates_same_hash(self):
        """Test that identical jobs generate the same hash."""
        job1 = JobCreate(
            source_id=1,
            title="Software Engineer",
            company="Test Corp",
            location="San Francisco, CA",
            url="https://example.com/jobs/1",
        )
        
        job2 = JobCreate(
            source_id=2,  # Different source
            title="Software Engineer",
            company="Test Corp",
            location="San Francisco, CA",
            url="https://example.com/jobs/2",  # Different URL
        )
        
        # Hash should be the same because title, company, location are the same
        assert job1.content_hash == job2.content_hash
    
    def test_different_jobs_generate_different_hashes(self):
        """Test that different jobs generate different hashes."""
        job1 = JobCreate(
            source_id=1,
            title="Software Engineer",
            company="Test Corp",
            location="San Francisco, CA",
            url="https://example.com/jobs/1",
        )
        
        job2 = JobCreate(
            source_id=1,
            title="Senior Software Engineer",  # Different title
            company="Test Corp",
            location="San Francisco, CA",
            url="https://example.com/jobs/2",
        )
        
        assert job1.content_hash != job2.content_hash


class TestSourceModel:
    """Tests for the Source model."""
    
    def test_compliance_status_enum(self):
        """Test compliance status enum values."""
        assert ComplianceStatus.APPROVED.value == "approved"
        assert ComplianceStatus.CONDITIONAL.value == "conditional"
        assert ComplianceStatus.PROHIBITED.value == "prohibited"
        assert ComplianceStatus.PENDING_REVIEW.value == "pending_review"


# Run with: pytest tests/ -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
