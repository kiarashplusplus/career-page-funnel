"""Source model for tracking job data origins and compliance."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class ComplianceStatus(str, Enum):
    """Compliance status for a job source."""
    
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    PROHIBITED = "prohibited"
    PENDING_REVIEW = "pending_review"


class SourceBase(BaseModel):
    """Base source fields."""
    
    name: str = Field(..., min_length=1, max_length=255)
    base_url: str = Field(..., max_length=2000)
    scraper_type: str = Field(..., max_length=100)  # greenhouse, lever, ashby, workday, direct
    compliance_status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    tos_url: Optional[str] = Field(None, max_length=2000)
    tos_reviewed_at: Optional[datetime] = None
    tos_notes: Optional[str] = None
    rate_limit_requests: int = Field(default=10, ge=1, le=1000)
    rate_limit_period: int = Field(default=60, ge=1, le=3600)  # seconds
    is_active: bool = True


class SourceCreate(SourceBase):
    """Fields required when creating a new source."""
    pass


class SourceUpdate(BaseModel):
    """Fields that can be updated on an existing source."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    base_url: Optional[str] = Field(None, max_length=2000)
    compliance_status: Optional[ComplianceStatus] = None
    tos_url: Optional[str] = Field(None, max_length=2000)
    tos_reviewed_at: Optional[datetime] = None
    tos_notes: Optional[str] = None
    rate_limit_requests: Optional[int] = Field(None, ge=1, le=1000)
    rate_limit_period: Optional[int] = Field(None, ge=1, le=3600)
    is_active: Optional[bool] = None


class Source(SourceBase):
    """Full source model including database fields."""
    
    model_config = {"from_attributes": True}
    
    id: int
    created_at: datetime
    updated_at: datetime
