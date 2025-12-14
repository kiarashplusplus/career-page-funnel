"""Job model for the career page funnel."""

import hashlib
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class JobBase(BaseModel):
    """Base job fields shared across create/update/read."""
    
    title: str = Field(..., min_length=1, max_length=500)
    company: str = Field(..., min_length=1, max_length=255)
    location: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    url: str = Field(..., max_length=2000)
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    salary_currency: Optional[str] = Field(None, max_length=3)
    job_type: Optional[str] = Field(None, max_length=50)  # full-time, part-time, contract, internship
    experience_level: Optional[str] = Field(None, max_length=50)  # entry, mid, senior, lead, executive
    remote_type: Optional[str] = Field(None, max_length=50)  # remote, hybrid, onsite


class JobCreate(JobBase):
    """Fields required when creating a new job."""
    
    source_id: int
    external_id: Optional[str] = Field(None, max_length=255)
    
    @computed_field
    @property
    def content_hash(self) -> str:
        """Generate a hash for deduplication based on title, company, and location."""
        content = f"{self.title.lower()}|{self.company.lower()}|{(self.location or '').lower()}"
        return hashlib.sha256(content.encode()).hexdigest()[:64]


class JobUpdate(BaseModel):
    """Fields that can be updated on an existing job."""
    
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    location: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    salary_currency: Optional[str] = Field(None, max_length=3)
    job_type: Optional[str] = Field(None, max_length=50)
    experience_level: Optional[str] = Field(None, max_length=50)
    remote_type: Optional[str] = Field(None, max_length=50)
    is_active: Optional[bool] = None


class Job(JobBase):
    """Full job model including database fields."""
    
    model_config = {"from_attributes": True}
    
    id: int
    source_id: int
    external_id: Optional[str] = None
    content_hash: str
    is_active: bool = True
    first_seen_at: datetime
    last_seen_at: datetime
    created_at: datetime
    updated_at: datetime
