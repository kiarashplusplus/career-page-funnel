-- Initialize the career-page-funnel database
-- This script runs automatically when the PostgreSQL container starts

-- Source registry for compliance tracking
CREATE TABLE IF NOT EXISTS sources (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL CHECK (source_type IN ('ats', 'direct', 'api', 'curated')),
    
    -- Compliance
    compliance_status TEXT NOT NULL CHECK (compliance_status IN ('approved', 'conditional', 'prohibited')),
    tos_url TEXT,
    robots_txt_allows BOOLEAN,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    
    -- Attribution (for conditional sources)
    requires_attribution BOOLEAN DEFAULT FALSE,
    attribution_text TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Jobs table with compliance tracking
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    
    -- Job data
    external_id TEXT,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    description TEXT,
    url TEXT NOT NULL,
    posted_at TIMESTAMP WITH TIME ZONE,
    
    -- Salary (if available)
    salary_min INTEGER,
    salary_max INTEGER,
    salary_currency TEXT DEFAULT 'USD',
    
    -- Classification
    experience_level TEXT CHECK (experience_level IN ('entry', 'mid', 'senior', 'lead', 'executive')),
    job_type TEXT CHECK (job_type IN ('full-time', 'part-time', 'contract', 'internship', 'temporary')),
    is_remote BOOLEAN,
    
    -- Deduplication
    content_hash TEXT NOT NULL,
    
    -- Metadata
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    UNIQUE(source_id, external_id)
);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_jobs_fts ON jobs USING GIN (
    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(company, '') || ' ' || coalesce(location, '') || ' ' || coalesce(description, ''))
);

-- Additional indexes for fast filtering
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);
CREATE INDEX IF NOT EXISTS idx_jobs_level ON jobs(experience_level);
CREATE INDEX IF NOT EXISTS idx_jobs_posted ON jobs(posted_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_active ON jobs(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_jobs_content_hash ON jobs(content_hash);

-- ============================================================================
-- APPROVED SOURCES - Public APIs designed for job aggregation
-- ============================================================================
INSERT INTO sources (name, base_url, source_type, compliance_status, tos_url, robots_txt_allows, reviewed_at) VALUES
    -- ATS Platforms with Public APIs
    ('Greenhouse', 'https://boards-api.greenhouse.io', 'ats', 'approved', 
     'https://www.greenhouse.io/legal/api-terms', TRUE, NOW()),
    ('Lever', 'https://api.lever.co', 'ats', 'approved', 
     'https://www.lever.co/terms', TRUE, NOW()),
    ('Ashby', 'https://api.ashbyhq.com', 'ats', 'approved', 
     'https://www.ashbyhq.com/legal/terms', TRUE, NOW()),
    
    -- Company career pages (manually verified)
    ('SimplifyJobs', 'https://github.com/SimplifyJobs', 'curated', 'approved', 
     NULL, TRUE, NOW())
ON CONFLICT (base_url) DO NOTHING;

-- ============================================================================
-- CONDITIONAL SOURCES - Require API registration or customer consent
-- ============================================================================
INSERT INTO sources (name, base_url, source_type, compliance_status, tos_url, robots_txt_allows, requires_attribution, reviewed_at) VALUES
    ('SmartRecruiters', 'https://api.smartrecruiters.com', 'ats', 'conditional', 
     'https://developers.smartrecruiters.com/terms', TRUE, TRUE, NOW()),
    ('JazzHR', 'https://api.resumatorapi.com', 'ats', 'conditional', 
     'https://www.jazzhr.com/terms', TRUE, TRUE, NOW()),
    ('BambooHR', 'https://api.bamboohr.com', 'ats', 'conditional', 
     'https://www.bamboohr.com/terms', TRUE, TRUE, NOW())
ON CONFLICT (base_url) DO NOTHING;

-- ============================================================================
-- PROHIBITED SOURCES - Explicitly prohibit scraping/redistribution
-- ============================================================================
INSERT INTO sources (name, base_url, source_type, compliance_status, tos_url, robots_txt_allows, reviewed_at) VALUES
    -- Major job boards that prohibit aggregation
    ('LinkedIn', 'https://linkedin.com', 'api', 'prohibited', 
     'https://www.linkedin.com/legal/user-agreement', FALSE, NOW()),
    ('Indeed', 'https://indeed.com', 'api', 'prohibited', 
     'https://www.indeed.com/legal', FALSE, NOW()),
    ('Glassdoor', 'https://glassdoor.com', 'api', 'prohibited', 
     'https://www.glassdoor.com/about/terms.htm', FALSE, NOW()),
    
    -- ATS Platforms without public APIs
    ('iCIMS', 'https://icims.com', 'ats', 'prohibited', 
     'https://www.icims.com/legal/terms-of-use', TRUE, NOW()),
    ('Workday', 'https://myworkdayjobs.com', 'ats', 'prohibited', 
     'https://www.workday.com/en-us/legal/service-terms.html', FALSE, NOW())
ON CONFLICT (base_url) DO NOTHING;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for sources table
DROP TRIGGER IF EXISTS update_sources_updated_at ON sources;
CREATE TRIGGER update_sources_updated_at
    BEFORE UPDATE ON sources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- View for redistributable jobs only
CREATE OR REPLACE VIEW redistributable_jobs AS
SELECT j.*, s.name as source_name, s.compliance_status, s.requires_attribution, s.attribution_text
FROM jobs j
JOIN sources s ON j.source_id = s.id
WHERE s.compliance_status IN ('approved', 'conditional')
  AND j.is_active = TRUE;

COMMENT ON VIEW redistributable_jobs IS 'Only jobs from compliant sources that can be legally redistributed';
