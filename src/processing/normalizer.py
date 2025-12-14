"""
Job data normalization for consistent storage and search.

This module normalizes:
1. Job titles: Case, abbreviations, common variations
2. Locations: City/state/country standardization
3. Salaries: Currency conversion, period normalization
4. Company names: Suffixes, variations
"""

import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SalaryPeriod(str, Enum):
    """Salary period for normalization."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    UNKNOWN = "unknown"


@dataclass
class NormalizedSalary:
    """Normalized salary information."""
    min_yearly: Optional[int]  # Annual salary in USD
    max_yearly: Optional[int]
    original_min: Optional[float]
    original_max: Optional[float]
    original_period: SalaryPeriod
    original_currency: str
    is_estimated: bool  # True if converted from hourly/monthly
    
    def __repr__(self) -> str:
        if self.min_yearly and self.max_yearly:
            return f"${self.min_yearly:,} - ${self.max_yearly:,}/year"
        elif self.min_yearly:
            return f"${self.min_yearly:,}+/year"
        elif self.max_yearly:
            return f"Up to ${self.max_yearly:,}/year"
        return "Not specified"


@dataclass
class NormalizedLocation:
    """Normalized location information."""
    city: Optional[str]
    state: Optional[str]
    state_code: Optional[str]
    country: Optional[str]
    country_code: Optional[str]
    is_remote: bool
    original: str
    
    def __repr__(self) -> str:
        parts = []
        if self.city:
            parts.append(self.city)
        if self.state_code:
            parts.append(self.state_code)
        elif self.state:
            parts.append(self.state)
        if self.country_code and self.country_code != "US":
            parts.append(self.country_code)
        
        if self.is_remote:
            if parts:
                return f"Remote ({', '.join(parts)})"
            return "Remote"
        
        return ", ".join(parts) if parts else self.original


# US State mappings
US_STATES = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
    'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
    'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
    'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
    'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
    'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
    'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
    'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
    'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
    'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC',
}

# Reverse lookup: code to name
US_STATE_CODES = {v: k.title() for k, v in US_STATES.items()}

# Common city name normalizations
CITY_NORMALIZATIONS = {
    'sf': 'San Francisco',
    'la': 'Los Angeles',
    'nyc': 'New York City',
    'ny': 'New York',
    'dc': 'Washington',
    'philly': 'Philadelphia',
    'chi': 'Chicago',
    'atl': 'Atlanta',
    'sea': 'Seattle',
    'den': 'Denver',
    'bos': 'Boston',
    'sfo': 'San Francisco',
    'lax': 'Los Angeles',
}

# Title abbreviation expansions
TITLE_ABBREVIATIONS = {
    'sr': 'Senior',
    'sr.': 'Senior',
    'jr': 'Junior',
    'jr.': 'Junior',
    'mgr': 'Manager',
    'eng': 'Engineer',
    'engr': 'Engineer',
    'dev': 'Developer',
    'swe': 'Software Engineer',
    'sde': 'Software Development Engineer',
    'pm': 'Product Manager',
    'tpm': 'Technical Program Manager',
    'em': 'Engineering Manager',
    'vp': 'Vice President',
    'svp': 'Senior Vice President',
    'evp': 'Executive Vice President',
    'cto': 'Chief Technology Officer',
    'cio': 'Chief Information Officer',
    'ceo': 'Chief Executive Officer',
    'cfo': 'Chief Financial Officer',
    'ml': 'Machine Learning',
    'ai': 'AI',
    'ui': 'UI',
    'ux': 'UX',
    'qa': 'QA',
    'ops': 'Operations',
    'devops': 'DevOps',
    'sre': 'Site Reliability Engineer',
    'dba': 'Database Administrator',
    'sysadmin': 'System Administrator',
}

# Company suffix normalizations (to remove)
COMPANY_SUFFIXES = [
    ', inc.', ', inc', ' inc.', ' inc',
    ', llc', ' llc',
    ', ltd', ' ltd', ', limited', ' limited',
    ', corp.', ', corp', ' corp.', ' corp', ', corporation', ' corporation',
    ', co.', ', co', ' co.', ' co',
    ', plc', ' plc',
    ', gmbh', ' gmbh',
    ', ag', ' ag',
    ', sa', ' sa',
    ', pty', ' pty',
]

# Currency conversion rates (approximate, USD as base)
CURRENCY_RATES = {
    'USD': 1.0,
    '$': 1.0,
    'EUR': 1.10,
    '€': 1.10,
    'GBP': 1.27,
    '£': 1.27,
    'CAD': 0.74,
    'C$': 0.74,
    'AUD': 0.66,
    'A$': 0.66,
    'CHF': 1.12,
    'JPY': 0.0067,
    '¥': 0.0067,
    'INR': 0.012,
    '₹': 0.012,
}

# Period multipliers to convert to yearly
PERIOD_MULTIPLIERS = {
    SalaryPeriod.HOURLY: 2080,   # 40 hours * 52 weeks
    SalaryPeriod.DAILY: 260,     # 5 days * 52 weeks
    SalaryPeriod.WEEKLY: 52,
    SalaryPeriod.MONTHLY: 12,
    SalaryPeriod.YEARLY: 1,
    SalaryPeriod.UNKNOWN: 1,     # Assume yearly if unknown
}


class JobNormalizer:
    """
    Normalize job data for consistent storage and search.
    
    Example:
        normalizer = JobNormalizer()
        
        # Normalize a title
        title = normalizer.normalize_title("Sr. SWE")
        print(title)  # "Senior Software Engineer"
        
        # Normalize a location
        location = normalizer.normalize_location("SF, CA")
        print(location)  # NormalizedLocation(city="San Francisco", state_code="CA", ...)
        
        # Normalize a salary
        salary = normalizer.normalize_salary(50, 75, "hourly", "USD")
        print(salary)  # "$104,000 - $156,000/year"
    """
    
    def __init__(
        self,
        expand_abbreviations: bool = True,
        normalize_case: bool = True,
    ):
        """
        Initialize the normalizer.
        
        Args:
            expand_abbreviations: Expand common abbreviations in titles
            normalize_case: Apply title case to job titles
        """
        self.expand_abbreviations = expand_abbreviations
        self.normalize_case = normalize_case
    
    def normalize_title(self, title: str) -> str:
        """
        Normalize a job title.
        
        - Expands abbreviations (Sr -> Senior)
        - Applies consistent casing
        - Removes extra whitespace
        
        Args:
            title: Raw job title
            
        Returns:
            Normalized job title
        """
        if not title:
            return ""
        
        # Strip and collapse whitespace
        normalized = ' '.join(title.split())
        
        # Expand abbreviations
        if self.expand_abbreviations:
            words = normalized.split()
            expanded_words = []
            
            for word in words:
                word_lower = word.lower().rstrip('.,')
                if word_lower in TITLE_ABBREVIATIONS:
                    expanded_words.append(TITLE_ABBREVIATIONS[word_lower])
                else:
                    expanded_words.append(word)
            
            normalized = ' '.join(expanded_words)
        
        # Apply title case (preserve acronyms)
        if self.normalize_case:
            normalized = self._smart_title_case(normalized)
        
        return normalized
    
    def _smart_title_case(self, text: str) -> str:
        """Apply title case while preserving acronyms like UI, UX, AI, ML."""
        # Common acronyms to preserve
        acronyms = {'UI', 'UX', 'AI', 'ML', 'API', 'SDK', 'QA', 'DevOps', 'SRE', 
                   'AWS', 'GCP', 'iOS', 'SQL', 'NoSQL', 'ETL', 'CI', 'CD'}
        
        words = text.split()
        result = []
        
        for word in words:
            # Check if it's an acronym (keep uppercase)
            if word.upper() in acronyms:
                result.append(word.upper())
            # Roman numerals
            elif re.match(r'^[IVX]+$', word.upper()):
                result.append(word.upper())
            else:
                result.append(word.capitalize())
        
        return ' '.join(result)
    
    def normalize_location(self, location: str) -> NormalizedLocation:
        """
        Normalize a location string.
        
        Parses and normalizes city, state, country information.
        Detects remote status.
        
        Args:
            location: Raw location string
            
        Returns:
            NormalizedLocation with parsed components
        """
        if not location:
            return NormalizedLocation(
                city=None, state=None, state_code=None,
                country=None, country_code=None,
                is_remote=False, original=""
            )
        
        original = location
        location = location.strip()
        
        # Detect remote
        is_remote = bool(re.search(
            r'\b(remote|work from home|wfh|distributed|anywhere)\b',
            location, re.IGNORECASE
        ))
        
        # Remove remote indicators for parsing
        location_clean = re.sub(
            r'\b(remote|work from home|wfh|distributed|anywhere)\b',
            '', location, flags=re.IGNORECASE
        ).strip(' ,/-')
        
        # Parse location parts
        city, state, state_code, country, country_code = self._parse_location_parts(location_clean)
        
        return NormalizedLocation(
            city=city,
            state=state,
            state_code=state_code,
            country=country,
            country_code=country_code,
            is_remote=is_remote,
            original=original,
        )
    
    def _parse_location_parts(
        self, location: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse location into city, state, country components."""
        if not location:
            return None, None, None, None, None
        
        # Split by comma, slash, or dash
        parts = [p.strip() for p in re.split(r'[,/\-]', location) if p.strip()]
        
        city = None
        state = None
        state_code = None
        country = None
        country_code = None
        
        for part in parts:
            part_lower = part.lower()
            part_upper = part.upper()
            
            # Check for city abbreviation
            if part_lower in CITY_NORMALIZATIONS:
                city = CITY_NORMALIZATIONS[part_lower]
                continue
            
            # Check for US state code (2 letters)
            if len(part) == 2 and part_upper in US_STATE_CODES:
                state_code = part_upper
                state = US_STATE_CODES[part_upper]
                continue
            
            # Check for full US state name
            if part_lower in US_STATES:
                state_code = US_STATES[part_lower]
                state = part.title()
                continue
            
            # Check for country codes/names
            if part_upper in ('US', 'USA', 'UNITED STATES'):
                country = 'United States'
                country_code = 'US'
                continue
            
            if part_upper in ('UK', 'UNITED KINGDOM', 'GB', 'GBR'):
                country = 'United Kingdom'
                country_code = 'UK'
                continue
            
            if part_upper in ('CA', 'CANADA') and not state_code:
                # CA could be California or Canada
                if state_code is None:
                    country = 'Canada'
                    country_code = 'CA'
                continue
            
            # Otherwise assume it's a city
            if city is None and len(part) > 2:
                city = part.title()
        
        # Default country to US if we have a US state
        if state_code and state_code in US_STATE_CODES and not country:
            country = 'United States'
            country_code = 'US'
        
        return city, state, state_code, country, country_code
    
    def normalize_salary(
        self,
        min_salary: Optional[float],
        max_salary: Optional[float],
        period: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> NormalizedSalary:
        """
        Normalize salary to annual USD.
        
        Args:
            min_salary: Minimum salary
            max_salary: Maximum salary
            period: Pay period (hourly, monthly, yearly, etc.)
            currency: Currency code or symbol
            
        Returns:
            NormalizedSalary with annual USD values
        """
        # Parse period
        salary_period = self._parse_period(period)
        
        # Parse currency
        currency = currency.upper() if currency else 'USD'
        if currency not in CURRENCY_RATES:
            currency = 'USD'
        
        # Calculate yearly USD values
        min_yearly = self._to_yearly_usd(min_salary, salary_period, currency)
        max_yearly = self._to_yearly_usd(max_salary, salary_period, currency)
        
        # Ensure min <= max
        if min_yearly and max_yearly and min_yearly > max_yearly:
            min_yearly, max_yearly = max_yearly, min_yearly
        
        return NormalizedSalary(
            min_yearly=min_yearly,
            max_yearly=max_yearly,
            original_min=min_salary,
            original_max=max_salary,
            original_period=salary_period,
            original_currency=currency,
            is_estimated=salary_period != SalaryPeriod.YEARLY,
        )
    
    def _parse_period(self, period: Optional[str]) -> SalaryPeriod:
        """Parse salary period string."""
        if not period:
            return SalaryPeriod.UNKNOWN
        
        period = period.lower()
        
        if any(p in period for p in ('hour', 'hr', '/h')):
            return SalaryPeriod.HOURLY
        if any(p in period for p in ('day', '/d')):
            return SalaryPeriod.DAILY
        if any(p in period for p in ('week', '/w', 'wk')):
            return SalaryPeriod.WEEKLY
        if any(p in period for p in ('month', '/m', 'mo')):
            return SalaryPeriod.MONTHLY
        if any(p in period for p in ('year', 'annual', '/y', 'yr')):
            return SalaryPeriod.YEARLY
        
        return SalaryPeriod.UNKNOWN
    
    def _to_yearly_usd(
        self,
        amount: Optional[float],
        period: SalaryPeriod,
        currency: str,
    ) -> Optional[int]:
        """Convert salary to yearly USD."""
        if amount is None or amount <= 0:
            return None
        
        try:
            # Convert to USD
            usd_amount = amount * CURRENCY_RATES.get(currency, 1.0)
            
            # Convert to yearly
            yearly = usd_amount * PERIOD_MULTIPLIERS[period]
            
            # Round to nearest 1000
            return int(round(yearly / 1000) * 1000)
        except (InvalidOperation, OverflowError):
            return None
    
    def normalize_company(self, company: str) -> str:
        """
        Normalize a company name.
        
        - Removes suffixes (Inc., LLC, etc.)
        - Applies consistent casing
        - Removes extra whitespace
        
        Args:
            company: Raw company name
            
        Returns:
            Normalized company name
        """
        if not company:
            return ""
        
        # Strip and collapse whitespace
        normalized = ' '.join(company.split())
        normalized_lower = normalized.lower()
        
        # Remove common suffixes
        for suffix in COMPANY_SUFFIXES:
            if normalized_lower.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
                normalized_lower = normalized.lower()
        
        return normalized
    
    def extract_salary_from_text(self, text: str) -> Optional[NormalizedSalary]:
        """
        Extract and normalize salary from job description text.
        
        Looks for patterns like:
        - $50,000 - $75,000
        - $50k - $75k
        - $50-75 per hour
        - 100,000 - 150,000 USD
        
        Args:
            text: Job description or salary string
            
        Returns:
            NormalizedSalary if found, None otherwise
        """
        if not text:
            return None
        
        # Patterns for salary extraction
        patterns = [
            # $50,000 - $75,000 / year
            r'\$\s*([\d,]+)\s*(?:k|K)?\s*[-–to]+\s*\$?\s*([\d,]+)\s*(?:k|K)?\s*(?:per\s+)?(year|annual|month|hour|hr)?',
            # $50k - $75k
            r'\$\s*([\d.]+)\s*(k|K)\s*[-–to]+\s*\$?\s*([\d.]+)\s*(k|K)',
            # 50,000 - 75,000 USD
            r'([\d,]+)\s*[-–to]+\s*([\d,]+)\s*(USD|EUR|GBP|CAD)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                try:
                    min_val = self._parse_salary_value(groups[0], 'k' in groups[1].lower() if len(groups) > 1 and groups[1] else False)
                    
                    if len(groups) >= 3:
                        max_idx = 2 if len(groups) > 3 else 1
                        is_k = len(groups) > 3 and groups[3] and 'k' in groups[3].lower()
                        max_val = self._parse_salary_value(groups[max_idx], is_k)
                    else:
                        max_val = None
                    
                    # Detect period
                    period = SalaryPeriod.YEARLY
                    period_text = groups[-1] if groups[-1] else ''
                    if 'hour' in period_text.lower() or 'hr' in period_text.lower():
                        period = SalaryPeriod.HOURLY
                    elif 'month' in period_text.lower():
                        period = SalaryPeriod.MONTHLY
                    
                    return self.normalize_salary(min_val, max_val, period.value)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _parse_salary_value(self, value: str, is_k: bool = False) -> float:
        """Parse a salary value string."""
        # Remove commas and whitespace
        clean = value.replace(',', '').replace(' ', '')
        amount = float(clean)
        
        # Apply k multiplier
        if is_k:
            amount *= 1000
        
        return amount


# Convenience function for CLI/testing
if __name__ == "__main__":
    normalizer = JobNormalizer()
    
    print("📝 Title Normalization Demo\n")
    titles = [
        "Sr. SWE",
        "junior dev",
        "SENIOR ENGINEER",
        "ml eng",
        "Staff SRE",
        "vp of engineering",
    ]
    for title in titles:
        print(f"  '{title}' -> '{normalizer.normalize_title(title)}'")
    
    print("\n📍 Location Normalization Demo\n")
    locations = [
        "SF, CA",
        "New York, NY",
        "Remote - US",
        "San Francisco, California, USA",
        "London, UK",
        "Remote",
        "Austin, TX, United States",
    ]
    for loc in locations:
        result = normalizer.normalize_location(loc)
        print(f"  '{loc}' -> {result}")
    
    print("\n💰 Salary Normalization Demo\n")
    salaries = [
        (50, 75, "hourly", "USD"),
        (100000, 150000, "yearly", "USD"),
        (80000, 100000, "yearly", "EUR"),
        (5000, 7000, "monthly", "USD"),
    ]
    for min_s, max_s, period, currency in salaries:
        result = normalizer.normalize_salary(min_s, max_s, period, currency)
        print(f"  {min_s}-{max_s} {period} {currency} -> {result}")
    
    print("\n🏢 Company Normalization Demo\n")
    companies = [
        "Acme, Inc.",
        "Google LLC",
        "Microsoft Corporation",
        "Apple Inc",
        "Meta Platforms, Inc.",
    ]
    for company in companies:
        print(f"  '{company}' -> '{normalizer.normalize_company(company)}'")
    
    print("\n📄 Salary Extraction Demo\n")
    texts = [
        "Salary: $120,000 - $180,000 per year",
        "Compensation: $50-75 per hour",
        "Pay range: $80k - $120k annually",
        "100,000 - 150,000 USD",
    ]
    for text in texts:
        result = normalizer.extract_salary_from_text(text)
        print(f"  '{text}' -> {result}")
