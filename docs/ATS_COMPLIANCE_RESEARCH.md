# ATS Platform ToS/API Compliance Research

**Research Date:** December 14, 2025  
**Purpose:** Evaluate compliance requirements for fetching job listings from various ATS platforms

---

## Executive Summary

| Platform | Public API | Scraping Policy | Verdict |
|----------|------------|-----------------|---------|
| **Ashby** | ✅ Yes (Posting API) | Requires partnership | **APPROVED** |
| **SmartRecruiters** | ✅ Yes (Job Ads API) | Requires API agreement | **CONDITIONAL** |
| **iCIMS** | ⚠️ Partner Only | Explicitly prohibited | **PROHIBITED** |
| **BambooHR** | ⚠️ Limited/Partner | No public job API | **CONDITIONAL** |
| **JazzHR** | ✅ Yes (Resumator API) | API key required | **CONDITIONAL** |
| **Workday** | ⚠️ Partner Only | Explicitly prohibited | **PROHIBITED** |

---

## 1. Ashby (jobs.ashbyhq.com / api.ashbyhq.com)

### Public API Availability
**YES** - Ashby provides a **Posting API** specifically designed for job board integrations.

**API Endpoint:**
```
POST https://api.ashbyhq.com/posting-api/job-board/{companyIdentifier}
```

**Documentation:** https://developers.ashbyhq.com/docs/posting-api-overview

### Key API Features
- `jobPosting.list` - List all published job postings
- `jobPosting.info` - Get details for a specific posting  
- `application.submit` - Submit applications via API
- Webhooks for `jobPostingPublish`, `jobPostingUnpublish`, `jobPostingUpdate`

### Terms of Service Summary
- Ashby has comprehensive API documentation on developers.ashbyhq.com
- The **Posting API** is specifically designed for job boards and aggregators
- API access requires authentication (Basic Auth)
- Rate limits apply

### Compliance Verdict: **APPROVED**
**Rationale:** Ashby explicitly provides a Posting API for job aggregation use cases. Using the official API with proper authentication is compliant with their intended usage patterns.

**Requirements:**
- Use the official Posting API
- Respect rate limits
- Include proper attribution/links back to Ashby job pages

---

## 2. SmartRecruiters (api.smartrecruiters.com)

### Public API Availability
**YES** - SmartRecruiters provides a comprehensive Job Ads API.

**API Endpoint:**
```
GET https://api.smartrecruiters.com/v1/companies/{companyId}/postings
```

**Documentation:** https://developers.smartrecruiters.com/reference

### Key API Features
- Job Ad API for fetching public postings
- Apply API for application submission
- Posting Management API
- Webhooks for job lifecycle events

### Terms of Service Summary
- SmartRecruiters requires API agreement for access
- The Job Ads API is designed for job board distribution
- Must comply with their Developer Terms of Use
- Rate limits and quotas apply

### Compliance Verdict: **CONDITIONAL**
**Rationale:** SmartRecruiters provides a Job Ads API, but requires registration and agreement to developer terms.

**Requirements:**
- Register for API access at developers.smartrecruiters.com
- Agree to Developer Terms of Use
- Use API keys for authentication
- Respect rate limits
- Attribution required

---

## 3. iCIMS (icims.com career pages)

### Public API Availability
**NO PUBLIC API** - iCIMS does not provide a public API for fetching job listings.

**Developer Program:** https://www.icims.com/products/platform/developer-program/

### Terms of Service Key Points
From iCIMS Terms of Use:
- **Explicitly prohibits** automated access: "Use any robot, spider or other automatic device, process, or means to access the Website"
- **Prohibits** copying/monitoring: "Use any manual process to monitor or copy any of the material on the Website"
- **No unauthorized access**: "Attempt to gain unauthorized access to, interfere with, damage or disrupt any parts of the Website"

### Compliance Verdict: **PROHIBITED**
**Rationale:** iCIMS explicitly prohibits scraping and automated data collection in their Terms of Use. They offer a partner Developer Program, but this requires a commercial relationship.

**Alternative Path:**
- Apply for iCIMS Developer Partner Program
- Negotiate formal data sharing agreement
- Use their official job board distribution partners

---

## 4. BambooHR (bamboohr.com job embeds)

### Public API Availability
**LIMITED** - BambooHR has an API but job listings access is restricted.

**API Documentation:** https://documentation.bamboohr.com/reference/get-job-summaries

### Key Points
- API requires authentication with company-level API keys
- Job-related endpoints are primarily for **customers' own data**
- No public job aggregation API is provided
- Career page embeds are intended for company websites only

### Terms of Service Summary
- API access is intended for BambooHR customers
- Third-party aggregation not explicitly supported
- Job embeds are JavaScript widgets meant for direct embedding

### Compliance Verdict: **CONDITIONAL**
**Rationale:** BambooHR does not provide a public job aggregation API. Access requires customer API keys, making third-party aggregation impractical without explicit partnership.

**Alternative Path:**
- Partner with BambooHR for job distribution
- Use official job board integration partners
- Companies can voluntarily share their API keys

---

## 5. JazzHR (app.jazz.co / resumatorapi.com)

### Public API Availability
**YES** - JazzHR provides the Resumator API for job data access.

**API Endpoint:**
```
GET https://api.resumatorapi.com/v1/jobs
GET https://api.resumatorapi.com/v1/jobs/{id}
```

**Documentation:** https://www.resumatorapi.com/v1/

### Key API Features
- List all jobs
- Get job details by ID
- Application submission
- Swagger/OpenAPI documentation available

### Terms of Service Summary
- API access requires API key from JazzHR customer account
- Designed for integrations with job boards
- Standard API usage terms apply

### Compliance Verdict: **CONDITIONAL**
**Rationale:** JazzHR provides a documented API, but access requires customer API keys. Aggregation requires partnership or customer consent.

**Requirements:**
- Obtain API key through JazzHR customer or partnership
- Follow API documentation
- Respect rate limits
- Attribution to JazzHR/job source

---

## 6. Workday (myworkdayjobs.com)

### Public API Availability
**NO PUBLIC API** - Workday does not provide a public API for job aggregation.

**Career Sites:** Hosted on myworkdayjobs.com subdomains (e.g., company.wd5.myworkdayjobs.com)

### Terms of Service Key Points
From Workday Terms of Use:
- Enterprise software with strict access controls
- Job data is customer-owned content
- No documented public API for job listings
- Job distribution typically via Workday-approved job board partners

### Technical Notes
- Workday career sites use internal JSON APIs (e.g., `/wday/cxs/{tenant}/{site}/jobs`)
- These are **internal APIs** not intended for third-party access
- No public documentation or terms of use for these endpoints

### Compliance Verdict: **PROHIBITED**
**Rationale:** Workday does not provide a public API for job aggregation. Accessing internal APIs violates their terms of service. Workday uses approved partner networks for job distribution.

**Alternative Path:**
- Become a Workday-approved job board partner
- Use Workday's Job Distribution Network partners
- Individual companies can push jobs to external boards

---

## Recommendations

### For Compliant Job Aggregation:

1. **Ashby**: ✅ Use the Posting API directly
2. **SmartRecruiters**: ✅ Register for API access
3. **JazzHR**: ✅ Partner with customers who share API keys
4. **BambooHR**: ⚠️ Requires customer partnership
5. **iCIMS**: ❌ Requires formal partnership program
6. **Workday**: ❌ Requires partner network membership

### Implementation Priority

| Priority | Platform | Action |
|----------|----------|--------|
| 1 | Ashby | Implement Posting API integration |
| 2 | SmartRecruiters | Register for API, implement integration |
| 3 | JazzHR | Document API, await customer partnerships |
| 4 | BambooHR | Deprioritize, await partnership opportunities |
| 5 | iCIMS | Explore partner program if high demand |
| 6 | Workday | Explore partner network if high demand |

---

## API Endpoint Reference

### Approved/Conditional Platforms

```yaml
ashby:
  base_url: https://api.ashbyhq.com/posting-api
  endpoint: /job-board/{companyId}
  method: POST
  auth: Basic Auth (API Key)
  docs: https://developers.ashbyhq.com/docs/posting-api-overview

smartrecruiters:
  base_url: https://api.smartrecruiters.com/v1
  endpoint: /companies/{companyId}/postings
  method: GET
  auth: API Key (requires registration)
  docs: https://developers.smartrecruiters.com/reference

jazzhr:
  base_url: https://api.resumatorapi.com/v1
  endpoint: /jobs
  method: GET
  auth: API Key (customer-provided)
  docs: https://www.resumatorapi.com/v1/
```

---

## Legal Disclaimer

This document is for informational purposes only and does not constitute legal advice. Terms of Service and API policies are subject to change. Always review the most current terms directly from each platform before implementing any integration. When in doubt, contact the platform's legal or partnership team for explicit authorization.

**Last Updated:** December 14, 2025
