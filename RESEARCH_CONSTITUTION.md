# SecondOrder Research Constitution

## Core Principles

### 1. Generic System
- **No hardcoded query-specific logic** - System must work for ANY research query
- No lists of specific terms (ambiguous terms, entity names, etc.)
- All decisions must be dynamic and query-driven
- No optimization for specific example queries

### 2. Chain Query Handling
For multi-hop queries (where answer depends on multiple steps):
- **Always find the ANCHOR first** - the initial entity/subject
- Emphasize "winner", "won", "1st place" in search queries to find actual winners
- Include FULL award context (e.g., "Phoenix New Times Best of Phoenix 2006")
- Verify the anchor before proceeding to next hop
- Each step must verify its predecessor's findings

### 3. Data Aggregation Queries
For queries requiring cross-referencing multiple data sources:
- Identify all required data points separately
- Search each independently first
- Cross-reference results AFTER gathering all data
- Don't assume correlation without evidence

### 4. Source Quality
- Prefer primary sources (official websites, databases)
- Cross-reference claims with multiple sources
- Flag contradictory evidence
- Prioritize recent data for "current" queries

## Plan Generation Rules

### Structured Plan Requirements
Every research plan MUST include:
- **id**: Unique step identifier
- **query**: Search-engine-ready query string
- **purpose**: What this step aims to find/resolve
- **dependencies**: Which steps must complete first

### Chain Query Plans
For chain queries, the plan MUST:
1. Lead with anchor resolution (who/what won)
2. Verify anchor with corroborating source
3. Then search for related entities
4. Final step synthesizes all findings

### Quality Indicators
A good plan has:
- At least 4 steps for complex queries
- Clear purpose for each step
- Logical dependency order
- Corroboration/verification step

## Search Query Construction

### Award/Winner Queries
When query mentions an award:
- Lead with "winner" or "won" - e.g., `"2023 winner" "Best Artist"`
- Include full award context - e.g., `"Grammy Awards 2023"`
- Verify winner from official source

### Role/Person Queries
When query asks about a person's role:
- Include timeframe if specified
- Verify person played that role
- Cross-reference with band/group records

### Timeline Queries
When query asks about history:
- Prioritize official sources
- Verify dates independently
- Note conflicting dates

## Error Handling

### Low Signal Detection
If search results are poor:
- Reformulate query with different keywords
- Try broader search first, then narrow
- Consider alternative sources

### Contradictory Evidence
If evidence contradicts:
- Flag the contradiction
- Search for clarification
- Present both perspectives in final report

### Verification Failed
If verification fails:
- Note uncertainty in report
- Suggest additional research
- Never fabricate sources

## Output Standards

### Report Requirements
Final report MUST:
- Cite sources with URLs
- Distinguish facts from analysis
- Acknowledge limitations
- Provide evidence for claims

### Answer Format
For specific answer queries (name, number, etc.):
- Provide direct answer first
- Show supporting evidence
- Note any uncertainty
