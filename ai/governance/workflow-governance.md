
# Workflow Governance

## Core Rule
All changes must follow Issue-First Development.

## Non-Negotiable Constraints
- No PR without linked issue
- No branch without issue number
- Conventional commit format required
- PR must reference issue
- Matching labels between Issue and PR
- Plain text formatting only (ASCII, markdown checkboxes)
- Issue type must be Bug, Feature, or Task

## Label Taxonomy Rules
- Task is an issue type, not a label
- Fix pairs with Bug
- Enhancement pairs with Feature
- Refactor pairs with Task
- Component labels required
- Priority and Profile labels optional but recommended

## Automated PR Rule
If automated PRs bypass issue-first:
1. Merge after review
2. Create retrospective issue documenting PR numbers

## Lint Requirement
If modifying agent-*.md, skill-*.md, or ai-report-*.md:
Run:
    ./scripts/check-agents.sh
