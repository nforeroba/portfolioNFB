---
# ============================================================================
# MULTIPURPOSE TEMPLATE - Use for both Blog Posts and Projects
# ============================================================================
# Instructions:
# - For BLOG POSTS: Fill only basic fields (title, date, summary, tags, authors)
# - For PROJECTS: Fill all fields including tech_stack, links, status, etc.
# - Delete unused fields or leave them empty
# ============================================================================

# === BASIC FIELDS (Required for all) ===
title: "Blog Title"
date: 2026-02-17
summary: "Brief description that appears in cards and meta tags"
authors:
  - me

# === TAGS (Required for all) ===
# For blog: Tutorial, DevOps, Frontend, Backend, Docker, React, Node.js, etc.
# For projects: Backend, Frontend, Full-Stack, API, etc.
tags:
  - Tag1
  - Tag2
  - Tag3

# === FEATURED IMAGE (Optional) ===
# If true, this item appears first in featured sections
featured: false

# ============================================================================
# PROJECT-SPECIFIC FIELDS (Leave empty/delete for blog posts)
# ============================================================================

# === TECH STACK (Projects only) ===
# List of technologies used
tech_stack:
  - React
  - TypeScript
  - Node.js
  - PostgreSQL
  - Docker

# === LINKS (Projects only) ===
# CODE and DEMO buttons
links:
  - type: github
    url: https://github.com/username/project
    label: Code
  - type: live
    url: https://demo.example.com
    label: Demo

# === PROJECT METADATA (Projects only) ===
status: "Live"                    # Live, In Progress, Completed, Archived
role: "Lead Developer"            # Your role in the project
duration: "4 months"              # How long it took
team_size: 2                      # Number of people

# === HIGHLIGHTS (Projects only) ===
# Key achievements or metrics
highlights:
  - "Handles 10k+ concurrent users"
  - "99.9% uptime SLA"
  - "60% performance improvement"

# ============================================================================
# END OF FRONTMATTER
# ============================================================================
---

<!-- 
  CONTENT STRUCTURE GUIDELINES:
  
  Write your content below using Markdown.
  The layout will automatically arrange elements in this order:
  
  1. Title (from frontmatter)
  2. Metadata (date, author, reading time, status/role/duration)
  3. CODE/DEMO Buttons (if links exist)
  4. Tags + Tech Stack (if exists)
  5. Highlights (if exist)
  6. Featured Image (featured.png/jpg in same folder)
  7. Table of Contents (auto-generated from headings)
  8. YOUR CONTENT BELOW ⬇️
  9. Share buttons (auto-added)
  10. Author info (auto-added)
-->

Brief introduction paragraph explaining what this project/post is about.

## Section 1: Main Topic

Content for your first major section. Use ## for main sections, ### for subsections.

### Subsection 1.1

More detailed content here.

### Subsection 1.2

Additional details.

## Section 2: Technical Details

Explain technical aspects, implementation, or methodology.

```javascript
// Code examples with syntax highlighting
const example = "Your code here";
```

## Section 3: Results or Outcomes

For projects: metrics, achievements, impact
For blog: conclusions, takeaways, recommendations

### Key Metrics (Projects)

- **Performance**: Specific improvements
- **Scale**: User numbers, throughput
- **Impact**: Business results

### Takeaways (Blog)

- Main point 1
- Main point 2
- Main point 3

## Challenges & Solutions (Optional)

### Challenge 1: Problem Description

**Problem**: What went wrong or was difficult

**Solution**: How you solved it

## Future Work (Optional)

- [ ] Feature or improvement 1
- [ ] Feature or improvement 2
- [ ] Feature or improvement 3

## Resources (Optional for blog)

- [Link to documentation](https://example.com)
- [Related article](https://example.com)
- [Tool or library](https://example.com)

## Conclusion

Summary paragraph wrapping up the main points.

---

<!-- 
  NOTES:
  - Don't add author info here (auto-generated at bottom)
  - Don't add share buttons (auto-generated)
  - Images: Place featured.png/jpg in same folder as index.md
  - Other images: Reference as ![Alt text](image.png)
-->
