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
title: "S&P500, Crypto & FX Forecasting Application"
date: 2026-03-17
summary: "Pick a symbol and run price forecasts. Check performance metrics and interact with validation and forecast plots."
authors:
  - me

# === TAGS (Required for all) ===
# For blog: Tutorial, DevOps, Frontend, Backend, Docker, React, Node.js, etc.
# For projects: Backend, Frontend, Full-Stack, API, etc.
tags:
  - Forecasting
  - Machine Learning
  - Finance
  - Time Series

# === FEATURED IMAGE (Optional) ===
# If true, this item appears first in featured sections
featured: false

# ============================================================================
# PROJECT-SPECIFIC FIELDS (Leave empty/delete for blog posts)
# ============================================================================

# === TECH STACK (Projects only) ===
# List of technologies used
tech_stack:
  - Python
  - Dash
  - Plotly
  - Prophet
  - statsforecast
  - XGBoost
  - scikit-learn
  - MAPIE
  - yfinance
  - Docker
  - Hugging Face Spaces

# === LINKS (Projects only) ===
# CODE and DEMO buttons
links:
  - type: github
    url: https://github.com/nforeroba/fin_fore_app
    label: Code
  - type: live
    url: https://huggingface.co/spaces/nikoniko23/fin_fore_app
    label: Demo

# === PROJECT METADATA (Projects only) ===
status: "Live"                    # Live, In Progress, Completed, Archived
role: "Solo Developer"            # Your role in the project
duration: "1 week"                # How long it took
team_size: 1                      # Number of people

# === HIGHLIGHTS (Projects only) ===
# Key achievements or metrics
highlights:
  - "Symbols: 500+ stocks, 100 crypto and 28 FX pairs available"
  - "8 forecasting models with different performance, bias and overfitting metrics"
  - "Deployed on Hugging Face Spaces via Docker"

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

**Project Status**: ✅ Live in Production  
**GitHub**: [View Source Code](https://github.com/alexjohnson/ecommerce-platform)  
**Demo**: [Try it Live](https://shop-demo.example.com)

<!-- 
  NOTES:
  - Don't add author info here (auto-generated at bottom)
  - Don't add share buttons (auto-generated)
  - Images: Place featured.png/jpg in same folder as index.md
  - Other images: Reference as ![Alt text](image.png)
-->
