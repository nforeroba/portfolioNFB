---
title: 'Blog'
date: 2024-05-19
type: landing

design:
  # Section spacing
  spacing: '5rem'

# Page sections
sections:
  - block: collection
    content:
      title: Blog Posts
      text: Thoughts on AI, technology, science and more.
      filters:
        folders:
          - blog
    design:
      view: article-grid
      fill_image: false
      columns: 3
      show_date: true
      show_read_time: true
      show_read_more: false
---