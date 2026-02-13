---
# Leave the homepage title empty to use the site title
title: ''
summary: ''
date: 2026-01-05
type: landing

design:
  # Default section spacing
  spacing: '0'

sections:
  # Developer Hero - Gradient background with name, role, social, and CTAs
  - block: dev-hero
    id: hero
    content:
      username: me
      greeting: "Hi, I'm"
      show_status: true
      show_scroll_indicator: true
      typewriter:
        enable: true
        prefix: "I have worked on"
        strings:
          - "LLM-powered Q&A systems"
          - "document intelligence with VLMs"
          - "voice cloning & speech synthesis"
          - "ML for predictive process optimization"
          - "deployment of WebSocket & batch AI services"
          - "web scraping for price monitoring & market research"
        type_speed: 50
        delete_speed: 40
        pause_time: 1000
      cta_buttons:
        - text: View My Work
          url: "#projects"
          icon: arrow-down
        - text: Get In Touch
          url: "#contact"
          icon: paper-airplane
    design:
      style: centered
      avatar_shape: circle
      animations: true
      background:
        color:
          light: "#fafafa"
          dark: "#0a0a0f"
      spacing:
        padding: ["6rem", "0", "4rem", "0"]
  
  # Filterable Portfolio - Alpine.js powered project filtering
  - block: collection
    id: projects
    content:
      title: "Featured Projects"
      subtitle: "Some of my recent work on Generative AI, Machine Learning and Deep Learning, using open-source or propietary technologies"
      count: 0
      filters:
        folders:
          - projects
      buttons:
        - name: All
          tag: '*'
        - name: Full-Stack
          tag: Full-Stack
        - name: Frontend
          tag: Frontend
        - name: Backend
          tag: Backend
      default_button_index: 0
      # Archive link auto-shown if more projects exist than 'count' above
      # archive:
      #   enable: false  # Set to false to explicitly hide
      #   text: "Browse All"  # Customize text
      #   link: "/work/"  # Custom URL
    design:
      columns: 3
      background:
        color:
          light: "#ffffff"
          dark: "#0d0d12"
      spacing:
        padding: ["4rem", "0", "4rem", "0"]
  
# Visual Tech Stack - Icons organized by category
  - block: tech-stack
    id: skills
    content:
      title: "Tech Stack"
      subtitle: "Technologies I've used to build AI/ML applications (web and on-premises)"
      categories:
        - name: Code, Data & Development
          items:
            - name: Python
              icon: devicon/python
            - name: R
              icon: devicon/rstudio
            - name: SQL
              icon: devicon/postgresql
            - name: Pandas
              icon: devicon/pandas
            - name: PostgreSQL
              icon: devicon/postgresql
            - name: Git
              icon: devicon/git
            - name: NVIDIA SMI
              icon: custom/nvidia
        - name: Machine & Deep Learning
          items:
            - name: PyTorch
              icon: devicon/pytorch
            - name: Scikit-learn
              icon: devicon/scikitlearn
            - name: Hugging Face Transformers
              icon: custom/huggingface
            - name: OpenCV
              icon: devicon/opencv
            - name: Fish Audio
              icon: custom/fishaudio
        - name: GenAI & LLMs
          items:
            - name: Ollama
              icon: custom/ollama
            - name: ChromaDB
              icon: custom/chromadb
            - name: LangChain & LangGraph
              icon: custom/langchain_2
            - name: Claude AI
              icon: custom/claudeai
            - name: Gemini AI
              icon: custom/geminiai
            - name: Mistral AI
              icon: custom/mistralai
            - name: OpenAI
              icon: custom/openai
        - name: Deployment & APIs
          items:
            - name: FastAPI
              icon: devicon/fastapi
            - name: Streamlit
              icon: devicon/streamlit
            - name: Docker
              icon: devicon/docker
            - name: Linux
              icon: devicon/linux
    design:
      style: grid
      show_levels: false
      background:
        color:
          light: "#f5f5f5"
          dark: "#08080c"
      spacing:
        padding: ["4rem", "0", "4rem", "0"]
  
  # Experience Timeline
  - block: resume-experience
    id: experience
    content:
      title: Experience
      date_format: Jan 2006
      items:
        - title: 'Data Scientist + AI Engineer/Developer (full-time)'
          company: 'AECSA BPO - Debt Collection'
          company_url: 'https://www.aecsa.com.co/'
          company_logo: custom/aecsa
          location: 'Bogotá, Colombia'
          date_start: 2024-07-15
          date_end: 2025-08-29
          description: |2-
            * Useless block. This information is actually taken from data\authors\me.yaml
        - title: 'Data Scientist (contractor/consultant)'
          company: 'IMPSERCOM S.A.S / Espumados Group - Manufacture & Retail'
          company_url: 'https://www.grupoespumados.com/'
          company_logo: custom/espumados
          location: 'Bogotá, Colombia'
          date_start: 2024-01-22
          date_end: 2024-04-22
          description: |2-
            * Useless block. This information is actually taken from data\authors\me.yaml
        - title: 'R&D | QC | QA Scientist (full-time)'
          company: 'Bank of the Republic (Colombia) - BRC'
          company_url: 'https://www.banrep.gov.co/'
          company_logo: custom/banrep
          location: 'Bogotá, Colombia'
          date_start: 2018-05-08
          date_end: 2023-01-31
          description: |2-
            * Useless block. This information is actually taken from data\authors\me.yaml
    design:
      columns: '1'
      background:
        color:
          light: "#ffffff"
          dark: "#0d0d12"
      spacing:
        padding: ["4rem", "0", "4rem", "0"]
  
  # Recent Blog Posts
  - block: collection
    id: blog
    content:
      title: "Recent Posts"
      subtitle: "Thoughts and tutorials on AI, technology, science and more!"
      count: 0
      filters:
        folders:
          - blog
      buttons:
        - name: All
          tag: '*'
        - name: Tutorial
          tag: Tutorial
        - name: Frontend
          tag: Frontend
        - name: Backend
          tag: Backend
        - name: DevOps
          tag: DevOps
      default_button_index: 0
    design:
      columns: 3
      background:
        color:
          light: "#f5f5f5"
          dark: "#08080c"
      spacing:
        padding: ["4rem", "0", "4rem", "0"]
  
  # Contact Section
  - block: contact-info
    id: contact
    content:
      title: Get In Touch
      #subtitle: ""
      text: |-
        Whether you're looking to hire, collaborate, or just want to say hi, feel free to reach out!
      email: nikorasu.fb@outlook.com
      autolink: true
    design:
      columns: '1'
      background:
        color:
          light: "#ffffff"
          dark: "#0d0d12"
      spacing:
        padding: ["4rem", "0", "4rem", "0"]
  
  # CTA Card
  - block: cta-card
    content:
      title: "Open to Opportunities"
      text: |-
        I'm looking for **AI engineering**, **AI development** and/or **Data Science** roles.
        Get to know me through this website or my resume below **↓**
      button:
        text: 'Download Resume'
        url: uploads/nicolas_forero_baena_resume_2026.pdf
        new_tab: true
    design:
      card:
        # Light mode: soft pastel theme gradient | Dark mode: rich deep gradient
        css_class: 'bg-gradient-to-br from-primary-200 via-primary-100 to-secondary-200 dark:from-primary-600 dark:via-primary-700 dark:to-secondary-700'
        text_color: dark
      background:
        color:
          light: "#f5f5f5"
          dark: "#08080c"
      spacing:
        padding: ["4rem", "0", "6rem", "0"]
---
