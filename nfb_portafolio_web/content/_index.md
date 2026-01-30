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
  - block: portfolio
    id: projects
    content:
      title: "Featured Projects"
      subtitle: "A selection of my recent work"
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
      subtitle: "Technologies I've used to build AI/ML applications (web or on-premises)"
      categories:
        - name: AI Hardware
          items:
            - name: NVIDIA
              icon: custom/nvidia
        - name: Languages & Core Tools
          items:
            - name: Python
              icon: devicon/python
            - name: R
              icon: devicon/r
            - name: Bash
              icon: custom/bash
            - name: PowerShell
              icon: devicon/powershell
            - name: Markdown
              icon: brands/markdown
            - name: JSON
              icon: devicon/json
            - name: NVIDIA SMI
              icon: custom/nvidia
        - name: Data Science & ML
          items:
            - name: NumPy
              icon: devicon/numpy
            - name: Pandas
              icon: devicon/pandas
            - name: Scikit-learn
              icon: devicon/scikitlearn
            - name: PyTorch
              icon: devicon/pytorch
            - name: Plotly
              icon: devicon/plotly
            - name: Kaggle
              icon: devicon/kaggle
        - name: Natural Language Processing
          items:
            - name: Gensim
              icon: custom/gensim
            - name: NLTK
              icon: custom/nltk
            - name: Hugging Face Transformers
              icon: custom/huggingface
        - name: GenAI
          items:
            - name: LangChain
              icon: custom/langchain
            - name: LangGraph
              icon: custom/langgraph
            - name: Claude AI
              icon: custom/claudeai
            - name: Open AI
              icon: custom/openai
            - name: Gemini AI
              icon: custom/geminiai
            - name: Mistral AI
              icon: custom/mistralai
            - name: Ollama
              icon: custom/ollama
            - name: Nano Banana
              icon: custom/nanobanana
            - name: ChromaDB
              icon: custom/chromadb
        - name: Computer Vision
          items:
            - name: OpenCV
              icon: devicon/opencv
            - name: Roboflow
              icon: custom/roboflow
        - name: Audio Edition & Speech Synthesis
          items:
            - name: Fish Audio
              icon: custom/fishaudio
            - name: Audacity
              icon: custom/audacity
        - name: Web & APIs
          items:
            - name: FastAPI
              icon: devicon/fastapi
            - name: Streamlit
              icon: devicon/streamlit
            - name: aiohttp
              icon: custom/aiohttp
            - name: Shiny
              icon: custom/shiny
            - name: Hugo
              icon: devicon/hugo
            - name: Netlify
              icon: devicon/netlify
        - name: Databases
          items:
            - name: PostgreSQL
              icon: devicon/postgresql
            - name: MySQL
              icon: devicon/mysql
            - name: SQLite
              icon: devicon/sqlite
            - name: ClickHouse
              icon: custom/clickhouse
        - name: Dev Tools & IDEs
          items:
            - name: VS Code
              icon: devicon/vscode
            - name: Jupyter
              icon: devicon/jupyter
            - name: Anaconda
              icon: devicon/anaconda
            - name: RStudio
              icon: devicon/rstudio
            - name: DBeaver
              icon: devicon/dbeaver
        - name: DevOps & Infrastructure
          items:
            - name: Git
              icon: devicon/git
            - name: GitHub
              icon: devicon/github
            - name: Linux
              icon: devicon/linux
            - name: Ubuntu
              icon: devicon/ubuntu
            - name: Windows 11
              icon: devicon/windows11
            # - name: SSH
            #   icon: custom/ssh
            - name: PuTTY
              icon: devicon/putty
            - name: FileZilla
              icon: devicon/filezilla
        - name: Team Communication
          items:
            - name: Slack
              icon: devicon/slack
            - name: Discord
              icon: custom/discord
        - name: Web Scraping
          items:
            - name: Selenium
              icon: devicon/selenium
            - name: Octoparse
              icon: custom/octoparse
            - name: ParseHub
              icon: custom/parsehub
        - name: My Learning Platforms
          items:
            - name: Udemy
              icon: custom/udemy
            - name: Coursera
              icon: custom/coursera
            - name: Business Science University
              icon: custom/bsu
            - name: Machine Learning School
              icon: custom/mls
            - name: DeepLearning.AI
              icon: custom/deeplearningai
            - name: YouTube
              icon: custom/youtube
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
          date_start: '2024-07-15'
          date_end: '2025-08-29'
          description: |2-
            * Test
        - title: 'Data Scientist (contractor/consultant)'
          company: 'IMPSERCOM S.A.S / Espumados Group - Manufacture & Retail'
          company_url: 'https://www.grupoespumados.com/'
          company_logo: custom/espumados
          location: 'Bogotá, Colombia'
          date_start: '2024-01-22'
          date_end: '2024-04-22'
          description: |2-
            * Test
        - title: 'R&D | QC | QA Scientist (full-time)'
          company: 'Bank of the Republic (Colombia) - BRC'
          company_url: 'https://www.banrep.gov.co/'
          company_logo: custom/banrep
          location: 'Bogotá, Colombia'
          date_start: '2018-05-08'
          date_end: '2023-01-31'
          description: |2-
            * Test
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
      title: Recent Posts
      subtitle: 'Thoughts on AI, technology, science and more'
      text: ''
      filters:
        folders:
          - blog
        exclude_featured: false
      count: 3
      order: desc
    design:
      view: card
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
      subtitle: "Let's build something amazing together"
      text: |-
        I'm always interested in hearing about new projects and opportunities.
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
        I'm looking for **AI engineering**, **AI development** & **Data Science** roles.
        
        Let's connect and discuss how I can help your team.
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
