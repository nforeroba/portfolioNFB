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
          icon: envelope
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
        - name: Languages & Core
          items:
            - name: Python
              icon: devicon/python
            - name: R
              icon: devicon/r
            - name: Bash
              icon: brands/gnubash
            - name: PowerShell
              icon: devicon/powershell
            - name: Markdown
              icon: brands/markdown
            - name: JSON
              icon: devicon/json
            - name: NVIDIA
              icon: custom/nvidia
            - name: Hugging Face
              icon: custom/huggingface
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
            - name: Matplotlib
              icon: devicon/matplotlib
            - name: Plotly
              icon: devicon/plotly
            - name: Kaggle
              icon: devicon/kaggle
        - name: Natural Language Processing (NLP)
          items:
            - name: Gensim
              icon: custom/gensim
            - name: NLTK
              icon: custom/nltk
            - name: Hugging Face Transformers
              icon: custom/huggingface
        - name: GenAI & LLMs
          items:
            - name: LangChain
              icon: custom/langchain
            - name: LangGraph
              icon: custom/langgraph
            - name: Claude AI (Anthropic)
              icon: custom/claudeai
            - name: Open AI API
              icon: custom/openai
            - name: Gemini AI (Google)
              icon: custom/geminiai
            - name: Mistral AI
              icon: custom/mistralai
            - name: Ollama
              icon: custom/ollama
            - name: Nano Banana
              icon: custom/nanobanana
            - name: DALL-E
              icon: custom/openai
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
              icon: devicon/clickhouse
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
            - name: Spyder
              icon: devicon/spyder
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
            - name: Debian
              icon: devicon/debian
            - name: Windows 11
              icon: devicon/windows11
            - name: SSH
              icon: custom/ssh
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
            - name: Beautiful Soup
              icon: custom/beautifulsoup
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
        - title: Senior Software Engineer
          company: Tech Corp
          company_url: ''
          company_logo: ''
          location: San Francisco, CA
          date_start: '2023-01-01'
          date_end: ''
          description: |2-
            * Lead development of microservices architecture serving 1M+ users
            * Improved API response time by 40% through optimization
            * Mentored team of 5 junior developers
            * Tech stack: React, Node.js, PostgreSQL, AWS
        - title: Full-Stack Developer
          company: Startup Inc
          company_url: ''
          company_logo: ''
          location: Remote
          date_start: '2021-06-01'
          date_end: '2022-12-31'
          description: |2-
            * Built and deployed 3 production applications from scratch
            * Implemented CI/CD pipeline reducing deployment time by 60%
            * Collaborated with design team on UI/UX improvements
            * Tech stack: Next.js, Express, MongoDB, Docker
        - title: Junior Developer
          company: Web Agency
          company_url: ''
          company_logo: ''
          location: New York, NY
          date_start: '2020-01-01'
          date_end: '2021-05-31'
          description: |2-
            * Developed client websites using modern web technologies
            * Maintained and updated legacy codebases
            * Participated in code reviews and agile ceremonies
            * Tech stack: React, WordPress, PHP, MySQL
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
