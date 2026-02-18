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
          - "web scraping for market research"
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
      count: 3
      filters:
        folders:
          - projects
      archive:
        enable: true
        text: "Browse All Projects"
        link: "/projects/"
    design:
      view: article-grid
      columns: 3
      fill_image: false
      show_read_more: false
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
            - name: PostgreSQL
              icon: devicon/postgresql
            - name: Pandas
              icon: custom/pandas
            - name: Git
              icon: devicon/git
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
        - name: Deployment & APIs
          items:
            - name: FastAPI
              icon: devicon/fastapi
            - name: Streamlit
              icon: devicon/streamlit
            - name: Docker
              icon: devicon/docker
            - name: Linux systemd
              icon: devicon/linux
            - name: NVIDIA SMI
              icon: custom/nvidia
        - name: AI Engineering
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
      count: 3
      filters:
        folders:
          - blog
      archive:
        enable: true
        text: "Browse All Posts"
        link: "/blog/"
    design:
      view: article-grid
      columns: 3
      fill_image: false
      show_read_more: false
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
  
  - block: markdown
    content:
      text: |
        <script>
        (function() {
          console.log('🍔 Mobile menu auto-close script loaded');
          
          document.addEventListener('click', function(e) {
            const link = e.target.closest('a');
            
            if (!link) return;
            
            const href = link.getAttribute('href');
            
            if (!href || (!href.startsWith('#') && !href.startsWith('/#'))) return;
            if (href === '#' || href === '/#') return;
            
            console.log('🔗 Navigation link clicked:', href);
            
            // Find the mobile menu container
            const mobileMenu = document.querySelector('nav[role="navigation"], nav');
            
            if (!mobileMenu) {
              console.log('⚠️ Menu not found');
              return;
            }
            
            // Check if menu is visible (mobile menu is displayed)
            const isMenuVisible = window.getComputedStyle(mobileMenu).display !== 'none';
            
            console.log('📱 Menu visible:', isMenuVisible);
            
            if (isMenuVisible) {
              console.log('🔒 Closing menu by clicking hamburger button...');
              
              // Find hamburger button by its distinctive classes
              const hamburgerBtn = document.querySelector('button.inline-block.px-3.text-xl');
              
              if (hamburgerBtn) {
                setTimeout(() => {
                  hamburgerBtn.click();
                  console.log('✅ Menu toggled');
                }, 100);
              } else {
                console.log('⚠️ Hamburger button not found');
              }
            }
          }, true);
          
          console.log('✅ Event delegation setup complete');
        })();
        </script>
---