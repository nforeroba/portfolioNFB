---
title: 'Projects'
date: 2024-05-19
type: landing

design:
  # Section spacing
  spacing: '5rem'

# Page sections
sections:
  - block: collection
    id: projects
    content:
      title: Selected Projects
      text: Some of my recent work on Generative AI, Machine Learning and Deep Learning, using open-source or propietary technologies
      filters:
        folders:
          - projects
    design:
      view: article-grid
      fill_image: true
      columns: 3
      show_date: true
      show_read_time: true
      show_read_more: true
  - block: markdown
    content:
      text: |
        <script>
        (function() {
          console.log('🚀 Filter script loaded for this page');
          
          document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
              initFilters();
            }, 1000);
          });
          
          function initFilters() {
            // Determine which config to use based on URL
            const isProjects = window.location.pathname.includes('/projects');
            const isBlog = window.location.pathname.includes('/blog');
            
            let buttons = [];
            if (isProjects) {
              buttons = [
                { name: 'All', tag: '*' },
                { name: 'Full-Stack', tag: 'Full-Stack' },
                { name: 'Frontend', tag: 'Frontend' },
                { name: 'Backend', tag: 'Backend' }
              ];
            } else if (isBlog) {
              buttons = [
                { name: 'All', tag: '*' },
                { name: 'Tutorial', tag: 'Tutorial' },
                { name: 'Frontend', tag: 'Frontend' },
                { name: 'Backend', tag: 'Backend' },
                { name: 'DevOps', tag: 'DevOps' }
              ];
            }
            
            if (buttons.length === 0) return;
            
            const main = document.querySelector('main');
            if (!main) return;
            
            const items = main.querySelectorAll('.col-12, article, [class*="col-md"]');
            console.log('Found items:', items.length);
            
            if (items.length <= 3) return;
            
            const title = main.querySelector('h1, h2');
            if (!title) return;
            
            const filterDiv = createButtons(buttons);
            const subtitle = title.nextElementSibling;
            if (subtitle && subtitle.tagName === 'P') {
              subtitle.after(filterDiv);
            } else {
              title.after(filterDiv);
            }
            
            setupFilters(main);
          }
          
          function createButtons(buttons) {
            const div = document.createElement('div');
            div.className = 'filter-buttons-container flex flex-wrap gap-2 justify-center mb-8 mt-6';
            
            buttons.forEach((btn, i) => {
              const button = document.createElement('button');
              button.className = i === 0
                ? 'filter-btn active px-4 py-2 rounded-lg font-medium bg-primary-600 text-white hover:bg-primary-700'
                : 'filter-btn px-4 py-2 rounded-lg font-medium bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600';
              button.textContent = btn.name;
              button.dataset.filter = btn.tag;
              div.appendChild(button);
            });
            
            return div;
          }
          
          function setupFilters(section) {
            const buttons = section.querySelectorAll('.filter-btn');
            buttons.forEach(btn => {
              btn.addEventListener('click', function() {
                const tag = this.dataset.filter;
                
                buttons.forEach(b => {
                  b.classList.remove('active', 'bg-primary-600', 'text-white', 'hover:bg-primary-700');
                  b.classList.add('bg-gray-200', 'dark:bg-gray-700', 'text-gray-700', 'dark:text-gray-300', 'hover:bg-gray-300', 'dark:hover:bg-gray-600');
                });
                this.classList.add('active', 'bg-primary-600', 'text-white', 'hover:bg-primary-700');
                this.classList.remove('bg-gray-200', 'dark:bg-gray-700', 'text-gray-700', 'dark:text-gray-300', 'hover:bg-gray-300', 'dark:hover:bg-gray-600');
                
                filterItems(section, tag);
              });
            });
          }
          
          function filterItems(section, tag) {
            const items = section.querySelectorAll('.col-12, article, [class*="col-md"]');
            
            items.forEach(item => {
              if (tag === '*') {
                item.style.display = '';
              } else {
                const tags = getTags(item);
                item.style.display = tags.includes(tag) ? '' : 'none';
              }
            });
          }
          
          function getTags(item) {
            const tags = [];
            item.querySelectorAll('a[href*="/tags/"]').forEach(link => {
              const parts = link.href.split('/tags/');
              if (parts[1]) {
                tags.push(parts[1].replace('/', ''));
              }
            });
            return tags;
          }
        })();
        </script>
---
