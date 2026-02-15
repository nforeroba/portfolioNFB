// Collection Filters - Custom filtering for HugoBlox collection blocks
// This script adds filtering capability to collection blocks

document.addEventListener('DOMContentLoaded', function() {
  
  // Configuration for each section
  const filterConfigs = {
    'projects': [
      { name: 'All', tag: '*' },
      { name: 'Full-Stack', tag: 'Full-Stack' },
      { name: 'Frontend', tag: 'Frontend' },
      { name: 'Backend', tag: 'Backend' }
    ],
    'blog': [
      { name: 'All', tag: '*' },
      { name: 'Tutorial', tag: 'Tutorial' },
      { name: 'Frontend', tag: 'Frontend' },
      { name: 'Backend', tag: 'Backend' },
      { name: 'DevOps', tag: 'DevOps' }
    ]
  };
  
  // Initialize filters for each section
  Object.keys(filterConfigs).forEach(sectionId => {
    initializeFilters(sectionId, filterConfigs[sectionId]);
  });
  
  function initializeFilters(sectionId, buttons) {
    const section = document.getElementById(sectionId);
    if (!section) return;
    
    // Find the title container to insert buttons after it
    const titleContainer = section.querySelector('h2, .section-heading');
    if (!titleContainer) return;
    
    // Create filter buttons container
    const filterContainer = createFilterButtons(sectionId, buttons);
    
    // Insert after title/subtitle
    const subtitleEl = titleContainer.nextElementSibling;
    if (subtitleEl && (subtitleEl.tagName === 'P' || subtitleEl.classList.contains('section-subtitle'))) {
      subtitleEl.after(filterContainer);
    } else {
      titleContainer.after(filterContainer);
    }
    
    // Set up filter functionality
    setupFilterLogic(sectionId, buttons);
  }
  
  function createFilterButtons(sectionId, buttons) {
    const container = document.createElement('div');
    container.className = 'filter-buttons-container flex flex-wrap gap-2 justify-center mb-8 mt-4';
    container.setAttribute('data-section', sectionId);
    
    buttons.forEach((button, index) => {
      const btn = document.createElement('button');
      btn.className = index === 0 
        ? 'filter-btn active px-4 py-2 rounded-lg font-medium transition-all duration-200 bg-primary-600 text-white'
        : 'filter-btn px-4 py-2 rounded-lg font-medium transition-all duration-200 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-primary-100 dark:hover:bg-gray-600';
      btn.textContent = button.name;
      btn.setAttribute('data-filter', button.tag);
      btn.setAttribute('data-section', sectionId);
      
      container.appendChild(btn);
    });
    
    return container;
  }
  
  function setupFilterLogic(sectionId, buttons) {
    const section = document.getElementById(sectionId);
    const filterButtons = section.querySelectorAll('.filter-btn');
    
    filterButtons.forEach(button => {
      button.addEventListener('click', function() {
        const filterTag = this.getAttribute('data-filter');
        
        // Update active button
        filterButtons.forEach(btn => {
          btn.classList.remove('active', 'bg-primary-600', 'text-white');
          btn.classList.add('bg-gray-200', 'dark:bg-gray-700', 'text-gray-700', 'dark:text-gray-300');
        });
        this.classList.add('active', 'bg-primary-600', 'text-white');
        this.classList.remove('bg-gray-200', 'dark:bg-gray-700', 'text-gray-700', 'dark:text-gray-300');
        
        // Filter items
        filterItems(sectionId, filterTag);
      });
    });
  }
  
  function filterItems(sectionId, filterTag) {
    const section = document.getElementById(sectionId);
    
    // Find all article items in this section
    const items = section.querySelectorAll('.col-12.col-md-6.col-lg-4, .article-item, [class*="col-"]');
    
    items.forEach(item => {
      if (filterTag === '*') {
        // Show all
        item.style.display = '';
        item.classList.remove('hidden');
      } else {
        // Get tags from the item
        const tags = getItemTags(item);
        
        if (tags.includes(filterTag)) {
          item.style.display = '';
          item.classList.remove('hidden');
        } else {
          item.style.display = 'none';
          item.classList.add('hidden');
        }
      }
    });
  }
  
  function getItemTags(item) {
    const tags = [];
    
    // Try to find tag links
    const tagLinks = item.querySelectorAll('a[href*="/tags/"]');
    tagLinks.forEach(link => {
      const href = link.getAttribute('href');
      if (href) {
        // Extract tag name from URL like /tags/frontend/ or /tags/frontend
        const parts = href.split('/tags/');
        if (parts.length > 1) {
          const tagName = parts[1].replace(/\/$/, ''); // Remove trailing slash
          if (tagName) {
            tags.push(tagName);
          }
        }
      }
    });
    
    return tags;
  }
  
});
