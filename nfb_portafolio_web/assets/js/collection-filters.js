// Collection Filters - Custom filtering for HugoBlox collection blocks
// Improved version with better section detection

document.addEventListener('DOMContentLoaded', function() {
  console.log('Collection filters script loaded');
  
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
  
  // Try multiple detection methods
  Object.keys(filterConfigs).forEach(sectionId => {
    let section = detectSection(sectionId);
    
    if (section) {
      console.log('Found section:', sectionId, section);
      initializeFilters(sectionId, filterConfigs[sectionId], section);
    } else {
      console.log('Section not found:', sectionId);
    }
  });
  
  function detectSection(sectionId) {
    // Method 1: Try to find by ID
    let section = document.getElementById(sectionId);
    if (section) return section;
    
    // Method 2: Try to find by data attribute
    section = document.querySelector(`[data-section="${sectionId}"]`);
    if (section) return section;
    
    // Method 3: Try to find by looking for the content folder
    // Look for sections that have articles from the projects or blog folder
    const allSections = document.querySelectorAll('section, div[class*="section"]');
    for (let s of allSections) {
      const links = s.querySelectorAll(`a[href*="/${sectionId}/"]`);
      if (links.length > 2) { // If multiple links to this folder exist
        return s;
      }
    }
    
    // Method 4: Look for main content area on individual pages
    if (window.location.pathname.includes(`/${sectionId}/`)) {
      section = document.querySelector('main, article, .article-container, [role="main"]');
      if (section) return section;
    }
    
    return null;
  }
  
  function initializeFilters(sectionId, buttons, section) {
    // Check if there are enough items to filter
    const items = section.querySelectorAll('.col-12, .article-item, [class*="col-md"]');
    console.log('Found items in section:', items.length);
    
    // Only add filters if there are more than 3 items
    if (items.length <= 3) {
      console.log('Not enough items to filter, skipping filters for:', sectionId);
      return;
    }
    
    // Find the title container to insert buttons after it
    const titleContainer = section.querySelector('h1, h2, .section-heading, header h1, header h2');
    if (!titleContainer) {
      console.warn('Title not found in section:', sectionId);
      return;
    }
    
    console.log('Creating filter buttons for:', sectionId);
    
    // Create filter buttons container
    const filterContainer = createFilterButtons(sectionId, buttons);
    
    // Insert after title/subtitle
    const subtitleEl = titleContainer.nextElementSibling;
    if (subtitleEl && (subtitleEl.tagName === 'P' || subtitleEl.classList.contains('section-subtitle') || subtitleEl.classList.contains('lead'))) {
      subtitleEl.after(filterContainer);
    } else {
      titleContainer.after(filterContainer);
    }
    
    // Set up filter functionality
    setupFilterLogic(sectionId, buttons, section);
  }
  
  function createFilterButtons(sectionId, buttons) {
    const container = document.createElement('div');
    container.className = 'filter-buttons-container flex flex-wrap gap-2 justify-center mb-8 mt-6';
    container.setAttribute('data-section', sectionId);
    
    buttons.forEach((button, index) => {
      const btn = document.createElement('button');
      btn.className = index === 0 
        ? 'filter-btn active px-4 py-2 rounded-lg font-medium transition-all duration-200 bg-primary-600 text-white shadow-sm hover:shadow-md'
        : 'filter-btn px-4 py-2 rounded-lg font-medium transition-all duration-200 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-primary-100 dark:hover:bg-gray-600 shadow-sm hover:shadow-md';
      btn.textContent = button.name;
      btn.setAttribute('data-filter', button.tag);
      btn.setAttribute('data-section', sectionId);
      
      container.appendChild(btn);
    });
    
    return container;
  }
  
  function setupFilterLogic(sectionId, buttons, section) {
    const filterButtons = section.querySelectorAll('.filter-btn');
    
    filterButtons.forEach(button => {
      button.addEventListener('click', function() {
        const filterTag = this.getAttribute('data-filter');
        
        console.log('Filter clicked:', filterTag, 'in section:', sectionId);
        
        // Update active button
        filterButtons.forEach(btn => {
          btn.classList.remove('active', 'bg-primary-600', 'text-white');
          btn.classList.add('bg-gray-200', 'dark:bg-gray-700', 'text-gray-700', 'dark:text-gray-300');
        });
        this.classList.add('active', 'bg-primary-600', 'text-white');
        this.classList.remove('bg-gray-200', 'dark:bg-gray-700', 'text-gray-700', 'dark:text-gray-300');
        
        // Filter items
        filterItems(section, filterTag);
      });
    });
  }
  
  function filterItems(section, filterTag) {
    // Find all article items in this section
    const items = section.querySelectorAll('.col-12, .article-item, [class*="col-md"], [class*="col-lg"]');
    
    console.log('Filtering', items.length, 'items with tag:', filterTag);
    
    let visibleCount = 0;
    
    items.forEach(item => {
      if (filterTag === '*') {
        // Show all
        item.style.display = '';
        item.classList.remove('hidden');
        visibleCount++;
      } else {
        // Get tags from the item
        const tags = getItemTags(item);
        
        if (tags.includes(filterTag)) {
          item.style.display = '';
          item.classList.remove('hidden');
          visibleCount++;
        } else {
          item.style.display = 'none';
          item.classList.add('hidden');
        }
      }
    });
    
    console.log('Visible items after filter:', visibleCount);
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
