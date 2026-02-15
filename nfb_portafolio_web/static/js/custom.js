console.log('Custom JS loaded from static folder');

document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM Ready - Initializing collection filters');
  
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
  
  Object.keys(filterConfigs).forEach(sectionId => {
    console.log('Looking for section:', sectionId, 'in path:', window.location.pathname);
    
    // Only init filters on the right pages
    if (window.location.pathname === '/' || 
        window.location.pathname.includes(`/${sectionId}/`)) {
      
      setTimeout(() => {
        let section = detectSection(sectionId);
        if (section) {
          console.log('✓ FOUND section:', sectionId);
          initializeFilters(sectionId, filterConfigs[sectionId], section);
        } else {
          console.log('✗ Section NOT found:', sectionId);
        }
      }, 500); // Wait for page to fully render
    }
  });
  
  function detectSection(sectionId) {
    console.log('Detecting section:', sectionId);
    
    // Try ID first
    let section = document.getElementById(sectionId);
    if (section) {
      console.log('Found by ID');
      return section;
    }
    
    // Try data attribute
    section = document.querySelector(`[data-section="${sectionId}"]`);
    if (section) {
      console.log('Found by data-section');
      return section;
    }
    
    // Look for links to the folder
    const allContainers = document.querySelectorAll('section, div, main, article');
    for (let container of allContainers) {
      const links = container.querySelectorAll(`a[href*="/${sectionId}/"]`);
      if (links.length >= 2) {
        console.log('Found by analyzing links, count:', links.length);
        return container;
      }
    }
    
    // On individual pages, use main content
    if (window.location.pathname.includes(`/${sectionId}/`)) {
      section = document.querySelector('main');
      if (section) {
        console.log('Found main on individual page');
        return section;
      }
    }
    
    return null;
  }
  
  function initializeFilters(sectionId, buttons, section) {
    const items = section.querySelectorAll('.col-12, article, [class*="col-md"]');
    console.log('Items found:', items.length);
    
    if (items.length <= 3) {
      console.log('Too few items, skipping');
      return;
    }
    
    const title = section.querySelector('h1, h2');
    if (!title) {
      console.log('No title found');
      return;
    }
    
    console.log('Creating buttons');
    const filterDiv = createButtons(sectionId, buttons);
    
    const subtitle = title.nextElementSibling;
    if (subtitle && subtitle.tagName === 'P') {
      subtitle.after(filterDiv);
    } else {
      title.after(filterDiv);
    }
    
    setupFilters(section, sectionId);
  }
  
  function createButtons(sectionId, buttons) {
    const div = document.createElement('div');
    div.className = 'filter-buttons-container flex flex-wrap gap-2 justify-center mb-8 mt-6';
    
    buttons.forEach((btn, i) => {
      const button = document.createElement('button');
      button.className = i === 0
        ? 'filter-btn active px-4 py-2 rounded-lg font-medium bg-primary-600 text-white'
        : 'filter-btn px-4 py-2 rounded-lg font-medium bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300';
      button.textContent = btn.name;
      button.dataset.filter = btn.tag;
      button.dataset.section = sectionId;
      div.appendChild(button);
    });
    
    return div;
  }
  
  function setupFilters(section, sectionId) {
    const buttons = section.querySelectorAll('.filter-btn');
    buttons.forEach(btn => {
      btn.addEventListener('click', function() {
        const tag = this.dataset.filter;
        console.log('Filter:', tag);
        
        buttons.forEach(b => {
          b.classList.remove('active', 'bg-primary-600', 'text-white');
          b.classList.add('bg-gray-200', 'dark:bg-gray-700');
        });
        this.classList.add('active', 'bg-primary-600', 'text-white');
        this.classList.remove('bg-gray-200', 'dark:bg-gray-700');
        
        filterItems(section, tag);
      });
    });
  }
  
  function filterItems(section, tag) {
    const items = section.querySelectorAll('.col-12, article, [class*="col-md"]');
    console.log('Filtering', items.length, 'items');
    
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
});
