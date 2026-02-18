// Auto-close mobile navigation menu when clicking on navigation links
// Works on any narrow screen where hamburger menu is visible

(function() {
  console.log('🍔 Mobile menu auto-close script loaded');
  
  // Wait for DOM to be ready
  document.addEventListener('DOMContentLoaded', function() {
    
    // Find all navigation links (links that start with #)
    const navLinks = document.querySelectorAll('nav a[href^="#"], header a[href^="#"]');
    
    if (navLinks.length === 0) {
      console.log('⚠️ No navigation links found');
      return;
    }
    
    console.log(`✅ Found ${navLinks.length} navigation links`);
    
    navLinks.forEach(link => {
      link.addEventListener('click', function(e) {
        
        // Check if mobile menu is currently open
        // HugoBlox typically uses a button to toggle menu and adds classes or attributes
        const mobileMenuButton = document.querySelector('button[data-toggle="navigation"], button[aria-expanded]');
        
        if (!mobileMenuButton) return;
        
        // Check if the menu button indicates the menu is open
        const isMenuOpen = mobileMenuButton.getAttribute('aria-expanded') === 'true' ||
                          mobileMenuButton.classList.contains('active') ||
                          document.body.classList.contains('menu-open');
        
        if (isMenuOpen) {
          console.log('📱 Menu is open, closing it after navigation');
          
          // Small delay to let navigation happen first
          setTimeout(() => {
            // Trigger click on menu button to close it
            mobileMenuButton.click();
          }, 100);
        }
      });
    });
  });
  
})();
