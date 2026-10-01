// Vizuara Books - Interactive Reader Script

(function() {
  // Theme Management
  const THEME_KEY = 'vb-reader-theme';
  const FONT_KEY = 'vb-reader-font-size';

  function initTheme() {
    const savedTheme = localStorage.getItem(THEME_KEY) || 'paper';
    setTheme(savedTheme, false);

    const themeSelect = document.getElementById('theme-select');
    if (themeSelect) {
      themeSelect.value = savedTheme;
      themeSelect.addEventListener('change', (e) => {
        setTheme(e.target.value, true);
      });
    }

    const savedFont = localStorage.getItem(FONT_KEY) || 'medium';
    setFontSize(savedFont, false);
  }

  function setTheme(theme, save = true) {
    document.documentElement.setAttribute('data-rtheme', theme);
    document.body.setAttribute('data-rtheme', theme);
    if (save) {
      try { localStorage.setItem(THEME_KEY, theme); } catch(e) {}
    }
  }

  function setFontSize(size, save = true) {
    document.body.setAttribute('data-font-size', size);
    if (save) {
      try { localStorage.setItem(FONT_KEY, size); } catch(e) {}
    }
  }

  // Reading Progress
  function initProgress() {
    const progressBar = document.querySelector('.vb-progress-bar');
    if (!progressBar) return;

    window.addEventListener('scroll', () => {
      const scrollTop = window.scrollY || document.documentElement.scrollTop;
      const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const progress = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0;
      progressBar.style.width = Math.min(100, Math.max(0, progress)) + '%';
    }, { passive: true });
  }

  // Table of Contents Drawer
  function initDrawer() {
    const toggleBtn = document.getElementById('toc-toggle');
    const overlay = document.getElementById('toc-overlay');
    const closeBtn = document.getElementById('toc-close');

    if (!overlay) return;

    function openDrawer() {
      overlay.classList.add('open');
      document.body.style.overflow = 'hidden';
    }

    function closeDrawer() {
      overlay.classList.remove('open');
      document.body.style.overflow = '';
    }

    if (toggleBtn) toggleBtn.addEventListener('click', openDrawer);
    if (closeBtn) closeBtn.addEventListener('click', closeDrawer);

    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeDrawer();
    });

    // Close on escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeDrawer();
        closeLightbox();
      } else if (e.key === 't' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        if (overlay.classList.contains('open')) closeDrawer();
        else openDrawer();
      }
    });
  }

  // Lightbox for figures
  let lightbox = null;
  function initLightbox() {
    lightbox = document.createElement('div');
    lightbox.className = 'vb-lightbox';
    lightbox.innerHTML = '<img src="" alt="Zoomed figure">';
    document.body.appendChild(lightbox);

    const lightboxImg = lightbox.querySelector('img');

    lightbox.addEventListener('click', closeLightbox);

    document.querySelectorAll('.art-body figure img').forEach(img => {
      img.addEventListener('click', () => {
        lightboxImg.src = img.src;
        lightboxImg.alt = img.alt || 'Zoomed figure';
        lightbox.classList.add('open');
        document.body.style.overflow = 'hidden';
      });
    });
  }

  function closeLightbox() {
    if (lightbox && lightbox.classList.contains('open')) {
      lightbox.classList.remove('open');
      document.body.style.overflow = '';
    }
  }

  // Copy Code Button
  function initCodeBlocks() {
    document.querySelectorAll('.art-body pre').forEach(pre => {
      const copyBtn = document.createElement('button');
      copyBtn.className = 'vb-copy-code-btn';
      copyBtn.innerText = 'Copy';
      copyBtn.style.cssText = 'position:absolute;top:8px;right:8px;font-family:var(--font-mono);font-size:11px;padding:3px 8px;border-radius:4px;border:1px solid var(--rt-rule);background:var(--rt-surface-card);color:var(--rt-ink-soft);cursor:pointer;opacity:0;transition:opacity 0.15s ease;';

      pre.style.position = 'relative';
      pre.appendChild(copyBtn);

      pre.addEventListener('mouseenter', () => copyBtn.style.opacity = '1');
      pre.addEventListener('mouseleave', () => copyBtn.style.opacity = '0');

      copyBtn.addEventListener('click', () => {
        const code = pre.querySelector('code') || pre;
        navigator.clipboard.writeText(code.innerText).then(() => {
          copyBtn.innerText = 'Copied!';
          copyBtn.style.color = 'var(--accent)';
          setTimeout(() => {
            copyBtn.innerText = 'Copy';
            copyBtn.style.color = 'var(--rt-ink-soft)';
          }, 2000);
        });
      });
    });
  }

  // Keyboard navigation between chapters
  function initKeyboardNav() {
    document.addEventListener('keydown', (e) => {
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      if (e.key === 'ArrowLeft' || e.key === 'h') {
        const prevLink = document.querySelector('a.vb-nav-card.prev');
        if (prevLink) prevLink.click();
      } else if (e.key === 'ArrowRight' || e.key === 'l') {
        const nextLink = document.querySelector('a.vb-nav-card.next');
        if (nextLink) nextLink.click();
      }
    });
  }

  // Font size buttons
  function initFontSizeControls() {
    const btnSm = document.getElementById('font-smaller');
    const btnLg = document.getElementById('font-larger');
    const sizes = ['small', 'medium', 'large'];

    function changeSize(delta) {
      const current = document.body.getAttribute('data-font-size') || 'medium';
      let idx = sizes.indexOf(current);
      if (idx === -1) idx = 1;
      let nextIdx = Math.min(2, Math.max(0, idx + delta));
      setFontSize(sizes[nextIdx]);
    }

    if (btnSm) btnSm.addEventListener('click', () => changeSize(-1));
    if (btnLg) btnLg.addEventListener('click', () => changeSize(1));
  }

  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initTheme();
      initProgress();
      initDrawer();
      initLightbox();
      initCodeBlocks();
      initKeyboardNav();
      initFontSizeControls();
    });
  } else {
    initTheme();
    initProgress();
    initDrawer();
    initLightbox();
    initCodeBlocks();
    initKeyboardNav();
    initFontSizeControls();
  }
})();
