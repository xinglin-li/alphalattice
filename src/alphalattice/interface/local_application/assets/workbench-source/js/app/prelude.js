/* The synchronous prelude applies preferences before the stylesheet paints, avoiding a frame in the wrong appearance or layout. It is plain ES5 because it runs before the app; the host's CSP allows only its external script. */
(function () {
  var html = document.documentElement;
  var pref = {};
  try { pref = JSON.parse(localStorage.getItem('alphalattice.workstation.viewer.v4') || '{}') || {}; } catch (e) { pref = {}; }
  var q = {};
  try {
    var pairs = (location.hash || '').replace(/^#/, '').split('&');
    for (var i = 0; i < pairs.length; i += 1) {
      var at = pairs[i].indexOf('=');
      if (at > 0) q[decodeURIComponent(pairs[i].slice(0, at))] = decodeURIComponent(pairs[i].slice(at + 1));
    }
  } catch (e) { q = {}; }
  // the appearance: the viewer's choice, or the host's when following (router.js's effectiveTheme)
  var aliases = {silver: 'light', titanium: 'light', emerald: 'light', ruby: 'light', obsidian: 'dark'};
  var theme = q.theme || pref.theme || 'follow';
  theme = aliases[theme] || theme;
  var dark = theme === 'dark' || (theme !== 'light' && typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches);
  html.setAttribute('data-theme', dark ? 'dark' : 'light');
  html.style.colorScheme = dark ? 'dark' : 'light';
  // the language (i18n.js's set)
  var lang = q.lang || pref.lang || 'en';
  var zh = lang === 'zh-CN' || lang === 'zh' || lang === 'cn';
  html.setAttribute('lang', zh ? 'zh-CN' : 'en');
  // the frame (window.js's syncFrame): the dock's side and the layout
  html.setAttribute('data-dock', pref.dockSide === 'left' ? 'left' : 'right');
  html.setAttribute('data-nav', pref.navigation === 'top' ? 'drawer' : 'docked');
  // the text size (window.js's applyZoom), within window.js's ZOOM_STEPS (80 to 175)
  var zoom = Number(pref.zoom);
  if (zoom >= 80 && zoom <= 175 && zoom !== 100) { html.style.zoom = String(zoom / 100); html.style.setProperty('--zoom', String(zoom / 100)); }
  // the dictionary, only for a zh reader, fetched now so the app does not wait for it (i18n.js's ensure)
  window.ALPHA_ASSETS = {zh: '{{zh_src}}'};
  if (zh) {
    var s = document.createElement('script');
    s.src = window.ALPHA_ASSETS.zh;
    s.async = false;
    s.setAttribute('data-alpha-zh', 'loading');
    s.addEventListener('load', function () { s.setAttribute('data-alpha-zh', 'ready'); });
    s.addEventListener('error', function () { s.setAttribute('data-alpha-zh', 'failed'); });
    (document.head || html).appendChild(s);
  }
})();
