/* Boot: viewer preferences, listeners, first render, the workspace read. */
(function boot() {
  const entry = window.ALPHA_ENTRY || {page: 'overview'};
  const initial = hashParams();

  // Viewer preferences are independent of the route.
  if (readPreference('opaqueControls')) document.body.classList.add('opaque-controls'); // law 150: the glass off (the parameters' `opaque` mode)
  if (readPreference('wideScrollbars')) document.body.classList.add('wide-scrollbars');
  if (readPreference('reduceFocusEffects')) document.body.classList.add('reduced-focus');
  const theme = initial.get('theme') || readPreference('theme') || app.theme;
  app.theme = THEMES.includes(THEME_ALIASES[theme] || theme) ? (THEME_ALIASES[theme] || theme) : 'follow';
  if (typeof matchMedia === 'function') matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (app.theme === 'follow') { applyTheme(); Window.renderTop(); } }); // following the host, live
  const lang = initial.get('lang') || readPreference('lang') || 'en';
  I18N.set(lang);

  Window.bind(() => render()); // the page's repaint, for a setting changed on the Settings page
  Geometry.bind();
  Events.bind();
  LiveViews.bind(); LiveTasks.bind(); LiveWorkspace.bind();

  if (!initial.has('page')) app.page = ROUTES[entry.page] ? entry.page : 'overview';
  readRoute();
  // A zh reader's first paint waits for the dictionary the prelude asked for (round 96); an
  // English reader's does not wait for anything.
  const dictionary = I18N.ensure(lang);
  const start = () => { render(); Data.connect(); };
  if (dictionary) dictionary.then(start); else start();
  // An error nobody caught (round 96): the card when the page is blank, the toast when it is drawn.
  const uncaught = (error) => {
    const main = $('#main');
    if (!main || !main.childElementCount) return renderFailure(error, 'uncaught');
    console.error('Uncaught', error);
    notify(t('Something failed: {message}', {message: String((error && error.message) || error || '')}));
  };
  window.addEventListener('error', (e) => uncaught(e.error || e.message));
  window.addEventListener('unhandledrejection', (e) => uncaught(e.reason));
  /* Read-only inspection surface for tests and tooling. */
  window.AlphaLattice = Object.freeze({
    version: '3.14',
    state: () => clone(app),
    routes: Object.keys(ROUTES),
    navigate,
    render,
    setTheme,
    setLocale,
    failNextRender: () => { app.failNextRender = true; },
    data: Data,
    untranslated: I18N.untranslated,
    i18nKeys: I18N.keys,
    action: (name, value) => dispatchAction(name, value),
    hasAction: (name) => Object.hasOwn(PRODUCT_ACTIONS, name) || READ_ACTIONS.has(name),
    actions: () => [...new Set([...Object.keys(PRODUCT_ACTIONS), ...READ_ACTIONS])],
  });
})();
