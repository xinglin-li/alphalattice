/* Post-render helpers: unavailable controls explain themselves, layout facts are measured (never
 * written back as widths), and each page remembers its reading position. */

/* Every disabled control names its reason (visible note or screen-reader text + title). */
const Controls = (() => {
  const REASONS = {
    'research-confirm': 'The declaration changed. Create a new PLAN before confirming.',
    'session-prev': 'This is the earliest saved holdings session.',
    'session-next': 'This is the latest saved holdings session.',
  };
  let seq = 0;
  /* `source` is the English reason; the visible note and title are rendered in the current locale. */
  function setReason(el, source, visible = true) {
    let id = el.dataset.reasonId;
    if (!id) {
      id = 'control-reason-' + ++seq;
      el.dataset.reasonId = id;
    }
    let note = document.getElementById(id);
    if (!note && visible) {
      // One note per surface and reason: a second held control on the same surface shares it.
      const surface = el.closest('.panel,section,.dialog-body,.dialog-foot,form,.tp-section,.feature-row,.object-actions,.top-actions');
      const shared = surface && [...surface.querySelectorAll('.control-reason')].find((n) => n.dataset.source === source && !n.hidden);
      if (shared) {
        note = shared;
        id = shared.id;
        el.dataset.reasonId = id;
      }
    }
    if (!note) {
      note = document.createElement('span');
      note.id = id;
      // CT7: a held control says why by its tip; a printed line beside it moved its row (the user, 2026-09-25:
      // "当你点击Submit the assessment 容易出现混乱"). Only a dialog's foot, where the decision is, prints it.
      note.className = visible && el.closest('.dialog-foot') ? 'control-reason' : 'sr-only';
      el.insertAdjacentElement('afterend', note);
    }
    if (note.dataset.source !== source || note.textContent !== t(source)) {
      note.dataset.source = source;
      note.textContent = t(source);
    }
    note.hidden = false;
    const described = (el.getAttribute('aria-describedby') || '').split(/\s+/).filter((x) => x && x !== id);
    el.setAttribute('aria-describedby', [...described, id].join(' '));
    el.dataset.tip = t(source); // round 93: the reason is the tooltip's, never the browser's title
    el.dataset.reasonTitle = 'true';
  }
  function clearReason(el) {
    const id = el.dataset.reasonId;
    const others = id ? [...document.querySelectorAll(`[data-reason-id="${id}"]`)].some((x) => x !== el && x.disabled) : false;
    if (id && !others && document.getElementById(id)) document.getElementById(id).hidden = true;
    const rest = (el.getAttribute('aria-describedby') || '').split(/\s+/).filter((x) => x && x !== id).join(' ');
    if (rest) el.setAttribute('aria-describedby', rest);
    else el.removeAttribute('aria-describedby');
    if (el.dataset.reasonTitle) {
      delete el.dataset.tip;
      delete el.dataset.reasonTitle;
    }
  }
  /* Round 90 (law 75, the sentence budget): outside rows and tables a grey line is one clause. A
   * caption, a table's note, a form row's line, a note's explanation, an empty state's sentence or
   * a definition longer than the budget keeps its first clause; the rest folds behind an (i) the
   * peek card reads (`data-explain`). The words are the same; the page reads one line. A line
   * with a control or a document inside, a lede, and a row's own line are left as written; a
   * folded line carries its (i), so a repaint that keeps the element folds only fresh words. */
  const FOLD_AT = 90, FOLD_LINES = '.caption, .table-note, .form-line, .form-note, .note-line > span, .data-readiness-line > span, .section-empty > p, .kv > dd, .term-list dd, .sub-cell, .stat-basis > span, .field > small, .panel-body > p:not([class]), .proof-intro';
  const FOLD_ROOTS = ['#main', '#inspector', '#dialog'].map((r) => `${r} :is(${FOLD_LINES})`).join(', ');
  // every repaint is folded, whichever path painted it: the observer coalesces a paint's mutations into one pass
  let foldQueued = false;
  const foldObserver = typeof MutationObserver === 'function' ? new MutationObserver(() => { if (foldQueued) return; foldQueued = true; requestAnimationFrame(() => { foldQueued = false; foldLines(); }); }) : null;
  function watchFolds() { if (!foldObserver) return; for (const id of ['main', 'inspector', 'dialog']) { const el = document.getElementById(id); if (el && !el.dataset.foldWatched) { el.dataset.foldWatched = 'true'; foldObserver.observe(el, {childList: true, subtree: true, characterData: true}); } } }
  function foldLines(root = document) {
    watchFolds();
    for (const el of root.querySelectorAll(FOLD_ROOTS)) {
      if (el.closest('.list-row, tr, .card-list, .menu, .picker-pop, .object-lede, .lede')) continue;
      if (el.querySelector('button, a, input, textarea, details, pre, code, table, .hint, .hint-more')) continue;
      const text = el.textContent.replace(/\s+/g, ' ').trim();
      if (text.length <= FOLD_AT) continue;
      const cut = foldCut(text);
      if (cut < 0) continue;
      const point = textPoint(el, cut);
      if (!point) continue;
      const range = document.createRange();
      range.setStart(point.node, point.offset);
      range.setEndAfter(el.lastChild);
      const rest = range.toString().replace(/^[\s.;:\u00b7\u3002\uff1b]+/, '').trim();
      if (!rest) continue;
      range.deleteContents();
      const more = document.createElement('button');
      more.type = 'button'; more.className = 'hint-more'; more.dataset.tip = rest; more.setAttribute('aria-label', t('More'));
      more.innerHTML = icon('info');
      // the (i) follows the clause it holds the rest of (inside a sub-line, not after the block); a trailing separator before the cut is not a sentence's end
      const host = point.node.parentElement && el.contains(point.node.parentElement) ? point.node.parentElement : el;
      const last = host.lastChild;
      if (last && last.nodeType === 3) last.nodeValue = last.nodeValue.replace(/[\s;:,\u00b7\uff1b\uff1a\uff0c]+$/, ''); // a separator before the cut is not the clause's end
      host.append(more); // N6: the (i) brings its own space (info-mark-gap), no space character before it
    }
  }
  /* Where the first clause ends: the first sentence end, semicolon, colon or middle dot from the
   * 16th character on (a kicker before its middle dot is a clause); past the 100th, the first one
   * at all; with none, a comma between the 40th and the 100th -- a line with none stays. */
  function foldCut(text) {
    const rx = /[.;:\u3002\uff1b\uff1a](?=\s|$)|\s\u00b7\s/g;
    let m, first = -1;
    while ((m = rx.exec(text))) {
      const at = m.index + (m[0].startsWith(' ') ? 0 : 1);
      if (at >= text.length - 1) break; // the line's own end is not a cut
      if (first < 0 && m.index >= 16) first = at;
      if (m.index >= 16 && m.index <= 100) return at;
    }
    const comma = text.slice(40, 100).search(/,\s/);
    return comma >= 0 ? 40 + comma + 1 : first;
  }
  /* The text node and offset at the n-th character of an element's normalised text. */
  function textPoint(el, n) {
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    let seen = 0, node, lastSpace = true;
    while ((node = walker.nextNode())) {
      const raw = node.nodeValue;
      for (let i = 0; i < raw.length; i++) {
        const space = /\s/.test(raw[i]);
        if (space && lastSpace) continue; // collapsed whitespace does not count
        if (seen === n) return {node, offset: i};
        seen += 1; lastSpace = space;
      }
    }
    return null;
  }
  function sync(root = document) {
    for (const b of root.querySelectorAll('button')) {
      if (b.disabled) {
        b.dataset.uiState = 'disabled';
        if (b.hasAttribute('data-reason')) setReason(b, b.dataset.reason, true);
        else if (b.dataset.action === 'session-prev' || b.dataset.action === 'session-next') {
          setReason(b, REASONS[b.dataset.action], false);
          const parent = b.closest('.session-selector');
          if (parent) {
            let s = parent.querySelector('.session-limit');
            if (!s) {
              s = document.createElement('small');
              s.className = 'session-limit';
              parent.append(s);
            }
            s.textContent = t(REASONS[b.dataset.action]);
          }
        } else if (b.dataset.action === 'research-confirm') setReason(b, REASONS['research-confirm'], true);
        else if (!b.getAttribute('aria-describedby')) setReason(b, b.dataset.tip || b.title || 'This control is unavailable in the current product state.', true);
      } else if (b.dataset.uiState === 'disabled') {
        delete b.dataset.uiState;
        clearReason(b);
      }
    }
    const selector = $('.session-selector');
    if (selector && !selector.querySelector('button:disabled')) selector.querySelector('.session-limit')?.remove();
    for (const el of root.querySelectorAll('select:disabled,input:disabled,textarea:disabled')) {
      if (!el.getAttribute('aria-describedby')) {
        const factor = el.matches('[data-factor],#factorLimits');
        // an option in a choice list (a radio, a checkbox) carries its own reason line: the note is
        // for assistive technology only, never a visible column inside the option (round 22)
        const own = el.matches('input[type="radio"],input[type="checkbox"]');
        setReason(el, el.dataset.tip || el.title || 'Not available yet: it follows a choice or read that has not happened.', !own);
      }
    }
    const invalid = $('#yamlEditor');
    if (invalid?.getAttribute('aria-invalid') === 'true') $('#validationNotice')?.setAttribute('role', 'alert');
    foldLines(root);
    mountCopy(root);
    keepWalk();
  }
  /* Code and span blocks carry one copy affordance, mounted once per block. */
  function mountCopy(root = document) {
    for (const el of root.querySelectorAll('.code-block,.tp-json pre,.es-json')) {
      if (el.querySelector(':scope > .code-copy')) continue;
      el.classList.add('copy-host'); // the sheet places the affordance by its host, not by the block's kind
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'icon-btn code-copy';
      b.dataset.action = 'copy-block';
      b.setAttribute('aria-label', t('Copy this block'));
      b.dataset.tip = t('Copy this block');
      b.innerHTML = icon('copy');
      el.append(b);
    }
  }
  /* Busy means "preparing a local preview", never queued, running or computed. */
  function setPending(selector, on) {
    for (const b of $$(selector)) {
      if (on) {
        b.dataset.uiState = 'pending';
        b.setAttribute('aria-busy', 'true');
        b.setAttribute('aria-disabled', 'true');
        if (!b.querySelector('.busy-dot')) { // the label stays, dimmed, with the live dot before it
          const s = document.createElement('i');
          s.className = 'busy-dot';
          s.setAttribute('aria-hidden', 'true');
          b.prepend(s);
        }
      } else {
        delete b.dataset.uiState;
        b.removeAttribute('aria-busy');
        b.removeAttribute('aria-disabled');
        b.querySelector('.busy-dot')?.remove();
      }
    }
  }
  /* The keyboard walks a list (round 15): j / k and the arrows move the selection, Home / End
   * jump, Enter opens the row's way, Escape clears it -- only when no field, editor, dialog or
   * drawer has focus and no modifier is held. Round 52: one held row (`data-held`) in one list
   * or table at a time (the list under focus, else the one holding a row, else the first); the
   * held row is not the open record (`aria-current`). Tables walk like lists: every body row is
   * a row, Enter follows its first link or button. */
  const ROW = '[data-row], tbody > tr';
  const LISTS = '#main .card-list, #main .data-table tbody';
  const rowsOf = (list) => [...list.querySelectorAll(ROW)].filter((r) => r.parentElement === list || r.closest('.card-list, tbody') === list);
  let walkKey = '';
  function walk(e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return false;
    if (!['j', 'k', 'ArrowDown', 'ArrowUp', 'Home', 'End', 'Enter', 'Escape', ' '].includes(e.key)) return false;
    const a = document.activeElement, tag = a?.tagName || '';
    if (a && (a.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(tag))) return false;
    if (a && a.closest('[data-chart], .chart-navigator, .choice-list')) return false; // the chart, its navigator and the choice list own their keys (round 95)
    if ($('#dialog')?.open) return false;
    if (e.key === ' ' && a && a.matches('button:not(.list-row-main):not(.holding-name), a[href]') && !a.closest('[data-held]')) return false; // Space presses a control the reader focused
    const lists = $$(LISTS).filter((l) => l.offsetParent && rowsOf(l).length);
    if (!lists.length) return false;
    // the list: the one the focus is in, else the one holding a row, else the first one in view
    const top = $('#top') ? layoutRect($('#top')).bottom : 0;
    const inView = (l) => { const b = layoutRect(l); return b.bottom > top && b.top < viewH(); };
    let list = lists.find((l) => l.contains(a)) || lists.find((l) => l.querySelector('[data-held]')) || lists.find(inView) || lists[0];
    let rows = rowsOf(list);
    let idx = rows.findIndex((r) => r.hasAttribute('data-held'));
    // past the last row the walk continues into the next list, before the first into the previous
    const li = lists.indexOf(list);
    if ((e.key === 'j' || e.key === 'ArrowDown') && idx === rows.length - 1 && lists[li + 1]) { list = lists[li + 1]; rows = rowsOf(list); idx = -1; }
    else if ((e.key === 'k' || e.key === 'ArrowUp') && idx === 0 && lists[li - 1]) { list = lists[li - 1]; rows = rowsOf(list); idx = rows.length; }
    if (e.key === 'Enter') {
      if (idx < 0) return false;
      const main = rows[idx].querySelector('.list-row-main, a[href], button[data-action]');
      if (!main) return false;
      e.preventDefault(); main.click(); return true;
    }
    if (e.key === 'Escape') {
      if (idx < 0) return false;
      rows[idx].removeAttribute('data-held'); walkKey = ''; e.preventDefault(); return true;
    }
    if (e.key === ' ') {
      if (idx < 0) return false;
      e.preventDefault();
      if (Inspect.peekOpen() && Inspect.peekRowIs(rows[idx])) Inspect.closePeek(); else Inspect.peek(rows[idx]);
      return true;
    }
    let next = idx;
    if (e.key === 'j' || e.key === 'ArrowDown') next = Math.min(rows.length - 1, idx + 1);
    else if (e.key === 'k' || e.key === 'ArrowUp') next = idx < 0 ? 0 : Math.min(rows.length - 1, Math.max(0, idx - 1));
    else if (e.key === 'Home') next = 0;
    else next = rows.length - 1;
    $$('#main [data-held]').forEach((r) => r.removeAttribute('data-held'));
    rows[next].setAttribute('data-held', '');
    walkKey = rows[next].dataset.key || '';
    rows[next].scrollIntoView({block: 'nearest'});
    rows[next].querySelector('.list-row-main, a[href], button[data-action]')?.focus?.({preventScroll: true});
    Inspect.peekFollow(rows[next]);
    if (rows[next].classList.contains('tp-task') && typeof LiveTasks !== 'undefined') LiveTasks.follow(rows[next].dataset.key); // round 58: the pane follows
    e.preventDefault();
    return true;
  }
  /* Round 42: the layer's keys, one owner. The topmost open surface (a popover, the drawer, a
   * dialog; Quick Open keeps its input's keys) takes the arrows: Down / Up move focus among its
   * items (wrapping in a menu, clamped in a sheet), Home / End jump; the row walker never sees a
   * key while a surface is open. */
  const LAYER = ['.menu:not(.picker-pop):not([hidden])', '#dialog[open]:not(.quick-dialog)'];
  function layerKeys(e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return false;
    const surface = LAYER.map((sel) => $(sel)).find(Boolean);
    if (!surface) return false;
    const a = document.activeElement, tag = a?.tagName || '';
    if (a && (a.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(tag))) return false;
    const menu = surface.matches('.menu');
    const typeAhead = menu && !surface.querySelector('.menu-search') && /^[a-z0-9]$/i.test(e.key); // round 55: first letters where there is no field
    if (!typeAhead && !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) return false;
    if (!menu && !surface.contains(a)) return false;
    const items = [...surface.querySelectorAll('a[href], button:not([disabled]), [tabindex="0"]')].filter((x) => x.tabIndex >= 0 && x.getClientRects().length && !x.closest('[hidden]'));
    if (!items.length) return false;
    const idx = items.indexOf(a);
    let next = idx;
    if (typeAhead) {
      const hit = [...items.slice(idx + 1), ...items.slice(0, idx + 1)].find((x) => x.textContent.trim().toLowerCase().startsWith(e.key.toLowerCase()));
      if (!hit) return false;
      next = items.indexOf(hit);
    } else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = items.length - 1;
    else if (e.key === 'ArrowDown') next = menu ? (idx + 1) % items.length : Math.min(items.length - 1, idx + 1);
    else next = menu ? (idx - 1 + items.length) % items.length : Math.max(0, idx - 1);
    items[next].focus({preventScroll: false});
    items[next].scrollIntoView?.({block: 'nearest'});
    e.preventDefault();
    return true;
  }
  /* A repaint keeps the held row by its key. */
  function keepWalk() {
    if (!walkKey || $('#main [data-held]')) return;
    const row = $$('#main [data-row], #main tbody > tr').find((r) => r.dataset.key === walkKey);
    if (row) row.setAttribute('data-held', '');
  }
  /* Round 51: the keyboard map (Linear's): one table, read by the key handler and rendered as
   * the shortcuts sheet, so they never disagree. A chord ("G then H") waits 800 ms for its
   * second key. A row with `run` is bound here; a row without one documents a key another owner
   * implements. No row binds inside a field or while a layer is open. */
  const KEYMAP = [
    {group: 'General', keys: 'Ctrl K', words: 'Search pages, records and actions'},
    {group: 'General', keys: '?', single: '?', words: 'Keyboard shortcuts', run: () => Inspect.openShortcuts()},
    {group: 'General', keys: '/', single: '/', words: 'Search this page', run: () => focusSearch()},
    {group: 'General', keys: 'Esc', words: 'Back one layer'},
    {group: 'General', keys: 'Ctrl \\', words: 'Show or hide the navigation'},
    {group: 'General', keys: 'Ctrl ]', words: 'Show or hide the inspector'},
    {group: 'General', keys: 'Ctrl =', words: 'Larger text'},
    {group: 'General', keys: 'Ctrl −', words: 'Smaller text'},
    {group: 'General', keys: 'Ctrl 0', words: 'Reset text size'},
    {group: 'General', keys: 'Ctrl Enter', words: 'Save the goal'}, // the composer's own (live-goals)
    {group: 'General', keys: '← / →', words: 'Resize the navigation from its focused edge'}, // the edge's own (Window.keyResize)
    // one chord for every place the dock lists, in its order, named by the dock's word (the sheet
    // composes `Go to <word>`); the letters kept where the place kept its meaning (2026-09-25: the
    // map still said the Lab, Team, Portfolio and Evidence of rounds before, and five lists had none)
    {group: 'Navigation', keys: 'G then O', chord: 'g o', page: 'overview'},
    {group: 'Navigation', keys: 'G then K', chord: 'g k', page: 'tasks'},
    {group: 'Navigation', keys: 'G then D', chord: 'g d', page: 'data'},
    {group: 'Navigation', keys: 'G then G', chord: 'g g', page: 'goals'},
    {group: 'Navigation', keys: 'G then F', chord: 'g f', page: 'factor'},
    {group: 'Navigation', keys: 'G then N', chord: 'g n', page: 'foundation'},
    {group: 'Navigation', keys: 'G then A', chord: 'g a', page: 'alpha'},
    {group: 'Navigation', keys: 'G then R', chord: 'g r', page: 'risk'},
    {group: 'Navigation', keys: 'G then P', chord: 'g p', page: 'portfolio'},
    {group: 'Navigation', keys: 'G then E', chord: 'g e', page: 'books'},
    {group: 'Navigation', keys: 'G then T', chord: 'g t', page: 'team-sessions'},
    {group: 'Navigation', keys: 'G then H', chord: 'g h', page: 'history'},
    {group: 'Navigation', keys: 'G then ,', chord: 'g ,', page: 'settings'},
    {group: 'Navigation', keys: 'G then L', chord: 'g l', page: 'lab', words: 'New experiment'},
    {group: 'Lists', keys: 'J / ↓', words: 'Next row'},
    {group: 'Lists', keys: 'K / ↑', words: 'Previous row'},
    {group: 'Lists', keys: 'Home / End', words: 'First / last row'},
    {group: 'Lists', keys: 'Space', words: 'Peek at the row'},
    {group: 'Lists', keys: 'Enter', words: 'Open the row'},
    {group: 'Lists', keys: 'Esc', words: 'Release the row'},
    {group: 'Lists', keys: 'Shift V', single: 'V', words: 'Display options', run: () => toggleDisplayMenu()},
    {group: 'Records', keys: 'L', single: 'l', words: 'Copy link to this view', run: () => copyLink()},
    {group: 'Records', keys: '[', single: '[', words: 'Previous record in the list', run: () => Inspect.step(-1)},
    {group: 'Records', keys: ']', single: ']', words: 'Next record in the list', run: () => Inspect.step(1)},
    {group: 'Properties', keys: 'I', single: 'i', on: 'lab', words: 'Change the input version', run: () => pressProperty('i')},
    {group: 'Properties', keys: 'S', single: 's', on: 'lab', words: 'Change the screening policy', run: () => pressProperty('s')},
    {group: 'Properties', keys: 'R', single: 'r', on: 'lab', words: 'Change the redundancy policy', run: () => pressProperty('r')},
    {group: 'Properties', keys: 'T', single: 't', on: 'lab', words: 'Change the Alpha target', run: () => pressProperty('t')},
    {group: 'Properties', keys: 'M', single: 'm', on: 'lab', words: 'Change the model capability', run: () => pressProperty('m')},
    {group: 'Properties', keys: 'F', single: 'f', on: 'lab', words: 'Change the model family', run: () => pressProperty('f')},
    {group: 'Properties', keys: 'D', single: 'd', on: 'lab', words: 'Go to the interval and cutoff', run: () => pressProperty('d')},
    {group: 'Charts', keys: '← / →', words: "Step through a focused chart's sessions"}, // the chart's own (Portfolio.bindCharts)
    {group: 'Charts', keys: 'Shift ← / →', words: "Move a focused window edge a month"},
    {group: 'Charts', keys: 'Page Up / Page Down', words: "Move a focused window edge a quarter"},
    {group: 'Menus', keys: '↑ / ↓', words: 'Move in the open menu'},
    {group: 'Menus', keys: 'Home / End', words: 'First / last item'},
    {group: 'Menus', keys: 'Enter', words: 'Choose'},
    {group: 'Menus', keys: 'Esc', words: 'Close the menu'},
  ];
  for (const row of KEYMAP) if (row.page) row.run = () => navigate(row.page);
  const keymap = () => KEYMAP;
  const chordFor = (page) => KEYMAP.find((r) => r.page === page)?.keys || '';
  const SEARCH = '#inspector input[type="search"], #main input.search-input, #main input[type="search"], #main .es-search input, #main .search-wrap input';
  const inField = () => { const a = document.activeElement, tag = a?.tagName || ''; return Boolean(a && (a.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(tag))); };
  const layerOpen = () => Boolean($('#dialog')?.open || LAYER.some((sel) => $(sel)));
  function focusSearch() {
    const field = $$(SEARCH).find((f) => f.offsetParent && !f.disabled);
    if (!field) return false;
    field.focus(); field.select();
    return true;
  }
  let chord = '', chordTimer = 0;
  const clearChord = () => { chord = ''; clearTimeout(chordTimer); delete document.body.dataset.chord; };
  function keys(e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return false;
    if (inField()) {
      // Esc leaves the page's search field (the walk and the chords never read inside a field)
      if (e.key === 'Escape' && document.activeElement.matches(SEARCH)) { document.activeElement.blur(); e.preventDefault(); return true; }
      return false;
    }
    if (layerOpen()) { clearChord(); return false; }
    if (e.key === ' ' && document.activeElement?.matches?.(Inspect.HOVER_LINKS)) { e.preventDefault(); Inspect.hoverCard(document.activeElement, true); return true; } // round 58: a focused link's card
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (chord) {
      const row = KEYMAP.find((r) => r.chord === chord + ' ' + key);
      clearChord();
      if (row) row.run();
      e.preventDefault();
      return true; // the second key of a chord is never something else
    }
    if (key === 'g') {
      chord = 'g'; document.body.dataset.chord = 'g';
      chordTimer = setTimeout(clearChord, 800);
      e.preventDefault();
      return true;
    }
    const row = KEYMAP.find((r) => r.single === e.key && r.run && (!r.on || r.on === app.page)); // a property's key binds on its page
    if (!row) return false;
    e.preventDefault();
    row.run();
    return true;
  }
  /* Esc steps back one layer in one order: a menu (its own listener), the command menu and the
   * dialogs (the native dialog), the peek, the detail (the window's column or the lane's reading,
   * law 149), the drawer, focus mode, then the held row (the walk). Never two at once. */
  function escape(e) {
    if (e.key !== 'Escape' || $('#dialog')?.open) return false;
    if (Inspect.peekOpen()) { e.preventDefault(); Inspect.closePeek(); return true; }
    if (Window.detailOpen()) { e.preventDefault(); Window.closeDetail(); return true; } // round 62: the inspector is one layer; law 149: the reading is the same layer
    if (!Window.docked() && Window.sideOpen()) { e.preventDefault(); Window.closeSideOverlay(); return true; } // the side's overlay
    if (Inspect.focus) { e.preventDefault(); Inspect.toggleFocus(); return true; }
    return false;
  }
  return {sync, foldLines, setReason, clearReason, setPending, walk, keepWalk, layerKeys, keys, escape, keymap, chordFor, focusSearch};
})();

/* Layout is CSS-owned. The observer only reports measured boxes and content density. */
const Geometry = (() => {
  let ticket = 0;
  let geometry = null;
  let quiet = false;
  /* What a Tab can land on; a landmark or heading focused for reading is not revealed by centering. */
  const REVEALED = 'a[href],button,input,select,textarea,summary,[contenteditable="true"],[tabindex]:not([tabindex="-1"])';
  /* Focus put back by the product itself -- a repaint returning it to the same control, a
   * remembered place on return -- must not move the reader: the reveal on focus below is for
   * focus the reader moves (a Tab into a control hidden under the sticky header or the dock). */
  function focusQuietly(el) {
    quiet = true;
    try { el.focus({preventScroll: true}); } finally { quiet = false; }
  }
  // Law 149: a lane reading's height limit is the window under where it stands: it pins under the top
  // row once the page scrolls, and until then (opened near the list's top, under the page's head) its
  // foot stays in view -- `--detail-top`, in layout px like `--vh`, held by the lane so a reading a
  // repaint draws anew stands at its height from its first frame (never the pinned height for one).
  let fitTicket = 0;
  function fitDetails() {
    fitTicket = 0;
    const pane = document.querySelector('#main .reading-pane'), main = document.getElementById('main');
    if (pane && main) main.style.setProperty('--detail-top', Math.max(0, Math.round(layoutRect(pane).top)) + 'px');
    // a detail over the lane (the page's Facts or Record) stands under the page's head, as a reading does
    const shell = document.getElementById('inspector');
    if (shell?.dataset.open) {
      // the page's head: its header, then its facts line and its tabs that follow it
      let last = [...document.querySelectorAll('#main .object-header')].find((x) => x.getClientRects().length && !x.closest('.lane-split')) || null;
      for (let n = last?.nextElementSibling; n && n.matches('.object-context, .tabs, [role="tablist"]'); n = n.nextElementSibling) if (n.getClientRects().length) last = n;
      shell.style.setProperty('--detail-under', last ? Math.max(0, Math.round(layoutRect(last).bottom)) + 'px' : '0px');
    }
  }
  const scheduleFit = () => { if (!fitTicket) fitTicket = requestAnimationFrame(fitDetails); };
  function update() {
    ticket = 0;
    fitDetails();
    const b = document.body;
    const main = $('#main');
    const wrap = $('.lane-frame');
    if (!main || !wrap) return;
    const box = layoutRect(main);
    const frame = layoutRect(wrap); // layout px (round 95): the reading space is judged in the layout's measure, not the zoomed one
    b.dataset.reading = readingRole();
    geometry = {page: app.page, profile: readingRole(), viewport: document.documentElement.clientWidth, frameLeft: frame.left, frameWidth: frame.width, contentLeft: box.left, contentWidth: box.width, authority: 'CSS_FRAME_1440', horizontalWrites: false};
  }
  function schedule() {
    if (!ticket) ticket = requestAnimationFrame(update);
  }
  function bind() {
    new MutationObserver(schedule).observe(document.body, {attributes: true, attributeFilter: ['class']});
    const ro = new ResizeObserver(schedule);
    ro.observe($('#main'));
    ro.observe($('#top'));
    window.addEventListener('resize', schedule, {passive: true});
    window.addEventListener('scroll', scheduleFit, {passive: true});
    document.addEventListener('visibilitychange', () => {
      document.body.dataset.pageHidden = String(document.hidden);
    });
    document.body.dataset.pageHidden = String(document.hidden);
    // The reveal is for focus the keyboard moves into a control (a Tab into a field hidden
    // under the sticky header or the dock). A pointer focuses what it clicked, already in view;
    // a click on blank space focuses `#main` or a sectioning element with tabindex="-1", and
    // centering one of those (taller than the window) is the page jumping for no reason.
    let pointer = false;
    // The body says which kind of input came last (round 24h): a focus the page restores after a
    // pointer interaction shows no ring; the first key press brings the rings back.
    document.addEventListener('pointerdown', () => { pointer = true; document.body.dataset.input = 'pointer'; }, true);
    document.addEventListener('keydown', () => { pointer = false; document.body.dataset.input = 'keyboard'; }, true);
    document.addEventListener('focusin', (event) => {
      const el = event.target;
      if(quiet || pointer || !el.closest?.('#main') || !el.matches?.(REVEALED)) return;
      const box=layoutRect(el);
      const top=layoutRect($('#top')).bottom;
      const bottom=viewH();
      if(box.height > bottom - top) return;
      if(box.top<top || box.bottom>bottom) el.scrollIntoView({block:'center',behavior:'instant'});
    });
  }
  return {
    update,
    schedule,
    bind,
    focusQuietly,
    fit: fitDetails,
    get geometry() {
      return geometry;
    },
  };
})();

/* In-tab UI memory (scroll, disclosure and focus per view), not job recovery. */
const Places = (() => {
  const views = new Map();
  let rendered = null;
  let restoring = false;
  let ticket = 0;
  const key = () => app.workspace + '|' + app.page;
  function capture() {
    if (!rendered || restoring) return;
    const main = $('#main');
    const active = document.activeElement;
    const scrolls = {};
    // N5 (law 113): the Team's thread is the page's own scroll, kept as every page's is
    for (const sel of ['#yamlEditor', '.table-scroll']) {
      const el = $(sel);
      if (el) scrolls[sel] = {top: el.scrollTop, left: el.scrollLeft};
    }
    const details = [...main.querySelectorAll('details')].map((e) => {
      const selector = e.id ? '#' + CSS.escape(e.id) : '#main details';
      return {selector, index: [...document.querySelectorAll(selector)].indexOf(e), open: e.open};
    });
    let focus = null;
    if (main.contains(active)) {
      if (active.id) focus = {id: active.id};
      else if (active.dataset?.action) focus = {action: active.dataset.action, value: active.dataset.value || ''};
      if (focus && typeof active.selectionStart === 'number') {
        focus.start = active.selectionStart;
        focus.end = active.selectionEnd;
      }
    }
    views.set(rendered, {y: scrollY, x: scrollX, scrolls, details, focus});
  }
  function restore(k, focus = false) {
    const p = views.get(k);
    if (!p) return;
    restoring = true;
    for (const d of p.details) {
      const e = document.querySelectorAll(d.selector)[d.index];
      if (e) e.open = d.open;
    }
    for (const [sel, pos] of Object.entries(p.scrolls)) {
      const e = $(sel);
      if (e) {
        e.scrollTop = pos.top;
        e.scrollLeft = pos.left;
      }
    }
    if (focus && p.focus) {
      const e = p.focus.id ? document.getElementById(p.focus.id) : [...$('#main').querySelectorAll('[data-action]')].find((x) => x.dataset.action === p.focus.action && (x.dataset.value || '') === p.focus.value);
      if (e) {
        Geometry.focusQuietly(e);
        if (p.focus.start !== undefined && typeof e.setSelectionRange === 'function') {
          try {
            e.setSelectionRange(p.focus.start, p.focus.end);
          } catch {
            /* not a text control */
          }
        }
      }
    }
    scrollTo({top: p.y, left: p.x, behavior: 'instant'});
    restoring = false;
  }
  /* Called at the end of render(): remember the new key and restore a known place. */
  function restoreAfterRender(previousKey, pageChanged) {
    const k = key();
    rendered = k;
    const token = ++ticket;
    if (views.has(k)) {
      restore(k, false);
      requestAnimationFrame(() => {
        if (token === ticket && key() === k) restore(k, pageChanged);
      });
    }
  }
  return {
    key,
    capture,
    restore,
    restoreAfterRender,
    forget: (k) => views.delete(k),
    remember: (k, place) => views.set(k, place),
    get: (k) => views.get(k),
    all: () => Object.fromEntries(views),
    reset: () => (rendered = null),
  };
})();
