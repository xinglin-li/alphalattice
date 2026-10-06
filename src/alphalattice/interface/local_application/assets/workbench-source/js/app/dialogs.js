/* Native dialog, toast, PLAN confirmation and the read-only inspection dialogs shared by pages. */
let returnFocus = null;
let toastTimer = null;

let toastMessage = '';
let toastVars = null;
let toastAct = null;
/* `message` is the English source text (`{name}` slots from `vars`); a language switch re-renders it.
   `act` ({action, value, word}): one press on what the toast is about -- a Task that ended opens. */
function notify(message, vars = null, act = null) {
  clearTimeout(toastTimer);
  toastMessage = message;
  toastVars = vars;
  toastAct = act;
  const failure=refusalParts(message), full=failure.detail ? `${failure.code}: ${said(failure.detail)}` : t(message,vars), source=String(message || '').toLowerCase();
  const brief=source==='link copied' ? t('Link copied') : source.includes('copied') ? t('Copied') : source.includes('filter')&&source.includes('clear') ? t('Filters cleared') : source.includes('verified') ? t('Result verified') : source.includes('cancel') ? t('Task cancelled') : source.includes('export') ? t('Exported') : source.includes('refus') ? t('Action refused') : full.trim().split(/\s+/).length<=4 ? full : t('Action needs attention');
  // round 56: the toast quotes the id it is about and shows a mark for what happened; a Task that
  // ended is said by its name (2026-09-25), cut at its measure and whole on hover
  const ref = vars?.ref ? String(vars.ref) : '', name = vars?.name ? String(vars.name) : '';
  $('#toastText').innerHTML = ref ? html`<code class="line-cut">${ref}</code> ${brief.toLowerCase()}` : name ? html`<b class="line-cut toast-name">${name}</b> ${brief.toLowerCase()}` : html`${brief}`;
  $('#toastIcon').innerHTML = icon(/copied|verified|exported|completed/.test(source) ? 'check' : 'info');
  const slot = $('#toastAction'); if (slot) slot.innerHTML = act ? btn(t(act.word), act.action, act.value, 'text-btn toast-action') : '';
  cancelLeave($('#toast')); $('#toast').hidden = false;
  toastTimer = setTimeout(hideToast, Math.min(8000, Math.max(3500, 1500 + 40 * String($('#toastText').textContent || '').length)));
}
function refreshToast() {
  if (!$('#toast').hidden) notify(toastMessage, toastVars, toastAct);
}
function hideToast() {
  clearTimeout(toastTimer);
  leave($('#toast'), () => { $('#toast').hidden = true; });
}

function closeDialog() {
  LiveResearch.dismissConfirmation(); LiveTasks.dismissConfirmation();
  LiveReview.dismissConfirmation();
  LiveStudy.dismissConfirmation();
  const d = $('#dialog');
  const wasOpen = d.open;
  document.body.classList.remove('modal-open');
  if (!wasOpen) { d.removeAttribute('aria-label'); return; }
  // round 94: the dialog leaves the way it came (its entrance reversed over --dur-1), then closes
  // and the focus returns to the opener
  leave(d, () => {
    d.close();
    d.removeAttribute('aria-label'); // a palette's name; the chassis is named by its title
    if (!returnFocus) return;
    const again = returnFocus.isConnected ? returnFocus : returnFocus.dataset?.action ? document.querySelector(`[data-action="${returnFocus.dataset.action}"]`) : null; // round 93: a repaint replaced the opener; its action names it
    if (again) again.focus({preventScroll: true});
  });
}
/* The palette (round 93): the dialog element with no head, no close glyph and no foot -- the input row is
 * the first row, the results the rest; it is named for assistive technology, not by a title. */
function openPalette(body, again = null) {
  const d = $('#dialog');
  Dialog.remember(again);
  cancelLeave(d);
  if (!d.open) returnFocus = document.activeElement;
  if (d.open) d.close();
  d.className = 'quick-dialog';
  d.setAttribute('aria-label', t('Quick Open'));
  d.innerHTML = body;
  document.body.classList.add('modal-open');
  d.showModal();
  Controls.sync(d);
}
/* One chassis: eyebrow, title, close, body, an optional footer; `sheet` is true or a kind (`alert`). */
function openDialog(eyebrow, heading, body, footer = '', sheet = false, again = null) {
  const d = $('#dialog');
  Dialog.remember(again);
  cancelLeave(d);
  if (!d.open) returnFocus = document.activeElement;
  if (d.open) d.close();
  d.className = sheet === true ? 'sheet' : sheet || '';
  d.innerHTML = html`<header class="dialog-head"><div><p class="dialog-kind">${eyebrow}</p><h2 id="dialogTitle">${heading}</h2></div>${btnAttrs(icon('close'), 'close', '', 'icon-btn', html`aria-label="${t('Close dialog')}"`)}</header><div class="dialog-body">${body}</div>${String(footer || '').trim() ? html`<footer class="dialog-foot">${footer}</footer>` : ''}`;
  document.body.classList.add('modal-open');
  d.showModal();
  Controls.sync(d);
}
/* The code dialog (round 92): a document read whole -- the object's name as the eyebrow, the
 * document's title, a copy glyph beside the close, the text in a block that never wraps and
 * scrolls both ways, `Close` in the foot. Nothing else in the product shows raw JSON or YAML. */
function codeDialog(title, kind, text, {lang = 'json'} = {}) {
  const body = html`<pre class="code-block code-document" tabindex="0" aria-label="${title}">${raw(codeMarkup(text, lang))}</pre>`;
  openDialog(kind || t('Document'), title, body, '', 'code-dialog');
  const d = $('#dialog'), close = d.querySelector('.dialog-head > .icon-btn');
  if (close) {
    const tools = document.createElement('div');
    tools.className = 'dialog-tools';
    close.replaceWith(tools);
    tools.insertAdjacentHTML('beforeend', btnAttrs(icon('copy'), 'code-copy', '', 'icon-btn', html`aria-label="${t('Copy the document')}" data-tip="${t('Copy the document')}"`));
    tools.append(close);
  }
  Controls.sync(d);
}
/* The document a `codeRef` link carries, by its id. The eyebrow is the open object's name (as the
 * Facts panel's), else the page's. */
function openCodeRef(id) {
  const tpl = document.querySelector(`template[data-code-id="${id}"]`);
  if (!tpl) return false;
  const name = String(document.querySelector('#main .object-header h1')?.textContent || '').replace(/\s+/g, ' ').trim() || t(ROUTES[app.page]?.[1] || '');
  codeDialog(tpl.dataset.codeTitle || t('Document'), name, tpl.content.textContent, {lang: tpl.dataset.codeLang || 'json'});
  return true;
}
function showInfo(title, text) {
  openDialog(t('Detail · prototype'), title, html`<p class="small muted">${text}</p>`, '');
}

/* ---- clipboard and exports ---- */
function copyText(text, word = 'Copied') {
  const vars = /^\S{4,64}$/.test(text) && !/^https?:/.test(text) ? {ref: text} : null; // a copied id is quoted in the toast; a link is not
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text).then(() => notify(word, vars)).catch(() => copyFallback(text));
  copyFallback(text);
}
function copyFallback(text) {
  openDialog(t('Copy · local fixture'), t('Copy this context'), html`<p class="caption">${t('Clipboard permission is unavailable. Select and copy this text; no remote service is used.')}</p><textarea id="copyField" class="yaml-editor copy-field" readonly aria-label="${t('Copyable context')}">${(text)}</textarea>`, '');
  $('#copyField').focus();
  $('#copyField').select();
}
function download(text, name, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], {type}));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 3000);
  notify('Exported');
}
function experienceSettings() { navigate('settings'); } // round 62: the comfort settings are the Settings page's Reading section
