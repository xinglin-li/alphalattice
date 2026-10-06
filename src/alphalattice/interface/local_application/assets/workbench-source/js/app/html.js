/* Markup primitives. Every template that produces markup is written as html`…`. Interpolated values
 * are escaped, unless they are fragments themselves (a nested html`…`, an array of fragments, or a
 * string explicitly marked with raw()), so data can never inject markup. The templates read the same
 * way lit-html templates do, should the renderer ever change.
 *
 *   html`<li class="${cls}">${t('Label')} ${count}</li>`      text and attribute values: escaped
 *   html`<ul>${items.map((x) => html`<li>${x}</li>`)}</ul>`    arrays flattened, fragments kept
 *   html`<a href="${url}"${current ? ' aria-current="page"' : ''}>`
 *                                                               between attributes: trusted structure
 *   raw(engine.svg())                                           trusted markup from an engine
 *
 * The one place a plain string is not escaped is between a tag's attributes (after `<tag` and
 * outside quotes): only attribute fragments such as ` hidden` or ` data-x="1"` belong there, never
 * data. null and undefined render as nothing; booleans render as text (aria-pressed="${on}"). A
 * fragment stringifies to its markup, so it can be assigned to innerHTML. */
class Markup {
  constructor(markup) {
    this.markup = markup;
  }
  toString() {
    // Trusted template style declarations are applied through CSSOM under the
    // product's strict CSP. Data interpolation remains escaped above this seam.
    return window.ALPHA_PRODUCT ? this.markup.replace(/\sstyle=/g, ' data-ui-style=') : this.markup;
  }
}
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[c]);
/* A fragment inside a quoted attribute is its words (2026-09-22: a reading pane's title carried a
 * mono span into its aria-label and broke the tag): the tags go, the text keeps its entities and a
 * bare quote is escaped. */
const markupWords = (markup) => markup.replace(/<[^>]*>/g, '').replace(/"/g, '&quot;');
const renderValue = (value, structural, quoted = false) => {
  if (value instanceof Markup) return quoted ? markupWords(value.markup) : value.markup;
  if (Array.isArray(value)) return value.map((v) => renderValue(v, structural, quoted)).join('');
  if (value === null || value === undefined) return '';
  return structural ? String(value) : esc(value);
};
/* Where each interpolation of a template sits: 'text', 'tag' (between attributes) or 'value' (inside
 * a quoted attribute). Computed once per template literal. */
const contexts = new WeakMap();
function contextOf(strings) {
  let ctx = contexts.get(strings);
  if (ctx) return ctx;
  ctx = [];
  let state = 'text';
  let quote = '';
  for (let i = 0; i < strings.length - 1; i++) {
    const s = strings[i];
    for (let k = 0; k < s.length; k++) {
      const c = s[k];
      if (state === 'text') {
        if (c === '<' && /[a-zA-Z/!]/.test(s[k + 1] || '')) state = 'tag';
      } else if (state === 'tag') {
        if (c === '"' || c === "'") { state = 'value'; quote = c; } else if (c === '>') state = 'text';
      } else if (c === quote) state = 'tag';
    }
    ctx.push(state);
  }
  contexts.set(strings, ctx);
  return ctx;
}
const html = (strings, ...values) => {
  const ctx = contextOf(strings);
  return new Markup(values.reduce((out, value, i) => out + renderValue(value, ctx[i] === 'tag', ctx[i] === 'value') + strings[i + 1], strings[0]));
};
const raw = (markup) => (markup instanceof Markup ? markup : new Markup(String(markup ?? '')));
