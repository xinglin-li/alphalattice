/* The declaration editor's marks: what a changed draft leaves on its dependent surfaces (its line
 * numbers and painted copy are the shared code editor's, components.js). */
const Lab = (() => {
  function markDraftChanged() {
    const n = $('#validationNotice');
    if (n) n.innerHTML = html`<span class="caption">${icon('edit')} ${t('Declaration changed · a new PLAN is required.')}</span>`;
    $$('[data-action="research-confirm"]').forEach((x) => (x.disabled = true));
    const plan = $('#planStateNotice');
    if (plan) {
      plan.hidden = false;
      plan.innerHTML = banner(t('PLAN is out of date'), t('The declaration has changed. Create a new PLAN before confirming.'), 'warning', '', 'warning');
    }
    // A shared PLAN beside a dirty draft is inspected, not adopted over it.
    $$('[data-action="research-adopt-shared"]').forEach((x) => (x.textContent = t('Inspect and adopt')));
    $$('[data-action="research-shared-dismiss"]').forEach((x) => (x.textContent = t('Keep my draft')));
    if (Inspect.lens === 'declaration') Inspect.refreshLens();
    Controls.sync();
  }
  function onYamlInput(e) {
    e.target.removeAttribute('aria-invalid');
    app.yaml = e.target.value;
    app.plan = null;
    CodeEditor.input(e.target); // the numbers and the painted copy follow the text (components.js)
    markDraftChanged();
  }
  return {markDraftChanged, onYamlInput};
})();
