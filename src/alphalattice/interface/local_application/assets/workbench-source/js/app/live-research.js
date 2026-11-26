/* Product declaration/preview interaction. Existing application operations own execution. */
const LiveResearch = (() => {
  /* The page repaints in place: the sections whose markup changed are replaced, the rest --
   * the header, the rail, the entry hero, an editor the reader is typing in -- stay as they
   * are. A navigation to the page (another page was on screen) is a full render. */
  const paint = () => { if (typeof patchMain === 'function') patchMain(); else render(); };
  const S = {initialized:false, kind:'factor.screening-development', input:'', binding:'',
    document:null, fields:[], request:{}, plan:null, revision:0, busy:'', error:'', dirty:false,
    ready:false, origin:null, handoff:null, missing:[], authoringOptions:null, featureInput:null, submission:null, confirming:null,
    shared:null, sharedTicket:0, adopting:null, adoptionSerial:0,
    // the reader's explicit editor view for this draft; the field a PLAN refusal named; the
    // guided Alpha/Portfolio entry; the admitted Task read in the work area; an exact reuse
    modeChoice:null, errorField:'', guide:null, work:null, reuse:null};
  /* A PLAN operated elsewhere (CLI, Codex) is read by its exact hash and shown beside the editor.
   * It never replaces the draft on its own: adopting is explicit, and a dirty draft is shown the
   * differences first. Reading a shared PLAN grants nothing and runs nothing. */
  const SHARED_TONE={AVAILABLE:'planned',EXPIRED:'expired',INVALID:'blocked',MISSING:'metadata',ADMITTED_AS_TASK:'queued'};
  const SHARED_MEANING={
    AVAILABLE:'This PLAN is retained and runnable by whoever confirms it in their own editor; reading it grants nothing.',
    EXPIRED:'This PLAN passed its expiry. It can be read, not run; PLAN it again explicitly before running.',
    INVALID:'The workspace moved beneath this PLAN; PLAN it again explicitly.',
    MISSING:'This service holds no preview under this hash: it expired, was displaced by newer previews, or the service restarted. Only an explicit new PLAN recreates it.',
    ADMITTED_AS_TASK:'This PLAN was already admitted as a Task; read the Task, not the preview.'};
  async function inspectShared(hash, wanted=()=>true) {
    if(!hash || !wanted()) return false;
    const ticket=++S.sharedTicket;
    S.shared={hash,ticket,loading:true,body:null,error:''};
    S.initialized=true; app.page='lab'; replaceHash({page:'lab',plan:hash}); dropReview(); paint();
    try {
      const body=await Data.read('/api/experiments/preview?'+new URLSearchParams({experiment_plan_hash:hash}));
      if(ticket!==S.sharedTicket || !wanted()) return false; // a later inspect or manual choice owns the panel
      S.shared={hash,ticket,loading:false,body,error:''};
      if(body?.status==='AVAILABLE') openReader(); // round 67 / 69: what is new for the person comes first — the reader opens on the shared PLAN
      return true;
    } catch(e) { if(ticket===S.sharedTicket) S.shared={hash,ticket,loading:false,body:null,error:e.message}; }
    if(ticket===S.sharedTicket && wanted()) paint();
    return wanted();
  }
  function dismissShared() { S.shared=null; S.sharedTicket++; replaceHash({plan:''}); dropReview(); paint(); }
  /* An adoption review is one transaction: the exact shared answer (hash + ticket), the draft
   * revision it was compared against and the action asked for. Anything else that arrives
   * before the confirmation -- another inspection, a new answer, an edit -- ends it. */
  function dropReview() { if(S.adopting) { S.adopting=null; closeDialog(); } }
  /* An unsaved draft worth protecting: edited, not merely loading, and not empty. */
  const unsaved=()=>S.dirty && S.busy!=='load' && app.yaml!=='';
  /* Line differences between the draft and the shared declaration: what adopting would change.
   * Exact (LCS) inside a bounded middle block after the common prefix and suffix; a larger block
   * is shown as the draft lines removed then the shared lines added, and says so. */
  const DIFF_CELLS=250000, DIFF_SHOWN=400;
  function lcsLines(a, b) {
    const n=a.length, m=b.length, table=Array.from({length:n+1},()=>new Array(m+1).fill(0));
    for(let i=n-1;i>=0;i--) for(let j=m-1;j>=0;j--) table[i][j]=a[i]===b[j] ? table[i+1][j+1]+1 : Math.max(table[i+1][j],table[i][j+1]);
    const out=[]; let i=0,j=0;
    while(i<n && j<m) { if(a[i]===b[j]) { out.push([' ',a[i]]); i++; j++; } else if(table[i+1][j]>=table[i][j+1]) out.push(['-',a[i++]]); else out.push(['+',b[j++]]); }
    while(i<n) out.push(['-',a[i++]]); while(j<m) out.push(['+',b[j++]]);
    return out;
  }
  function diffLines(mine, theirs) {
    const a=mine.split('\n'), b=theirs.split('\n');
    let start=0; while(start<a.length && start<b.length && a[start]===b[start]) start++;
    let endA=a.length, endB=b.length; while(endA>start && endB>start && a[endA-1]===b[endB-1]) { endA--; endB--; }
    const exact=(endA-start)*(endB-start)<=DIFF_CELLS;
    const middle=exact ? lcsLines(a.slice(start,endA),b.slice(start,endB)) : [...a.slice(start,endA).map(l=>['-',l]),...b.slice(start,endB).map(l=>['+',l])];
    const lines=[...a.slice(0,start).map(l=>[' ',l]),...middle,...a.slice(endA).map(l=>[' ',l])];
    return {lines,exact,removed:middle.filter(([k])=>k==='-').length,added:middle.filter(([k])=>k==='+').length};
  }
  function diffView(diff) {
    const first=Math.max(0,diff.lines.findIndex(([k])=>k!==' ')-3);
    const shown=diff.lines.slice(first,first+DIFF_SHOWN), hidden=diff.lines.length-shown.length;
    return html`${first ? html`<p class="caption">${countText(first, '{n} unchanged line above', '{n} unchanged lines above')}</p>` : ''}<pre class="code-block">${raw(shown.map(([k,line])=>codeEsc(k+' ')+codeMarkup(line,'yaml')).join('\n'))}</pre>${hidden>first ? html`<p class="caption">${countText(hidden-first, '{n} more line not shown', '{n} more lines not shown')}</p>` : ''}`;
  }
  function reviewAdoption(action) {
    const sh=S.shared, diff=diffLines(app.yaml,sh.body.yaml);
    S.adopting={id:++S.adoptionSerial,hash:sh.hash,ticket:sh.ticket,revision:S.revision,action};
    openDialog(t('Shared PLAN · your draft'),t(action==='replan' ? 'Adopt the shared declaration over your unsaved draft and PLAN it again?' : 'Adopt the shared declaration over your unsaved draft?'),
      html`<p class="caption">${t('Adopting replaces {removed} of your draft with {added} of the shared declaration. Your draft is kept unless you adopt.',{removed:countText(diff.removed,'{n} line','{n} lines'),added:countText(diff.added,'{n} line','{n} lines')})}${diff.exact ? '' : html` ${t('The differing block is too large to compare line by line; it is shown as your lines removed, then the shared lines added.')}`}</p>${diffView(diff)}`,
      html`${btn(t(action==='replan' ? 'Adopt and PLAN again' : 'Adopt shared declaration'),'research-adopt-confirm',String(S.adopting.id),'button primary')}`);
  }
  /* The confirmed transaction must still be the reviewed one: same shared answer, same draft. */
  function confirmAdoption(id) {
    const txn=S.adopting;
    if(!txn || String(txn.id)!==String(id)) { closeDialog(); throw Error('This confirmation is no longer current; review the shared PLAN again.'); }
    S.adopting=null;
    const sh=S.shared;
    if(!sh?.body || sh.hash!==txn.hash || sh.ticket!==txn.ticket || S.revision!==txn.revision) {
      closeDialog(); paint();
      throw Error('The shared PLAN or your draft changed since this review; nothing was adopted. Review it again.');
    }
    return performAdoption(txn.action);
  }
  function performAdoption(action) {
    const b=S.shared.body;
    closeDialog(); changed();
    const request={...(b.next_requests?.replan || {})}; delete request.operation; delete request.experiment_document;
    adopt({document:b.document,yaml:b.yaml,research_input_id:b.research_input_id,input_binding_hash:b.input_binding_hash,plan_request:request},b.origin_task_id || null);
    S.handoff=null; S.initialized=true; replaceHash({page:'lab',origin:b.origin_task_id || '',draft_source:'',plan:S.shared.hash});
    const ticket=S.revision;
    fieldSpecs({document:b.document,research_input_id:b.research_input_id,input_binding_hash:b.input_binding_hash}).then(fields=>{ if(ticket===S.revision){ S.fields=fields; settleMode(); paint(); } }).catch(()=>{});
    paint();
    if(action==='replan') return preview().then((sent)=>{ if(!sent) throw Error('The adopted declaration was not PLANned: the editor is busy or not ready. Create the PLAN explicitly.'); return true; }); // exactly one PLAN; never RUN
    return true;
  }
  /* Adopt the shared declaration into the editor. A dirty draft is reviewed first (the
   * differences, bound to this exact answer and revision); a clean draft adopts at once. */
  function adoptShared(action='adopt') {
    const b=S.shared?.body;
    if(!b || typeof b.yaml!=='string') throw Error('This shared PLAN has no readable declaration to adopt.');
    if(S.busy==='run') throw Error('Wait for the current submission to return.');
    if(unsaved()) { reviewAdoption(action); return false; }
    return performAdoption(action);
  }
  const replanShared=()=>adoptShared('replan');
  function impactWords(impact) {
    const change=impact?.change, upstream=impact?.execution?.preserved_upstream || [];
    return {
      change:change?.origin_task_id ? t('Input: {input}; program: {program}; implementation: {implementation}. The original result remains historical.',{
        input:t(change.input_changed ? 'impact|Changed' : 'impact|Unchanged'),
        program:t(change.program_changed ? 'impact|Changed' : 'impact|Unchanged'),
        implementation:t(change.implementation_changed ? 'impact|Changed' : 'impact|Unchanged'),
      }) : '',
      upstream:upstream.length ? t('No new upstream work is planned for: {work}. RUN revalidates the bindings.',{work:upstream.map(codeWords).join(', ')}) : '',
    };
  }
  function sharedPanel() {
    const sh=S.shared; if(!sh) return '';
    if(sh.loading) return noteLine(t('Reading the shared PLAN'),short(sh.hash, SHORT.hash));
    if(sh.error) return banner(t('Shared PLAN unreadable'),sh.error,'warning',btn(t('Dismiss'),'research-shared-dismiss','','button compact'));
    const b=sh.body, status=b.status || 'MISSING';
    const rows=[[t('Plan identity'),mono(sh.hash,SHORT.hash)],[t('Preview state'),stateLine(SHARED_TONE[status] || 'metadata',{word:codeWords(status)})]];
    if(b.caller) rows.push([t('Previewed by'),html`${actorWords(b.caller, b.producer_id)} · ${b.previewed_at || ''}`]);
    if(b.expires_at) rows.push([t('Runnable until'),b.expires_at]);
    if(b.research_input_id) rows.push([t('Exact input'),html`${b.research_input_id} · ${mono(b.input_binding_hash)}`]);
    if(b.program?.kind) rows.push([t('Method'),b.program.kind]);
    if(b.existing_task) rows.push([t('Admitted task'),html`${mono(b.existing_task.task_id,SHORT.id)} · ${codeWords(b.existing_task.lifecycle)} ${btn(t('Inspect task'),'task',b.existing_task.task_id,'button compact')}`]);
    if(b.invalidated_by) rows.push([t('Invalidated by'),coded(b.invalidated_by)]); // the owner's code in words (WD2)
    const impact=impactWords(b.impact);
    if(impact.change) rows.push([t('Compared with the origin'),impact.change]);
    if(impact.upstream) rows.push([t('Will reuse'),impact.upstream]);
    if(b.impact?.execution?.work_if_new) rows.push([t('Work if new'),codeWords(b.impact.execution.work_if_new)]);
    const readable=typeof b.yaml==='string';
    return html`<section class="panel pad"><h3>${t('Shared PLAN')}</h3><h2>${t('Research context operated elsewhere')}</h2>${kv(rows)}<p class="caption">${t(SHARED_MEANING[status] || SHARED_MEANING.MISSING)}</p>${readable ? html`${codeRef(t('Read the exact declaration'), String(b.yaml || ''), 'yaml')}` : ''}<div class="flow">${readable ? btn(t(unsaved() ? 'Inspect and adopt' : 'Adopt into editor'),'research-adopt-shared','','button compact') : ''}${readable && status!=='ADMITTED_AS_TASK' ? btn(t('Adopt and PLAN again'),'research-replan-shared','','button compact') : ''}${btn(t(unsaved() ? 'Keep my draft' : 'Dismiss'),'research-shared-dismiss','','button compact')}</div></section>`;
  }
  /* ---- what the Lab creates, and where each kind starts ---- */
  const kinds = [['factor.screening-development','Factor screening'], ['alpha.model-development','Alpha modeling'],
    ['risk.covariance-development','Risk modeling'], ['portfolio.policy-development','Portfolio study']]; // the research chain's order (N6); the dock's names (law 134)
  const kindLabel=(kind)=>t(kinds.find(v=>v[0]===kind)?.[1] || 'Research declaration');
  /* Alpha and Portfolio have no standalone declaration: the owner admits their drafts only from
   * saved sources (a Factor decision or a sealed Foundation; a saved Alpha candidate). */
  const GUIDED=new Set(['alpha.model-development','portfolio.policy-development']);
  const KIND_LINES={
    'factor.screening-development':'Screens the declared factors on the exact input version. PLAN previews the work; one explicit confirmation admits one Factor Task whose evidence report is read on the Factor page.',
    'risk.covariance-development':'Evaluates the installed covariance estimator over the declared formations of the exact input. One explicit confirmation admits one Risk Task whose diagnostics are read on the Risk page; a Risk report never sizes a Portfolio.',
    'alpha.model-development':'Fits one explicit model declaration on the factors a saved research decision carries forward. One explicit confirmation admits one Alpha Task with development candidates; opening the draft fits nothing.',
    'portfolio.policy-development':'Applies the declared policy to the saved scores of one evaluated Alpha candidate; no model is refitted. One explicit confirmation admits one Portfolio Task with a saved book.'};
  /* The owners' typed refusals, in words a reader can act on. The code stays beside the text. */
  const CODES={
    'factor_research.authoring_factor_id_not_in_inventory':'A declared factor is not in this input version\'s factor inventory; choose factors from the list.',
    'factor_research.authoring_factor_ids_invalid':'Declare at least one factor from the inventory.',
    'research_authoring.session_range_invalid':'The interval start must not be after its end, and the cutoff must lie within the input version.',
    'research_authoring.document_unparsable':'The YAML could not be parsed; fix its syntax and PLAN again.',
    'research_experiment.document_invalid':'The declaration does not match the method schema; the named field says where.',
    'research_experiment.preview_required':'No PLAN is retained for this declaration (it expired, was displaced or the service restarted); create the PLAN again, then confirm.',
    'research_experiment.preview_expired':'The PLAN passed its expiry before confirmation; create it again, then confirm.',
    'research_experiment.workspace_changed':'The workspace changed beneath this declaration; PLAN again.',
    'research_experiment.standalone_kind_not_installed':'This kind has no standalone declaration; it starts from the saved sources chosen on this page.',
    'research_experiment.input_selection_required':'Choose one exact input version.',
    'research_experiment.input_not_admitted':'The requested research input is not one this workspace admits for experiments; choose one of the versions offered here.',
    'RESEARCH_INPUT_NOT_ADMITTED':'This workspace admits no research input for experiments yet (its manifest lists none); prepare the workspace first.',
    'INPUT_SELECTION_REQUIRED':'Several inputs are admitted; choose one exact version.',
    'alpha_research.factor_decision_selection_required':'An Alpha draft needs a saved Factor decision or a sealed Foundation as its source.',
    'alpha_research.authoring_model_recipe_not_admissible':'The model declaration is not admitted by the installed capability: choose an installed capability and family, and give every parameter that family needs (ridge needs its regularization; lasso and elastic net their multiplier) within the installed domain.',
    'alpha_research.authoring_target_recipe_not_installed':'Choose one of the installed Alpha targets.',
    'alpha_research.development_work_budget_exceeded':'The numerical call budget is below what this declaration needs; raise it within the admitted bound or narrow the declaration.',
    'research_experiment.one_document_required':'Send the declaration once, as YAML or as a document, not both.',
    'research_foundation.handoff_source_mismatch':'The named Foundation belongs to another Factor decision or input version than the ones requested.',
    'factor_research.completed_experiment_required':'The source study has not completed and published; no decision or draft can start from it yet.'};
  const codeOf=(message)=>String(message || '').split(':')[0].trim();
  /* The most specific known code in the message: an inner validation code before the envelope's. */
  const innerCode=(message)=>{ const text=String(message || ''); return Object.keys(CODES).find(k=>text.includes(k) && k!==codeOf(text)) || ''; };
  const explain=(message)=>{ const inner=innerCode(message), code=codeOf(message); return inner ? t(CODES[inner]) : CODES[code] ? t(CODES[code]) : ''; };
  /* Which declared field a PLAN refusal names: the owner's `fields` paths first, then the code. */
  const FIELD_CODES={'factor_research.authoring_factor_id_not_in_inventory':'factor.factor_ids','factor_research.authoring_factor_ids_invalid':'factor.factor_ids','research_authoring.session_range_invalid':'experiment.sessions','research_authoring.document_unparsable':'yaml'};
  function fieldOf(error) {
    const named=(error?.body?.fields || []).map(p=>Array.isArray(p) ? p.join('.') : String(p)).filter(Boolean);
    if(named.length) return named[0];
    const message=String(error?.message || '');
    const code=Object.keys(FIELD_CODES).find(k=>message.includes(k));
    return code ? FIELD_CODES[code] : '';
  }
  /* A control the owner's schema binds to the refusal code (`refusal` on the control). */
  const refusedBy=(f,message)=>Boolean(f.refusal) && String(message || '').includes(f.refusal);
  /* The named field, as a full path or as the owner's relative path (`start` for `experiment.sessions.start`). */
  const namesField=(f,named)=>{ if(!named) return false; const p=f.path.join('.'); return p===named || p.startsWith(named+'.') || p.endsWith('.'+named); };
  /* The owner's own detail for a schema refusal (the text after the code), which names the field. */
  const ownerDetail=(message)=>{ const m=String(message || ''); const i=m.indexOf(': '); return codeOf(m)==='research_experiment.document_invalid' && i>0 ? m.slice(i+2).replace(/^:\s*/,'').trim() : ''; };
  const eligible = () => Data.inputs().filter(v => v.available && ['REGISTERED','SUCCEEDED'].includes(v.lifecycle));
  const inputKey = (v) => JSON.stringify([v.id, v.binding_hash]);
  const preparation=()=>typeof Data.preparation==='function' ? Data.preparation() : null;
  const saved=()=>typeof Data.experiments==='function' ? (Data.experiments() || []) : [];
  const cutoffOf=(binding)=>typeof LiveViews!=='undefined' ? LiveViews.cutoffText(LiveViews.inputState(binding)) : '';
  const changed = () => {
    S.revision++; S.plan=null; S.confirming=null; S.dirty=true; S.error=''; S.errorField=''; S.errorBody=null; app.plan=null;
    if(S.adopting) S.adopting=null; // the reviewed draft moved; a confirmation must be renewed
    // Work owned by the previous revision (a load, a format, a PLAN) is invalidated: its answer
    // will be ignored by its ticket, so its busy state is released now, never left behind.
    if(['load','format','plan'].includes(S.busy)) S.busy='';
    Lab.markDraftChanged();
  };
  /* ---- controls and YAML: two views of one declaration ---- */
  const optionValue=v=>v && typeof v==='object' && Object.hasOwn(v,'value') ? v.value : v;
  /* A control is shown while every condition it carries holds in the document -- the capability
   * handle it belongs to, the model family or training policy that reads it (`when`, the owner's
   * list of {path, values}; none means always). Visibility never stands in for the owner's PLAN
   * check: a value the page hid is still the owner's to refuse. */
  const visibleField=(f,doc=S.document)=>(f.when || []).every(c=>c.values.includes(c.path.reduce((v,k)=>v?.[k],doc)));
  const fieldValue=(f,doc)=>f.path.reduce((v,k)=>v?.[k],doc);
  /* The declared values these controls cannot show: an option field whose value is outside the
   * owner's option list. Such a draft stays in YAML with its exact values. */
  const unrepresented=(fields,doc)=>fields.filter(f=>{
    if(!visibleField(f,doc) || !f.options) return false;
    const value=fieldValue(f,doc); if(value==null) return false;
    return (Array.isArray(value) ? value : [value]).some(v=>!f.options.map(optionValue).includes(v));
  }).map(f=>f.label || f.path.join('.'));
  /* Declared entries no control covers: kept exactly as declared, read and edited in YAML. An
   * entry a control addresses is covered; an object some control reaches into is walked. */
  function uncovered(fields, doc) {
    const covered=fields.map(f=>f.path.join('.')), out=[];
    const walk=(value,prefix)=>{
      for(const [key,child] of Object.entries(value || {})) {
        const p=prefix ? prefix+'.'+key : key;
        if(p==='experiment.kind' || covered.includes(p)) continue;
        if(child && typeof child==='object' && !Array.isArray(child) && covered.some(c=>c.startsWith(p+'.'))) walk(child,p);
        else out.push(p);
      }
    };
    walk(doc,'');
    return out;
  }
  const defaultMode=()=>S.fields.length && S.document && !unrepresented(S.fields,S.document).length ? 'controls' : 'yaml';
  /* Controls are the first view when the owner's schema represents the draft; the reader's
   * explicit choice for this draft (`modeChoice`) is kept over every later read of it. */
  const settleMode=()=>{ app.mode=S.modeChoice || defaultMode(); };
  function adopt(body, origin=null) {
    S.document=clone(body.document || body.template); S.answered=body.document || body.template; S.fields=body.controls || body.authoring_options?.controls || [];
    S.missing=body.missing_fields || [];
    S.authoringOptions=body.authoring_options || null;
    S.featureInput=body.feature_input || null;
    S.prereq=body.prerequisites || null; // U60: what the kind's flow needs on this input
    S.request={...(body.plan_request || {} )}; delete S.request.operation;
    S.input=body.research_input_id || body.input_id || S.request.research_input_id || S.input;
    S.binding=body.input_binding_hash || S.request.input_binding_hash || S.binding;
    S.kind=S.document.experiment.kind; S.origin=origin; S.ready=true; S.dirty=false; S.modeChoice=null; S.guide=null;
    app.yaml=body.yaml; app.declaration={kind:S.kind,input:S.input}; settleMode();
  }
  async function fieldSpecs(body) {
    if(body.controls) return body.controls;
    if(body.authoring_options?.controls) return body.authoring_options.controls;
    const kind=(body.document || body.template)?.experiment?.kind;
    if(!['factor.screening-development','risk.covariance-development'].includes(kind)) return [];
    const handle=(body.document || body.template)?.experiment?.data_snapshot_handle || '';
    const controls=await Data.read('/api/experiments/controls?'+new URLSearchParams({
      research_input_id:body.research_input_id || body.input_id,
      input_binding_hash:body.input_binding_hash,experiment_kind:kind,
      ...(handle.startsWith('research-features@')?{feature_preparation_hash:handle.slice('research-features@'.length)}:{})}));
    return controls.controls || [];
  }
  /* A kind's declaration and controls for one exact version, as the owner answered them: kept
   * for the session so a second switch is instant; a Refresh of the page forgets them. */
  const controlsRead=new Map();
  const forget=()=>controlsRead.clear();
  async function load(kind, key, origin=null) {
    if (S.busy === 'run') return;
    changed(); const ticket=S.revision; S.busy='load'; S.ready=false; S.origin=null; S.modeChoice=null;
    S.kind=kind; S.fields=[]; app.yaml=''; S.document=null; S.request={}; S.missing=[]; S.authoringOptions=null; S.prereq=null;
    [S.input,S.binding]=key ? JSON.parse(key) : ['', ''];
    // No standalone declaration exists for these kinds: the owner would refuse the read. The
    // saved sources it requires are chosen here instead; nothing is read until one is named.
    if(GUIDED.has(kind) && !origin) { S.busy=''; S.dirty=false; S.guide=guideFor(kind); restoreLocator(); paint(); return guideRead(); } // a guided entry reads nothing until a source is named; its guide lists the sources the flow needs (U60)
    S.guide=null;
    // No exact version chosen: the choice is the page, not a refusal; there is no draft to protect.
    if(!origin && (!S.input || !S.binding)) { S.busy=''; S.dirty=false; restoreLocator(); paint(); return; }
    restoreLocator(); // the route names the kind and the exact version the draft is read for
    const readKey=origin ? null : [kind,S.input,S.binding].join('|');
    const kept=readKey && controlsRead.get(readKey);
    if(kept) { adopt(kept,null); S.fields=await fieldSpecs(kept); if(ticket!==S.revision) return; settleMode(); S.busy=''; paintKeepingHeight(); return; }
    paintKeepingHeight();
    try {
      const selection={research_input_id:S.input,input_binding_hash:S.binding};
      const body=origin ? await Data.post('/api/experiments/draft',{task_id:origin,...selection}) :
        await Data.read('/api/experiments/controls?' + new URLSearchParams({...selection,experiment_kind:kind}));
      if (ticket !== S.revision) return;
      if (!(body.document || body.template) || typeof body.yaml !== 'string') throw Error(body.status || 'Declaration unavailable');
      adopt(body,origin);
      const fields=await fieldSpecs(body);
      if(ticket===S.revision) { S.fields=fields; settleMode(); if(readKey) controlsRead.set(readKey,body); }
    } catch(e) { if(ticket===S.revision) S.error=e.message; }
    finally { if(ticket===S.revision) { S.busy=''; paintKeepingHeight(); } }
  }
  function ready() {
    if(S.initialized || app.page!=='lab' || Data.workspaceStatus!=='ready') return;
    S.initialized=true; app.yaml='';
    const work=hashParams().get('work');
    if(work && !S.work) void watch(work); // the exact Task the route names, read beside the draft
    const handoff=hashParams().get('draft_source');
    if(handoff) {try{return receiveHandoff(JSON.parse(handoff)).catch(e=>{S.error=e.message;paint();});}catch(e){S.error=e.message;paint();return;}}
    const origin=hashParams().get('origin');
    if(origin) return continueFrom(origin);
    const input=hashParams().get('research_input'),binding=hashParams().get('input_binding');
    if(input||binding)return load(hashParams().get('experiment_kind') || S.kind,JSON.stringify([input || '',binding || '']));
    const plan=hashParams().get('plan');
    if(plan) void inspectShared(plan); // beside the draft, never instead of it
    const kind=hashParams().get('experiment_kind');
    if(kind && kinds.some(v=>v[0]===kind)) S.kind=kind;
    const inputs=eligible();
    // A guided kind initializes from the route exactly as from its start button: the saved
    // sources it requires, whatever the number of eligible versions (none, one or several).
    if(GUIDED.has(S.kind)) return load(S.kind,inputs.length===1 ? inputKey(inputs[0]) : '');
    if(inputs.length===1) return load(S.kind,inputKey(inputs[0]));
    paint();
  }
  /* An edited draft is kept nowhere but this page, so replacing it is asked in the product's own
   * dialog (a host may suppress the browser's confirm, which then reads as a dead control). The
   * answer is awaited; closing the dialog any other way keeps the draft. */
  let replaceAnswer=null;
  function askReplace() {
    return new Promise((resolve)=>{
      replaceAnswer=resolve;
      alertDialog(t('Replace the edited draft?'), t('This declaration was edited on this page and is kept nowhere else. Replacing it drops those edits; saved results, admitted PLANs and Tasks are unchanged.'),
        html`${btn(t('Keep editing'),'research-replace','keep','button')}${btn(t('Copy declaration'),'copy-yaml','','button')}${btn(t('Replace draft'),'research-replace','replace','button primary')}`); // round 82: the alert
      const dialog=typeof $==='function' ? $('#dialog') : null; if(dialog) dialog.addEventListener('close',()=>answerReplace('keep'),{once:true});
    });
  }
  function answerReplace(answer) {
    const resolve=replaceAnswer; replaceAnswer=null;
    if(!resolve) return;
    closeDialog();
    resolve(answer==='replace');
  }
  async function choose(kind,key,origin=null) {
    if(unsaved() && !(await askReplace())) return paint();
    S.handoff=null; replaceHash({origin:origin || '',draft_source:''}); return load(kind,key,origin);
  }
  const routeContext=()=>({origin:S.origin || '',draft_source:S.handoff?JSON.stringify(S.handoff):'',research_input:S.input || '',input_binding:S.binding || '',experiment_kind:S.kind,plan:S.shared?.hash || '',work:S.work?.task || ''});
  const restoreLocator=()=>replaceHash(routeContext());
  async function followRoute() {
    if(app.page!=='lab'||!S.initialized)return;
    const q=hashParams(),source=q.get('draft_source'),origin=q.get('origin');
    const plan=q.get('plan') || '';
    if(plan!==(S.shared?.hash || '')) { if(plan) void inspectShared(plan); else { S.shared=null; S.sharedTicket++; } }
    const work=q.get('work') || '';
    if(work!==(S.work?.task || '')) { if(work) void watch(work); else S.work=null; }
    try {
      if(source){
        const reference=JSON.parse(source);
        if(JSON.stringify(reference)!==JSON.stringify(S.handoff))await receiveHandoff(reference);
      }else if(origin && origin!==S.origin)await continueFrom(origin);
      else if(!source&&!origin){const input=q.get('research_input'),binding=q.get('input_binding'),kind=q.get('experiment_kind') || S.kind;if(input&&binding&&(input!==S.input||binding!==S.binding||kind!==S.kind))await choose(kind,JSON.stringify([input,binding]));else if(!input&&!binding&&GUIDED.has(kind)&&kind!==S.kind)await choose(kind,'');else restoreLocator();}
    }catch(e){S.error=e.message;restoreLocator();paint();}
  }
  async function continueFrom(task) {
    if(S.busy) return restoreLocator();
    const chosen=Data.navigationIntent();
    if(unsaved() && !(await askReplace())) { if(Data.navigationCurrent(chosen)) restoreLocator(); return; }
    if(!Data.navigationCurrent(chosen)) return;
    if(app.page!=='lab') objectEntry('page:lab');
    closeDialog(); changed(); const ticket=S.revision; S.busy='load'; S.ready=false; S.guide=null;
    S.handoff=null; app.page='lab'; replaceHash({page:'lab',origin:task,draft_source:''}); paint();
    try {
      const body=await Data.post('/api/experiments/draft',{task_id:task});
      if(ticket!==S.revision) return;
      adopt(body,task); S.initialized=true;
      const fields=await fieldSpecs(body);
      if(ticket===S.revision) { S.fields=fields; settleMode(); }
    } catch(e) { if(ticket===S.revision) S.error=e.message; }
    finally { if(ticket===S.revision) { S.busy=''; paint(); } }
  }
  async function receiveHandoff(source) {
    const routes={
      features:['/api/experiments/controls',['feature_preparation_hash','input_binding_hash']],
      foundation:['/api/experiments/foundations/draft',['foundation_admission_hash']],
      factor:['/api/experiments/handoff',['task_id','curation_receipt_hash','research_input_id','input_binding_hash']],
      alpha_candidate:['/api/experiments/portfolio-draft',['task_id','candidate_id']],
    };
    if(!source || !Object.hasOwn(routes,source.kind)) throw Error('Unknown read-only handoff kind');
    const [path,keys]=routes[source.kind];
    if(Object.keys(source).some(k=>k!=='kind'&&!keys.includes(k)) || keys.some(k=>typeof source[k]!=='string'||!source[k]))throw Error('Incomplete handoff reference');
    if(S.busy==='run')return restoreLocator();
    const chosen=Data.navigationIntent();
    if(unsaved() && !(await askReplace())) { if(Data.navigationCurrent(chosen)) restoreLocator(); return; }
    if(!Data.navigationCurrent(chosen))return;
    if(app.page!=='lab')objectEntry('page:lab');
    closeDialog();changed();const ticket=S.revision;S.busy='load';S.ready=false;S.origin=null;S.guide=null;
    S.handoff=clone(source);S.initialized=true;app.page='lab';
    replaceHash({page:'lab',origin:'',draft_source:JSON.stringify(source)});paint();
    try{
      const request=Object.fromEntries(keys.map(k=>[k,source[k]]));
      const body=source.kind==='features'
        ? await Data.read(path+'?'+new URLSearchParams({...request,experiment_kind:'factor.screening-development'}))
        : await Data.post(path,request);
      if(ticket!==S.revision)return;
      if(source.kind==='factor')body.plan_request={factor_task_id:source.task_id,curation_receipt_hash:source.curation_receipt_hash,research_input_id:source.research_input_id,input_binding_hash:body.input_binding_hash};
      adopt(body);S.handoff=clone(source);
      const fields=await fieldSpecs(body);if(ticket===S.revision){S.fields=fields;settleMode();}
    }catch(e){if(ticket===S.revision)S.error=e.message;}
    finally{if(ticket===S.revision){S.busy='';paint();}}
  }
  /* ---- the guided entry for Alpha and Portfolio: the saved sources the owner requires ---- */
  const guideFor=(kind)=>({kind,study:'',curation:null,decision:'',target:'',foundations:null,foundationsError:'',foundation:'',alpha:'',alphaBody:null,candidate:'',busy:'',error:''});
  /* The sealed Foundations, read once per entry (again only on an explicit retry); nothing else
   * is read until a source is named. A failed read stays a failed read, never an empty list. */
  async function guideRead(retry=false) {
    const g=S.guide,ticket=S.revision; if(!g || g.kind!=='alpha.model-development' || g.busy) return;
    if(g.foundations!==null && !retry) return;
    g.foundationsError=''; g.busy='foundations'; paint();
    try { const b=await Data.read('/api/experiments/foundations'); if(S.guide===g && ticket===S.revision) g.foundations=b.foundations || []; }
    catch(e) { if(S.guide===g && ticket===S.revision && e.name!=='AbortError') { g.foundations=null; g.foundationsError=e.message; } }
    finally { if(S.guide===g && ticket===S.revision) { g.busy=''; paint(); } }
  }
  /* A saved Factor study's decisions and the inputs the decision owner admits for a handoff. */
  async function guideStudy(task) {
    const g=S.guide,ticket=S.revision; if(!g) return;
    g.study=task; g.curation=null; g.decision=''; g.target=''; g.error='';
    if(!task) return paint();
    g.busy='curation'; paint();
    try {
      const c=await Data.read('/api/experiments/curation?'+new URLSearchParams({task_id:task}));
      if(S.guide!==g || ticket!==S.revision || g.study!==task) return;
      g.curation=c;
      // The study's own exact version is offered as the handoff input when it is still admitted.
      const row=saved().find(v=>v.task_id===task), version=row ? Data.inputs().find(v=>v.available && v.binding_hash===row.input_binding_hash) : null;
      if(version && (c.inputs || []).includes(version.id)) g.target=inputKey(version);
    } catch(e) { if(S.guide===g && ticket===S.revision && g.study===task && e.name!=='AbortError') g.error=e.message; }
    finally { if(S.guide===g && ticket===S.revision && g.study===task) { g.busy=''; paint(); } }
  }
  /* A saved Alpha study's evaluated candidates, read from its exact readback. */
  async function guideAlpha(task) {
    const g=S.guide,ticket=S.revision; if(!g) return;
    g.alpha=task; g.alphaBody=null; g.candidate=''; g.error='';
    if(!task) return paint();
    g.busy='alpha'; paint();
    try {
      const b=await Data.read('/api/experiments/readback?'+new URLSearchParams({task_id:task}));
      if(S.guide!==g || ticket!==S.revision || g.alpha!==task) return;
      if(b.status!=='EXPERIMENT_PUBLISHED') throw Error(b.status);
      g.alphaBody=b;
    } catch(e) { if(S.guide===g && ticket===S.revision && g.alpha===task && e.name!=='AbortError') g.error=e.message; }
    finally { if(S.guide===g && ticket===S.revision && g.alpha===task) { g.busy=''; paint(); } }
  }
  function guideChange(field, value) {
    const g=S.guide; if(!g || g.busy) return;
    if(field==='guideStudy') return guideStudy(value);
    if(field==='guideAlpha') return guideAlpha(value);
    if(field==='guideDecision') g.decision=value; else if(field==='guideInput') g.target=value; else if(field==='guideFoundation') g.foundation=value; else if(field==='guideCandidate') g.candidate=value;
    paint();
  }
  /* The existing source-bound handoff, with the sources chosen here. Opening a draft runs nothing. */
  function guideOpen(which) {
    const g=S.guide; if(!g || g.busy) return;
    if(which==='decision') {
      const [id,hash]=g.target ? JSON.parse(g.target) : ['',''];
      if(!g.study || !g.decision || !id || !hash) throw Error(t('Choose the saved Factor study, one of its saved decisions and the exact input version first.'));
      return receiveHandoff({kind:'factor',task_id:g.study,curation_receipt_hash:g.decision,research_input_id:id,input_binding_hash:hash});
    }
    if(which==='foundation') { if(!g.foundation) throw Error(t('Choose a sealed Foundation first.')); return receiveHandoff({kind:'foundation',foundation_admission_hash:g.foundation}); }
    if(!g.alpha || !g.candidate) throw Error(t('Choose the saved Alpha study and one of its evaluated candidates first.'));
    return receiveHandoff({kind:'alpha_candidate',task_id:g.alpha,candidate_id:g.candidate});
  }
  /* The previous refusal's marks go with the draft that moved; the next PLAN answers anew. */
  function clearFieldMarks() {
    if(typeof document==='undefined') return;
    for(const el of document.querySelectorAll('.editor-workbench [aria-invalid="true"]')) el.removeAttribute('aria-invalid');
    for(const el of document.querySelectorAll('.editor-workbench .field.is-invalid')) el.classList.remove('is-invalid');
    for(const el of document.querySelectorAll('.editor-workbench .field-error')) el.remove();
  }
  function edit(e) {
    if(S.busy==='run') return;
    const marked=Boolean(S.error);
    changed();
    if(marked) clearFieldMarks();
    if(e.target.id==='yamlEditor') return Lab.onYamlInput(e);
    const f=S.fields[Number(e.target.dataset.researchField)];
    if(!f || !S.document || !visibleField(f)) return;
    if(f.path.some(k=>['__proto__','prototype','constructor'].includes(k)))throw Error('Control path unavailable');
    let target=S.document;
    for(const key of f.path.slice(0,-1)) {
      if(!Object.hasOwn(target,key) || target[key]===null)target[key]={}; // U42: a null section (no policy) is opened by its first field
      if(typeof target[key]!=='object' || Array.isArray(target[key])) throw Error('Control path unavailable');
      target=target[key];
    }
    const key=f.path.at(-1);
    if(f.type==='universe') target[key]=e.target.value==='' ? f.whole : f.whole+f.sample_mark+e.target.value; // R4 (V339): empty is the whole universe
    else if(f.options && f.type!=='multiple' && e.target.value==='') { const was=f.path.slice(0,-1).reduce((v,k)=>v?.[k],S.answered); if(was && typeof was==='object' && was[key]===null) target[key]=null; else { delete target[key]; settleEmpty(f.path); } }
    else target[key]=f.options ? (f.type==='multiple' ? (Array.isArray(e.target.value) ? e.target.value : []).map(j=>optionValue(f.options[Number(j)])) : optionValue(f.options[Number(e.target.value)])) : f.type==='number' ? e.target.valueAsNumber : e.target.value; // a multiple choice reaches here as the list's chosen values (round 95)
    if(f.type==='multiple' && typeof document!=='undefined') { const count=document.getElementById('researchFieldCount'+e.target.dataset.researchField); if(count) count.textContent=chosenText(f,target[key]); }
    if(S.fields.some(field=>(field.when || []).some(c=>c.path.join('.')===f.path.join('.')))) {
      // A value other controls are conditioned on changed: the parameters those controls
      // wrote no longer belong to the declaration, so they leave the document with them.
      // Only the fields of controls now hidden leave, and only when no control still shown owns
      // the same path (two families may share a parameter's path under different conditions).
      const kept=new Set(S.fields.filter(field=>visibleField(field)).map(field=>field.path.join('.')));
      for(const field of S.fields)if(!visibleField(field) && !kept.has(field.path.join('.'))){
        const parent=field.path.slice(0,-1).reduce((v,k)=>v?.[k],S.document);
        if(parent && Object.hasOwn(parent,field.path.at(-1))) { delete parent[field.path.at(-1)]; settleEmpty(field.path); }
      }
      paint();
    }
  }
  /* A section the owner answered as null and emptied again by the controls is null again: a
   * cleared catalog policy is the tranche book the draft began with (U42), never an empty policy. */
  function settleEmpty(path) {
    for(let n=path.length-1;n>0;n--) {
      const at=path.slice(0,n), parent=at.slice(0,-1).reduce((v,k)=>v?.[k],S.document), key=at.at(-1), v=parent?.[key];
      if(!v || typeof v!=='object' || Array.isArray(v) || Object.keys(v).length || at.reduce((x,k)=>x==null ? x : x[k],S.answered)!==null) return;
      parent[key]=null;
    }
  }
  /* Controls and YAML are two views of one declaration: switching formats it through the owner
   * and is not an edit -- the draft's edited state and its PLAN are kept unless the formatted
   * document actually differs (an edited YAML being read into controls). Nothing is repainted
   * while the owner answers (the two buttons are only held), and the repaint keeps the editor's
   * height, so the page does not jump under the reader. */
  async function setMode(mode) {
    if(S.busy || !S.ready || mode===app.mode) return;
    if(mode==='controls' && !S.fields.length) return;
    const ticket=++S.revision; S.busy='format';
    const held=$$('.editor-workbench [data-action="mode"]'); held.forEach((b)=>{ b.disabled=true; });
    const before=JSON.stringify(S.document);
    try {
      const body=await Data.post('/api/experiments/declaration', app.mode==='yaml' ? {experiment_yaml:app.yaml} : {experiment_document:S.document});
      if(ticket!==S.revision) return;
      if(mode==='controls') {
        if(body.document.experiment?.kind!==S.kind) throw Error(t('The YAML declares another method than these controls; keep editing YAML, or start the other kind.'));
        const kept=unrepresented(S.fields,body.document);
        if(kept.length) throw Error(t('The declared value of {fields} cannot be shown by these controls. The draft stays in YAML with its exact values; PLAN says whether the owner admits them.',{fields:kept.join(', ')}));
      }
      if(JSON.stringify(body.document)!==before) { S.dirty=true; S.plan=null; S.confirming=null; app.plan=null; }
      S.document=body.document; app.yaml=body.yaml; app.mode=mode; S.modeChoice=mode; S.error=''; S.errorField=''; S.errorBody=null;
    } catch(e) { if(ticket===S.revision) S.error=e.message; }
    finally { if(ticket===S.revision) { S.busy=''; held.forEach((b)=>{ b.disabled=false; }); paintKeepingHeight(); } }
  }
  /* Repaint the page with the editor no shorter than it was: the page keeps its height while the
   * sections are replaced (so the window is never clamped), and the new editor content keeps at
   * least the old height until the next full render. */
  function paintKeepingHeight() {
    if(typeof $!=='function') return paint(); // no document (the probes' harness): the paint alone
    const main=$('#main'), content=$('.editor-workbench .declaration-content');
    const keep=content ? content.offsetHeight : 0;
    if(main && keep) main.style.minHeight=main.offsetHeight+'px';
    paint();
    const next=$('.editor-workbench .declaration-content');
    if(next && keep) next.style.minHeight=keep+'px';
    if(main) main.style.minHeight='';
  }
  /* One PLAN of the current draft. Returns whether it was requested at all: a busy or unready
   * editor sends nothing, and callers that promised a PLAN must not report one they did not get. */
  async function preview() {
    if(S.busy || !S.ready) return false;
    S.plan=null; S.confirming=null; app.plan=null; S.error=''; S.errorField=''; S.errorBody=null; S.busy='plan';
    const ticket=S.revision;
    const request={...S.request,research_input_id:S.input,input_binding_hash:S.binding,
      ...(app.mode==='yaml' ? {experiment_yaml:app.yaml} : {experiment_document:clone(S.document)})};
    paintKeepingHeight();
    try {
      const body=await Data.post('/api/experiments/plan',request);
      if(ticket!==S.revision) return true; // requested, but the draft moved on; its answer is not shown
      if(body.status!=='PLANNED' || !body.plan_hash) throw Error('PLAN was not admitted');
      S.plan=body; S.missing=[]; app.plan=body; openReader();
    } catch(e) { if(ticket===S.revision) { S.error=e.message; S.errorField=fieldOf(e); S.errorBody=e.body || null; } }
    finally { if(ticket===S.revision) S.busy=''; paint(); }
    return true;
  }
  /* What a PLAN refusal carries beyond its words (the owner's body): where a YAML text stopped parsing (U17),
   * the bound a policy's top_k broke (U45), and the ways back it names -- the Alpha study's Portfolio draft
   * (U45) and the explored study's promotion, which is its page's (R4). */
  function refusalWays(b) {
    if(!b) return '';
    const at=b.document_location, most=b.expected?.['portfolio.policy.top_k']?.maximum, next=b.next_requests || {}, draft=next.portfolio_draft, promote=next.promote;
    const words=[at ? t('The YAML stopped at line {line}, column {column}.',{line:at.line,column:at.column}) : '',most!=null ? t('At most {n} names: the listings the Alpha study scored.',{n:count(most)}) : ''].filter(Boolean).join(' ');
    const ways=html`${draft?.task_id && draft?.candidate_id ? btn(t('Open the Portfolio draft again'),'research-portfolio-redraft',JSON.stringify([draft.task_id,draft.candidate_id]),'button compact') : ''}${promote?.task_id ? btn(t('Open the explored study'),'task-result',promote.task_id,'button compact') : ''}`;
    return words || String(ways) ? html`${words ? html`<p class="caption">${words}</p>` : ''}${String(ways) ? html`<div class="flow">${ways}</div>` : ''}` : '';
  }
  function reviewRun() {
    if(!S.plan || S.busy) return;
    S.confirming={hash:S.plan.plan_hash,revision:S.revision};
    openDialog(t('PLAN · explicit confirmation'), t('Run this exact declaration?'),
      html`${previewFacts()}${codeRef(t('Read the exact declaration'), S.plan.document)}<p class="caption">${t('Confirmation may consume local compute. No data download, strategy activation or current-input change is requested.')}</p>`,
      html`${btn(t('Confirm and run'),'research-run','', 'button primary',true)}`);
  }
  /* ---- the admitted Task, read beside the draft in the shared work area ---- */
  /* What the confirmed PLAN declared, kept for the area's facts; a Task restored from the route
   * alone has only what Task Control and the readback say. */
  function declaredOf(plan) {
    const x=plan.execution_preview || {}, d=plan.document || {}, risk=riskScopeOf(x,null);
    return {kind:plan.program?.kind || S.kind,input:S.input,binding:S.binding,factors:d.factor?.factor_ids?.length ?? null,context:x.context_factor_count ?? null,calls:x.expected_numerical_calls ?? null,
      interval:x.statistical_start && x.statistical_end ? `${x.statistical_start} — ${x.statistical_end}` : '',origin:S.origin || null,handoff:S.handoff ? clone(S.handoff) : null,
      // the family-specific scope the PLAN declared: Alpha's configurations and folds, Risk's
      // complete formation window, dated scopes and coverage, Portfolio's sessions and holds;
      // absent for the others
      configurations:x.model_configuration_count ?? null,folds:x.fold_count ?? null,features:d.alpha?.ordered_feature_ids?.length ?? null,
      window:risk.window,scopes:risk.scopes,sizes:risk.sizes,listings:x.listing_count ?? null,
      sessions:x.score_session_count ?? null,holds:x.hold_session_count ?? null,candidate:d.portfolio?.candidate_id || null};
  }
  /* A Risk declaration's whole formation scope, never its first dated scope alone: the PLAN's
   * statistical interval, or the published series' formation sessions, or the span of every
   * dated scope; each scope keeps its own matrix axis size. The coverage (the owner's listing
   * union across scopes) is reported by the caller as coverage, not as a matrix. A legacy
   * single-scope surface is one scope. */
  function riskScopeOf(x, surface) {
    const scopes=surface?.scope_surfaces?.length ? surface.scope_surfaces : x.listing_scopes?.length ? x.listing_scopes : surface?.formation_sessions?.length ? [surface] : [];
    const firsts=scopes.map(v=>v.first_session || v.formation_sessions?.[0]).filter(Boolean).sort(), lasts=scopes.map(v=>v.last_session || v.formation_sessions?.at(-1)).filter(Boolean).sort();
    const sessions=surface?.formation_sessions || [];
    const window=x.statistical_start && x.statistical_end ? `${x.statistical_start} — ${x.statistical_end}` : sessions.length ? `${sessions[0]} — ${sessions.at(-1)}` : firsts.length ? `${firsts[0]} — ${lasts.at(-1)}` : '';
    return {window,scopes:scopes.length,sizes:scopes.map(v=>v.ordered_listing_ids?.length ?? v.listing_count ?? null)};
  }
  const EXPERIMENT_KIND='research_experiment', AUTOMATIC_READBACK_READS=3;
  function watch(task, declared=null) {
    S.reuse=null; S.work={task,view:null,body:null,mismatch:null,readback:null,readAt:0,stale:null,recovered:false,error:'',lateReads:0,reading:false,declared};
    replaceHash({work:task});
    return readWork();
  }
  /* The readback that counts as this Task's completion: the owner's, for exactly this Task. */
  const published=(w)=>Boolean(w.body) && w.body.status==='EXPERIMENT_PUBLISHED' && w.body.task_id===w.task;
  /* Task Control's recovery view (stages, lifecycle, liveness) -- adopted as experiment work only
   * when it answers for the requested Task and that Task is an experiment; any other answer is
   * disclosed, never composed into the experiment scene. Once the view says the Task succeeded,
   * the experiment readback is read again until it says the result is published: a bounded number
   * of times on the cadence, and once more on every explicit Read again, which also renews that
   * automatic budget. A failed readback is kept and disclosed, never a Task fact. A failed view
   * read keeps the last observation and marks it as not current. */
  async function readWork(quiet=false, explicit=false) {
    const w=S.work; if(!w || w.reading) return;
    w.reading=true;
    if(explicit) w.lateReads=0;
    try {
      const view=await Data.readShared('/api/tasks/recovery?'+new URLSearchParams({task_id:w.task}));
      if(S.work!==w) return;
      if(view.task_id!==w.task || view.task_kind!==EXPERIMENT_KIND) { w.mismatch={task_id:view.task_id,task_kind:view.task_kind,lifecycle:view.lifecycle}; w.view=null; w.stale=null; w.readAt=Date.now(); w.error=''; return; }
      const wasMoving=Boolean(w.view && stateMoving(w.view.lifecycle));
      w.mismatch=null; w.view=view; w.recovered=Boolean(w.stale); w.stale=null; w.readAt=Date.now(); w.error='';
      const area=typeof LiveWorkArea!=='undefined' ? (app.page==='lab' ? LiveWorkArea.areaFor(EXPERIMENT,w.task) : LiveWorkArea.entryFor(EXPERIMENT,w.task)) : null;
      if(view.lifecycle==='SUCCEEDED' && !published(w) && (explicit || w.lateReads<AUTOMATIC_READBACK_READS)) {
        w.lateReads+=1;
        try {
          const body=await Data.readShared('/api/experiments/readback?'+new URLSearchParams({task_id:w.task}),true);
          if(S.work!==w) return;
          if(body.task_id===w.task) { w.body=body; w.readback=null; }
          else w.readback={failures:(w.readback?.failures || 0)+1,error:t('The readback answered for Task {task}, not for this one; nothing was adopted.',{task:body.task_id || ''}),at:Date.now()};
        } catch(e) { if(S.work===w) w.readback={failures:(w.readback?.failures || 0)+1,error:e.message,at:Date.now()}; }
        if(S.work!==w) return;
        if(published(w)) { if(area && wasMoving) area.accent=true; void Data.refreshHistory(); }
      }
      if(area) LiveWorkArea.noteTransition(view,area);
    } catch(e) {
      if(S.work===w) { if(w.view) w.stale={since:w.readAt || Date.now(),error:e.message,failures:(w.stale?.failures || 0)+1,at:Date.now()}; else w.error=e.message; }
    } finally { if(S.work===w) { w.reading=false; if(quiet && typeof patchMain==='function') patchMain(); else paint(); } }
  }
  /* On the activity cadence: read again while the Task moves, while its projection disagrees
   * with the view, or while a succeeded Task's readback has not said "published" yet (bounded).
   * A Task the owner answered for as something else is not read again by itself. */
  function observe() {
    const w=S.work; if(!w || app.page!=='lab' || w.reading || w.mismatch) return;
    const v=w.view, projection=(typeof Data.tasks==='function' ? Data.tasks() : []).find(x=>x.task_id===w.task);
    const differs=Boolean(v && projection && (projection.lifecycle!==v.lifecycle || projection.verified_stage_count!==v.verified_stage_count));
    const late=Boolean(v && v.lifecycle==='SUCCEEDED' && !published(w) && w.lateReads<AUTOMATIC_READBACK_READS);
    if(v && !stateMoving(v.lifecycle) && !v.operation_running && !(projection && stateMoving(projection.lifecycle)) && !differs && !late) return;
    void readWork(true);
  }
  function dismissWork() { S.work=null; S.reuse=null; replaceHash({work:''}); paint(); }
  async function run() {
    if(S.busy || !S.plan || !S.confirming || S.confirming.revision!==S.revision || S.confirming.hash!==S.plan.plan_hash) return;
    const hash=S.confirming.hash, plan=S.plan; S.confirming=null; S.busy='run'; S.error=''; closeDialog(); paint();
    try {
      const body=await Data.post('/api/experiments/run',{experiment_plan_hash:hash});
      S.submission=body; S.plan=null; app.plan=null;
      const task=body.task_id || body.publication_task_id;
      if(task) app.labTask=task;
      // Exact reuse is an answer, not work: the completed Task and its reader are named at once.
      if(body.status==='REUSED_EXACT' && body.publication_task_id) { S.work=null; S.reuse={task:body.publication_task_id,kind:plan.program?.kind || S.kind,realization:body.realization || null}; replaceHash({work:''}); }
      else if(body.task_id) await watch(body.task_id,declaredOf(plan));
      else S.error=(body.failure_code || body.status)+' · '+t('Nothing was admitted; create the PLAN again, then confirm.');
      await Data.refreshHistory();
    } catch(e) {
      S.error=e.message+' · '+t(e.body?.next_action==='EXPERIMENT_PLAN' ? 'Nothing was admitted; create the PLAN again, then confirm.' : 'Submission may be uncertain. Inspect Tasks before confirming again.');
    } finally { S.busy=''; paint(); }
  }
  const STAGE_NAMES=[['execute_sealed_experiment','Execute the sealed experiment'],['verify_experiment_evidence','Verify the experiment evidence']];
  /* The experiment families this owner runs, each in its own words for the shared area: what the
   * execution stage does, what the verification does, which reader the completion opens and what
   * continues there. Task Control's two stages are the same for all of them; nothing here reports
   * progress the owner does not, and an unknown family gets the generic words, never Factor's. */
  const FAMILIES={
    'factor.screening-development':{title:'Factor work',reader:'Open the saved Factor study',
      execute:'The research owner runs the admitted program on the sealed input as one unit: features, targets, walk-forward folds and the evidence report. It reports at stage boundaries only.',
      verify:'Task Control verifies the recorded evidence against the sealed program before the result is published; nothing is recomputed.',
      done:'The Factor evidence, a curation decision and the Foundation or Alpha handoff continue from that reader.'},
    'alpha.model-development':{title:'Alpha work',reader:'Open the saved Alpha study',
      execute:'The Alpha owner fits the declared model per causal fold on the input\'s bound features, scores the validation sessions and records each candidate\'s development metrics, as one unit. It reports at stage boundaries only; no intermediate metric exists.',
      verify:'Task Control verifies the recorded candidate evidence against the sealed program before the result is published; nothing is refitted.',
      done:'The candidates, their folds and the Portfolio draft from one explicit candidate continue from that reader. A candidate\'s score is development evidence, not a realized return.'},
    'risk.covariance-development':{title:'Risk work',reader:'Open the saved Risk study',
      execute:'The Risk owner estimates the declared covariance at each formation of the window and evaluates it against the next realized session, as one unit. It reports at stage boundaries only.',
      verify:'Task Control verifies the recorded diagnostics against the sealed program before the result is published; nothing is re-estimated.',
      done:'The dated scopes, diagnostics and the report-only association continue from that reader; the matrix sizes no Portfolio.'},
    'portfolio.policy-development':{title:'Portfolio work',reader:'Open the saved book',
      execute:'The Portfolio owner replays the declared policy over the candidate\'s saved scores session by session -- selection, sleeves, costs -- as one unit; no model is refitted. It reports at stage boundaries only.',
      verify:'Task Control verifies the recorded replay against the sealed program before the book is published; nothing is replayed again.',
      done:'The dated holdings, report facts and a compatible comparison continue from that reader. A stored replay is not live holdings.'}};
  const GENERIC_FAMILY={title:'Experiment work',reader:'Open the saved result',
    execute:'The research owner runs the admitted program on the sealed input as one unit and reports at stage boundaries only.',
    verify:'Task Control verifies the recorded evidence against the sealed program before the result is published.',
    done:'The result continues from its own reader.'};
  /* The family of the Task on the page: the program the confirmed PLAN declared, the published
   * readback's program, or the experiment listing's kind for a Task restored by route; unknown
   * otherwise (the goal summary is text, not a typed kind). */
  const familyOf=(w)=>w ? (w.declared?.kind || w.body?.program?.kind || saved().find(v=>v.task_id===w.task)?.kind || '') : '';
  /* What a published readback records of the declaration, in the same shape as the confirmed
   * PLAN's facts, for a Task restored by its route: the recorded scope and the work the owner
   * actually counted (fit/prediction/metric calls, numerical calls), never a fresh estimate. */
  function declaredFromReadback(b) {
    if(!b || b.status!=='EXPERIMENT_PUBLISHED') return null;
    const x=b.execution_preview || {}, d=b.document || {}, risk=riskScopeOf(x,b.risk_surface), r=b.result || {};
    // Only Alpha (fit/prediction/metric) and Risk count their execution's calls in the readback;
    // the readback's own evidence count is its verification, not the execution, so the other
    // families keep the PLAN's estimate, labelled as such.
    const counted=r.fit_call_count!=null ? r.fit_call_count+(r.predict_call_count || 0)+(r.metric_call_count || 0) : b.execution_numerical_call_count ?? null;
    return {kind:b.program?.kind || '',input:b.research_input_id || '',binding:b.input_binding_hash || '',recorded:true,callsCounted:counted!=null,
      factors:d.factor?.factor_ids?.length ?? null,context:r.evidence_report?.hypothesis_count ?? null,calls:counted ?? x.expected_numerical_calls ?? null,
      interval:x.statistical_start && x.statistical_end ? `${x.statistical_start} — ${x.statistical_end}` : (b.evidence?.formation_sessions?.length ? `${b.evidence.formation_sessions[0]} — ${b.evidence.formation_sessions.at(-1)}` : ''),
      configurations:x.model_configuration_count ?? null,folds:x.fold_count ?? null,features:d.alpha?.ordered_feature_ids?.length ?? null,
      window:risk.window,scopes:risk.scopes,sizes:risk.sizes,listings:x.listing_count ?? b.risk_surface?.ordered_listing_ids?.length ?? null,
      sessions:x.score_session_count ?? null,holds:x.hold_session_count ?? null,candidate:d.portfolio?.candidate_id || null};
  }
  const declaredFor=(w)=>w ? (w.declared || declaredFromReadback(w.body)) : null;
  // each dated scope's own axis size, in scope order; an unrecorded size stays unknown
  // the scopes' axes, only when every scope's size is known: a sentence leaves an unknown clause out (ST7)
  const scopeSizes=(d)=>(d.sizes || []).length && d.sizes.every(v=>v!=null) ? d.sizes.join(', ') : '';
  const familyWords=(w)=>FAMILIES[familyOf(w)] || GENERIC_FAMILY;
  const EXPERIMENT={key:'experiment',page:'lab',railLabel:'Experiment stages; select to inspect, not execute',factsLabel:'Experiment facts',steps:STAGE_NAMES,
    get title() { return familyWords(S.work).title; },
    get lines() { const w=familyWords(S.work); return {execute_sealed_experiment:w.execute,verify_experiment_evidence:w.verify}; },
    fallbackLine:'Continuing the reported experiment stage.',
    oneUnit:{execute_sealed_experiment:'The execution is one unit: the owner reports no finer count, and its heartbeat is the only sign of life while it runs.',verify_experiment_evidence:'The verification is one unit; its evidence reference is the next fact.'},
    view:()=>S.work?.view || null,work:experimentWork,aside:experimentAside,absorb:()=>{},completedNow:(b)=>b?.status==='EXPERIMENT_PUBLISHED'};
  /* The shown stage's count: the scope the PLAN declared for this family (never a computation
   * percentage), or one unit when the declaration is not known to this page. */
  function declaredCount(d, kind) {
    if(!d) return null;
    if(kind==='factor.screening-development' && d.factors!=null) return html`<strong>${d.factors}</strong><span class="tp-denom"> ${pluralText(d.factors,'factor declared · {n} in the multiple-testing denominator','factors declared · {n} in the multiple-testing denominator',{n:d.context ?? ''})}</span>`;
    if(kind==='alpha.model-development' && (d.configurations!=null || d.folds!=null)) return html`<strong>${d.configurations ?? ''}</strong><span class="tp-denom"> ${pluralText(d.configurations,'model configuration · {f} · {n}','model configurations · {f} · {n}',{f:countText(d.folds ?? '','{n} causal fold','{n} causal folds'),n:countText(d.features ?? '','{n} feature','{n} features')})}</span>`;
    if(kind==='risk.covariance-development' && (d.window || d.listings!=null)) return html`<strong>${d.listings ?? ''}</strong><span class="tp-denom"> ${pluralText(d.listings,scopeSizes(d) ? 'listing in coverage · formations {window} · {k}, axes of {sizes} assets · {n}' : 'listing in coverage · formations {window} · {k} · {n}',scopeSizes(d) ? 'listings in coverage · formations {window} · {k}, axes of {sizes} assets · {n}' : 'listings in coverage · formations {window} · {k} · {n}',{window:d.window || '',k:countText(d.scopes || '','{n} dated scope','{n} dated scopes'),sizes:scopeSizes(d),n:countText(d.calls ?? '',d.callsCounted ? '{n} numerical call recorded' : '{n} numerical call estimated',d.callsCounted ? '{n} numerical calls recorded' : '{n} numerical calls estimated')})}</span>`;
    if(kind==='portfolio.policy-development' && d.sessions!=null) return html`<strong>${d.sessions}</strong><span class="tp-denom"> ${pluralText(d.sessions,'scored session to replay · {h} · {n}','scored sessions to replay · {h} · {n}',{h:countText(d.holds ?? '','{n} hold session','{n} hold sessions'),n:countText(d.listings ?? '','{n} listing','{n} listings')})}</span>`;
    return null;
  }
  function experimentWork(b, v, state, shown) {
    const w=S.work, count=declaredCount(declaredFor(w),familyOf(w)) || html`<strong>1</strong><span class="tp-denom"> ${t('unit')}</span>`;
    return {count,bar:null,detail:t(shown==='verify_experiment_evidence' ? 'One verification unit; no finer count is reported. Task Control records its evidence reference when it is done.' : 'Declared scope, not a computation percentage: the owner reports no finer count for this stage. Its liveness is the heartbeat below.')};
  }
  /* What the PLAN declared, in this family's terms; a Task restored by route has only what Task
   * Control recorded (its goal summary) until its readback names the program. */
  function declaredLine(d, kind, v) {
    if(d && kind==='factor.screening-development' && d.factors!=null) return t('{f}; statistical interval {interval}',{f:countText(d.factors,'{n} factor','{n} factors'),interval:d.interval || ''});
    if(d && kind==='alpha.model-development' && d.folds!=null) return t('{c} on {n}; {f}; statistical interval {interval}',{c:countText(d.configurations ?? '','{n} model configuration','{n} model configurations'),n:countText(d.features ?? '','{n} feature','{n} features'),f:countText(d.folds,'{n} causal fold','{n} causal folds'),interval:d.interval || ''});
    if(d && kind==='risk.covariance-development' && d.window) return t(scopeSizes(d) ? 'formations {window} in {k} (axes of {sizes} assets); coverage {n}, the union across scopes, not one matrix' : 'formations {window} in {k}; coverage {n}, the union across scopes, not one matrix',{window:d.window,k:countText(d.scopes || '','{n} dated scope','{n} dated scopes'),sizes:scopeSizes(d),n:countText(d.listings ?? '','{n} listing','{n} listings')});
    if(d && kind==='portfolio.policy-development' && d.sessions!=null) return t('candidate {candidate}; {n}, {h}; replay interval {interval}',{candidate:d.candidate || '',n:countText(d.sessions,'{n} scored session','{n} scored sessions'),h:countText(d.holds ?? '','{n} hold','{n} holds'),interval:d.interval || ''});
    if(d && d.interval) return t('statistical interval {interval}',{interval:d.interval});
    return v.status?.goal_summary || '';
  }
  function experimentAside(b, v, state, shown, isCurrent) {
    if(!isCurrent) return null; // the stage record the area keeps
    const w=S.work, d=declaredFor(w), kind=familyOf(w);
    const rows=[[t('Method'),kind ? kindLabel(kind) : t('experiment (kind not recorded on this page)')],[t('Exact input'),d?.input ? html`${d.input} · ${cutoffOf(d.binding)}` : t('as recorded on the Task')],
      [t('Declared'),html`${declaredLine(d,kind,v)}${d?.recorded ? html`${infoMark(t('from the published readback'))}` : ''}`],
      [t('Numerical calls'),d?.calls!=null ? t(d.callsCounted ? '{n} recorded by the completing execution' : '{n} estimated by the PLAN for a fresh execution',{n:d.calls}) : t('not previewed on this page')],
      [t('Liveness'),typeof LiveTasks!=='undefined' && LiveTasks.liveness ? (LiveTasks.liveness(v) || t('the Task is not executing; its lifecycle says where it stands')) : v.liveness?.status || '']];
    const rail=html`<section class="ui-log-rail prep-log-rail"><div><span>${t('Task')}</span><strong class="mono">${short(v.task_id)}</strong></div><div><span>${t('Task Control')}</span><strong>${v.verified_stage_count} / ${v.total_stage_count} ${t('verified')}</strong></div></section>`;
    return LiveWorkArea.factsShell(EXPERIMENT,{label:t('Experiment facts'),title:t('This experiment'),caption:t('The declaration this Task runs and the owner\'s liveness; this owner reports no per-unit activity'),facts:kv(rows),rail});
  }
  /* The Task the route or the RUN named, as the owner answered for it: a mismatch (another kind
   * of Task, or another id) is explained with the way to its own reader; a failed readback is
   * disclosed with its count and the explicit retry. */
  function workMismatch(w) {
    const m=w.mismatch;
    if(m.task_id!==w.task) return t('The Task owner answered for Task {other} instead of the requested Task {task}; nothing was adopted.',{other:short(m.task_id),task:short(w.task)});
    return t('Task {task} is a {kind} Task ({state}), not an experiment; this page reads experiment Tasks only. It is read in its own place on Tasks, unchanged.',{task:short(w.task),kind:codeWords(m.task_kind),state:codeWords(m.lifecycle)});
  }
  function readbackNotice(w) {
    const r=w.readback; if(!r) return '';
    const exhausted=w.lateReads>=AUTOMATIC_READBACK_READS;
    return banner(t('The result could not be read ({n})',{n:countText(r.failures,'{n} attempt','{n} attempts')}),html`<span class="mono">${r.error}</span>${explain(r.error) ? html`<br>${explain(r.error)}` : ''}<br>${t(exhausted ? 'Automatic reads stopped after {n} attempts; Read again tries once more and renews them. The Task itself is unchanged.' : 'It is read again on the cadence, a bounded number of times; Read again tries at once.',{n:AUTOMATIC_READBACK_READS})}`,'warning',btn(t('Read again'),'research-work-refresh','','button compact'));
  }
  function workSection() {
    const w=S.work; if(!w) return '';
    const v=w.view, b=w.body, published=Boolean(b) && b.status==='EXPERIMENT_PUBLISHED' && b.task_id===w.task;
    const standing=v && typeof LiveTasks!=='undefined' && LiveTasks.standing ? LiveTasks.standing(v) : null;
    const head=html`<header class="prep-scene-head lab-work-head"><div><p class="caption">${w.mismatch ? t('Task named by the route') : familyOf(w) ? kindLabel(familyOf(w))+' · '+t('Task') : t('Experiment Task')}</p><h2>${t('Task')} <span class="mono" data-tip="${w.task}">${short(w.task)}</span>${infoMark(t('The same Task as on Tasks and in the activity feed. The declaration above stays editable; nothing here replaces it.'))}</h2>${standing ? html`<p class="prep-liveness lab-standing" data-tone="${standing[0]}">${icon(standing[1])}<span>${standing[2]}</span></p>` : ''}</div><div class="flow">${btn(t('Inspect task'),'task',w.task,'button compact')}${published ? btn(t(familyWords(w).reader),'task-result',w.task,'button primary compact') : ''}${btn(t('Read again'),'research-work-refresh','','button compact')}${btn(t('Dismiss'),'research-work-dismiss','','text-btn')}</div></header>`;
    if(w.mismatch) return html`<section class="prep-scene lab-work" data-scene="mismatch">${head}${noteLine(t('Not this page\'s Task'),workMismatch(w),'warning',btn(t('Inspect task'),'task',w.task,'button compact'))}</section>`;
    if(!v) return html`<section class="prep-scene lab-work" data-scene="pending">${head}${w.error ? notRead(t('Task not read'),w.error,explain(w.error),btn(t('Read again'),'research-work-refresh','','button compact')) : noteLine(t('Reading the Task'),t('Its stages and liveness come from Task Control; nothing is started by reading.'))}</section>`;
    const words=familyWords(w);
    // N6 (laws 72, 98): a finished Task is one line -- done, what it is, its reader; its stages are the Task's own record
    if(v.lifecycle==='SUCCEEDED' && published) return html`<section class="prep-scene lab-work lab-work-done" data-scene="experiment" data-lifecycle="SUCCEEDED">${noteLine(html`<strong>${t('Completed')}</strong> · ${familyOf(w) ? kindLabel(familyOf(w)) : t('Experiment Task')}`,t(words.done),'ok',html`${btn(t(words.reader),'task-result',w.task,'button primary compact')}${btn(t('Dismiss'),'research-work-dismiss','','text-btn')}`,'checkcircle')}</section>`;
    const complete=v.lifecycle==='SUCCEEDED' ? (w.readback ? readbackNotice(w) : noteLine(t('Completed; its result is being read'),w.lateReads>=AUTOMATIC_READBACK_READS ? t('Task Control verified the last stage, but the readback did not say "published" after {n} reads; Read again tries once more, or open the Task for its owner\'s answer.',{n:AUTOMATIC_READBACK_READS}) : t('Task Control verified the last stage; the readback is read once more.'),'neutral',w.lateReads>=AUTOMATIC_READBACK_READS ? btn(t('Read again'),'research-work-refresh','','button compact') : ''))
      : w.stale ? noteLine(t('Not re-read since {time}',{time:when(new Date(w.stale.since).toISOString())}),html`${w.stale.error} · ${t('The last observation is kept below; nothing about the Task is inferred from a failed read.')}`,'warning') : '';
    const area=typeof LiveWorkArea!=='undefined' ? LiveWorkArea.workArea(EXPERIMENT,b,v,{readAt:w.readAt,stale:w.stale,recovered:w.recovered}) : '';
    return html`<section class="prep-scene lab-work" data-scene="experiment" data-lifecycle="${v.lifecycle}">${head}${complete}${area}</section>`;
  }
  /* ---- the editor ---- */
  const GROUPS=[['experiment.sessions','Interval and cutoff'],['experiment.budget','Budget'],['factor','Factors and policies'],['risk','Estimator'],['alpha','Model declaration'],['portfolio','Policy'],['experiment','Experiment']];
  const groupOf=(f)=>{ const p=f.path.join('.'); return (GROUPS.find(([prefix])=>p===prefix || p.startsWith(prefix+'.')) || ['','Declaration'])[1]; };
  const chosenText=(f,value)=>t('{n} of {m} chosen',{n:Array.isArray(value) ? value.length : 0,m:f.options.length});
  /* R4 (V339): the universe, whole or a sample of its names, as one size: empty is the whole
   * profile, N writes `whole + sample_mark + N`; the owner's bounds are its minimum cross-section
   * and one fewer than the Panel's names, and a Panel too small to sample holds the field. */
  /* A refused field's reason (V248): the owner's code in words, or the contract's error type in words. */
  const REASON_WORDS={missing:'Required',extra_forbidden:'Not a field of this section',greater_than:'Too small',greater_than_equal:'Too small',less_than:'Too large',less_than_equal:'Too large',
    int_parsing:'Not a whole number',int_from_float:'Not a whole number',float_parsing:'Not a number',bool_parsing:'Not true or false',string_type:'Not text',literal_error:'Not one of the choices',date_from_datetime_parsing:'Not a date',date_parsing:'Not a date',too_short:'Too few',too_long:'Too many'};
  const reasonWords=(r)=>REASON_WORDS[r] ? t(REASON_WORDS[r]) : coded(r);
  function universeInput(f,value,name,attrs) {
    const prefix=f.whole+f.sample_mark, size=String(value || '').startsWith(prefix) ? String(value).slice(prefix.length) : '', can=Number(f.sample_max)>=Number(f.sample_min);
    return {input:html`<input id="${name}" class="ui-field" type="number" inputmode="numeric" value="${size}" placeholder="${t('Whole universe')}" min="${f.sample_min}" max="${f.sample_max}" step="1"${attrs}${can ? '' : html` disabled`}>`,
      note:can ? t('Empty for the whole universe ({whole}); a sample holds {min} to {max} names.',{whole:f.whole,min:f.sample_min,max:f.sample_max}) : t('This Panel has too few names to sample; the study runs on the whole universe ({whole}).',{whole:f.whole})};
  }
  function control(f, i) {
    const value=valueAt(f.path), name='researchField'+i, invalid=Boolean(S.error) && ((S.errorBody?.fields || []).some(p=>namesField(f,Array.isArray(p) ? p.join('.') : String(p))) || namesField(f,S.errorField) || refusedBy(f,S.error)); // U39: every field the answer names
    const attrs=html`data-research-field="${i}"${S.busy==='run' ? html` disabled data-tip="${t('Submitting the confirmed PLAN; editing resumes when the owner answers.')}"` : ''}${invalid ? html` aria-invalid="true"` : ''}`;
    const sample=f.type==='universe' ? universeInput(f,value,name,attrs) : null;
    const bounds=sample ? sample.note : f.min!=null && f.max!=null ? t(f.min_exclusive ? 'More than {min}, at most {max}' : 'Between {min} and {max}',{min:f.min,max:f.max}) : f.min!=null ? t(f.min_exclusive ? 'More than {min}' : 'At least {min}',{min:f.min}) : f.max!=null ? t('At most {max}',{max:f.max}) : '';
    const unit=f.unit ? t('Unit: {unit}',{unit:f.unit}) : '';
    const input=sample ? sample.input : f.options && f.type==='multiple' ? choiceList(name, f.options.map((v,j)=>({value:String(j), label:v?.label ?? String(optionValue(v)), mono:v?.label==null, checked:Array.isArray(value) && value.includes(optionValue(v))})), attrs, html`<span id="researchFieldCount${i}">${chosenText(f,value)}</span>`)
      : f.options ? picker(name, [['',t('Choose explicitly')],...f.options.map((v,j)=>[String(j), v?.label ?? String(optionValue(v))])], {selected: value==null ? '' : String(f.options.findIndex(v=>optionValue(v)===value)), attrs, disabled:S.busy==='run', label:t(f.label || name)})
      : html`<input id="${name}" class="ui-field${f.type==='date' ? ' mono' : ''}" type="${f.type==='number' ? 'number' : 'text'}" value="${value ?? ''}"${attrs}${f.type==='date' ? html` data-date="true" inputmode="numeric" placeholder="YYYY-MM-DD" pattern="[0-9]{4}-[0-9]{2}-[0-9]{2}" autocomplete="off"` : ''}${f.min==null || f.min_exclusive ? '' : html` min="${f.min}"`}${f.max==null ? '' : html` max="${f.max}"`}${f.step==null ? '' : html` step="${f.step}"`}>`; // N6: a date in the product's ISO form, not the browser's order
    // one refusal names several fields: every one is ringed, the message is said once, under the first
    const said=invalid && !errorSaid; if(invalid) errorSaid=true;
    const reason=invalid && !said ? S.errorBody?.reasons?.[f.path.join('.')] : ''; // U39 (V248): a further refused field's own reason
    return html`<div class="field${invalid ? ' is-invalid' : ''}"><label for="${name}">${t(f.label || f.path.join('.'))}</label>${input}${[f.help ? t(f.help) : '',bounds,unit].filter(Boolean).map(x=>html`<small>${x}</small>`)}${reason ? html`<small class="field-error">${reasonWords(reason)}</small>` : ''}${said ? html`<small class="field-error" role="alert">${innerCode(S.error) ? explain(S.error) : ownerDetail(S.error) || explain(S.error) || S.error}</small>` : ''}</div>`;
  }
  let errorSaid=false;
  function controls() {
    errorSaid=false;
    const groups=new Map();
    S.fields.forEach((f,i)=>{ if(!visibleField(f) || (f.options && f.type!=='multiple')) return; const g=groupOf(f); if(!groups.has(g)) groups.set(g,[]); groups.get(g).push(control(f,i)); }); // single choices are the head's chips (round 56)
    const kept=S.document ? uncovered(S.fields,S.document) : [];
    return html`${[...groups].map(([g,fields])=>html`<section class="lab-control-group" data-stack-box="field"><h3>${t(g)}</h3><div class="form-grid">${fields}</div></section>`)}${kept.length ? html`${factsRef(html`${countText(kept.length,'{n} other declared field, kept exactly as declared','{n} other declared fields, kept exactly as declared')}`, html`<p class="caption">${t('These entries have no control here and travel unchanged with the declaration: {fields}. Switch to YAML to read or edit them.',{fields:kept.join(' · ')})}</p>`)}` : ''}`;
  }
  /* Where this draft comes from, as recorded references. Links reopen the exact object; the
   * origin travels in the route, so reload, back and forward keep the same source. */
  const STUDY_PAGES={'factor.screening-development':'factor','alpha.model-development':'alpha','risk.covariance-development':'risk','portfolio.policy-development':'portfolio'};
  const openLink=(label,page,extra)=>link(html`${label} ${icon('arrow')}`,page,'button compact',extra);
  function sourceRows() {
    const h=S.handoff;
    const rows=[[t('Exact input'),S.input ? html`${S.input} · ${cutoffOf(S.binding)}` : t('Select an input')]];
    if(h?.kind==='alpha_candidate') rows.push([t('Alpha candidate'),html`${t('candidate')} ${mono(String(h.candidate_id || '').replace(/^alpha-candidate-/,''),SHORT.id)} · ${t('Task')} ${mono(h.task_id,SHORT.id)} ${openLink(t('Open Alpha study'),'alpha',{study:h.task_id})}`]);
    else if(h?.kind==='foundation') rows.push([t('Foundation admission'),html`${mono(h.foundation_admission_hash,SHORT.hash)} ${openLink(t('Open Foundation'),'foundation',{foundation:h.foundation_admission_hash})}`]);
    else if(h?.kind==='factor') rows.push([t('Factor decision'),html`${t('Task')} ${mono(h.task_id,SHORT.id)} · ${t('the saved curation decision')} ${openLink(t('Open Factor study'),'factor',{study:h.task_id})}`]);
    else if(h?.kind==='features') rows.push([t('Prepared research features'),mono(h.feature_preparation_hash)]);
    if(S.origin) rows.push([t('Continued from'),html`${t('Task')} ${mono(S.origin,SHORT.id)} ${STUDY_PAGES[S.kind]==='portfolio' ? openLink(t('Open origin study'),'portfolio',{book:S.origin,session:''}) : openLink(t('Open origin study'),STUDY_PAGES[S.kind] || 'history',{study:S.origin})}`]);
    if(!h && !S.origin) rows.push([t('Source'),t('Authored directly from the selected input; no saved study is referenced.')]);
    return rows;
  }
  function scopeReader() {
    return html`${kv([...sourceRows(),[t('Creates'),S.work ? html`${t('Task')} ${btn(html`<span class="mono">${short(S.work.task)}</span>`,'task',S.work.task,'text-btn')}` : t('One Task, once the PLAN is confirmed')]])}${featuresSection()}`; // N6: after a run it names the Task, never `No task yet`
  }
  // a Factor draft's research-local definitions, from the head's ··· (the product's head button, 2026-09-25)
  const featureTool=()=>{
    if(S.kind!=='factor.screening-development') return [];
    const plan=S.featureInput?.definition_plan_hash;
    return plan || S.binding ? [{ic:'edit',action:plan ? 'feature-open-plan' : 'feature-definitions',value:plan || S.binding,word:t('Manage formula definitions'),why:t('A separate research-local definition. The experiment draft and global daily catalog stay unchanged.')}] : [];
  };
  function featureDefinitionsButton() {
    const plan=S.featureInput?.definition_plan_hash;
    return plan?btn(t('Manage formula definitions'),'feature-open-plan',plan,'button compact')
      :S.binding?btn(t('Manage formula definitions'),'feature-definitions',S.binding,'button compact'):'';
  }
  /* The features this study may use (the product's feature input): the three scopes as facts, each
   * column's definition and membership in Facts, and the way to the research-local definitions. */
  function featuresSection() {
    const input=S.featureInput?.input_binding_hash===S.binding ? S.featureInput : null, manage=featureDefinitionsButton();
    if(!input && !manage) return '';
    const status={RESEARCH_LOCAL_PREPARED:'Prepared for this research; not installed globally',MATCHES_RECORDED_INPUT:'Numerical method matches the recorded input',DIFFERS_FROM_RECORDED_INPUT:'Current numerical method differs from the recorded input',NOT_REGISTERED_IN_THIS_BUILD:'Historical column; no definition installed in this build',NOT_IN_SELECTED_INPUT:'Registered formula; not built in this input'};
    const definitions=input ? (input.columns || []).map(entry=>html`<section>
      <h4 class="mono">${entry.factor_id}</h4>
      <p class="caption">${t(status[entry.definition_status] || entry.definition_status)}</p>
      ${entry.current_definition ? html`<p class="mono">${entry.current_definition.formula}</p>${kv([[t('Required source fields'),entry.current_definition.required_fields.join(', ')],
        [t('Window / lag (sessions)'),`${entry.current_definition.window_sessions} / ${entry.current_definition.lag_sessions}`]])}` : ''}
    </section>`) : [];
    const scopes=input ? html`${kv([
      [t('Columns in the selected input'),count(input.input_factor_count)],
      [t('Registered formulas in this build'),count(input.registered_formula_count)],
      [t('Registered but absent from this input'),count(input.registered_not_in_input_count)]
    ])}${factsRef(t('Formula definitions and input membership'),html`${definitions}<p class="caption">${t('Definitions describe the current build. Registration does not build a column, and a published column does not guarantee usable values for every date and listing.')}</p>${codeRef(t('Exact definitions and recorded identities'),input)}`)}` : '';
    return html`<div class="lab-submitted"><h3>${t('Features available to this study')}${input ? infoMark(t('These are different scopes, not conflicting totals. Screening choices do not remove the multiple-testing context; curation chooses downstream inputs, and the model declaration names the columns actually used.')) : ''}</h3>${scopes}${manage ? html`<div class="flow">${manage}</div>` : ''}</div>`;
  }
  const planLine=(ic,title,body)=>html`<div class="plan-line">${icon(ic)}<div><strong>${title}</strong><p>${body}</p></div></div>`;
  /* The PLAN in reading order: proposed work, reuse, recorded source and boundaries; then the
   * exact identity and execution details the owner returned, unchanged. */
  function previewFacts() {
    const p=S.plan;
    if(!p) return '';
    const intent=p.execution_intent || {}, x=p.execution_preview || {}, ps=p.portfolio_source, as=p.alpha_source;
    // An existing candidate is whatever task the owner matched, in whatever state it is in. RUN
    // reuses only a completed one; in-flight work is joined, interrupted work needs recovery, and
    // any other state is returned as the answer. None of these is a finished result.
    const existing=intent.disposition==='EXISTING_EXECUTION_CANDIDATE', state=intent.lifecycle || '', ref=short(intent.task_id || '', SHORT.id);
    const creates=!existing ? t('One new {kind} Task. No result exists until it completes.',{kind:kindLabel(p.program.kind)})
      : state==='SUCCEEDED' ? t('An identical execution already completed as task {task}. RUN re-verifies that published result and reuses it; nothing new is computed.',{task:ref})
      : ['QUEUED','RUNNING'].includes(state) ? t('An identical execution is already {state} as task {task}. RUN joins that in-flight work instead of starting another; its computation is unfinished and no result exists yet.',{state:codeWords(state),task:ref})
      : state==='RECOVERY_REQUIRED' ? t('An identical execution was interrupted as task {task} and needs recovery. RUN reports it as in flight and does not restart it; recover it from Tasks before a result can exist.',{task:ref})
      : t('An identical execution exists as task {task} in state {state}. RUN returns that state instead of starting new work; resolve it before this declaration can produce a result.',{task:ref,state:codeWords(state)});
    const inspect=existing && intent.task_id ? html` ${btn(t('Inspect task'),'task',intent.task_id,'button compact')}` : '';
    const reuseSource=intent.upstream==='SAVED_ALPHA_SCORES_NO_REFIT'
      ? t('Saved Alpha scores of candidate {candidate} from task {task}; no model is refitted.',{candidate:ps?.candidate_id || '',task:short(ps?.alpha_task_id || '', SHORT.id)})
      : intent.upstream==='DECLARED_METHOD_ON_SEALED_INPUT' ? t('The declared method on the sealed input {input}; no data is downloaded.',{input:S.input || ''}) : (intent.upstream || '');
    const source=as ? t('Factor task {task}, curation {curation}, Foundation {foundation}.',{task:short(as.factor_task_id, SHORT.id),curation:short(as.curation_receipt_hash, SHORT.hash),foundation:as.foundation_admission_hash ? short(as.foundation_admission_hash, SHORT.hash) : t('none recorded')})
      : ps ? t('Alpha task {task}, candidate {candidate}, input binding {binding}.',{task:short(ps.alpha_task_id, SHORT.id),candidate:ps.candidate_id,binding:short(ps.input_binding_hash, SHORT.hash)})
      : t('The sealed input {input} only.',{input:S.input || ''});
    const impact=p.impact, words=impactWords(impact);
    const reuse=html`${reuseSource}${words.upstream ? html` ${words.upstream}` : ''}`;
    const changes=impact?.change?.origin_task_id ? impact.change.declared_fields : p.declaration_changes;
    const declarationChange=changes===null || changes===undefined ? t('Not a continuation: there is no earlier declaration of the same kind to compare.') : changes.length ? t('{n} differ from the origin study: {fields}.',{n:countText(changes.length,'{n} declared field','{n} declared fields'),fields:changes.map(c=>c.path.join('.')).join(', ')}) : t('Identical to the origin declaration.');
    const generated=impact?.change?.generated_output_workspace ? t('The output path is generated for this Program, not another choice.') : '';
    const changed=html`${declarationChange}${generated ? html` ${generated}` : ''}${words.change ? html` ${words.change}` : ''}`;
    // R4: the lane in the Host's own words (`lane_detail`, research_experiment_projection.lane_fields), never the page's
    const lane=p.research_lane==='EXPLORATION' ? planLine('flag',t('On a sample'),p.lane_detail ? html`<span class="owner-text">${p.lane_detail}</span>` : '') : '';
    const risk=x.risk_disposition ? planLine('evidence',t('Reads a Risk study'),html`${codeWords(x.risk_disposition)}${x.solver_calls!=null ? html` · ${countText(x.solver_calls,'{n} solver call','{n} solver calls')}` : ''}`) : '';
    return html`${lane}${planLine('plus',t(existing ? 'Existing work' : 'Will create'),html`${creates}${inspect}`)}${risk}${planLine('copy',t('Will reuse'),reuse)}${planLine('cube',t('Recorded source'),source)}${planLine('edit',t('Compared with the origin'),changed)}${planLine('lock',t('Will not change'),t('Saved results, current/default inputs and strategy authority. Numerical calls estimated for a fresh execution: {n}; this preview made none.',{n:x.expected_numerical_calls ?? ''}))}<p class="caption">${(p.limitations || []).map((code,i)=>html`${i ? ' · ' : ''}${coded(code)}`)}</p>${factsRef(html`${t('Exact plan identity and execution details')}`, html`${kv([[t('Method'),p.program.kind],[t('Plan identity'),p.plan_hash],[t('Execution intent'),intent.disposition || ''],[t('Candidate task'),intent.task_id || ''],[t('Numerical calls in preview'),p.numerical_call_count],[t('Statistical start'),x.statistical_start || ''],[t('Statistical end'),x.statistical_end || ''],[t('Statistical sessions'),x.statistical_session_count ?? ''],[t('Context factors'),x.context_factor_count ?? ''],[t('Minimum listings'),x.minimum_listings ?? ''],[t('Latest outcome session'),x.latest_outcome_session || '']])}`)}${LiveViews.researchTiming(p.timing)}${codeRef(t('Exact execution preview (JSON)'), x)}`;
  }
  /* ---- the entry: what this page is for, the four starts, the saved work to continue ---- */
  function valueAt(path) { return path.reduce((v,k)=>v?.[k],S.document); }
  const entryState=()=>{
    if(S.busy==='load') return 'loading';
    if(S.ready) return 'ready';
    if(GUIDED.has(S.kind) && !S.handoff && !S.origin) return 'guided';
    if(S.error) return 'refused';
    const prep=preparation();
    if(!Data.inputs().length && prep && !prep.inputs?.length) return 'unprepared';
    return 'no-input';
  };
  const studyWords=(v)=>{ const n=v.factor_ids?.length; return v.kind==='factor.screening-development' ? countText(n ?? 0,'{n} factor','{n} factors') : v.kind==='alpha.model-development' ? [methodWords(v.model_parameters?.family),methodWords(v.target_recipe_id)].filter(Boolean).join(' · ') || t('Alpha') : v.kind==='risk.covariance-development' ? (methodWords(v.risk_capability_handle) || t('Risk')) : v.candidate_id ? t('candidate')+' '+short(String(v.candidate_id).replace(/^alpha-candidate-/,''), SHORT.id) : t('Portfolio'); };
  const studyLabel=(v)=>`${t('Task')} ${short(v.task_id)} · ${studyWords(v)} · ${v.input_id || ''} · ${v.sessions?.end || ''}`;
  /* The kinds are the composer's tabs in every state (the user, 2026-09-24: 胶囊换成下划线标题, 这样有哪种实验
   * 一目了然): every kind the Lab makes, in the research chain's order, the current one lit; a draft being
   * run keeps its own. A creation chooses its kind where it declares; saved work continues from its study. */
  const kindTabs=(held)=>tabStrip(t('Experiment kind'),kinds.map(([kind,label])=>({word:t(label),on:kind===S.kind,off:held && kind!==S.kind,action:'research-start',value:kind})),'lab-kind-tabs');
  const versionLabel=(v)=>`${v.id} · ${v.date} · ${short(v.binding_hash, SHORT.hash)}`;
  function inputSelect(disabled=false) {
    const inputs=eligible();
    return html`<div class="field"><label id="expInputLabel" for="expInput">${t('Research input version')}</label>${picker('expInput',[['',t('Select an input')],...inputs.map(v=>[inputKey(v),versionLabel(v)])],{selected:JSON.stringify([S.input,S.binding]),kind:'text',disabled,attrs:disabled ? 'aria-describedby="researchModeReason"' : '',labelId:'expInputLabel'})}<small>${S.binding ? cutoffOf(S.binding) : t('Published versions only; the working store is never used for research.')}</small></div>`;
  }
  /* Round 56: the declaration's properties as chips under its title — the exact input, the kind,
   * every single-choice control of the schema (the Controls view's selects) and the interval; a
   * chip's choice writes the same field the select wrote, through the same owner reads. */
  const PROPERTY_KEYS={'factor.screening_policy':'s','factor.redundancy_policy':'r','alpha.target_recipe_id':'t','alpha.model_capability_handle':'m','alpha.model_parameters.family':'f'};
  const choiceWords=(f,v)=>v?.label ?? (f.path.join('.')==='portfolio.risk_task_id' ? `${t('Risk study')} ${LiveViews.shortRef(optionValue(v))}` : methodWords(optionValue(v))); // U41: a linked Risk study by its reference
  const optionLabel=(f,value)=>{ const o=f.options.find(v=>optionValue(v)===value); return o==null ? '' : choiceWords(f,o); };
  const unsettable=(f)=>f.path.reduce((v,k)=>v==null ? v : v[k],S.answered)===null; // U41, U42: a choice the owner answered as none may be none again
  function propertyRow() {
    const held=Boolean(S.busy) || S.busy==='run';
    const context=html`${kindLabel(S.kind)} · ${S.input || t('Select an input')}`;
    const inputs=eligible(), inputRows=inputs.map(v=>({value:inputKey(v),title:`${v.id} · ${v.date}`,meta:html`<span class="mono">${short(v.binding_hash, SHORT.hash)}</span> · ${cutoffOf(v.binding_hash)}`}));
    const bound=S.handoff && S.handoff.kind!=='factor';
    const chips=[propertyChip(t('Input'),S.input ? html`${S.input} · ${cutoffOf(S.binding)}` : '',{icon:'cube',key:'i',rows:held || bound ? [] : inputRows,context,action:'research-input',current:JSON.stringify([S.input,S.binding]),note:bound ? t('Bound to the saved source this draft continues from.') : t('Waiting for the product owner')})];
    if(app.mode==='controls') S.fields.forEach((f,i)=>{ if(!f.options || f.type==='multiple' || !visibleField(f)) return; const value=valueAt(f.path), key=PROPERTY_KEYS[f.path.join('.')] || '', none=unsettable(f) ? [{value:`${i}:-1`,title:t('None')}] : [];
      chips.push(propertyChip(t(f.label || f.path.join('.')),value==null && none.length ? t('None') : optionLabel(f,value),{key,rows:held ? [] : [...none,...f.options.map((v,j)=>({value:`${i}:${j}`,title:choiceWords(f,v)}))],context,action:'research-set',current:value==null ? (none.length ? `${i}:-1` : '') : `${i}:${f.options.findIndex(v=>optionValue(v)===value)}`,note:t('Waiting for the product owner')})); });
    const dates=S.fields.filter(f=>f.type==='date' && visibleField(f)).map(f=>valueAt(f.path)).filter(Boolean);
    if(app.mode==='controls' && dates.length) chips.push(propertyChip(t('Interval'),html`${dates[0]}${dates.length>1 ? html` — ${dates[1]}` : ''}${dates.length>2 ? html` · ${t('cutoff')} ${dates[2]}` : ''}`,{icon:'calendar',key:'d',press:'research-dates'}));
    return html`<div class="property-row" aria-label="${t('Declared properties')}">${chips}</div>`;
  }
  /* a property chip's choice: the same edit the select made, then the chips redrawn */
  function setField(i,j) {
    const f=S.fields[i]; if(!f || (j<0 ? null : optionValue(f.options[j]))===(valueAt(f.path) ?? null)) return; // choosing the current value edits nothing
    edit({target:{dataset:{researchField:String(i)},value:j<0 ? '' : String(j),id:''}}); // -1: none, as the owner answered it
    if(f.path.join('.')!=='alpha.model_parameters.family') paintKeepingHeight();
  }
  function focusDates() {
    const field=document.querySelector('#main [data-research-field][data-date]');
    if(field){ field.scrollIntoView({block:'center'}); field.focus(); }
  }
  function stateView(state) {
    const shell=(kind,body)=>html`<section class="panel pad lab-state" data-box="${kind==='refused' ? 'decision' : 'workspace'}" data-state="${kind}">${body}</section>`;
    if(state==='loading') return shell('loading',html`<h2>${t('Reading the {kind} declaration for {input}',{kind:kindLabel(S.kind),input:S.input || t('the selected source')})}</h2><p class="caption" role="status">${t('Waiting for the product owner')} · ${t('No PLAN, execution or preparation follows from this read.')}</p>`);
    if(state==='unprepared') { const prep=preparation(); return html`<div class="lab-state" data-state="unprepared">${emptyState(html`${t('This workspace holds no verified research input yet')}${infoMark(`${t(prep?.task_id ? 'A preparation Task is recorded for this workspace; its scene says where it stands. Research declarations need a published input version.' : 'One explicit preparation makes it research-ready: sources are captured, a Panel is built and one input version is published. Opening this page starts nothing.')} ${t('Saved research, if any, stays readable on the Factor, Alpha and Risk pages.')}`)}`,link(t(prep?.task_id ? 'Open the preparation' : 'Prepare the workspace'),'overview','button primary'))}</div>`; } // N6: the one empty state, not framed twice
    if(state==='refused') { const prep=preparation(); const unprepared=Boolean(prep && !prep.inputs?.length); return shell('refused',html`<h2>${t('The owner did not open this {kind} declaration',{kind:kindLabel(S.kind)})}</h2>${explain(S.error) ? html`<p>${explain(S.error)}</p>` : ''}<p class="mono refusal-code">${S.error}</p>${unprepared ? html`<p>${t('The workspace readiness read also reports no verified research input yet; the preparation path is on the Data pages.')} ${link(t('Open the preparation'),'overview','button compact')}</p>` : ''}<p class="caption">${t('Choose another input version or start another kind above; nothing was planned or run.')}</p>${GUIDED.has(S.kind) ? '' : inputSelect(Boolean(S.handoff && S.handoff.kind!=='factor'))}`); }
    const all=Data.inputs(), inputs=eligible(), other=all.filter(v=>!inputs.includes(v));
    const others=other.length ? html`<p class="caption">${countText(other.length,'{n} version not eligible:','{n} versions not eligible:')} ${other.map(v=>`${versionLabel(v)} · ${v.available ? codeWords(v.lifecycle) : t('source files unavailable')}`).join('; ')}</p>${link(t('Input versions'),'inputs','button compact')}` : '';
    return shell('no-input',html`<h2>${t('{kind}: choose the exact input version',{kind:kindLabel(S.kind)})}</h2><p class="caption">${t('The declaration is bound to one published input version and its cutoff; none is chosen for you.')}</p>${inputs.length ? inputSelect() : html`<p>${t('No published input version is available for research.')}</p>${link(t('Input versions'),'inputs','button compact')}`}${others}`);
  }
  function guideView() {
    const g=S.guide; if(!g) return '';
    const shell=(body)=>html`<section class="panel pad lab-state lab-guide" data-box="workspace" data-state="guided">${body}</section>`;
    const notice=g.error ? notRead(t('Source not read'),g.error,explain(g.error)) : '';
    const busy=g.busy ? html`<p class="caption" role="status">${t('Waiting for the product owner')}</p>` : '';
    if(g.kind==='portfolio.policy-development') {
      const alphas=saved().filter(v=>v.kind==='alpha.model-development' && v.lifecycle==='SUCCEEDED');
      const candidates=(g.alphaBody?.result?.candidates || []).filter(v=>v.status==='DEVELOPMENT_EVALUATED');
      return shell(html`<h2>${t('A Portfolio study starts from a saved Alpha candidate')}</h2><p class="caption">${t('The Portfolio owner applies the declared policy to the saved scores of one evaluated candidate of a saved Alpha study; no model is refitted. Choose the study, then the candidate; the draft opens bound to that source.')}</p>${notice}<div class="field"><label id="guideAlphaLabel" for="guideAlpha">${t('Saved Alpha study')}</label>${picker('guideAlpha',[['',alphas.length ? t('Choose explicitly') : t('No saved Alpha study yet')],...alphas.map(v=>[v.task_id,studyLabel(v)])],{selected:g.alpha,disabled:Boolean(g.busy),labelId:'guideAlphaLabel'})}</div>${busy}${g.alphaBody ? html`<div class="field"><label id="guideCandidateLabel" for="guideCandidate">${t('Evaluated candidate')}</label>${picker('guideCandidate',[['',candidates.length ? t('Choose explicitly') : t('No evaluated candidate in this study')],...candidates.map(v=>[v.candidate_id,v.candidate_id])],{selected:g.candidate,disabled:Boolean(g.busy),labelId:'guideCandidateLabel'})}<small>${countText(candidates.length,'{n} evaluated candidate.','{n} evaluated candidates; compare them on the Alpha page before choosing.')}</small></div>` : ''}<div class="flow">${typedBtn(t('Open the Portfolio draft from this candidate'),'research-guide-open','candidate','button primary',g.alpha && g.candidate && !g.busy ? '' : t('Choose the saved Alpha study and one of its evaluated candidates first.'))}${g.alpha ? openLink(t('Open Alpha study'),'alpha',{study:g.alpha}) : ''}</div><p class="caption">${t('Opening a draft fits nothing and admits no Task; PLAN and one explicit confirmation come after.')}</p>`);
    }
    const factors=saved().filter(v=>v.kind==='factor.screening-development' && v.lifecycle==='SUCCEEDED');
    const c=g.curation, decisions=c?.decisions || [], targets=c ? Data.inputs().filter(v=>v.available && (c.inputs || []).includes(v.id)) : [];
    const listed=g.foundations || [], foundations=listed.filter(v=>v.standing==='CURRENT'), historical=listed.filter(v=>v.standing==='HISTORICAL');
    const decisionPanel=html`<div class="lab-guide-option"><h3>${t('From a saved Factor decision')}</h3><div class="field"><label id="guideStudyLabel" for="guideStudy">${t('Saved Factor study')}</label>${picker('guideStudy',[['',factors.length ? t('Choose explicitly') : t('No saved Factor study yet')],...factors.map(v=>[v.task_id,studyLabel(v)])],{selected:g.study,disabled:Boolean(g.busy),labelId:'guideStudyLabel'})}</div>${g.busy==='curation' ? busy : ''}${c ? html`<div class="field"><label id="guideDecisionLabel" for="guideDecision">${t('Saved curation decision')}</label>${picker('guideDecision',[['',decisions.length ? t('Choose explicitly') : t('No decision saved for this study yet')],...decisions.map(d=>[d.receipt_hash,`${actorWords(d.actor_submission?.actor_kind)} · ${short(d.receipt_hash, SHORT.hash)}`])],{selected:g.decision,disabled:Boolean(g.busy),labelId:'guideDecisionLabel'})}<small>${decisions.length ? t('An immutable decision: which declared factors are carried forward and in which role.') : t('Make one on the Factor page: choose roles, give rationales and acknowledge the recorded limitations.')}</small></div><div class="field"><label id="guideInputLabel" for="guideInput">${t('Exact handoff input')}</label>${picker('guideInput',[['',t('Choose explicitly')],...targets.map(v=>[inputKey(v),versionLabel(v)])],{selected:g.target,disabled:Boolean(g.busy),labelId:'guideInputLabel'})}<small>${t('The published version the Alpha draft will be bound to; the study\'s own version is offered when it is still admitted.')}</small></div>` : ''}<div class="flow">${typedBtn(t('Open the Alpha draft from this decision'),'research-guide-open','decision','button primary',g.study && g.decision && g.target && !g.busy ? '' : t('Choose the saved Factor study, one of its saved decisions and the exact input version first.'))}${g.study ? openLink(t('Open Factor study'),'factor',{study:g.study}) : ''}</div></div>`;
    const unread=g.foundationsError ? html`<p class="caption">${t('The sealed Foundations could not be read:')} <span class="mono">${g.foundationsError}</span>${explain(g.foundationsError) ? html`<br>${explain(g.foundationsError)}` : ''}</p>${btn(t('Read again'),'research-guide-foundations','','button compact')}` : '';
    const foundationPanel=html`<div class="lab-guide-option"><h3>${t('From a sealed Foundation')}</h3>${g.foundations===null ? (g.foundationsError ? unread : busy) : foundations.length ? html`<div class="field"><label id="guideFoundationLabel" for="guideFoundation">${t('Sealed Foundation')}</label>${picker('guideFoundation',[['',t('Choose explicitly')],...foundations.map(v=>[v.admission.admission_hash,`${v.admission.input_id} · ${countText(v.admission.foundation?.ordered_factor_ids?.length ?? 0,'{n} factor','{n} factors')} · ${short(v.admission.admission_hash, SHORT.hash)}`])],{selected:g.foundation,disabled:Boolean(g.busy),labelId:'guideFoundationLabel'})}<small>${t('Immutable research material sealed from a saved decision; not a foundation model.')}</small></div>` : html`<p class="caption">${t(historical.length ? 'No current Foundation is sealed in this workspace; new work starts on a current one, sealed from a saved decision on the Factor page.' : 'No Foundation is sealed in this workspace. One is sealed from a saved decision on the Factor page (preview, then an explicit seal); it is immutable research material, not a model.')}</p>`}${historical.length ? html`<p class="caption">${countText(historical.length,'{n} historical Foundation is not offered','{n} historical Foundations are not offered')} · ${[...new Set(historical.map(v=>v.standing_code).filter(Boolean))].map(code=>coded(code))}</p>` : ''}<div class="flow">${typedBtn(t('Open the Alpha draft from this Foundation'),'research-guide-open','foundation','button primary',g.foundation && !g.busy ? '' : 'Choose a sealed Foundation first.')}${link(t('Foundations'),'foundation','button compact')}</div></div>`;
    return shell(html`<h2>${t('Alpha modeling starts from a saved research decision')}</h2><p class="caption">${t('The Alpha owner admits a draft only from a saved Factor curation decision, or from a Foundation sealed from one. The draft it opens is source-bound: the factors it may use are the decision\'s, and the model declaration is completed here before PLAN. Opening it fits nothing.')}</p>${notice}<div class="lab-guide-grid">${decisionPanel}${foundationPanel}</div>`);
  }
  const STATE_WORDS={loading:['queued','reading'],refused:['error','not opened'],unprepared:['blocked','not ready'],'no-input':['draft','no input chosen'],guided:['draft','saved source required']};
  const labState=(state)=>S.plan ? stateLine('planned') : state==='ready' ? stateLine('draft',{word:t(S.dirty ? 'edited draft' : 'draft')}) : stateLine(STATE_WORDS[state][0],{word:t(STATE_WORDS[state][1])});
  const button=(label,action)=>typedBtn(t(label),action,'','button primary',S.busy || !S.ready ? t('Wait for a ready declaration') : '',S.busy || !S.ready ? 'aria-describedby="labStateReason"' : '');
  /* The Scope / PLAN reader (round 69; round 92: a reading pane in the lane, never the side
   * column) -- the shared PLAN, an exact reuse or the Task on the page, then the PLAN preview with
   * its confirmation (or the scope before a PLAN) and the exact references. It opens with a PLAN,
   * a shared PLAN, a Task on the page or from ···, and repaints with the draft. */
  function reader() {
    return html`<div id="planStateNotice" hidden></div>${sharedPanel()}${S.reuse ? html`<div class="lab-submitted"><h3>${t('Exact reuse')}</h3><p class="caption">${t('An identical execution already completed as Task {task}; nothing new was computed and no Task was admitted.',{task:short(S.reuse.task)})}${S.reuse.realization ? html` ${t('Computed')}: ${LiveViews.realizationWords(S.reuse.realization)}` : ''}</p><div class="flow">${btn(t((FAMILIES[S.reuse.kind] || GENERIC_FAMILY).reader),'task-result',S.reuse.task,'button primary compact')}${btn(t('Inspect task'),'task',S.reuse.task,'button compact')}</div></div>` : S.work ? html`<div class="lab-submitted"><h3>${t('Task on this page')}</h3><p class="caption">${t('Task {task} is read below on this page; Tasks shows the same Task.',{task:short(S.work.task)})}</p>${btn(t('Inspect task'),'task',S.work.task,'button compact')}</div>` : ''}${S.plan ? '' : html`<p class="caption">${hint(t('Before a PLAN'), t('PLAN validates the declaration with its owner and previews the work: what would be created, reused and left unchanged. It runs nothing; only the confirmation after it admits a Task.'))}</p>`}${S.plan ? html`${previewFacts()}<div class="plan-commit">${button('Review and run','research-confirm')}<p class="caption">${t('Opens the exact confirmation; only that confirmation admits a Task.')}</p></div>` : scopeReader()}${S.handoff || S.origin ? html`${codeRef(t('Exact source reference'), {origin_task_id:S.origin || null,handoff_source:S.handoff || null,research_input_id:S.input || null,input_binding_hash:S.binding || null})}` : ''}${S.authoringOptions ? html`${codeRef(t('Installed authoring options'), S.authoringOptions)}` : ''}${factsRef(html`${t('Research contract')}`, html`<div>${t('PLAN is not execution. Only explicit confirmation can admit a Task. Saved inputs, strategies and results are not activated or replaced.')}</div>`)}`;
  }
  const readerTitle=()=>S.plan ? t('PLAN preview') : S.shared ? t('Shared PLAN') : S.work ? t('Task on this page') : t('Execution scope');
  function openReader() { S.readerOpen=true; paint(); const pane=typeof document!=='undefined' ? document.querySelector('#main .reading-pane') : null; if(pane && typeof Geometry!=='undefined' && Geometry.focusQuietly) Geometry.focusQuietly(pane); return true; }
  function closeReader() { if(!S.readerOpen) return false; S.readerOpen=false; paint(); return true; }
  const readingPaneMarkup=()=>S.readerOpen ? readingPane(readerTitle(), kindLabel(S.kind), reader(), 'lab-reader-close', 'lab-reading', ['lab-reader','']) : '';
  function page() {
    const state=entryState();
    // While no declaration is ready the primary button is held; its reason is the state badge in
    // the meta line (aria-describedby), so Controls.sync prints no note under the actions and
    // the head keeps its height.
    const h=S.handoff, sourceNote=h?.kind==='alpha_candidate' ? html`<span>${t('from Alpha candidate')} ${mono(String(h.candidate_id || '').replace(/^alpha-candidate-/,''),SHORT.id)} · ${t('Task')} ${mono(h.task_id,SHORT.id)}</span>` : h?.kind==='foundation' ? html`<span>${t('from Foundation')} ${mono(h.foundation_admission_hash,SHORT.hash)}</span>` : h?.kind==='factor' ? html`<span>${t('from Factor decision')} ${mono(h.task_id,SHORT.id)}</span>` : S.origin ? html`<span>${t('continued from')} ${t('Task')} ${mono(S.origin,SHORT.id)}</span>` : '';
    const heading=objectHead(kindLabel(S.kind), html`<span id="labStateReason">${labState(state)}</span><span>${S.input ? html`${S.input} · ${cutoffOf(S.binding)}` : t('Select an input')}</span>${sourceNote}`, state==='ready' ? button('Preview work','research-plan') : '','',[{ic:'eye',action:'lab-reader',word:t('Scope / PLAN'),why:t('The execution scope, the PLAN preview and its confirmation, in the inspector')},{ic:'copy',action:'copy-yaml',word:t('Copy declaration'),why:t('The draft as YAML, to the clipboard')},...featureTool()]);
    const unrep=S.ready && S.document ? unrepresented(S.fields,S.document) : [];
    const modeReason=S.busy ? t('Waiting for the product owner') : !S.fields.length ? t('This origin provides YAML editing; no control schema was supplied.') : unrep.length ? t('The declared value of {fields} cannot be shown by these controls; the YAML keeps them exactly.',{fields:unrep.join(', ')}) : t('Controls and YAML are two views of the same declaration; switching formats it, and never plans or runs.');
    const yamlInvalid=S.error && S.errorField && app.mode==='yaml' ? ' aria-invalid="true"' : ''; // the named field lives in the YAML text
    const editor=app.mode==='controls' ? html`<div class="controls-content">${controls()}</div>` :
      codeEditor('yamlEditor',app.yaml,{lang:'yaml',label:t('YAML research declaration'),attrs:[!S.ready || ['run','load','format'].includes(S.busy) ? 'readonly' : '',yamlInvalid.trim()].filter(Boolean).join(' ')});
    const workbench=state==='ready' ? html`<section class="editor-workbench panel" data-box="workspace"><header class="editor-workbench-header"><span class="editor-file">${icon(app.mode==='controls' ? 'edit' : 'file')}<strong>${app.mode==='controls' ? t('Declaration') : 'research.yaml'}</strong><span class="editor-file-note">${t(S.dirty ? 'Edited draft · kept on this page only; copy it to keep it' : 'Draft declaration')}</span></span>${tabStrip(t('Declaration views'),[{word:t('Controls'),action:'mode',value:'controls',on:app.mode==='controls',attrs:!S.fields.length || S.busy ? 'disabled aria-describedby="researchModeReason"' : 'aria-describedby="researchModeReason"'},{word:t('YAML'),action:'mode',value:'yaml',on:app.mode==='yaml',attrs:S.busy ? 'disabled aria-describedby="researchModeReason"' : 'aria-describedby="researchModeReason"'}],'editor-tabs')}</header><div class="declaration-context">${propertyRow()}<div class="lab-produces"><span class="editor-kind">${hint(t('What this declares'), t(KIND_LINES[S.kind] || 'A research declaration on the exact input.'))}</span></div></div><span class="sr-only" id="researchModeReason">${modeReason}</span><div class="declaration-content">${editor}</div><footer class="editor-statusbar"><div id="validationNotice" aria-live="polite">${S.error ? html`<span class="caption">${S.error}${explain(S.error) ? html` · ${explain(S.error)}` : ''}${S.errorBody?.document_location ? html` · ${t('line {line}, column {column}',S.errorBody.document_location)}` : ''}</span>` : S.busy ? t('Waiting for the product owner') : S.plan ? t('PLAN matches this declaration') : hint(t('Draft'), t('Editing is not execution; create a PLAN to validate.'))}</div></footer></section>`
      : state==='guided' ? guideView() : stateView(state);

    // a refusal a field carries (its ring and its message) is not repeated above the form (round 21)
    const carried=Boolean(S.error) && state==='ready' && app.mode==='controls' && S.fields.some(f=>visibleField(f) && (namesField(f,S.errorField) || refusedBy(f,S.error)));
    // the kinds' tabs lead the page in every state (the user, 2026-09-24)
    const kindRow=kindTabs(S.busy==='run');
    return html`${heading}${kindRow}${S.error && state==='ready' && !carried ? refusal({code:explain(S.error) ? S.error : '',reason:explain(S.error) || String(S.error).replace(/:\s*:\s*Value error,\s*/,' · ')},TONE.attention,{word:t('Declaration refused'),more:refusalWays(S.errorBody)}) : ''}${S.error ? prerequisitesPanel(S.errorBody?.prerequisites,{id:S.input,binding:S.binding}) : ''}${S.missing.length ? noteLine(t('Explicit model declaration required'),html`${S.missing.join(', ')} · ${t('Complete these fields before PLAN; nothing is chosen for you.')}`) : ''}<div class="lane-split${S.readerOpen ? ' has-reading' : ''}"><div class="lane-main"><div class="lab-workspace" id="labWorksurface" data-stack-box="box" data-state="${state}">${workbench}</div>${S.error && S.errorBody?.prerequisites ? '' : prerequisitesPanel(S.prereq,{id:S.input,binding:S.binding})}${workSection()}</div>${readingPaneMarkup()}</div>`;
  }
  function leaveReads() {
    const retry=S.busy==='load' || S.shared?.loading || Boolean(S.guide?.busy);
    S.revision++;S.sharedTicket++;
    if(['load','format','plan'].includes(S.busy))S.busy='';
    if(S.shared?.loading)S.shared=null;
    if(S.guide?.busy){S.guide.busy='';S.guide.error='';S.guide.foundationsError='';}
    if(retry)S.initialized=false;
  }
  return {ready,page,leaveReads,reader,openReader,closeReader,controls,edit,setField,focusDates,setMode,preview,reviewRun,run,continueFrom,receiveHandoff,routeContext,followRoute,
    inspectShared,adoptShared,replanShared,confirmAdoption,dismissShared,diffLines,shared:()=>S.shared,adopting:()=>S.adopting,
    dirty:unsaved,observe,watch,dismissWork,readWork:()=>readWork(false,true),guideRead,guideChange,guideOpen,entryState,
    // the area's words and declared scope for the Task on the page, for consumers without the composed area
    declaredOf,
    sceneFacts:()=>{const w=S.work,kind=familyOf(w),d=declaredFor(w),count=declaredCount(d,kind);return {family:kind || null,title:familyWords(w).title,reader:familyWords(w).reader,count:count ? String(count).replace(/<[^>]+>/g,'') : t('1 unit'),declared:String(declaredLine(d,kind,w?.view || {})),calls:d?.calls!=null ? t(d.callsCounted ? '{n} recorded by the completing execution' : '{n} estimated by the PLAN for a fresh execution',{n:d.calls}) : t('not previewed on this page')};},
    // An explicit move into the editor from another page (the preparation scene, the inputs page)
    // makes the one history entry for the place being left, like every other explicit navigation.
    useInput:async key=>{const chosen=Data.navigationIntent();if(unsaved()&&!(await askReplace()))return;if(!Data.navigationCurrent(chosen))return;S.initialized=true;S.handoff=null;objectEntry('page:lab');app.page='lab';replaceHash({page:'lab',origin:'',draft_source:''});const loading=load('factor.screening-development',key),navigation=Data.navigationIntent();await loading;if(Data.navigationCurrent(navigation))replaceHash({page:'lab',...routeContext()});},
    dismissConfirmation:()=>{S.confirming=null;S.adopting=null;},
    answerReplace,
    selectKind:(kind)=>choose(kind,JSON.stringify([S.input,S.binding])),forget,
    selectInput:(key)=>{if(S.handoff){if(S.handoff.kind!=='factor')return;const [id,hash]=JSON.parse(key);return receiveHandoff({...S.handoff,research_input_id:id,input_binding_hash:hash});}return choose(S.kind,key,S.origin);},
    copy:()=>copyText(app.mode==='yaml' ? app.yaml : JSON.stringify(S.document,null,2)),
    work:()=>S.work ? {task:S.work.task,family:familyOf(S.work) || null,lifecycle:S.work.view?.lifecycle || null,published:published(S.work),stale:Boolean(S.work.stale),lateReads:S.work.lateReads,mismatch:S.work.mismatch ? (S.work.mismatch.task_id!==S.work.task ? 'other-task:'+S.work.mismatch.task_id : S.work.mismatch.task_kind) : null,readback:S.work.readback ? {failures:S.work.readback.failures,error:S.work.readback.error} : null} : null,
    context:()=>({input_id:S.input,input_binding_hash:S.binding,origin_task_id:S.origin,handoff_source:S.handoff,plan_hash:S.plan?.plan_hash || null,work_task_id:S.work?.task || null})};
})();
