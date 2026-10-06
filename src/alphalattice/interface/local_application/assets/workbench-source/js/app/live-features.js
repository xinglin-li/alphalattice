/* Research-local formulas (the product's Local Feature, read in the UI's places on 2026-09-25; LS5,
 * LY5): the existing research-local definition and Task owners, without replacing the Lab draft.
 * A saved definition is not a materialized input; a raw build is not Panel admission. One page,
 * three places, each its own address: the change composed over one research input
 * (`input_binding`; `feature_parent`, the saved revision it builds on), a saved PLAN
 * (`feature_plan`, read as its page, never in a dialog), and the values a build Task prepared
 * (`feature_build`, the Task's result). The page follows its address: a link, Back and a reload
 * read the same place. */
const LiveFeatures = (() => {
  const S={key:'',generation:0,controls:null,controlsKey:'',operation:'CREATE',selected:'',spec:null,reason:'',plan:null,build:null,error:'',errorBody:null,busy:false,drafts:new Map()};
  const clone=value=>JSON.parse(JSON.stringify(value));
  const operations={CREATE:'Add a local formula',UPDATE:'Revise a local formula',RENAME:'Rename in this research',RETIRE:'Retire from this research'};
  const fields=[['factor_id','Factor ID','text'],['family','Family','text'],['formula','Formula statement (identity-bound)','text'],
    ['window_sessions','Window (trading sessions)','number'],['lag_sessions','Economic lag (trading sessions)','number'],
    ['minimum_observations','Minimum observed source rows','number'],['literature_sources','Sources / notes (one per line)','lines']];
  const choices=()=>S.operation==='CREATE'
    ? (S.controls?.registered_formulas || []).filter(v=>(S.controls.registered_kernels || []).some(k=>k.formula_ref===v.formula_ref))
    : S.controls?.definitions || [];
  // a ticket is one visit's: a new place on this page, or leaving it, ends the visit it was taken in
  const current=ticket=>ticket===S.generation && app.page==='features';
  const paint=()=>{ if(app.page==='features') (typeof patchMain==='function' ? patchMain : render)(); };
  const fail=(e)=>{ S.error=e.message; S.errorBody=e.body && typeof e.body==='object' ? e.body : null; };
  /* The place the address names: a build's values, a saved PLAN, or a change over an input. */
  function route() {
    const h=hashParams(), build=h.get('feature_build') || '', plan=h.get('feature_plan') || '';
    if(build) return {view:'build',key:'build:'+build,id:build};
    if(plan) return {view:'plan',key:'plan:'+plan,id:plan};
    const binding=h.get('input_binding') || '', parent=h.get('feature_parent') || '';
    return {view:'compose',key:`compose:${binding}:${parent}`,binding,parent};
  }
  // the editor a composer address holds while the reader is elsewhere (a PLAN, Back): kept on this page only
  const keep=()=>{ if(S.key.startsWith('compose:')) S.drafts.set(S.key,{operation:S.operation,selected:S.selected,reason:S.reason,spec:S.spec}); };
  /* Leaving the page ends its visit (the router, when it paints another page). A PLAN or a build
   * already sent completes with its owner, but its answer no longer moves the reader: a return --
   * a link, Back or Forward, even to the same address -- is a new visit, and an answer that
   * arrives meanwhile is said by a toast, never by taking the page. */
  function leave() { if(!S.key) return; keep(); S.generation++; S.key=''; S.busy=false; }
  function follow(r) {
    keep();
    const ticket=++S.generation; S.key=r.key; S.error=''; S.errorBody=null; S.busy=false;
    if(r.view==='build') {
      if(S.build?.task_id===r.id) return; // handed over by the Task's result
      S.build=null; S.busy=true;
      Data.read('/api/features/build?'+new URLSearchParams({task_id:r.id}))
        .then(b=>{ if(current(ticket)) S.build=b; },e=>{ if(current(ticket)) fail(e); })
        .finally(()=>{ if(current(ticket)) { S.busy=false; paint(); } });
      return;
    }
    if(r.view==='plan') {
      const known=S.plan?.plan_hash===r.id;
      if(!known) { S.plan=null; S.busy=true; }
      (known ? Promise.resolve(S.plan) : Data.read('/api/features/readback?'+new URLSearchParams({feature_plan_hash:r.id})))
        .then(p=>{ if(!current(ticket)) return; S.plan=p; S.busy=false; paint(); return controlsFor(p.input_binding_hash,p.document.parent_plan_hash || '',ticket); },
          e=>{ if(current(ticket)) { fail(e); S.busy=false; paint(); } });
      return;
    }
    S.plan=null;
    const draft=S.drafts.get(r.key);
    Object.assign(S,{operation:draft?.operation || 'CREATE',selected:draft?.selected || '',reason:draft?.reason || '',spec:draft?.spec ? clone(draft.spec) : null});
    if(r.binding) void controlsFor(r.binding,r.parent,ticket);
  }
  /* The owner's controls for one input and the revision a change builds on; read once per pair.
   * Called while the page renders: nothing before the read repaints. */
  async function controlsFor(binding,parent,ticket) {
    const key=binding+':'+parent;
    if(S.controlsKey===key && S.controls) return retire();
    S.controls=null; S.controlsKey=key;
    try {
      const c=await Data.read('/api/features/controls?'+new URLSearchParams({input_binding_hash:binding,...(parent?{feature_plan_hash:parent}:{})}));
      if(current(ticket) && S.controlsKey===key) { S.controls=c; retire(); }
    } catch(e) { if(current(ticket)) fail(e); }
    finally { if(current(ticket)) paint(); }
  }
  // a retirement carries no specification: the editor shows the definition it retires, once the controls name it
  function retire() {
    if(S.key.startsWith('compose:') && S.operation==='RETIRE' && S.selected && !S.spec) {
      const found=S.controls?.definitions?.find(v=>v.factor_id===S.selected);
      if(found) S.spec=clone(found);
    }
  }
  function select(id) {
    if(S.busy)return;
    const spec=choices().find(v=>v.factor_id===id);
    S.selected=id;S.spec=spec?clone(spec):null;S.error='';S.errorBody=null;paint();
  }
  function operation(value) {
    if(S.busy || !Object.hasOwn(operations,value) || value===S.operation)return;
    S.operation=value;select('');
  }
  function edit(key,value) {
    if(S.busy)return;
    if(key==='reason'){S.reason=value;return;}
    const field=fields.find(v=>v[0]===key);
    if(!field||!S.spec)return;
    if(key==='factor_id' && !['CREATE','RENAME'].includes(S.operation))return;
    S.spec[key]=field[2]==='number' ? (value===''?null:Number(value)) : field[2]==='lines' ? [...new Set(value.split('\n').map(s=>s.trim()).filter(Boolean))].sort() : value;
  }
  const inputWords=(binding)=>{ const v=Data.inputVersion(binding); return v ? `${v.id} · ${LiveViews.cutoffText(LiveViews.inputState(binding))}` : mono(binding); };
  const notice=()=>S.error ? refusal(S.errorBody || {message:S.error},TONE.attention,{word:t('The owner refused this step')}) : '';
  /* ---- the change being composed: a form page (PG5), its kinds a segmented choice (CT3) ---- */
  function field(key,label,type) {
    const id='featureField-'+key, held=S.busy ? ' disabled' : '';
    const input=type==='lines'
      ? html`<textarea id="${id}" class="ui-field" data-feature-field="${key}"${held}>${(S.spec[key] || []).join('\n')}</textarea>`
      : html`<input id="${id}" class="ui-field" type="${type}" data-feature-field="${key}" value="${S.spec[key] ?? ''}"${held}${key==='factor_id'&&!['CREATE','RENAME'].includes(S.operation)?' readonly aria-describedby="featureRenameNote"':''}>`;
    const note=key==='factor_id' && S.operation==='UPDATE' ? html`<small id="featureRenameNote">${t('Use Rename to change the identifier; an ordinary revision retains it.')}</small>`
      : key==='formula' ? html`<small>${t('The formula statement participates in numerical identity. Use Sources / notes for description-only changes.')}</small>` : '';
    return html`<div class="field"><label for="${id}">${t(label)}</label>${input}${note}</div>`;
  }
  function editor() {
    const c=S.controls;
    const scope=html`<section class="lab-control-group"><h3>${t('Research input version')}</h3>${kv([[t('Input'),inputWords(c.input_binding_hash)],
      [t('Columns in the selected input'),count(c.source_input_factor_count)],[t('Local definitions'),count(c.local_definition_factor_count)],
      [t('Supported interval'),`${c.source_period.start} — ${c.source_period.end}`]])}</section>`;
    const definition=S.spec && S.operation!=='RETIRE' ? html`<section class="lab-control-group"><h3>${t(operations[S.operation])}</h3><div class="form-grid">${fields.map(([key,label,type])=>field(key,label,type))}</div>
      ${kv([[html`${t('Registered computation')}${infoMark(t('Only installed arithmetic runs. Changing its description does not install a new algorithm. Validation tolerances and source conventions stay bound to the selected recipe.'))}`,html`<span class="mono">${S.spec.formula_ref}</span>`],
        [t('Required source fields'),S.spec.required_fields.join(', ')],[t('Return convention'),S.spec.return_convention]])}</section>` : '';
    const retirement=S.operation==='RETIRE' ? html`<section class="lab-control-group"><h3>${t(operations.RETIRE)}</h3><p>${t('Retirement removes the definition from this research revision, not old artifacts or the global catalog.')}</p></section>` : '';
    const reason=html`<section class="lab-control-group"><div class="field"><label for="featureReason">${t('Reason for this change')}</label><textarea id="featureReason" class="ui-field" data-feature-field="reason"${S.busy?' disabled':''}>${S.reason}</textarea></div></section>`;
    return html`<section class="editor-workbench panel" data-box="workspace"><header class="editor-workbench-header"><span class="editor-file">${icon('edit')}<strong>${t('Research-local formulas')}</strong>${infoMark(t('A separate research-local definition. The experiment draft and global daily catalog stay unchanged.'))}</span></header><div class="declaration-content"><div class="controls-content">${scope}${definition}${retirement}${reason}</div></div></section>`;
  }
  function composeView(r) {
    const chosen=S.operation==='CREATE' ? 'Registered formula preset' : 'Local formula';
    const kinds=html`<div class="segmented ui-segments" role="group" aria-label="${t('Change')}">${Object.entries(operations).map(([key,word])=>segBtn(t(word),'feature-operation',key,S.operation===key,S.busy ? 'disabled' : ''))}</div>`;
    const selection=S.controls ? html`${subjectChoice(t(chosen),'featureSelection',[['',t('Choose explicitly')],...choices().map(v=>[v.factor_id,v.factor_id])],{selected:S.selected,disabled:S.busy})}${S.operation==='CREATE' ? infoMark(t('These presets have standalone registered kernels; required source dependencies are still checked by the build. The full catalog remains available in input discovery.')) : ''}` : '';
    const primary=typedBtn(t('Preview and save definition (PLAN)'),'feature-plan','','button primary',S.busy ? t('Waiting for the product owner') : !S.spec ? t('Choose a formula first') : '');
    const heading=objectHead(t(operations[S.operation]),'',r.binding ? primary : '',stateLine('draft'),[],{top:r.binding ? html`${kinds}${selection}` : ''});
    if(!r.binding) return html`${heading}${emptyState(t('Select an input'),link(t(pageWord('lab')),'lab','button compact'),'','elsewhere')}`;
    return html`${heading}${notice()}${S.controls ? editor() : skeleton('body')}`;
  }
  /* ---- a saved PLAN: an object's page (PG5, PG6); the confirmation is its primary ---- */
  function planName(p) {
    const edits=p.document.edits || [];
    return edits.length===1 ? `${t(operations[edits[0].operation] || edits[0].operation)} · ${edits[0].factor_id}` : countText(edits.length,'{n} local formula change','{n} local formula changes');
  }
  function planView() {
    const p=S.plan, w=p.work;
    const tools=[{ic:'edit',action:'feature-edit-request',word:t('Edit this request'),why:t('Back to the editor with this exact change')},
      {ic:'branch',action:'feature-revise',word:t('Revise this definition'),why:t('A new change on top of this saved revision')},
      {ic:'file',action:'feature-export',word:t('Export YAML'),why:t('The saved request as YAML')}];
    const head=objectHead(planName(p),html`<p class="lede">${t('Definition saved; values are not implied by this PLAN.')}</p>`,
      typedBtn(t('Confirm and build local columns'),'feature-build','','button primary',S.busy?t('Waiting for the product owner'):''),stateLine('planned'),tools,
      {object:true,id:p.plan_hash,facts:[[t('Input'),inputWords(p.input_binding_hash)]]});
    const rows=[[t('Definition revision'),mono(p.candidate_revision.revision_hash)],[t('Local definitions'),count(p.candidate_revision.features.length)],
      [t('Numerically changed columns'),w.base_compute_factor_ids.join(', ')||'0'],...(p.required_fields.length ? [[t('Required source fields'),p.required_fields.join(', ')]] : []),
      ...(p.trial_baseline ? [[t('A trial runs against'),html`<span class="owner-text">${p.trial_baseline}</span>`]] : [])]; // V354: stated before any build
    return html`${head}${notice()}${panel(t('Properties'),t('Build reuses unchanged raw values and prepares only the local additions in one Task. The original input and global catalog stay unchanged; no Factor experiment or model training runs.'),
      html`${kv(rows,'kv-columns')}${codeRef(t('Exact change and impact'),{declaration:p.document,delta:p.delta,work:p.work})}`)}`;
  }
  /* ---- the values a build Task prepared: the Task's result, read as its page ---- */
  // U52: the build's window against T0 beside its facts, every statement in the Facts (V347)
  const factsRefIf=(scope)=>LiveViews.temporalAll(scope) ? factsRef(t('Time and survivorship'),LiveViews.temporalAll(scope)) : '';
  function buildView() {
    const b=S.build, r=b.receipt, columns=b.columns || [], state=String(b.status || '').toLowerCase();
    if(b.status!=='SUCCEEDED' || !r) return html`${objectHead(t('Research-local Formula values'),'',btn(t('Inspect task'),'task',b.task_id,'button primary'),stateLine(state),[],{object:true,id:b.task_id})}${notice()}${refusal({failure_code:b.failure_code || b.status},'warning',{state})}`;
    const research=b.next_requests?.research;
    const primary=research ? link(t('Research with these features'),'lab','button primary',{draft_source:JSON.stringify({kind:'features',input_binding_hash:b.preparation.input_binding_hash,feature_preparation_hash:b.preparation.content_hash}),origin:'',plan:''}) : '';
    const tools=[{ic:'edit',action:'feature-open-plan',value:r.definition_plan_hash,word:t('Manage formula definitions'),why:t('The saved definition PLAN these values were built from')},
      {ic:'task',action:'task',value:b.task_id,word:t('Inspect task'),why:t('The Task that built these values')}];
    const head=objectHead(columns.map(c=>c.factor_id).join(', ') || t('Research-local Formula values'),
      html`<p class="lede">${t('Verified raw Formula columns. Not a new admitted Panel or Factor/Alpha input; the global catalog is unchanged.')}</p>`,primary,stateLine(state),tools,{object:true,id:b.task_id});
    const prepared=b.preparation ? panel(t('Prepared research features'),t('Prepared research features can be used directly in a Factor declaration. PLAN checks the exact input before any experiment runs.'),
      html`${kv([[t('Prepared local columns'),count(columns.length)],[t('Preprocessing calls in the recorded attempt'),count(b.preparation.preprocessing_calls_in_this_attempt)]])}${codeRef(t('Exact preparation receipt'),b.preparation)}`) : '';
    const headers=[{label:t('Factor ID'),type:'id'},{label:t('Supported interval'),type:'date'},{label:t('Trading sessions'),type:'num'},{label:t('Listings'),type:'num'},{label:t('Available values / total cells'),type:'num'},{label:t('Physical bytes'),type:'num'}];
    const rows=columns.map(c=>tr([html`<span class="mono">${c.factor_id}</span>`,`${c.start} — ${c.end}`,count(c.sessions),count(c.listings),`${count(c.available)} / ${count(c.cells)}`,count(c.physical_bytes)]));
    const values=html`<section class="panel" data-box="table">${sectionHead(html`${t('Research-local Formula values')} <span class="num">${count(columns.length)}</span>`)}${table(headers,rows,'',{report:true,grid:true,countLine:false})}</section>`;
    const receipt=panel(t('Properties'),'',html`${kv([[t('Computed columns'),r.computed_factor_ids.join(', ') || '0'],[t('Reused columns'),r.reused_factor_ids.join(', ') || '0'],
      [t('Inherited input columns'),count(r.inherited_input_factor_ids.length)],[t('Kernel calls in the recorded attempt'),count(r.kernel_calls_in_this_attempt)],...LiveViews.temporalRow(b.temporal_scope)],'kv-columns')}${factsRefIf(b.temporal_scope)}${codeRef(t('Exact materialization receipt'),r)}`);
    return html`${head}${notice()}${prepared}${values}${receipt}`;
  }
  function page() {
    const r=route();
    if(r.key!==S.key) follow(r);
    if(r.view==='compose') return composeView(r);
    const body=r.view==='plan' ? S.plan : S.build;
    if(!body) return S.busy ? html`${skeleton('head')}${skeleton('body')}` : notice();
    return r.view==='plan' ? planView() : buildView();
  }
  /* ---- the owner's operations ---- */
  async function plan() {
    if(S.busy||!S.controls||!S.spec||!S.key.startsWith('compose:'))return;
    const ticket=S.generation;
    const doc={...clone(S.controls.template),reason:S.reason,edits:[{operation:S.operation,
      factor_id:S.operation==='CREATE'?S.spec.factor_id:S.selected,specification:S.operation==='RETIRE'?null:clone(S.spec)}]};
    S.busy=true;S.error='';S.errorBody=null;paint();
    try {
      const p=await Data.post('/api/features/plan',{feature_document:doc});
      if(!current(ticket)){notify(t('The definition PLAN was saved while you were elsewhere; it did not open over this page.'));return;}
      S.plan=p;S.busy=false;
      navigate('features',{feature_plan:p.plan_hash,input_binding:'',feature_parent:'',feature_build:''}); // the saved PLAN is its own place
    } catch(e) { if(current(ticket)) fail(e); else notify(t('The definition PLAN asked for earlier did not complete while you were elsewhere.')); }
    finally { if(current(ticket)) { S.busy=false; paint(); } }
  }
  async function build() {
    if(S.busy||!S.plan||!S.key.startsWith('plan:'))return;
    const ticket=S.generation;S.busy=true;S.error='';S.errorBody=null;paint();
    try {
      const b=await Data.post('/api/features/build',{feature_plan_hash:S.plan.plan_hash,feature_output:S.plan.next_requests.build.feature_output});
      const task=b.task_id||b.publication_task_id;
      if(!current(ticket)){notify(task ? t('The build Task was admitted while you were elsewhere; it is on Tasks.') : t('The build asked for earlier did not complete while you were elsewhere.'));return;}
      if(!task)throw Error(b.failure_code||b.status);
      S.busy=false;paint();await LiveTasks.open(task);
    } catch(e) { if(current(ticket)) { fail(e); S.busy=false; paint(); } else notify(t('The build asked for earlier did not complete while you were elsewhere.')); }
  }
  /* The saved request back in the editor, over the same input and revision (the owner's parent). */
  function editRequest() {
    if(S.busy||!S.plan)return;
    const request=S.plan.document,entry=request.edits[0];
    if(request.edits.length!==1)throw Error('This saved request contains multiple edits; export its YAML to keep all edits.');
    const parent=request.parent_plan_hash || '';
    S.drafts.set(`compose:${S.plan.input_binding_hash}:${parent}`,{operation:entry.operation,selected:entry.factor_id,reason:request.reason,spec:entry.specification?clone(entry.specification):null});
    navigate('features',{input_binding:S.plan.input_binding_hash,feature_parent:parent,feature_plan:'',feature_build:''});
  }
  function revise() {
    if(S.busy||!S.plan)return;
    S.drafts.delete(`compose:${S.plan.input_binding_hash}:${S.plan.plan_hash}`);
    navigate('features',{input_binding:S.plan.input_binding_hash,feature_parent:S.plan.plan_hash,feature_plan:'',feature_build:''});
  }
  /* The Task's result: the build it read is this page's body, at the build's address. */
  function showBuild(body) {
    S.build=body;
    navigate('features',{feature_build:body.task_id,feature_plan:'',input_binding:'',feature_parent:''});
  }
  const open=(binding)=>navigate('features',{input_binding:binding || '',feature_parent:'',feature_plan:'',feature_build:''});
  const openPlan=(hash)=>navigate('features',{feature_plan:hash,input_binding:'',feature_parent:'',feature_build:''});
  // the way up from a saved PLAN or a build is the composer over the same input
  const routeContext=()=>{ const binding=S.plan?.input_binding_hash || S.controls?.input_binding_hash || ''; return binding ? {input_binding:binding} : {}; };
  return {page,leave,open,openPlan,select,operation,edit,plan,build,editRequest,revise,showBuild,routeContext,
    exportYaml:()=>{if(S.plan?.yaml)download(S.plan.yaml,'AlphaLattice-feature-change.yaml','text/yaml');}};
})();
PAGES.features = LiveFeatures.page;
