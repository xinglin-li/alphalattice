/* Saved research results and explicit handoffs, using existing application operations.
 * Reading order on a saved study: what it is (method, input, interval, state) and its recorded
 * limits first; then decision-relevant evidence and the continuation actions; exact identity,
 * declaration JSON and exports stay reachable inside disclosures. */
const LiveStudy = (() => {
  const kinds={factor:'factor.screening-development',alpha:'alpha.model-development',risk:'risk.covariance-development','alpha-compare':'alpha.model-development'};
  const alphaPage=(page=app.page)=>page==='alpha' || page==='alpha-compare'; // N2: the comparison reads its first study as the Alpha page does
  const pages=new Set([...Object.keys(kinds),'foundation']);
  const S={task:null,body:null,status:'empty',error:'',busy:'',revision:0,readPage:'',writeTicket:null,verificationError:'',
    curation:null,roles:{},reasons:{},limits:new Set(),decision:'',target:'',factorQuery:'',
    preview:null,admission:null,foundations:null,foundationsError:'',foundationLabel:null,pending:null,riskOffset:0,sealed:null,
    alphaComparison:{leftCandidate:'',rightTask:'',rightBody:null,rightPage:'',rightCandidate:'',result:null,error:'',busy:'',revision:0}};
  const ready=()=>S.status==='ready'&&S.body?.status==='EXPERIMENT_PUBLISHED'&&!S.busy;
  // The listing names target/model fields only for an alpha_source plan; a lifecycle
  // shares the kind but has no candidate-fold graph. Strict readback still decides the pair.
  const comparisonListed=v=>v.kind===kinds.alpha && v.lifecycle==='SUCCEEDED' && Boolean(v.target_recipe_id && v.model_parameters);
  const comparisonReady=b=>b?.status==='EXPERIMENT_PUBLISHED' && b.program?.kind===kinds.alpha && Boolean(b.alpha_source) && !b.lifecycle_research && !b.alpha_qualification;
  const ordinaryPage=b=>Object.keys(kinds).find(k=>k!=='alpha-compare' && kinds[k]===b?.program?.kind);
  function comparisonPairReady() {
    const c=S.alphaComparison, selected=(b,id)=>comparisonReady(b) && (b.result?.candidates || []).some(v=>v.status==='DEVELOPMENT_EVALUATED' && v.candidate_id===id);
    return ready() && !c.busy && Boolean(c.rightTask) && selected(S.body,c.leftCandidate) && selected(c.rightBody,c.rightCandidate);
  }
  const COMPARISON_UNSUPPORTED='workbench.alpha_comparison_unsupported_study';
  const value=v=>v && typeof v==='object' && Object.hasOwn(v,'availability') ? v.value ?? v.availability : v ?? '';
  const text=v=>{const x=value(v);return typeof x==='object'?JSON.stringify(x):String(x);};
  const pairs=(v)=>Object.entries(v || {}).map(([k,x])=>[html`<span data-tip="${k}">${codeWords(k)}</span>`,text(x)]); // the key worded, the code in its title (round 86)
  /* A displayed statistic: four decimals for reading; a non-zero value that four decimals would
   * print as zero is shown in exponent form instead, never as 0.0000; the exports and each item's
   * evidence carry the owner's exact value. (`stat` is the card component; this formats a number.) */
  const figure=v=>{const x=value(v);if(typeof x!=='number' || !Number.isFinite(x))return text(v);return x!==0 && Math.abs(x)<0.00005 ? x.toExponential(2).replace(/^-/,'−') : fmt(x,4);};
  /* The owners' typed refusals of the decision and Foundation chain, in words a reader can act on. */
  const CODES={
    [COMPARISON_UNSUPPORTED]:'This study has no verified candidate-fold evaluation for comparison. Open its ordinary study readback.',
    'research_foundation.preview_required':'No Foundation preview is retained for this admission (it was never previewed here, or the service restarted); preview it again, then seal.',
    'research_foundation.preview_stale':'The Foundation candidate changed between the preview and the seal (the decision, the input version or the evidence moved); preview it again.',
    'research_foundation.source_changed':'The recorded Foundation no longer matches what its sources derive to; its source study, decision or input changed beneath it.',
    'research_foundation.handoff_source_mismatch':'The named Foundation belongs to another Factor decision or input version than the ones requested.',
    'factor_research.completed_experiment_required':'The source study has not completed and published; no decision, Foundation or draft can start from it yet.',
    'factor_research.curation_receipt_mismatch':'The evidence receipt changed since these choices were read; reload the study and decide again.',
    'factor_research.curation_role_not_admissible':'A chosen role is not one the decision owner admits for that factor; the admissible roles are listed beside it.',
    'factor_research.curation_limitations_not_acknowledged':'Every recorded limitation must be acknowledged before a decision is saved.',
    'research_experiment.factor_parent_required':'Only a saved Factor study can carry a decision or a Foundation.',
    'risk_report.sessions_outside_portfolio':'The Risk study\'s formations lie outside this book\'s report interval; a report-only link needs formations within the book\'s dates. Choose a Risk study over those dates, or a book that covers this window.',
    'risk_report.not_compatible':'The owner found this Risk study incompatible with the book (input or formation axis); nothing was attached.',
    'alpha_research.saved_comparison_requires_two_tasks':'Both sides name the same saved study; a comparison needs two different saved Alpha studies.',
    'alpha_research.saved_comparison_alpha_tasks_required':'One selected Task is not saved Alpha evidence.',
    'alpha_research.saved_comparison_candidate_unavailable':'The named candidate is not among that study\'s recorded candidates.',
    'alpha_research.saved_comparison_candidate_incomplete':'The named candidate did not complete its evaluation; only an evaluated candidate can be compared.',
    'alpha_research.saved_comparison_receipt_unavailable':'One study\'s development receipt could not be read from its recorded artifacts.',
    'alpha_research.saved_comparison_receipt_mismatch':'One study\'s recorded receipt does not match its published evidence; nothing is compared against a moved artifact.',
    'alpha_research.saved_comparison_fold_set_incomplete':'One candidate lacks a recorded result for every fold; the pair cannot be read fold by fold.',
    'alpha_research.saved_comparison_not_alpha':'One recorded receipt is not Alpha development evidence.',
    'alpha_research.saved_comparison_input_binding_hash_mismatch':'The two studies were fitted on different exact inputs; the owner compares only candidates of the same input version.',
    'alpha_research.saved_comparison_target_recipe_binding_hash_mismatch':'The two studies predict different targets; the owner compares only candidates of the same target recipe.',
    'alpha_research.saved_comparison_target_materialization_binding_hash_mismatch':'The two studies\' targets were materialized differently; the owner compares only candidates of the same materialization.',
    'alpha_research.saved_comparison_ordered_feature_ids_mismatch':'The two studies use different feature axes; the owner compares only candidates on the same ordered features.',
    'alpha_research.saved_comparison_metric_policy_hash_mismatch':'The two studies were scored under different metric policies; their numbers are not in the same units.',
    'alpha_research.saved_comparison_split_policy_hash_mismatch':'The two studies were evaluated on different causal fold policies; their folds do not pair.',
    'alpha_research.saved_comparison_fold_commitment_mismatch':'The two candidates\' fold commitments differ; the owner pairs only identical fold commitments.',
    'alpha_research.saved_comparison_fold_commitment_duplicate':'A candidate records the same fold commitment twice; the pair cannot be read fold by fold.',
    'alpha_research.saved_comparison_score_support_mismatch':'The two candidates score different rows of the common surface; nothing is intersected or zero-filled to force a comparison.'};
  const explain=(message)=>{const code=String(message || '').split(':')[0].trim();if(CODES[code])return t(CODES[code]);const inner=Object.keys(CODES).find(k=>String(message || '').includes(k));return inner ? t(CODES[inner]) : '';};
  // Human-facing fold labels count from 1; the owner's zero-based fold_index is kept in every
  // selector, route, export and detail and is named beside the label where a reader may need it.
  const foldLabel=(index)=>Number.isInteger(index) ? t('Fold {n}',{n:index+1}) : t('Fold');
  const invalidate=()=>{S.revision++;S.preview=null;S.pending=null;S.foundationLabel=null;named=null;S.error='';S.errorBody=null;if(S.busy!=='write')S.busy='';};
  const alphaRoute=task=>{
    const h=hashParams();
    return h.get('alpha_left_task')===task?{
      leftCandidate:h.get('alpha_left_candidate') || '',rightTask:h.get('alpha_right_task') || '',rightCandidate:h.get('alpha_right_candidate') || ''
    }:null;
  };
  const sameAlphaSelection=(c,s)=>Boolean(s)&&['leftCandidate','rightTask','rightCandidate'].every(k=>c[k]===s[k]);
  function persistAlphaSelection() {
    const c=S.alphaComparison;
    replaceHash({alpha_left_task:S.task,alpha_left_candidate:c.leftCandidate,alpha_right_task:c.rightTask,alpha_right_candidate:c.rightCandidate});
  }
  function routeContext(page=app.page) {
    if(page==='foundation')return S.admission?{foundation:S.admission.admission_hash}:{};
    if(S.body?.program?.kind!==kinds[page])return {};
    if(!alphaPage(page))return {study:S.task};
    const c=S.alphaComparison;
    if(!c.leftCandidate&&!c.rightTask&&!c.rightCandidate)return {study:S.task};
    return {study:S.task,alpha_left_task:S.task,alpha_left_candidate:c.leftCandidate,alpha_right_task:c.rightTask,alpha_right_candidate:c.rightCandidate};
  }
  /* The saved-experiment listing is shared with History through Data; this only refreshes it. */
  async function catalog() {
    if(S.busy)return;
    S.busy='catalog';const ticket=S.revision;render();
    try{await Data.refreshExperiments();}
    finally{if(ticket===S.revision){S.busy='';render();}}
  }
  /* Opening navigates at once (page and route), then reads. The answer updates this module's
   * own state whenever it is still the newest open; it moves the page or the route only while
   * this study is still the view being shown -- a reader who left for another page meanwhile,
   * or whose route no longer names this study, is not brought back by a late answer. */
  const shown=(task)=>pages.has(app.page)&&hashParams().get('study')===task;
  async function open(task,page=null,readback=null) {
    // The Task opener may already have needed the strict read to discover its kind.
    // Reuse only that exact published Task/body binding, never a listing or summary.
    const accepted=readback?.task_id===task && readback.status==='EXPERIMENT_PUBLISHED' && readback.program?.kind===kinds[page] ? readback : null;
    closeDialog();invalidate();const ticket=S.revision;S.task=task;S.body=null;S.status='loading';
    S.error='';S.verificationError='';S.busy='read';S.curation=null;S.roles={};S.reasons={};S.limits=new Set();S.decision='';S.riskOffset=0;
    S.alphaComparison={leftCandidate:'',rightTask:'',rightBody:null,rightPage:'',rightCandidate:'',result:null,error:'',busy:'',revision:0};
    if(page)app.page=page;S.readPage=app.page;
    const keepAlphaSelection=alphaPage()&&hashParams().get('alpha_left_task')===task;
    replaceHash({page:app.page,study:task,foundation:'',...(keepAlphaSelection?{}:{alpha_left_task:'',alpha_left_candidate:'',alpha_right_task:'',alpha_right_candidate:''})});render();
    try{
      if(app.page==='alpha' && !accepted) {
        try {
          const summary=await Data.read('/api/experiments/summary?'+new URLSearchParams({task_id:task}));
          if(ticket!==S.revision)return;
          if(summary.status==='EXPERIMENT_SUMMARY') {S.body=summary;S.status='ready';render();}
        } catch(error) {
          if(error?.body?.failure_code!=='research_experiment.summary_kind_not_installed')throw error;
        }
      }
      const body=accepted || await Data.read('/api/experiments/readback?'+new URLSearchParams({task_id:task}));
      if(ticket!==S.revision)return;
      S.body=body;S.status='ready';
      const actual=Object.keys(kinds).find(k=>kinds[k]===body.program?.kind);
      if(actual && shown(task) && app.page!=='alpha-compare' && kinds[app.page]!==body.program?.kind){app.page=actual;replaceHash({page:actual,study:task});}
      if(actual==='alpha')restoreAlphaSelection(task);
      if(body.program?.kind===kinds.factor && body.status==='EXPERIMENT_PUBLISHED'){
        S.target=JSON.stringify([body.research_input_id,body.input_binding_hash]);
        const choices=await Data.read('/api/experiments/curation?'+new URLSearchParams({task_id:task}));
        if(ticket===S.revision)S.curation=choices;
      }
    }catch(e){if(ticket===S.revision){if(S.body?.status==='EXPERIMENT_SUMMARY'){S.verificationError=e.message;S.status='ready';}else{S.error=e.message;S.errorBody=e.body && typeof e.body==='object' ? e.body : null;S.status='error';}}}
    finally{if(ticket===S.revision){S.busy='';render();}}
  }
  async function foundations(hash=null) {
    if(S.busy)return;
    invalidate();const ticket=S.revision;S.busy='foundation';S.error='';if(!hash)S.foundationsError='';
    if(hash){S.admission=null;app.page='foundation';replaceHash({page:'foundation',foundation:hash,study:''});}
    S.readPage=app.page;render();
    try{
      const b=await Data.read(hash?'/api/experiments/foundations/readback?'+new URLSearchParams({foundation_admission_hash:hash}):'/api/experiments/foundations');
      if(ticket!==S.revision)return;
      if(hash)S.admission=b.admission;else S.foundations=b.foundations || [];
    }catch(e){if(ticket===S.revision){S.error=e.message;S.errorBody=e.body && typeof e.body==='object' ? e.body : null;if(!hash){S.foundations=null;S.foundationsError=e.message;}}} // a failed list stays unread, never an empty one
    finally{if(ticket===S.revision){S.busy='';render();}}
  }
  function ensure() {
    if(!pages.has(app.page)||Data.workspaceStatus!=='ready'||S.busy)return;
    const h=hashParams();
    if(app.page==='foundation'){
      const id=h.get('foundation');
      if(id && S.admission?.admission_hash!==id && !S.error)foundations(id);
      else if(!id && S.admission){S.admission=null;render();} // the address went up to the list: the reader forgets its admission, and no link carries it back
      else if(!id && !S.foundations && !S.foundationsError)foundations(); // a failed list waits for Refresh, not for the next paint
      return;
    }
    const id=addressed();
    if(id && (S.task!==id || S.status==='empty'))open(id,app.page);
    else if(alphaPage()&&S.task===id&&S.body?.program?.kind===kinds.alpha)restoreAlphaSelection(id);
    else if(!id && S.task)leave(); // the address went up to the list: the reader forgets its object
    else if(!id && Data.experiments()===null)catalog();
  }
  /* The study the address names (an alpha comparison names its left member). The page shows what
   * its address says: no study in it is the list, whatever the reader held (2026-09-21, the user's
   * reading: the crumb's way up re-opened the study it was meant to leave). */
  const addressed=()=>{const h=hashParams();return alphaPage()?(h.get('alpha_left_task') || h.get('study')):h.get('study');};
  function leave() {
    invalidate();S.task=null;S.body=null;S.status='empty';S.curation=null;S.roles={};S.reasons={};S.limits=new Set();S.decision='';S.riskOffset=0;
    S.alphaComparison={leftCandidate:'',rightTask:'',rightBody:null,rightPage:'',rightCandidate:'',result:null,error:'',busy:'',revision:0};
    render();
  }
  /* ---- shared reading helpers ---- */
  const facts=()=>LiveViews.studyFacts(LiveViews.declaredFromReadback(S.body));
  const cutoff=(binding)=>LiveViews.cutoffText(LiveViews.inputState(binding));
  const studyLink=(page,id,label,extra={})=>link(html`${label} ${icon('arrow')}`,page,'button compact',{study:id,...extra});
  const originRow=(origin,page)=>[t('Continued from'),origin?html`${mono(origin,SHORT.id)} ${studyLink(page,origin,t('Open origin study'))}`:t('Not continued · authored directly')];
  function statisticalInterval(b) {
    const p=b.execution_preview;
    if(p?.statistical_start)return `${p.statistical_start} — ${p.statistical_end}`;
    const f=b.evidence?.formation_sessions;
    return f?.length?`${f[0]} — ${f.at(-1)}`: '';
  }
  function list() {
    const compare=app.page==='alpha-compare';
    const rows=(Data.experiments() || []).filter(v=>compare ? comparisonListed(v) : v.kind===kinds[app.page]);
    const refused=Data.experimentRefusals() || [];
    // One row per saved experiment (round 13): the state dot, the declared label, the reference,
    // the interval, the input and its cutoff; the row opens the exact object. The state is the
    // listing's, overlaid by Task Control's newer report (Data.lifecycleOf).
    // the list's one verb (law 129; the user, 2026-09-24: 现在没地方创建new experiments): a new experiment of this kind -- the empty list's own way says it when nothing is saved
    const verb=!compare && rows.length ? link(html`${icon('plus')}${t('New experiment')}`,'lab','button primary',{experiment_kind:kinds[app.page]}) : '';
    const refusals=refused.length ? panel(t('Unreadable saved study plans'),t('The recorded plans could not be read; their kind is unknown.'),html`<div class="card-list">${refused.map((item)=>html`<div>
      <h3>${t('Task')} ${hashCell(item.task_id,SHORT.id)}</h3>
      ${refusal({failure_code:item.failure_code,reason:item.detail},TONE.attention,{state:'refused',word:t('Refused'),catalog:true,next:html`<span class="flow">${btn(t('Inspect task'),'task',item.task_id,'text-btn')} ${link(t('Research inputs'),'inputs','text-btn')}</span>`})}
    </div>`)}</div>`) : '';
    return html`${objectHead(t(ROUTES[app.page][1]),`${t(compare ? 'Choose the first saved Alpha study; a candidate of it is compared with a candidate of another.' : 'Choose an exact saved experiment. No latest result is inferred.')} ${t('Labels summarize declared parameters, never results or a winner.')}`,verb,'',[])}${Data.experimentsError?notRead(t('Saved experiments unavailable'),Data.experimentsError):''}${refusals}${rows.length ? LiveViews.studyLobby(alphaPage() ? 'alpha' : app.page,rows.map(v=>{const f=LiveViews.studyFacts(v),h=Data.history().find(x=>x.task_id===v.task_id);return {key:v.task_id,state:Data.lifecycleOf(v.task_id,v.lifecycle) || 'metadata',name:f.wordsMarkup || f.words || f.summary || t(ROUTES[app.page][1]),words:f.words || f.summary || '',ref:short(v.task_id,SHORT.id),interval:html`${f.interval || ''}${v.research_lane==='EXPLORATION' ? html`${f.interval ? ' · ' : ''}${t('on a sample')}` : ''}`,input:v.input_id ? html`${v.input_id} · ${cutoff(v.input_binding_hash)}` : '',inputKey:v.input_binding_hash,inputText:v.input_id,at:h?.raw?.recorded_at || '',to:{action:'study-open',value:v.task_id}};}).reverse()) : compare && !refused.length ? emptyState(t('No completed candidate study to compare.'),link(t('Alpha modeling'),'alpha','button primary',{study:'',alpha_left_task:'',alpha_left_candidate:'',alpha_right_task:'',alpha_right_candidate:''}),'page-empty') : refused.length ? '' : emptyState(t('No experiment yet'),link(t('New experiment'),'lab','button primary',{experiment_kind:kinds[app.page]}),'page-empty')}`;
  }
  /* First screen: the sources as a row of chips (each the way to its object), what the study
   * admits next in one sentence, the owner's recorded limitations as pills; the four named facts
   * and the exact references stay in key/value rows under a disclosure (they wrap machine
   * identifiers; stat cards would not). */
  const LEDES={factor:'The evidence each declared factor left on one exact input, and the curation decisions it admits.',
    alpha:'The candidates fitted on one exact input, read fold by fold; development evidence, never a live model.',
    risk:'The estimator\'s diagnostics over dated formations of one exact input; development evidence, not an admitted Risk publication.'};
  const NEXT={alpha:'Choose one candidate below to create a Portfolio draft, or continue this declaration as a new draft. Neither runs research.',
    factor:'Use a saved curation decision below to preview a Foundation or an Alpha draft, or continue this declaration as a new draft.',
    risk:'Attach this diagnostic to a saved Portfolio book as a report-only association (the book\'s weights are unchanged), or continue this declaration as a new draft.'};
  function limitations() {
    const l=S.body?.limitations || [];
    return l.length?html`<div class="card-list">${l.map((code)=>objectRow({lead: statusDot('metadata',t('Recorded limitation')), name: coded(code), cls: 'limit-row'}))}</div>`:'';
  }
  /* The study's facts (round 64): the sources as one kv (each value the way to its object), the
   * recorded limitations, the declared facts, the lineage and the declaration as JSON — sections
   * of the inspector's Facts mode, opened from the head's `Facts` chip and `···`; the page keeps
   * its figures. What the study admits next is the first section's line. */
  const hasFacts=()=>Boolean(S.body) && S.body.status==='EXPERIMENT_PUBLISHED' && S.body.program?.kind===kinds[app.page];
  function factsSections() {
    const b=S.body;
    if(!hasFacts()) return [];
    const rows=[];
    if(app.page==='alpha'){
      const a=b.alpha_source || {};
      rows.push({icon:'lab',label:t('Factor study'),value:a.factor_task_id ? html`<span class="mono">${mono(a.factor_task_id,SHORT.id)}</span>` : t('none recorded'),to:a.factor_task_id ? {page:'factor',extra:{study:a.factor_task_id}} : null});
      rows.push({icon:'check',label:t('Curation decision'),value:t(a.curation_receipt_hash ? 'recorded' : 'none recorded')});
      rows.push({icon:'archive',label:t('Foundation'),value:a.foundation_admission_hash ? html`<span class="mono">${mono(a.foundation_admission_hash,SHORT.hash)}</span>` : t('none recorded'),to:a.foundation_admission_hash ? {page:'foundation',extra:{foundation:a.foundation_admission_hash}} : null});
    } else rows.push({icon:'branch',label:t('Upstream'),value:t('the sealed input itself')});
    rows.push({icon:'cube',label:t('Exact input'),value:html`${b.research_input_id || ''} · ${cutoff(b.input_binding_hash)}`});
    rows.push({icon:'history',label:t('Continued from'),value:b.origin_task_id ? html`<span class="mono">${mono(b.origin_task_id,SHORT.id)}</span>` : t('authored directly'),to:b.origin_task_id ? {page:app.page,extra:{study:b.origin_task_id}} : null});
    if(typeof LiveViews!=='undefined' && LiveViews.collaborationRows) rows.push(...LiveViews.collaborationRows(S.task));
    const limits=limitations();
    return [{title:t('Sources'),body:html`${kv(sourceRows(rows))}<p class="caption">${t(NEXT[app.page])}</p>`},
      limits ? {title:t('Recorded limitations'),body:limits} : null,
      {title:t('This study'),body:summary()},{title:t('Sources and continuation'),body:lineage()},
      {title:t('Documents'),body:codeRef(t('Read the exact declaration (JSON)'), b.document)}].filter(Boolean);
  }
  /* A study's Properties (N2 of the open-issues plan, law 126 amended): what it declared and where it comes
   * from -- the method, the continuation, an Alpha study's Factor study and Foundation -- a section after its
   * content, the facts in the lane's columns, each value whole; the head's line and Facts hold the rest. */
  // a source is named as its own list names it, and the name is the way to it (law 134)
  const studyName=(id)=>{const x=Data.history().find(r=>r.task_id===id);return x ? (x.words || x.summary || t(x.name)) : html`<span class="mono">${LiveViews.shortRef(id)}</span>`;};
  // The Foundation page's standing remains the owner's full list reading.
  let named=null;
  function nameFoundations() {
    if(named?.revision===S.revision)return;
    const reading={revision:S.revision,rows:[]};named=reading;
    Data.read('/api/experiments/foundations').then(b=>{
      if(named!==reading || reading.revision!==S.revision || app.page!=='foundation')return;
      reading.rows=b.foundations || [];render();
    },error=>{if(named===reading && error?.name==='AbortError')named=null;});
  }
  function readFoundationLabel(hash) {
    const reading={hash,revision:S.revision,ids:null};S.foundationLabel=reading;
    Data.read('/api/experiments/foundations/summary?'+new URLSearchParams({foundation_admission_hash:hash})).then(b=>{
      if(S.foundationLabel!==reading || reading.revision!==S.revision || app.page!=='alpha')return;
      if(b.foundation_admission_hash===hash && Array.isArray(b.ordered_factor_ids))reading.ids=b.ordered_factor_ids;
      render();
    },()=>{});
    return reading;
  }
  const foundationName=(hash)=>{const reading=S.foundationLabel?.hash===hash && S.foundationLabel.revision===S.revision ? S.foundationLabel : readFoundationLabel(hash);return reading.ids ? foundationFactorsTitle(reading.ids) : html`<span class="mono">${LiveViews.shortRef(hash)}</span>`;};
  // U37: how this read was proved -- in full, or the files' identity unchanged since the full check (the owner's words on hover)
  const verificationWords=(basis)=>basis==='FULL' ? t('in full: every sealed file read') : String(basis).startsWith('FILES_UNCHANGED') ? hint(t('files unchanged since the full check'),String(basis).replace(/^FILES_UNCHANGED\s*/,'')) : codeWords(basis);
  function propertiesSection(lead=[]) {
    const b=S.body,a=b.alpha_source || {};
    const rows=[...lead];
    if(app.page==='factor'){
      rows.push([t('Recorded work'),html`${countText(b.execution_numerical_call_count ?? '','{n} numerical call','{n} numerical calls')}${infoMark(t('by the completing execution; this page estimates nothing'))}`]);
      rows.push([t('Current readback'),html`${countText(b.numerical_call_count ?? '','{n} numerical call','{n} numerical calls')}${infoMark(t('Strict verification of this saved result; no new Factor execution'))}`]);
      if(b.execution_evidence_hash)rows.push([t('Original execution evidence'),hashCell(b.execution_evidence_hash,SHORT.hash)]);
    }
    if(app.page==='alpha'){
      if(a.factor_task_id) rows.push([t('Factor study'),link(studyName(a.factor_task_id),'factor','text-btn',{study:a.factor_task_id})]);
      rows.push([t('Foundation'),a.foundation_admission_hash ? link(foundationName(a.foundation_admission_hash),'foundation','text-btn',{foundation:a.foundation_admission_hash}) : t('none · the development input')]);
    }
    rows.push([t('Continued from'),b.origin_task_id ? link(studyName(b.origin_task_id),app.page,'text-btn',{study:b.origin_task_id}) : t('authored directly')]);
    if(b.verification_basis) rows.push([t('Verified'),verificationWords(b.verification_basis)]);
    if(b.realization) rows.push([t('Computed'),LiveViews.realizationWords(b.realization)]);
    const l=b.limitations || [];
    if(l.length) rows.push([t('Recorded limitations'),html`${count(l.length)}${infoMark(l.map(code=>codeWords(code)).join(' · '))}`]);
    return panel(t('Properties'),'',kv(rows,'kv-columns'));
  }
  function summary() {
    const b=S.body,d=b.document,s=d.experiment?.sessions || {},recorded=Data.history().find(x=>x.task_id===S.task);
    const method=app.page==='alpha'?html`${[methodWords(d.alpha?.model_parameters?.family),methodWords(d.alpha?.target_recipe_id)].filter(Boolean).join(' · ') || ''} <span class="sub-cell">${countText(d.alpha?.ordered_feature_ids?.length ?? '', '{n} feature', '{n} features')}</span>`
      :app.page==='risk'?html`${d.risk?.estimator?.capability ? methodWords(d.risk.estimator.capability) : ''}${infoMark(t('Estimator parameters below'))}`
      :html`${t('Factor screening')} <span class="sub-cell">${countText(d.factor?.factor_ids?.length ?? 0, '{n} factor declared', '{n} factors declared')}</span>`;
    return html`<p class="caption">${t('Declared facts of the saved result; not a performance claim.')}</p>${kv([[t('Method'),method],[t('Source input'),html`${b.research_input_id || ''} <span class="sub-cell">${cutoff(b.input_binding_hash)}</span>`],[t('Declared interval'),html`${s.start&&s.end?`${s.start} — ${s.end}`: ''} <span class="sub-cell">${s.as_of?.session?t('as-of')+' '+s.as_of.session:t('as-of unknown')}</span>`],[t('Recorded'),html`${recorded?.recordedAt || ''}${infoMark(t('Published development evidence; not current authority'))}`]])}`;
  }
  /* Only references recorded in this result are linked. An absent reference is stated, never
   * replaced by a latest lookup. */
  function lineage() {
    const b=S.body,rows=[];
    if(app.page==='alpha'){
      const a=b.alpha_source || {};
      rows.push([t('Factor study'),a.factor_task_id?html`${mono(a.factor_task_id,SHORT.id)} ${studyLink('factor',a.factor_task_id,t('Open Factor study'))}`:t('No Factor reference recorded')]);
      rows.push([t('Curation decision'),a.curation_receipt_hash?mono(a.curation_receipt_hash,SHORT.hash):t('No curation reference recorded')]);
      rows.push([t('Foundation admission'),a.foundation_admission_hash?html`${mono(a.foundation_admission_hash,SHORT.hash)} ${link(html`${t('Open Foundation')} ${icon('arrow')}`,'foundation','button compact',{foundation:a.foundation_admission_hash})}`:t('None recorded · drafted from the development input, not a sealed Foundation')]);
    } else rows.push([t('Upstream study'),t('None · this method starts from the sealed input')]);
    rows.push([t('Exact input'),html`${b.research_input_id || ''} · ${cutoff(b.input_binding_hash)} · ${mono(b.input_binding_hash)}`]);
    rows.push(originRow(b.origin_task_id,app.page));
    return html`<p class="caption">${t('Recorded references only; nothing is looked up as the latest result.')}</p>${kv(rows)}`;
  }
  function provenance() {
    const b=S.body;if(!b)return '';
    return html`${kv([[t('Task'),html`<span class="mono">${S.task}</span>`],[t('Receipt'),mono(b.receipt?.receipt_hash,Infinity)],[t('Program'),mono(b.program?.program_hash,Infinity)],[t('Input'),b.research_input_id || ''],[t('Input binding'),mono(b.input_binding_hash,Infinity)],[t('Declared interval'),JSON.stringify(b.document?.experiment?.sessions || {})],[t('Statistical interval'),statisticalInterval(b)]])}<p class="caption">${t('Readback is not a new execution or prospective validation.')}</p><ul>${(b.limitations || []).map(x=>html`<li>${x}</li>`)}</ul>`;
  }
  function factorRows() {
    const selected=new Set(S.body.document.factor.factor_ids), items=S.body.result.evidence_report.items;
    return items.filter(v=>selected.has(v.factor_id)&&(!S.factorQuery||v.factor_id.includes(S.factorQuery))).map(v=>tr([
      btnAttrs(html`<span class="mono">${v.factor_id}</span>${icon('arrow')}`,'study-factor-detail',v.factor_id,'text-btn cell-opener','data-row-press'),coded(v.classification),figure(v.mean_oriented_rank_ic),figure(v.rank_ic_by_q_value),figure(v.validation_pair_coverage_mean),Number.isFinite(Number(v.validation_period_count)) ? count(v.validation_period_count) : text(v.validation_period_count)
    ])); // N6 (Stripe's list): the factor's name opens its evidence; no trailing link column
  }
  const FACTOR_HEADERS=()=>[{label:t('Factor'),type:'id'},{label:t('Classification'),type:'text'},{label:t('Rank IC (mean, oriented)'),type:'num'},{label:t('BY q-value'),type:'num'},{label:t('Pair coverage (fraction)'),type:'num'},{label:t('Validation periods'),type:'num'}];
  const factorNote=()=>t('{n} of {m} hypotheses shown: the factors this study declared, each read against the full denominator. Values to four decimals (tiny non-zero values in exponent form); the exact values are in each factor\'s evidence and in the JSON export.',{n:S.body.document.factor.factor_ids.length,m:S.body.result.evidence_report.hypothesis_count});
  function factorDetail(id) {
    const item=S.body?.result?.evidence_report?.items.find(v=>v.factor_id===id);
    if(!item)return;
    const choice=S.curation?.choices.find(v=>v.factor_id===id);
    const exact=(v)=>typeof v==='number' ? String(v) : text(v);
    // N2 (law 131): a row's detail opens in the inspector beside its table, never a dialog
    Window.openInspector({mode:'detail',readHeader:()=>({title:id,kind:t('Factor · recorded evidence')}),title:id,kind:t('Factor · recorded evidence'),by:['study-factor-detail',id],body:html`<section class="inspector-section">${kv([[t('Classification'),coded(item.classification)],[t('Reasons'),(item.reason_codes || []).map(codeWords).join(' · ') || ''],[t('Admissible roles'),choice?.roles.join(', ') || t('none · the decision owner admits no role from this evidence')],[t('Redundancy group'),choice?.cluster_id || ''],[t('Rank IC (mean, oriented) · exact'),exact(item.mean_oriented_rank_ic)],[t('Rank IC t-statistic (HAC)'),exact(item.rank_ic_hac_t_stat)],[t('Raw p-value'),exact(item.rank_ic_raw_p_value)],[t('BY q-value · exact'),exact(item.rank_ic_by_q_value)],[t('Validation periods'),exact(item.validation_period_count)],[t('Pair coverage (fraction) · exact'),exact(item.validation_pair_coverage_mean)]])}<h3>${t('Exact item evidence')}</h3><pre class="code-block code-document">${json(item)}</pre></section>`});
  }
  const curationCount=()=>{const c=S.curation;return t('{a} of {n} have an admissible role; {c} chosen so far. The recorded limitations below must each be acknowledged. Nothing is preselected.',{a:c.choices.filter(v=>v.roles.length).length,n:countText(c.choices.length,'{n} declared factor','{n} declared factors'),c:Object.values(S.roles).filter(Boolean).length});};
  function curationEditor() {
    const c=S.curation;if(!c)return '';
    const chosen=Object.values(S.roles).filter(Boolean).length;
    // the handoff binds a published version of the study's input: with none available here its verbs are held by that reason
    const handoffChoices=Data.inputs().filter(v=>v.available&&c.inputs.includes(v.id));
    const handoffOff=!ready() || !S.decision ? t('Select a saved decision') : !handoffChoices.some(v=>JSON.stringify([v.id,v.binding_hash])===S.target) ? t('No published version of this study\'s input is available here') : '';
    const choice=(v,i)=>html`<details class="reveal-details"><summary><span class="mono">${v.factor_id}</span> · ${codeWords(v.classification)}${v.roles.length ? html` · ${t('admissible: {roles}',{roles:v.roles.join(', ')})}` : html` · ${t('no admissible role')}`}${S.roles[v.factor_id] ? html` · <strong>${S.roles[v.factor_id]}</strong>` : ''}</summary>${v.roles.length ? html`<div class="field"><label for="studyRole${i}">${t('Admissible role')}</label>${picker('studyRole'+i,[['',t('Do not select')],...v.roles.map(r=>[r,codeWords(r)])],{selected:S.roles[v.factor_id] || '',disabled:S.busy==='write',attrs:html`data-study-role="${v.factor_id}"`,label:t('Role')})}<small>${t('Roles the decision owner admits from this factor\'s evidence classification; no other role can be saved.')}</small><label class="sr-only" for="studyReason${i}">${t('Rationale')}</label><textarea id="studyReason${i}" class="ui-field" data-study-reason="${v.factor_id}"${S.busy==='write'?' disabled':''} maxlength="500" placeholder="${t('Reason for this choice')}">${S.reasons[v.factor_id] || ''}</textarea><small>${t('A rationale is required for every selected factor; it is saved with the decision.')}</small></div>` : html`<p class="caption">${t('No admissible role from its evidence: {reasons}. It cannot be carried forward by this decision.',{reasons:(v.reason_codes || []).map(codeWords).join(' · ') || codeWords(v.classification)})}</p>`}</details>`;
    return html`${panel(t('Curation decision'),t('Which declared factors are carried forward, in which role, and why. Immutable metadata once saved; no computation. Saving writes metadata beside this study and runs nothing; the product refuses a role the evidence does not admit.'),html`<p class="caption" id="studyCurationCount">${curationCount()}</p><details class="reveal-details"${chosen || S.limits.size ? ' open' : ''}><summary>${t('Choose factors and acknowledge limitations')}</summary>${c.choices.map(choice)}<p class="caption" id="studyCurationNotice">${t('Unavailable roles are refused by the product, not replaced with keep/reject labels.')}</p>${c.limitations.map(v=>html`<label class="check-row"><input type="checkbox" data-study-limit="${v}"${S.busy==='write'?' disabled':''}${S.limits.has(v)?' checked':''}>${coded(v)}</label>`)}${typedBtn(t('Review curation'),'study-curate-preview','','button primary',ready()?'':t('Wait for readback'))}</details>`,'','data-box="workspace"')}
      ${c.decisions.length ? panel(t('Next step'),t('The handoffs start from a saved decision, never from unsaved controls. Preview derives the Foundation candidate from the decision without publishing it; sealing is a separate explicit confirmation. The Alpha draft opens on the Lab page bound to the decision and fits nothing.'),html`<div class="field"><label id="studyDecisionLabel" for="studyDecision">${t('Saved curation')}</label>${picker('studyDecision',[['',c.decisions.length ? t('Choose explicitly') : t('No decision saved yet')],...c.decisions.map(d=>[d.receipt_hash,`${actorWords(d.actor_submission.actor_kind)} · ${short(d.receipt_hash, SHORT.hash)}`])],{selected:S.decision,disabled:S.busy==='write',labelId:'studyDecisionLabel'})}</div><div class="field"><label id="studyHandoffInputLabel" for="studyHandoffInput">${t('Exact handoff input')}</label>${picker('studyHandoffInput',handoffChoices.map(v=>[JSON.stringify([v.id,v.binding_hash]),`${v.id} · ${v.date} · ${short(v.binding_hash, SHORT.hash)}`]),{selected:S.target,disabled:S.busy==='write',labelId:'studyHandoffInputLabel',placeholder:t('No published version available')})}<small>${t('The published version the handoff binds to; the study\'s own version is offered first.')}</small></div>${S.decision?html`${codeRef(t('Selected decision and rationale'), c.decisions.find(d=>d.receipt_hash===S.decision))}`:''}<div class="flow next-step-actions">${typedBtn(t('Preview Foundation'),'study-foundation-preview','','button',handoffOff)}${typedBtn(t('Create Alpha draft'),'study-alpha-draft','','button',handoffOff)}</div>${S.preview?html`<div class="subpanel"><h3>${t('Foundation previewed · not sealed')}</h3>${foundationFacts(S.preview.admission)}${btn(t('Review Foundation publication'),'study-foundation-confirm','','button primary')}</div>`:''}`,'','data-box="workspace"') : ''}`;
  }
  function factor() {
    // The three counts as tiles; the basis they share -- development evidence, the full
    // denominator -- said once under them.
    const rows=factorRows();
    const evidence=html`<section class="panel" data-box="table"><div class="table-toolbar"><div class="panel-label"><h2>${t('Factor evidence')} <span class="num">${count(S.body.document.factor.factor_ids.length)}</span></h2>${infoMark(`${factorNote()} ${t('Development evidence')}: ${t('Selection scopes the report, not the full multiple-testing denominator. No prospective validation is implied.')}`)}</div><div class="search-field">${icon('search')}<input id="studyFactorQuery" class="search-input" type="search" placeholder="${t('Find factor')}" aria-label="${t('Find factor')}" value="${S.factorQuery}" autocomplete="off"></div></div><div id="studyFactorTable">${table(FACTOR_HEADERS(),rows,'',{classes:'compact',report:true,grid:true,countLine:false})}</div></section>`;
    return html`${curationEditor()}${detailSplit(evidence,'detail')}${propertiesSection()}`; // N2 (law 126 amended): the decision first, the evidence the lane's width, the properties after; law 149: a factor's detail beside its table
  }
  /* A sealed Foundation's page (N6): what it admits -- its ordered factors and the limitations
   * it carries -- first; where it came from as a side box; the exact identities in Facts. */
  function foundationBody(a) {
    const f=a.foundation, ids=f.ordered_factor_ids || [];
    const factors=panel(html`${t('Admitted factors')} <span class="num">${count(ids.length)}</span>`,t('The factors this Foundation carries into an Alpha draft, in their admitted order.'),html`<div class="card-list lines slotted">${ids.map((id,i)=>objectRow({lead: statusDot('verified'), name: html`<span class="mono">${id}</span>`},{key:id,columns:['order'],props:[t('#{n} in order',{n:i+1})]}))}</div>`);
    const limits=(f.limitations || []).length ? panel(html`${t('Recorded limitations')} <span class="num">${count(f.limitations.length)}</span>`,'',html`<div class="card-list">${f.limitations.map((code)=>objectRow({lead: statusDot('metadata',t('Recorded limitation')), name: coded(code), cls: 'limit-row'}))}</div>`) : '';
    const props=panel(t('Properties'),'',kv([[t('Factor study'),link(studyName(a.factor_task_id),'factor','text-btn',{study:a.factor_task_id})],[t('Input'),html`${a.input_id} · ${cutoff(a.input_binding_hash)}`],[t('Market as-of'),f.execution_outcome?.market_as_of || ''],[t('Curation decision'),t('recorded')]],'kv-columns'));
    return html`${factors}${limits}${props}${factsRef(html`${t('Exact identity')}`, html`${kv([[t('Admission'),html`<span class="mono">${a.admission_hash}</span>`],[t('Foundation'),html`<span class="mono">${f.foundation_hash}</span>`],[t('Factor task'),html`<span class="mono">${a.factor_task_id}</span>`],[t('Curation'),html`<span class="mono">${a.curation_receipt_hash}</span>`],[t('Input binding'),html`<span class="mono">${a.input_binding_hash}</span>`]])}`)}`;
  }
  /* A study's page (N2 of the open-issues plan, law 126 amended): one column -- the notice of its next
   * step or decision first, its result the lane's width, its Properties after; nothing stands beside. */
  function foundationFacts(a) {
    const f=a.foundation;
    return html`${kv([[t('Factor study'),html`${mono(a.factor_task_id,SHORT.id)} ${studyLink('factor',a.factor_task_id,t('Open Factor study'))}`],[t('Curation decision'),mono(a.curation_receipt_hash,SHORT.hash)],[t('Exact input'),html`${a.input_id} · ${cutoff(a.input_binding_hash)} · ${mono(a.input_binding_hash)}`],[t('Ordered factors'),html`${countText(f.ordered_factor_ids.length, '{n} factor', '{n} factors')} <span class="sub-cell">${f.ordered_factor_ids.join(', ')}</span>`],[t('Market as-of'),f.execution_outcome.market_as_of]])}${(f.limitations || []).length?html`<h3>${t('Recorded limitations')}</h3><div class="card-list">${(f.limitations || []).map((code)=>objectRow({lead: statusDot('metadata',t('Recorded limitation')), name: coded(code), cls: 'limit-row'}))}</div>`:''}${factsRef(html`${t('Exact identity')}`, html`${kv([[t('Admission'),html`<span class="mono">${a.admission_hash}</span>`],[t('Foundation'),html`<span class="mono">${f.foundation_hash}</span>`],[t('Factor task'),html`<span class="mono">${a.factor_task_id}</span>`],[t('Curation'),html`<span class="mono">${a.curation_receipt_hash}</span>`],[t('Input binding'),html`<span class="mono">${a.input_binding_hash}</span>`]])}`)}`;
  }
  /* The seal this page just confirmed, while its admission is the one shown: the owner's own
   * disposition (a new publication, or the exact reuse of an admission already sealed). */
  const sealedNow=()=>Boolean(S.sealed) && S.admission?.admission_hash===S.sealed.hash;
  /* The Foundation page is a chooser and an object like the other Studies pages (2026-09-21, the
   * user's reading of the split it was: the admission had no close, opened by itself from a
   * retained address, and its 1.4 s read moved nothing): the list of sealed admissions, one row
   * each, and the admission the address names as the page -- its title the crumb's last word,
   * the page's word the way up, `<-` the way back. */
  const foundationFactorsTitle=(ids)=>`${countText(ids.length, '{n} factor', '{n} factors')}${ids.length ? ' · '+ids.slice(0,3).join(', ')+(ids.length>3 ? ' +'+(ids.length-3) : '') : ''}`;
  const foundationTitle=(a)=>foundationFactorsTitle(a.foundation?.ordered_factor_ids || []); // N6 (law 134): its factors, as a Factor study's label says them
  function foundationRefusal(b) {
    const reason=b.missing_factor_task_id ? html`${t('The recorded Factor Task is missing from Task Control.')} ${hashCell(b.missing_factor_task_id,SHORT.id)} ${t('A new Factor study starts from the recorded input; the original declaration is not retained.')}` : t('The recorded Foundation sources could not be verified.');
    const ways=prerequisiteWays(b.next_requests) || link(t('Research inputs'),'inputs','button compact');
    return refusal({code:b.failure_code,reason},TONE.attention,{word:t('Foundation could not be verified'),next:'',more:html`<div class="flow">${ways}</div>`});
  }
  function foundationPage() {
    const sealed=sealedNow() ? (S.sealed.status==='REUSED_EXACT' ? noteLine(t('Foundation already sealed · exact reuse'),t('The confirmed seal matched an admission this workspace had already sealed; the owner returned it exactly and published nothing new. Its sources and decision are the ones below.'),'neutral','','info') : noteLine(t('Foundation sealed'),t('The previewed candidate is now an immutable admission in this workspace; nothing was trained, activated or replaced.'),'ok','','checkcircle')) : '';
    const id=hashParams().get('foundation');
    if(!id)return foundationList(sealed);
    if(S.errorBody?.foundation_admission_hash===id)return html`${objectHead(S.errorBody.admission ? foundationTitle(S.errorBody.admission) : t('Research Foundation'),'','', '',[],{object:true,id})}${foundationRefusal(S.errorBody)}`;
    if(!S.admission || S.admission.admission_hash!==id)return S.busy==='foundation' ? html`${skeleton('head')}${skeleton('body')}` : foundationList(sealed); // being read, it stands in its place; refused, the list stands under the notice
    const a=S.admission;
    const lede=html`<p class="lede">${t(sealedNow() ? (S.sealed.status==='REUSED_EXACT' ? 'This admission was already sealed: the owner returned the existing artifact exactly; nothing new was published.' : 'Sealed just now from the previewed candidate; immutable from here on.') : 'An immutable research handoff: the explicit source for an Alpha draft.')} ${t('Reading or publishing a Foundation does not train a model, change current inputs or activate a strategy.')}</p>`;
    const tools=[{ic:'file',action:'study-foundation-export',value:a.admission_hash,word:t('Export JSON'),why:t('The admission exactly as sealed')}];
    // U26: its standing is the list's (the owner's one reading of every admission), read quietly when not yet read
    const listed=[...(S.foundations || []),...(named?.revision===S.revision ? named.rows : [])].find(f=>f.admission?.admission_hash===a.admission_hash);
    if(!listed && !S.foundations) nameFoundations();
    const old=listed?.standing==='HISTORICAL';
    const draft=typedBtn(t('Create Alpha draft'),'study-foundation-draft',a.admission_hash,'button primary',old ? 'A historical Foundation offers its export only; new work starts on a current one.' : '');
    const standing=old ? noteLine(t('Historical'),html`${t('It reads by its recorded graph; a draft or a PLAN on it is refused.')}${listed.standing_code ? html` · ${coded(listed.standing_code)}` : ''}`,'neutral') : '';
    return html`${objectHead(foundationTitle(a),lede,draft,badge(old ? 'historical' : 'recorded', old ? undefined : t('foundation-standing|Recorded')),tools,{object:true,id:a.admission_hash})}${sealed}${standing}${foundationBody(a)}`;
  }
  function foundationList(sealed) {
    // a sealed admission records no day (its fields are its sources and its outcome): the list groups by its input version
    const list=(S.foundations || []).map(v=>{const a=v.admission,id=a?.admission_hash || v.foundation_admission_hash,asOf=a?.foundation?.execution_outcome?.market_as_of,old=v.standing==='HISTORICAL',blocked=v.status==='REFUSED',name=a ? foundationTitle(a) : t('Research Foundation');return {key:id,...(blocked ? {state:'refused'} : old ? {state:'historical'} : {}),name,words:name,ref:blocked ? hashCell(id,SHORT.hash) : short(id,SHORT.hash),interval:blocked ? html`${codeWords(v.failure_code)}${v.missing_factor_task_id ? html` · ${t('Factor task')} ${hashCell(v.missing_factor_task_id,SHORT.id)}` : ''}` : html`${asOf ? t('as-of {d}',{d:asOf}) : ''}${old && v.standing_code ? html`${asOf ? ' · ' : ''}${codeWords(v.standing_code)}` : ''}`,input:a ? html`${a.input_id} · ${cutoff(a.input_binding_hash)}` : t('Not verified'),inputKey:a?.input_binding_hash || '',inputText:a?.input_id || t('Not verified'),to:{action:'study-foundation-open',value:id}};});
    const state=S.foundations===null&&S.foundationsError?notRead(t('The Foundation list could not be read'),S.foundationsError,t('Refresh reads it again; nothing is inferred about what is sealed.')):S.foundations===null?html`<p class="caption" role="status">${t(S.busy==='foundation' ? 'Reading the sealed Foundations; each is re-derived from its sources before it is listed.' : 'The Foundation list has not been read; Refresh reads it.')}</p>`:!S.foundations.length?emptyState(html`${t('No Foundation yet')}${infoMark(t('One is sealed from a saved curation decision on a Factor study: preview it there, then confirm the seal.'))}`,'','page-empty'):'';
    return html`${objectHead(t(ROUTES.foundation[1]),html`<p class="lede">${t('An immutable research handoff, not a foundation model or current strategy.')} ${t('Reading or publishing a Foundation does not train a model, change current inputs or activate a strategy.')} ${t('One is sealed from a saved curation decision on a Factor study: preview it there, then confirm the seal.')}</p>`)}${sealed}${list.length ? LiveViews.studyLobby('foundation',list,{axes:['input']}) : state}`;
  }
  const RELATION_WORDS={PARAMETERS:'The declared model parameters alone, over one target and one feature axis',FEATURE_ADDITION:'One feature axis holds every feature of the other and more, over the same target values'};
  function alphaComparison() {
    const c=S.alphaComparison,b=S.body;
    const current=(b.result?.candidates || []).filter(v=>v.status==='DEVELOPMENT_EVALUATED');
    const saved=(Data.experiments() || []).filter(v=>comparisonListed(v)&&v.task_id!==S.task);
    const candidates=(c.rightBody?.result?.candidates || []).filter(v=>v.status==='DEVELOPMENT_EVALUATED');
    const r=c.result;
    const folds=(r?.folds || []).map(v=>tr([
      mono(v.fold_commitment_hash,SHORT.hash),figure(v.left?.metrics?.zero_relative_oos_r2),
      figure(v.right?.metrics?.zero_relative_oos_r2),figure(v.left?.metrics?.rank_ic),
      figure(v.right?.metrics?.rank_ic),`${codeWords(v.left?.status)} / ${codeWords(v.right?.status)}`
    ]));
    // The owner's answer for the exact pair: both operands as it named them, its disposition and
    // status, the paired folds when it found the pair comparable, an incomplete side explained,
    // and the export of exactly what it returned.
    const side=(v,fallbackTask,fallbackCandidate)=>html`<span class="mono">${LiveViews.shortRef(v?.task_id || fallbackTask)}</span> · <span class="mono">${v?.candidate_id || fallbackCandidate || ''}</span>${v?.status && v.status!=='EXPERIMENT_PUBLISHED' ? html` <span class="sub-cell">${codeWords(v.status)}${v.failure_code ? html` · <span class="mono">${v.failure_code}</span>` : ''}</span>` : ''}`;
    const result=r?html`<div class="subpanel">${noteLine(t('Descriptive saved evidence'),t('The product compares exactly selected stored candidates. It does not select a winner, refit a model or create independent validation.'),'neutral','','info')}${kv([[t('Left (this study)'),side(r.left,S.task,c.leftCandidate)],[t('Right'),side(r.right,c.rightTask,c.rightCandidate)],[t('Disposition'),codeWords(r.disposition)],[t('Status'),codeWords(r.status)],...(r.relation ? [[t('What differs'),RELATION_WORDS[r.relation] ? t(RELATION_WORDS[r.relation]) : codeWords(r.relation)]] : []),...(r.relation==='FEATURE_ADDITION' ? [[t('Features added'),featureList(r.declared_feature_difference?.added || [])],[t('Features removed'),(r.declared_feature_difference?.removed || []).length ? featureList(r.declared_feature_difference.removed) : t('none')]] : []),[t('Left model'),JSON.stringify(r.declared_parameter_difference?.left || {})],[t('Right model'),JSON.stringify(r.declared_parameter_difference?.right || {})],[t('Scored rows'),r.score_support?.scored_row_count ?? ''],[t('Surface cells'),r.score_support?.common_surface_row_count ?? '']])}${r.status==='COMPARABLE'?html`<section class="section-gap"><div class="table-toolbar"><h3>${t('Paired fold evidence')}</h3></div>${table([t('Fold commitment'),t('Left OOS R²'),t('Right OOS R²'),t('Left Rank IC'),t('Right Rank IC'),t('State')],folds)}${codeRef(t('Exact prerequisites, support and stored metrics'), {prerequisites:r.prerequisites,score_support:r.score_support,folds:r.folds})}</section>`:r.status==='INCOMPLETE'?html`<p class="caption">${t('One selected candidate is not complete saved evidence (its Task or candidate state is shown beside it); the owner reports the pair without inventing any comparison metric. Choose a completed candidate on that side.')}</p>`:''}<p class="caption">${t('Limitations')}: ${(r.limitations || []).map(coded).join(' · ')}</p><div class="flow">${btn(t('Export comparison JSON'),'study-alpha-comparison-export','','button compact')}</div></div>`:'';
    // An owner refusal names the incompatible prerequisite; the selection is kept as chosen and
    // nothing is aligned, truncated or refitted to make the pair comparable.
    const refused=c.error?refusal({code:c.error,reason:html`${explain(c.error) || codeWords(c.error)} ${t('The owner decides comparability; the selection is kept and nothing is re-aligned, truncated or refitted.')}`},'warning',{word:t(explain(c.error) ? 'Comparison not admitted' : 'Saved comparison unavailable'),next:'',action:c.error===COMPARISON_UNSUPPORTED ? (c.rightPage ? studyLink(c.rightPage,c.rightTask,t('Open ordinary study'),{alpha_left_task:'',alpha_left_candidate:'',alpha_right_task:'',alpha_right_candidate:''}) : btn(t('Inspect task'),'task',c.rightTask,'button compact')) : ''}):'';
    return html`<section class="panel pad section-gap study-compare-form" data-box="workspace"><div class="field"><label id="studyAlphaComparisonLeftLabel" for="studyAlphaComparisonLeft">${t('This study candidate')}</label>${picker('studyAlphaComparisonLeft',[['',t('Choose explicitly')],...current.map(v=>[v.candidate_id,v.candidate_id])],{selected:c.leftCandidate,disabled:!ready()||Boolean(c.busy),labelId:'studyAlphaComparisonLeftLabel'})}</div><div class="field"><label id="studyAlphaComparisonTaskLabel" for="studyAlphaComparisonTask">${t('Other saved Alpha study')}</label>${picker('studyAlphaComparisonTask',[['',t('Choose explicitly')],...saved.map(v=>[v.task_id,`${v.task_id} · ${v.summary || t('Saved Alpha study')}`])],{selected:c.rightTask,disabled:!ready()||Boolean(c.busy),labelId:'studyAlphaComparisonTaskLabel'})}</div><div class="field"><label id="studyAlphaComparisonCandidateLabel" for="studyAlphaComparisonCandidate">${t('Other study candidate')}</label>${picker('studyAlphaComparisonCandidate',[['',t('Choose explicitly')],...candidates.map(v=>[v.candidate_id,v.candidate_id])],{selected:c.rightCandidate,disabled:!c.rightBody||Boolean(c.busy),attrs:!c.rightBody||c.busy?html`title="${t(c.busy ? 'Reading the selected study…' : 'Choose the other saved Alpha study first; its candidates are listed here.')}"`:'',labelId:'studyAlphaComparisonCandidateLabel'})}</div>${c.busy==='read'?html`<p class="caption">${t('Reading only the explicitly selected saved study…')}</p>`:''}${refused}${typedBtn(t('Compare exact saved evidence'),'study-alpha-comparison-run','','button',comparisonPairReady()?'':t('Choose both exact candidates'))}${result}</section>`;
  }
  /* The Alpha reader in reading order: what was trained (target, model, features, their
   * source), how it was evaluated (causal folds, interval, maturity, holdout, recorded work),
   * the candidates with their development metrics in the owner's units, the folds with their
   * windows, and the two permitted continuations. A score is development evidence on validation
   * folds, never a realized return; nothing is trained, scored or chosen by reading. */
  /* U27: an Alpha qualification, a study of its own: the family it qualified over (the development studies on its
   * goal's question since the goal opened, those attempted without evidence), the candidates it nominated and its
   * disposition -- a candidate set, or no stable current model, a result and never a failure -- with its receipt. */
  const QUALIFIED={CURRENT_ALPHA_CANDIDATE_SET_READY:['verified','Candidate set ready'],NO_STABLE_CURRENT_ALPHA_MODEL:['metadata','No stable current model']};
  function qualification(b) {
    const q=b.alpha_qualification, f=b.qualification_family || {}, [state,word]=QUALIFIED[q.disposition] || ['metadata',''];
    const ids=(list)=>list?.length ? html`${list.map((id,i)=>html`${i ? ', ' : ''}<span class="mono" data-tip="${id}">${short(String(id).replace(/^alpha-candidate-/,''),SHORT.id)}</span>`)}` : t('none');
    const members=(f.members || []).map(m=>objectRow({state:'succeeded',name:studyName(m.task_id),to:{action:'study-open',value:m.task_id}},{key:m.task_id,columns:['admitted'],props:[when(m.admitted_at)]}));
    const unfinished=(f.unfinished_task_ids || []).map(id=>objectRow({state:'cancelled',name:studyName(id),why:t('attempted without evidence')},{key:id}));
    const facts=kv([[t('Disposition'),stateLine(state,{word:word ? t(word) : codeWords(q.disposition),next:''})],[t('Goal'),f.goal_id ? link(html`${t('Goal')} <span class="mono">${short(f.goal_id,SHORT.id)}</span>`,'goal','text-btn',{goal:f.goal_id}) : ''],[t('Family opened'),f.opened_at ? when(f.opened_at) : ''],[t('Candidates attempted'),ids(q.attempted_candidate_ids)],[t('Nominated'),ids(q.selected_candidate_ids)],[t('Currently qualified'),ids(q.current_qualified_ids)],[t('Recorded work'),t('{f} fits · {p} predictions · {m} metric calls',{f:count(q.fit_call_count),p:count(q.predict_call_count),m:count(q.metric_call_count)})]],'kv-columns');
    const family=members.length || unfinished.length ? panel(html`${t('Family')} <span class="num">${count(members.length+unfinished.length)}</span>`,t('Every development study on the question since the goal opened; one cancelled before it sealed a result adds no hypothesis.'),html`<div class="card-list lines slotted">${members}${unfinished}</div>`) : '';
    return html`${panel(t('Qualification'),t('Qualified over the whole attempted family; the sealed holdout is unread, and nothing is activated.'),facts)}${family}${codeRef(t('Exact qualification receipt (JSON)'),q)}${propertiesSection()}`;
  }
  function alpha() {
    const b=S.body,p=b.execution_preview || {},d=b.document.alpha || {},a=b.alpha_source || {},sp=p.split_policy || {},r=b.result || {};
    if(b.alpha_qualification) return qualification(b);
    if(b.lifecycle_research) {
      const receipt=b.lifecycle_research, rule=receipt.lifecycle;
      return panel(t('Model lifecycle research'),t('Saved model training and ensemble scoring. Predictive metrics are not produced by this lifecycle method; no strategy is activated.'),html`${kv([
        [t('Component'),p.component_id ? html`<span class="coded" data-tip="${p.component_id}">${codeWords(p.component_id)}</span>` : ''],[t('Features'),html`${countText((p.ordered_feature_ids || []).length,'{n} feature','{n} features')}${(p.ordered_feature_ids || []).length ? html` · ${featureList(p.ordered_feature_ids)}` : ''}`],
        [t('Refit cadence'),t('Every {n} months; anchor month {m}',{n:rule.month_interval,m:rule.anchor_month})],
        [t('Training window / purge (sessions)'),`${rule.training_window_sessions} / ${rule.purge_sessions}`],
        [t('Seeds'),rule.seeds.join(', ')],[t('Vintage weights'),rule.vintage_weights.join(', ')],
        [t('Required vintages'),(p.required_vintages || []).join(', ')],
        [t('Original fits'),receipt.fit_call_count ?? ''],[t('Original score predictions'),receipt.prediction_call_count ?? ''],
        [t('Scored formations'),receipt.formation_count ?? receipt.formation_sessions?.length ?? ''],
        [t('Statistical interval'),statisticalInterval(b)]
      ])}${codeRef(t('Exact lifecycle receipt'),{content_hash:receipt.content_hash,program_hash:receipt.program_hash,lifecycle:receipt.lifecycle})}`);
    }
    const features=d.ordered_feature_ids || [];
    // N6: what was trained and how it was evaluated are the study's properties, one line each; the
    // parameters, the split and the recorded work are their (i)s, the exact declaration the Facts'
    const params=Object.entries(d.model_parameters || {}).filter(([k])=>k!=='family').map(([k,v])=>`${k.replaceAll('_',' ')} ${typeof v==='object' ? JSON.stringify(v) : v}`).join(' · ');
    const alphaRows=[
      [t('study|Target'),html`${methodWords(d.target_recipe_id) || ''}${infoMark(t('the quantity every score predicts, by the installed recipe'))}`],
      [t('Model'),html`${methodWords(d.model_parameters?.family || p.model_adapter_id) || ''}${infoMark(`${params || t('no parameters recorded')} · ${codeWords(p.determinism)}`)}`],
      [t('Features'),html`${countText(features.length,'{n} feature','{n} features')}${features.length ? html` · ${featureList(features)}` : ''}`],
      [t('Folds'),html`${p.fold_count ?? ''}${sp.mode ? ` · ${codeWords(sp.mode)}` : ''}${infoMark(t('train {a} · validation {b} · purge {c} · embargo {d} sessions',{a:sp.train_sessions ?? '',b:sp.validation_sessions ?? '',c:sp.purge_sessions ?? '',d:sp.embargo_sessions ?? ''}))}`],
      [t('Statistical interval'),html`${statisticalInterval(b)}${infoMark(t('{n} sessions; the declared authority interval does not reslice the folds',{n:p.statistical_session_count==null ? '' : count(p.statistical_session_count)}))}`],
      [t('Maturity lag'),html`${countText(p.maturity_lag_sessions ?? '','{n} session','{n} sessions')}${infoMark(t('sessions between a score and its matured outcome'))}`],
      [t('Sealed holdout'),html`${countText(sp.sealed_holdout_sessions ?? '','{n} session','{n} sessions')}${infoMark(t('sessions kept unread by this study'))}`],
      [t('Recorded work'),html`${countText(r.fit_call_count ?? '','{n} fit','{n} fits')}${infoMark(t('{f} fit, {p} prediction and {m} metric calls by the completing execution; this page trains nothing',{f:r.fit_call_count ?? '',p:r.predict_call_count ?? '',m:r.metric_call_count ?? ''}))}`]];
    // Each row leads with its identity, state and action in one cell and keeps the metrics in
    // labelled numeric columns (the label is the header at width, the caption in the narrow
    // card arrangement); the action is never squeezed by the metric columns.
    // Round 60 (Stripe's table): one line per row — the identity leads, the state is a column, the
    // metrics are labelled numeric columns and the action is a quiet link at the end.
    const metric=(v)=>html`<td class="num">${figure(v)}</td>`;
    const shortId=(id)=>short(String(id || '').replace(/^alpha-candidate-/,''),SHORT.id);
    const candidateHeaders=[{label:t('Candidate'),type:'id'},{label:t('State'),type:'text'},...['OOS R² (pooled, vs zero)','Rank IC (mean per session)','Decile spread (gross, fraction)','Fold coverage (fraction)'].map((x)=>({label:t(x),type:'num'}))];
    const candidates=(r.candidates || []).map(v=>html`<tr><td><span class="main-cell mono" data-tip="${v.candidate_id}">${shortId(v.candidate_id)}</span></td><td>${[codeWords(v.status),...(v.failure_codes || [])].filter(Boolean).join(' · ')}</td>${metric(v.pooled_oos_r2)}${metric(v.mean_rank_ic)}${metric(v.mean_gross_decile_spread)}${metric(v.fold_coverage_mean)}</tr>`);
    // N6 (the plan's next steps): an evaluated candidate seeds a Portfolio draft from the side column's box, not a column the table rolls away
    const evaluatedCandidates=(r.candidates || []).filter(v=>v.status==='DEVELOPMENT_EVALUATED');
    const nextStep=evaluatedCandidates.length ? banner(t('Next step'),t('"Create Portfolio draft" opens a source-bound draft on the Lab page: PLAN and one explicit confirmation follow; nothing is refitted or run by opening it.'),'neutral',html`<div class="flow next-step-actions">${evaluatedCandidates.map(v=>typedBtn(evaluatedCandidates.length>1 ? t('Create Portfolio draft from {id}',{id:shortId(v.candidate_id)}) : t('Create Portfolio draft'),'study-portfolio-draft',v.candidate_id,'button primary',b.status==='EXPERIMENT_SUMMARY' ? t('Verify evidence before continuing') : ''))}</div>`,'arrow') : '';
    const windows=Object.fromEntries((p.folds || []).map(f=>[f.fold_index,f]));
    const oneCandidate=new Set((b.fold_results || []).map(v=>v.candidate_id)).size<=1; // one candidate's folds name it once, in the caption
    const oneState=(b.fold_results || []).length>0 && new Set((b.fold_results || []).map(v=>v.status)).size===1; // law 143: a state every fold shares is said once, in the caption, as the one candidate is (it filled a column beside a fold's detail at 125 %)
    const foldHeaders=[{label:t('Fold'),type:'id',cls:'col-tight'},...(oneCandidate ? [] : [{label:t('Candidate'),type:'id'}]),...(oneState ? [] : [{label:t('State'),type:'text',cls:'col-tight'}]),{label:t('Train window'),type:'date'},{label:t('Validation window'),type:'date'},...['MAE','MSE','OOS R²','Rank IC'].map((x)=>({label:t(x),type:'num',cls:'col-tight'}))]; // the short columns tight, so the windows keep their line at 1673 x 1186 @ 125 %
    const span=(w,a,b)=>w[a] ? `${w[a]} — ${w[b]}` : '';
    const folds=(b.fold_results || []).map((v,i)=>{const w=windows[v.fold_index] || {};return html`<tr><td>${btnAttrs(html`${foldLabel(v.fold_index)}${icon('arrow')}`,'study-fold',String(i),'text-btn cell-opener','data-row-press')}</td>${oneCandidate ? '' : html`<td><span class="mono">${shortId(v.candidate_id)}</span></td>`}${oneState ? '' : html`<td>${codeWords(v.status) || ''}</td>`}<td>${span(w,'train_start','train_end')}</td><td>${span(w,'validation_start','validation_end')}</td>${metric(v.metrics?.mae)}${metric(v.metrics?.mse)}${metric(v.metrics?.zero_relative_oos_r2)}${metric(v.metrics?.rank_ic)}</tr>`;}); // the fold's name opens it (Stripe's list: the first column is the link)
    const foldCaption=html`${oneCandidate && (b.fold_results || []).length ? html` ${t('All folds belong to candidate {id}.',{id:shortId(b.fold_results[0].candidate_id)})}` : ''}${oneState ? html` ${t('Every fold is {state}.',{state:String(codeWords(b.fold_results[0].status) || '').toLowerCase()})}` : ''}`;
    const candidatesMarkup=html`<section class="panel" data-box="table"><div class="table-toolbar"><div class="panel-label"><h2>${t('Candidates')} <span class="num">${count(candidates.length)}</span></h2>${infoMark(`${t('A candidate is one model configuration. Its numbers are development evidence on the validation folds: OOS R² is pooled against a zero forecast, rank IC is the mean per-session rank correlation of score and matured target, the gross decile spread is per session before costs. A score is not a realized return and no winner is inferred.')} ${t('Values to four decimals, tiny non-zero values in exponent form; the exact values are in the JSON export.')}`)}</div></div>${table(candidateHeaders,candidates,'',{report:true,grid:true,countLine:false})}</section>`;
    const foldsMarkup=html`<section class="panel" data-box="table"><div class="table-toolbar"><div class="panel-label"><h2>${t('Causal folds')} <span class="num">${count(folds.length)}</span></h2>${infoMark(`${t('Each fold trains before its validation window with the declared purge and embargo; the sealed holdout is not among them.')}${foldCaption ? ' '+String(foldCaption).replace(/<[^>]+>/g,'').trim() : ''}`)}</div></div>${b.status==='EXPERIMENT_SUMMARY' ? emptyState(t('Detailed fold evidence appears after verification.')) : table(foldHeaders,folds,'',{report:true,grid:true,countLine:false})}</section>`;
    return html`${nextStep}${candidatesMarkup}${detailSplit(foldsMarkup,'detail')}${propertiesSection(alphaRows)}`; // N2 (law 126 amended): the next step a notice first, both tables the lane's width, the properties after
  }
  /* The Risk reader: the estimator and its parameters, the dated formation window and listing
   * axis, the diagnostics per formation in the owner's units (absent ones marked), the exact
   * report, and the report-only association explained before its confirmation. A Risk matrix
   * sizes no Portfolio here. */
  function risk() {
    const b=S.body,rows=b.result?.evaluations || [],shown=rows.slice(S.riskOffset,S.riskOffset+40),target=Data.subject(),est=b.document.risk?.estimator || {},x=b.execution_preview || {},surface=b.risk_surface || {};
    const scopes=surface.scope_surfaces || [surface];
    const sessions=surface.formation_sessions || [];
    const parameters=JSON.stringify(est.parameters || {}); // exact owner parameters, not inferred labels
    const riskRows=([
      [t('Estimator'),html`${methodWords(est.capability || surface.capability_handle) || ''}${infoMark(`${t('parameters')} ${parameters}`)}`],
      [t('Formations'),html`${rows.length}${infoMark(`${sessions.length ? `${sessions[0]} — ${sessions.at(-1)}` : t('window not recorded')} · ${t('each with one realized next session')}`)}`], // the count; its window the (i) (a value is never cut)
      [t('Listing coverage'),html`${(surface.ordered_listing_ids || []).length || x.listing_count || ''}${infoMark(t('listings across all dated scopes; the historical union, not a matrix axis'))}`],
      [t('Dated scopes'),html`${scopes.length}${infoMark(t('axes of {sizes} assets; each formation uses its own scope\'s axis',{sizes:scopes.map(v=>(v.ordered_listing_ids || []).length).join(', ')}))}`],
      [t('Sector snapshot'),x.sector_revision ? mono(x.sector_revision,SHORT.hash) : t('not recorded')],
      [t('Recorded work'),html`${countText(b.execution_numerical_call_count ?? '','{n} numerical call','{n} numerical calls')}${infoMark(t('by the completing execution; this page estimates nothing'))}`]]);
    const linked=Data.riskLinks(target?.task_id);
    const refusedLinks=(Data.riskLinkRefusals?.(target?.task_id) || []).map(r=>refusal(r,'warning',{catalog:true,more:html`<p>${t('Risk link')} ${hashCell(r.link_hash)}</p>`,next:prerequisiteWays(r.next_requests)}));
    // V616 (U92): the subject a report reference admits, named by its owner before any association is offered -- an
    // installed strategy book carries no Portfolio study's receipt and captured input, so both ways are held with the
    // owner's words and its way on (the completed studies); a Portfolio study goes on to the owner's exact checks
    const installed=target?.source_kind==='INSTALLED_RESULT' ? b.report_reference_selection?.installed_book : null;
    const refused=installed && installed.available===false ? installed : null;
    const linkHeld=!(ready()&&target) ? t('Open a Portfolio study first') : refused ? (refused.detail ? t(refused.detail) : explainCode(refused.failure_code)) : '';
    const studiesWay=refused?.next_requests?.studies ? link(t('Choose a completed Portfolio study'),'portfolio','text-btn',{book:''}) : '';
    // N2 (law 126 amended): the next step is a notice first in the content, its verb its button; what it made --
    // the recorded associations, once read -- a section after the figures
    const association=banner(t('Next step'),html`${t('Links this diagnostic to a saved Portfolio book for reading beside it; the book\'s weights, sizing and returns are unchanged (the replay admits no risk matrix).')}${target ? html` ${t('Selected book')}: ${LiveViews.bookWords(target.policy) || LiveViews.shortRef(target.task_id)} · ${t('holdings date')} ${target.session}` : ''}${refused ? html`<p class="caption">${linkHeld} ${studiesWay}</p>` : ''}`,'neutral',html`<div class="flow next-step-actions">${typedBtn(t('Attach this report to a book'),'study-risk-preview','','button primary',linkHeld)}${typedBtn(t('Review retrospective Portfolio-window reference'),'study-risk-window-preview','','button',linkHeld)}${infoMark(t('The retrospective option requires every Portfolio formation, keeps the original Risk source and both cutoffs, and does not claim the assessment was available when the Portfolio was decided. No estimation or allocation follows.'))}${target ? btn(t('Read existing associations'),'study-risk-links',target.task_id,'button compact') : ''}</div>`,'arrow');
    const associations=linked ? html`${refusedLinks}${panel(t('Linked Risk reports'),'',html`${linked.map(v=>html`<div class="feature-row"><div class="grow"><strong>${t('Risk study')} ${LiveViews.shortRef(v.risk_task_id)}</strong><p class="mono">${v.link_hash}</p>${v.report_window ? html`<p class="caption">${t('Post-observed Portfolio-window reference')} · ${v.risk_start} — ${v.risk_end}</p>` : ''}</div>${btn(t('Export JSON'),'study-risk-export',v.link_hash,'button compact')}</div>`)}${!linked.length && !refusedLinks.length ? html`<p class="caption">${t('No Risk report is linked to this Portfolio task.')}</p>` : ''}`)}` : '';
    const absent=shown.some(v=>['shrinkage','condition_number','annualized_volatility_median','equal_weight_predicted_variance','equal_weight_realized_squared_return','gaussian_log_score_per_asset'].some(k=>v[k]==null));
    const scopeColumns=[{label:t('First formation'),type:'date'},{label:t('Last formation'),type:'date'},{label:t('Assets'),type:'num'}];
    const riskColumns=[{label:t('Formation'),type:'date'},{label:t('Realized'),type:'date'},...['Shrinkage (fraction)','Condition (number)','Volatility (median, annualized)','Predicted variance (equal weight)','Realized sq. return (equal weight)','Log score (per asset)'].map(x=>({label:t(x),type:'num'}))];
    // N6 (the plan: one figure of predicted against realized): the owner's two recorded series per formation, one axis (both are variances)
    const expText=(z)=>z==null || !Number.isFinite(Number(z)) ? '' : Number(z)===0 ? '0' : Number(z).toExponential(1);
    const series=rows.map(v=>({date:v.formation_session,predicted:v.equal_weight_predicted_variance==null ? null : Number(v.equal_weight_predicted_variance),realized:v.equal_weight_realized_squared_return==null ? null : Number(v.equal_weight_realized_squared_return)}));
    // the user (2026-09-24): the figure in its box, the lane's width, its axis in one power of ten
    const figureMarkup=rows.length>1 ? figureBox(t('Predicted against realized'),chart('indexed',true,{id:'risk-'+S.task,rows:series,lines:[{key:'predicted',cls:'series',label:'Predicted variance'},{key:'realized',cls:'benchmark',label:'Realized squared return'}],axis:'power',value:expText,note:t('The owner\'s recorded values'),title:t('Predicted against realized'),label:t('Predicted variance and realized squared return at each formation; read-only chart.')}),{info:infoMark(t('At each formation, the equal-weight variance the matrix implies, and the next session\'s realized squared return of the same equal-weight book, as the owner recorded them; one axis, both variances.')),legend:html`<span class="chart-legend"><span><i class="legend-line"></i>${t('Predicted variance')}</span><span><i class="legend-line benchmark"></i>${t('Realized squared return')}</span></span>`}) : '';
    const scopesMarkup=scopes.length>1 ? html`<section class="panel section-gap" data-box="table"><div class="table-toolbar"><div class="panel-label"><h2>${t('Dated estimation scopes')}</h2>${infoMark(t('Each dated scope has its own matrix axis. Stability comparisons restart at scope boundaries; the historical coverage union is not a covariance matrix.'))}</div></div>${table(scopeColumns,scopes.map(v=>tr([v.formation_sessions[0],v.formation_sessions.at(-1),v.ordered_listing_ids.length])),'',{report:true,grid:true,countLine:false})}</section>` : '';
    const formations=html`<section class="panel" data-box="table"><div class="table-toolbar"><div class="panel-label"><h2>${t('Formation and realization')} <span class="num">${count(rows.length)}</span></h2>${infoMark(`${t('Predicted variance is the equal-weight variance the matrix implies at the formation; the realized squared return is the next session\'s. Values to four decimals, tiny non-zero values in exponent form; the exact values are in the JSON export.')}${absent ? ' '+t('An empty cell is a diagnostic the owner did not report for that formation.') : ''}`)}</div></div>${table(riskColumns,shown.map(v=>tr([v.formation_session,v.next_session,figure(v.shrinkage),figure(v.condition_number),figure(v.annualized_volatility_median),figure(v.equal_weight_predicted_variance),figure(v.equal_weight_realized_squared_return),figure(v.gaussian_log_score_per_asset)])),'',{report:true,grid:true,countLine:false})}${rows.length>40 ? pager({page:Math.floor(S.riskOffset/40),pages:Math.ceil(rows.length/40),prev:['study-risk-page','-40'],next:['study-risk-page','40']}) : ''}</section>`;
    const axes=codeRef(t('Exact scope axes and source identities'), scopes.map(v=>({surface_hash:v.surface_hash,formation_sessions:v.formation_sessions,ordered_listing_ids:v.ordered_listing_ids})));
    return html`${association}${figureMarkup || formations}${figureMarkup ? formations : ''}${scopesMarkup}${associations}${propertiesSection(riskRows)}${axes}`; // N2: the next step first, the figure and the formations the lane's width, the properties after
  }
  function page() {
    const notice=S.error?(S.errorBody ? refusal({...S.errorBody, reason: S.errorBody.reason || S.errorBody.detail || S.errorBody.message || explain(S.error) || ''}) : refusal({code:S.error,reason:explain(S.error) || ''},'warning',{word:t(explain(S.error) ? 'The owner refused this step' : 'Research readback needs attention')})):'';
    if(app.page==='foundation')return html`${S.errorBody?.foundation_admission_hash ? '' : notice}${foundationPage()}`; // the item's refusal stays on its own page
    if(app.page==='alpha-compare' && addressed() && comparisonReady(S.body))
      return html`${objectHead(t(ROUTES['alpha-compare'][1]),`${t('Choose both saved task and candidate identities. This comparison is descriptive and uses no inferred latest result.')} ${t('What is compared: the two candidates\' paired causal folds on their common surface -- OOS R² and rank IC per fold, in the same units as above. The owner decides whether the pair is comparable; an incompatible pair is refused with its reason, and no winner is inferred.')}`,'','',[],{})}${notice}${alphaComparison()}`;
    if(!addressed())return list();
    // a study being read stands in its own place -- a head and a body of placeholders, never a
    // skeleton stacked above the list it replaces (2026-09-21, the Studies audit)
    if(!S.body && S.busy)return html`${skeleton('head')}${skeleton('body')}`;
    const readPage=app.page==='alpha-compare' ? ordinaryPage(S.body) : app.page;
    const unsupported=app.page==='alpha-compare' && S.body ? refusal({code:COMPARISON_UNSUPPORTED,reason:t(CODES[COMPARISON_UNSUPPORTED])},'warning',{word:t('Comparison not admitted'),next:'',action:readPage ? studyLink(readPage,S.task,t('Open ordinary study'),{alpha_left_task:'',alpha_left_candidate:'',alpha_right_task:'',alpha_right_candidate:''}) : btn(t('Inspect task'),'task',S.task,'button compact')}) : '';
    if(!S.body || !['EXPERIMENT_PUBLISHED','EXPERIMENT_SUMMARY'].includes(S.body.status) || !readPage || S.body.program?.kind!==kinds[readPage])return html`${notice}${unsupported || html`${list()}${S.body?.status && S.body.status!=='EXPERIMENT_PUBLISHED'?refusal({code:S.body.failure_code || ''},'warning',{state:S.body.status,action:btn(t('Inspect task'),'task',S.task,'button compact')}):''}`}`;
    const f=facts(), b=S.body, s=b.document?.experiment?.sessions || {}, recorded=Data.history().find(x=>x.task_id===S.task);
    // The head: what the study declared, in words, with its input, interval and recorded date
    // as one meta line; the catalog as a quiet entry; the one next action -- continuing this
    // declaration -- as the primary button; the exports and identities behind the menu.
    const meta=html`<p class="lede">${t(b.lifecycle_research ? 'Declared model vintages and their saved forecasts, not a candidate-fold evaluation or live strategy.' : LEDES[readPage])}</p>`;
    const headFacts=[[t('Study'),t(ROUTES[readPage][1])],[t('Input'),b.research_input_id || ''],[t('Declared interval'),dateRange(s.start,s.end)],[t('As-of'),s.as_of?.session || ''],[t('Recorded'),recorded?.recordedAt ? when(recorded.recordedAt) : '']];
    const summary=b.status==='EXPERIMENT_SUMMARY';
    const actions=typedBtn(t('Continue as a new draft'),'research-continue',S.task,'button primary',summary ? t('Verify evidence before continuing') : '');
    // the light summary first, the strict verification after: a failed verification keeps the summary and opens nothing that needs it
    const verifying=t('Saved metrics and model metadata are readable; numerical artifacts have not yet been checked for this read.');
    const verificationNotice=!summary ? '' : S.verificationError ? notRead(t('Saved summary · evidence not verified'),S.verificationError,verifying,btn(t('Retry verification'),'study-open',S.task,'button compact')) : noteLine(t('Saved summary · verifying full evidence'),verifying);
    const promoted=S.promoted?.task===S.task ? S.promoted.answer : null;
    const lane=b.research_lane!=='EXPLORATION' ? '' : noteLine(t('Explored on a sample'),promoted?.task_id ? html`${t('Its promotion runs as Task {id}.',{id:short(promoted.task_id)})} ${btn(t('Follow the Task'),'task',promoted.task_id,'text-btn')}` : promoted ? codeWords(promoted.status) : b.lane_detail ? html`<span class="owner-text">${b.lane_detail}</span>` : '','neutral',promoted || !b.promotion ? '' : btn(t('Promote to the whole universe'),'study-promote',S.task,'button compact'));
    const standing=b.method_standing==='NOT_CURRENT' ? noteLine(t('Read as recorded'),html`${t('It was sealed under a method this build no longer runs; a continuation declares it again under the installed one.')}${b.method_refusal ? html` · ${coded(b.method_refusal)}` : ''}`,'neutral') : '';
    const tools=[...(app.page==='alpha' && b.status==='EXPERIMENT_PUBLISHED' && !b.lifecycle_research ? [{ic:'grid',action:'alpha-compare-open',value:S.task,word:t('Compare with…'),why:t('A candidate of this study beside one of another saved study')}] : []),{ic:'history',action:'go',value:'history',word:t('Show in History'),why:t('Every saved object, exact readback')},{ic:'file',action:'study-export',value:'html',word:t('Export HTML'),why:t('The owner-rendered report, exactly as saved')},{ic:'file',action:'study-export',value:'json',word:t('Export JSON'),why:t('The full readback')},{ic:'file',action:'study-export',value:'yaml',word:t('Export YAML'),why:t('The declaration as YAML')}].filter(v=>!summary || v.action==='go');
    return html`${objectHead(f.words || f.brief || t(ROUTES[readPage][1]),meta,actions,badge('historical'),tools,{object:true,id:S.task,facts:headFacts})}${LiveTasks.currentGroup?.() || ''}${notice}${unsupported}${verificationNotice}${lane}${standing}${({factor,alpha,risk}[readPage])()}${standingPanel(b.standing)}${LiveViews.researchTiming(b.timing)}`; // U59: the five marks in one place
  }
  function edit(kind,id,v) {
    if(S.busy==='write')return;
    invalidate();S.decision='';
    if(kind==='role')S.roles[id]=v;else if(kind==='reason')S.reasons[id]=v;else if(v)S.limits.add(id);else S.limits.delete(id);
    const count=$('#studyCurationCount');if(count&&S.curation)count.textContent=curationCount();
    const decision=$('#studyDecision');if(decision)decision.value='';
    const preview=$('[data-action="study-foundation-confirm"]');if(preview)preview.disabled=true;
  }
  function chooseDecision(hash) {if(S.busy==='write')return;invalidate();S.decision=hash;render();}
  function target(v) {if(S.busy==='write')return;invalidate();S.target=v;render();}
  function selection() {
    if(!S.curation?.decisions.some(v=>v.receipt_hash===S.decision))throw Error('Select a saved decision');
    const [id,binding]=JSON.parse(S.target);
    if(!id||!binding)throw Error('Select an exact handoff input');
    return {task_id:S.task,curation_receipt_hash:S.decision,research_input_id:id,input_binding_hash:binding};
  }
  async function previewFoundation() {
    if(!ready())return;
    S.pending=null;S.preview=null;const ticket=S.revision;S.busy='preview';S.error='';render();
    try{const body=await Data.post('/api/experiments/foundations/preview',selection());if(ticket===S.revision)S.preview=body;}
    catch(e){if(ticket===S.revision)S.error=e.message;}
    finally{if(ticket===S.revision){S.busy='';render();}}
  }
  function confirm(kind) {
    if(!ready())return;
    let path,payload;
    try{
      if(kind==='curate'){
        if(!S.curation)throw Error('Load curation choices first');
        const choices=S.curation.choices.filter(v=>S.roles[v.factor_id]).map(v=>({factor_id:v.factor_id,role:S.roles[v.factor_id],rationale:(S.reasons[v.factor_id] || '').trim()}));
        if((!choices.length && S.curation.choices.some(v=>v.roles.length)) || choices.some(v=>!v.rationale) || S.curation.limitations.some(v=>!S.limits.has(v)))throw Error('Select admissible roles, provide rationales and acknowledge every stated limitation.');
        path='/api/experiments/curation';payload={task_id:S.task,experiment_curation:{expected_receipt_hash:S.curation.receipt_hash,choices,limitations_acknowledged:[...S.limits]}};
      }else if(kind==='seal'){
        if(!S.preview?.admission)throw Error('Preview Foundation first');
        path='/api/experiments/foundations/seal';payload={foundation_admission_hash:S.preview.admission.admission_hash};
      }else if(kind==='promote'){
        if(!S.body?.promotion)throw Error('This study is not explored on a sample');
        path='/api/experiments/promote';payload={task_id:S.task};
      }else{
        const p=Data.subject();if(!p)throw Error('Select a Portfolio study first');
        const sel=p.source_kind==='INSTALLED_RESULT' ? S.body?.report_reference_selection?.installed_book : null;
        if(sel && sel.available===false)throw Error(sel.detail ? t(sel.detail) : String(sel.failure_code));
        path='/api/experiments/risk-link';payload={task_id:p.task_id,risk_task_id:S.task,...(kind==='link-window'?{risk_report_scope:'POST_OBSERVED_PORTFOLIO_WINDOW'}:{})};
      }
      closeDialog();S.error='';S.pending={path,payload,kind,revision:S.revision};
      openDialog(t('Research · explicit confirmation'),t(kind==='promote' ? 'Promote this study to the whole universe?' : 'Confirm this exact metadata operation?'),html`<p>${t(kind==='promote' ? 'One Task runs the same declaration on every name of its input; the explored study stays as it is. A Portfolio study builds on its Alpha study\'s promotion, which runs first when it has not run.' : 'The product revalidates references and permissions. No model training, strategy activation or current-input change is requested.')}</p><pre class="code-block code-document">${json(payload)}</pre>`,html`${btn(t('Confirm'),'study-commit','','button primary',true)}`);
    }catch(e){S.error=e.message;render();}
  }
  async function commit() {
    const p=S.pending;if(!p||p.revision!==S.revision||S.busy)return;
    S.pending=null;S.busy='write';S.writeTicket=p;closeDialog();render();
    try{
      const b=await Data.post(p.path,p.payload);if(p.revision!==S.revision)return;
      if(p.kind==='curate'){const c=await Data.read('/api/experiments/curation?'+new URLSearchParams({task_id:p.payload.task_id}));if(p.revision===S.revision){S.curation=c;S.decision=b.decision.receipt_hash;S.roles={};S.reasons={};S.limits=new Set();}}
      if(p.kind==='seal'){S.admission=b.admission;S.sealed={hash:b.admission.admission_hash,status:b.status || 'FOUNDATION_SEALED'};S.foundations=null;S.foundationsError='';S.preview=null;app.page='foundation';replaceHash({page:'foundation',foundation:b.admission.admission_hash,study:''});}
      if(p.kind==='link'||p.kind==='link-window')await Data.refreshRiskLinks(p.payload.task_id);
      if(p.kind==='promote')S.promoted={task:p.payload.task_id,answer:b}; // R4: said in place, with its Task
    }catch(e){if(p.revision===S.revision)S.error=e.message;}
    finally{if(S.writeTicket===p){S.writeTicket=null;S.busy='';render();}}
  }
  async function exportStudy(format) {if(!ready()||!S.task)return;const ticket=S.revision;const b=await Data.read('/api/experiments/export?'+new URLSearchParams({task_id:S.task}));if(ticket!==S.revision)return;if(typeof b[format]!=='string')throw Error('Export unavailable');download(b[format],'AlphaLattice-study.'+format,format==='html'?'text/html':format==='yaml'?'text/yaml':'application/json');}
  async function exportFoundation(hash) {const b=await Data.read('/api/experiments/foundations/export?'+new URLSearchParams({foundation_admission_hash:hash}));download(b.json,'AlphaLattice-foundation.json');}
  const links=(task)=>Data.refreshRiskLinks(task);
  /* The linked report export is the owner's document for one Portfolio task and link; the Risk page
   * and the Portfolio report area both call it. */
  async function riskExport(hash,task=Data.subject()?.task_id) {if(!task)return;const d=await Data.readDocument('/api/experiments/risk-export?'+new URLSearchParams({task_id:task,risk_report_hash:hash}));download(d.text,'AlphaLattice-linked-risk.json');}
  function portfolioDraft(id) {if(!ready()||!S.body.result.candidates.some(v=>v.candidate_id===id))return;return LiveResearch.receiveHandoff({kind:'alpha_candidate',task_id:S.task,candidate_id:id});}
  async function alphaComparisonTask(task,candidate='') {
    const c=S.alphaComparison,ticket=++c.revision;c.rightTask=task;c.rightBody=null;c.rightPage='';c.rightCandidate=candidate;c.result=null;c.error='';persistAlphaSelection();
    if(!task){render();return;}c.busy='read';c.busyTicket=ticket;render();
    try {
      const body=await Data.read('/api/experiments/readback?'+new URLSearchParams({task_id:task}));
      if(ticket!==c.revision||S.task!==hashParams().get('alpha_left_task'))return;
      c.rightPage=ordinaryPage(body) || '';
      if(!comparisonReady(body))throw Error(COMPARISON_UNSUPPORTED);
      c.rightBody=body;
      if(c.rightCandidate&&!body.result?.candidates?.some(v=>v.status==='DEVELOPMENT_EVALUATED'&&v.candidate_id===c.rightCandidate))throw Error(t('Selected saved candidate is unavailable.'));
    }catch(e){if(ticket===c.revision)c.error=String(e?.message || e);}
    finally{if(c.busyTicket===ticket){c.busy='';c.busyTicket=0;}render();}
  }
  function alphaComparisonCandidate(side,candidate) {const c=S.alphaComparison;++c.revision;c[side]=candidate;c.result=null;c.error='';persistAlphaSelection();render();}
  function restoreAlphaSelection(task) {
    const selected=alphaRoute(task),c=S.alphaComparison;
    if(!selected || (sameAlphaSelection(c,selected) && (!selected.rightTask || c.rightBody || c.busy || c.error)))return;
    c.leftCandidate=selected.leftCandidate;
    if(selected.rightTask){alphaComparisonTask(selected.rightTask,selected.rightCandidate);return;}
    ++c.revision;c.rightTask='';c.rightBody=null;c.rightPage='';c.rightCandidate='';c.result=null;c.error='';render();
  }
  async function alphaComparisonRun() {
    const c=S.alphaComparison;if(!comparisonPairReady())return;
    const ticket=++c.revision,task=S.task;c.busy='compare';c.busyTicket=ticket;c.result=null;c.error='';render();
    try {const result=await Data.read('/api/experiments/alpha-compare?'+new URLSearchParams({left_task_id:task,left_candidate_id:c.leftCandidate,right_task_id:c.rightTask,right_candidate_id:c.rightCandidate}));if(ticket!==c.revision||task!==S.task)return;const reopen=result.next_requests?.reopen;if(reopen?.left_task_id===task&&reopen.left_candidate_id===c.leftCandidate&&reopen.right_task_id===c.rightTask&&reopen.right_candidate_id===c.rightCandidate){persistAlphaSelection();}c.result=result;}
    catch(e){if(ticket===c.revision)c.error=String(e?.message || e);}
    finally{if(c.busyTicket===ticket){c.busy='';c.busyTicket=0;}render();}
  }
  function exportAlphaComparison() {const c=S.alphaComparison;if(!c.result)throw Error('No saved Alpha comparison is available to export.');download(JSON.stringify(c.result,null,2),`AlphaLattice-alpha-comparison-${S.task}-${c.rightTask}.json`);}
  function alphaDraft(){return LiveResearch.receiveHandoff({kind:'factor',...selection()});}
  function fold(i){const b=S.body.fold_results[Number(i)];if(!b)return;Window.openInspector({mode:'detail',readHeader:()=>({title:foldLabel(b.fold_index),kind:t('Alpha · recorded fold')}),title:foldLabel(b.fold_index),kind:t('Alpha · recorded fold'),by:['study-fold',String(i)],body:html`<section class="inspector-section">${kv([[t('Candidate'),html`<span class="mono">${b.candidate_id}</span>`],[t('Recorded fold index'),html`${b.fold_index}${infoMark(t('the owner\'s zero-based index; displayed as {label}',{label:foldLabel(b.fold_index)}))}`]])}${kv(pairs(b.metrics))}<h3>${t('Actual fold boundaries and fit evidence')}</h3><pre class="code-block code-document">${json({preview:S.body.execution_preview.folds?.find(v=>v.fold_index===b.fold_index),fit_ledger:b.fit_ledger,role:b.role,status:b.status,failure:b.failure})}</pre></section>`});} // N2 (law 131): the fold's record beside its table
  function leaveReads(page) {
    if(S.readPage && S.readPage!==page)return;
    const unfinished=S.busy==='read' || S.status==='loading';
    invalidate();S.readPage='';if(unfinished)S.status='empty';
    const c=S.alphaComparison;if(c.busy){++c.revision;c.busy='';c.busyTicket=0;}
  }
  return {pages,leaveReads,ensure,page,open,catalog,foundations,factsSections,hasFacts,
    routeContext,confirm,commit,edit,chooseDecision,target,previewFoundation,alphaDraft,portfolioDraft,alphaComparisonTask,alphaComparisonCandidate,alphaComparisonRun,exportAlphaComparison,fold,factorDetail,exportStudy,exportFoundation,links,riskExport,
    foundationDraft:hash=>LiveResearch.receiveHandoff({kind:'foundation',foundation_admission_hash:hash}),
    filter:v=>{S.factorQuery=v;const e=$('#studyFactorTable');if(e)e.innerHTML=table(FACTOR_HEADERS(),factorRows(),factorNote(),{classes:'compact',report:true,grid:true,count:factorRows().length});},
    riskPage:v=>{S.riskOffset=Math.max(0,S.riskOffset+Number(v));render();},dismissConfirmation:()=>{S.pending=null;},
    proof:()=>app.page==='foundation'&&S.admission?foundationFacts(S.admission):provenance(),
    context:()=>app.page==='foundation'?{foundation_admission_hash:S.admission?.admission_hash || null,input_id:S.admission?.input_id,input_binding_hash:S.admission?.input_binding_hash}:S.body?.program?.kind===kinds[app.page]?{task_id:S.task,receipt_hash:S.body.receipt?.receipt_hash,input_id:S.body.research_input_id,input_binding_hash:S.body.input_binding_hash}:null};
})();
