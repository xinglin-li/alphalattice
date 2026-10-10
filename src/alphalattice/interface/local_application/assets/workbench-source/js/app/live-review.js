/* Exact-book review pages: owner facts and context-bound drafts, receipts and reads. */
const LiveReview = (() => {
  const pages=new Set(['books','evidence','evidence-stream','evidence-reading','report','handoff']);
  const EVIDENCE_KIND='alternative_evidence.document_intelligence', REVIEW_KIND='chief_risk_officer.portfolio_review';
  const PREPARE_WORDS='Prepare exact evidence for external analysis.', PREPARE_BOOK_WORDS='Prepare exact evidence for every unit of the book.';
  const preparing=(goal)=>goal===PREPARE_WORDS || goal===PREPARE_BOOK_WORDS; // one packet's preparation, or a book's coverage run (10.8)
  const DRAFT_CONTEXTS=16;
  const S={key:null,selector:null,pin:'',view:null,readPage:'',ledgers:new Map(),ledgerTopic:'',status:'empty',error:'',refusal:'',revision:0,bundleRev:0,busy:'',bookDate:'',
    preview:null,packet:null,dossier:null,role:'analyst',drafts:{analyst:'',cro:''},draftEdit:{analyst:0,cro:0},draftsPersisted:true,draftsNote:'',editorNote:'',lastRefusal:null,packetTask:'',packetUnit:'',finding:null,exportRequest:null,unitless:null,
    item:'',collections:{},reportCount:'',reportConclusion:'',folded:new Set(),foldKey:'',viewKey:'',viewPin:'',sourceQuery:'',sourcePage:0,issuerQuery:'',issuerState:'',issuerPage:0,checksPage:0,findingPage:0,findingOpen:new Set(),runPage:0,runsOpen:false,citePages:new Map(),packetPage:0,unitPage:0,previewPending:false,docsOpen:new Set(),storage:null,storagePending:false,exportPending:false,ledgerOpen:new Set(),ledgerQuery:'',ledgerPage:0,autoDossier:'',autoPacket:'',roleChosen:'',reading:null,excerpts:new Map(),documents:new Map(),packetDelivery:null,viewFilter:{entity:'',topic:'',days:''},spanQuery:'',spanPage:0,pending:null,work:null,unresolved:null,exportDoc:null,delivery:null,riskLinks:null,risk:'',comparison:'',question:'',nav:0,routeSig:'',readGen:0,editSeq:0,held:new Map(),legacy:null,
    // what this page itself recorded for each exact book: Tasks the owner admitted for the
    // selector it was sent, prepared Tasks whose packet the owner returned for that book, and
    // the answers this page imported (their attribution is known because the page sent them)
    bound:new Map(),taskBooks:new Map(),taskFacts:new Map()};
  const explain=explainCode;
  const key=(v)=>JSON.stringify(Object.fromEntries(Object.entries(v || {}).sort()));
  const query=(extra={})=>new URLSearchParams({...(S.selector || {}),...extra});
  /* The product's composed next requests (`view.next_requests`): each is dispatched by the
   * `operation` it names -- never by its key, which differs between projections (`packet`,
   * `packet_<unit>`, `prepare`, `dossier`, ...) -- and carries every binding field the product
   * wrote (the selector, a Task and its unit, a cutoff, a preparation binding, a review handle).
   * Three kinds: a read or a way to a page, an execution that does bounded work the person
   * asks for, and managed Provider work the product admits and the person triggers. */
  const NEXT_WORDS={EVIDENCE_PREVIEW:['read','Prepare sources'],EVIDENCE_PACKET:['read','Read the Analyst packet'],CRO_REVIEW_DOSSIER:['read','Read the CRO dossier'],CRO_REVIEW_FINDING:['read','Read the finding'],EVIDENCE_CRO_EXPORT:['read','Read the exact report'],STATUS:['read','Inspect the Task'],
    EVIDENCE_PREPARE:['run','Prepare the sources'],EVIDENCE_CONTINUE:['run','Continue reading the sources'],STORAGE_EVIDENCE_REBUILD:['run','Rebuild the evidence index'],EVIDENCE_REFRESH:['managed','Refresh evidence with the Provider'],CRO_REVIEW:['managed','Review with the Provider']};
  /* Where each operation is sent: the session's `routes` (`Data.route`, U13), the Host's one table; no map here. */
  /* A request the page dispatches is the object the product wrote, captured at the press (a
   * later paint may replace the view) and sent, confirmed or read with exactly its fields. The
   * page's state validates that it still applies -- a request that names a book names this
   * page's -- and never substitutes a field of its own for one the request carries. */
  const requestFields=(r)=>Object.fromEntries(Object.entries(r || {}).filter(([k,v])=>k!=='operation' && v!==undefined && v!==null));
  /* The four forms a book selector takes (the product's `BookSelector`). A request that names
   * a book is proved to be this page's by its form first -- the same form as the page's, then
   * every key of that form present on both and equal -- never by the keys the two happen to
   * share (a result selector and an experiment selector share none, and shared nothing is not
   * agreement). Two forms are never one book here: an alias between them (a result and the
   * experiment that produced it) is the product's to state, and the page holds no proof of
   * it, so it refuses. A request of no form names no book (a Task's status, a workspace
   * rebuild) and needs no proof; one of two forms at once is refused as none of them. */
  const SELECTOR_FORMS={result:['result_hash'],handoff:['handoff_hash'],update:['update_task_id','update_publication_hash','position_basis'],experiment:['experiment_task_id','experiment_receipt_hash','portfolio_session']};
  const named=(v,k)=>v?.[k]!==undefined && v[k]!==null && v[k]!=='';
  const selectorForm=(v)=>{ const forms=Object.keys(SELECTOR_FORMS).filter(f=>SELECTOR_FORMS[f].some(k=>named(v,k))); return forms.length===1 ? forms[0] : forms.length ? 'mixed' : ''; };
  const taskSubjects=(view)=>[...new Map((Array.isArray(view?.subjects) ? view.subjects : view?.next_requests?.subject ? [view.next_requests.subject] : []).flatMap(s=>{const f=selectorForm(s);return s.operation==='EVIDENCE_CRO' && f && f!=='mixed' && SELECTOR_FORMS[f].every(k=>named(s,k)) ? [Object.fromEntries(SELECTOR_FORMS[f].map(k=>[k,s[k]]))] : [];}).map(s=>[key(s),s])).values()];
  const appliesHere=(r)=>{ const form=selectorForm(r); if(!form) return true; return form!=='mixed' && selectorForm(S.selector)===form && SELECTOR_FORMS[form].every(k=>named(r,k) && named(S.selector,k) && String(r[k])===String(S.selector[k])); };
  const bookName=()=>S.selector?.experiment_task_id || S.selector?.update_task_id || S.selector?.result_hash || S.selector?.handoff_hash || '';
  // the object's name for the head (round G1): the book's words from History, its short handle when History has no words
  const bookTitle=()=>{ const k=S.selector ? key(S.selector) : ''; const b=k ? books().find(x=>key(x.selector)===k) : null; return b ? b.name : (S.selector ? short(bookName(),SHORT.id) : t('Evidence')); };
  const historical=()=>Boolean(S.pin);
  // a re-read of the view already shown (status `refreshing` for the same book and pin) leaves the page ready: its actions never blink disabled (the user's reading, 2026-09-22); a foreign view held painted for the hold is not ready
  const viewShown=()=>S.viewKey===S.key && S.viewPin===S.pin;
  const ready=()=>(S.status==='ready' || (S.status==='refreshing' && viewShown())) && !S.busy;
  const current=()=>ready() && !historical();
  const session=()=>S.selector?.portfolio_session || S.view?.book?.formation_session || S.preview?.book?.formation_session || S.bookDate || '';
  const isExperimentBook=()=>Boolean(S.selector?.experiment_task_id);
  const isUpdateBook=()=>selectorForm(S.selector)==='update';
  const workspaceId=()=>(typeof Data.workspace==='function' ? Data.workspace() : '') || '';
  const pct=pctFraction; // the shared percentage rule (components.js); owner strings go through pctText
  /* The saved review publications of this exact book, from history, newest first. */
  const publications=()=>Data.history().filter(v=>v.raw?.review_publication_hash && key(v.raw.book)===S.key).map(v=>({hash:v.raw.review_publication_hash,recorded:String(v.raw.recorded_at || v.recordedAt || ''),lapsed:v.raw.status==='HISTORICAL_REVIEW'})).sort((a,b)=>b.recorded.localeCompare(a.recorded));
  /* The working review's publication is the identity the owner names in its projection (for
   * every subject kind); when the projection names none, none is claimed. */
  const workingPublication=()=>S.view?.review_publication_hash || '';
  /* This page's own recorded evidence for the exact book (session-bound), with the evidence
   * identity each Task was established under (the publication Task Control names, the packet's
   * context and cutoff, the dossier's analysis): a Task of another identity is history. */
  const boundFor=(k=S.key)=>{if(!S.bound.has(k))S.bound.set(k,{prepare:new Set(),analysis:new Set(),review:new Set(),imported:new Set()});return S.bound.get(k);};
  const factsOf=(task)=>S.taskFacts.get(task) || {};
  const noteFacts=(task,facts)=>{if(!task) return;const known=S.taskFacts.get(task) || {};for(const [k,v] of Object.entries(facts || {})) if(v) known[k]=v;S.taskFacts.set(task,known);};
  const bind=(group,task,k=S.key,selector=(k===S.key ? S.selector : null),facts=null)=>{if(!task || !k) return;boundFor(k)[group].add(task);if(selector)S.taskBooks.set(task,{...selector});if(facts)noteFacts(task,facts);};
  const artifactHash=(view,kind)=>(view?.artifact_refs || []).map(r=>String(r).match(new RegExp(kind+'/([^/?#]+)'))?.[1]).find(Boolean) || '';
  // what Task Control's recovery view says a Task published
  const artifactFacts=(view)=>({publication:artifactHash(view,'cro_review_publication'),analysis:artifactHash(view,'alternative_evidence_analysis_publication'),request:artifactHash(view,'alternative_evidence_request'),asOf:view?.subject_context?.evidence_as_of || ''});
  /* The current evidence identity the projection (and a dossier read for this book) names: the
   * review publication, the selected analysis publication, the evidence as-of. */
  // a bundle read under the projection read the page is showing: its context is this read's;
  // one read earlier is retained content that establishes nothing about the refreshed context
  const inThisRead=(bundle)=>Boolean(bundle) && bundle.key===S.key && bundle.readGen===S.readGen;
  const continuationRequired=()=>!historical() && (S.view?.required_actions || []).some(a=>a.action==='SETTLE_EVIDENCE_CONTINUATION' && a.blocking);
  const requiredActions=()=> (S.view?.required_actions || []).filter(a=>a.action!=='NONE');
  // NONE carries the review's limits; it asks for no decision or next action.
  const actionStanding=a=>a.action==='NONE' ? {state:null,tone:TONE.rest,word:codeWords(a.action)} : {state:a.blocking ? 'blocked' : 'deferred',tone:TONE.attention,word:t(a.blocking ? 'Blocking' : 'Advisory')};
  const bundleOf=(role)=>{if(role==='cro' && continuationRequired()) return null;const x=role==='analyst' ? S.packet : S.dossier;return x && x.key===S.key ? x.value : null;};
  const dossierOf=()=>bundleOf('cro')?.dossier || null;
  function currentIdentity() {
    const v=S.view || {}, d=inThisRead(S.dossier) ? S.dossier.value?.dossier : null;
    const selected=(v.eligible_versions || []).find(x=>x.is_selected)?.analysis_publication_hash || '';
    return {publication:v.review_publication_hash || '',analysis:selected || (d ? d.analysis_publication_hash || '' : ''),asOf:v.evidence_as_of || '',unique:v.evidence_selection==='UNIQUE_CURRENT'};
  }
  /* A bound Task's currency: 'current' only on an exact match with an identity this read names,
   * 'historical' only on an established mismatch, else 'unresolved'; a lifecycle is never a
   * version fact and a timestamp alone can only refute. */
  // two owner timestamps naming one instant, whatever their offset spelling
  const sameInstant=(a,b)=>a===b || (Boolean(a) && Boolean(b) && Number.isFinite(Date.parse(a)) && Date.parse(a)===Date.parse(b));
  // states in which the owner says no review is published for the current version, and the one
  // in which it says no current analysis exists at all
  const NO_REVIEW=new Set(['ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW','AWAITING_ALTERNATIVE_EVIDENCE','MODEL_AUTHORITY_NOT_ADMITTED','ALTERNATIVE_EVIDENCE_EXPIRED','EVIDENCE_SELECTION_AMBIGUOUS','EVIDENCE_REFRESH_IN_PROGRESS','REVIEW_INPUT_INCOMPLETE']);
  function currency(group,task) {
    const f=factsOf(task), now=currentIdentity(), state=S.view?.state || '';
    if(group==='review'){
      if(!f.publication) return 'unresolved';
      if(now.publication) return f.publication===now.publication ? 'current' : 'historical';
      return NO_REVIEW.has(state) ? 'historical' : 'unresolved'; // the read says this version has no review: the Task's is an earlier one
    }
    if(group==='analysis'){
      if(f.analysis && now.analysis) return f.analysis===now.analysis ? 'current' : 'historical';
      if(f.analysis && state==='AWAITING_ALTERNATIVE_EVIDENCE' && !continuationRequired()) return 'historical'; // the read says no current analysis exists; an exact pending continuation says otherwise
      if(f.asOf && now.asOf && !sameInstant(f.asOf,now.asOf)) return 'historical';
      return 'unresolved';
    }
    // a prepared packet: the book's current preparation while the read names no analysis at all;
    // an older cutoff than the read's is history; the same cutoff proves nothing by itself
    if(f.asOf && now.asOf && !sameInstant(f.asOf,now.asOf)) return 'historical';
    return now.asOf ? 'unresolved' : 'current';
  }
  const taskOf=(id)=>(Data.tasks() || []).find(v=>v.task_id===id) || null;
  const taskState=(id)=>{const w=S.work?.task===id ? S.work.view : null;return w ? {lifecycle:w.lifecycle,verified:w.verified_stage_count,total:w.total_stage_count} : (()=>{const p=taskOf(id);return p ? {lifecycle:p.lifecycle,verified:p.verified_stage_count,total:p.total_stage_count} : {lifecycle:'',verified: '',total: ''};})();};
  /* Workspace discovery: the evidence and review Tasks Task Control lists for the whole
   * workspace -- offered for binding, never counted as this book's progress. */
  const evidenceTasks=()=>(Data.tasks() || []).filter(v=>v.task_kind===EVIDENCE_KIND);
  const prepareTasks=()=>evidenceTasks().filter(v=>v.total_stage_count===6 || preparing(v.goal_summary));
  function routeContext() {return S.selector ? {review_selector:JSON.stringify(S.selector),review_publication:S.pin,prepared_task:S.packetTask,prepared_unit:S.packetTask ? S.packetUnit : '',view_entity_id:S.viewFilter.entity,view_topic:S.viewFilter.topic,view_last_days:S.viewFilter.days,work:S.work?.task || '',review_lists:Object.keys(S.collections).length ? JSON.stringify(S.collections) : '',report_count:S.reportCount,report_conclusion:S.reportConclusion} : {work:S.work?.task || ''};}
  // the route rewritten in place: every review key is stated, so a cleared one leaves the route
  const routeUpdate=(extra={})=>({review_selector:'',review_publication:'',prepared_task:'',prepared_unit:'',view_entity_id:'',view_topic:'',view_last_days:'',work:'',review_lists:'',report_count:'',report_conclusion:'',...routeContext(),...extra});
  /* The route (page and review keys) is remembered whenever this module writes it; a render
   * whose route differs was navigated by the reader and supersedes every pending intent. */
  const routeSignature=()=>{const q=hashParams();return ['page','review_selector','review_publication','prepared_task','prepared_unit','view_entity_id','view_topic','view_last_days','work','review_lists','report_count','report_conclusion'].map(k=>q.get(k) || '').join('\u0001');};
  const repaintMain=()=>(typeof patchMain==='function' ? patchMain : render)(); // a read's start and finish repaint by the diff: the page never shakes (the user's reading, 2026-09-22)
  const writeRoute=(update)=>{replaceHash(update);S.routeSig=routeSignature();};
  /* A choice that changes what the page holds is a level: the address is pushed, so `<-` returns to the one before it (law 86, the way back). A view's filter or a search is not a level (writeRoute). Without a history (the harness) it is a replace. */
  const pushRoute=(update)=>{ if(typeof pushHash!=='function'){writeRoute(update);return;} pushHash(update); S.routeSig=routeSignature(); }; // the router owns the entry and its depth
  function leaveReads(page) {
    if(!pages.has(page))return;
    S.nav++;S.bundleRev++;S.revision++; // every departed page's auxiliary reads lose their state-writing generation
    if(S.readPage===page || !pages.has(app.page) || app.page==='books') {
      S.readPage='';
      if(S.status==='loading' || S.status==='refreshing' || !pages.has(app.page) || app.page==='books')S.status='empty';
    }
    S.previewPending=false;S.storagePending=false;S.exportPending=false;S.retainedPending=false;
    if(S.work)S.work.reading=false;
    S.ledgerSig='';if(S.ledgerRead?.status==='reading')S.ledgerRead=null;if(S.reading?.status==='reading')S.reading=null;
    if(['part','bundle','read','finding','sources','export','links'].includes(S.busy))S.busy='';
    for(const [k,v] of S.ledgers)if(v.status==='reading')S.ledgers.delete(k);
  }
  function observeRoute() {
    const sig=routeSignature(); if(sig===S.routeSig) return false;
    S.routeSig=sig; S.nav+=1; return true;
  }
  function reset() {S.collections={};S.reportCount='';S.reportConclusion='';S.previewPending=false;S.storagePending=false;S.exportPending=false;S.retainedPending=false;S.pending=null; S.retained=null; S.revision++; S.bundleRev++; S.packet=null; S.dossier=null; S.exportDoc=null; S.exportRequest=null; S.finding=null; S.roleChosen=''; S.autoDossier=''; S.autoPacket=''; S.ledgers=new Map(); S.ledgerOpen=new Set(); S.ledgerQuery=''; S.ledgerPage=0; S.sourceQuery=''; S.sourcePage=0; S.issuerQuery=''; S.issuerState=''; S.issuerPage=0; S.docsOpen=new Set(); S.reading=null; S.excerpts=new Map(); S.documents=new Map(); S.packetDelivery=null; S.spanQuery=''; S.spanPage=0; S.preview=null; S.delivery=null; S.deliveryAttempt=''; S.deliveryRefusal=null; S.question=''; S.riskLinks=null; S.risk=''; S.comparison=''; S.lastRefusal=null; S.refusal='';}
  /* ---- drafts, kept under the exact workspace/book/prepared-Task(/unit) context and role ---- */
  const draftContext=(unit=S.packetUnit)=>workspaceId()+'|'+S.key+'|'+S.packetTask+(unit ? '|'+unit : '');
  /* The viewer store: contexts keyed by workspace, book and prepared Task, plus a legacy
   * single-slot draft ({key, analyst, cro}) that named no workspace and is carried untouched
   * until the reader recovers, exports or discards it. */
  const isLegacy=(saved)=>Boolean(saved) && typeof saved==='object' && typeof saved.key==='string' && !saved.contexts;
  function draftStore() {
    const saved=readPreference('reviewDrafts');
    if(saved && saved.contexts && typeof saved.contexts==='object') return {contexts:saved.contexts,legacy:saved.legacy && typeof saved.legacy==='object' && typeof saved.legacy.key==='string' ? saved.legacy : null};
    if(isLegacy(saved)) return {contexts:{},legacy:{key:saved.key,analyst:typeof saved.analyst==='string' ? saved.analyst : '',cro:typeof saved.cro==='string' ? saved.cro : ''}};
    return {contexts:{},legacy:null};
  }
  const writeStore=(store)=>{savePreference('reviewDrafts',{contexts:store.contexts,...(store.legacy ? {legacy:store.legacy} : {})});return readPreference('reviewDrafts');};
  // the legacy draft that names this book and prepared Task (it cannot name the workspace)
  const legacyFor=(store)=>store.legacy && (store.legacy.analyst || store.legacy.cro) && store.legacy.key===S.key+'|'+S.packetTask ? store.legacy : null;
  // the draft kept for this Task before its unit was named (the context without a unit): shown
  // beside a unit's editor as recoverable, never adopted as that unit's own
  const unitlessFor=(store)=>{ if(!S.packetUnit) return null; const entry=store.contexts[draftContext('')]; return entry && (entry.analyst || entry.cro) ? {id:draftContext(''),analyst:entry.analyst || '',cro:entry.cro || ''} : null; };
  /* The context's drafts: unsent text this page could not persist is held in memory for the
   * session and comes back before anything the store says; a transition never drops it. */
  function loadDrafts() {
    S.importRefusal=null; // a refused import belonged to the context it was made in
    const id=draftContext(), store=draftStore(), entry=store.contexts[id], held=S.held.get(id);
    S.drafts={analyst:held ? held.analyst : (entry?.analyst || ''),cro:held ? held.cro : (entry?.cro || '')};S.draftEdit={analyst:++S.editSeq,cro:++S.editSeq};
    S.draftsPersisted=!held;S.draftsNote=held ? held.note : '';S.editorNote='';S.legacy=legacyFor(store);S.unitless=unitlessFor(store);
  }
  const hold=(id,note)=>{S.held.set(id,{analyst:S.drafts.analyst,cro:S.drafts.cro,note});S.draftsPersisted=false;S.draftsNote=note;};
  /* Persist the context's drafts: only empty contexts are evicted; when none can be, or the
   * store does not read back, the text stays held here and the editor says so. */
  function saveDrafts() {
    const id=draftContext(), store=draftStore(), contexts=store.contexts;
    const dirty=S.drafts.analyst || S.drafts.cro;
    if(!dirty){
      delete contexts[id];S.held.delete(id);
      const back=writeStore(store);
      S.draftsPersisted=!(back && back.contexts && back.contexts[id]);
      S.draftsNote=S.draftsPersisted ? '' : t('The cleared draft is still in the viewer store: it could not be removed there.');
      return;
    }
    if(!contexts[id] && Object.keys(contexts).length>=DRAFT_CONTEXTS){
      const empty=Object.entries(contexts).filter(([,v])=>!v.analyst && !v.cro).sort((a,b)=>(a[1].at || 0)-(b[1].at || 0))[0];
      if(empty) delete contexts[empty[0]];
      else { hold(id,t('This draft is kept in this page only: {n} other contexts still hold unsent text. Clear or export one of them to persist this one.',{n:Object.keys(contexts).length})); return; }
    }
    contexts[id]={analyst:S.drafts.analyst,cro:S.drafts.cro,at:Date.now()};
    const back=writeStore(store)?.contexts?.[id];
    if(back && back.analyst===S.drafts.analyst && back.cro===S.drafts.cro){S.held.delete(id);S.draftsPersisted=true;S.draftsNote='';}
    else hold(id,t('This draft is kept in this page only: the viewer store could not persist it. Export or copy the text before leaving.'));
  }
  /* The reader's decision on a legacy draft: recovered into this context's empty fields (typed
   * text is kept), exported, or discarded; nothing else removes it. */
  function settleLegacy(decision) {
    const store=draftStore(), legacy=legacyFor(store); if(!legacy) return;
    if(decision==='export'){download(JSON.stringify(legacy,null,2),'AlphaLattice-unsent-draft.json','application/json');return;}
    if(decision==='recover'){
      const kept=[];
      for(const role of ['analyst','cro']){ if(!legacy[role]) continue; if(S.drafts[role]){kept.push(role);continue;} S.drafts[role]=legacy[role];S.draftEdit[role]=++S.editSeq; }
      if(kept.length){store.legacy={...legacy,...Object.fromEntries(['analyst','cro'].filter(r=>!kept.includes(r)).map(r=>[r,'']))};notify(t('The {roles} text you already typed here was kept; the legacy text for it stays until you clear this draft or discard it.',{roles:kept.join(', ')}));}
      else store.legacy=null;
      writeStore(store);saveDrafts();S.legacy=legacyFor(draftStore());render();return;
    }
    store.legacy=null;writeStore(store);S.legacy=legacyFor(draftStore());render();
  }
  /* The reader's decision on a draft kept for this Task with no unit named: recovered into this
   * unit's empty fields (typed text is kept), exported, or discarded; it is never assigned to a
   * unit by the page. */
  function settleUnitless(decision) {
    const store=draftStore(), found=unitlessFor(store); if(!found) return;
    if(decision==='export'){download(JSON.stringify({key:S.key+'|'+S.packetTask,analyst:found.analyst,cro:found.cro},null,2),'AlphaLattice-unsent-draft.json','application/json');return;}
    const entry=store.contexts[found.id];
    if(decision==='recover'){
      const kept=[];
      for(const role of ['analyst','cro']){ if(!found[role]) continue; if(S.drafts[role]){kept.push(role);continue;} S.drafts[role]=found[role];S.draftEdit[role]=++S.editSeq;entry[role]=''; }
      if(kept.length) notify(t('The {roles} text you already typed here was kept; the unit-less text for it stays until you clear this draft or discard it.',{roles:kept.join(', ')}));
      if(!entry.analyst && !entry.cro) delete store.contexts[found.id];
      writeStore(store);saveDrafts();S.unitless=unitlessFor(draftStore());render();return;
    }
    delete store.contexts[found.id];writeStore(store);S.unitless=unitlessFor(draftStore());render();
  }
  const unsentHeld=()=>[...S.held.values()].some(h=>h.analyst || h.cro);
  if(typeof window!=='undefined' && typeof window.addEventListener==='function') window.addEventListener('beforeunload',(e)=>{ if(unsentHeld()){e.preventDefault();e.returnValue='';} });
  /* Open one exact book (or the owner's default book when none is named yet), working or
   * pinned to one historical publication; the page named stays the page shown. */
  /* The same book opened under another reading (working or a pinned publication) keeps what the page painted -- the view, the report, the bundles -- until their successors arrive; only what belongs to the reading itself (a pending confirmation, a composed request, a refusal, the auto-reads' marks) is cleared. Another book resets the page. */
  function softReset() {S.deliveryAttempt='';S.deliveryRefusal=null;S.previewPending=false;S.storagePending=false;S.exportPending=false;S.retainedPending=false;S.pending=null; S.revision++; S.bundleRev++; S.exportRequest=null; S.finding=null; S.roleChosen=''; S.autoDossier=''; S.autoPacket=''; S.ledgerPage=0; S.spanQuery=''; S.spanPage=0; S.sourcePage=0; S.issuerPage=0; S.findingPage=0; S.findingOpen=new Set(); S.runPage=0; S.citePages=new Map(); S.packetPage=0; S.unitPage=0; S.riskLinks=null; S.risk=''; S.comparison=''; S.lastRefusal=null; S.refusal='';}
  async function open(selector,pin='',page='evidence') {
    S.nav+=1; closeDialog(); if(S.selector && key(selector)===S.key) softReset(); else reset();
    S.unresolved=null; if(key(selector)!==S.key)S.bookDate='';S.selector=selector ? {...selector} : null;S.pin=pin || '';S.key=key(selector);
    S.packetTask='';S.packetUnit='';S.work=null;S.item='';S.folded=new Set();app.page=pages.has(page) ? page : 'evidence';loadDrafts();if(selector)savePreference('reviewLastBook',selector);
    writeRoute(routeUpdate({page:app.page}));return refresh();
  }
  async function refresh() {
    if(!S.selector && S.unresolved) return; // an unresolved Task names no book; the owner's default is not guessed for it
    S.readPage=app.page;S.previewPending=false;S.storagePending=false;S.exportPending=false;S.retainedPending=false;S.pending=null;S.error='';S.refusal='';S.busy='';
    // the shown view stays painted while the next one loads (status `refreshing`: no read starts against it); a skeleton only when the owner takes longer than 400 ms
    const stale=Boolean(S.view), sameBook=stale && S.viewKey===S.key; if(stale)S.status='refreshing'; else S.status='loading';
    // the same book under another reading (a pin) keeps its page painted until the next view arrives -- the lane dims, nothing collapses (the user's reading, 2026-09-22); another book's view holds 400 ms, then the skeleton
    repaintMain();
    // The first paint establishes the destination visit and invalidates the departed page before this read owns a ticket.
    const ticket=++S.revision;let navigation=Data.navigationIntent();
    const hold=stale && !sameBook && typeof setTimeout==='function' ? setTimeout(()=>{ if(ticket===S.revision && S.status==='refreshing'){S.status='loading';S.view=null;repaintMain();} },400) : null;
    try {
      const view=await Data.read('/api/evidence-cro?'+query(S.pin ? {review_publication_hash:S.pin} : {}));
      if(hold!==null)clearTimeout(hold);
      if(ticket!==S.revision) return;
      // an identical projection is the same read: what was read under it (the dossier, the packet) stays bound; a changed one is a new read generation
      if(!S.view || JSON.stringify(view)!==JSON.stringify(S.view))S.readGen+=1;
      S.view=view;S.status='ready';const addressed=Data.navigationCurrent(navigation);adoptPacketRequest(view,addressed);if(addressed)navigation=Data.navigationIntent();if(view.book?.formation_session)S.bookDate=view.book.formation_session;
      // law 123: a book page with no book chosen and none the owner defaults to opens Books, the Home
      if(view.state==='NO_BOOK_TO_REVIEW' && !S.selector && Data.navigationCurrent(navigation) && pages.has(app.page) && app.page!=='books'){app.page='books';replaceHash({page:'books'});render();return;}
      // a cold link names the object read in the inspector (`review_item`): it opens once the view is here
      const linked=hashParams().get('review_item'); if(linked && Window.inspectorMode()!=='evidence') queueMicrotask(()=>{ if(S.view && ticket===S.revision && Data.navigationCurrent(navigation)) setItemQuiet(linked); });
      // the owner's default book becomes the exact selector of this page
      if(!S.selector && view.book) { const b=view.book; S.selector=b.experiment_subject ? {experiment_task_id:b.experiment_subject.experiment_task_id,experiment_receipt_hash:b.experiment_subject.experiment_receipt_hash,portfolio_session:b.experiment_subject.portfolio_session} : b.update_subject ? {update_task_id:b.update_subject.update_task_id,update_publication_hash:b.update_subject.update_publication_hash,position_basis:b.update_subject.position_basis} : b.result_hash ? {result_hash:b.result_hash} : null; S.key=key(S.selector); loadDrafts(); if(S.selector)savePreference('reviewLastBook',S.selector); if(Data.navigationCurrent(navigation)){writeRoute(routeUpdate());navigation=Data.navigationIntent();} }
      // after the selector's adoption (S.key moves there): the ledgers of every packet the projection names, and the issuers folded by default -- the overview is a summary, the items open on demand (round E2)
      if(app.page==='evidence') void readLedgers();
      if(S.foldKey!==S.key){S.foldKey=S.key;S.folded=new Set((view.issuer_rows || []).map(r=>r.entity_id));}
      S.viewKey=S.key;S.viewPin=S.pin; // the painted view is this book's and pin's
      if(Data.navigationCurrent(navigation))followBookTask();
      // a report painted before this read stays until its successor arrives (`ensure` reads the publication this view names); a delivery composed against another publication is gone
      const named=S.pin || workingPublication(); if(S.delivery && (S.delivery.review || '')!==named)S.delivery=null;
    } catch(e) {if(hold!==null)clearTimeout(hold);if(ticket===S.revision){S.view=null;S.viewKey='';S.viewPin='';S.error=e.message;S.refusal=explain(e.message) && !transport(e.message) ? e.message : '';S.status=S.refusal ? 'ready' : 'error';}}
    if(ticket===S.revision) repaintMain();
  }
  /* The route is the reader's exact selection (book, pin, prepared Task, followed Task); a
   * change to any is adopted, after an open only while its book and pin are still current. */
  function ensure() {
    if(!pages.has(app.page) || app.page==='books' || Data.workspaceStatus!=='ready') return;
    const h=hashParams();
    let selector=null;
    try { selector=h.get('review_selector') ? JSON.parse(h.get('review_selector')) : Data.raw()?.reviewSelector || null; }
    catch { if(S.error!=='Invalid exact review selector'){S.status='error';S.error='Invalid exact review selector';render();}return; }
    const restoreLists=()=>{let lists={};try{const value=JSON.parse(h.get('review_lists') || '{}');for(const [k,v] of Object.entries(value))if(v && typeof v==='object')lists[k]={query:String(v.query || ''),page:Math.max(0,Math.floor(Number(v.page) || 0))};}catch{}const changed=JSON.stringify(lists)!==JSON.stringify(S.collections) || (h.get('report_count') || '')!==S.reportCount || (h.get('report_conclusion') || '')!==S.reportConclusion;S.collections=lists;S.reportCount=h.get('report_count') || '';S.reportConclusion=h.get('report_conclusion') || '';return changed;};
    const pin=h.get('review_publication') || '', task=h.get('prepared_task') || '', unit=task ? h.get('prepared_unit') || '' : '', work=h.get('work') || '';
    const vf={entity:h.get('view_entity_id') || '',topic:h.get('view_topic') || '',days:h.get('view_last_days') || ''}, viewChanged=JSON.stringify(vf)!==JSON.stringify(S.viewFilter); if(viewChanged){S.viewFilter=vf;S.reading=null;S.spanPage=0;} // the view's keys (round E3)
    const item=h.get('review_item') || ''; if(S.status==='ready' && item!==S.item){ if(item) queueMicrotask(()=>{ if(S.status==='ready') setItemQuiet(item); }); else {S.item='';if(typeof Window!=='undefined')Window.clearDetailTrail?.();if(typeof patchMain==='function')patchMain();} } // the way back: the address without the item closes the pane
    if((selector && (key(selector)!==S.key || pin!==S.pin || S.status==='empty')) || (!selector && S.status==='empty' && !S.unresolved)) {
      const page=app.page, opened=open(selector,pin,page), gen=S.nav; // the generation this open holds
      opened.then(()=>{
        if(gen!==S.nav) return; // the reader moved on; nothing of this open is adopted
        if(task){S.packetTask=task;S.packetUnit=unit;loadDrafts();}
        restoreLists();
        writeRoute(routeUpdate());
        if(work)watch(work,true);
        render();
      });
      return;
    }
    if(S.selector && (task!==S.packetTask || unit!==S.packetUnit) && S.status==='ready'){S.packetTask=task;S.packetUnit=unit;S.packet=null;S.pending=null;S.reading=null;S.spanPage=0;loadDrafts();S.routeSig=routeSignature();render();}
    if(restoreLists() || viewChanged) repaintMain(); // Back/Forward may change only the view keys while the same cell stays open.
    if(work && work!==(S.work?.task || '') && !S.work?.reading){void watch(work,true);}
    if(ready() && app.page==='report' && isUpdateBook() && S.deliveryAttempt!==S.key+'|'+(S.pin || workingPublication())) void assembleDelivery(true);
    if(ready() && pages.has(app.page) && !S.preview && !S.previewPending && !historical() && !S.refusal) inspectSources(null,app.page!=='evidence-stream'); // every page's rail reads the store's state (round F1); one read at a time, or the patch -> ensure -> read chain never ends
    if(ready() && (app.page==='report' || (app.page==='handoff' && (reviewPublished() || historical()))) && !S.error && (!S.exportDoc || S.exportDoc.hash!==(S.pin || workingPublication())) && (workingPublication() || S.pin)) readExport(null,app.page!=='report'); // a painted report of another reading stays until this one's arrives; the Review page reads the published dispositions from it (round F4)
  }
  /* ---- the exact book, its date, its evidence version and the review the page reads ---- */
  function books() {
    const seen=new Map();
    // N3 (law 134): a book's name is its History name's words (the policy it declared, the package's words), the same as in Studies
    const named=(sel,name)=>({selector:sel,name,ref:short(sel.experiment_task_id || sel.update_task_id || sel.result_hash || sel.handoff_hash,SHORT.id)});
    for(const row of Data.history()) if(row.raw?.book && !seen.has(key(row.raw.book))) seen.set(key(row.raw.book),named(row.raw.book,row.words || t(row.kind))); // round H1 (R4): the package and the hash; the session is the head's chip
    if(S.selector && !seen.has(S.key)) seen.set(S.key,named(S.selector,isExperimentBook() ? t('authored book') : t('installed result')));
    return [...seen.values()];
  }
  /* The book is chosen on every review page; a historical publication is pinned where it is
   * read (the desk and the report), and shown on the other pages only while one is pinned. */
  /* E1: the book is chosen from the name's switch (GitHub's branch picker); the reading (the
   * current one or a pinned publication) stays a chooser on the context line. */
  const bookChoices=()=>{const all=books(),admitted=taskSubjects(S.unresolved?.view),choices=admitted.length ? admitted.map(selector=>all.find(b=>key(b.selector)===key(selector)) || {selector,name:t('Book'),ref:short(selector.experiment_task_id || selector.update_task_id || selector.result_hash || selector.handoff_hash,SHORT.id)}) : all;return [['',t('Choose a saved book')],...choices.map(v=>({value:JSON.stringify(v.selector),title:v.name,meta:v.ref}))];};
  function bookSwitch() { return picker('reviewBook',bookChoices(),{selected:S.selector ? JSON.stringify(S.selector) : '',kind:'switch',label:t('Choose the book')}); }
  function readingChooser() {
    const pubs=publications(); // B1: the reading is the object's, the context line's last fact on every tab -- the head does not change between tabs
    const note=t(pubs.length ? 'A pinned publication is an exact historical readback; preparing newer evidence never rewrites it.' : 'No review publication is recorded for this book yet.');
    return subjectChoice(t('Publication'),'reviewSavedPublication',[['',t('Current')],...pubs.map(v=>[v.hash,t('Pinned {hash} · {date}',{hash:short(v.hash, SHORT.hash),date:String(v.recorded).slice(0,10)})])],{selected:S.pin,disabled:!S.selector,attrs:html`title="${note}"`});
  }
  function selectorBar() {
    const pubs=publications(), pinnable=['evidence','report'].includes(app.page) || Boolean(S.pin);
    // round 32: one line of words — the book and the publication are text controls (a field box
    // only when hovered or focused); what a pin means is the control's title, said once on the desk
    const note=S.selector ? t(pubs.length ? 'A pinned publication is an exact historical readback; preparing newer evidence never rewrites it.' : 'No review publication is recorded for this book yet.') : t('Choose a book first; its publications are listed here.');
    const publication=pinnable ? subjectChoice(t('Publication'),'reviewSavedPublication',[['',t('Current')],...pubs.map(v=>[v.hash,t('Pinned {hash} · {date}',{hash:short(v.hash, SHORT.hash),date:String(v.recorded).slice(0,10)})])],{selected:S.pin,disabled:!S.selector,attrs:html`title="${note}"`}) : '';
    return html`${subjectChoice(t('Book'),'reviewBook',bookChoices(),{selected:S.selector ? JSON.stringify(S.selector) : ''})}${publication}`; // round 93: the header's subject line
  }
  function stateBanner(always=true) {
    if(S.refusal) return notRead(t('Evidence review is not available for this workspace'),S.refusal,explain(S.refusal));
    const v=S.view; if(!v) return '';
    if(!always && (!v.task_id || !stateMoving(v.task_lifecycle))) return ''; // the desk's facts and layers say the state; a banner only while a Task can be followed (round E2)
    const a=stateOf(v.state), title=a.word, text=a.line || '', tone=a.tone==='good' ? 'ok' : a.tone;
    const words=v.state==='REVIEW_PUBLISHED' ? html`${t(text)} ${t('Disposition')}: <span class="review-strong">${coded(v.disposition)}</span> · ${t('completeness')}: <span class="review-strong">${coded(v.review_state)}</span>. ${t('A published review is a Host-routed recommendation, not permission to trade.')}` : t(text);
    return noteLine(t(title),html`${words}${v.explanation ? html` ${said(v.explanation)}` : ''}${v.task_id ? html` · ${t('Task')} <span class="mono">${short(v.task_id)}</span>${v.task_lifecycle ? html` · ${codeWords(v.task_lifecycle)}` : ''}` : ''}`,tone==='ok' ? 'ok' : tone==='warning' ? 'warning' : 'neutral',v.task_id && S.work?.task!==v.task_id ? btn(t('Follow this Task'),'review-watch',v.task_id,'button compact') : '');
  }
  /* The five steps. Each status comes from this exact book and evidence version: the owner's
   * projection, the packet or dossier the owner returned for it, the Tasks admitted for it from
   * this page; unbound Tasks are discovery; a pinned, expired or ambiguous projection borrows
   * no later work. */
  function stepRows() {
    const v=S.view, refusal=S.lastRefusal, b=boundFor(), published=historical() || v?.state==='REVIEW_PUBLISHED', blocked=blockedState();
    const day=(s)=>dayWord(s) || ''; // the rail says the day (the reader's, law 133); the facts say the time
    // this book's bound Tasks, each with its version currency against the projection: only a
    // current one is progress; a historical one is named as history; an unresolved one is not
    // promoted either way
    const owned=(group,set)=>[...set].map(id=>({id,...taskState(id),currency:currency(group,id)}));
    const running=(list)=>list.find(x=>stateMoving(x.lifecycle)), succeeded=(list)=>list.filter(x=>x.lifecycle==='SUCCEEDED' && x.currency==='current').at(-1);
    const history=(list)=>list.filter(x=>x.lifecycle==='SUCCEEDED' && x.currency==='historical'), open=(list)=>list.filter(x=>x.lifecycle==='SUCCEEDED' && x.currency==='unresolved');
    const prepared=owned('prepare',b.prepare), analysed=owned('analysis',b.analysis), reviewed=owned('review',b.review);
    // a packet read speaks for the read it was made under, or for a read that names no version yet
    const packetBound=Boolean(S.packet) && S.packet.key===S.key && S.packet.task===S.packetTask && currency('prepare',S.packetTask)!=='historical' && (inThisRead(S.packet) || !v?.evidence_as_of);
    // a dossier proves a current analysis only when it was read under this projection read (or
    // names the analysis this read selects); a dossier from an earlier read is retained content
    const dossierHere=dossierOf();
    const dossierBound=Boolean(dossierHere) && (inThisRead(S.dossier) || (currentIdentity().analysis && currentIdentity().analysis===dossierHere.analysis_publication_hash));
    const dossierRetained=Boolean(dossierHere) && !dossierBound;
    const analysisAsOf=v?.evidence_as_of || (dossierBound ? dossierHere.evidence_as_of : '') || '';
    const analysisCurrent=dossierBound || ['ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW','REVIEW_PUBLISHED'].includes(v?.state); // an evidence as-of alone is the packet's stamp, not an analysis (round E2)
    const attribution=attributionWords(v?.review_attribution || []);
    // what an earlier version's Task is to this one (readable history), and what a Task whose
    // version no owner fact settles against this read is (open, neither current nor history)
    const past=(list,words)=>{const h=history(list), u=open(list);return (h.length ? ' · '+t(words,{task:short(h.at(-1).id),n:h.length}) : '')+(u.length ? ' · '+t('Task {task}: version not established against this read',{task:short(u.at(-1).id)}) : '');};
    let rows;
    if(blocked){
      const unavailable=['todo',t('not available: no evidence authority is admitted for this workspace')];
      rows=[['sources',unavailable],['analyst',unavailable],['validation',unavailable],['cro',unavailable],['published',['todo',t('no review published for this book')]]];
    } else if(historical()){
      const recorded=['done',t('as recorded in this publication · evidence as-of {date}',{date:day(v?.evidence_as_of)})];
      rows=[['sources',recorded],['analyst',recorded],['validation',recorded],['cro',['done',attribution || t('as recorded in this publication')]],['published',['done',t('pinned historical publication')]]];
    } else if(v?.state==='ALTERNATIVE_EVIDENCE_EXPIRED' || v?.state==='EVIDENCE_SELECTION_AMBIGUOUS'){
      const why=v.state==='ALTERNATIVE_EVIDENCE_EXPIRED' ? ['refused',t('the selected analysis expired at {date}; newer evidence is needed',{date:day(v.evidence_expires_at)})] : ['waiting',t('{n} eligible analyses; choose the exact one to review against',{n:(v.eligible_versions || []).length})];
      rows=[['sources',why],['analyst',why],['validation',why],['cro',['todo',t('needs the chosen current analysis')]],['published',['todo',t('no current review for this evidence version')]]];
    } else {
      const preparing=running(prepared), preparedDone=succeeded(prepared), validating=running(analysed), analysedDone=succeeded(analysed), publishing=running(reviewed), reviewedDone=succeeded(reviewed);
      const olderPrepared=past(prepared,'Task {task} prepared an earlier version (history)'), olderAnalysed=past(analysed,'Task {task} published an earlier analysis (history)'), olderReviewed=past(reviewed,'Task {task} assessed an earlier version (history, not this one)');
      rows=[
        ['sources',preparing ? ['running',t('preparing · Task {task} · {n} / {m} stages',{task:short(preparing.id),n:preparing.verified,m:preparing.total})] : packetBound ? ['done',t('prepared · Task {task} · packet read for this book',{task:short(S.packetTask)})+olderPrepared] : preparedDone ? ['done',t('prepared · Task {task} · bound to this book by its owner',{task:short(preparedDone.id)})+olderPrepared] : analysisCurrent ? ['done',t('prepared · as the published analysis records · evidence as-of {date}',{date:day(analysisAsOf)})+olderPrepared] : ['todo',t('no prepared packet is bound to this book on this page')+olderPrepared]],
        ['analyst',refusal?.role==='analyst' && refusal.kind==='refused' ? ['refused',t('last answer refused · {code}',{code:refusal.code})] : refusal?.role==='analyst' && refusal.kind==='correct' ? ['waiting',t('last answer returned for correction · {code}',{code:refusal.code})] : refusal?.role==='analyst' ? ['waiting',t('last submission outcome uncertain; reconcile before submitting again')] : analysedDone ? ['done',(b.imported.has(analysedDone.id) ? t('accepted · imported here as HUMAN · Task {task}',{task:short(analysedDone.id)}) : t('accepted · Task {task}',{task:short(analysedDone.id)}))+olderAnalysed] : validating ? ['running',t('being validated · Task {task}',{task:short(validating.id)})] : analysisCurrent ? ['done',t('analysis published · as-of {date}',{date:day(analysisAsOf)})+olderAnalysed] : packetBound ? ['waiting',t('packet read; awaiting the external answer')+olderAnalysed] : ['todo',t('needs a packet read for this book')+olderAnalysed]],
        ['validation',validating ? ['running',t('Task {task} · {n} / {m} stages verified',{task:short(validating.id),n:validating.verified,m:validating.total})] : analysedDone ? ['done',t('analysis published · Task {task}',{task:short(analysedDone.id)})] : analysisCurrent ? ['done',t('analysis published for this book')] : ['todo',t('runs when an answer is submitted')]],
        ['cro',refusal?.role==='cro' && refusal.kind==='refused' ? ['refused',t('last assessment refused · {code}',{code:refusal.code})] : refusal?.role==='cro' && refusal.kind==='correct' ? ['waiting',t('last CRO answer returned for correction · {code}',{code:refusal.code})] : refusal?.role==='cro' ? ['waiting',t('last submission outcome uncertain; reconcile before submitting again')] : publishing ? ['running',t('being sealed · Task {task}',{task:short(publishing.id)})] : published ? ['done',attribution || t('assessed · as the projection records')] : reviewedDone ? ['done',(b.imported.has(reviewedDone.id) ? t('assessed · imported here as HUMAN · Task {task}',{task:short(reviewedDone.id)}) : t('assessed · Task {task}',{task:short(reviewedDone.id)}))+olderReviewed] : dossierBound ? ['waiting',t('dossier read; awaiting the external assessment')+olderReviewed] : dossierRetained && analysisCurrent ? ['todo',t('dossier read before this projection read; read it again for this version')+olderReviewed] : analysisCurrent ? ['todo',t('dossier not read yet')+olderReviewed] : ['todo',t('needs a published analysis')+olderReviewed]],
        ['published',published ? ['done',t('{disposition} · {state}',{disposition:codeWords(v.disposition),state:codeWords(v.review_state)})] : ['todo',t('no review published for this book')]]];
    }
    return rows;
  }
  const blockedState=()=>Boolean(S.refusal) || ['EVIDENCE_AUTHORITY_NOT_ADMITTED','NO_BOOK_TO_REVIEW'].includes(S.view?.state);
  // the Overview's stage of actions, only where there is one (an empty section held its margins open under the coverage figure)
  const stageActions=()=>{ const a=actions(); return a ? html`<section class="review-stage">${a}</section>` : ''; };
  /* The primary actions, each saying what it does before it is pressed. */
  function actions() {
    const managed=(S.view?.available_actions || []);
    const provider=(action,label)=>managed.includes(action) ? typedBtn(label,'review-provider',action==='REFRESH_EVIDENCE' ? 'refresh' : 'review','button',current() ? '' : t('Read-only while pinned or busy')) : '';
    const buttons=[provider('REFRESH_EVIDENCE',t('Refresh evidence with the configured Provider')),provider('REVIEW_WITH_CRO',t('Review with the configured Provider'))].filter(Boolean);
    if(!buttons.length) return ''; // no Provider action admitted: no stage at all
    return html`<div class="flow review-actions">${buttons}</div><p class="caption">${t('The Provider buttons invoke the configured model Provider and consume its quota; they are not the native handoff and are refused when no credential is admitted.')}</p>`;
  }
  /* The page's one primary action at the head (round 32): the desk leads to the report, Sources
   * prepares them once the preview says they can be, the report exports. */
  function headActions() {
    const v=S.view;
    if(!(S.view || S.refusal)) return '';
    if(v?.state==='EVIDENCE_AUTHORITY_NOT_ADMITTED' && !['evidence-stream','report'].includes(app.page)) return ''; // N4 (law 130): the empty state holds the one way; the top row does not repeat it (the Report's delivery keeps its verb)
    if(app.page==='evidence'){ const p=cycle().primary; return p ? typedBtn(p.word,p.action,p.value,'button primary') : ''; } // the state's one next lawful action (round F1)
    // the reason is the page's own words (already in the reader's language), named by the button so Controls never re-reads it as a source
    if(app.page==='handoff'){ const p=reviewWords().primary; return p ? html`${typedBtn(p.word,p.action,p.value,'button primary',p.disabled || '','aria-describedby="reviewTopReason"')}<span class="sr-only" id="reviewTopReason">${p.disabled || ''}</span>` : ''; } // the turn's submit, or the report (round F4)
    if(app.page==='evidence-reading'){ const p=readingWords().primary || cycle().primary; return p ? typedBtn(p.word,p.action,p.value,'button primary',p.action==='review-continue' && !current() ? t('Read-only while pinned or busy') : '') : ''; } // `Read more` while the owner says pending (round F3)
    if(app.page==='evidence-stream'){ const p=storeWords().primary; return p ? typedBtn(p.word,p.action,p.value,'button primary',current() ? '' : t('Read-only while pinned or busy')) : ''; } // the store's one action (round F2)
    if(app.page==='report') return isUpdateBook() ? (S.delivery ? btn(t('Export delivery HTML'),'review-delivery-export','html','button primary') : '') : isExperimentBook() ? typedBtn(t('Compose the delivery'),'review-deliver','','button primary',ready() ? '' : t('Busy')) : S.exportDoc ? btn(t('Export review HTML'),'review-export','html','button primary') : '';
    return '';
  }
  /* ---- the reading surface (round 77): one list of the owners' objects in the lane (the
   * projection's issuers, issues and citations; the dossier's findings once read; the preview's
   * issuers and filings and the packet's spans on the sources page), one reading in the inspector. ---- */
  const KINDS={issuer:'Issuers',issue:'review|Issues',finding:'Findings',citation:'Citations',filing:'Filings'};
  function objects() {
    const v=S.view, out=[];
    if(app.page==='evidence-stream'){
      const p=S.preview?.status==='EVIDENCE_PREPARATION_READY' ? S.preview : null, facts=packetJson();
      // round E1: every issuer of the inventory with its holdings (its filings live in its reading), the packet's spans as citations
      const inv=inventory(), byId=Object.fromEntries((inv?.rows || []).map(r=>[r.entity_id,r]));
      for(const i of p?.scope?.selected_issuers || []) out.push({type:'issuer',id:i.entity_id,x:{...i,tickers:i.tickers || [i.entity_id]},issuer:i.entity_id,holdings:byId[i.entity_id] || null});
      for(const r of inv?.rows || []) if(!out.some(o=>o.type==='issuer' && o.id===r.entity_id)) out.push({type:'issuer',id:r.entity_id,x:{entity_id:r.entity_id,tickers:[r.entity_id]},issuer:r.entity_id,holdings:r});
      for(const c of Array.isArray(facts?.spans) ? facts.spans : []) if(c?.span_handle) out.push({type:'citation',id:c.span_handle,x:c,issuer:c.entity_id || ''});
      return out;
    }
    if(!v) return out;
    for(const r of v.issuer_rows || []) out.push({type:'issuer',id:r.entity_id,x:r,issuer:r.entity_id});
    for(const c of v.issue_cards || []) out.push({type:'issue',id:c.issue_handle,x:c,issuer:(c.affected_entities || [])[0] || ''});
    for(const f of dossierOf()?.findings || reportOf()?.review?.dossier?.findings || []) out.push({type:'finding',id:f.finding_handle,x:f,issuer:(f.affected_entities || [])[0] || ''});
    const pkg=findingPackage(); if(pkg && !out.some(o=>o.type==='finding' && o.id===pkg.finding.finding_handle)) out.push({type:'finding',id:pkg.finding.finding_handle,x:pkg.finding,issuer:(pkg.affected_entities || [])[0] || ''});
    for(const c of v.citations || []) out.push({type:'citation',id:c.span_handle,x:spanOf(c.span_handle) || c,issuer:c.entity_id || ''});
    return out;
  }
  const objectOf=(id)=>objects().find(o=>o.id===id) || null;
  // how the list is shown (the viewer's, kept per list): the tree by issuer, or flat by kind or
  // state; the rows' facts and availability by choice
  const DISPLAY=()=>({groupings:[['issuer',t('Issuer')],['kind',t('Kind')],['state',t('State')],['none',t('No grouping')]],orderings:[],properties:[['facts',t('Facts')]],apply:()=>patchMain()});
  function setFilter(spec){const [name,value='']=String(spec).split(':');if(S.pendingFilter===name)S.pendingFilter='';
    if(name==='evidence-report-count' || name==='evidence-report-conclusion'){S[name==='evidence-report-count' ? 'reportCount' : 'reportConclusion']=value==='all' ? '' : value;collectionState('report-issuers',true).page=0;writeRoute(routeUpdate());patchMain();return;}
    if(!name.startsWith('evidence-view-'))return;viewFilter(name.slice('evidence-view-'.length),value==='all' ? '' : value);}
  // Evidence collections share the product's table size, search, indexed rows and count.
  const collectionState=(name,keep=false)=>S.collections[name] || (keep ? (S.collections[name]={query:'',page:0}) : {query:'',page:0});
  function collectionQuery(name,value){const state=collectionState(name,true);state.query=String(value || '');state.page=0;writeRoute(routeUpdate());patchMain();}
  function collectionPage(spec){const [name,way]=String(spec).split('|'),state=collectionState(name,true);state.page=Math.max(0,state.page+(way==='prev' ? -1 : 1));writeRoute(routeUpdate());patchMain();}
  function collectionTable(name,list,columns,cells,{words=(x)=>JSON.stringify(x),total=list.length,tools=''}={}){
    const state=collectionState(name),query=state.query.trim().toLowerCase(),found=query ? list.filter(x=>String(words(x)).toLowerCase().includes(query)) : list;
    const {shown,start,page,pages}=pageOf(found,state.page);if(state.page!==page){state.page=page;if(S.collections[name])writeRoute(routeUpdate());}
    const search=total>LIST_PAGE ? searchBar('evidenceQuery-'+name,t('Search'),t('Search'),state.query) : '';
    return html`<div data-evidence-collection="${name}">${tools}${search}${table([{label:'#',type:'num',index:true},...columns],shown.map((x,i)=>tr([count(start+i+1),...cells(x)])),'',{report:true,countLine:false})}${pager({total:found.length,one:'{n} entry',many:'{n} entries',page,pages,prev:['review-collection-page',name+'|prev'],next:['review-collection-page',name+'|next']})}</div>`;
  }
  const textCollection=(name,list,words=x=>x)=>collectionTable(name,list,[{label:t('Reason'),type:'text',absorb:true}],x=>[words(x)]);
  const issuerObjects=(id)=>objects().filter(k=>k.type!=='issuer' && ((k.x.affected_entities || []).includes(id) || k.x.entity_id===id || (k.type==='filing' && k.issuer===id)));
  function objectCollection(name,list){return collectionTable(name,list,[{label:t('Kind'),type:'text',cls:'col-tight'},{label:t('Object'),type:'text',absorb:true},{label:t('State'),type:'text'}],k=>{const n=evidenceName(k.type,k.x);return [t(KINDS[k.type]),btnAttrs(n.subject || k.id,'review-live-item',k.id,'text-btn','data-row-press'),n.word || ''];},{words:k=>[k.id,entityNames(k.x.affected_entities),k.x.entity_id,k.x.summary,k.x.topic,k.x.document_type].filter(Boolean).join(' ')});}

  const clearFilter=(name)=>setFilter(name+':');
  function addClause(name){S.pendingFilter=name;patchMain();const trigger=$('#filter-'+name);if(trigger)Picker.open(trigger);} // round 95: `+ Filter` chose a field
  /* The limits, the provenance and the unmapped positions, read in the inspector's Facts. */
  /* The Overview's Facts (B1; the book plan's item 5): what the review recorded beyond the head's
   * line -- the analysis, the reviewer, the policy, its rules and reasons, the limits and the
   * provenance. The head says the days and the completeness. */
  function overviewFacts() {
    const v=S.view; if(!v || S.refusal) return [];
    const d=dossierOf(), selected=(v.eligible_versions || []).find(x=>x.is_selected) || null, published=reviewPublished() || historical();
    const rows=[[t('Analysis'),selected ? t('selected · {d}',{d:dayWord(selected.evidence_as_of)}) : d?.evidence_as_of ? dayWord(d.evidence_as_of) : ''],...(published ? [[t('Reviewer'),attributionWords(v.review_attribution || [],{facts:true}) || ''],[t('Policy'),v.policy_version ? codeCell(v.policy_version) : ''],[t('Rules'),(v.rule_ids || []).length ? html`<span class="mono">${v.rule_ids.join(', ')}</span>` : ''],[t('Reasons'),(v.reasons || []).length ? html`<span class="owner-text">${v.reasons.map((r,i)=>html`${i ? ' · ' : ''}${reasonWords(r)}`)}</span>` : '']] : [])].filter(([,x])=>x!=='');
    const c=v.scope_coverage,u=v.book?.update_subject,x=v.book?.experiment_subject,lines=[...gapLines(v.gaps || []),...(v.claim_limits || [])]; // the Report's words for the same lines: one way to say a gap
    return [{title:t('The review'),body:kv(rows)},...teamFacts(),
      ...(lines.length ? [{title:t('Limits'),body:html`${textCollection('overview-limits',lines,said)}`}] : []),
      ...(u || x ? [{title:t('Provenance'),body:html`${u ? kv([[t('Position basis'),u.position_basis],[t('Observed through'),u.observed_through],[t('Entry session'),u.entry_session],[t('Update publication'),html`<span class="mono">${u.update_publication_hash}</span>`]]) : ''}${x ? kv([[t('Previous holdings session'),x.preceding_session || ''],[t('Research cutoff'),x.research_as_of_session || '']]) : ''}`}] : []),
      ...(c?.unmapped?.length ? [{title:countText(c.unmapped_positions,'{n} position without admitted issuer mapping','{n} positions without admitted issuer mapping'),body:collectionTable('unmapped-positions',c.unmapped,[{label:t('Listing'),type:'text'},{label:t('Weight'),type:'num',cls:'col-tight'},{label:t('State'),type:'text'},{label:t('Reason'),type:'text',absorb:true}],r=>[r.ticker || r.listing_id,pctText(r.ending_weight),codeWords(r.status),r.reason])}] : [])];
  }
  /* Coverage and mapping as one bar each (round 77, the single visuals round 5 left): the words
   * first, the bar under them; the required minimum is the bar's mark. */
  const share=(text)=>{const m=String(text || '').match(/^(-?\d+(?:\.\d+)?)\s*%$/);return m ? Number(m[1])/100 : null;};
  /* An issue's days (contract 10.8): when its earliest cited finding was found, since when an
   * earlier review has held it open, and the day the reviewer last assessed it when this review
   * carried it as it stood. */
  const issueDates=(c)=>[...(c.found_on ? [[t('Found'),dayWord(c.found_on)]] : []),...(c.open_since ? [[t('Open since'),dayWord(c.open_since)]] : []),...(c.carried_on ? [[t('Carried'),t('as last assessed {d}',{d:dayWord(c.carried_on)})]] : [])];
  const issueDayWords=(c)=>c.carried_on ? t('carried from {d}',{d:dayWord(c.carried_on)}) : c.open_since ? t('open since {d}',{d:dayWord(c.open_since)}) : c.found_on ? t('found {d}',{d:dayWord(c.found_on)}) : '';
  function issue(card) {
    return html`${evidenceRow('issue',card,{named:true,brief:true,cls:'evidence-head'})}
      <div class="review-claim-blocks">
        <section><h3>${t('What the sources say (Analyst, cited)')}</h3><ul>${(card.observed_source_text || []).map(x=>html`<li>${x}</li>`)}</ul>${Array.isArray(card.cited_finding_handles) ? html`<p class="caption review-cites">${t('Cited findings')}: ${card.cited_finding_handles.length ? card.cited_finding_handles.map(h=>citePill(h)) : ''}</p>` : ''}</section>
        <section><h3>${t('CRO assessment')}</h3><p>${card.cro_inference}</p>${kv([[t('Severity if true'),evidenceState(card.severity_if_true).word],[t('Interpretation'),html`${codeWords(card.effective_interpretation)} <span class="sub-cell">${t('stated')} ${codeWords(card.stated_interpretation)}</span>`],[hint(t('Support'),t('How a finding is held up by its own citations: supported, single source, contested, unsupported.')),codeWords(card.evidence_structure)],[t('Position impact'),codeWords(card.position_impact_direction)],[hint(t('Exposure'),t('The owner\u2019s ordinal band of a position from its weight and change: low, medium, high, critical.')),bandWords(card.exposure_band)]])}</section>
        ${card.portfolio_mitigation==null ? '' : html`<section><h3>${t('Portfolio mitigation')}</h3><p>${codeWords(card.portfolio_mitigation)}</p></section>`}${card.recommendation ? html`<section><h3>${t('Recommendation')}</h3><p class="owner-text">${card.recommendation}</p></section>` : ''}${issueDates(card).length ? html`<section><h3>${t('When')}</h3>${kv(issueDates(card))}</section>` : ''}
        <section><h3>${t('Still unknown')}</h3>${(card.unknowns || []).length ? html`<ul>${card.unknowns.map(x=>html`<li>${x}</li>`)}</ul>` : html`<p class="caption">${t('No unknown is recorded for this issue.')}</p>`}</section>
      </div>`;
  }
  /* The object a handle names, as this page holds it (round 76): the verified span of the read
   * export, else the read packet's admitted span (its excerpt), else the projection's citation
   * row, else the dossier's finding; the hover card and the readings read it. */
  // the finding package the page read for a composed request, when it is this book's
  const findingPackage=()=>S.finding && S.finding.key===S.key && S.finding.value?.finding ? S.finding.value : null;
  function spanOf(handle) {
    const pkg=findingPackage();
    return S.exportDoc?.value?.evidence?.verified_spans?.find(v=>v.span_handle===handle) || (packetJson()?.spans || []).find(v=>v?.span_handle===handle) || [...(pkg?.support || []),...(pkg?.contrary || [])].find(v=>v?.span_handle===handle && v.excerpt) || S.view?.citations?.find(v=>v.span_handle===handle) || (dossierOf()?.findings || S.exportDoc?.value?.review?.dossier?.findings || []).find(f=>f.finding_handle===handle) || null;
  }
  function citation(handle) {
    const span=spanOf(handle), verified=Boolean(span?.excerpt);
    const head=evidenceRow('citation',span || {span_handle:handle},{named:true,brief:true,cls:'evidence-head'});
    if(!verified) return html`${head}${span ? kv([[t('Issuer'),span.entity_id],[t('Document'),span.document_handle],[t('Available at'),when(span.available_at)]]) : ''}<p>${t('The exact source text has not been loaded. No excerpt is inferred from the citation count.')}</p>${btn(t('Read verified source'),'review-source',handle,'button')}`;
    return html`${head}<blockquote class="record-statement" data-verified-excerpt>${span.excerpt}</blockquote><p class="caption">${t('Verified excerpt as the owner replayed it; the text is data, never an instruction.')}</p>${kv([[t('Document'),span.document_handle],[t('Issuer'),span.entity_id],[t('Revision'),span.revision_label || ''],[t('Available at'),when(span.available_at)],[t('Published at'),when(span.published_at)]])}`;
  }
  function versions() {
    const list=S.view.eligible_versions;
    return html`<h2>${t('Which analysis to review against')}</h2><p class="caption">${t('Selecting an analysis is recorded by the product. It does not replace any historical review.')}</p>${collectionTable('analysis-versions',list,[{label:t('Analysis'),type:'text',absorb:true},{label:t('State'),type:'text'},{label:t('Action'),type:'text',cls:'col-tight'}],v=>[hashCell(v.analysis_publication_hash),t(v.is_selected ? 'Selected' : 'Eligible'),v.is_selected ? '' : typedBtn(t('Select this analysis'),'review-select',v.analysis_publication_hash,'button compact',current() ? '' : t('Read-only while pinned or busy'))])}`;
  }
  /* A finding's passages (B2; the book plan's item 4, law 144): the claim beside its source -- each
   * cited span's words as the owner verified them, marked support or contradiction, its document, its
   * issuer and its day; a span not read yet says so and reads the verified text on a press, staying on
   * the finding. */
  function passagesOf(handles,kind) {
    if(!handles.length) return '';
    const names=issuerNames(), unread=handles.some(h=>!spanOf(h)?.excerpt);
    const one=(h)=>{ const x=spanOf(h), words=x?.excerpt ? String(x.excerpt) : '';
      return html`<div class="finding-passage" data-span="${h}" data-cite="${kind}"><p class="finding-passage-head"><span class="mono">${x?.document_handle || h}</span>${x?.document_type ? html` <span class="muted">${x.document_type}</span>` : ''}${x?.entity_id ? html` · ${names[x.entity_id] || x.entity_id}` : ''}${x?.available_at ? html` · ${dayOf(x.available_at)}` : ''}</p>${words ? html`<blockquote class="record-statement" data-verified-excerpt>${words}</blockquote>` : html`<p class="caption">${t('The exact source text has not been loaded. No excerpt is inferred from the citation count.')}</p>`}</div>`; };
    return html`<h3>${t(kind==='contra' ? 'Contradicting passages' : 'Supporting passages')} <span class="num">${count(handles.length)}</span></h3>${collectionTable('finding-'+S.item+'-'+kind,handles,[{label:t('Passage'),type:'text',absorb:true}],h=>[one(h)])}${unread && S.item && (reviewPublished() || historical()) ? btn(t('Read the verified passages'),'review-source',S.item,'button compact') : ''}`;
  }
  /* The reading of one object (the inspector's body), per type. */
  function evidenceReading(id=S.item) {
    if(String(id).startsWith('cell:')) return cellReading(id);
    if(String(id).startsWith('span:')) return spanReading(String(id).slice(5));
    if(id==='analysis') return S.view?.eligible_versions ? versions() : '';
    const o=objectOf(id);
    if(!o) return html`<p class="caption">${t('This object is not in the current read.')}</p>`;
    const head=evidenceRow(o.type,o.x,{named:true,brief:true,cls:'evidence-head'});
    const rows=(list,type)=>list.length ? objectCollection('issuer-'+id+'-'+type,list) : '';
    if(o.type==='issue'){ const a=(S.view?.required_actions || []).find(a=>(o.x.affected_entities || []).includes(a.entity_id)), standing=a ? actionStanding(a) : null; return html`${a ? refusal({reason:said(a.reason)},standing.tone,{state:standing.state,word:standing.word,next:a.action==='NONE' ? '' : html`${codeWords(a.action)} · ${entityNames([a.entity_id])}`}) : ''}${issue(o.x)}`; }
    if(o.type==='citation') return citation(o.id);
    // B2 (item 4): the claim -- the finding's issuers and stance, its summary once -- then its sources, then the review's words on it; its handle is the pane's caption
    if(o.type==='finding'){ const f=o.x, sup=(f.supporting_span_handles || []), con=(f.contradicting_span_handles || []), disp=dispositionOf(f.finding_handle); return html`${evidenceRow('finding',f,{named:true,brief:true,cls:'evidence-head',why:html`${entityNames(f.affected_entities) || ''}${f.direction ? html` · ${cellWord(f.direction)}` : ''}`})}${f.summary ? html`<p class="owner-text">${f.summary}</p>` : ''}${passagesOf(sup,'support')}${passagesOf(con,'contra')}${kv([[t('Lifecycle'),codeWords(f.lifecycle)],[t('Stance'),cellWord(f.direction)],[hint(t('Support'),t('How a finding is held up by its own citations: supported, single source, contested, unsupported.')),codeWords(f.structure)],...(disp ? [[t('Disposition'),codeWords(disp.disposition)],...(disp.rationale ? [[t('CRO rationale'),html`<span class="owner-text">${disp.rationale}</span>`]] : [])] : [])])}${(f.limitations || []).length ? html`<h3>${t('Limitations')}</h3>${textCollection('finding-'+f.finding_handle+'-limits',f.limitations)}` : ''}`; }
    if(o.type==='filing') return html`${head}${kv([[t('Issuer'),o.x.entity_id],[t('Document type'),o.x.document_type],[t('Revision'),o.x.revision],[t('Available at'),when(o.x.available_at)]])}<p class="caption">${t('A candidate is a document the source package holds; it becomes admitted passages only after preparation, in the exact packet.')}</p>`;
    const r=o.x, kids=issuerObjects(o.id), of=(type)=>kids.filter(k=>k.type===type);
    const summary=r.findings_summary_parts?.length ? r.findings_summary_parts.map(([words,authored])=>said(words)+(authored ? ': '+authored : '')).join(I18N.zh ? '；' : '; ') : said(r.findings_summary);
    const review=o.holdings && !r.conclusion ? '' : kv([[t('Conclusion'),r.conclusion ? codeWords(r.conclusion) : ''],[t('Findings'),summary || (r.selection_reason ? codeWords(r.selection_reason) : '')],[t('Weight'),weightWords(r.ending_weight) || ''],[t('Change'),changeWords(r.signed_change) || ''],[hint(t('Exposure'),t('The owner\u2019s ordinal band of a position from its weight and change: low, medium, high, critical.')),bandWords(r.exposure_band)]]);
    return html`${head}${review}${o.holdings ? holdingsReading(o) : ''}${['issue','finding','filing','citation'].map(type=>of(type).length ? html`<h3>${t(KINDS[type])} <span class="num">${count(of(type).length)}</span></h3>${rows(of(type),type)}` : '')}`;
  }
  /* The reading pane reads the object the route names (`review_item`) beside the list (round 92:
   * in the lane, never the side column); a repaint refreshes it. */
  function setItem(id) {
    const was=S.item; S.item=id || '';
    if(was && !S.item && typeof Window!=='undefined') Window.focusAfterPaint(['review-live-item',was]); // law 149: closing returns the focus to the row
    if(S.item && !was){if(typeof pushDetail==='function'){pushDetail({review_item:S.item});S.routeSig=routeSignature();}else pushRoute({review_item:S.item});render();}
    else if(!S.item && was){if(typeof Window!=='undefined')Window.clearDetailTrail?.();if(typeof closeDetailRoute==='function'){if(closeDetailRoute({review_item:''}))return;}else replaceHash({review_item:''});}
    else replaceHash({review_item:S.item});
    patchMain();
    if(S.item && typeof document!=='undefined') { const pane=document.querySelector('#main .reading-pane'); if(pane && typeof Geometry!=='undefined' && Geometry.focusQuietly) Geometry.focusQuietly(pane); }
  }
  /* The packet request the product composed for this book: a prepared Task and, in a coverage
   * run, the unit it holds. Adopted when the page holds no prepared Task yet; a run of several
   * units offers its packets as next steps and adopts none by itself. */
  // U79 (V547): a prepared unit under authority the Host no longer holds -- the workspace's Evidence package or source mode
  // moved since -- carries the owner's words on its progress row; its packet is refused, so it is offered nowhere
  const staleOf=(p)=>String(p?.state || '')==='PREPARED' && Boolean(p?.detail);
  const staleUnit=(view,unit)=>(view?.coverage_progress?.units || []).some(u=>u.unit_id===unit && staleOf(u));
  function packetRequests(view=S.view){ return Object.values(view?.next_requests || {}).filter(r=>r && r.operation==='EVIDENCE_PACKET' && !staleUnit(view,r.evidence_unit_id)); }
  function adoptPacketRequest(view, addressed=true) {
    const packets=packetRequests(view);
    if(!S.packetTask){ if(packets.length!==1 || !packets[0].task_id) return; S.packetTask=packets[0].task_id;S.packetUnit=packets[0].evidence_unit_id || '';loadDrafts();if(addressed)writeRoute(routeUpdate());return; }
    // the Task the page holds without a unit: the product names exactly one unit for it, or none
    if(S.packetUnit) return;
    const own=packets.filter(r=>r.task_id===S.packetTask && r.evidence_unit_id);
    if(own.length===1){S.packetUnit=own[0].evidence_unit_id;loadDrafts();if(addressed)writeRoute(routeUpdate());}
  }
  /* One request, dispatched by the operation it names with every field it carries. A read
   * opens or navigates; an execution goes through the same confirmation its page action does;
   * managed work only when the product admits it and the person pressed. */
  function next(k) {
    const r=S.view?.next_requests?.[k] ? {...S.view.next_requests[k]} : null; // captured whole at the press
    if(!r || !current()) return;
    if(!appliesHere(r)){S.error=t('This request names another book; open that book to run it.');render();return;}
    const op=r.operation;
    if(op==='EVIDENCE_PREVIEW'){app.page='evidence-stream';writeRoute(routeUpdate({page:'evidence-stream'}));render();return inspectSources(r);}
    if(op==='EVIDENCE_PACKET'){S.nav+=1;S.packetTask=r.task_id;S.packetUnit=r.evidence_unit_id || '';S.packet=null;S.pending=null;S.bundleRev++;loadDrafts();pushRoute(routeUpdate());return getBundle('analyst',r);}
    if(op==='CRO_REVIEW_DOSSIER') return getBundle('cro',r);
    if(op==='CRO_REVIEW_FINDING') return readFinding(r);
    if(op==='EVIDENCE_CRO_EXPORT') return openExport(r);
    if(op==='STATUS'){ if(r.task_id) return LiveTasks.select(r.task_id); return; }
    if(op==='EVIDENCE_PREPARE') return confirm('prepare','',r);
    if(op==='EVIDENCE_REFRESH') return confirm('refresh','',r);
    if(op==='CRO_REVIEW') return confirm('cro','',r);
    if(op==='EVIDENCE_CONTINUE' || op==='STORAGE_EVIDENCE_REBUILD') return confirmRequest(op,r);
    S.error=t('No page action is mapped for {op}',{op});render();
  }
  /* An execution the product composed and the page has no dedicated action for: confirmed
   * with its own words -- what it will do, its scope and budget as the request states them --
   * and sent with the request's fields exactly. It is an execution, never an answer: the
   * outcome binds a continuation to the book's preparations and follows a rebuild's Task,
   * and nothing about it is an imported answer or a review. */
  function confirmRequest(op,r) {
    const [,word]=NEXT_WORDS[op] || ['run',op];
    const payload=requestFields(r);
    const facts=Object.entries(payload).map(([k,v])=>[codeWords(k),String(v)]);
    const effect=t(op==='EVIDENCE_CONTINUE' ? 'Reads the sources further within the declared budget; a new bounded read, nothing recomputed.' : 'Builds a derived evidence index beside the stored evidence; nothing about the evidence itself changes.');
    closeDialog();S.pending={path:Data.route(op),payload,revision:S.revision,kind:'execute',operation:op,key:S.key,role:S.role,task:S.packetTask,unit:S.packetUnit};
    openDialog(t(word),bookName(),html`<p>${effect}</p>${kv(facts)}`,html`${btn(t(word),'review-commit','','button primary')}`);
  }
  /* The finding a composed request names, read as the product's bounded package with the
   * request's fields exactly (its dossier or its sealed review, never the page's): the
   * reading pane shows it, its excerpts serve the citations, and a package of another book
   * or an older read is not adopted. */
  async function readFinding(r) {
    if(!r.finding_handle){S.error=t('The request names no finding.');render();return;}
    S.busy='finding';S.error='';const ticket=S.revision, asked={key:S.key,handle:r.finding_handle,request:{...r}};render();
    try{
      const doc=await Data.readDocument(Data.route('CRO_REVIEW_FINDING')+'?'+new URLSearchParams(requestFields(asked.request)));
      if(!doc.value?.finding)throw Error(doc.value?.failure_code || doc.value?.detail || doc.value?.disposition || 'Finding unavailable');
      if(ticket!==S.revision || S.key!==asked.key) return;
      S.finding={key:asked.key,handle:asked.handle,request:asked.request,value:doc.value};
    }catch(e){if(ticket===S.revision && S.key===asked.key)S.error=e.message;}
    finally{if(ticket===S.revision){S.busy='';repaintMain();}}
    if(ticket===S.revision && S.key===asked.key && findingPackage()) setItem(asked.handle);
  }
  /* The exact report a composed request names: its publication becomes the page's pin when it
   * is not the one shown (the route restores it), and the report is read with the request's
   * fields, not re-assembled from the pin. */
  async function openExport(r) {
    const hash=r.review_publication_hash || '';
    if(!hash){S.error=t('The request names no review publication.');render();return;}
    if(hash!==(S.pin || workingPublication())){
      // the open resets the page synchronously, then reads the pinned section; the captured
      // request is recorded after that reset so the report's own read on arrival uses it
      const opened=open(S.selector,hash,'report'), gen=S.nav; S.exportRequest={key:S.key,request:{...r}};
      await opened; if(gen!==S.nav) return; if(S.exportDoc?.hash===hash) return;
    }
    else { S.exportRequest={key:S.key,request:{...r}}; app.page='report';writeRoute(routeUpdate({page:'report'}));render(); }
    return readExport(S.exportRequest.request);
  }
  const setItemQuiet=(id)=>{S.item=id || '';patchMain();}; // the route's item, adopted without writing the route
  const readingPaneMarkup=()=>{ if(!S.item) return ''; const id=S.item, o=objectOf(id), r=o ? evidenceName(o.type,o.x) : null, cell=String(id).startsWith('cell:') ? cellOf(id) : null, span=String(id).startsWith('span:') ? String(id).slice(5) : ''; const x=span ? spanOf(span) : null, handled=r && ['finding','issue','citation'].includes(o.type); // item 6: the words title the pane; the handle follows small in its caption
    return readingPane(span ? (x?.entity_id ? html`${issuerNames()[x.entity_id] || x.entity_id}${x.document_type ? html` · ${x.document_type}` : ''}` : html`<span class="mono">${span}</span>`) : cell ? html`${cell.entity_id} · ${codeWords(cell.topic)}` : r ? r.subject : id==='analysis' ? t('Which analysis') : id, span ? html`${t('Passage')}${x?.entity_id ? html` · <span class="mono">${span}</span>` : ''}` : cell ? t('Ledger cell') : handled ? html`${r.kind} · <span class="mono">${id}</span>` : r ? r.kind : t('Evidence'), evidenceReading(id), 'review-item-close', 'desk-reading', ['review-live-item', id]); };
  /* ---- the sources: candidates versus admitted passages ---- */
  async function inspectSources(request=null,quiet=false) { // quiet (round E2): the desk's Sources layer reads the inventory; a failed read is its own words, not the desk's notice
    if(!current())return;
    // a quiet read (the rail's Sources stage on every page, round F1) is background: it never makes the page busy, so an editor's keystroke or a button is never swallowed by it
    if(!quiet)S.busy='sources';S.previewPending=true;const ticket=S.revision;repaintMain();
    try{const value=await Data.read(Data.route('EVIDENCE_PREVIEW')+'?'+(request ? new URLSearchParams(requestFields(request)) : query()),true);if(ticket===S.revision)S.preview=value;}
    catch(e){if(ticket===S.revision){if(!quiet)S.error=e.message;S.preview={status:'READ_REFUSED',failure_code:e.message};}}
    finally{if(ticket===S.revision){S.previewPending=false;if(!quiet)S.busy='';repaintMain();}}
  }
  /* What the owner's source mode means for preparation: the words come from its mode and
   * declared work, never from a fixed promise. */
  function modeWords(p) {
    const mode=p?.source_mode || '', work=p?.source_work || '';
    if(mode==='RECORDED' || work==='RECORDED_LOCAL_READ') return {kind:'recorded',prepare:t('Preparation is one local Task of the evidence owner over the recorded source package: it reads the recorded documents, canonicalizes them, builds the retrieval index (the embedding work of that stage) and selects admitted passages. It fetches nothing from a network and runs no analyst or CRO model; it needs one explicit confirmation.'),acquire:t('The recorded documents of the source package are read locally; nothing is fetched.'),resolve:t('The issuer registry names the official sources for the scope; no network is used in recorded mode.')};
    if(mode==='LIVE_OFFICIAL' || work==='EXPLICITLY_ADMITTED_OFFICIAL_ACQUISITION') return {kind:'live',prepare:t('Preparation is one Task of the evidence owner that ACQUIRES official documents from the approved source families over the network under the explicitly admitted acquisition policy, then canonicalizes them, builds the retrieval index (embedding work) and selects admitted passages. It runs no analyst or CRO model; it needs one explicit confirmation and the owner revalidates the acquisition authority at submission.'),acquire:t('Official documents are acquired from the approved source families over the network under the admitted acquisition policy.'),resolve:t('The issuer registry names the official sources to acquire for the scope.')};
    return {kind:'unknown',prepare:t('The owner did not state the source mode of this preparation on this page; its cost and side effects are not known here. Read the source preview again before preparing.'),acquire:t('How the documents are obtained depends on the owner\'s source mode, which is not stated on this page.'),resolve:t('The issuer registry names the official sources for the scope.')};
  }
  /* The packet's one JSON block, read with a quote- and escape-aware scan (a brace inside an
   * excerpt cannot end it early); null when the packet holds no readable block. */
  function packetFacts(text) {
    if(typeof text!=='string') return null;
    const start=text.indexOf('{'); if(start<0) return null;
    let depth=0, inString=false, escaped=false;
    for(let i=start;i<text.length;i+=1){
      const ch=text[i];
      if(inString){ if(escaped) escaped=false; else if(ch==='\\') escaped=true; else if(ch==='"') inString=false; continue; }
      if(ch==='"') inString=true;
      else if(ch==='{') depth+=1;
      else if(ch==='}'){ depth-=1; if(!depth){ try{const value=JSON.parse(text.slice(start,i+1));return value && typeof value==='object' && !Array.isArray(value) ? value : null;}catch{return null;} } }
    }
    return null;
  }
  const packetJson=()=>packetFacts(bundleOf('analyst')?.packet);
  // the evidence identity a packet carries: its analysis context and the cutoff its obligation names
  const packetIdentity=(value)=>({context:value?.analysis_context_hash || '',asOf:packetFacts(value?.packet)?.research_obligation?.evidence_as_of || ''});
  /* ---- the sources: the book's data layer (round E1) -- what the workspace holds for each
   * issuer, what the last check found at the source, the units of a wide book, and what a
   * preparation would do; every figure the preview's or the projection's. ---- */
  const BYTE_WORDS=(n)=>typeof bytesWords==='function' ? bytesWords(n) : String(n);
  // the preview's inventory as one shape for both source modes: a recorded package's documents,
  // or the live branch's retained bodies -- the owner's counts, never the page's
  function inventory() {
    const p=S.preview?.status==='EVIDENCE_PREPARATION_READY' ? S.preview : null; if(!p) return null;
    const inv=p.source_inventory || {}, live=(inv.mode || p.source_mode)==='LIVE_OFFICIAL', scope=Object.fromEntries((p.scope?.selected_issuers || []).map(i=>[i.entity_id,i]));
    const rows=(inv.issuers || []).map(r=>{ const held=Number(live ? r.retained_documents : r.documents) || 0, deferred=live ? (r.deferred_documents || []) : [];
      return {entity_id:r.entity_id,unit:r.unit_id || '',cik:r.cik || '',held,deferred,bytes:live ? Number(r.retained_bytes) || 0 : null,types:r.document_types || [],latest:(live ? r.latest_accepted_at : r.latest_available_at) || '',state:held>0 ? 'held' : deferred.length ? 'deferred' : 'no_source',scope:scope[r.entity_id] || null,live}; });
    const withSource=live ? inv.issuers_with_retained_documents : inv.issuers_with_source;
    return {live,rows,units:inv.units || [],withSource:withSource==null ? null : Number(withSource),bytes:live ? Number(inv.retained_bytes) || 0 : null,without:inv.issuers_without_source || [],claim:inv.claim || ''};
  }
  const unitsOf=()=>S.preview?.coverage?.units || [];
  const wide=()=>unitsOf().length>1;
  const unitProgress=(id)=>(S.view?.coverage_progress?.units || []).find(u=>u.unit_id===id) || null;
  // the checks the prepared Tasks sealed at the source: the book's for one unit, each unit's in a run
  const checksOfBook=()=>wide() ? unitsOf().map(u=>u.source_check).filter(Boolean) : (S.preview?.source_check ? [S.preview.source_check] : []);
  const issuerDocs=(entity)=>checksOfBook().flatMap(c=>(c.documents || []).filter(d=>d.entity_id===entity));
  const issuerWords=(scopeRow,id)=>entityNames(scopeRow?.tickers) || id;
  // a filing type: an owner's code in words (RELEASE -> Release), a form's own name as it is (10-K)
  // Document forms are free source metadata, not UI state codes.
  const typeWords=(types)=>(types || []).map(String).join(', ');
  /* The holdings: one row per issuer -- words first, figures last (laws 32-38); the state is the
   * owner's holding said as a word (a neutral `held` is its word alone, law 81). */
  function holdingsTable(inv) {
    const isWide=wide();
    // the Portfolio holdings table's shape (the user's reading, 2026-09-22: a hundred issuers must not fill the screen): an index, a search field, a table page of rows (law 92), the foot's count and Previous / Next
    const q=S.sourceQuery.trim().toLowerCase();
    const found=q ? inv.rows.filter(r=>[r.entity_id,issuerWords(r.scope,r.entity_id),r.unit,typeWords(r.types)].some(x=>String(x || '').toLowerCase().includes(q))) : inv.rows;
    const PAGE=LIST_PAGE, {shown,start,page,pages}=pageOf(found,S.sourcePage); S.sourcePage=page;
    const toolbar=searchBar('sourceQuery',t('Find an issuer'),t('Find an issuer…'),S.sourceQuery);
    const turns={page,pages,prev:['review-source-page','prev'],next:['review-source-page','next']}, foot=pager({total:found.length,one:'{n} entry',many:'{n} entries',...turns});
    const columns=[{label:'',type:'text',cls:'col-tight col-fold'},{label:'#',type:'num',index:true},{label:t('Issuer'),type:'text'},...(isWide ? [{label:t('Unit'),type:'id',cls:'col-tight'}] : []),{label:t('State'),type:'status',cls:'col-tight'},{label:t('Filings'),type:'text',absorb:true},{label:t(inv.live ? 'Latest accepted' : 'Latest available'),type:'date',cls:'col-tight'},{label:t('Held'),type:'num',cls:'col-tight'},...(inv.live ? [{label:t('Deferred'),type:'num',cls:'col-tight'},{label:t('Retained'),type:'num',cls:'col-tight'}] : [])]; // W: a one-word state and the counts are tight columns (as the Overview's issuers, round H1), so the table fits beside the side column before it rolls
    // B3 (item 5): an issuer's row opens its documents in place, under it; a held issuer the published review did not reach says the review's record beside its forms -- never a reason the owner does not record
    // law 149: the row is one press -- its documents, the same press folds them -- its chevron leading the row; the issuer's reading is the Overview's and the Review's
    const docs=sourceDocuments(), docsOf=(id)=>docs.rows.filter(d=>d.entity===id);
    const reached=(id)=>{ const r=reviewPublished() || historical() ? (S.view?.issuer_rows || []).find(x=>x.entity_id===id) : null; if(!r || issuerReviewed(r)) return ''; const e=evidenceState(r.conclusion), why=unreviewedReason(r); return html` · ${stateLine(e.state,{word:e.word,next:''})}${why && why!==String(r.conclusion) ? html` <span class="muted">· ${codeWords(why)}</span>` : ''}`; };
    const cells=(list)=>html`${list.map(x=>html`<td>${x}</td>`)}`;
    const rows=shown.map((r,i)=>{ const list=docsOf(r.entity_id), open=S.docsOpen.has(r.entity_id), name=issuerWords(r.scope,r.entity_id);
      const toggle=list.length ? btnAttrs(icon('chevron'),'review-issuer-docs',r.entity_id,'icon-btn compact doc-toggle',html`aria-expanded="${open}" aria-label="${t('Documents of {issuer}',{issuer:name})}" data-tip="${countText(list.length,'{n} document','{n} documents')}" data-row-press`) : '';
      const main=html`<tr data-issuer="${r.entity_id}">${cells([toggle,count(start+i+1),html`<span class="doc-issuer">${name}</span>`,...(isWide ? [html`<span class="mono">${r.unit}</span>`] : []),stateLine(r.state,{next:''}),html`${typeWords(r.types)}${reached(r.entity_id)}`,r.latest ? when(r.latest) : '',count(r.held),...(inv.live ? [r.deferred.length ? count(r.deferred.length) : '',r.bytes ? BYTE_WORDS(r.bytes) : ''] : [])])}</tr>`;
      const kids=open ? list.map(d=>html`<tr class="doc-row" data-depth="1">${cells(['','','',...(isWide ? [''] : []),stateLine(d.status,{next:''}),html`${typeWords([d.form])}${d.title ? html` · <span class="muted">${d.title}</span>` : ''}`,d.when ? when(d.when) : '',docs.fromPacket && d.chars ? html`${count(d.chars)} <span class="num-unit">${t('characters')}</span>` : '',...(inv.live ? ['',''] : [])])}</tr>`) : '';
      return html`${main}${kids}`; });
    // the panel's one line is the owner's totals and which documents these are; what a holding means is a Facts sentence
    const known=inv.withSource==null ? inv.rows.filter(r=>r.held>0).length : inv.withSource;
    const totals=html`${t('{w} of {n} issuers hold a document',{w:count(known),n:count(inv.rows.length)})}${inv.without.length ? html` · ${t('without a source')}: ${inv.without.join(', ')}` : ''}${inv.live && inv.bytes ? html` · ${t('{b} retained',{b:BYTE_WORDS(inv.bytes)})}` : ''}${S.preview?.scope?.mapping_failures?.length ? html` · ${countText(S.preview.scope.mapping_failures.length,'{n} mapping failure','{n} mapping failures')}` : ''} · ${docs.caption}`;
    return panel(html`${t('Issuers and their documents')} <span class="num">${count(inv.rows.length)}</span>`,totals,html`${inv.rows.length>PAGE ? toolbar : ''}${found.length>PAGE ? pagerTop(turns) : ''}${!found.length ? emptyState(t('No issuer matches this search.'),'','','nomatch') : table(columns,rows,'',{report:true,countLine:false,classes:'holdings-table'})}${foot}`,'','data-box="table" data-jump="issuers"');
  }
  /* The last check: the acquisition stage's own accounting -- traffic apart from bodies, the
   * outcome of every resource -- and the one sentence in its figures. */
  function checkFacts(check,live) {
    if(!check){
      // the owner binds a check to the preparation that ran it (its intent names the cutoff): a
      // packet prepared as of an earlier cutoff has its check, but the preview of this cutoff
      // does not project it -- said as that, never as "nothing was checked"
      const prepared=Object.values(S.view?.next_requests || {}).some(r=>r && r.operation==='EVIDENCE_PACKET');
      return html`<p class="caption">${t(!live ? 'A recorded source package: no source is checked at preparation; the recorded documents are read locally.' : prepared ? 'A packet is prepared for this book as of an earlier cutoff; the owner binds its source check to that preparation and projects none for this cutoff. A new preparation checks the source as of now.' : 'No source check yet for this book: the first preparation reads the official inventory and fetches only what the workspace does not hold.')}</p>`;
    }
    const inventoryN=Number(check.inventory_request_count) || 0, bodies=Number(check.body_request_count) || 0;
    const docs=check.documents || [];
    const rows=docs.map(d=>tr([d.entity_id,d.form,html`<span class="mono">${d.accession}</span>`,stateLine(String(d.outcome || '').toLowerCase(),{next:''}),d.content_bytes ? BYTE_WORDS(d.content_bytes) : '',said(d.detail)]));
    return html`${kv([[t('Checked at'),whenText(check.checked_at)],[t('As of'),whenText(check.evidence_as_of)],[t('Outcome'),stateLine(String(check.snapshot_status || '').toLowerCase(),{next:''})],[t('Inventory requests'),html`${count(inventoryN)}${check.history_shard_request_count ? html` <span class="sub-cell">${countText(check.history_shard_request_count,'{n} history shard','{n} history shards')}</span>` : ''}`],[t('Body requests'),count(bodies)],[t('Reused'),html`${count(check.reused_local_count)}${check.reused_bytes ? html` <span class="sub-cell">${BYTE_WORDS(check.reused_bytes)}</span>` : ''}`],[t('Fetched'),html`${count(check.fetched_count)}${check.fetched_bytes ? html` <span class="sub-cell">${BYTE_WORDS(check.fetched_bytes)}</span>` : ''}`],[t('Deferred'),count(check.deferred_count)],[t('Failed'),count(check.failed_count)]],'sources-check')}${docs.length ? factsRef(countText(docs.length,'{n} resource of the check','{n} resources of the check'),table([{label:t('Issuer'),type:'text'},{label:t('Form'),type:'text'},{label:t('Accession'),type:'id'},{label:t('Outcome'),type:'status'},{label:t('Bytes'),type:'num'},{label:t('Detail'),type:'text',absorb:true}],rows,'',{report:true,count:rows.length})) : ''}`;
  }
  const trafficWords=(check)=>{ if(!check) return ''; const inventoryN=count(check.inventory_request_count), bodies=Number(check.body_request_count) || 0;
    return bodies===0 ? t('Checked without a download: {n} inventory requests, no body request.',{n:inventoryN}) : t('{n} inventory and {b} body requests: {f} fetched, {r} reused without a request.',{n:inventoryN,b:count(bodies),f:count(check.fetched_count),r:count(check.reused_local_count)}); };
  /* A wide book's units: the owner's cut, each unit's state, Task and check. */
  function unitsTable() {
    const units=unitsOf(); if(units.length<2) return '';
    const scope=Object.fromEntries((S.preview?.scope?.selected_issuers || []).map(i=>[i.entity_id,i]));
    const checked=units.some(u=>u.source_check); // a fact no unit carries takes no column (law 81)
    // the row's action (FT3): a prepared group's packet is ready to read and answer, while the run continues and after it,
    // until its analysis is published (10.9)
    const readyOf=(p)=>String(p?.state || '')==='PREPARED' && Boolean(p?.packet_task_id) && !staleOf(p), reading=units.some(u=>readyOf(unitProgress(u.unit_id)) || staleOf(unitProgress(u.unit_id)));
    const row=(u)=>{ const p=unitProgress(u.unit_id), c=u.source_check || null, stale=staleOf(p), state=stale ? 'superseded' : p ? String(p.state || '').toLowerCase() : (u.prepared_task_id || p?.packet_task_id ? 'prepared' : 'not_started'), task=p?.packet_task_id || u.prepared_task_id || '';
      // U78 (V541): a unit's words are the owner's -- why it failed or went stale, its way on -- the state's hover card;
      // where its sources fell short, its issuers without one by name
      const without=Array.isArray(p?.issuers_without_source) ? p.issuers_without_source : [];
      const said=html`${hint(stateLine(state,{next:''}),p?.detail ? t(p.detail) : '')}${p?.failure_code ? html` <span class="sub-cell">${coded(p.failure_code)}</span>` : ''}${without.length ? html` <span class="sub-cell">${hint(countText(without.length,'{n} without a source','{n} without a source'),without.map(id=>issuerWords(scope[id],id)).join(', '))}</span>` : ''}`;
      // a pending unit's own count while the run continues (10.10); a prepared unit's packet is ready then, whatever the rest of the run does (10.9)
      const work=p?.work ? workOf(p.work) : null, working=work ? html` <span class="sub-cell">${preparingWord(p.work.stage)}${work.count ? html` · ${work.count}` : ''}</span>` : '';
      const ready=readyOf(p) ? btn(t('Read its packet'),'review-use-packet',p.packet_task_id+'|'+(p.packet_unit_id || ''),'text-btn compact') : stale && current() ? btn(t('Prepare again'),'review-prepare','','text-btn compact') : '';
      return [(u.ordered_entity_ids || []).map(id=>issuerWords(scope[id],id)).join(', '),html`<span class="mono">${u.unit_id}</span>`,html`${said}${working}`,task ? html`<span class="mono">${short(task,SHORT.id)}</span>` : '',...(checked ? [c ? when(c.checked_at) : '',c ? count(c.reused_local_count) : '',c ? count(c.fetched_count) : '',c ? count((Number(c.deferred_count) || 0)+(Number(c.failed_count) || 0)) : ''] : []),...(reading ? [ready] : [])]; };
    const packing=S.preview?.coverage?.packing;
    return panel(html`${t('Units')} <span class="num">${count(units.length)}</span>`,t('Units of at most {n} issuers, in priority order; each has its own check and packet.',{n:count(S.preview?.coverage?.unit_limit)}),html`${collectionTable('source-units',units,[{label:t('Issuers'),type:'text',absorb:true},{label:t('Unit'),type:'id'},{label:t('State'),type:'status'},{label:t('Task'),type:'id'},...(checked ? [{label:t('Checked at'),type:'date'},{label:t('Reused'),type:'num'},{label:t('Fetched'),type:'num'},{label:t('Deferred or failed'),type:'num'}] : []),...(reading ? [{label:'',type:'link'}] : [])],row)}${packing ? factsRef(t('How the units were packed'),html`<p>${said(packing.rule || '')}</p>${kv([[t('Rule'),html`<span class="mono">${packing.rules_id || ''}</span>`],[t('Documents per unit'),count(packing.document_capacity_per_unit)],[t('Issuers per unit'),count(packing.issuer_limit_per_unit)],[t('Source counts'),collectionTable('unit-source-basis',Object.entries(packing.source_count_basis || {}),[{label:t('Issuer'),type:'text'},{label:t('Source counts'),type:'text',absorb:true}],([id,basis])=>[issuerWords(scope[id],id),said(basis)])]])}`) : ''}`,'','data-box="table" data-jump="units"');
  }
  /* An issuer's holdings in its reading: its state and counts, its deferred bodies with the cap
   * and the recheck date, its resources' outcomes from the last check, its recorded candidates. */
  function holdingsReading(o) {
    const h=o.holdings, docs=issuerDocs(o.id), candidates=(S.preview?.source_candidates || []).filter(c=>c.entity_id===o.id);
    const facts=kv([[t('Holding'),stateLine(h.state,{next:''})],...(h.unit ? [[t('Unit'),html`<span class="mono">${h.unit}</span>`]] : []),...(h.cik ? [[t('CIK'),html`<span class="mono">${h.cik}</span>`]] : []),[t('Held documents'),count(h.held)],...(h.types.length ? [[t('Filings'),typeWords(h.types)]] : []),...(h.latest ? [[t(h.live ? 'Latest accepted' : 'Latest available'),when(h.latest)]] : []),...(h.bytes ? [[t('Retained'),BYTE_WORDS(h.bytes)]] : [])]);
    const deferred=h.deferred.length ? html`<h3>${t('Deferred')} <span class="num">${count(h.deferred.length)}</span></h3>${collectionTable('issuer-'+o.id+'-deferred',h.deferred,[{label:t('Document type'),type:'text'},{label:t('Accession'),type:'id'},{label:t('Reason'),type:'text',absorb:true}],d=>[d.form,d.accession,d.recheck_after ? t('{o} observed against a {c} cap; checked again after {d}',{o:BYTE_WORDS(d.observed_bytes),c:BYTE_WORDS(d.admitted_cap_bytes),d:when(d.recheck_after)}) : t('{o} observed against a {c} cap; it stands until a larger cap or an explicit scope retries it',{o:BYTE_WORDS(d.observed_bytes),c:BYTE_WORDS(d.admitted_cap_bytes)})])}` : '';
    const outcomes=docs.length ? html`<h3>${t('Last check')} <span class="num">${count(docs.length)}</span></h3><div class="card-list lines">${docs.map(d=>objectRow({state:String(d.outcome || '').toLowerCase(),name:html`${d.form} · <span class="mono">${d.accession}</span>`,why:html`${d.content_bytes ? BYTE_WORDS(d.content_bytes) : ''}${d.content_bytes && d.detail ? ' · ' : ''}${said(d.detail)}`,cls:'evidence-row'},{key:'check:'+d.accession}))}</div>` : '';
    const cands=candidates.length ? html`<h3>${t('Filings')} <span class="num">${count(candidates.length)}</span></h3><div class="card-list lines">${candidates.map(c=>evidenceRow('filing',c,{key:[c.entity_id,c.document_type,c.revision].join('·')}))}</div>` : '';
    return html`<h3>${t('Holdings')}</h3>${facts}${deferred}${outcomes}${cands}`;
  }
  /* A refusal's recovery, the control beside it (law 58): a capacity refusal opens the storage
   * page's cleanup preview; a superseded, expired or changed binding reads the preview again. */
  const RECOVERY=[[/storage\.managed_capacity_exceeded/,()=>html`${link(t('Storage cap'),'settings','button compact',{row:'storageCap'})}${link(t('Open Storage'),'storage','button compact')}`],[/storage\.disk_space_insufficient/,()=>link(t('Open Storage'),'storage','button compact')],[/preparation_superseded|brief_source_stale|REFUSED_PREPARATION_(EXPIRED|SUPERSEDED|BINDING_CHANGED)|REFUSED_ACQUISITION_WINDOW_CLOSED/,()=>btn(t('Read source preview again'),'review-preview','','button compact')]];
  const recoveryFor=(code)=>{ const text=String(code || ''); const hit=RECOVERY.find(([re])=>re.test(text)); return hit ? hit[1]() : ''; };
  /* The Facts panel's Sources section: the mode, the families, the binding and the claims. */
  function sourcesFacts() {
    const p=S.preview, inv=inventory();
    if(p?.status!=='EVIDENCE_PREPARATION_READY') return [];
    return [{title:t('Sources'),body:html`<p>${t(inv?.live ? 'What this workspace holds for the book\'s issuers, from the sealed commitments; a held body is reused without a download when the official inventory still selects it.' : 'What the recorded source package holds for the book\'s issuers at this cutoff; reading fetches nothing.')}</p>${inv?.live ? html`<p>${t('What the acquisition stage of the last preparation found at the source: its traffic, and every resource\'s outcome.')}</p>` : ''}${kv([[t('Source mode'),coded(p.source_mode)],[t('Families'),(p.approved_source_families || []).map(f=>codeWords(f)).join(', ')],[t('Work'),coded(p.source_work)],[t('Evidence as-of'),whenText(p.evidence_as_of)],[t('Acquisition deadline'),whenText(p.acquisition_deadline)],[t('Binding'),mono(p.preparation_binding_hash,SHORT.hash)],[t('Candidate documents'),count(p.recorded_candidate_count)],[t('Managed model required'),t(p.managed_model_required ? 'yes' : 'no')]])}${inv?.claim ? html`<p class="caption">${said(inv.claim)}</p>` : ''}<p class="caption">${said(p.claim)}</p>`}];
  }
  /* ---- the Sources page (round F2, redesign-2 section 7): the evidence store's management in
   * the Data page's pattern -- the store's state in one sentence and four figures, then Set up
   * (the prerequisites, each met or not, with the exact step the product cannot run itself),
   * Prepare (the preview as facts, the last check), Keep current (the version, what is due,
   * reuse, the preparations, the runs), the tables (issuers, groups, documents) and the storage
   * line. Everything comes from the preview, the projection, Task Control and the storage
   * readback; what the backend does not project is said in the page's words. ---- */
  const previewOf=()=>S.preview?.status==='EVIDENCE_PREPARATION_READY' ? S.preview : null;
  const modelShort=(id)=>String(id || '').split('/').pop();
  const familyWords=(p)=>(p?.approved_source_families || []).map(f=>codeWords(f)).join(' · ');
  /* The four prerequisites of the store, each met or not, with the step that meets it: the retrieval pack first, as the
   * authority binds its recipe (V405); `setup` is the Host's steps for the row, when the preview refused for it. */
  function setupRows() {
    const p=S.preview, ok=previewOf(), v=S.view, st=v?.state || '', code=String(p?.failure_code || ''), setup=p?.setup || {};
    const retired=/matter_selection_policy_retired/.test(code);
    const authorityMissing=retired || st==='EVIDENCE_AUTHORITY_NOT_ADMITTED' || /NO_ADMITTED_EVIDENCE_AUTHORITY/.test(code);
    const runtimeMissing=/evidence_task_adapter_absent|NO_ADMITTED_EVIDENCE_RUNTIME/.test(code);
    const ms=ok?.matter_selection || null, allowance=ms?.per_session_allowance || null;
    const adm=ok?.admission || null, window=adm?.acquisition_window_seconds!=null ? Math.round(adm.acquisition_window_seconds/60) : null; // U57 (A2): the admitted bounds, as the preview states them
    const policy=ok ? [ms?.method ? codeWords(ms.method) : '',allowance ? t('{r} reads · {w} windows · {b} a window',{r:count(allowance.reads),w:count(allowance.windows),b:BYTE_WORDS(allowance.window_bytes)}) : '',window!==null && Number.isFinite(window) ? t('a preparation submits within {n} min of its preview',{n:count(window)}) : '',adm?.maximum_documents_per_issuer ? t('at most {n} documents an issuer',{n:count(adm.maximum_documents_per_issuer)}) : '',adm?.maximum_document_bytes ? t('at most {b} a document',{b:BYTE_WORDS(adm.maximum_document_bytes)}) : '',v?.evidence_expires_at ? t('the current version lives until {t}',{t:dayWord(v.evidence_expires_at)}) : ''].filter(Boolean).join(' · ') : '';
    const authority=
      {id:'authority',met:!authorityMissing && (Boolean(ok) || (!runtimeMissing && Boolean(v) && st!=='EVIDENCE_AUTHORITY_NOT_ADMITTED')),name:t('Evidence authority'),why:retired ? t('installed under a retired matter selection') : authorityMissing ? t('no recorded package is bound into this workspace and no live acquisition is admitted') : ok ? t(ok.source_mode==='LIVE_OFFICIAL' ? 'live official acquisition admitted' : 'a recorded package is bound into this workspace') : t('admitted'),command:String(p?.setup_help || 'scripts/materialize_evidence_cro_authority.py --help'),shows:p?.setup_help ? 'The command the Host names' : 'Its options',sentence:t('One script writes the recorded package into the workspace and binds it into its manifest, or admits the live acquisition; the product cannot run it. Run it in a terminal, then read again.'),setup:setup.authority || null,
        setupWords:t(retired ? 'Set up under a retired matter selection: bind it again under the integrated selection, in a terminal at the product folder.' : 'The retrieval pack comes first: the authority binds its recipe. Each step runs in a terminal at the product folder.')};
    return [
      {id:'pack',met:Boolean(ok?.retrieval?.encoder),name:t('Retrieval pack'),why:ok?.retrieval?.encoder ? html`${modelShort(ok.retrieval.encoder.model_id)} · ${modelShort(ok.retrieval.reranker?.model_id)}${ok.retrieval.runtime ? html` · ${codeWords(ok.retrieval.runtime)}` : ''}` : runtimeMissing ? t('no evidence runtime is admitted: the retrieval pack is not installed, or not bound') : t('not stated until the authority is admitted'),command:'scripts/install_retrieval_pack.py --status',shows:'What is installed',sentence:t('The pinned encoder and reranker are installed into the application model store by one script (its --install <recipe> --network fetches the two packs once; installing changes no workspace); the product cannot run it. Run it in a terminal at this machine, then read again.'),setup:setup.pack || null,
        setupWords:t('Installed once for this machine, into the application model store; it changes no workspace. Each step runs in a terminal at the product folder.')},
      authority,
      {id:'mode',met:Boolean(ok),name:t('Mode and families'),why:ok ? html`${codeWords(ok.source_mode)}${familyWords(ok) ? html` · ${familyWords(ok)}` : ''} · ${t(ok.source_mode==='LIVE_OFFICIAL' ? 'network consent given · official acquisition admitted' : 'no network is used')}` : t('not stated until the authority is admitted'),command:'',sentence:''},
      {id:'policy',met:Boolean(ok),name:t('Policy'),why:policy || t('not stated until the authority is admitted'),command:'',sentence:''}];
  }
  /* The store's sentence and its one action (the head of the Sources page). */
  function storeWords() {
    const p=S.preview, ok=previewOf(), v=S.view, st=v?.state || '';
    if(S.refusal) return {sentence:t('Evidence review is not available for this workspace.'),primary:null};
    if(!p) return {sentence:t('Reading the sources.'),primary:null};
    const unmet=setupRows().filter(r=>!r.met);
    if(!ok && unmet.length) return {sentence:t('Not set up: {what}.',{what:unmet.map(r=>r.name).join(', ')}),primary:null};
    if(!ok) return {sentence:t('The sources cannot be prepared yet.'),primary:null};
    if(st==='EVIDENCE_REFRESH_IN_PROGRESS'){ const c=cycle(); return {sentence:c.sentence,primary:c.primary}; }
    const again=Boolean(v?.evidence_as_of) || ['ALTERNATIVE_EVIDENCE_EXPIRED','ALTERNATIVE_EVIDENCE_SUPERSEDED','REVIEW_PUBLISHED'].includes(st);
    return {sentence:'',primary:current() ? {word:t(again ? 'Prepare again' : 'Prepare sources'),action:'review-prepare',value:''} : null}; // W: a store that can prepare is said by its four figures (the mode, the documents held, the version, the pack); a sentence restating them is a second copy
  }
  /* The four figures (the Data page's rail). */
  function storeRail() {
    const ok=previewOf(), inv=inventory(), v=S.view, rows=setupRows(), firstUnmet=rows.find(r=>r.id!=='pack' && !r.met), packRow=rows.find(r=>r.id==='pack');
    const mode=ok ? t(inv?.live ? 'live official sources' : 'recorded package') : t('not set up');
    const modeNote=ok ? (familyWords(ok) || t('no family stated')) : (firstUnmet ? firstUnmet.why : '');
    const held=inv ? inv.rows.reduce((n,r)=>n+r.held,0) : null;
    const docs=inv ? (inv.live && inv.bytes ? html`${count(held)} <span class="num-unit">${BYTE_WORDS(inv.bytes)}</span>` : count(held)) : t('not read');
    // W: how many issuers hold a document is the holdings table's caption, right under the figures -- said once; a recorded
    // package's documents are the package's, not the workspace's retained store (Retained documents below; the main line's reading, 2026-09-30)
    const docsNote=inv && !inv.live ? t('in the recorded package') : '';
    const expired=v?.state==='ALTERNATIVE_EVIDENCE_EXPIRED';
    const version=v?.evidence_as_of ? dayWord(v.evidence_as_of) : t('none');
    const versionNote=v?.evidence_as_of ? t(expired ? 'expired {t}' : 'expires {t}',{t:dayWord(v.evidence_expires_at)}) : t('nothing prepared');
    const pack=ok?.retrieval?.encoder ? t('installed') : t('not stated');
    const packNote=ok?.retrieval?.encoder ? modelShort(ok.retrieval.encoder.model_id)+' · '+modelShort(ok.retrieval.reranker?.model_id) : (packRow.met ? '' : packRow.why);
    return rail(html`${stat(t('Source mode'),mode,modeNote,true)}${stat(t('Documents held'),docs,docsNote,!inv)}${stat(t('Evidence version'),version,versionNote,true)}${stat(t('Retrieval pack'),pack,packNote,true)}`);
  }
  /* U68 (V405, the user's R2-02): the Host's steps for one prerequisite, each as a person types it at the product
   * folder with what the Host knows filled in -- the environment when it is absent, Check, Set up with what is left to
   * choose and what to do before it, Read again. The Host's sentences are its English, worded through the dictionary. */
  const setupStep=(name,note,command,under='')=>html`<li><p><strong>${name}</strong>${note ? html` · ${note}` : ''}</p>${command ? html`<pre class="code-block setup-command owner-text">${command}</pre>` : ''}${under}</li>`;
  function setupSteps(id,s) {
    const choose=Array.isArray(s.choose) && s.choose.length ? html`<p class="caption">${t('Left for you to choose')}</p>${kv(s.choose.map(c=>[html`<code class="owner-text">${c.arg}</code>`,html`<span class="owner-text">${t(c.why)}</span>`]),'setup-choices')}` : '';
    const install=id==='pack' && s.recipe ? t('recipe {recipe}, fetched once over the network',{recipe:s.recipe}) : s.before ? html`<span class="owner-text">${t(s.before)}</span>` : '';
    return html`<ol class="setup-steps">${[
      s.environment ? setupStep(t('Create the retrieval environment'),t('not created yet'),s.environment) : '',
      s.check ? setupStep(t('setup|Check'),t(id==='pack' ? 'says whether it is installed' : 'installs nothing'),s.check) : '',
      s.install ? setupStep(t('Set up'),install,s.install,choose) : '',
      html`<li>${btn(t('Read again'),'review-preview','','button compact')}${s.before ? html` <span class="caption">${t('once the install ends')}</span>` : ''}</li>`]}</ol>`;
  }
  /* Set up: the checklist, whole while a prerequisite is unmet, one line when all are met. */
  function setupSection(rows) {
    return checkList({title:t('Set up'),key:'setup',rowCls:'evidence-row setup-row',cls:'setup-details',attrs:'data-overview="setup"',rows:rows.map(r=>({...r,detail:r.setup ? html`<div class="setup-card" data-setup="${r.id}"><p>${r.setupWords}</p>${setupSteps(r.id,r.setup)}</div>` : r.command ? html`<div class="setup-card"><p>${r.sentence}</p><p class="caption">${t(r.shows)}</p><pre class="code-block setup-command">${r.command}</pre>${btn(t('Read again'),'review-preview','','button compact')}</div>` : ''}))});
  }
  /* Prepare: what one preparation reads and does, as facts; the last check's result. */
  function workEstimate(p) {
    if (!p?.work_estimate) return '';
    const {unit_count:n,unit_limit:limit}=p.coverage || p.preparation_binding || {};
    if (!Number.isInteger(n) || n<0 || !Number.isInteger(limit) || limit<1) return t('The owner has not stated the work estimate’s unit count and issuer limit.');
    const units=countText(n,'{n} bounded source/canonicalization/retrieval unit','{n} bounded source/canonicalization/retrieval units');
    return t('One Task of {units} of at most {limit} issuers, in priority order; no analyst or CRO inference. Candidate count is not accepted document count; time and size estimates are unavailable. The matter selection reads under one per-session allowance and leaves its pending scope explicit; see matter_selection.',{units,limit:count(limit)});
  }
  function prepareSection() {
    const p=previewOf(), words=modeWords(p), inv=inventory(), sc=p.scope || {};
    const sentence=words.kind==='recorded' ? t('One local Task reads the recorded documents, canonicalizes them, builds the retrieval index and selects admitted passages; nothing is fetched and no analyst or CRO model runs.') : words.kind==='live' ? t('One Task acquires official documents from the approved families over the network under the admitted policy, then canonicalizes them, builds the retrieval index and selects admitted passages; no analyst or CRO model runs.') : words.prepare;
    const scope=html`${countText((sc.selected_issuers || []).length,'{n} issuer selected','{n} issuers selected')}${(sc.mapping_failures || []).length ? html` · ${countText(sc.mapping_failures.length,'{n} mapping failure','{n} mapping failures')}` : ''}`;
    const units=wide() ? unitsOf().filter(u=>u.source_check) : [];
    const check=units.length ? factsRef(countText(units.length,'{n} group check','{n} group checks'),html`${units.map(u=>html`<h3>${t('Group')} <span class="mono">${u.unit_id}</span></h3><p>${trafficWords(u.source_check)}</p>${checkFacts(u.source_check,inv.live)}`)}`)
      : p.source_check ? factsRef(t('The last check'),html`<p>${trafficWords(p.source_check)}</p>${checkFacts(p.source_check,inv.live)}`)
      : hint(t('none'),String(checkFacts(null,inv.live)).replace(/<[^>]+>/g,'')); // the reason a check is absent, on hover
    const facts=kv([[t('Scope'),scope],[t('Cutoff'),dayWord(p.evidence_as_of)],[hint(t('Candidates'),t('recorded locally; not yet admitted passages')),count(p.recorded_candidate_count || 0)],[t('Managed model'),t(p.managed_model_required ? 'required' : 'not required')],...(p.work_estimate ? [[t('Owner\'s estimate'),workEstimate(p)]] : []),[t('Last check'),check],...campaignRow(p.campaign)],'kv-columns');
    return panel(t('Prepare'),sentence,facts,'','data-overview="prepare"'); // B3: a section of the column, its facts in the lane's columns, the prose its (i)
  }
  const campaignRow=(c)=>c ? [[t('Campaign'),html`<span class="mono">${c.campaign_id}</span> · ${t('{a} attempts and {b} left',{a:count(c.attempts_remaining),b:BYTE_WORDS(c.bytes_remaining)})}${c.unsettled_reservations ? html` · ${countText(c.unsettled_reservations,'{n} reservation unsettled','{n} reservations unsettled')}` : ''}`]] : [];
  /* The reuse accounting (A7): this preparation's selections and documents from its own sealed receipts and checks; the
   * service's counters since it started, said as what they are, in Facts. */
  function reuseWords(r) {
    const n=(r.selections_reused || 0)+(r.selections_made || 0);
    const counters=Object.entries(r.service_counters || {}).map(([k,v])=>[codeWords(k),count(v)]);
    return html`${t('{a} of {n} selections reused',{a:count(r.selections_reused || 0),n:count(n)})}${r.documents_reused!=null ? html` · ${t('{a} documents reused, {b} fetched',{a:count(r.documents_reused),b:count(r.documents_fetched || 0)})}` : ''}${counters.length ? factsRef(t('Reuse since the service started'),html`<p class="caption owner-text">${said(r.basis)}</p>${kv(counters)}`) : ''}`;
  }
  /* Keep current: the version and its clock, what is due, reuse -- a section of the column (B3). */
  function updateSection() {
    const p=previewOf(), v=S.view, inv=inventory(), st=v?.state || '', check=p.source_check || null;
    const word=st==='ALTERNATIVE_EVIDENCE_EXPIRED' ? 'expired' : st==='ALTERNATIVE_EVIDENCE_SUPERSEDED' ? 'superseded' : v?.evidence_as_of ? 'prepared' : 'pending';
    const version=v?.evidence_as_of ? html`${stateLine(word,{next:''})} <span class="sub-cell">${dayWord(v.evidence_as_of)} · ${t(word==='expired' ? 'expired {t}' : 'expires {t}',{t:dayWord(v.evidence_expires_at)})}</span>` : html`${stateLine('pending',{next:''})} <span class="sub-cell">${t('nothing prepared')}</span>`;
    const {deferred,due}=dueOf(p,inv);
    const standing=deferred.filter(d=>!d.recheck_after).length; // since W6 a deferral carries no recheck date (10.8)
    const dueWords=due.length ? countText(due.length,'{n} deferred document past its recheck date','{n} deferred documents past their recheck date') : standing ? countText(standing,'{n} deferred until a larger cap or an explicit scope','{n} deferred until a larger cap or an explicit scope') : deferred.length ? t('{n} deferred, none due yet',{n:count(deferred.length)}) : inv?.live ? t('nothing is due') : hint(t('none'),t('a recorded package does not age; a new package is a new setup'));
    const selection=(v?.eligible_versions || []).length>1 ? [[t('Selection'),html`${countText(v.eligible_versions.length,'{n} analysis eligible','{n} analyses eligible')} · ${btn(t('Choose the analysis'),'review-live-item','analysis','text-btn')}`]] : [];
    const reuse=p.reuse ? reuseWords(p.reuse) : hint(t('unchanged documents are reused'),t('unchanged documents are reused, not fetched again')); // U57 (A7): the preparation's own accounting
    const facts=kv([[t('Version'),version],[t('Due'),dueWords],[t('Reuse'),reuse],...selection],'kv-columns');
    return panel(t('Keep current'),t('An update is a new preparation under a new cutoff.'),facts,'','data-overview="update"');
  }
  /* The deferred documents of the inventory, and those past their recheck by the owner's clock
   * (never the browser's). */
  function dueOf(p,inv) {
    const clock=p?.evidence_as_of ? Date.parse(p.evidence_as_of) : NaN;
    const deferred=(inv?.rows || []).flatMap(r=>(r.deferred || []).map(d=>({...d,entity:r.entity_id})));
    return {deferred,due:deferred.filter(d=>d.recheck_after && Number.isFinite(clock) && Date.parse(d.recheck_after)<=clock)};
  }
  /* The preparations (W; law 114, a list's facts are columns): the documents past their recheck,
   * then the evidence Tasks with their stages and binding -- a list on the spine, where its facts
   * have their columns; the side keeps the properties. */
  function preparationsSection() {
    const {due}=dueOf(previewOf(),inventory()), b=boundFor();
    const dueRows=due.length ? collectionTable('due-documents',due,[{label:t('Issuer'),type:'text'},{label:t('Document type'),type:'text'},{label:t('Accession'),type:'id'},{label:t('Reason'),type:'text',absorb:true}],d=>[d.entity,d.form || '',d.accession || '',t('{o} observed against a {c} cap; due since {d}',{o:BYTE_WORDS(d.observed_bytes),c:BYTE_WORDS(d.admitted_cap_bytes),d:when(d.recheck_after)})]) : '';
    const boundRows=[...b.prepare].map(id=>({id,...taskState(id),bound:true}));
    const otherRows=prepareTasks().filter(x=>!b.prepare.has(x.task_id)).map(x=>({id:x.task_id,lifecycle:x.lifecycle,verified:x.verified_stage_count,total:x.total_stage_count,bound:false}));
    const rows=[...boundRows,...otherRows];
    if(!rows.length && !due.length) return '';
    const bindingWords=(x)=>x.bound ? t('bound: admitted for this book from this page, or its packet read for it') : t('unbound · a workspace Task listed by Task Control; reading its packet for this book asks the owner');
    const ran=(id)=>{ const v=(Data.tasks() || []).find(x=>x.task_id===id); return v?.running_since || v?.last_activity_at || ''; };
    const tasks=rows.length ? collectionTable('preparation-tasks',rows,[{label:t('Preparation'),type:'date'},{label:t('State'),type:'status'},{label:t('Verified stages'),type:'num'},{label:t('Binding'),type:'text'},{label:t('Task'),type:'id'},{label:t('Action'),type:'link'}],x=>[ran(x.id) ? when(ran(x.id)) : '',x.lifecycle ? stateLine(String(x.lifecycle).toLowerCase(),{next:''}) : '',`${x.verified} / ${x.total}`,hint(t(x.bound ? 'Bound' : 'Unbound'),bindingWords(x)),html`<span class="mono">${short(x.id,SHORT.id)}</span>`,html`${btn(t('Follow'),'review-watch',x.id,'text-btn')}${x.lifecycle==='SUCCEEDED' ? btn(t('Read its packet for this book'),'review-use-task',x.id,'text-btn') : ''}`]) : '';
    return panel(rows.length ? html`${t('Preparations')} <span class="num">${count(rows.length)}</span>` : t('Preparations'),'',html`${dueRows}${tasks}`,'','data-overview="preparations"');
  }
  /* The runs of the evidence Tasks (E3): the latest three, then all of them, as the Overview's
   * Activity. */
  /* The store's trace (the reviews, 2026-09-24: the preparations and the runs pushed the page three
   * screens long): one line with their counts, folded; open by itself while a preparation moves, has
   * failed or has a document due; the reader's press decides after. */
  function executionSection() {
    const b=boundFor(), preps=prepareTasks(), runs=(typeof Data.runsOf==='function' ? Data.runsOf('task') : []).filter(r=>r.kind===EVIDENCE_KIND), {due}=dueOf(previewOf(),inventory());
    const nPrep=new Set([...b.prepare,...preps.map(x=>x.task_id)]).size;
    if(!nPrep && !runs.length && !due.length) return '';
    const attention=due.length>0 || preps.some(x=>!['SUCCEEDED','CANCELLED'].includes(String(x.lifecycle || '').toUpperCase()));
    const open=S.executionOpen ?? attention; S.executionShown=open;
    const head=btnAttrs(html`${icon('chevron')}<span>${t('Preparations and runs')}</span><b class="num">${count(nPrep)} · ${count(runs.length)}</b>`,'review-execution','','lobby-fold',html`aria-expanded="${open}"`);
    return html`<section class="execution-record" data-overview="execution"><div class="group-head lobby-head">${head}${due.length ? html`<span class="lobby-note">${countText(due.length,'{n} document due','{n} documents due')}</span>` : ''}</div>${open ? html`${preparationsSection()}${runsSection()}` : ''}</section>`;
  }
  function runsSection() {
    const list=(typeof Data.runsOf==='function' ? Data.runsOf('task') : []).filter(r=>r.kind===EVIDENCE_KIND);
    return activityFeed({title:t('Runs'),list,shown:ACTIVITY_SHOWN,open:S.runsOpen,render:pagedRuns,more:'review-runs-all',attrs:'data-overview="runs"'});
  }
  /* The documents of the book's sources (B3): the read packet's admitted index when one is read, else
   * the recorded package's candidates; a live store lists them once a preparation admits them (A6).
   * The issuers' table opens them under their issuer; the caption says which they are. */
  function sourceDocuments() {
    const p=previewOf(), facts=packetJson(), inv=inventory();
    const fromPacket=Array.isArray(facts?.document_index) ? facts.document_index : null;
    const rows=fromPacket ? fromPacket.map(d=>({entity:d.entity_id || '',form:d.document_type || '',title:d.title || '',when:d.time?.accepted_at || d.time?.available_at || d.time?.published_at || '',chars:d.character_count,status:'admitted'})) : (p?.source_candidates || []).map(c=>({entity:c.entity_id || '',form:c.document_type || '',title:c.revision || '',when:c.available_at || '',chars:null,status:'candidate'}));
    const unreadable=Boolean(bundleOf('analyst')) && !fromPacket; // a packet is read but its block could not be read on this page: said, never an empty list
    const caption=fromPacket ? t('{n} admitted by the read packet · {r} rejected',{n:count(rows.length),r:count((facts.document_rejections || []).length)}) : unreadable ? t('The read packet\'s document index could not be read on this page; its passages are read on the Reading page.') : rows.length ? t(inv?.live ? '{n} candidate documents at the source' : '{n} candidate documents of the recorded package',{n:count(rows.length)}) : inv?.live ? t('The documents are listed once a preparation admits them; the holdings count what is retained per issuer.') : t('No document is listed for this book.');
    return {rows:unreadable ? [] : rows,caption,fromPacket:Boolean(fromPacket)};
  }
  /* The retained documents (U57, A6): every registered issuer's in this workspace, newest accepted first, the Host's
   * pages of table-rows (`EVIDENCE_DOCUMENTS`) turned by its page number; read once a book, again on a page turn. */
  async function readDocuments(page=1) {
    if(S.retainedPending) return; const rev=S.revision;S.retainedPending=true;
    // the first page is the Host's default: a page number is sent only past it (the Local Web parses it as text, the closing stills)
    try{ const b=await Data.read(Data.route('EVIDENCE_DOCUMENTS')+(page>1 ? '?'+new URLSearchParams({documents_page:page}) : '')); if(rev===S.revision)S.retained={status:'ready',value:b}; }
    catch(e){ if(rev===S.revision && e.name!=='AbortError')S.retained={status:'refused',code:String(e?.message || e)}; }
    finally{ if(rev===S.revision){S.retainedPending=false;repaintMain();} }
  }
  function documentsSection() {
    if(!S.retained){ if(typeof queueMicrotask==='function') queueMicrotask(()=>void readDocuments(1)); return ''; }
    const x=S.retained; // `S.documents` is the packets' document index
    if(x.status!=='ready') return notRead(t('Retained documents not read'),x.code,'',btn(t('Read again'),'review-documents-page','1','button compact'));
    const b=x.value, docs=b.documents || [];
    const refusedRows=b.refused_documents || [],refused=refusedRows.length ? panel(t('Unreadable retained documents'),'',collectionTable('refused-documents',refusedRows,[{label:t('Recorded reference'),type:'text'},{label:t('Reason'),type:'text',absorb:true},{label:t('Next step'),type:'text'}],r=>[hashCell(r.artifact_hash),html`<span data-tip="${r.failure_code}">${literalWords(t(r.detail || CODE_LINES[r.failure_code] || ''))}</span>`,prerequisiteWays(r.next_requests)]),'','data-box="table"') : '';
    const recordedHere=inventory() && !inventory().live; // the figures above count the recorded package; this store is the official sources'
    if(!docs.length) return html`${refused}${refusedRows.length ? '' : panel(t('Retained documents'),'',emptyState(t(recordedHere ? 'This workspace retains no official-source document; the documents above are the recorded package\'s' : 'No document is retained in this workspace')))}`;
    const rows=docs.map(d=>tr([html`<span class="mono">${d.entity_id}</span>`,html`${d.title ? html`<span class="owner-text">${d.title}</span>` : String(d.document_type)}<span class="sub-cell">${d.document_type} · <span class="mono">${short(d.revision,SHORT.id)}</span></span>`,d.report_period_end ? dayWord(d.report_period_end) : '',d.accepted_at ? when(d.accepted_at) : '',BYTE_WORDS(d.content_bytes)]));
    const docTurns={page:(b.page || 1)-1,pages:b.page_count || 1,prev:['review-documents-page',String((b.page || 1)-1)],next:['review-documents-page',String((b.page || 1)+1)]}, foot=pager({total:b.total,one:'{n} document',many:'{n} documents',...docTurns});
    return html`${refused}${panel(t('Retained documents'),t('What this workspace holds, from its sealed references; nothing was fetched, and no source\'s freshness is claimed.'),html`${pagerTop(docTurns)}${table([{label:t('Issuer'),type:'id'},{label:t('Document'),type:'text',absorb:true},{label:t('Period end'),type:'date'},{label:t('Accepted'),type:'date'},{label:t('Size'),type:'num'}],rows,'',{report:true,countLine:false,classes:'compact'})}${foot}`,'','data-box="table" data-jump="documents"')}`;
  }
  const documentsPage=(v)=>{ const n=Number(v); if(Number.isInteger(n) && n>=1) void readDocuments(n); };
  /* The storage line: the evidence category's figures from the readback, read once. */
  async function readStorage() {
    if(S.storage || S.storagePending) return; const rev=S.revision;S.storagePending=true;
    try{ const b=await Data.read('/api/workspace/storage',true); if(rev===S.revision)S.storage={status:'ready',value:b}; }
    catch(e){ if(rev===S.revision && e.name!=='AbortError')S.storage={status:'refused',code:String(e?.message || e)}; }
    finally{ if(rev===S.revision){S.storagePending=false;repaintMain();} }
  }
  const workspaceWide=(x)=>x?.status==='refused' && /research_workspace\./.test(String(x.code || ''));
  function storageLine() {
    if(!S.storage){ if(typeof queueMicrotask==='function') queueMicrotask(()=>void readStorage()); return ''; }
    const x=S.storage, e=x.status==='ready' ? x.value?.evidence : null;
    const code=x.code ? (String(x.code).match(/[a-z_]+\.[a-z_]+/) || [String(x.code).split(':')[0]])[0] : ''; // the owner's code, not the error's class
    // B3: a section of the column; a category the owner does not report is not shown (law 81), a refusal is said with its code
    // the reviews (2026-09-24): a refusal is a notice with its way -- read again -- the code in its (i), never a loose paragraph
    // a refusal about the workspace itself (its manifest) leads the page and says what it leaves out (the user's phase 6 reading)
    if(!e || typeof e!=='object') return code ? workspaceWide(x) ? banner(t('The workspace manifest could not be read'),html`${explain(code) || codeWords(code)} ${t('The sections below were read from the evidence owner; only the storage figures are missing.')}${infoMark(code)}`,'warning',btn(t('Read again'),'review-storage','','button compact')) : banner(t('Evidence storage not read'),html`${explain(code) || codeWords(code)}${infoMark(code)}`,'warning',btn(t('Read again'),'review-storage','','button compact')) : '';
    const sum=e.summary && typeof e.summary==='object' ? e.summary : e;
    return panel(t('Storage'),'',html`<p>${t('originals')} ${BYTE_WORDS(sum.source_object_bytes || 0)} · ${t('canonical texts')} ${BYTE_WORDS(sum.source_blob_bytes || 0)} · ${t('vectors')} ${BYTE_WORDS(sum.vector_payload_bytes || 0)} · ${t('indexes')} ${BYTE_WORDS(sum.index_payload_bytes || 0)} · ${t('sealed records')} ${BYTE_WORDS(sum.sealed_artifact_bytes || 0)}</p>`);
  }
  /* U78 (V541, V546): when units fell short of sources under the recorded package, the person's ways on as the Host words
   * them -- official SEC acquisition, their decision, first; a package only where one covers every unit's issuers, its
   * steps as Set up shows an authority's -- and once, what a recorded package can review. */
  function waysSection() {
    const w=S.view?.source_ways; if(!w || typeof w!=='object') return '';
    const o=w.official || null, pk=w.package || null;
    const official=o ? html`<ol class="setup-steps">${setupStep(t('Official SEC acquisition'),o.before ? html`<span class="owner-text">${t(o.before)}</span>` : '',o.command || '')}</ol>` : '';
    const packaged=pk ? html`<p><strong>${t('A recorded package for this book')}</strong>${pk.covers ? html` · <span class="owner-text">${t(pk.covers)}</span>` : ''}</p>${setupSteps('authority',pk)}` : '';
    return panel(t('Ways on'),'',html`${w.package_rule ? html`<p class="owner-text">${t(w.package_rule)}</p>` : ''}${official}${packaged}`,'','data-overview="ways"');
  }
  function sourcesPage() {
    const p=S.preview, ok=previewOf(), inv=inventory(), rows=setupRows(), unmet=rows.filter(r=>!r.met); // the checklist waits for the preview: a first paint without it would open the fold, and the patch keeps what a paint opened
    // a refusal that is not a prerequisite (the transport, a typed refusal) is still said once
    const other=!p ? '' : p.status==='READ_REFUSED' && !unmet.length ? refusal({code:p.failure_code,reason:explain(p.failure_code)},'warning',{word:t('Source preview refused'),action:recoveryFor(p.failure_code)}) : p.status!=='EVIDENCE_PREPARATION_READY' && !unmet.length ? refusal({code:p.failure_code || p.status,reason:said(p.explanation) || explain(p.failure_code) || '',next:p.next_action},'warning',{word:t('Sources cannot be prepared yet'),action:recoveryFor(p.failure_code)}) : '';
    const setup=p ? setupSection(rows) : '';
    // N4: a store that is not set up is its checklist alone -- no figure of a store that does not exist (law 81)
    if(S.view?.state==='EVIDENCE_AUTHORITY_NOT_ADMITTED' && !S.refusal) return html`${setup}${other}`;
    // B3 (item 5): one column -- Set up one line at the top (open while a prerequisite is unmet), the store's figures, Prepare and Keep current, the issuers and their documents, the units, the preparations, the runs, the storage
    const holdings=inv ? holdingsTable(inv) : '', units=inv ? unitsTable() : '', documents=ok ? documentsSection() : '';
    const jumps=pageJumps([[t('Issuers'),'[data-jump="issuers"]',Boolean(String(holdings))],[t('Units'),'[data-jump="units"]',Boolean(String(units))],[t('Retained documents'),'[data-jump="documents"]',Boolean(String(documents))]]);
    return html`<div class="overview-main sources-flow">${setup}${workspaceWide(S.storage) ? storageLine() : ''}${storeRail()}${jumps}${other}${waysSection()}${workSection('evidence-stream')}${ok ? prepareSection() : ''}${ok ? updateSection() : ''}${holdings}${units}${documents}${ok ? executionSection() : ''}${workspaceWide(S.storage) ? '' : storageLine()}</div>`;
  }
  /* ---- the two handoffs ---- */
  async function getBundle(role,request=null,stay=false,quiet=false) {
    if(!current())return;
    if(role==='cro' && continuationRequired()){S.error=said(S.view.explanation);render();return;}
    if(role==='analyst'&&!S.packetTask) {S.error='Choose a prepared evidence Task.';render();return;}
    // the page's own read names what the page shows; a composed request is read with its fields
    const fields=request ? requestFields(request) : {...S.selector,...(role==='analyst' ? {task_id:S.packetTask,...(S.packetUnit ? {evidence_unit_id:S.packetUnit} : {})} : {})};
    S.busy='bundle';S.error='';S.pending=null;const ticket=++S.bundleRev, gen=S.nav, asked={key:S.key,pin:S.pin,task:S.packetTask,unit:S.packetUnit || '',selector:{...S.selector},readGen:S.readGen};repaintMain();
    try{
      const doc=await Data.readDocument(Data.route(role==='analyst' ? 'EVIDENCE_PACKET' : 'CRO_REVIEW_DOSSIER')+'?'+new URLSearchParams(fields));
      if(!doc.value.submission_template)throw Error(doc.value.failure_code || doc.value.detail || doc.value.disposition || 'Bundle unavailable');
      // the owner answered for the book, Task and unit it was asked about: that binding is
      // recorded whatever the page shows now, and the bundle is adopted only into the exact
      // context it was asked for (the same book, pin and, for a packet, the same Task and unit)
      if(role==='analyst') bind('prepare',asked.task,asked.key,asked.selector,packetIdentity(doc.value));
      if(ticket!==S.bundleRev || gen!==S.nav || S.key!==asked.key || S.pin!==asked.pin || S.readGen!==asked.readGen || (role==='analyst' && (S.packetTask!==asked.task || S.packetUnit!==asked.unit))) return;
      const bundle={...doc,key:asked.key,task:role==='analyst' ? asked.task : '',unit:role==='analyst' ? asked.unit : '',readGen:asked.readGen};
      // a packet asked for without a unit that the product answered for one unit: the product
      // named the unit, the page records it (the drafts follow; a unit-less draft is offered back)
      if(role==='analyst' && !asked.unit && doc.value.coverage_unit?.unit_id){S.packetUnit=String(doc.value.coverage_unit.unit_id);bundle.unit=S.packetUnit;loadDrafts();writeRoute(routeUpdate());}
      if(role==='analyst'){S.packet=bundle;adoptExcerpts(doc.value);}else S.dossier=bundle;
      S.role=role;if(!quiet)S.roleChosen=role;if(!stay && app.page!=='handoff'){app.page='handoff';pushRoute(routeUpdate({page:'handoff'}));}
    }catch(e){if(ticket===S.bundleRev && gen===S.nav && S.key===asked.key && S.pin===asked.pin && S.readGen===asked.readGen){S.error=e.message;}}
    finally{if(ticket===S.bundleRev){S.busy='';repaintMain();}}
  }
  function packetChoice() {
    const b=boundFor(), discovered=prepareTasks().filter(v=>v.lifecycle==='SUCCEEDED');
    const options_=[...new Set([...b.prepare,...discovered.map(v=>v.task_id)])].map(id=>{const s=taskState(id);return [id,`${id} · ${s.verified} / ${s.total}${b.prepare.has(id) ? ' · '+t('bound to this book') : ' · '+t('unbound workspace Task')}`];});
    return html`<div class="field"><label id="reviewPreparedTaskLabel" for="reviewPreparedTask">${t('Prepared evidence Task')}</label>${picker('reviewPreparedTask',[['',t('Choose the exact prepared Task')],...options_],{selected:S.packetTask,labelId:'reviewPreparedTaskLabel'})}<small>${t(options_.length ? 'Only a Task prepared for this exact book is accepted.' : 'No prepared evidence Task exists yet; prepare sources first.')}</small></div>`;
  }
  /* The held answer parsed for an advisory preview: its findings or issues and their handles
   * against the reference set this page read (the packet's spans, the dossier's handles and
   * issuers); malformed items are named; an unreadable or empty set certifies nothing. */
  const isObject=(v)=>Boolean(v) && typeof v==='object' && !Array.isArray(v);
  // the handles a CRO assessment may cite: the dossier's sealed findings (an explicit list wins
  // when the owner sends one); null when the dossier states neither
  const citableHandles=(d)=>!isObject(d) ? null : Array.isArray(d.finding_handles) ? d.finding_handles.filter(x=>typeof x==='string') : Array.isArray(d.findings) ? d.findings.map(f=>f?.finding_handle).filter(x=>typeof x==='string') : null;
  const stringList=(v)=>Array.isArray(v) ? v.filter(x=>typeof x==='string') : [];
  // the judgment-only answers (the agent seam; contract 10.1): the field each role's answer travels
  // in; the dossier's finding aliases (F1.. -> its finding) and its open issues (O1.. in dossier order)
  const ANSWER_FIELD={analyst:'analysis_answer',cro:'review_answer'};
  const findingAliases=()=>{ const m=bundleOf('cro')?.finding_aliases; return isObject(m) ? m : null; };
  const aliasOf=(handle)=>{ const m=findingAliases(); if(!m) return ''; for(const [a,h] of Object.entries(m)) if(h===handle) return a; return ''; };
  const openIssueAliases=()=>{ const d=dossierOf(); return Array.isArray(d?.open_issues) ? Object.fromEntries(d.open_issues.map((v,i)=>['O'+(i+1),v?.open_issue_handle || ''])) : null; };
  function answerPreview(role) {
    const text=S.drafts[role]; if(!text) return null;
    let body; try{body=JSON.parse(text);}catch(e){return {error:t('Not valid JSON: {e}',{e:e.message})};}
    if(!isObject(body)) return {error:t('The structured answer must be one JSON object.')};
    const envelope=Object.hasOwn(body,'operation'), field=ANSWER_FIELD[role], inner=envelope ? body[field] : body;
    if(!isObject(inner)) return {error:t('The response field {f} is missing or is not an object.',{f:field})};
    const listName=role==='analyst' ? 'findings' : 'risks', raw=inner[listName];
    const unsupported=[];
    if(raw!==undefined && !Array.isArray(raw)) unsupported.push(t('{f} is not a list',{f:listName}));
    const valid=(Array.isArray(raw) ? raw : []).map((item,i)=>{ if(!isObject(item)){unsupported.push(t('{f}[{i}] is not an object',{f:listName,i}));return null;} return item; }).filter(Boolean);
    let reference, cited, rows, resolved=0;
    if(role==='analyst'){
      const facts=packetJson();
      reference=facts===null ? {status:'unavailable'} : Array.isArray(facts.spans) ? {status:facts.spans.length ? 'available' : 'empty',set:new Set(facts.spans.map(s=>s?.alias).filter(Boolean))} : {status:'unavailable'};
      cited=[...new Set(valid.flatMap(f=>[...stringList(f.cite),...stringList(f.contrary)]))];
      rows=valid.map((f,i)=>({handle:t('Finding {n}',{n:i+1}),entities:typeof f.issuer==='string' ? f.issuer : '',direction:typeof f.direction==='string' ? f.direction : '',cites:stringList(f.cite).length}));
    } else {
      // a risk cites findings (F..) and may carry an open issue (O..); `resolved` closes open issues
      const d=dossierOf(), aliases=findingAliases(), opened=openIssueAliases() || {}, byHandle=new Map((d?.findings || []).map(f=>[f?.finding_handle,f]));
      const done=Array.isArray(inner.resolved) ? inner.resolved.filter(isObject) : []; resolved=done.length;
      reference=!aliases ? {status:'unavailable'} : {status:Object.keys(aliases).length || Object.keys(opened).length ? 'available' : 'empty',set:new Set([...Object.keys(aliases),...Object.keys(opened)])};
      cited=[...new Set([...valid.flatMap(r=>[...stringList(r.findings),...(typeof r.carries==='string' ? [r.carries] : [])]),...done.flatMap(r=>[...(typeof r.issue==='string' ? [r.issue] : []),...stringList(r.findings)])])];
      const issuersOf=(r)=>[...new Set(stringList(r.findings).flatMap(a=>stringList(byHandle.get(aliases?.[a])?.affected_entities)))];
      rows=valid.map((r,i)=>({handle:t('Risk {n}',{n:i+1}),entities:entityNames(issuersOf(r)) || '',cites:stringList(r.findings).length,severity:typeof r.severity==='string' ? r.severity : '',carries:typeof r.carries==='string' ? r.carries : ''}));
    }
    const unknown=reference.status==='available' ? cited.filter(h=>!reference.set.has(h)) : cited;
    const verification=reference.status==='available' ? (unknown.length ? 'unknown' : cited.length ? 'checked' : 'none') : reference.status; // unavailable | empty | unknown | checked | none
    const summary=role==='analyst' ? inner.notes : inner.summary;
    return {envelope,summary:typeof summary==='string' ? summary : '',count:valid.length,items:rows,cited:cited.length,unknown,verification,unsupported,resolved};
  }
  /* A pydantic validation text ("N validation errors for Model" then each field path on its own line
   * and its message indented under it) as {cause, fields}; any other text is left to be read whole. */
  function validationWords(detail) {
    const lines=String(detail || '').split('\n'), head=lines.findIndex(l=>/\d+ validation errors? for /.test(l));
    if(head<0) return null;
    const fields=[];
    for(let i=head+1;i<lines.length;i++) if(lines[i].trim() && !/^\s/.test(lines[i])) fields.push([lines[i].trim(),(lines[i+1] || '').trim().replace(/\s*\[.*$/,'')]);
    return fields.length ? {cause:lines[head].replace(/^[A-Za-z]*Error:\s*/,'').trim(),fields} : null;
  }
  function refusalNotice() {
    const r=S.lastRefusal; if(!r || r.role!==S.role) return '';
    const what=t('answer');
    const at=new Date(r.at).toLocaleTimeString('en-GB',{hour12:false});
    if(r.kind==='done') return ''; // an admitted answer's outcome is said where its Task is listed (submittedSection)
    if(r.kind==='correct'){ // contract 10.4: nothing admitted; each problem by its item; the answers still read and what is already acceptable as facts
      const a=isObject(r.answer) ? r.answer : {}, problems=Array.isArray(a.problems) ? a.problems.filter(isObject) : [], kept=Array.isArray(a.accepted_items) ? a.accepted_items.length : 0, left=Number(a.rounds_left) || 0;
      const facts=kv([[hint(t('Answers still read'),t('After the last, the Host keeps the acceptable items and drops the rest.')),count(left)],[t('Items already acceptable'),count(kept)]]);
      return refusal({code:r.code,reason:t('The Host read this answer and returned it with the problems below; nothing was admitted.')},TONE.attention,{state:'returned_for_correction',next:problems.length ? t('correct the named items and submit again') : '',cls:'review-refusal',attrs:html`data-outcome="correct"`,more:html`${problems.length ? html`<ul class="refusal-fields">${problems.map(p=>html`<li>${p.item ? t('Item {n}',{n:p.item}) : t('The answer')} · ${p.text || ''}</li>`)}</ul>` : ''}${facts}<p class="caption">${t('The answer you submitted is kept below exactly as written.')} · ${at}</p>`,action:btn(t('Dismiss'),'review-refusal-dismiss','','button compact')});
    }
    if(r.kind==='uncertain') return refusal({code:r.code,reason:t('The request left this page but no owner answer arrived: it may have been admitted, refused, or never received. Nothing is assumed either way, and it is not resent. Read again to see whether the owner admitted a Task for this {what}; an identical resubmission of an admitted {what} is an exact reuse, not a second publication.',{what})},'warning',{state:'metadata',word:t('The outcome of the last submission is uncertain'),cls:'review-refusal',attrs:html`data-outcome="uncertain"`,more:html`<p class="caption">${t('The {what} you submitted is kept below exactly as written.',{what})} · ${at}</p>`,action:html`<div class="flow">${btn(t('Read again'),'review-reconcile','','button compact')}${btn(t('Dismiss'),'review-refusal-dismiss','','button compact')}</div>`});
    const fields=(r.fields || []).map(f=>html`<li><span class="mono">${(f.path || []).join('.')}</span>: ${t('invalid')} <span class="mono">${(f.invalid_values || []).join(', ')}</span> · ${t('allowed')} <span class="mono">${(f.allowed_values || []).join(', ')}</span></li>`);
    // a validator's text is the owner's words, read as its fields: its first line the cause, each field
    // with its message; the exact text stays one press away (ST6: said once; ST1: nothing invented)
    const said=validationWords(r.detail) || validationWords(r.code); // the service sends a validator's text as the refusal's code
    const sentRefused=r.kind==='renewal';
    return refusal({code:said && !r.detail ? '' : r.code,reason:html`${said ? said.cause : r.detail || ''}${explain(r.code) ? html`${r.detail ? ' · ' : ''}${explain(r.code)}` : ''}`,next:r.next},'warning',{word:sentRefused ? t('Sent and refused before dispatch') : t('The owner refused this {what}',{what}),cls:'review-refusal',attrs:html`data-outcome="${sentRefused ? 'sent-refused' : 'refused'}"`,more:html`${said ? html`<ul class="refusal-fields">${said.fields.map(([path,why])=>html`<li><span class="mono">${path}</span> · ${why}</li>`)}</ul>${codeRef(t('The owner\'s exact refusal'),r.detail || r.code,'text')}` : ''}${fields.length ? html`<ul>${fields}</ul>` : ''}<p class="caption">${sentRefused ? t('Nothing was admitted; the {what} you submitted is kept below exactly as written. It is not resent by itself: confirm again to send it.',{what}) : t('Nothing was recorded. The {what} you submitted is kept below exactly as written; edit it or submit it to the right binding.',{what})} · ${at}</p>`,action:btn(t('Dismiss'),'review-refusal-dismiss','','button compact')});
  }
  /* The role's answer (the user's readings of 2026-09-25 on this form: its verbs, their spacing, then
   * the form as a whole): one editor box in the Lab's shape -- the JSON painted with its line numbers
   * (codeEditor), a file imported or dropped into the same draft, the quiet verbs in its head (Format
   * and Clear only while there is a draft to format or clear), one Submit at its foot whose held
   * reason is its tip (CT7); the draft's notices above the box, the parsed preview under it. */
  // an answer travels in one request, which the service caps (web.py MAXIMUM_REQUEST_BODY_BYTES), and its owner bounds
  // the answer itself (the Analyst brief's MAXIMUM_SUBMISSION_BYTES, the CRO assessment's MAXIMUM_REVIEW_SUBMISSION_BYTES);
  // a test keeps the three equal to their owners'. A draft is measured as the exact JSON it would send; an owner's own
  // count adds only the fields it defaults, so at its bound the owner's refusal decides (said in words, CODE_LINES)
  const REQUEST_LIMIT=128*1024, ANSWER_LIMIT={analyst:64*1024,cro:256*1024}, kib=(n)=>Math.ceil(n/1024), bytesOf=(text)=>new TextEncoder().encode(text || '').length;
  const answerMax=(role)=>Math.min(REQUEST_LIMIT,ANSWER_LIMIT[role]);
  // the bound a draft passes, with its size against it; null while it fits. A text that is not yet one JSON object is its own size
  function overLimit(role,text) {
    if(!text) return null;
    let request, answer=0;
    try { const payload=submissionPayload(role,text); delete payload.operation; request=bytesOf(JSON.stringify(payload)); answer=bytesOf(JSON.stringify(payload[ANSWER_FIELD[role]])); }
    catch { request=bytesOf(text); }
    return request>REQUEST_LIMIT ? {size:request,max:REQUEST_LIMIT} : answer>ANSWER_LIMIT[role] ? {size:answer,max:ANSWER_LIMIT[role]} : null;
  }
  const overWords=(o)=>t('This answer is {size} KB as sent; at most {max} KB is accepted.',{size:kib(o.size),max:kib(o.max)});
  const draftOff=(role)=>{
    if(!current()) return t('Read-only while pinned or busy');
    if(!(role==='analyst' ? S.packet : S.dossier)) return t(role==='analyst' ? 'Read the Analyst packet first' : 'Read the CRO dossier first');
    if(!S.drafts[role]) return t('Paste or import the structured answer first');
    const over=overLimit(role,S.drafts[role]);
    return over ? overWords(over) : '';
  };
  // the editor's foot says what stops the answer, in place: a file refused for its size, or a draft over a bound
  function answerStatus(role){
    const r=S.importRefusal;
    if(r) return noteLine(t('Not imported'),r.read ? t('{name} is {size} KB as sent; at most {max} KB is accepted.',{name:r.name,size:kib(r.size),max:kib(r.max)}) : t('{name} is {size} KB; at most {max} KB is accepted.',{name:r.name,size:kib(r.size),max:kib(r.max)}),'warning');
    const over=overLimit(role,S.drafts[role]);
    return over ? noteLine(t('Too large to send'),overWords(over),'warning') : '';
  }
  const parses=(text)=>{ try { JSON.parse(text); return true; } catch { return false; } };
  /* The answer's place (the user's phase 6 reading): the material is read first, so the editor opens on an explicit
   * action -- or by itself once a draft, an import or its refusal is held, so nothing in hand is hidden. */
  function answerSlot(role) {
    if(S.answerOpen===role || S.drafts[role] || S.importRefusal || S.draftsNote || S.editorNote || S.unitless || S.legacy) return answerEditor(role); // a held draft, even an earlier version's, opens it
    const what=t(role==='analyst' ? 'packet' : 'dossier');
    return panel(t(role==='analyst' ? 'Structured Analyst answer (JSON)' : 'Structured CRO answer (JSON)'),t('Read the material, then paste or import the structured answer; its citations are checked against the {what} before anything is sent.',{what}),btnAttrs(t('Paste or import an answer'),'review-answer-open',role,'button compact','data-answer-open'),'','data-answer-closed="'+role+'"');
  }
  const openAnswer=(role)=>{ S.answerOpen=role; repaintMain(); };
  function answerEditor(role) {
    const preview=answerPreview(role), draft=S.drafts[role] || '', off=draftOff(role);
    const facts=previewSlot(role,preview), title=t(role==='analyst' ? 'Structured Analyst answer (JSON)' : 'Structured CRO answer (JSON)');
    // the answer's shape is the title's (i): one sentence of what the Host reads (contract 10.1)
    const shape=t(role==='analyst' ? 'One JSON object: findings, each citing excerpt aliases (S12); a subset or an empty list is an answer.' : 'One JSON object: risks, each citing finding aliases (F3) with why, severity, confidence and a recommendation; a risk may carry an open issue (O1), resolved closes one; an empty list says no major negative was found.');
    // the editor's own notes, in place (LY7): the drafts' store, then what the last press on this editor did
    const notices=html`${S.draftsNote ? html`<p class="caption review-draft-note" role="status">${S.draftsNote}</p>` : ''}${S.editorNote ? html`<p class="caption review-editor-note" role="status">${S.editorNote}</p>` : ''}${S.unitless ? html`<div class="banner neutral review-unitless" role="status">${icon('info')}<div class="grow"><strong>${t('An unsent draft for this prepared Task names no unit')}</strong><p>${t('Kept before the unit was named, so which unit it answers is unproven. Recover it into this unit\'s empty fields, export it, or discard it; nothing happens otherwise.')}</p><p class="caption">${t('Analyst')}: ${S.unitless.analyst ? t('{n} characters',{n:S.unitless.analyst.length}) : ''} · CRO: ${S.unitless.cro ? t('{n} characters',{n:S.unitless.cro.length}) : ''}</p></div><div class="flow">${btn(t('Recover here'),'review-unitless','recover','button compact')}${btn(t('Export'),'review-unitless','export','button compact')}${btn(t('Discard'),'review-unitless','discard','button compact')}</div></div>` : ''}${S.legacy ? html`<div class="banner neutral review-legacy" role="status">${icon('info')}<div class="grow"><strong>${t('An unsent draft from an earlier version of this page names this book and prepared Task')}</strong><p>${t('Stored before drafts were kept per workspace, so its workspace is unproven. Recover it into the empty fields, export it, or discard it; nothing happens otherwise.')}</p><p class="caption">${t('Analyst')}: ${S.legacy.analyst ? t('{n} characters',{n:S.legacy.analyst.length}) : ''} · CRO: ${S.legacy.cro ? t('{n} characters',{n:S.legacy.cro.length}) : ''}</p></div><div class="flow">${btn(t('Recover here'),'review-legacy','recover','button compact')}${btn(t('Export'),'review-legacy','export','button compact')}${btn(t('Discard'),'review-legacy','discard','button compact')}</div></div>` : ''}`;
    const tools=html`<span class="flow review-answer-tools">${btnAttrs(html`${icon('file')}${t('Import a file')}`,'review-file-open','','text-btn',html`aria-controls="reviewReplyFile"`)}${btnAttrs(t('Format JSON'),'review-draft-format','','text-btn',html`data-draft-tool="format"${parses(draft) ? '' : ' hidden'}`)}${btnAttrs(t('Clear draft'),'review-draft-clear','','text-btn',html`data-draft-tool="clear"${draft ? '' : ' hidden'}`)}</span>`;
    const editor=codeEditor('reviewReply',draft,{lang:'json',label:title,placeholder:t('Paste the structured answer here, or drop a JSON file (up to {max} KB).',{max:kib(answerMax(role))}),attrs:'data-drop-json'});
    return html`${notices}<section class="editor-workbench panel review-answer" data-box="workspace"><header class="editor-workbench-header"><span class="editor-file">${icon('file')}<strong>${hint(title,shape)}</strong></span>${tools}</header><input id="reviewReplyFile" type="file" accept="application/json,.json" hidden><div class="declaration-content">${editor}</div><footer class="editor-statusbar"><div class="review-answer-status" id="reviewAnswerStatus">${answerStatus(role)}</div><span class="sr-only" id="reviewSubmitReason">${off}</span>${typedBtn(t(role==='analyst' ? 'Submit the Analyst answer' : 'Submit the CRO answer'),'review-submit-preview','','button primary',off,'aria-describedby="reviewSubmitReason"')}</footer></section><div class="review-answer-preview-slot">${facts}</div>`;
  }
  /* The preview of the held answer alone, so typing repaints it without repainting the page. */
  function previewSlot(role,preview=answerPreview(role)) {
    if(!preview) return '';
    if(preview.error) return html`<p class="caption review-answer-error">${preview.error}</p>`;
    const what=role==='analyst' ? t('packet') : t('dossier');
    // the aliases are the fact, in sight: the grey line's fold keeps its title's clause and folds the explanation
    const verdict=preview.verification==='unknown' ? noteLine(html`${t('References outside the {what}',{what})} <span class="mono">${preview.unknown.join(', ')}</span>`,t('These aliases are not in the {what}; the Host names each one as a problem and admits nothing for that item.',{what}),'warning')
      : preview.verification==='unavailable' ? noteLine(t('Citations not checked on this page'),t('The aliases of the {what} could not be read here, so no citation is confirmed or flagged; the Host checks every one on submission.',{what}),'neutral')
      : preview.verification==='empty' ? noteLine(t('The {what} names no aliases',{what}),t('Every cited alias is unknown to it; the Host names each as a problem.'),'warning')
      : preview.verification==='none' ? html`<p class="caption">${t('No alias is cited (an advisory reading; the Host decides).')}</p>`
      : html`<p class="caption">${t('Every cited alias is in the {what} this page read (an advisory check; the Host decides).',{what})}</p>`;
    const rows=[[t(role==='analyst' ? 'Findings' : 'Risks'),html`${count(preview.count)} <span class="sub-cell">${countText(preview.cited,'{n} citation','{n} citations')}</span>`],...(role==='cro' && preview.resolved ? [[t('Open issues resolved'),count(preview.resolved)]] : []),...(preview.summary ? [[t('Summary'),preview.summary]] : []),[t('Envelope'),preview.envelope ? t('the owner\'s submission template (context checked on submit)') : t('the bare response; the {what}\'s template is added on submit',{what})]];
    return html`<div class="review-answer-preview"><h3>${t(role==='analyst' ? 'Answer preview · Analyst answer' : 'Answer preview · CRO answer')}</h3>${kv(rows)}${preview.unsupported.length ? noteLine(t('Parts of this answer could not be previewed'),html`<span class="mono">${preview.unsupported.join(' · ')}</span> · ${t('The text is kept exactly as written; the owner validates the whole answer on submission.')}`,'warning') : ''}${preview.items.length ? html`<div class="card-list lines">${preview.items.map(i=>role==='cro' ? evidenceRow('issue',{issue_handle:i.handle,affected_entities:[i.entities],severity_if_true:i.severity,cro_inference:i.carries ? t('carries {o}',{o:i.carries}) : '',cited_finding_handles:Array.from({length:i.cites})}) : evidenceRow('finding',{finding_handle:i.handle,affected_entities:[i.entities],direction:i.direction,cites:i.cites}))}</div>` : ''}${verdict}</div>`;
  }
  /* ---- the Review page (round E4, the user's reading: the page follows the book's state; a read
   * runs nothing, so the packet and the dossier are read when the page opens; the Analyst's part is
   * the sealed answer once an analysis is published; the CRO's part is the composed dossier and,
   * once a review is published, the published review). ---- */
  const analysisPublished=()=>['ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW','REVIEW_PUBLISHED'].includes(S.view?.state) || Boolean(dossierOf());
  const reviewPublished=()=>S.view?.state==='REVIEW_PUBLISHED';
  function autoRead() { // the page's reads, once per book and packet; a failure is the section's own words
    if(!current() || !['handoff','evidence'].includes(app.page) || S.busy) return; // the Review page and the cycle's status page (round F1)
    const dossierRequests=Object.values(S.view?.next_requests || {}).filter(r=>r?.operation==='CRO_REVIEW_DOSSIER' && appliesHere(r));
    const dossierKey=S.key+'|'+S.readGen;
    if(analysisPublished() && !continuationRequired() && !inThisRead(S.dossier) && dossierRequests.length===1 && S.autoDossier!==dossierKey){S.autoDossier=dossierKey;void getBundle('cro',dossierRequests[0],true,true);return;}
    // a group's packet the reader chose is read whatever the book's turn: each prepared group is answered on its own (10.9)
    if(analysisPublished() && S.roleChosen!=='analyst') return;
    if(S.packetTask && !bundleOf('analyst') && S.autoPacket!==S.key+'|'+S.packetTask+'|'+S.packetUnit){S.autoPacket=S.key+'|'+S.packetTask+'|'+S.packetUnit;void getBundle('analyst',null,true,true);}
  }
  const croTurn=()=>Boolean(bundleOf('cro')) && !reviewPublished(); // the CRO's turn: a dossier read and no review published yet
  /* ---- the Review page (round F4, GitHub's pull request): the two answers as a timeline in
   * recorded order -- the prepared evidence, the Analyst's answer with its checks per issuer and
   * its findings as threads on the passages (each with the CRO's disposition as its resolution),
   * the CRO's assessment with its material issues, the verdict -- and the editor in the slot of
   * the role whose turn it is. Every row its owner's words. ---- */
  // A retained export establishes an assessment only for the publication this reading names.
  const publishedReceipt=()=>{const hash=S.pin || workingPublication();return hash && S.exportDoc?.hash===hash ? reportOf()?.review?.receipt || null : null;};
  const publishedAssessment=()=>publishedReceipt()?.submission || null;
  // each finding's disposition: as sealed, or -- a review answered in the judgment-only format asks
  // for none -- as the Host derives it (contract 10.1): a finding a named risk cites is a material
  // issue (the risk's reason), any other was read and not named
  const dispositionOf=(handle)=>{ const receipt=publishedReceipt(), pub=receipt?.submission || null; if(!pub) return null;
    if(isObject(receipt.answer)){ const issue=(pub.material_issues || []).find(i=>stringList(i?.cited_finding_handles).includes(handle)); return issue ? {finding_handle:handle,disposition:'MATERIAL_ISSUE',rationale:issue.causal_channel || ''} : {finding_handle:handle,disposition:'NOT_ADDRESSED',rationale:t('Read by the reviewer and not named as a risk.')}; }
    return (pub.finding_dispositions || []).find(x=>isObject(x) && x.finding_handle===handle) || null; };
  // whether the held CRO answer names a risk citing this finding (by its alias)
  const heldRisk=(handle)=>{ const alias=aliasOf(handle); if(!alias) return false; const text=S.drafts.cro; if(!text) return false; let body; try{body=JSON.parse(text);}catch{return false;} const inner=isObject(body) ? (Object.hasOwn(body,'operation') ? body.review_answer : body) : null; return (Array.isArray(inner?.risks) ? inner.risks : []).some(r=>isObject(r) && stringList(r.findings).includes(alias)); };
  /* The head's primary on the Review page: the turn's submit, or the report once published. */
  function reviewWords() {
    if(!S.view || S.refusal) return {primary:null};
    const writing=Boolean(S.roleChosen) && current(); // a new answer or assessment the reader chose to write here (the review, 2026-09-24: the lawful way to judge findings)
    if((reviewPublished() || historical()) && !writing) return {primary:{word:t('Read the report'),action:'review-step',value:'report'}};
    const role=S.role, bundle=role==='analyst' ? bundleOf('analyst') : bundleOf('cro');
    if(!bundle) return {primary:null};
    return {primary:{word:t(role==='analyst' ? 'Submit the Analyst answer' : 'Submit the CRO answer'),action:'review-submit-preview',value:'',disabled:draftOff(role)}}; // the editor's one reason (CT7)
  }
  /* ---- the Review page (E2; Linear's grouped list and Sentry's issue stream, measured; the turns
   * stay GitHub's pull request): the reviewer's order -- one banner when a person must review, the
   * packet and the editor while it is a role's turn, the findings as one table grouped by issuer
   * (one word a cell, colour on the materiality alone; a row reads its finding in the pane), the
   * material issues as a table, the open questions folded under each -- and beside them the three
   * entries (prepared, the Analyst's answer, the CRO's assessment) as short property sections
   * with their ways to the passages, the checks, the dossier and the report. ---- */
  const reviewSection=(title,n,body,attrs)=>html`<section class="panel" ${attrs}><header class="panel-head"><div><h2>${title}${n===null ? '' : html` <span class="num">${count(n)}</span>`}</h2></div></header><div class="panel-body">${body}</div></section>`;
  /* Whether a person must review: the Analyst's request, the assessment's route. */
  const humanReview=()=>{ const d=dossierOf(), pub=publishedAssessment(), v=S.view; return {analyst:Boolean(d?.analyst_requires_human_review),cro:Boolean(pub?.requires_human_review || (reviewPublished() && v?.disposition==='HUMAN_REVIEW_REQUIRED'))}; };
  const humanReviewAsked=()=>{ const h=humanReview(); return h.analyst || h.cro; };
  /* One notice, first (B2; the book plan's item 9): when a person must review, `Human review
   * required` with who asked and its one way; the CRO's recorded words -- why the review stands as it
   * does -- under it, clamped (whole on More); said once, above the list. */
  function reviewBanner() {
    const {analyst,cro}=humanReview(), words=said(publishedAssessment()?.overall_rationale || '');
    if(!analyst && !cro && !words) return '';
    const asked=analyst || cro, who=analyst && cro ? t('The Analyst and the CRO\'s assessment route this review to a person.') : analyst ? t('The Analyst asked for a human review of this answer.') : cro ? t('The assessment routes this review to a human.') : '';
    const act=asked && bundleOf('cro') && current() && S.role!=='cro' ? btnAttrs(t(reviewPublished() ? 'Write a new CRO answer here' : 'Write the CRO answer here'),'review-role','cro','button compact') : '';
    const open=S.findingOpen.has('rationale'), more=words.length>240 ? btnAttrs(t(open ? 'Less' : 'More'),'review-finding-more','rationale','text-btn card-more',html`aria-expanded="${open}"`) : '';
    return html`<div class="review-banner" data-box="decision" role="status">${statusMark(asked ? 'warning' : 'neutral')}<div class="review-banner-words"><p>${asked ? html`<b>${t('Human review required')}</b> · ${who}` : html`<b>${t('The CRO\'s assessment')}</b>`}</p>${words ? html`<p class="owner-text review-rationale${open ? '' : ' clamp-3'}">${words}</p>${more}` : ''}</div>${act}</div>`;
  }
  /* The packet while it is the Analyst's turn: the packets to choose, the obligation and the
   * question; nothing once the analysis is published (the passages are the Reading's). */
  function preparedMain() {
    const b=bundleOf('analyst'), facts=packetJson(), o=facts?.research_obligation, v=S.view;
    if(analysisPublished()) return '';
    const composed=!S.packetTask ? packetRequests(v) : [];
    const PAGE=LIST_PAGE, {shown:shownPackets,page,pages}=pageOf(composed,S.packetPage); S.packetPage=page; // round H4 (R3): twenty groups a page
    const packetFoot=composed.length>PAGE ? pager({total:composed.length,one:'{n} group',many:'{n} groups',page,pages,prev:['review-packet-page','prev'],next:['review-packet-page','next']}) : '';
    // the clip census (2026-09-24): every row said the same words and Task and at 375 the group that tells them apart was cut -- the shared words and Task are said once, a row is its group (law 143)
    const oneTask=new Set(composed.map(r=>r.task_id)).size===1;
    const packetLead=composed.length ? html`<p class="caption">${t('Choose a group to read its Analyst packet')}${oneTask ? html` · ${t('Task')} <span class="mono">${short(composed[0].task_id,SHORT.id)}</span>` : ''}</p>` : '';
    const choice=!S.packetTask ? (composed.length ? html`${packetLead}<div class="card-list lines">${shownPackets.map(r=>objectRow({name:r.evidence_unit_id ? html`${t('Group')} <span class="mono">${r.evidence_unit_id}</span>` : t('Read the Analyst packet'),to:{action:'review-use-packet',value:r.task_id+'|'+(r.evidence_unit_id || '')},cls:'evidence-row'},{key:'pk:'+(r.evidence_unit_id || r.task_id),props:oneTask ? [] : [html`${t('Task')} <span class="mono">${short(r.task_id,SHORT.id)}</span>`]}))}</div>${packetFoot}` : prepareTasks().some(x=>x.lifecycle==='SUCCEEDED') ? packetChoice() : html`<p class="caption">${t('No prepared packet is named for this book.')}</p>`) : '';
    const question=b ? html`${obligationFacts(facts)}${o?.question ? html`<p class="prose owner-text">${said(o.question)}</p>` : ''}` : '';
    // B2 (item 5): the packet's facts and its read's state are the packet's section (the side's first entry went)
    const reading=S.packetTask && !b ? (S.busy==='bundle' ? skeleton('rows') : S.error ? notRead(t('The packet was not read'),S.error,explain(S.error),btn(t('Read again'),'review-packet','','button compact')) : '') : '';
    const held=b ? kv([[t('Packet'),codeWords(b.status)],[t('Prepared Task'),mono(b.prepared_task_id,SHORT.id)],[t('Evidence as-of'),dayWord(o?.evidence_as_of)],[t('Source expires'),dayWord(b.source_expires_at)],[t('Admitted material'),facts ? t('{d} documents · {s} spans',{d:(facts.document_index || []).length,s:(facts.spans || []).length}) : t('not readable here')]].filter(([,x])=>x!=null && x!==''),'kv-columns') : '';
    return choice || question || reading || held ? reviewSection(t('The packet'),null,html`${reading}${choice}${held}${question}`,html`data-review="packet"`) : '';
  }

  /* 2. The Analyst's turn, in the column: the checks to answer while it is awaited, the editor in the
   * Analyst's turn; once published the findings are the one list's (B2). */
  function analystMain() {
    const b=bundleOf('analyst'), facts=packetJson(), o=facts?.research_obligation, published=analysisPublished();
    const names=issuerNames(), name=(id)=>names[id] || entityNames([id]) || id;
    const toAnswer=!published && o ? reviewSection(t('Checks to answer'),(o.issuer_axis || []).length,html`<div class="card-list lines">${(o.issuer_axis || []).map(id=>objectRow({state:'not_reported',name:name(id),why:(o.required_checks || []).map(said).join(' · '),cls:'evidence-row'},{key:'todo:'+id,attrs:html`data-tip="${(o.required_checks || []).map(said).join(' · ')}"`}))}</div>`,html`data-review="to-answer"`) : '';
    const editor=S.role==='analyst' && b ? html`${refusalNotice()}${answerSlot('analyst')}` : '';
    return html`${toAnswer}${editor}`;
  }
  /* The Analyst's checks by issuer (B2; item 5: the side's ways became sections): each issuer's state,
   * its findings, its topics without evidence, why it was selected; what is missing, by issuer and
   * topic, folded under the table. */
  function checksSection() {
    const d=dossierOf(); if(!d || !analysisPublished()) return '';
    const missing=(d.coverage?.missing_evidence || []), missingOf=(id)=>missing.filter(line=>String(line).replace(/^u\d+:\s*/,'').startsWith(id+' ')).length; // a book of several groups prefixes each line with its group (10.8)
    const issuers=[...(d.issuers || [])].sort((x,y)=>(x.weight_rank || 0)-(y.weight_rank || 0));
    if(!issuers.length) return '';
    const stateKey=(i)=>REVIEW_STATE_KEY[String(i.review_state || '').toUpperCase()] || 'pending';
    const tally=new Map(); for(const i of issuers){ const k=stateKey(i); tally.set(k,(tally.get(k) || 0)+1); }
    const findingsOf=(id)=>(d.findings || []).filter(f=>(f.affected_entities || []).includes(id)).length;
    const byIssuer=collectionTable('checks',issuers,[{label:t('Issuer'),type:'text',cls:'col-tight'},{label:t('State'),type:'text'},{label:t('Findings'),type:'num',cls:'col-tight'},{label:t('Topics without evidence'),type:'num',cls:'col-tight'},{label:t('Selection'),type:'text',absorb:true}],i=>[btnAttrs(entityNames(i.tickers) || i.entity_id,'review-live-item',i.entity_id,'text-btn','data-row-press'),t(stateOf(stateKey(i)).word),count(findingsOf(i.entity_id)),count(missingOf(i.entity_id)),i.selection_reason ? codeWords(i.selection_reason) : '']);
    const gaps=missing.length ? html`<details class="reveal-details"><summary>${t('What is missing, by issuer and topic')} <span class="num">${count(missing.length)}</span></summary>${collectionTable('check-gaps',gapLines(missing),[{label:t('Reason'),type:'text',absorb:true}],x=>[x])}</details>` : '';
    return panel(html`${t('Checks by issuer')} <span class="num">${count(issuers.length)}</span>`,[...tally].map(([k,n])=>`${t(stateOf(k).word)} ${count(n)}`).join(' · '),html`${byIssuer}${gaps}`,'','data-review="checks" data-box="table"');
  }
  /* The one list (B2; the book plan's item 9; E2's table): once a review is published, each material
   * issue is a group -- its issuers, its severity if true and the position's impact over the CRO's
   * inference (two lines; whole in the pane), its handle small -- over the findings it names (the
   * owner's link); the findings no issue names follow. Before a review, the findings by the book's
   * groups where the dossier names them, else by issuer. A finding row is its topic and summary, the
   * stance, the support, the citations and the materiality -- the published disposition, else the
   * held draft's, else not answered; during the CRO's turn the materiality is the disposition's
   * picker. A row opens its passages beside the list (item 4). Twenty findings a page. */
  function findingsTable(d,issues=[]) {
    const findings=Array.isArray(d.findings) ? d.findings : null; // no sealed list is an unreadable reference set, not an empty one
    if(!findings) return html`<p class="caption">${t('The dossier seals no findings list; nothing is checked or certified here.')}</p>`;
    if(!findings.length) return html`<p class="caption">${t('No finding was reported.')}</p>`;
    const canAnswer=S.role==='cro' && Boolean(bundleOf('cro')) && current() && (!reviewPublished() || S.roleChosen==='cro'); // a new assessment written here judges each finding in its row (the held draft)
    const material=(f)=>{ const pub=dispositionOf(f.finding_handle); return pub ? pub.disposition==='MATERIAL_ISSUE' : heldRisk(f.finding_handle); };
    // the CRO's turn names a risk citing the finding's alias where a disposition picker stood (contract 10.7); a finding a drafted risk cites says so
    const materiality=(f)=>{
      const h=f.finding_handle, pub=dispositionOf(h), alias=aliasOf(h), code=pub ? pub.disposition : '';
      if(canAnswer && alias) return heldRisk(h) ? html`<span class="muted" data-tip="${t('A risk in the draft cites {a}',{a:alias})}">${t('Named in the draft')} · <span class="mono">${alias}</span></span>` : '';
      if(!code) return html`<span class="muted">${t('Not answered')}</span>`;
      const word=codeWords(code), tip=pub?.rationale ? html` data-tip="${said(pub.rationale)}"` : '';
      return code==='MATERIAL_ISSUE' ? html`<span class="state warning"${STATUS_ATTR}${tip}><i aria-hidden="true"></i><span class="state-word">${word}</span></span>` : html`<span class="muted"${tip}>${word}</span>`;
    };
    // the row's action (FT3): in the CRO's turn, a finding no drafted risk cites offers to name one, by its alias
    const span=canAnswer ? 7 : 6, nameRisk=(f)=>{ const h=f.finding_handle, alias=aliasOf(h); return alias && !heldRisk(h) ? btn(t('Name a risk citing {a}',{a:alias}),'review-name-risk',h,'text-btn compact') : ''; };
    const row=(f,index)=>{ const h=f.finding_handle, cites=(f.supporting_span_handles || []).length+(f.contradicting_span_handles || []).length;
      return tableRow(html`data-finding="${h}"${S.item===h ? html` aria-selected="true"` : ''}`,[['col-index col-num',count(index)],['col-text col-absorb',rowTitle({title:codeWords(f.topic),sub:f.summary || null,action:'review-live-item',value:h,attrs:html`aria-label="${t('Read finding {h}',{h})}"`})],['col-text col-tight',cellWord(f.direction)],['col-text col-tight',html`<span class="muted">${codeWords(f.structure)}</span>`],['col-num col-tight',count(cites)],['col-text col-tight',materiality(f)],...(canAnswer ? [['col-link',nameRisk(f)]] : [])]); };
    let groups;
    if(issues.length){
      const byHandle=new Map(findings.map(f=>[f.finding_handle,f])), named=new Set(issues.flatMap(c=>c.cited_finding_handles || []));
      const issueHead=(c)=>{ const h=c.issue_handle, sev=evidenceState(c.severity_if_true), list=(c.cited_finding_handles || []).map(x=>byHandle.get(x)).filter(Boolean);
        const severity=/HIGH/.test(String(c.severity_if_true || '')) ? html`<span class="state ${TONE.attention}"${STATUS_ATTR}><i aria-hidden="true"></i><span class="state-word">${sev.word}</span></span>` : sev.word;
        const words=c.cro_inference || (c.rule_id && !/^[A-Z][A-Z_]+$/.test(String(c.rule_id)) ? c.rule_id : '') || h; // the issue's words lead (law 96); its handle follows small
        return {list,head:html`<tr class="table-group issue-group" data-issue="${h}"${S.item===h ? ' aria-selected="true"' : ''}><th colspan="${span}" scope="rowgroup">${rowTitle({title:html`${entityNames(c.affected_entities) || ''} · ${severity}${c.position_impact_direction ? html` · ${cellWord(c.position_impact_direction)}` : ''}`,sub:words,action:'review-live-item',value:h,cls:'finding-open issue-open',attrs:html`aria-label="${t('Read issue {h}',{h})}"`})}<span class="table-group-note">${issueDayWords(c) ? html`${issueDayWords(c)} · ` : ''}${countText(list.length,'{n} finding','{n} findings')} · <span class="mono">${h}</span></span></th></tr>`}; };
      groups=issues.map(issueHead);
      const rest=findings.filter(f=>!named.has(f.finding_handle));
      if(rest.length) groups.push({list:rest,head:tableGroup(span,t('Findings no issue names'),countText(rest.length,'{n} finding','{n} findings'))});
    } else {
      const children=Array.isArray(d.evidence_children) ? d.evidence_children : [];
      const groupOf=(f)=>children.length ? (children.find(c=>(c.ordered_entity_ids || []).some(id=>(f.affected_entities || []).includes(id)))?.unit_id || '') : (entityNames(f.affected_entities) || '');
      groups=[...new Set(findings.map(groupOf))].map(g=>{ const list=findings.filter(f=>groupOf(f)===g); return {list,head:tableGroup(span,children.length ? html`${t('Group')} <span class="mono">${g}</span>` : g,html`${countText(list.length,'{n} finding','{n} findings')} · ${countText(list.filter(material).length,'{n} material','{n} material')}`)}; });
    }
    // twenty findings a page, the groups' heads where their first row falls (an issue naming none is its head alone)
    const flat=groups.flatMap(g=>g.list.length ? g.list.map(f=>({g,f})) : [{g,f:null}]);
    const state=collectionState('findings'),query=state.query.trim().toLowerCase(),found=query ? flat.filter(({g,f})=>[f?.finding_handle,f?.summary,entityNames(f?.affected_entities),f?.topic,String(g.head)].some(x=>String(x || '').toLowerCase().includes(query))) : flat;
    const {shown,start,page,pages}=pageOf(found,state.page);if(state.page!==page){state.page=page;if(S.collections.findings)writeRoute(routeUpdate());}
    const foot=pager({total:found.length,one:'{n} entry',many:'{n} entries',page,pages,prev:['review-collection-page','findings|prev'],next:['review-collection-page','findings|next']});
    let last=null;
    const rows=shown.map(({g,f},i)=>{ const head=g!==last ? g.head : ''; last=g; return html`${head}${f ? row(f,start+i+1) : ''}`; });
    const columns=[['#','col-index col-num'],[t('Finding'),'col-text col-absorb'],[t('Stance'),'col-text col-tight'],[t('Support'),'col-text col-tight'],[t('Citations'),'col-num col-tight'],[t('Materiality'),'col-text col-tight'],...(canAnswer ? [['','col-link']] : [])];
    return html`<div data-evidence-collection="findings">${flat.length>LIST_PAGE ? searchBar('evidenceQuery-findings',t('Search'),t('Search'),state.query) : ''}${groupedTable({columns,rows,cls:'findings-table'})}${foot}</div>`;
  }
  /* 3. The CRO's turn, in the column: the editor while it is the CRO's (B2: the issues are the one
   * list's groups, the open questions a section). */
  function croMain() {
    const b=bundleOf('cro');
    if(!analysisPublished()) return '';
    return S.role==='cro' && b ? html`${refusalNotice()}${answerSlot('cro')}` : '';
  }
  /* The list's section (B2): the issues and their findings once a review is published, else the
   * findings; its way on is the Reading's every passage. */
  function reviewList() {
    const d=dossierOf(); if(!d || !analysisPublished()) return '';
    const issues=reviewPublished() || historical() ? (S.view?.issue_cards || []) : [];
    const way=btnAttrs(html`${t('Read the passages')}${icon('arrow')}`,'review-step','evidence-reading','text-btn compact');
    if(!issues.length) return panel(html`${t('Findings')} <span class="num">${count((d.findings || []).length)}</span>`,t('A finding opens its passages beside the list.'),findingsTable(d),way,'data-review="findings" data-box="table"');
    return panel(html`${t('Material issues')} <span class="num">${count(issues.length)}</span>`,t('Each issue over the findings it names, then the findings no issue names; a finding opens its passages beside the list.'),findingsTable(d,issues),way,'data-review="issues" data-box="table"');
  }
  /* The published review's own limits (contract 10.6): what it did not read, in the program's words,
   * and the issuers whose findings no risk names (an answered review) -- the Report's Limits, said
   * once there (PG11); a review sealed before carries neither. */
  const reviewLimits=(v)=>{ if(!v || (!reviewPublished() && !historical())) return []; const unnamed=stringList(v.not_addressed_issuers);
    return [...(v.limitations || []).map(reviewLimitWords),...(unnamed.length ? [html`${t('Issuers whose findings no risk names')}: ${entityNames(unnamed) || unnamed.join(', ')}`] : [])]; };
  /* The open questions (B2; item 9), named by whose they are -- the Analyst's, the CRO's -- each its
   * own count, folded; the question whole at its measure (a unit said where they span more than one). */
  function openQuestions() {
    const d=dossierOf(), pub=publishedAssessment();
    const analyst=(d?.unresolved_questions || []).map(q=>({unit:q.unit_id || '',text:q.text || ''})), cro=(pub?.unresolved_questions || []).map(q=>({unit:'',text:typeof q==='string' ? q : q.text || ''}));
    if(!analyst.length && !cro.length) return '';
    const units=new Set(analyst.map(q=>q.unit)).size>1;
    const list=(qs,who)=>collectionTable('questions-'+who,qs,[...(units ? [{label:t('Unit'),type:'text',cls:'col-tight'}] : []),{label:t('Question'),type:'text',absorb:true}],q=>[...(units ? [q.unit] : []),q.text]);
    // reached from the Report's questions: the Analyst's open, the section brought into view once
    // reached from a Report question (this navigation only): once the Analyst's questions have arrived with the dossier, the section comes into view with the Analyst's fold open -- opened in the page, so a repaint keeps whatever the reader does next
    if(S.questionsFocus===S.nav && analyst.length){ S.questionsFocus=0; if(typeof requestAnimationFrame==='function') requestAnimationFrame(()=>{ const sec=document.querySelector('#main [data-review="questions"]'), first=sec?.querySelector('details'); if(first) first.open=true; sec?.scrollIntoView({block:'start'}); }); }
    const fold=(who,qs,key)=>qs.length ? html`<details class="reveal-details"><summary>${who} <span class="num">${count(qs.length)}</span></summary>${list(qs,key)}</details>` : '';
    return panel(t('Open questions'),t('What the Analyst and the CRO left open, as they recorded it.'),html`${fold(t('The Analyst\'s'),analyst,'analyst')}${fold(t('The CRO\'s'),cro,'cro')}`,'','data-review="questions"');
  }
  /* The Tasks this page's submissions bound to the book (round 79), with their verified stages. */
  /* The last answer the Host reads keeps its acceptable items (contract 10.4): what it kept and dropped is said once,
   * beside the Task the answer made (79313812); the editor closes when the review is published, this stays. */
  function doneNotice() {
    const r=S.lastRefusal; if(r?.kind!=='done') return '';
    const at=new Date(r.at).toLocaleTimeString('en-GB',{hour12:false});
    const a=isObject(r.answer) ? r.answer : {}, dropped=Array.isArray(a.dropped) ? a.dropped.filter(isObject) : [];
    return refusal({reason:t('The Host kept the acceptable items and dropped the rest; no more answers are read for this bundle.')},TONE.attention,{state:'kept_acceptable',cls:'review-refusal',attrs:html`data-outcome="done"`,next:'',more:html`${dropped.length ? html`<ul class="refusal-fields">${dropped.map(p=>html`<li>${p.item ? t('Item {n}',{n:p.item}) : t('The answer')} · ${p.text || ''}</li>`)}</ul>` : ''}${kv([[t('Items kept'),count(Array.isArray(a.accepted_items) ? a.accepted_items.length : 0)],[t('Items dropped'),count(dropped.length)]])}<p class="caption">${at}</p>`,action:btn(t('Dismiss'),'review-refusal-dismiss','','button compact')});
  }
  function submittedSection() {
    const b=boundFor(), ids=new Set([...b.analysis,...b.review]); if(!ids.size) return '';
    return reviewSection(t('Submitted from this page'),ids.size,html`${doneNotice()}${boundRows(ids)}`,html`data-review="submitted"`);
  }

  // a table cell's word starts as every cell's does (the audit: the Stance column read `adverse` beside `Mixed`)
  const cellWord=(code)=>{const w=String(codeWords(code));return /^[a-z]/.test(w) ? w[0].toUpperCase()+w.slice(1) : w;};

  /* "Name a risk citing F.." writes the held CRO answer (the exact submission stays the text; this is
   * its assisted part): one risk citing the finding's alias, its judgment left empty for the reviewer
   * to write -- no severity or confidence is chosen for them. */
  function nameRisk(handle) {
    if(S.busy || S.role!=='cro' || !bundleOf('cro')) return;
    const alias=aliasOf(handle); if(!alias){S.editorNote=t('The dossier names no alias for this finding; read the dossier again.');render();return;}
    const text=S.drafts.cro; let body={};
    if(text){ try{body=JSON.parse(text);}catch{S.editorNote=t('The held answer is not valid JSON; fix it before naming a risk from a finding.');render();return;} if(!isObject(body)){S.editorNote=t('The structured answer must be one JSON object.');render();return;} }
    const envelope=Object.hasOwn(body,'operation'), inner=envelope ? (isObject(body.review_answer) ? body.review_answer : (body.review_answer={})) : body;
    const list=Array.isArray(inner.risks) ? inner.risks : (inner.risks=[]);
    if(!list.some(r=>isObject(r) && stringList(r.findings).includes(alias))) list.push({findings:[alias],why:'',severity:'',confidence:'',recommendation:''});
    reply(JSON.stringify(body,null,2)); render();
  }
  /* The Facts panel's sections on the handoffs page (round 79): each read bundle's binding and
   * envelope, and how this page works. */
  /* The exact publication's Committee and the sessions that declare this book or analysis. */
  function teamFacts() {
    const v=S.refusal ? null : S.view, c=v?.book?.committee_context || deliveryOf()?.committee_context;
    if(!(v || c) || typeof LiveViews==='undefined' || !LiveViews.collaborationRows) return [];
    const rows=LiveViews.collaborationRows(v?.book?.result_hash,v?.review_publication_hash,v && currentIdentity().analysis,v && dossierOf()?.analysis_publication_hash,c?.update_task_id,c?.review_selector?.update_publication_hash,c);
    return rows.length ? [{title:t('Team'),body:kv(sourceRows(rows))}] : [];
  }
  function handoffFacts() {
    const p=bundleOf('analyst'), d=bundleOf('cro'), out=[];
    if(p) out.push({title:t('Analyst packet'),body:html`${kv([[t('Context'),mono(p.analysis_context_hash,SHORT.hash)],[t('Packet'),mono(p.packet_hash,SHORT.hash)],[t('Prepared Task'),mono(p.prepared_task_id,Infinity)],[t('Host usage'),codeWords(p.external_host_usage)],[t('Claim'),codeWords(p.claim)]])}<p class="caption">${t('The answer must be bound to exactly these; a changed packet is refused.')}</p>${codeRef(t('Read the submission template and response schema'), {submission_template:p.submission_template,response_schema:p.response_schema})}`});
    // B2 (item 5): the preparation's and the assessment's facts, once the side's entries
    const a=dossierOf(), v=S.view;
    if(analysisPublished()) out.push({title:t('The analysis'),body:html`${kv([[t('Analysis'),mono(a?.analysis_publication_hash || currentIdentity().analysis || '',SHORT.hash)],[t('Evidence as-of'),dayWord(a?.evidence_as_of || v?.evidence_as_of) || ''],[t('Expires'),dayWord(a?.evidence_expires_at || v?.evidence_expires_at) || '']])}${(a?.evidence_children || []).length ? html`<h3>${t('Groups\' publications')} <span class="num">${count(a.evidence_children.length)}</span></h3><ul class="fact-list">${a.evidence_children.map(c=>html`<li>${t('Group')} <span class="mono">${c.unit_id}</span> · ${entityNames(c.ordered_entity_ids) || ''} · ${t('analysis')} <span class="mono">${short(c.analysis_publication_hash,SHORT.id)}</span> · ${t('as-of {d}',{d:dayWord(c.evidence_as_of)})}</li>`)}</ul>` : ''}`});
    if(d) out.push({title:t('CRO dossier'),body:html`${kv([[t('Dossier'),html`${codeWords(d.status)} · ${mono(d.dossier?.dossier_hash,SHORT.hash)}`],[t('Analysis reviewed'),mono(d.dossier?.analysis_publication_hash,SHORT.hash)],[t('Citable findings'),(citableHandles(d.dossier) || []).length ? count((citableHandles(d.dossier) || []).length) : t('none')],[t('Dossier delivery'),(d.delivery || {}).delivery_mode==='WHOLE_DOSSIER' ? t('whole') : d.delivery?.part ? t('part {p} of {n}',{p:count(d.delivery.part),n:count(d.delivery.part_count)}) : ''],[t('Routes the policy allows'),(d.dossier?.allowed_routes || []).map(codeWords).join(' · ') || ''],[t('Policy'),mono(d.decision_policy_hash,SHORT.hash)],[t('Schema'),mono(d.assessment_schema_hash,SHORT.hash)],[t('Host usage'),codeWords(d.external_host_usage)],[t('Claim'),codeWords(d.claim)]])}${[...(d.dossier?.claim_limits || []),...(d.dossier?.limitations || [])].length ? html`<h3>${t('Claim limits and limitations')}</h3><ul class="fact-list">${[...(d.dossier?.claim_limits || []),...(d.dossier?.limitations || [])].map(x=>html`<li>${said(x)}</li>`)}</ul>` : ''}${codeRef(t('Read the submission template and assessment schema'), {submission_template:d.submission_template,assessment_schema:d.assessment_schema})}`});
    const bundleRequest=inThisRead(S.dossier) ? d?.next_requests?.cro_bundle : null;
    if(bundleRequest?.operation==='AGENT_BUNDLE_PREPARE' && bundleRequest.agent_role==='CRO' && appliesHere(bundleRequest)) out.push({title:t('Prepare agent bundle'),body:codeRef(t('Read the exact request'),bundleRequest)});
    return [...out,...teamFacts(),{title:t('How this page works'),body:html`<p>${t('The product prepares, validates, seals and publishes; the answers come from outside it. A dossier is compiled only from a current published analysis. Opening, reading or downloading here starts no actor, model or fetch.')}</p>`}];
  }
  /* The handoff's record (round 79): what the owners recorded about the Tasks this page bound. */
  const boundTasks=()=>{const b=boundFor();return [...b.prepare,...b.analysis,...b.review];};
  function handoffRecord() {
    const lines=boundTasks().flatMap(id=>LiveActivity.logLines(id));
    return lines.length ? runLog({id:'handoffRecord',title:t('Record'),lines,count:lines.length}) : html`<p class="caption">${t('Nothing recorded for this object in the retained activity window.')}</p>`;
  }
  function handoffState() {
    const r=Object.fromEntries(stepRows().map(([id,[tone]])=>[id,tone]));
    const [code,word]=r.cro==='refused' || r.analyst==='refused' ? ['refused','Last submission refused'] : r.published==='done' ? ['succeeded','Answered · Analyst and CRO'] : r.cro==='running' ? ['running','Sealing the assessment'] : r.validation==='running' ? ['running','Validating the answer'] : r.analyst==='done' || r.validation==='done' ? ['review_pending','Awaiting the CRO assessment'] : r.analyst==='waiting' || S.view?.state==='ANALYST_PACKET_PREPARED' || packetRequests(S.view).length ? ['review_pending','Awaiting the Analyst answer'] : ['pending',r.sources==='done' ? 'Packet not read yet' : 'Sources not prepared yet'];
    return stateLine(code,{word:t(word),next:''});
  }
  // the Tasks a role's submission created, bound to this book by this page (round 79)
  const boundRows=(set)=>set.size ? html`<div class="card-list lines">${[...set].map(id=>{const s=taskState(id);return objectRow({state:s.lifecycle || 'pending',name:html`${t('Task')} ${mono(id,SHORT.id)}`,why:t('{n} / {m} stages verified',{n:s.verified,m:s.total}),to:{action:'task',value:id},cls:'evidence-row'},{key:id});})}</div>` : '';
  function handoffPage() {
    S.role=continuationRequired() ? 'analyst' : S.roleChosen || (croTurn() ? 'cro' : 'analyst'); // the current owner gate precedes an earlier chosen role
    queueMicrotask(autoRead);
    // B1: the head is the book's; where the handoff stands is the Review's first line -- the turn, then each party's day
    const days=['prepared','analysed','reviewed'].map(id=>cycle().stages.find(x=>x.id===id)).filter(st=>st && st.time).map(st=>html`${t(st.name)} ${st.time}`);
    const where=S.view ? html`<p class="review-where">${handoffState()}${days.length ? html` <span class="muted">· ${days.map((d,i)=>html`${i ? ' · ' : ''}${d}`)}</span>` : ''}</p>` : '';
    // B2: one column in the reviewer's order; the changes since the prior review are the Report's
    const list=reviewList(), questions=openQuestions(), answer=croMain(), checks=checksSection(); // the sections, composed once: the jumps name those the page holds
    return html`${where}${workSection('handoff')}${reviewBanner()}<div class="overview-main">${preparedMain()}${analystMain()}${submittedSection()}${pageJumps([[t('Findings'),'[data-review="findings"], [data-review="issues"]',Boolean(String(list))],[t('Open questions'),'[data-review="questions"]',Boolean(String(questions))],[t('The answer'),'[data-answer-closed], .review-answer',Boolean(String(answer))],[t('Checks by issuer'),'[data-review="checks"]',Boolean(String(checks))]])}${list}${questions}${answer}${checks}</div>`;
  }
  /* The Review's ··· (B2; item 5): the packet's and the dossier's exact downloads, and the way to write
   * a role's answer on this page when it is not the role in view (a human review's way is the notice's). */
  function handoffTools() {
    const b=bundleOf('analyst'), d=bundleOf('cro');
    return [...(b ? [{ic:'file',action:'review-bundle-export',value:'',word:t('Download the packet (JSON)'),why:t('The owner\'s exact packet')}] : []),...(d ? [{ic:'file',action:'review-bundle-export',value:'cro',word:t('Download the dossier (JSON)'),why:t('The owner\'s exact dossier')}] : []),
      ...(b && S.role!=='analyst' ? [{ic:'edit',action:'review-role',value:'analyst',word:t(analysisPublished() ? 'Write a new answer here' : 'Write the answer here'),why:t('The Analyst\'s answer, written on this page')}] : []),
      ...(d && S.role!=='cro' && !humanReviewAsked() ? [{ic:'edit',action:'review-role',value:'cro',word:t(reviewPublished() ? 'Write a new CRO answer here' : 'Write the CRO answer here'),why:t('The CRO\'s answer, written on this page')}] : [])];
  }
  /* ---- the answer's submission ---- */
  function submissionPayload(role=S.role,text=S.drafts[role]) {
    const bundle=bundleOf(role);
    if(!bundle)throw Error(t(role==='analyst' ? 'Read the Analyst packet first' : 'Read the CRO dossier first'));
    if(!text)throw Error(t('Paste or import the structured answer first'));
    const body=JSON.parse(text), template=bundle.submission_template;
    if(!isObject(body))throw Error(t('The structured answer must be one JSON object.'));
    const field=ANSWER_FIELD[role];
    if(Object.hasOwn(body,'operation')) {
      for(const [k,v] of Object.entries(template))if(body[k]!==v)throw Error(t('The response envelope names a different context: {k}. Read the current packet or dossier and answer that one.',{k}));
      if(!Object.hasOwn(body,field))throw Error(t('Missing structured response field: {f}',{f:field}));
      return {...body};
    }
    return {...template,[field]:body};
  }
  /* The page's own confirmations assemble their request from what the page shows; a composed
   * request (`request`) is confirmed and sent with its own fields, the page's state only
   * saying whether it is admitted here. */
  function confirm(kind,value='',request=null) {
    if(!current())return;
    if(continuationRequired() && (kind==='cro' || (kind==='submit' && S.role==='cro'))){S.error=said(S.view.explanation);render();return;}
    S.error='';
    let path,payload,explanation,title,facts=[];
    try{
      if(kind==='select'){
        if(!S.view.eligible_versions.some(v=>v.analysis_publication_hash===value))throw Error(t('Analysis is not an eligible choice.'));
        path='/api/evidence-select';payload={...S.selector,analysis_publication_hash:value};title=t('Record this analysis selection');explanation=t('Records this exact analysis as the one to review against; prior publications stay unchanged. No model or Task runs.');
      }else if(kind==='prepare'){
        if(!request && S.preview?.status!=='EVIDENCE_PREPARATION_READY')return;
        const words=modeWords(S.preview?.status==='EVIDENCE_PREPARATION_READY' ? S.preview : {source_mode:S.view?.source_mode || ''});
        path=Data.route('EVIDENCE_PREPARE');payload=request ? requestFields(request) : {...S.preview.next_requests.prepare};title=t(words.kind==='live' ? 'Prepare sources by admitted official acquisition' : 'Prepare sources for this book');
        // the confirmation is facts and one sentence (the user's reading, 2026-09-22: the prose was one block): what the owner reads, from where, under what, and what never runs; the owner's own estimate is a caption
        const p=S.preview?.status==='EVIDENCE_PREPARATION_READY' ? S.preview : null, inv=p?.source_inventory || null;
        facts=[[t('Source mode'),p ? html`${codeWords(p.source_mode || '')}${p.source_work ? html` · ${codeWords(p.source_work)}` : ''}` : t('not stated')],
          [t('Documents'),p ? html`${countText(p.recorded_candidate_count || 0,'{n} candidate','{n} candidates')}${inv && Array.isArray(inv.issuers) ? html` · ${t('{w} of {n} issuers hold a source',{w:count(inv.issuers_with_source ?? inv.issuers.filter(r=>(r.documents || 0)>0).length),n:count(inv.issuers.length)})}` : ''}` : ''],
          [t('Evidence as-of'),p?.evidence_as_of ? dayWord(p.evidence_as_of) : ''],
          [t('Network'),words.kind==='live' ? t('official sources over the network, under the admitted acquisition policy') : words.kind==='recorded' ? t('none: nothing is fetched') : t('not stated')],
          [t('Model'),p?.managed_model_required ? t('a managed model is required') : t('none: no analyst or CRO model runs')],
          ...(p?.work_estimate ? [[t('Owner\'s estimate'),workEstimate(p)]] : [])]; // both readings use the owner's count and bound
        explanation=words.kind==='recorded' ? t('Confirming admits one local Task of the evidence owner over the recorded source package: it reads the recorded documents, canonicalizes them, builds the retrieval index and selects admitted passages.')
          : words.kind==='live' ? t('Confirming admits one Task of the evidence owner that acquires official documents from the approved source families over the network under the admitted acquisition policy, then canonicalizes them, builds the retrieval index and selects admitted passages; the owner revalidates the acquisition authority at submission.')
          : words.prepare;
      }else if(kind==='submit'){
        path=S.role==='analyst' ? '/api/evidence/analysis' : '/api/cro/assessment';payload=submissionPayload();
        const over=overLimit(S.role,S.drafts[S.role]); if(over)throw Error(overWords(over)); // the held Submit's own bound, whichever entry asked
        title=t(S.role==='analyst' ? 'Submit the Analyst answer' : 'Submit the CRO answer');explanation=t('The Host checks the answer against its exact binding and names any problem by item: an answer with problems is returned for correction (twice at most; the third keeps its acceptable items). An admitted answer is sealed and published as one Task with HUMAN attribution. An imported answer does not authorize its own route and does not prove an independent reviewer.');
      }else{
        const action=kind==='refresh' ? 'REFRESH_EVIDENCE' : 'REVIEW_WITH_CRO';
        if(!S.view.available_actions.includes(action))return;
        path=Data.route(kind==='refresh' ? 'EVIDENCE_REFRESH' : 'CRO_REVIEW');payload=request ? requestFields(request) : {...S.selector};title=t(kind==='refresh' ? 'Refresh evidence with the configured Provider' : 'Review with the configured Provider');explanation=t('This path invokes the configured model Provider and consumes its quota; without an admitted credential the owner refuses it. It is not the native handoff.');
      }
      delete payload.operation;
      if(bytesOf(JSON.stringify(payload))>REQUEST_LIMIT)throw Error(t('Response exceeds the product request limit.'));
      closeDialog();S.pending={path,payload,revision:S.revision,kind,key:S.key,role:S.role,task:S.packetTask,unit:S.packetUnit};
      // the exact request stays in the confirmation, folded (a reader who opens it is still where the decision is)
      openDialog(t('Evidence · explicit confirmation'),title,html`${kv([[t('Book'),mono(bookName(),SHORT.id)],[t('Holdings date'),session()],...facts])}<p>${explanation}</p><details class="reveal-details"><summary>${t('Exact request')}</summary><pre class="code-block code-document">${json(payload)}</pre></details>`,html`${btn(t('Confirm'),'review-commit','','button primary',true)}`);
    }catch(e){S.error=e.message;render();}
  }
  async function commit() {
    if(S.busy || !S.pending)return;
    const request=S.pending;
    if(request.revision!==S.revision || request.key!==S.key || request.role!==S.role || request.task!==S.packetTask || (request.unit ?? '')!==(S.packetUnit || '')){S.pending=null;closeDialog();S.error=t('The page changed since this confirmation was opened; nothing was sent. Review it and confirm again.');render();return;}
    const ticket=S.revision, gen=S.nav, sentFor={key:S.key,selector:{...S.selector},role:S.role,task:S.packetTask,unit:S.packetUnit,mode:request.kind==='prepare' ? (S.preview?.source_mode || '') : '',
      // the evidence identity the answer was written against: the packet's cutoff for an
      // analysis, the dossier's analysis and cutoff for an assessment
      facts:request.kind==='submit' ? (S.role==='analyst' && bundleOf('analyst') ? packetIdentity(bundleOf('analyst')) : S.role==='cro' && dossierOf() ? {analysis:dossierOf().analysis_publication_hash || '',asOf:dossierOf().evidence_as_of || ''} : null) : null};
    if(request.operation==='EVIDENCE_CONTINUE'){const missing=(request.declare || []).filter(k=>request.payload[k]==null); if(missing.length){S.error=t('Declare {k} before continuing: the owner names the limit and the page holds no default.',{k:missing.map(k=>codeWords(String(k).toUpperCase())).join(', ')});render();return;}} // round E3
    S.pending=null;S.busy='submit';closeDialog();render();
    try{
      const body=await Data.post(request.path,request.payload);
      const task=body.task_id || body.publication_task_id;
      // what the owner admitted belongs to the book it was sent for, whatever the page shows
      // now; a continuation is a preparation of the book, a rebuild is workspace work bound to
      // no book, and neither is an answer this page imported
      const group=request.kind==='prepare' ? 'prepare' : request.kind==='submit' ? (sentFor.role==='analyst' ? 'analysis' : 'review') : request.kind==='refresh' ? 'analysis' : request.kind==='execute' ? (request.operation==='EVIDENCE_CONTINUE' ? 'prepare' : '') : 'review';
      if(task && group){bind(group,task,sentFor.key,sentFor.selector,sentFor.facts);if(request.kind==='submit' && body.task_id)boundFor(sentFor.key).imported.add(task);}
      if(ticket!==S.revision || S.key!==sentFor.key || gen!==S.nav){if(task)notify(t('Admitted for earlier book'),null,{word:'Open',action:'task',value:task});return;}
      S.busy='';
      if(request.kind==='submit' && S.lastRefusal?.role===sentFor.role) S.lastRefusal=null;
      // the packet the outcome composes: adopted whole (its Task and its unit) when it is one;
      // a run of several units offers them as next steps after the refresh and adopts none
      const composed=packetRequests(body); if(composed.length===1 && composed[0].task_id){S.packetTask=composed[0].task_id;S.packetUnit=composed[0].evidence_unit_id || '';loadDrafts();writeRoute(routeUpdate());}
      if(body.disposition==='REUSED_EXACT'){notify(t('Exact reuse'));}
      if(body.disposition==='REUSED_IN_FLIGHT'){notify(t('Already in flight'));}
      // the third answer to one bundle keeps its acceptable items (contract 10.4): what was kept and dropped is said once
      if(request.kind==='submit' && body.answer?.verdict==='DONE') S.lastRefusal={role:sentFor.role,kind:'done',code:'',detail:'',fields:[],next:'',answer:body.answer,at:Date.now()};
      // A typed refusal is the product's answer even when it names a Task: nothing is waited for.
      const refused=typeof body.disposition==='string' && /^REFUSED/.test(body.disposition);
      if(refused){S.error=body.disposition+(body.detail ? ' · '+body.detail : '');}
      else if(task){await watch(task,false,sentFor.mode);}
      if(request.kind!=='submit') await refresh(); else render();
      void Data.refreshHistory();
    }catch(e){
      if(ticket!==S.revision || S.key!==sentFor.key) return;
      S.busy='';
      if(request.kind==='submit') keepOutcome(sentFor.role,e); else S.error=String(e?.message || e)+(e?.body ? '' : ' · '+t('no owner answer arrived; the outcome is uncertain -- Read again before repeating it'));
      render();
    }
  }
  /* A failed submission is kept as what it was: an owner refusal, a request refused before
   * dispatch (the session renewed), or an uncertain transport; each keeps the draft. */
  function keepOutcome(role,e) {
    const message=String(e?.message || e), body=e?.body && typeof e.body==='object' ? e.body : null;
    const renewal=message.startsWith('local_web.session_renewed') || message.startsWith('local_web.session_renewal_failed');
    const kind=renewal ? 'renewal' : body ? (body.status==='CORRECT' ? 'correct' : 'refused') : 'uncertain';
    S.lastRefusal={role,kind,code:renewal ? message.split(':')[0] : body ? String(body.failure_code || body.refused || body.disposition || message) : message,detail:body?.detail || '',fields:body?.field_errors || [],next:body?.next_action || '',answer:isObject(body?.answer) ? body.answer : null,at:Date.now()};
  }
  /* ---- the owner's Tasks in the shared work area ---- */
  // Stages accumulate independently across groups; no furthest group is a global current stage.
  const BOOK_STAGES=['admit_evidence_request','resolve_official_sources','acquire_source_evidence','canonicalize_documents','build_retrieval_generation','select_evidence_spans'];
  const groupStage=(id)=>{ const m=/^u\d+_(.+)$/.exec(String(id || '')); return m ? m[1] : null; };
  const coverageRun=(v)=>Boolean(v?.stages?.length) && v.stages.every(s=>groupStage(s.stage_id));
  function bookView(v) {
    if(!coverageRun(v)) return v;
    const stages=BOOK_STAGES.map(kind=>{ const of=v.stages.filter(s=>groupStage(s.stage_id)===kind), verified=of.filter(s=>s.lifecycle==='VERIFIED').length;
      const lifecycle=of.some(s=>s.lifecycle==='BLOCKED') ? 'BLOCKED' : verified===of.length ? 'VERIFIED' : verified || of.some(s=>['IN_PROGRESS','READY_FOR_VERIFICATION'].includes(s.lifecycle)) ? 'IN_PROGRESS' : 'PENDING';
      return {stage_id:kind,lifecycle,groups:of.length,verified,evidence_count:of.reduce((a,s)=>a+(Number(s.evidence_count) || 0),0),words:`${codeWords(lifecycle)} · ${count(verified)} / ${countText(of.length,'{n} group','{n} groups')}`}; }).filter(s=>s.groups);
    const groups=[...new Set(v.stages.map(s=>s.stage_id.split('_')[0]))].map(id=>{
      const own=v.stages.filter(s=>s.stage_id.startsWith(id+'_'));
      const failed=v.current_scope?.failed_unit_ids?.includes(id);
      const blocked=own.find(s=>s.lifecycle==='BLOCKED');
      const active=own.find(s=>['IN_PROGRESS','READY_FOR_VERIFICATION'].includes(s.lifecycle));
      const lifecycle=failed ? 'failed' : blocked ? 'blocked' : own.every(s=>s.lifecycle==='VERIFIED') ? 'done' : active ? 'running' : 'waiting';
      return {id,lifecycle,current_stage:groupStage((blocked || active)?.stage_id) || ''};
    });
    const parallel={total:groups.length,groups,scope:v.current_scope};
    for(const state of ['done','running','waiting','blocked','failed']) parallel[state]=groups.filter(g=>g.lifecycle===state).length;
    return {...v,stages,parallel,status:{...v.status,current_stage:''}};
  }
  const workView=()=>bookView(S.work?.view || null);
  // the evidence owner's count of a book stage while its run continues (10.10), for the book the page shows
  const runCount=(stage)=>S.view?.state==='EVIDENCE_REFRESH_IN_PROGRESS' && S.view?.task_id===S.work?.task ? (S.view?.coverage_progress?.preparation || []).find(r=>r.stage===stage) || null : null;
  // the mode the followed Task's preparation runs in: the preview it was confirmed from, else
  // the packet it produced, else not stated
  const workMode=()=>{const w=S.work;if(!w)return modeWords(null);if(w.mode)return modeWords({source_mode:w.mode});const facts=packetJson();if(facts && S.packetTask===w.task)return modeWords({source_mode:(facts.document_index || []).some(d=>d?.source_name==='ISSUER_RECORDED') ? 'RECORDED' : ''});return modeWords(null);};
  const managedWork=(w)=>Boolean(w) && w.kind===EVIDENCE_KIND && !preparing(w.goal) && !(boundFor(S.key).imported.has(w.task) || [...S.bound.values()].some(b=>b.imported.has(w.task)));
  const WORK={key:'review',get page(){return pages.has(app.page) ? app.page : 'evidence';},railLabel:'Task stages; select to inspect, not execute',factsLabel:'Task facts',
    get steps(){return (workView()?.stages || []).map(s=>[s.stage_id,stageOf(s.stage_id).word]);},
    get title(){return S.work?.kind===REVIEW_KIND ? 'CRO review work' : preparing(S.work?.goal) ? 'Source preparation work' : managedWork(S.work) ? 'Analysis work' : 'Answer validation work';},
    get lines(){const m=workMode();return {...stageLines(),resolve_official_sources:m.resolve,acquire_source_evidence:m.acquire,...(managedWork(S.work) ? {analyze_evidence:t('The analysis stage runs the configured actor over the admitted passages when a managed refresh was requested (Provider quota), or validates an external answer when one was submitted; this page recorded no submission for this Task.')} : {})};},
    fallbackLine:'Continuing the reported stage.',
    parallelTable:groups=>collectionTable('parallel-groups',groups,[{label:t('Group'),type:'id'},{label:t('State'),type:'status'},{label:t('Stage'),type:'text',absorb:true}],g=>[mono(g.id,SHORT.id),t({done:'Done',running:'Running',waiting:'Waiting',blocked:'Blocked',failed:'Failed'}[g.lifecycle]),g.current_stage ? codeWords(g.current_stage) : '']),
    get oneUnit(){const run=coverageRun(S.work?.view);return Object.fromEntries((workView()?.stages || []).map(s=>[s.stage_id,run ? 'One unit a group; Task Control verifies each group of the book.' : 'This stage is one unit of the owner\'s work; it reports at stage boundaries only.']));},
    view:()=>workView(),
    work:(b,v,state,shown)=>{
      if(!coverageRun(S.work?.view)) return {count:html`<strong>${v.verified_stage_count}</strong><span class="tp-denom"> / ${v.total_stage_count} ${t('stages verified')}</span>`,bar:null,detail:t('Stage count, not a percentage of documents, passages or review: the owner reports no finer unit. Document availability, issuer coverage and review clearance are different counts and are read from their owners.')};
      // a book stage: the evidence owner's own count while the run continues, else its groups verified
      const s=v.stages.find(x=>x.stage_id===shown), r=runCount(shown), total=Number(r?.total) || 0;
      if(r && total>0) return {...workCount(Number(r.completed) || 0,total,t(PREPARING_UNITS[r.unit_name] || r.unit_name)),detail:t('Counted by the evidence owner across the book\'s groups as the run continues; not a time estimate.')};
      if(r) return {count:html`<span class="tp-denom">${t('The owner has not counted this stage\'s work yet')}</span>`,bar:null,detail:t('Counted by the evidence owner across the book\'s groups as the run continues; not a time estimate.')};
      return {...workCount(s?.verified || 0,s?.groups || 0,t('groups verified')),detail:t('One unit a group; Task Control verifies each group of the book.')};
    },
    aside:(b,v,state,shown,isCurrent)=>{if(!isCurrent){ const s=coverageRun(S.work?.view) ? v.stages.find(x=>x.stage_id===shown) : null; return s ? LiveWorkArea.factsShell(WORK,{label:t('Stage record'),title:t('Stage record'),caption:t('One unit a group; each group\'s evidence is its own Task Control record.'),facts:kv([[t('Groups verified'),`${count(s.verified)} / ${count(s.groups)}`],[t('Evidence'),countText(s.evidence_count,'{n} reference','{n} references')]]),rail:''}) : null; }const w=S.work,bookOf=w && S.taskBooks.get(w.task);const bookRow=bookOf ? html`${mono(bookOf.experiment_task_id || bookOf.update_task_id || bookOf.result_hash || bookOf.handoff_hash,SHORT.id)}${infoMark(t('the book the owner admitted this Task for'))}` : t('scope unresolved: no owner fact binds this Task to a book on this page');const rows=[[t('Task'),html`${codeWords(v.task_kind)} <span class="sub-cell">${v.status?.goal_summary || ''}</span>`],[t('Book'),bookRow],[t('Kind of work'),t(w?.kind===REVIEW_KIND ? 'independent CRO assessment sealed and published by the product' : preparing(w?.goal) ? (workMode().kind==='live' ? 'source preparation by admitted official acquisition; no analyst or CRO inference' : workMode().kind==='recorded' ? 'local source preparation over the recorded package; no analyst or CRO inference' : 'source preparation; its source mode is not stated on this page') : managedWork(w) ? 'analysis publication; whether an external answer or the configured actor produced it is not recorded on this page' : 'product validation and publication of an answer this page submitted')],[t('Liveness'),typeof LiveTasks!=='undefined' && LiveTasks.liveness ? (LiveTasks.liveness(v) || t('the Task is not executing; its lifecycle says where it stands')) : v.liveness?.status || '']];return LiveWorkArea.factsShell(WORK,{label:t('Task facts'),title:t('This Task'),caption:t('What Task Control records; completion of a Task never implies CRO clearance'),facts:kv(rows),rail:html`<section class="ui-log-rail prep-log-rail"><div><span>${t('Task')}</span><strong class="mono">${short(v.task_id)}</strong></div><div><span>${t('Task Control')}</span><strong>${v.verified_stage_count} / ${v.total_stage_count} ${t('verified')}</strong></div></section>`});},
    absorb:()=>{},completedNow:(b)=>Boolean(b?.done)};
  async function watch(task,quiet=false,mode='',initial=null) {
    if(!task) return;
    const known=taskOf(task);
    S.work={task,kind:known?.task_kind || '',goal:known?.goal_summary || '',mode:mode || (S.work?.task===task ? S.work.mode : ''),view:null,readAt:0,stale:null,error:'',reading:false,done:false};
    writeRoute(routeUpdate());
    return readWork(quiet,initial);
  }
  async function readWork(quiet=false,initial=null) {
    const w=S.work; if(!w || w.reading) return;
    const nav=S.nav,reading={};w.reading=reading;
    const currentRead=()=>S.work===w && w.reading===reading && nav===S.nav && pages.has(app.page) && app.page!=='books';
    try {
      const view=initial || await Data.read('/api/tasks/recovery?'+new URLSearchParams({task_id:w.task}));
      if(!currentRead()) return;
      if(view.task_id!==w.task || ![EVIDENCE_KIND,REVIEW_KIND].includes(view.task_kind)) { w.error=t('Task {task} is {kind}, not evidence or review work; it is read on Tasks.',{task:short(view.task_id),kind:view.task_kind || ''}); w.view=null; w.readAt=Date.now(); return; }
      const wasMoving=Boolean(w.view && stateMoving(w.view.lifecycle));
      w.kind=view.task_kind; w.goal=view.status?.goal_summary || w.goal; w.view=view; w.stale=null; w.readAt=Date.now(); w.error='';
      noteFacts(w.task,artifactFacts(view));
      const area=typeof LiveWorkArea!=='undefined' ? LiveWorkArea.areaFor(WORK,w.task) : null;
      if(view.lifecycle==='SUCCEEDED' && !w.done) { w.done=true; if(area && wasMoving) area.accent=true; if(w.goal===PREPARE_WORDS && !S.packetTask && boundFor().prepare.has(w.task)){S.packetTask=w.task;S.packetUnit='';loadDrafts();writeRoute(routeUpdate());} void Data.refreshHistory(); if(!quiet || wasMoving) void refresh(); }
      if(area) LiveWorkArea.noteTransition(bookView(view),area);
    } catch(e) {
      if(currentRead() && e.name!=='AbortError') { if(w.view) w.stale={since:w.readAt || Date.now(),error:e.message}; else w.error=e.message; }
    } finally { if(S.work===w && w.reading===reading) { w.reading=false; if(nav===S.nav && pages.has(app.page) && app.page!=='books'){if(quiet && typeof patchMain==='function') patchMain(); else render();} } }
  }
  /* While a preparation runs, the book's progress is re-read on the activity's beat (contract 10.9,
   * 10.10): its counts and its units' states patched in place, the page otherwise untouched; a new
   * state is a refresh. A pinned reading never moves. */
  let progressReading=false;
  async function readProgress() {
    if(progressReading || S.busy || S.pin || S.view?.state!=='EVIDENCE_REFRESH_IN_PROGRESS' || !pages.has(app.page)) return;
    progressReading=true; const ticket=S.revision;
    try {
      const view=await Data.read('/api/evidence-cro?'+query({}));
      if(ticket!==S.revision || !S.view) return;
      if(view.state!==S.view.state) { void refresh(); return; }
      S.view={...S.view,coverage_progress:view.coverage_progress};
      if(typeof patchMain==='function') patchMain(); else render();
    } catch { /* the next beat reads again; the painted counts stay as they were */ } finally { progressReading=false; }
  }
  /* The book's own running Task, named by its projection (an owner fact), is followed on its page as
   * the data scenes follow theirs: its work area shows without a press; another followed Task stays. */
  function followBookTask() {
    const v=S.view, id=v?.task_id; if(!id || S.pin || S.work || v.state!=='EVIDENCE_REFRESH_IN_PROGRESS' || !pages.has(app.page)) return;
    S.taskBooks.set(id,{...S.selector}); void watch(id,true);
  }
  function observe() {
    void readProgress();
    followBookTask();
    const w=S.work; if(!w || !pages.has(app.page) || w.reading) return;
    const v=w.view, projection=taskOf(w.task);
    const differs=Boolean(v && projection && (projection.lifecycle!==v.lifecycle || projection.verified_stage_count!==v.verified_stage_count));
    if(v && !stateMoving(v.lifecycle) && !v.operation_running && !(projection && stateMoving(projection.lifecycle)) && !differs) return;
    void readWork(true);
  }
  function dismissWork() { S.nav+=1; S.work=null; S.unresolved=null; writeRoute({work:''}); render(); }
  function workSection(page) {
    const w=S.work; if(!w || app.page!==page) return '';
    const v=w.view, bookOf=S.taskBooks.get(w.task);
    const standing=v && typeof LiveTasks!=='undefined' && LiveTasks.standing ? LiveTasks.standing(v) : null;
    const kicker=w.kind===REVIEW_KIND ? 'CRO review · Task' : preparing(w.goal) ? 'Source preparation · Task' : managedWork(w) ? 'Analysis · Task' : 'Answer validation · Task';
    const head=html`<header class="prep-scene-head lab-work-head"><div><p class="caption">${t(kicker)}</p><h2>${t('Task')} <span class="mono" data-tip="${w.task}">${short(w.task)}</span></h2>${standing ? html`<p class="prep-liveness lab-standing" data-tone="${standing[0]}">${icon(standing[1])}<span>${standing[2]}</span></p>` : ''}${!bookOf ? html`<p class="caption">${t('Scope unresolved on this page: no owner fact binds this Task to the book shown; its stages are read without claiming the book.')}</p>` : ''}</div><div class="flow">${btn(t('Inspect task'),'task',w.task,'button compact')}${btn(t('Read again'),'review-work-refresh','','button compact')}${btn(t('Dismiss'),'review-dismiss-work','','button compact')}</div></header>`;
    if(!v) return html`<section class="prep-scene lab-work review-work" data-scene="pending">${head}${w.error ? notRead(t('Task not read'),w.error) : noteLine(t('Reading the Task'),t('Its stages and liveness come from Task Control; nothing is started by reading.'))}</section>`;
    // N6 (laws 72, 98): a finished Task is one line -- what it published, what it was, its reader; its stages are the Task's own record
    if(v.lifecycle==='SUCCEEDED') {
      const whole=Boolean(preparing(w.goal) && coverageRun(v));
      const title=t(w.kind===REVIEW_KIND ? 'Review published' : whole ? 'Source preparation finished' : preparing(w.goal) ? 'Sources prepared' : managedWork(w) ? 'Analysis published' : 'Answer validated and published');
      const words=t(w.kind===REVIEW_KIND ? 'The published review is read on the Review desk and exported on the report page. Publication is a routed recommendation, not clearance to trade.' : whole ? 'Read the groups to see the prepared sources, any failures and the next steps. Preparation is not analysis or a CRO review.' : preparing(w.goal) ? 'The exact Analyst packet of this Task can be read and handed off; a prepared packet is not an Analyst conclusion.' : 'The analysis is published. It is not a CRO review; the dossier can now be read for the independent assessment.');
      const reader=w.kind===REVIEW_KIND ? btn(t('Open the Review desk'),'review-step','evidence','button primary compact') : w.goal===PREPARE_WORDS ? (S.selector ? btn(t('Read the Analyst packet'),'review-use-task',w.task,'button primary compact') : '') : preparing(w.goal) ? (S.selector ? btn(t('Read the groups\' packets'),'review-step','evidence-stream','button primary compact') : '') : (S.selector ? btn(t('Read the CRO dossier'),'review-dossier','','button primary compact') : '');
      return html`<section class="prep-scene lab-work review-work lab-work-done" data-scene="review-task" data-lifecycle="SUCCEEDED" data-bound="${Boolean(bookOf)}">${noteLine(html`<strong>${title}</strong> <span class="mono">${short(w.task)}</span>${infoMark(bookOf ? words : words+' '+t('Scope unresolved on this page: no owner fact binds this Task to the book shown; its stages are read without claiming the book.'))}`,'','ok',html`${reader}${btn(t('Dismiss'),'review-dismiss-work','','text-btn')}`,'checkcircle')}</section>`;
    }
    const done=['FAILED','BLOCKED','RECOVERY_REQUIRED','CANCELLED'].includes(v.lifecycle) ? refusal({code:v.worker_failure?.failure_code || '',reason:t('Nothing was published; the drafts and the read packet or dossier stay as they were. Tasks holds the owner\'s recovery actions.')},'warning',{state:v.lifecycle})
      : w.stale ? noteLine(t('Not re-read since {time}',{time:when(new Date(w.stale.since).toISOString())}),html`${w.stale.error} · ${t('The last observation is kept below; nothing about the Task is inferred from a failed read.')}`,'warning') : '';
    const area=typeof LiveWorkArea!=='undefined' ? LiveWorkArea.workArea(WORK,{done:w.done},bookView(v),{readAt:w.readAt,stale:w.stale,recovered:false}) : '';
    return html`<section class="prep-scene lab-work review-work" data-scene="review-task" data-lifecycle="${v.lifecycle}" data-bound="${Boolean(bookOf)}">${head}${done}${area}</section>`;
  }
  /* A cold open reads the Task owner's admitted subjects. A legacy Task can only be bound
   * by an explicit book choice verified by its packet or analysis, never by page memory. */
  async function resolveScope(id,view,candidate) {
    if(!view || view.task_id!==id || ![EVIDENCE_KIND,REVIEW_KIND].includes(view.task_kind)) return null;
    const subjects=taskSubjects(view),book=candidate ? subjects.find(s=>key(s)===key(candidate)) : subjects.length===1 ? subjects[0] : null;
    if(book){
      const group=view.task_kind===REVIEW_KIND ? 'review' : preparing(view.status?.goal_summary) ? 'prepare' : 'analysis';
      return {group,book,page:group==='analysis' ? 'handoff' : 'evidence',view,facts:artifactFacts(view)};
    }
    if(subjects.length) return null;
    if(view.task_kind!==EVIDENCE_KIND || !candidate) return null;
    const prepareOnly=preparing(view.status?.goal_summary) || view.total_stage_count===6;
    try{
      if(prepareOnly){
        const doc=await Data.readDocument('/api/evidence/packet?'+new URLSearchParams({...candidate,task_id:id}));
        return doc.value.submission_template ? {group:'prepare',book:candidate,page:'handoff',packet:doc,view,facts:packetIdentity(doc.value)} : null;
      }
      const pub=artifactHash(view,'alternative_evidence_analysis_publication'); if(!pub) return null;
      const projection=await Data.read('/api/evidence-cro?'+new URLSearchParams(candidate));
      let asOf='';
      const named=(projection.eligible_versions || []).some(v=>v.analysis_publication_hash===pub) || (await Data.readDocument('/api/cro/dossier?'+new URLSearchParams(candidate)).then(d=>{const dd=d.value?.dossier;if(dd?.analysis_publication_hash===pub){asOf=dd.evidence_as_of || '';return true;}return false;}).catch(()=>false));
      return named ? {group:'analysis',book:candidate,page:'handoff',view,facts:{analysis:pub,asOf}} : null;
    }catch{ return null; } // the owner did not bind the Task to that book
  }
  /* The Task's book is opened and the Task followed, as one navigation: the continuation
   * belongs to the generation this open created and does nothing once the reader has moved. */
  function adopt(id,scope) {
    bind(scope.group,id,key(scope.book),scope.book,scope.facts);
    const opened=open(scope.book,scope.group==='review' ? scope.facts.publication || '' : '',scope.page), gen=S.nav;
    const watched=watch(id,true,'',scope.view);
    return Promise.all([opened,watched]).then(async()=>{ if(gen!==S.nav) return false; if(scope.group==='prepare'){S.packetTask=id;S.packetUnit=scope.packet?.value?.coverage_unit?.unit_id || '';loadDrafts();adoptPacketRequest(S.view);writeRoute(routeUpdate());if(scope.packet)S.packet={...scope.packet,key:S.key,task:id,unit:S.packetUnit,readGen:S.readGen};else if(scope.view.lifecycle==='SUCCEEDED' && (!scope.view.current_scope || scope.view.current_scope.units_total===1)) await getBundle('analyst',null,true);} return gen===S.nav; });
  }
  /* From Task Center, the owner's answer is accepted only in this navigation generation. */
  async function openTask(id,wanted=()=>true) {
    if(!wanted()) return false;
    let addressed=null;try{addressed=JSON.parse(hashParams().get('review_selector') || 'null');}catch{addressed={};}
    const gen=++S.nav; // this return is a navigation; any later one supersedes it
    let view=null,error='';
    try { view=await Data.read('/api/tasks/recovery?'+new URLSearchParams({task_id:id})); } catch(e) { error=e.message; }
    if(gen!==S.nav || !wanted()) return false; // the reader navigated while the owner was answering: nothing is opened for this reply
    const scope=await resolveScope(id,view,addressed);
    if(gen!==S.nav || !wanted()) return false;
    if(scope) return adopt(id,scope);
    S.nav+=1; closeDialog();reset();S.selector=null;S.key=key(null);S.pin='';S.view=null;S.status='empty';S.packetTask='';S.packetUnit='';S.work=null;
    S.taskBooks.delete(id);
    const why=view?.subject_refusal;
    S.unresolved={task:id,kind:view?.task_kind || '',view,reason:error || (why ? why.detail || why.explanation || codeWords(why.failure_code) : taskSubjects(view).length ? t('Choose the book admitted onto this Task.') : t('no owner fact binds this Task to a book on this page: choose the exact book to bind it (the owner verifies), or read it on Tasks'))};
    app.page='handoff';writeRoute(routeUpdate({page:'handoff',work:id}));render();
    return watch(id,true,'',view);
  }
  /* A book chosen while a Task's scope is unresolved is offered to the owner as that Task's
   * book; the Task stays unresolved when the owner does not bind it. */
  // the Reading picker: the reading already shown (working, or the same pin) is read again in place -- the page never shrinks and grows (the user's reading); another reading opens
  const reread=(pin)=>{ if(S.selector && (pin || '')===S.pin && (S.status==='ready' || S.status==='refreshing')){ S.nav+=1; writeRoute(routeUpdate()); return refresh(); }
    if(S.selector && (pin || '')!==S.pin){ S.pin=pin || ''; pushRoute(routeUpdate()); } // a pin change is a pushed level: `<-` returns to the reading before it (law 86, round F6)
    return open(S.selector,pin || '',app.page); };
  async function choose(value) {
    if(!value) return;
    const selector=JSON.parse(value), pending=S.unresolved;
    // the book already shown, chosen again (the user's reading: the page shook): its view is read again and what the page read stays painted; another book opens
    if(!pending && key(selector)===S.key && (S.status==='ready' || S.status==='refreshing')){ S.nav+=1; writeRoute(routeUpdate()); return refresh(); } // the reading (working or pinned) stays
    if(!pending) return open(selector,'',app.page);
    const gen=++S.nav;
    const scope=await resolveScope(pending.task,pending.view,selector);
    if(gen!==S.nav) return;
    if(scope && key(scope.book)===key(selector)) { S.unresolved=null; return adopt(pending.task,scope); }
    const opened=open(selector,'',app.page), g=S.nav; await opened;
    if(g!==S.nav) return;
    S.unresolved={...pending,reason:t('the owner did not bind Task {task} to this book; its scope stays unresolved',{task:short(pending.task)})};
    return watch(pending.task,true);
  }
  /* ---- the report and the delivery ---- */
  async function readExport(request=null,quiet=false) { // quiet (round F4): the Review page reads the published dispositions in the background; the page is never busy for it
    if(!ready() || S.exportPending || (app.page==='report' && isUpdateBook())) return;
    // the report the page shows is the request's when one was captured for it, else the pin's
    const captured=request || (S.exportRequest && S.exportRequest.key===S.key && S.exportRequest.request.review_publication_hash===(S.pin || workingPublication()) ? S.exportRequest.request : null);
    const hash=captured ? captured.review_publication_hash : (S.pin || workingPublication());
    if(!hash){S.exportDoc=null;render();return;}
    if(!quiet)S.busy='export';S.exportPending=true;S.error='';const ticket=S.revision, asked={key:S.key,hash};repaintMain();
    try {const doc=await Data.readDocument(Data.route('EVIDENCE_CRO_EXPORT')+'?'+(captured ? new URLSearchParams(requestFields(captured)) : query({review_publication_hash:hash})));if(ticket===S.revision && S.key===asked.key && (S.pin || workingPublication())===asked.hash)S.exportDoc={...doc,hash};}
    catch(e){if(ticket===S.revision)S.error=e.message;}
    finally{if(ticket===S.revision){S.exportPending=false;if(!quiet)S.busy='';repaintMain();}}
  }
  async function deliveryOptions() {
    if(!ready() || !S.selector?.experiment_task_id)return;
    const ticket=S.revision;S.busy='links';render();
    try{const b=await Data.read('/api/experiments/risk-links?'+new URLSearchParams({task_id:S.selector.experiment_task_id}));if(ticket===S.revision){S.riskLinks=b.links || [];S.riskLinkRefusals=b.refused_links || [];}}
    catch(e){if(ticket===S.revision){S.error=e.message;S.riskLinks=[];S.riskLinkRefusals=[];}}
    finally{if(ticket===S.revision){S.busy='';render();if($('#dialog')?.open)deliveryDialog();}}
  }
  async function assembleDelivery(quiet=false) {
    const s=S.selector;
    const update=isUpdateBook();
    if(!ready() || !(update ? SELECTOR_FORMS.update.every(k=>named(s,k)) : s?.experiment_task_id && s.experiment_receipt_hash && s.portfolio_session))return;
    S.busy='delivery';S.error='';S.deliveryRefusal=null;S.deliveryAttempt=S.key+'|'+(S.pin || workingPublication());const mutation={};S.deliveryMutation=mutation;const ticket=S.revision,risk=S.risk,comparison=S.comparison,question=S.question,intent=Data.navigationIntent();render();
    const current=()=>ticket===S.revision && Data.navigationCurrent(intent);
    try{
      const payload=update ? Object.fromEntries(SELECTOR_FORMS.update.map(k=>[k,s[k]])) : {task_id:s.experiment_task_id,experiment_receipt_hash:s.experiment_receipt_hash,portfolio_session:s.portfolio_session};
      const review=S.pin || workingPublication(); if(review)payload.review_publication_hash=review;
      if(!update && S.risk)payload.risk_report_hash=S.risk;
      if(!update && S.comparison){payload.left_task_id=s.experiment_task_id;payload.right_task_id=S.comparison;}
      if(S.question.trim())payload.delivery_question=S.question.trim();
      const doc=await Data.postDocument('/api/research/delivery',payload);
      if(current() && risk===S.risk && comparison===S.comparison && question===S.question){
        S.delivery={...doc,review:review || ''};const exported=doc.value?.sections?.evidence_cro?.value;
        if(update && exported)S.exportDoc={value:exported,text:JSON.stringify(exported),hash:review};
        if(!quiet){closeDialog();notify(t('Delivery composed · export {hash}',{hash:short(doc.value?.export_hash || '', SHORT.hash)}));}
      }
    }catch(e){if(current()){S.deliveryRefusal=e.body || null;S.error=String(e?.message || e)+(e?.body ? '' : ' · '+t('no owner answer arrived; nothing is assumed'));}}
    finally{if(S.deliveryMutation===mutation){S.deliveryMutation=null;if(!current())S.deliveryAttempt='';if(S.busy==='delivery')S.busy='';repaintMain();}}
  }
  /* ---- the report as a document (round 78): Scope, one section per issuer (its issues,
   * findings and citations as rows), Decision, Delivery -- in the object model's order, each
   * section's provenance in the Facts panel's section of the same name. ---- */
  const reportOf=()=>S.exportDoc?.value || null;
  const deliveryOf=()=>S.delivery?.value || null;
  const SECTION_WORDS={PRESENT:'present',NOT_SELECTED:'absent · not selected (not zero, not clearance)',NOT_PUBLISHED:'absent · not published',INCOMPATIBLE:'refused by the owner',HISTORICAL_REVIEW:'historical review',EXPIRED:'review present but expired'};
  /* The Report page's headline (Stripe's object head, round F5): the verdict, its completeness, the
   * published day, the reviewer and the reviewed weight in one line; the cycle's sentence otherwise. */
  function reportWords() {
    const v=S.view, x=reportOf();
    if(!v || S.refusal) return {sentence:'',primary:null};
    if(!(reviewPublished() || historical())) return {sentence:cycle().sentence,primary:null};
    const parts=[codeWords(v.disposition),v.review_state ? codeWords(v.review_state) : '',publishedAt() ? t('published {d}',{d:publishedAt()}) : '',attributionWords(v.review_attribution || [],{plain:true}) || ''].filter(Boolean); // round H4 (R1): the reviewed weight is Scope's figure, not the headline's sentence
    return {sentence:parts.join(' · ')+'.',primary:null};
  }
  /* The publications of this book (round F5): every review publication history records, newest
   * first, the current and the pinned one marked; a row pins its publication (a soft open). One
   * line a row -- the day and the state word; the hash and the Task on hover (B1: a section). */
  function publicationsSection() {
    const pubs=publications(); if(!pubs.length) return '';
    const list=collectionTable('report-publications',pubs,[{label:t('Review'),type:'text',absorb:true},{label:t('State'),type:'text'},{label:t('Publication'),type:'text'},{label:t('Action'),type:'text',cls:'col-tight'}],p=>{const named=p.hash===workingPublication(),current=named && reviewPublished(),pinned=p.hash===S.pin,lapsed=p.lapsed || (named && !reviewPublished());return [html`${t(lapsed ? 'Historical review' : 'Review')} · ${dayWord(p.recorded)}`,t(pinned ? 'Pinned' : lapsed ? 'Historical' : current ? 'Current' : 'Historical'),hashCell(p.hash),pinned ? btn(t('Unpin'),'review-current','','text-btn') : btn(t('Pin'),'review-pin',p.hash,'text-btn')];});
    return panel(t('Publications'),t('Publications in the loaded History; a pinned one is read exactly as sealed.'),html`${list}${Data.hasMoreHistory?.() ? html`<p class="caption">${t('Earlier History is not read yet; it may contain earlier publications for this book.')} ${btn(t('Read earlier History'),'history-more','','text-btn')}${link(t('History'),'history','text-btn')}</p>` : ''}`,'','data-report-section="publications" data-box="table"');
  }
  /* ---- the report's history (round E5): what changed since the prior review of this book, from
   * the owner's comparison -- a prior finding no longer mentioned is never "resolved"; the review's
   * times as facts; the dossier's unresolved questions carried. ---- */
  const changeRows=(c)=>{if(!c || typeof c!=='object')return [];if(Array.isArray(c.findings))return c.findings;return ['new','changed','no_longer_found','unchanged'].flatMap(kind=>(c.findings?.[kind] || []).map(r=>({...r,changeKind:kind})));};
  const changeKindWords=(kind)=>({new:t('New'),changed:t('Changed'),no_longer_found:t('not mentioned again'),unchanged:t('Unchanged')})[kind] || '';
  function changesSection() {
    const x=reportOf(), c=x?.changes_since_prior_review; if(!x || c===undefined) return '';
    if(c===null) return ''; // N3 (law 81): no prior review, nothing to compare -- not shown
    const rows=changeRows(c);
    const facts=kv([[t('Comparison'),html`${codeWords(c.status)}${infoMark(said(c.reason))}`],[t('Prior review'),html`<span class="mono">${short(c.prior_publication_hash,SHORT.hash)}</span>`],[t('This review'),html`<span class="mono">${short(c.current_publication_hash,SHORT.hash)}</span>`]]);
    const list=rows.length ? collectionTable('report-changes',rows,[{label:t('Finding'),type:'text',cls:'col-tight'},{label:t('Issuers'),type:'text',cls:'col-tight'},{label:t('Topic'),type:'text',cls:'col-tight'},{label:t('Before'),type:'text',cls:'col-tight'},{label:t('Now'),type:'text',cls:'col-tight'},{label:t('Summary'),type:'text',absorb:true}],r=>[html`<span class="mono">${r.finding_handle || r.prior_finding_handle}</span>${r.changeKind ? html` <span class="sub-cell">${changeKindWords(r.changeKind)}</span>` : ''}${r.prior_finding_handle && r.finding_handle && r.prior_finding_handle!==r.finding_handle ? html` <span class="sub-cell mono">${r.prior_finding_handle}</span>` : ''}`,entityNames(r.affected_entities) || '',codeWords(r.topic),r.prior_disposition ? codeWords(r.prior_disposition) : t('not in the prior review'),r.disposition ? codeWords(r.disposition) : t('not mentioned again'),html`<span class="owner-text">${r.summary || ''}</span>${r.meaning ? html` <span class="sub-cell">${said(r.meaning)}</span>` : ''}`]) : (c.status && String(c.status).startsWith('REVIEWS_NOT') ? '' : emptyState(t('The owner compares no finding between the two reviews.')));
    return panel(t('Changes since the prior review'),t('The owner\'s comparison; a finding not mentioned again is not resolved.'),html`${facts}${list}`,'','data-report-section="changes" data-box="table"');
  }
  function reviewTimes() {
    const x=reportOf(); if(!x) return '';
    const rec=x.review?.recommendation || {}, pub=x.review?.publication || {}, ev=x.evidence?.publication || {};
    return kv([[t('Evidence as-of'),dayWord(rec.evidence_as_of)],[t('Evidence expires'),dayWord(rec.evidence_expires_at)],[t('Evidence published'),when(ev.published_at)],[t('Review published'),when(pub.published_at)],[t('Readback'),html`${codeWords(x.review_status)}${infoMark(t('exact in its sealed content; the rendering here may differ from the exported HTML'))}`]]);
  }
  function questionsSection() {
    const q=reportOf()?.review?.dossier?.unresolved_questions || []; if(!q.length) return '';
    // B1 (item 8): a question is a row -- its unit a small reference, the question whole at its measure -- and the way to the Review, where the Analyst's open questions are read
    const units=new Set(q.map(x=>x.unit_id || '')).size>1; // a unit is said where the questions span more than one
    const list=collectionTable('report-questions',q,[...(units ? [{label:t('Unit'),type:'text',cls:'col-tight'}] : []),{label:t('Question'),type:'text',absorb:true}],x=>[...(units ? [x.unit_id || ''] : []),btnAttrs(x.text || '','review-questions','','text-btn','data-row-press')]);
    return panel(html`${t('Unresolved questions')} <span class="num">${count(q.length)}</span>`,t('Carried from the dossier; the review answered around them. A question opens the Review.'),list,'','data-report-section="questions" data-box="table"');
  }
  function reportIssuers(){
    const rows=issuerOrder(S.view?.issuer_rows || []),citations=S.view?.citations || [];
    const figures=(r)=>({findings:r.finding_count,issues:r.adverse_issue_count,citations:citations.filter(c=>c.entity_id===r.entity_id).length});
    const groups=[['any','With findings, issues or citations',n=>n.findings>0 || n.issues>0 || n.citations>0],['none','No findings, issues or citations',n=>n.findings===0 && n.issues===0 && n.citations===0],['findings','With findings',n=>n.findings>0],['issues','With issues',n=>n.issues>0],['citations','With citations',n=>n.citations>0]];
    const all=t('All ({n})',{n:count(rows.length)}),countOptions=[['all',all],...groups.map(([key,words,predicate])=>[key,t('{label} ({n})',{label:t(words),n:count(rows.filter(r=>predicate(figures(r))).length)})])];
    const conclusions=[...new Set(rows.map(r=>r.conclusion).filter(Boolean))],conclusionOptions=[['all',all],...conclusions.map(code=>[code,t('{label} ({n})',{label:codeWords(code),n:count(rows.filter(r=>r.conclusion===code).length)})])];
    const predicate=groups.find(([key])=>key===S.reportCount)?.[2],found=rows.filter(r=>(!predicate || predicate(figures(r))) && (!S.reportConclusion || r.conclusion===S.reportConclusion));
    const filters=rows.length ? html`<div class="holdings-filters">${filterBar([{name:'evidence-report-count',label:t('Counts'),value:S.reportCount,options:countOptions},{name:'evidence-report-conclusion',label:t('Conclusion'),value:S.reportConclusion,options:conclusionOptions}],{pending:S.pendingFilter})}<span class="caption">${all}</span></div>` : '';
    const list=collectionTable('report-issuers',found,[{label:t('Issuer'),type:'text',absorb:true},{label:t('Conclusion'),type:'text'},{label:t('Findings'),type:'num',cls:'col-tight'},{label:t('review|Issues'),type:'num',cls:'col-tight'},{label:t('Citations'),type:'num',cls:'col-tight'}],r=>{const n=figures(r);return [btnAttrs(issuerName(r),'review-live-item',r.entity_id,'text-btn','data-row-press'),r.conclusion ? codeWords(r.conclusion) : '',n.findings==null ? '' : count(n.findings),n.issues==null ? '' : count(n.issues),count(n.citations)];},{words:r=>[r.entity_id,entityNames(r.tickers),codeWords(r.conclusion)].join(' '),total:rows.length,tools:filters});
    return detailSplit(panel(t('Issuers'),'',rows.length ? list : emptyState(t('No issuers are named in this review.')),'','data-report-section="issuers" data-box="table"'),'review',{pane:readingPaneMarkup()});
  }
  function reportDocument(withDelivery=true) {
    const v=S.view, x=reportOf();
    const sc=v?.scope_coverage, cv=v?.coverage, reviewed=cv && cv.reviewed_ending_weight!=null ? cv.reviewed_ending_weight : sc?.whole_book_reviewed_weight;
    // the third review (2026-09-24): the reviewed weight, the floor and the mapping are the Overview's live ruler -- the Report says what it covered beyond them, and where the ruler is
    const measures=[...(cv && cv.reviewed_absolute_change!=null ? [[t('Change reviewed'),pctText(cv.reviewed_absolute_change)]] : []),...(sc ? [[t('Scope'),t('{p} of {n} positions \u00b7 {i}',{p:count(sc.reviewed_positions),n:count(sc.book_positions),i:countText(sc.reviewed_issuers,'{n} issuer','{n} issuers')})]] : []),...(reviewed!=null ? [[t('Coverage'),btnAttrs(html`${t('On the Overview')}${icon('arrow')}`,'review-step','evidence','text-btn compact')]] : [])]; // figures (R1), as the publication sealed them
    // B1 (item 8): the book and its evidence are the head's; Scope says what the review covered and how this reading holds it (the readback's exactness, its words the (i))
    const scope=panel(t('Scope'),'',html`${kv([...(x ? [[t('Readback'),html`${codeWords(x.review_status)}${infoMark(`${said(x.claim)} ${t('Exact in its sealed content; the rendering here may differ from the exported HTML.')}`)}`]] : []),...measures],'kv-columns')}`,'','data-report-section="scope"');
    const issuers=x ? reportIssuers() : '';
    // the reviews (2026-09-24): a required action is the decision's first line and the way on it names -- the issuer it reads, else its pipeline (the Overview's blocker says the same)
    const decided=Boolean(v && (reviewPublished() || historical()));
    const actions=(decided ? v.required_actions || [] : []).map(a=>objectRow({state:actionStanding(a).state,name:codeWords(a.action),why:said(a.reason),to:actionWay(a),cls:'evidence-row'},{key:'action:'+a.action+':'+(a.entity_id || ''),word:actionStanding(a).word,columns:['scope'],props:[a.entity_id || actionWay(a)?.word || ''],attrs:a.reason ? html`data-tip="${said(a.reason)}"` : ''}));
    const limits=v ? [...gapLines(v.gaps || []),...(v.claim_limits || []),...reviewLimits(v)] : [];
    const decision=!v ? '' : !decided ? panel(t('Decision'),'',html`${emptyState(t(publications().some(p=>p.lapsed) || workingPublication() ? 'No current review is published for this book; nothing is decided' : 'No review is published for this book; nothing is decided.'))}${publications().some(p=>p.lapsed) || workingPublication() ? html`<p class="caption">${t('A review sealed under an earlier binding reads back as recorded; the book can be reviewed again.')}</p>` : ''}`,'','data-report-section="decision"') : panel(t('Decision'),'',html`${actions.length ? html`<div class="card-list lines decision-actions">${actions}</div>` : ''}${kv([[t('Route'),v.disposition ? codeWords(v.disposition) : '\u2014'],[t('Completeness'),v.review_state ? codeWords(v.review_state) : '\u2014'],...(x?.review?.recommendation?.action_activation ? [[t('Activation'),codeWords(x.review.recommendation.action_activation)]] : []),[t('Reviewer'),attributionWords(v.review_attribution || [],{facts:true}) || '\u2014'],[t('Policy'),v.policy_version ? codeCell(v.policy_version) : '\u2014'],[t('Rules'),(v.rule_ids || []).length ? html`<span class="mono">${v.rule_ids.join(', ')}</span>` : '\u2014'],[t('Reasons'),(v.reasons || []).length ? html`<span class="owner-text">${v.reasons.map((r,i)=>html`${i ? ' · ' : ''}${reasonWords(r)}`)}</span>` : '\u2014']].filter(([,x])=>x!=='\u2014'),'kv-columns')}${limits.length ? html`<details class="reveal-details"><summary>${t('Limits')} <span class="num">${count(limits.length)}</span></summary>${textCollection('report-limits',limits,said)}</details>` : ''}`,'','data-report-section="decision"');
    const questions=questionsSection(), changes=changesSection();
    // B1 (item 8): one column, the conclusion first -- the decision, then what it covered, issuer by issuer (the folds are the outline), what changed, what stays open, the delivery and the publications; the clock is the head's and Facts'
    return html`${decision}${scope}${issuers}${changes}${questions}${withDelivery ? panel(t('Delivery'),'',deliverySection(),'','data-report-section="delivery"') : ''}${publicationsSection()}`;
  }
  /* An authored book's delivery: the composed facts, or the way to compose; an installed result's review is the owner's export. */
  function deliverySection() {
    const d=deliveryOf();
    if(isUpdateBook() && d){
      const p=d.sections?.positions?.value || {},schedule=p.schedule || {},date=schedule.entry_session;
      const rows=p.position_rows?.length ? collectionTable('delivery-positions',p.position_rows,[{label:t('Listing'),type:'text',absorb:true},{label:t('Weight'),unit:'%',type:'num'},{label:t('Change'),unit:'bps',type:'num'}],r=>[r.name || r.listing_id,pctText(r.weight),Number.isFinite(Number(r.change)) ? signed(r.change,'ratio',2) : t(r.change || '')]) : '';
      const facts=factsRef(t('Facts'),kv([[t('Formation session'),schedule.formation_session || ''],[t('Entry session'),date || ''],[t('Holding end'),schedule.holding_end_session || ''],[t('Position basis'),codeWords(p.basis)],[t('Update publication'),hashCell(d.selection?.update_publication_hash)],[t('Review section'),t(SECTION_WORDS[d.sections?.evidence_cro?.status] || '')],[t('Export hash'),hashCell(d.export_hash)]]));
      return html`${stateLine('recorded',{word:codeWords(d.status),next:''})}${panel(date ? t('Positions for {date}',{date}) : t('Positions'),'',html`${p.position_rows?.[0]?.basis ? html`<p class="caption">${said(p.position_rows[0].basis)}</p>` : ''}${rows}${facts}`,'','data-report-section="positions"')}${panel(t('Commentary'),'',html`${(d.commentary || []).map(c=>html`<article><h3>${c.attribution_word ? t(c.attribution_word,Object.fromEntries(Object.entries(c.attribution_words || {}).map(([k,v])=>[k,declaredCodeWord(v) ? codeWords(v) : v]))) : c.attribution}</h3><p class="owner-text">${c.question ?? c.text}</p>${c.question != null ? html`<p class="caption">${t('The person’s answer, relayed by the PM')}</p><p class="owner-text">${c.answer ?? t('Not yet given')}</p>` : ''}</article>`)}${factsRef(t('Facts'),kv([[t('Attribution'),codeWords(d.commentary_provenance?.attribution)]]))}`,'','data-report-section="commentary"')}<p class="caption">${said(d.claim)}</p>`;
    }
    if(!isExperimentBook()) return html`<p>${t('This book is an installed result; its review is exported through the review owner (the ··· menu). A joined delivery is composed for an authored Portfolio book from the Lab.')}</p>`;
    if(!d) return emptyState(html`${t('No delivery composed yet')}${infoMark(t('No delivery composed yet; composing joins this exact book with its sources and the sections you select.'))}`,isExperimentBook() ? '' : typedBtn(t('Compose the delivery'),'review-deliver','','button primary',ready() ? '' : t('Busy'))); // round 90: the head's primary is the one action; round 93: one sentence
    const sections=d.sections || {}, status=d.summary?.section_status || {};
    return html`${stateLine('recorded',{word:codeWords(d.status),next:''})}${kv([[t('Question'),html`${d.question || ''} <span class="sub-cell">${codeWords(d.question_status)}</span>`],[t('Input'),html`${d.input?.research_input_id} · ${mono(d.input?.input_binding_hash,SHORT.hash)}`],[t('Sections'),html`${Object.entries(status).map(([k,w],i)=>html`${i ? ' · ' : ''}${k}: ${t(SECTION_WORDS[w] || w || '')}`)}`],[t('Portfolio result'),html`${t('annualized return')} ${pct(d.summary?.portfolio_result?.annualized_return)} · ${t('Sharpe')} ${fmt(d.summary?.portfolio_result?.sharpe)}`],[t('Review'),d.summary?.review ? html`${mono(d.summary.review.publication_hash,SHORT.hash)} · ${codeWords(d.summary.review.route)} · ${codeWords(d.summary.review.completeness)}` : t('absent')],[t('Risk'),d.summary?.risk_reference ? html`${mono(d.summary.risk_reference.link_hash,SHORT.hash)} · ${d.summary.risk_reference.selected_session_covered ? t('covers the holdings date') : t('does not cover the holdings date')}` : t('absent')],[t('Comparison'),sections.comparison?.status==='INCOMPATIBLE' ? html`${t('refused')} · <span class="mono">${sections.comparison.failure_code}</span>` : d.summary?.comparison_tasks ? html`${mono(d.summary.comparison_tasks.left,SHORT.id)} ↔ ${mono(d.summary.comparison_tasks.right,SHORT.id)}` : t('absent')],[t('Commentary'),html`${(d.commentary || []).length} · ${codeWords(d.commentary_provenance?.attribution)}`]])}<p class="caption">${said(d.claim)}</p><div class="flow">${btn(t('Compose again'),'review-deliver','','button compact')}${btn(t('Reopen this exact delivery'),'review-delivery-reopen','','button compact')}</div>`;
  }
  /* The composition dialog (round 78): the book's facts, the optional sections, the question. */
  function deliveryDialog() {
    if(!isExperimentBook() || !S.selector) return;
    const books=Data.history().filter(v=>v.kind==='portfolio.policy-development' && v.raw?.status==='SUCCEEDED' && v.raw?.task_id && v.raw.task_id!==S.selector.experiment_task_id);
    const review=S.pin || workingPublication();
    const refusedLinks=S.riskLinks===null ? [] : S.riskLinkRefusals || [];
    openDialog(t('Delivery'),t('Compose the delivery'),html`${kv([[t('Book'),html`${mono(S.selector.experiment_task_id,SHORT.id)} <span class="sub-cell">${t('receipt')} ${mono(S.selector.experiment_receipt_hash,SHORT.hash)}</span>`],[t('Holdings date'),S.selector.portfolio_session],[t('Review section'),review ? html`${t('the review publication')} <span class="sub-cell">${mono(review,SHORT.hash)}</span>` : t('absent: no review publication exists for this book')]])}
      ${refusedLinks.map(r=>refusal(r,'warning',{catalog:true,more:html`<p>${t('Risk link')} ${hashCell(r.link_hash)}</p>`,next:prerequisiteWays(r.next_requests)}))}<div class="field"><label id="reviewRiskLabel" for="reviewRisk">${t('Linked Risk report (optional)')}</label>${picker('reviewRisk',[['',S.riskLinks===null ? t('Not read yet') : S.riskLinks.length ? t('None (absent)') : refusedLinks.length ? t('No readable Risk link') : t('None linked to this book (absent)')],...(S.riskLinks || []).map(l=>[l.link_hash,`${short(l.risk_task_id)} · ${short(l.link_hash, SHORT.hash)}`])],{selected:S.risk,labelId:'reviewRiskLabel'})}<small>${t('Only a report-only link the owner recorded for this exact book can be selected; nothing is attached here.')}</small></div>
      <div class="field"><label id="reviewComparisonLabel" for="reviewComparison">${t('Saved comparison book (optional)')}</label>${picker('reviewComparison',[['',t('None (absent)')],...books.map(v=>[v.raw.task_id,`${short(v.raw.task_id)} · ${v.raw.book?.portfolio_session || ''}`])],{selected:S.comparison,labelId:'reviewComparisonLabel'})}<small>${t('The owner compares both books at this holdings date and refuses an incompatible pair with its reason.')}</small></div>
      <div class="field"><label for="reviewQuestion">${t('Delivery question (optional)')}</label><input id="reviewQuestion" class="ui-field" value="${S.question}" placeholder="${t('The question this delivery answers; recorded as provided for this delivery, not as the original experiment intent')}"></div>
      <p class="caption">${t('An unselected or unavailable section is absent, never zero and never approval. Composing reads the owner\'s export; nothing runs.')}</p>`,
      html`${btn(t('Read linked Risk reports'),'review-risk-options','','button compact')}${typedBtn(t('Compose the delivery (read-only export)'),'review-delivery','','button primary',ready() ? '' : t('Busy'))}`);
  }
  function reportPage() {
    if(isUpdateBook()) return html`<div class="overview-main report-document">${deliveryOf() ? html`${deliverySection()}${reportOf() ? reportDocument(false) : ''}` : S.error ? html`${S.deliveryRefusal ? refusal(S.deliveryRefusal,'warning',{next:prerequisiteWays(S.deliveryRefusal.next_requests)}) : notRead(t('Export not read'),S.error)}${btn(t('Read again'),'review-delivery','','button compact')}` : skeleton('body')}</div>`;
    const x=reportOf(), hash=S.pin || workingPublication();
    const notice=!hash ? (S.view?.state==='REVIEW_PUBLISHED' && !historical() ? noteLine(t('Working publication not named by the owner'),html`${t('The projection reports a published review but names no publication identity; nothing is inferred from history.')}<br>${t('Pin one of the recorded publications above to read its exact export; nothing is chosen for you.')}`,'neutral') : noteLine(t('No review publication for this book'),t(isExperimentBook() ? 'The delivery below can still be composed; its review section is absent, which is not clearance.' : 'Nothing is exported until a review is published for this book.')))
      : !x ? (S.error && S.busy!=='export' ? notRead(t('Export not read'),S.error,'',btn(t('Read again'),'review-refresh','','button compact')) : skeleton('body')) : ''; // B1: the readback's exactness is Scope's line, its words the (i)
    return html`${notice}<div class="overview-main report-document">${reportDocument()}</div>`;
  }
  /* The report's provenance per section (Facts, round 78). */
  function reportFacts() {
    const v=S.view, x=reportOf(), d=deliveryOf(), out=[];
    if(x || v) out.push({title:t('Scope'),body:kv([[t('Report hash'),mono(x?.report_hash,SHORT.hash)],[t('Publication'),mono(x?.review?.publication?.publication_hash || S.pin || workingPublication(),SHORT.hash)],[t('Export hash'),mono(x?.export_hash,SHORT.hash)],[t('Book authority'),codeWords(x?.book_authority || v?.book?.authority)],[t('Evidence selection'),v?.evidence_selection ? codeWords(v.evidence_selection) : ''],[t('Verified excerpts'),count((x?.evidence?.verified_spans || []).length)]])});
    if(x) out.push({title:t('Times'),body:reviewTimes()}); // B1: sealed clock
    if(v) out.push({title:t('Decision'),body:kv([[t('Reviewer'),attributionWords(v.review_attribution || [],{facts:true}) || ''],[t('Policy'),v.policy_version ? codeCell(v.policy_version) : ''],[t('Rules'),(v.rule_ids || []).join(', ') || ''],[t('Reasons'),(v.reasons || []).length ? html`${v.reasons.map((r,i)=>html`${i ? ' · ' : ''}${reasonWords(r)}`)}` : '']])});
    if(d) out.push({title:t('Delivery'),body:html`${kv([[t('Input binding'),mono(d.input?.input_binding_hash,SHORT.hash)],[t('Export hash'),mono(d.export_hash,SHORT.hash)],...(d.question_status ? [[t('Question provenance'),codeWords(d.question_status)]] : [])])}${codeRef(t('Read the exact reopen request'), d.next_requests?.reopen ?? null)}`});
    return [...out,...teamFacts()];
  }
  /* ---- the page ---- */
  /* No book chosen and none resolved by the owner: the way in, not the empty desk. The owner's
   * own sentence stays readable; the workbench names its own routes to a book. */
  /* ---- the spine (round F1, redesign-2 sections 4-5): the book's evidence cycle as five stages,
   * one sentence of state and one next lawful action, derived once from what the page holds --
   * the projection, the preview, the packets named, the dossier read, the bound Tasks -- and read
   * by every page's head, rail and checks. Nothing here composes a request: the primary is the
   * owner's `next_requests` entry consumed whole, or a page. ---- */
  const STAGES=[['sources','Sources','evidence-stream'],['prepared','Prepared','evidence-reading'],['analysed','Analysed','handoff'],['reviewed','Reviewed','handoff'],['reported','Reported','report']];
  const dayWord=(x)=>{ const s=String(x || ''); return /\d[T ]\d/.test(s) ? dayOf(s) : s.slice(0,10); }; // N3 (law 133): the day of an instant is the reader's day; a date stays a date
  /* A book's next step by its state (N4): the word and the page it is taken on -- the Overview's
   * primary and the Books row say the same (law 58: the way on, named once, the same everywhere). */
  const BOOK_NEXT={EVIDENCE_AUTHORITY_NOT_ADMITTED:['Set up evidence','evidence-stream'],AWAITING_ALTERNATIVE_EVIDENCE:['Prepare sources','evidence-stream'],EVIDENCE_REFRESH_IN_PROGRESS:['Follow the run','evidence'],ANALYST_PACKET_PREPARED:['Answer as the Analyst','handoff'],ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW:['Assess as the CRO','handoff'],REVIEW_PUBLISHED:['Read the report','report'],ALTERNATIVE_EVIDENCE_EXPIRED:['Prepare again','evidence-stream'],ALTERNATIVE_EVIDENCE_SUPERSEDED:['Prepare again','evidence-stream'],EVIDENCE_SELECTION_AMBIGUOUS:['Choose the analysis','evidence']};
  const publishedAt=()=>{ const h=S.pin || workingPublication(); const row=h ? publications().find(p=>p.hash===h) : null; return row ? dayWord(row.recorded) : ''; };
  const stepTo=(word,page)=>({word:t(word),action:'review-step',value:page});
  const nextOrStep=(word,key,page)=>S.view?.next_requests?.[key]?.operation ? {word:t(word),action:'review-next',value:key} : stepTo(word,page);
  const BAND_WORDS={LOW:'Low',MEDIUM:'Medium',HIGH:'High',CRITICAL:'Critical'};
  const bandWords=(x)=>BAND_WORDS[x] ? t(BAND_WORDS[x]) : codeWords(x); // W: under an Exposure label the band is its one word ("critical"), not "critical exposure" again
  const reasonWords=(x)=>{ const v=String(x || ''); return /^[A-Z][A-Z_]+$/.test(v) ? codeWords(v) : /^[a-z][a-z0-9_.]+$/.test(v) ? codeWords(v.toUpperCase().replace(/\./g,'_')) : html`<span class="owner-text">${said(v)}</span>`; }; // generated sentences have exact templates; authored sentences keep their words
  function requiredPrimary(a) {
    const kind=String(a?.action || '');
    if(['SETTLE_EVIDENCE_CONTINUATION','ANALYZE_CONTINUED_EVIDENCE','READ_CRO_DOSSIER','SUBMIT_CRO_ASSESSMENT'].includes(kind)) return actionWay(a);
    if(kind==='REFRESH_EVIDENCE') return nextOrStep('Prepare again','preview','evidence-stream');
    if(kind==='RESOLVE_ISSUER_MAPPING') return {word:t('Resolve the mapping'),action:'go',value:'portfolio'};
    if(kind==='HUMAN_REVIEW') return stepTo('Human review','handoff');
    return stepTo('Read the report','report');
  }
  function cycle() {
    const v=S.view, st=v?.state || '', blocked=blockedState(), pinned=historical(), inv=inventory(), cp=v?.coverage_progress || null;
    const actions=requiredActions(), blocking=actions.find(a=>a.blocking) || actions[0] || null;
    let sentence='', tone='neutral', primary=null;
    if(S.refusal){ sentence=t('Evidence review is not available for this workspace.'); tone='warning'; }
    else if(pinned){ sentence=t('Pinned: the review published {d}, read exactly as it was sealed.',{d:publishedAt() || ''}); tone='historical'; primary=stepTo('Read the report','report'); }
    else if(v) switch(st){
      case 'NO_BOOK_TO_REVIEW': sentence=t('No book is chosen.'); break;
      case 'REVIEW_INPUT_INCOMPLETE': sentence=t('This book cannot be reviewed yet.'); tone='warning'; break;
      case 'EVIDENCE_AUTHORITY_NOT_ADMITTED': sentence=t('Evidence is not set up for this workspace.'); tone='warning'; primary=stepTo(...BOOK_NEXT[st]); break;
      case 'AWAITING_ALTERNATIVE_EVIDENCE': sentence=continuationRequired() ? t('The selected analysis needs its evidence continuation before CRO.') : t('No evidence is prepared for this book.'); primary=continuationRequired() ? requiredPrimary(blocking) : nextOrStep('Prepare sources','prepare','evidence-stream'); break;
      case 'EVIDENCE_REFRESH_IN_PROGRESS': sentence=cp && cp.units_total>1 ? t('Preparing: {k} of {n} groups done.',{k:count(cp.units_prepared || 0),n:count(cp.units_total)}) : t('Preparing the evidence.'); tone='accent'; primary=v.task_id ? {word:t('Follow the run'),action:'review-watch',value:v.task_id} : null; break;
      case 'ANALYST_PACKET_PREPARED': sentence=t('Evidence prepared; awaiting the Analyst\'s answer.'); primary=stepTo(...BOOK_NEXT[st]); break;
      case 'ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW': sentence=t('Analysis published as of {d}; awaiting the CRO\'s assessment.',{d:dayWord(v.evidence_as_of) || ''}); primary=actions.length ? requiredPrimary(blocking) : stepTo(...BOOK_NEXT[st]); break;
      case 'REVIEW_PUBLISHED': sentence=t('Reviewed · {n}.',{n:actions.length ? countText(actions.length,'{n} required action','{n} required actions') : t('no required action')}); tone='good'; primary=blocking ? requiredPrimary(blocking) : stepTo('Read the report','report'); break;
      case 'ALTERNATIVE_EVIDENCE_EXPIRED': sentence=t('The evidence expired {d}; a published review stays readable by its handle.',{d:dayWord(v.evidence_expires_at) || ''}); tone='warning'; primary=nextOrStep('Prepare again','preview','evidence-stream'); break;
      case 'ALTERNATIVE_EVIDENCE_SUPERSEDED': sentence=t('The preparation\'s policy moved; the evidence is superseded.'); tone='warning'; primary=nextOrStep('Prepare again','preview','evidence-stream'); break;
      case 'EVIDENCE_SELECTION_AMBIGUOUS': sentence=t('{n} analyses are current; choose the one to review against.',{n:count((v.eligible_versions || []).length)}); tone='warning'; primary={word:t('Choose the analysis'),action:'review-live-item',value:'analysis'}; break;
      default: sentence=st ? t(stateOf(st).word) : '';
    }
    // the five stages: the words from the bound Tasks' currency (stepRows: a current Task is progress, an earlier version's is history, an unsettled one is open), the inventory, the ledgers, the dossier, the export
    const rail=Object.fromEntries((v && !blocked ? stepRows() : []).map(([id,[tn,text]])=>[id,{tone:tn,text}]));
    const word=(x,done)=>x==='done' ? done : x==='running' ? 'running' : x==='waiting' ? 'awaiting' : x==='refused' ? 'refused' : 'pending';
    const unavailable=t('not available: no evidence authority is admitted for this workspace');
    const check=S.preview?.source_check || null;
    const sources=blocked ? {word:'unavailable',line:unavailable}
      : inv ? {word:inv.without.length ? 'no_source' : 'held',time:check?.checked_at ? dayWord(check.checked_at) : '',line:t('{w} of {n} issuers hold a document',{w:count(inv.withSource==null ? inv.rows.filter(r=>r.held>0).length : inv.withSource),n:count(inv.rows.length)})+' · '+t(inv.live ? 'live sources' : 'recorded package')}
      : v?.evidence_as_of && !['AWAITING_ALTERNATIVE_EVIDENCE','REVIEW_INPUT_INCOMPLETE'].includes(st) ? {word:'held',time:dayWord(v.evidence_as_of),line:t('held at the preparation; the inventory is read on the Sources page')} // the projection's own fact (law 95): a prepared book had its sources
      : {word:'pending',line:t('the sources are not read yet')};
    const cells=cellsHere(), named=packetsNamed(), byState=(code)=>cells.filter(c=>c.delivery_state===code).length;
    const cont=readyLedgers().some(l=>l.scope) ? ' · '+t('continuation')+' '+[...new Set(readyLedgers().map(l=>contWords(l.scope?.state)))].filter(Boolean).join(', ') : '';
    const cellsLine=cells.length ? t('{a} complete · {b} partial · {c} pending · {d} no route · {e} no source',{a:count(byState('COMPLETE')),b:count(byState('PARTIAL')),c:count(byState('PENDING')),d:count(byState('NO_ROUTE')),e:count(byState('NO_SOURCE'))})+cont : '';
    const groups=cp && cp.units_total>1 ? ' · '+t('{k} of {n} groups prepared',{k:count(cp.units_prepared || 0),n:count(cp.units_total)}) : '';
    const prepared=blocked ? {word:'unavailable',line:unavailable}
      : st==='EVIDENCE_REFRESH_IN_PROGRESS' ? {word:'running',line:rail.sources?.text || t('preparing')}
      : st==='ALTERNATIVE_EVIDENCE_EXPIRED' ? {word:'expired',time:dayWord(v.evidence_expires_at),line:rail.sources?.text || ''}
      : st==='EVIDENCE_SELECTION_AMBIGUOUS' && v?.evidence_as_of ? {word:'prepared',time:dayWord(v.evidence_as_of),line:t('prepared · more than one analysis is current')} // the packets are prepared; the choice belongs to the Analysed stage
      : st==='ALTERNATIVE_EVIDENCE_SUPERSEDED' ? {word:'superseded',line:t('the preparation\'s policy moved')}
      : st==='ANALYST_PACKET_PREPARED' || (named.length && rail.sources?.tone!=='done' && rail.sources?.tone!=='running') ? {word:'prepared',time:v?.evidence_as_of ? dayWord(v.evidence_as_of) : '',line:(named.length ? t('prepared · {n}',{n:countText(named.length,'{n} packet named by the owner','{n} packets named by the owner')}) : t('prepared · as the owner\'s state records'))+(cellsLine ? ' · '+cellsLine : '')+groups} // the owner's state says prepared: the packets it names are the evidence, whether or not this page has read one
      : rail.sources ? {word:word(rail.sources.tone,'prepared'),time:v?.evidence_as_of && rail.sources.tone==='done' ? dayWord(v.evidence_as_of) : '',line:rail.sources.text+(cellsLine ? ' · '+cellsLine : !named.length ? ' · '+t('no prepared packet is named for this book') : '')+groups}
      : {word:'pending',line:t('nothing prepared')};
    if(rail.sources && !cells.length && named.length && ledgersHere().some(l=>l.status==='refused')) prepared.line+=' · '+t('no ledger: the preparation did not run the integrated selection');
    const d=dossierOf(), states=(d?.issuers || []).map(i=>String(i.review_state || '')).filter(Boolean);
    const tally=states.length ? [...new Set(states)].map(x=>count(states.filter(y=>y===x).length)+' '+t(stateOf(REVIEW_STATE_KEY[x] || x.toLowerCase()).word)).join(' · ') : '';
    const analysed=blocked ? {word:'unavailable',line:unavailable}
      : st==='EVIDENCE_SELECTION_AMBIGUOUS' ? {word:'ambiguous',line:rail.analyst?.text || ''}
      : st==='ANALYST_PACKET_PREPARED' && !['refused','waiting','running'].includes(rail.analyst?.tone) ? {word:'awaiting',line:t('awaiting the Analyst\'s answer')+(rail.analyst?.tone==='done' ? ' · '+rail.analyst.text : '')}
      : rail.analyst ? {word:word(rail.analyst.tone,'published'),time:analysisPublished() && v?.evidence_as_of ? dayWord(v.evidence_as_of) : '',line:rail.analyst.text+(rail.validation?.tone==='running' ? ' · '+rail.validation.text : '')+(tally ? ' · '+tally : '')+((d?.unresolved_questions || []).length ? ' · '+countText(d.unresolved_questions.length,'{n} unresolved question','{n} unresolved questions') : '')}
      : {word:'pending',line:t('no analysis')};
    const unreviewed=cp ? (cp.units_total || 0)-(cp.units_reviewed || 0) : 0;
    const reviewed=blocked ? {word:'unavailable',line:t('no review published for this book')}
      : rail.published?.tone==='done' ? {word:'review_published',time:publishedAt(),line:rail.published.text+(actions.length ? ' · '+countText(actions.length,'{n} required action','{n} required actions') : '')+(rail.cro?.text && ['refused','waiting','running'].includes(rail.cro.tone) ? ' · '+rail.cro.text : '')+(unreviewed ? ' · '+countText(unreviewed,'{n} group not reviewed','{n} groups not reviewed') : '')}
      : rail.cro ? {word:word(rail.cro.tone,'review_published'),line:rail.cro.text}
      : {word:'pending',line:t('no review published for this book')};
    const reported=S.delivery ? {word:'delivered',line:t('delivery composed')} : (reviewPublished() || pinned) && !blocked ? {word:'available',line:t('the report is ready to read and export')} : {word:'pending',line:t('after a review is published')};
    const by={sources,prepared,analysed,reviewed,reported};
    const stages=v || S.refusal ? STAGES.map(([id,name,page])=>({id,name,page:blocked ? null : page,...by[id]})) : [];
    // the policy's own requirements as checks (GitHub's list): the mapping, the reviewed weight against the floor, the Analyst's checks per issuer, a human review the policy routes to
    const checks=[];
    if(v && !blocked && (analysisPublished() || reviewPublished() || pinned || named.length)){
      const sc=v.scope_coverage, cv=v.coverage;
      if(sc){
        const mapped=Math.max(0,(sc.book_positions || 0)-(sc.unmapped_positions || 0));
        checks.push({id:'mapping',state:sc.unmapped_positions ? 'partial' : 'verified',name:t('Mapping'),why:t('{m} of {n} positions mapped to an issuer',{m:count(mapped),n:count(sc.book_positions || 0)})+(sc.unmapped_positions ? ' · '+countText(sc.unmapped_positions,'{n} unmapped','{n} unmapped') : '')});
        const reviewed=cv && cv.reviewed_ending_weight!=null ? cv.reviewed_ending_weight : sc.whole_book_reviewed_weight, accounted=cv?.accounted_ending_weight, w=share(accounted), m=share(sc.minimum_required_weight); // compare only owner-accounted material plus official quiet coverage
        checks.push({id:'floor',state:w!==null && m!==null && w>=m ? 'verified' : w!==null && sc.attainable_below_minimum ? 'blocked' : 'partial',name:t('Accounted weight'),why:w===null ? t('Whole-book coverage accounting was not recorded; the floor comparison is unavailable.') : t('{w} of the book\'s weight accounted',{w:pctText(accounted)})+' · '+t('{m} required',{m:pctText(sc.minimum_required_weight)})+(sc.attainable_below_minimum ? ' · '+t('not attainable with this issuer scope') : '')});
      } else if(cv){
        checks.push({id:'mapping',state:'verified',name:t('Mapping'),why:t('{m} of the positions mapped · {s} of the issuers selected',{m:pctText(cv.mapping),s:pctText(cv.selected_issuers)})});
        checks.push({id:'floor',state:cv.accounted_ending_weight!=null ? 'verified' : 'partial',name:t('Accounted weight'),why:cv.accounted_ending_weight==null ? t('Whole-book coverage accounting was not recorded; the floor comparison is unavailable.') : t('{w} of the book accounted · {c} of the change · as this publication sealed it',{w:pctText(cv.accounted_ending_weight),c:pctText(cv.reviewed_absolute_change)})});
      }
      if(d){ const bad=states.some(x=>!['EXECUTED_WITH_FINDINGS','EXECUTED_NO_FINDINGS'].includes(x)); checks.push({id:'analyst',state:bad ? 'partial' : 'verified',name:t('Analyst checks'),why:tally || t('no issuer state recorded'),to:{action:'review-step',value:'handoff'}}); if(d.analyst_requires_human_review) checks.push({id:'human',state:'awaiting',name:t('Human review'),why:t('required by the Analyst')}); }
      else if(analysisPublished()) checks.push({id:'analyst',state:'available',name:t('Analyst checks'),why:t('read with the dossier on the Review page'),to:{action:'review-step',value:'handoff'}});
      if(v.disposition==='HUMAN_REVIEW_REQUIRED' && !checks.some(k=>k.id==='human')) checks.push({id:'human',state:'awaiting',name:t('Human review'),why:t('the policy routes this review to a human')});
    }
    return {state:st,sentence,tone,primary,stages,checks};
  }
  /* The rail: the five stages as one row under every page's head, the page's own stage marked;
   * a stage is a link to its page (none while the authority is not admitted). */
  /* The checks: the five stages with their lines, then the policy's requirements; every row
   * read from its owner (law 17), none merged into one green word. */
  /* The measures (round H1; Mercury's cards, GitHub's run summary): the cycle's figures as stat
   * tiles, never as sentences -- once a review is published the reviewed weight against the
   * route's floor (a meter with the mark), the mapping, the issuers, the findings, the issues,
   * the publications; before it the preparation's own figures. Words stay in the labels. */
  const pctOf=(a,b)=>Number(b) ? pctText(((Number(a) || 0)/Number(b)*100).toFixed(3)+'%') : '';
  /* The preparation while its run continues (contract 10.10): the filings fetched, the documents read,
   * the passages embedded and the groups selected, each its owner's count toward its own total --
   * the book stage's count in the work area (the data scenes' rendering), a group's own stage in the
   * Units table. Counts, never a percentage across stages nor a time; telemetry of the Host, gone when
   * the run ends or the Host restarts. */
  const PREPARING=[['acquire_source_evidence','Fetch filings'],['canonicalize_documents','Read documents'],['build_retrieval_generation','Embed passages'],['select_evidence_spans','Select passages']];
  const PREPARING_UNITS={filings:'filings',documents:'documents',chunks:'passages',units:'groups'};
  const preparingWord=(stage)=>t((PREPARING.find(([id])=>id===stage) || [stage,stage])[1]);
  // a running stage that has not yet learned its denominator has no count to show: its word alone (ST7)
  const workOf=(w)=>w.running && !(Number(w.total)>0) ? {count:'',bar:null} : workCount(w.completed,w.total,t(PREPARING_UNITS[w.unit_name] || w.unit_name));
  function overviewStrip() {
    const v=S.view; if(!v || S.refusal || blockedState() || reviewPublished() || historical()) return ''; // B1: once a review is published the coverage figure is the one drawing (its words carry the mapping)
    const cp=v.coverage_progress, pubs=publications(), inv=inventory();
    if(!cp) return '';
    const pw=share(cp.prepared_weight), tiles=[];
    tiles.push(stat(t('Prepared weight'),pctText(cp.prepared_weight),'',false,'',pw!==null ? coverageBar([{n:pw,tone:'ok'}],{total:1,words:'',label:t('Prepared weight')}) : ''));
    tiles.push(stat(t('Mapping'),pctOf(cp.mapped_issuers,cp.book_listings),''));
    tiles.push(stat(t('Issuers'),count(cp.issuers_prepared),Number(cp.issuers_carried) ? t('{n} carried · {w} of the weight',{n:count(cp.issuers_carried),w:pctText(cp.carried_weight)}) : '')); // carried: no new filing in the window, reviewed by what earlier readings found (10.8)
    tiles.push(stat(t('Groups'),html`${count(cp.units_prepared)} ${unit('/ '+count(cp.units_total))}`));
    if(inv) tiles.push(stat(t('Documents'),count(inv.rows.reduce((a,r)=>a+r.held,0)),t(inv.live ? 'live sources' : 'recorded package')));
    tiles.push(stat(t('Publications'),count(pubs.length),''));
    return measureStrip(tiles,t('Measures')); // the box the strip measures itself by (six or three across)
  }
  /* The issuers the review did not reach, by the owner's record (B1; the book plan's items 2, 3 and
   * 7): a coverage line `<issuer> is not reviewed: <CODE>`, else a conclusion that says no review
   * (an evidence gap, not evaluated); the reason is that line's code, else the conclusion. */
  const NOT_REVIEWED=new Set(['EVIDENCE_GAP','NOT_EVALUATED']);
  const unreviewedReason=(r)=>{ const line=(S.view?.gaps || []).map(String).find(x=>x.startsWith(r.entity_id+' is not reviewed: ')); return line ? line.slice(line.indexOf(': ')+2).trim() : NOT_REVIEWED.has(String(r.conclusion)) ? String(r.conclusion) : ''; };
  const issuerReviewed=(r)=>!unreviewedReason(r);
  const issuerName=(r)=>entityNames(r.tickers) || r.entity_id;
  const byWeight=(a,b)=>(share(b.ending_weight) || 0)-(share(a.ending_weight) || 0);
  // the book's one order of its issuers (the reviews, 2026-09-24: the figure and the table disagreed): the coverage figure's -- what the review reached first, then what it did not, each largest first
  const issuerOrder=(rows)=>[...rows.filter(issuerReviewed).sort(byWeight),...rows.filter(r=>!issuerReviewed(r)).sort(byWeight)];
  // what the review says of an issuer it did not reach: its conclusion, and the recorded reason where it is another word
  const heldWords=(r)=>{ const why=unreviewedReason(r); return [codeWords(r.conclusion),why && why!==String(r.conclusion) ? codeWords(why) : ''].filter(Boolean).join(' · '); };
  /* Why the book is held, drawn (B1; the book plan's items 2 and 7, law 89): one gauge across the
   * lane -- the reviewed share of the book's ending weight filled, the rest the track, the route's
   * floor a labelled mark, the whole book 0 % to 100 % at its ends, the shortfall in words. It
   * measures one thing at any number of issuers (a review, 2026-09-24: an issuer named under each
   * part smeared at 35, and their order read as if some "reached" the floor): its key is the two
   * groups -- reviewed, and not reached with the recorded reason -- each its count and share and the
   * Issuers table's filter; the issuers are the table's. Every figure is the owner's; the rest of the
   * book is its complement. */
  function coverageFigure() {
    const v=S.view; if(!v || S.refusal || blockedState() || !(reviewPublished() || historical())) return '';
    const rows=v.issuer_rows || [], sc=v.scope_coverage, cv=v.coverage;
    const reviewed=cv && cv.reviewed_ending_weight!=null ? cv.reviewed_ending_weight : sc?.whole_book_reviewed_weight, read=share(reviewed), accounted=share(cv?.accounted_ending_weight), quiet=share(cv?.nothing_filed_ending_weight), unreached=share(cv?.unreached_ending_weight), floor=sc ? share(sc.minimum_required_weight) : null;
    if(read===null && accounted===null) return '';
    const done=rows.filter(issuerReviewed), held=rows.filter(r=>!issuerReviewed(r));
    const mapping=sc ? pctOf(Math.max(0,(sc.book_positions || 0)-(sc.unmapped_positions || 0)),sc.book_positions) : cv?.mapping ? pctText(cv.mapping) : '';
    const points=accounted===null || floor===null ? null : Math.round((floor-accounted)*10000)/100;
    const words=html`${read===null ? '' : html`<b class="num">${pctText(reviewed,sc?.minimum_required_weight)}</b> ${t('reviewed material')}`}${quiet===null ? '' : html` · <b class="num">${pctText(cv.nothing_filed_ending_weight)}</b> ${t('officially nothing filed')}`}${unreached===null ? '' : html` · <b class="num">${pctText(cv.unreached_ending_weight)}</b> ${t('unreached')}`}${accounted===null ? html` · ${t('Whole-book coverage accounting was not recorded; the floor comparison is unavailable.')}` : html` · <b class="num">${pctText(cv.accounted_ending_weight)}</b> ${t('accounted weight')}${points===null ? '' : html` · ${points>0 ? t('{d} points under the floor',{d:num(points)}) : t('at or above the floor')}`}`}${sc?.attainable_below_minimum && accounted!==null ? html` · ${t('not attainable with this issuer scope')}` : ''}${mapping ? html` · ${t('mapping {m}',{m:mapping})}` : ''}`;
    // the key: two groups, each its issuers' count and share of the book and the table's filter; what the review did not reach says its recorded reasons, each once
    const reasons=[...new Set(held.map(heldWords).filter(Boolean))];
    const item=(state,on,n,weight,words)=>btnAttrs(html`<i${on ? ' class="on"' : ''} aria-hidden="true"></i><span>${words}</span><span class="muted">${countText(n,'{n} issuer','{n} issuers')} · ${weight}</span>`,'review-issuer-state',S.issuerState===state ? '' : state,'coverage-key-item',html`aria-pressed="${S.issuerState===state}" data-tip="${t(S.issuerState===state ? 'Show every issuer in the table' : 'Show these issuers in the table')}"`);
    const key=cv?.accounted_ending_weight!=null ? html`<div class="coverage-key" role="group" aria-label="${t('Coverage')}">${done.length ? item('reviewed',true,done.length,pctText(cv.accounted_ending_weight),t('Accounted weight')) : ''}${held.length ? item('held',false,held.length,pctText(cv.unreached_ending_weight),t('Unreached')) : ''}${cv?.nothing_filed_window_days!=null ? html`<span>${t('Quiet window')}: ${count(cv.nothing_filed_window_days)} ${t('days')}</span>` : ''}</div>` : !rows.length ? '' : html`<div class="coverage-key" role="group" aria-label="${t('Coverage')}">${done.length ? item('reviewed',true,done.length,pctText(reviewed),t('Reviewed')) : ''}${held.length ? item('held',false,held.length,'',html`${t('Not reviewed')}${reasons.length ? html` <span class="muted">· ${reasons.join(' · ')}</span>` : ''}`) : ''}</div>`;
    const segments=accounted===null ? [] : [{n:accounted,tone:'ok'}];
    const bar=accounted===null ? '' : meter({kind:'share',segments,total:1,mark:floor,markLabel:floor===null ? '' : t('floor {m}',{m:pctText(sc.minimum_required_weight)}),ends:['0%','100%'],words:'',label:t('Coverage'),cls:'coverage-meter'});
    return figureBox(t('Coverage'),html`<p class="coverage-words">${words}</p>${bar}${key}`,{info:infoMark(t('The book\'s ending weight by what the review reached, as the owner recorded it; the mark is the route\'s floor.')),attrs:'data-overview="coverage"'});
  }
  /* A required action's way (B1; the book plan's item 3): where the owner's pipeline takes it -- a
   * refresh is prepared on Sources, a mapping resolved on the book's study, a human review written
   * on Review; an action naming an issuer reads that issuer. */
  const ACTION_WAYS={REFRESH_EVIDENCE:['review-step','evidence-stream','Sources › Prepare'],RESOLVE_ISSUER_MAPPING:['go','portfolio','Resolve the mapping'],HUMAN_REVIEW:['review-step','handoff','Human review'],SELECT_ANALYSIS:['review-live-item','analysis','Choose the analysis']};
  function actionWay(a) {
    if(a.action==='NONE')return null;
    if(a.entity_id) return {action:'review-live-item',value:a.entity_id,word:''};
    const kind=a.action;
    if(kind==='PREPARE_EVIDENCE') return nextOrStep('Prepare sources','prepare','evidence-stream');
    if(kind==='SETTLE_EVIDENCE_CONTINUATION' || kind==='ANALYZE_CONTINUED_EVIDENCE' || kind==='READ_CRO_DOSSIER'){
      const operation=kind==='READ_CRO_DOSSIER' ? 'CRO_REVIEW_DOSSIER' : 'EVIDENCE_PACKET';
      const requests=Object.entries(S.view?.next_requests || {}).filter(([,r])=>r?.operation===operation && (operation!=='EVIDENCE_PACKET' || !staleUnit(S.view,r.evidence_unit_id)));
      if(requests.length===1) return {word:t(NEXT_WORDS[operation][1]),action:'review-next',value:requests[0][0]};
      return requests.length ? stepTo('Choose a group to read its Analyst packet','handoff') : null;
    }
    if(kind==='SUBMIT_CRO_ASSESSMENT') return stepTo('Assess as the CRO','handoff');
    const way=ACTION_WAYS[kind];return way ? {action:way[0],value:way[1],word:t(way[2])} : null;
  }
  /* The status (E1; GitHub's merge box, measured): the verdict -- or, before a review, the cycle's
   * sentence -- with its mark; the five stages as checks, each a mark, its line and its day, a row
   * opening its reading in the pane; the required action that blocks. The Overview's one box; its
   * facts stay in the Properties column, the figures in the strip under it. */
  const STAGE_TONE={held:'good',prepared:'good',published:'good',review_published:'good',available:'good',delivered:'good',running:'accent',awaiting:'warning',ambiguous:'warning',expired:'warning',superseded:'warning',no_source:'warning',refused:TONE.attention,unavailable:TONE.failure,pending:'neutral'};
  function overviewStatus() {
    const v=S.view, c=cycle(); if(!c.stages.length) return '';
    const published=Boolean(v) && !S.refusal && (reviewPublished() || historical());
    const actions=v?.required_actions || [], blocking=requiredActions().find(a=>a.blocking);
    const human=c.checks.find(k=>k.id==='human');
    const checks=c.stages.map(st=>{ const asked=st.id==='reviewed' && Boolean(human), tone=asked ? 'warning' : STAGE_TONE[st.word] || 'neutral'; return {tone,name:t(st.name),why:asked ? t('human review required') : tone==='good' ? '' : t(stateOf(st.word).word),time:st.time || '',action:st.page===null ? '' : 'review-step',value:st.page || '',tip:st.line || ''}; }); // W (R4): a mark, a word, a day -- a done stage's name is its word; law 149: the row goes to the stage's page, the owner's line is its tip
    const tone=published && v.disposition ? stateOf(evidenceState(v.disposition).state).tone : blocking ? TONE.attention : published ? (requiredActions().length ? 'warning' : 'good') : ({accent:'accent',warning:'warning',good:'good'})[c.tone] || 'neutral'; // B1 (item 7): the route's standing, the head's
    const notReached=published ? (v.issuer_rows || []).filter(r=>!issuerReviewed(r)).sort(byWeight).map(issuerName) : [];
    const foot=actions.map(a=>{ const way=actionWay(a), standing=actionStanding(a); return {tone:standing.tone,word:standing.word,why:html`${a.action==='NONE' ? '' : codeWords(a.action)}${a.reason ? html`${a.action==='NONE' ? '' : ' \u00b7 '}<span class="owner-text">${said(a.reason)}</span>` : ''}${a.action==='REFRESH_EVIDENCE' && notReached.length ? html` <span class="status-names">${t('Not reviewed: {names}.',{names:notReached.join(', ')})}</span>` : ''}`,action:way?.action || '',value:way?.value || '',label:way?.word || ''}; }); // the owner pipeline is visible before publication; only a published review has a Report decision
    // B1 (item 7): the route is the head's state; the box says why -- the owner's first reason, the rest under it
    // the third review (2026-09-24; the user: 各管一件事): the Overview is the live standing -- the stages, the blocker and its way; the decision and its reasons are the Report's
    return statusBox({tone,title:c.sentence,reasons:[],checks,foot,label:t('Status'),attrs:'data-overview="status"'});
  }
  /* The ways out (E1): the object's other readings, one row each -- the report, the limits and the
   * decision trace; the stages open from the status box, the runs from Activity. */
  /* The issuers with their conclusions: the holdings-shaped table (law 92); a row opens the
   * issuer's issues and citations in the reading pane. */
  function issuersTable() {
    const v=S.view, rows=issuerOrder(v?.issuer_rows || []); if(!rows.length) return '';
    const q=S.issuerQuery.trim().toLowerCase(), grouped=S.issuerState ? rows.filter(r=>issuerReviewed(r)===(S.issuerState==='reviewed')) : rows; // the coverage key's filter
    const found=q ? grouped.filter(r=>[r.entity_id,entityNames(r.tickers),codeWords(r.conclusion)].some(x=>String(x || '').toLowerCase().includes(q))) : grouped;
    const PAGE=LIST_PAGE, {shown,start,page,pages}=pageOf(found,S.issuerPage); S.issuerPage=page;
    const toolbar=searchBar('issuerQuery',t('Find an issuer'),t('Find an issuer…'),S.issuerQuery);
    const foot=pager({total:found.length,one:'{n} entry',many:'{n} entries',page,pages,prev:['review-issuer-page','prev'],next:['review-issuer-page','next']});
    const columns=[{label:'#',type:'num',index:true},{label:t('Issuer'),type:'text'},{label:t('Evidence'),type:'status',absorb:true},{label:t('Weight'),type:'num',cls:'col-tight'},{label:t('Change'),type:'num',cls:'col-tight'},{label:t('Transition'),type:'status',cls:'col-tight'},{label:t('Exposure'),type:'status',cls:'col-tight'}]; // round H1: a short figure is a tight column
    // B1 (items 1 and 7): a row leads with what the review reached -- reviewed, or the owner's word and recorded reason for what it did not -- then the figures; the conclusion and the findings are the report's
    const reached=(r)=>{ if(issuerReviewed(r)) return stateLine('evidence_reviewed',{next:''}); const e=evidenceState(r.conclusion), why=unreviewedReason(r); return html`${stateLine(e.state,{word:e.word,next:''})}${why && why!==String(r.conclusion) ? html` <span class="muted">· ${codeWords(why)}</span>` : ''}`; };
    const trs=shown.map((r,i)=>tr([count(start+i+1),btnAttrs(issuerName(r),'review-live-item',r.entity_id,'text-btn','data-row-press'),reached(r),weightWords(r.ending_weight),changeWords(r.signed_change),codeWords(r.transition),bandWords(r.exposure_band)]));
    const report=reviewPublished() || historical() ? btnAttrs(html`${t('Conclusions in the report')}${icon('arrow')}`,'review-step','report','text-btn compact') : '';
    const shownOf=S.issuerState ? html`<p class="caption issuers-filter">${t(S.issuerState==='reviewed' ? 'Reviewed' : 'Not reviewed')} · ${t('{n} of {total}',{n:count(grouped.length),total:count(rows.length)})} ${btn(t('Show all'),'review-issuer-state','','text-btn compact')}</p>` : '';
    return panel(t('Issuers'),t('What the review reached, issuer by issuer; what it concluded of each is the report\'s.'),html`${shownOf}${rows.length>PAGE ? toolbar : ''}${!found.length ? emptyState(t('No issuer matches this search.'),'','','nomatch') : table(columns,trs,'',{report:true,countLine:false,classes:'issuers-table'})}${found.length>PAGE ? foot : ''}`,report,'data-overview="issuers" data-box="table"');
  }
  /* A page of runs (round H4, R3): the latest twenty of a list, grouped by day, then the foot with
   * Previous / Next; one page state for the evidence pages' Runs lists. */
  function pagedRuns(list) {
    const PAGE=LIST_PAGE, {shown,page,pages}=pageOf(list,S.runPage); S.runPage=page;
    const bound=new Set(boundTasks());
    const foot=list.length>PAGE ? pager({total:list.length,one:'{n} run',many:'{n} runs',page,pages,prev:['review-run-page','prev'],next:['review-run-page','next']}) : '';
    return html`<div class="card-list lines slotted">${[...Data.groupByDay(shown).entries()].map(([dayLabel,rs])=>html`${groupHead(dayLabel,rs.length)}${rs.map(r=>runRow(r,{inDay:true,columns:['bound'],props:[bound.has(r.id) ? t('Bound') : ''],attrs:bound.has(r.id) ? html`data-tip="${t('bound to this book')}"` : ''}))}`)}</div>${foot}`; // one list, the days its groups: the columns hold across days (S1)
  }
  /* Activity (law 55; E1): every evidence Task of the workspace, newest first, the moving one
   * first, a run bound to this book marked -- the latest three, then all of them a table page at a time. */
  const ACTIVITY_SHOWN=3;
  function activityList() {
    const list=(typeof Data.runsOf==='function' ? Data.runsOf('task') : []).filter(r=>r.kind===EVIDENCE_KIND || String(r.kind || '').startsWith('chief_risk_officer'));
    return activityFeed({title:t('Activity'),list,shown:ACTIVITY_SHOWN,open:S.runsOpen,render:pagedRuns,more:'review-runs-all',attrs:'data-overview="runs"'});
  }
  /* The book's one state (B1; the book plan's item 7, law 95), the same on every tab: once a review
   * is published -- the working one or a pinned one -- the route's standing in its mark's colour (a
   * refresh requested, an objection, a human review, accepted); before it, the projection's state.
   * The publication is the context line's chooser; a tab's own progress is its content's. */
  function headState() {
    const v=S.view; if(!v && !S.refusal) return '';
    const route=v && (reviewPublished() || historical()) && v.disposition ? evidenceState(v.disposition) : null;
    const line=route ? stateLine(route.state,{word:route.word,next:''}) : v?.state ? (v.state==='EVIDENCE_AUTHORITY_NOT_ADMITTED' ? stateLine('metadata',{word:t('Not set up'),next:''}) : stateLine(v.state,{next:''})) : '';
    return line && v?.explanation ? html`<span class="hint" data-tip="${said(v.explanation)}" tabindex="0">${line}</span>` : line; // round H1: the owner's paragraph is the state's hover card, not a lede
  }
  /* A book whose evidence is not set up (N4, law 130): one empty state with its one way -- on the
   * Overview, the Reading and the Review, and the Report of an installed result; an authored
   * book's Report still composes its delivery (its review section absent, never clearance), and
   * Sources is the way (its setup checklist). */
  function notSetUp() {
    return emptyState(t('Evidence is not set up for this workspace.'),link(t('Set up evidence'),'evidence-stream','button primary',S.selector ? {review_selector:JSON.stringify(S.selector)} : {}),'review-state page-empty');
  }
  function noBook() {
    return emptyState(t('No book to review'),link(t('Choose a book'),'books','button primary'),'review-state page-empty'); // round 93 (rule 3): one sentence, one action; N3: the way is the Evidence Home
  }
  function entry() {
    const v=S.view;
    if(v && !S.refusal && current()) queueMicrotask(()=>void readLedgers()); // the packets named now: the route's Task arrives after the open (round E2)
    if(v && !S.refusal && current()) queueMicrotask(autoRead); // the dossier once per book: the Analyst's checks are counted on the status page (round F1)
    if(v && v.state==='NO_BOOK_TO_REVIEW' && !S.selector) return noBook();
    if(v && !S.refusal && v.state==='EVIDENCE_AUTHORITY_NOT_ADMITTED') return notSetUp();
    const withBook=Boolean(v) && !S.refusal && v.state!=='NO_BOOK_TO_REVIEW';
    // the cycle's status page (round F1; B1, one column): the verdict with its way, why the book is held drawn, the Provider actions where admitted, the run, the issuers by what the review reached, the runs; its properties are the head's line and Facts
    return html`${stateBanner(false)}<div class="overview-main">${withBook ? html`${overviewStatus()}${coverageFigure()}${overviewStrip()}${stageActions()}` : S.refusal ? overviewStatus() : ''}${workSection('evidence')}${withBook ? html`${issuersTable()}${activityList()}` : ''}</div>`;
  }
  /* ---- the overview (round E2): the ledger of every packet the projection names, the topic
   * strip, the issuer x topic matrix, the cell reading and the four layers of progress. ---- */
  const TOPICS=['LIQUIDITY_GOING_CONCERN','CAPITAL_DILUTION','LEGAL_REGULATORY','OPERATIONS_SUPPLY','PRODUCT_SAFETY_CYBER','GOVERNANCE_CONTROLS','COMMERCIAL_COUNTERPARTY','CORPORATE_ACTION_LISTING'];
  const CELL_STATE={COMPLETE:'complete',PARTIAL:'partial',PENDING:'pending_delivery',NO_ROUTE:'no_route',NO_SOURCE:'no_source'};
  const TOPIC_SHORT={LIQUIDITY_GOING_CONCERN:'Liquidity',CAPITAL_DILUTION:'Capital',LEGAL_REGULATORY:'Legal',OPERATIONS_SUPPLY:'topic|Operations',PRODUCT_SAFETY_CYBER:'Safety & cyber',GOVERNANCE_CONTROLS:'Governance',COMMERCIAL_COUNTERPARTY:'Commercial',CORPORATE_ACTION_LISTING:'Corporate'}; // the ledger's heads; the strip says the whole name
  const contWords=(st)=>st ? t({PENDING:'pending',COMPLETE:'complete',NOTHING_RESUMABLE:'nothing resumable'}[String(st).toUpperCase()] || String(st).toLowerCase()) : '';
  const cellKey=(code)=>CELL_STATE[String(code || '').toUpperCase()] || String(code || '').toLowerCase();
  const ledgerId=(task,unit)=>task+'|'+(unit || '');
  /* The packets the projection names for this book: the composed packet requests, a wide book's
   * units by their packet Task, the Task the page holds, the Tasks bound here -- never a guess. */
  function packetsNamed() {
    const v=S.view, out=new Map();
    const add=(task,unit)=>{ if(task && !out.has(ledgerId(task,unit))) out.set(ledgerId(task,unit),{task,unit:unit || ''}); };
    for(const r of packetRequests(v)) add(r.task_id,r.evidence_unit_id);
    for(const u of v?.coverage_progress?.units || []) if(u.packet_task_id) add(u.packet_task_id,u.packet_unit_id || '');
    if(!out.size && S.packetTask) add(S.packetTask,S.packetUnit);
    if(!out.size) for(const id of boundFor().prepare) add(id,'');
    return [...out.values()];
  }
  /* The book's ledger, read whole (U6, R20): the Host's one answer (`EVIDENCE_LEDGER`, a page of groups by its next
   * request) carries each prepared group's cells, topics, continuation scope and coverage unit as its packet read
   * answers them, so no packet is read here. The map lists the book's prepared groups as the ledger does, before and
   * after publication alike (U55, 2026-09-30: after publication the view names no packet, and the page said nothing was
   * prepared); a group the Host cannot give a ledger keeps its code (a preparation that did not run the integrated
   * selection has none, never an empty matrix). Read again when the book's view moves. */
  async function readLedgers() {
    if(!current()) return;
    const repaint=()=>(typeof patchMain==='function' ? patchMain : render)();
    const key=S.key, rev=S.revision, v=S.view, requestedPage=app.page; // the view's revision, not S.nav: a route write moves S.nav
    const sig=[key,v?.state,v?.evidence_as_of,...packetsNamed().map(p=>ledgerId(p.task,p.unit))].join('|');
    if(S.ledgerSig===sig) return;
    S.ledgerSig=sig; S.ledgerRead={key,status:'reading'};
    try{
      const groups=[];
      for(let page=1;page && app.page===requestedPage && rev===S.revision;){
        const b=await Data.read(Data.route('EVIDENCE_LEDGER')+'?'+query({ledger_page:page}));
        if(rev!==S.revision || S.key!==key || app.page!==requestedPage) return;
        groups.push(...(b.groups || []));
        page=b.next_request?.ledger_page || 0;
      }
      for(const [id,l] of S.ledgers) if(l.key===key) S.ledgers.delete(id);
      for(const g of groups){
        if(g.status==='REFUSED') continue; // kept below with its own identity and recovery route
        if(!g.packet_task_id) continue; // a group not prepared (yet) has no packet and no ledger
        const task=g.packet_task_id, unit=g.packet_unit_id || '';
        S.ledgers.set(ledgerId(task,unit),g.refusal ? {status:'refused',key,task,unit,code:String(g.refusal).split(':')[0],message:g.refusal}
          : {status:'ready',key,task,unit,coverage:{cells:g.cells || [],topics:g.topics || []},scope:g.continuation || null,coverageUnit:g.coverage_unit || null});
      }
      S.ledgerRead={key,status:'ready',refusals:groups.filter(g=>g.status==='REFUSED')};
    }catch(e){
      if(e.name==='AbortError' || rev!==S.revision || S.key!==key || app.page!==requestedPage) return;
      const message=String(e?.message || e);
      S.ledgerRead={key,status:'failed',code:message.split(':')[0],message}; // kept, as a refusal was: a repaint never reads it again
    }
    repaint();
  }
  // what the ledger holds for this book: 'reading' until it answers, then its groups (none: nothing prepared)
  const ledgerHolds=()=>S.ledgerRead?.key!==S.key || S.ledgerRead.status==='reading' ? 'reading' : S.ledgerRead.status==='failed' ? 'failed' : ledgersHere().length || S.ledgerRead.refusals?.length ? 'groups' : 'none';
  const ledgersHere=()=>[...S.ledgers.values()].filter(l=>l.key===S.key);
  const readyLedgers=()=>ledgersHere().filter(l=>l.status==='ready' && l.coverage);
  const ledgerWords=(l)=>l.status==='refused' ? (l.code==='alternative_evidence.topic_coverage_absent' ? t('no ledger: this preparation did not run the integrated selection') : explain(l.code) || l.code) : l.status==='reading' ? t('reading the ledger') : '';
  const unitWord=(l)=>l.unit || l.coverageUnit?.unit_id || '';
  /* The topic strip: eight rows in the topics' declared order, the owner's per-unit figures
   * summed by the page; each unit's own ledger is a Facts reference. */
  function topicStrip() {
    const holds=ledgerHolds();
    if(holds==='failed') return notRead(t('Ledger not read'),S.ledgerRead.message);
    if(holds!=='groups') return '';
    const refused=(S.ledgerRead.refusals || []).map(r=>refusal(r,'warning',{catalog:true,more:html`<p>${t('Evidence group')} ${hashCell(r.group_id)}${r.packet_task_id ? html` · ${t('Task')} ${hashCell(r.packet_task_id)}` : ''}</p>`,next:prerequisiteWays(r.next_requests)}));
    const ledgers=ledgersHere(), named=ledgers, ready=readyLedgers(), order=ready[0]?.coverage?.topics?.map(x=>x.topic) || TOPICS;
    const sum=(topic,field)=>ready.reduce((a,l)=>a+(Number((l.coverage.topics || []).find(x=>x.topic===topic)?.[field]) || 0),0);
    const rows=order.map(topic=>tr([codeWords(topic),count(sum(topic,'issuers')),html`${count(sum(topic,'unit_needs_delivered'))} / ${count(sum(topic,'unit_needs'))}`,html`${count(sum(topic,'tables_complete'))} / ${count(sum(topic,'tables'))}`,count(sum(topic,'cells_with_gaps')),count(sum(topic,'cells_no_route')),count(sum(topic,'cells_no_source'))]));
    const without=ledgers.filter(l=>l.status==='refused'), reading=ledgers.filter(l=>l.status==='reading');
    const line=ready.length ? t('{n} of {m} packets with a ledger',{n:count(ready.length),m:count(named.length)})+(without.length ? ' · '+t('{n} without',{n:count(without.length)}) : '')+(reading.length ? ' · '+t('{n} reading',{n:count(reading.length)}) : '') : refused.length ? t('Ledger groups could not be read') : without.length===named.length ? t('No ledger: the preparations of this book did not run the integrated selection.') : t('Reading the ledgers');
    const perUnit=ledgers.length>1 || ledgers.some(l=>l.status!=='ready') ? factsRef(t('Each packet\'s ledger'),html`<div class="card-list lines">${ledgers.map(l=>objectRow({state:l.status==='ready' ? 'ready' : l.status==='refused' ? 'refused' : 'running',name:html`${t('Unit')} ${l.status==='refused' ? html`${locatorCell(unitWord(l))} · ${hashCell(l.task,SHORT.id)}` : html`<span class="mono">${unitWord(l) || ''}</span> · <span class="mono">${short(l.task,SHORT.id)}</span>`}`,why:l.status==='ready' ? t('{c} cells · continuation {s}',{c:count((l.coverage.cells || []).length),s:contWords(l.scope?.state)}) : ledgerWords(l),cls:'evidence-row'},{key:'ledger:'+ledgerId(l.task,l.unit)}))}</div>`) : '';
    return html`${refused}${panel(t('Topics'),ready.length ? line : '',html`${!ready.length && !refused.length ? emptyState(line) : ''}${ready.length ? table([{label:t('Topic'),type:'text',absorb:true},{label:t('Issuers'),type:'num'},{label:t('Needs'),type:'num'},{label:t('Tables'),type:'num'},{label:t('Gaps'),type:'num'},{label:t('No route'),type:'num'},{label:t('No source'),type:'num'}],rows,'',{report:true,countLine:false}) : ''}${perUnit}`,'','data-overview="topics" data-box="table"')}`;
  }
  /* The issuer x topic matrix: at the lane's width a table (the issuer, eight status cells: the
   * state's dot and word, the count of incomplete reasons, the reasons on hover), under it one
   * topic at a time over issuer rows. A cell opens its reading. */
  const cellsHere=()=>readyLedgers().flatMap(l=>(l.coverage.cells || []).map(c=>({...c,ledger:l,id:'cell:'+ledgerId(l.task,l.unit)+':'+c.entity_id+':'+c.topic})));
  function cellOf(id) { return cellsHere().find(c=>c.id===id) || null; }
  const cellWords=(c)=>(c.incomplete || []).map(x=>codeWords(x)).join(' · ');
  function cellButton(c) {
    const key=cellKey(c.delivery_state), n=(c.incomplete || []).length;
    return btnAttrs(html`${stateLine(key,{next:''})}${n>1 ? html` <span class="sub-cell">${count(n)}</span>` : ''}`,'review-live-item',c.id,'text-btn ledger-cell',html`data-cell-state="${c.delivery_state}" data-tip="${n ? cellWords(c) : t(stateOf(key).line || '')}" aria-pressed="${S.item===c.id}"`);
  }
  function ledgerMatrix() {
    const cells=cellsHere(); if(!cells.length) return '';
    const order=readyLedgers()[0]?.coverage?.topics?.map(x=>x.topic) || TOPICS;
    const issuers=[...new Set(cells.map(c=>c.entity_id))];
    const scope=Object.fromEntries((S.preview?.scope?.selected_issuers || S.view?.issuer_rows || []).map(i=>[i.entity_id,i]));
    const byCell=new Map(cells.map(c=>[c.entity_id+':'+c.topic,c]));
    const ORDER=['no_source','no_route','pending_delivery','partial','complete'];
    // the Portfolio holdings table's shape (the user's word on long tables): an index, a search field, a table page of rows (law 92), the foot's count and Previous / Next
    const q=S.ledgerQuery.trim().toLowerCase(), name=(id)=>entityNames(scope[id]?.tickers) || id;
    const found=q ? issuers.filter(id=>String(id).toLowerCase().includes(q) || String(name(id)).toLowerCase().includes(q)) : issuers;
    const PAGE=LIST_PAGE;
    const toolbar=searchBar('ledgerQuery',t('Find an issuer'),t('Find an issuer…'),S.ledgerQuery);
    // a row whose eight cells share one state is not eight words (the user's reading): it folds into its state's group; the table keeps the rows whose cells differ
    const EVERY={complete:'Complete in every topic',partial:'Partial in every topic',pending_delivery:'Pending in every topic',no_route:'No route in every topic',no_source:'No source in every topic'};
    const rowState=(id)=>{const ks=new Set(order.map(topic=>byCell.get(id+':'+topic)?.delivery_state).filter(Boolean)); return ks.size===1 ? cellKey([...ks][0]) : null;};
    const uniform=new Map(), mixed=[]; for(const id of found){const k=rowState(id); if(k){if(!uniform.has(k)) uniform.set(k,[]); uniform.get(k).push(id);} else mixed.push(id);}
    const {shown:mshown,start:mstart,page:mpage,pages:mpages}=pageOf(mixed,S.ledgerPage); S.ledgerPage=mpage;
    const mfoot=pager({total:mixed.length,one:'{n} entry',many:'{n} entries',page:mpage,pages:mpages,prev:['review-ledger-page','prev'],next:['review-ledger-page','next']});
    const uniformRows=[...uniform.entries()].sort((a,b)=>ORDER.indexOf(a[0])-ORDER.indexOf(b[0])).map(([k,list])=>{const gid='all:'+k, open=S.ledgerOpen.has(gid);
      return html`${objectRow({lead:statusDot(k),name:t(EVERY[k] || stateOf(k).word),cls:'evidence-branch'},{key:'g:'+gid,attrs:html`data-depth="0"`,props:[btnAttrs(html`${icon('chevron')}${countText(list.length,'{n} issuer','{n} issuers')}`,'review-ledger-group',gid,'text-btn evidence-fold compact',html`aria-expanded="${open}"`)]})}${open ? collectionTable('ledger-all-'+k,list,[{label:t('Issuer'),type:'text',absorb:true}],id=>{const c=byCell.get(id+':'+order[0]);return [c ? btnAttrs(name(id),'review-live-item',c.id,'text-btn','data-row-press') : name(id)];},{words:id=>name(id)+' '+id}) : ''}`;});
    const wide=html`${issuers.length>PAGE ? toolbar : ''}${!found.length ? emptyState(t('No issuer matches this search.'),'','','nomatch') : ''}${mixed.length ? table([{label:'#',type:'num',index:true},{label:t('Issuer'),type:'text'},...order.map(topic=>({label:t(TOPIC_SHORT[topic] || codeWords(topic)),type:'status',cls:'col-tight'}))],mshown.map((id,i)=>tr([count(mstart+i+1),name(id),...order.map(topic=>{const c=byCell.get(id+':'+topic);return c ? cellButton(c) : '';})])),'',{report:true,countLine:false,classes:'ledger-table compact'}) : ''}${mixed.length ? mfoot : ''}${uniformRows.length ? html`<div class="card-list evidence-tree ledger-uniform">${uniformRows}</div>` : ''}`;
    const topic=order.includes(S.ledgerTopic) ? S.ledgerTopic : order[0];
    const chooser=html`<div class="ledger-topics">${picker('ledgerTopic',order.map(x=>[x,codeWords(x)]),{selected:topic,action:'review-ledger-topic',kind:'text',label:t('Topic')})}</div>`;
    // under the lane's width: one topic, the issuers grouped by the cell's state, each group folded until opened
    const groups=new Map(); for(const id of issuers){const c=byCell.get(id+':'+topic); if(!c) continue; const k=cellKey(c.delivery_state); if(!groups.has(k)) groups.set(k,[]); groups.get(k).push(c);}
    const narrow=html`${chooser}<div class="card-list evidence-tree">${[...groups.entries()].sort((a,b)=>ORDER.indexOf(a[0])-ORDER.indexOf(b[0])).map(([k,list])=>{const gid=topic+':'+k, open=S.ledgerOpen.has(gid);
      return html`${objectRow({lead:statusDot(k),name:t(stateOf(k).word),cls:'evidence-branch'},{key:'g:'+gid,attrs:html`data-depth="0"`,props:[btnAttrs(html`${icon('chevron')}${countText(list.length,'{n} issuer','{n} issuers')}`,'review-ledger-group',gid,'text-btn evidence-fold compact',html`aria-expanded="${open}"`)]})}${open ? collectionTable('ledger-'+topic+'-'+k,list,[{label:t('Issuer'),type:'text',absorb:true},{label:t('Reason'),type:'text'}],c=>[btnAttrs(entityNames(scope[c.entity_id]?.tickers) || c.entity_id,'review-live-item',c.id,'text-btn','data-row-press'),(c.incomplete || []).length ? cellWords(c) : '']) : ''}`;})}</div>`;
    return html`<section class="panel ledger" data-overview="ledger" data-box="table">${sectionHead(t('Ledger'),t('The sealed reading plan of each issuer × topic cell, never that the topic was checked.'))}<div class="panel-body"><div class="ledger-wide">${wide}</div><div class="ledger-narrow">${narrow}</div></div></section>`;
  }
  /* The reading of one cell: its state, every reason it is not complete, its gaps, the counts of
   * its plan, and the unit's continuation (the bundles arrive with the reading page, E3). */
  function cellReading(id) {
    const c=cellOf(id); if(!c) return html`<p class="caption">${t('This cell is not in the current read.')}</p>`;
    const key=cellKey(c.delivery_state), l=c.ledger, scope=l.scope;
    const rows=(list,word,key)=>list.length ? html`<h3>${word} <span class="num">${count(list.length)}</span></h3>${textCollection('cell-'+id+'-'+key,list,x=>html`${codeWords(String(x).split(':')[0])}${String(x).includes(':') ? html` · ${ownerWords(String(x).slice(String(x).indexOf(':')+1).trim())}` : ''}`)}` : '';
    const basis=(c.regions_by_basis || []).map(([b,n])=>html`${codeWords(String(b).split(':').pop())} <span class="num">${count(n)}</span>`);
    const facts=kv([[t('State'),stateLine(key,{next:''})],[t('Unit'),html`<span class="mono">${unitWord(l) || ''}</span>`],[t('Forms held'),(c.forms_held || []).length ? c.forms_held.join(' · ') : t('none')],[t('Routed regions'),html`${count(c.regions)}${basis.length ? html` <span class="sub-cell">${joinMarkup(basis)}</span>` : ''}`],[t('Unit needs'),html`${count(c.unit_needs_delivered)} / ${count(c.unit_needs)} <span class="sub-cell">${t('{n} pending',{n:count(c.unit_needs_pending)})}</span>`],[t('Typed observations'),count(c.typed_observations)],[t('Tables'),html`${count(c.tables_complete)} ${t('complete')} · ${count(c.tables_partial)} ${t('partial')} · ${count(c.tables_pending)} ${t('not dealt')}${c.tables_progress_unknown ? html` · ${count(c.tables_progress_unknown)} ${t('unknown progress')}` : ''}${c.tables_unrenderable ? html` · ${count(c.tables_unrenderable)} ${t('unrenderable')}` : ''}${c.tables_without_original ? html` · ${count(c.tables_without_original)} ${t('without an original')}` : ''}`],[t('Candidates'),html`${count(c.candidates_pending)} ${t('pending')} · ${count(c.candidates_read)} ${t('read')}${c.candidates_covered ? html` · ${count(c.candidates_covered)} ${t('covered')}` : ''}${c.candidates_overlapping ? html` · ${count(c.candidates_overlapping)} ${t('overlapping')}` : ''}`],[t('Residual search'),html`${codeWords(c.residual)}${c.residual_hits ? html` · ${countText(c.residual_hits,'{n} hit','{n} hits')}` : ''}${c.residual_windows ? html` · ${countText(c.residual_windows,'{n} window','{n} windows')}` : ''}`],...(scope ? [[t('Continuation'),contWords(scope.state)],...(scope.next_session ? [[t('Next session'),t('{w} windows · {r} reads · {b} · {p} table pages · {c} candidates',{w:count(scope.next_session.windows),r:count(scope.next_session.reads),b:bytesWords(scope.next_session.source_bytes),p:count(scope.next_session.table_pages),c:count(scope.next_session.candidates)})],[t('Model work'),codeWords(scope.next_session.model_work)]] : [])] : [])]);
    const reasons=rows(c.incomplete || [],t('Why not complete'),'reasons'), gaps=rows(c.gaps || [],t('Gaps'),'gaps');
    const remainders=scope?.remainders && Object.keys(scope.remainders).length ? html`<h3>${t('Beyond the plan')}</h3>${kv(Object.entries(scope.remainders).map(([k,v])=>[codeWords(String(k).toUpperCase()),count(v)]))}` : '';
    const rule=scope?.scope_rule ? html`<p class="caption">${t(scope.scope_rule)}</p>` : html`<p class="caption">${t(stateOf('complete').line || '')}</p>`;
    const passages=btnAttrs(t('Read the passages of this cell'),'review-read-cell',c.entity_id+':'+c.topic,'button compact');
    return html`${facts}<p class="flow">${passages}</p>${reasons}${gaps}${remainders}${rule}`;
  }
  /* ---- the reading workbench (round E3): one prepared packet's passages as the owner's time
   * view (whole), filtered by issuer, topic and an interval ending at the run's cutoff with the
   * unfiltered coverage beside it; a passage read as its source facts and its derived
   * annotations, apart; the excerpts from every packet part read; three actions, each its own
   * request. A filter is a view, never the scope (contract section 5). ---- */
  const viewWant=()=>JSON.stringify([S.key,S.pin || workingPublication(),S.packetTask,S.packetUnit,S.viewFilter]);
  async function readView() {
    if(!ready() || (!S.packetTask && !(S.pin || workingPublication()))) return;
    const want=viewWant(); if(S.reading && S.reading.want===want) return;
    const f=S.viewFilter, filters={...(f.entity ? {view_entity_id:f.entity} : {}),...(f.topic ? {view_topic:f.topic} : {}),...(f.days ? {view_last_days:f.days} : {})};
    const fields={...S.selector,...filters,evidence_detail:'time_view',...(S.packetTask ? {task_id:S.packetTask,...(S.packetUnit ? {evidence_unit_id:S.packetUnit} : {}),delivery_part:1} : {review_publication_hash:S.pin || workingPublication()})};
    const rev=S.revision, repaint=()=>(typeof patchMain==='function' ? patchMain : render)();
    S.reading={status:'reading',want,key:S.key,task:S.packetTask,unit:S.packetUnit};
    try{ const path=S.packetTask ? Data.route('EVIDENCE_PACKET') : '/api/evidence-cro'; const value=await Data.read(path+'?'+new URLSearchParams(fields)); if(rev!==S.revision || S.reading?.want!==want) return; if(!S.packetTask && value.published_reading?.review_publication_hash!==fields.review_publication_hash)throw Error(t('This reading names another review; no passages are shown.')); S.reading={...S.reading,status:'ready',value}; }
    catch(e){ if(rev===S.revision && S.reading?.want===want && e.name!=='AbortError'){const message=String(e?.message || e); S.reading={...S.reading,status:'refused',code:message.split(':')[0],message};} }
    repaint();
  }
  const viewOf=()=>S.reading?.status==='ready' && S.reading.want===viewWant() ? S.reading.value : null;
  const evOf=()=>viewOf()?.evidence_view || viewOf()?.published_reading || null;
  /* The excerpts live in the markdown packet: every part read adds its spans (by handle) and its
   * document index; the delivery of the last part read names the next part, consumed whole. */
  function adoptExcerpts(value) {
    const facts=packetFacts(value?.packet);
    for(const sp of facts?.spans || []) if(sp?.span_handle) S.excerpts.set(sp.span_handle,{...sp,part:value?.delivery?.part || 1});
    for(const d of facts?.document_index || []) if(d?.document_handle) S.documents.set(d.document_handle,d);
    if(value?.delivery) S.packetDelivery=value.delivery;
  }
  async function readPart() {
    const r=S.packetDelivery?.next_request; if(!r || !current()) return;
    const asked={key:S.key,task:S.packetTask,revision:S.revision,request:{...r}}; S.busy='part';S.error='';render();
    try{ const doc=await Data.readDocument(Data.route('EVIDENCE_PACKET')+'?'+new URLSearchParams(requestFields(asked.request))); if(S.key!==asked.key || S.packetTask!==asked.task || S.revision!==asked.revision) return; if(!doc.value?.packet) throw Error(doc.value?.failure_code || doc.value?.detail || 'Part unavailable'); adoptExcerpts(doc.value); }
    catch(e){ if(S.key===asked.key && S.revision===asked.revision && e.name!=='AbortError') S.error=e.message; }
    finally{ if(S.key===asked.key && S.revision===asked.revision){S.busy='';render();} }
  }
  const excerptOf=(handle)=>S.excerpts.get(handle) || null;
  const docOf=(handle)=>S.documents.get(handle) || null;
  const dayOf=(x)=>x ? String(x).slice(0,10) : '';
  // a document's time at its stated precision: an acceptance names the day it was accepted; a date-only or unknown-precision publication never gets an instant
  const timeWords=(tm)=>{ if(!tm) return ''; const at=tm.accepted_at || tm.published_at || tm.available_at; if(!at) return t('no stated time'); const word=tm.accepted_at ? t('accepted') : tm.published_at ? (String(tm.published_precision || '')==='DATE' ? '' : t('precision {p}',{p:codeWords(tm.published_precision || 'UNKNOWN')})) : t('available'); return html`<span data-tip="${word}">${dayOf(at)}</span>`; };
  const progressWord=(st)=>st ? t(stateOf(String(st).toLowerCase()).word) : '';
  const tableWords=(tb)=>{ const p=tb?.progress || {}; return html`${progressWord(p.state)} · ${t('{d} of {n} rows delivered',{d:count(p.rows_delivered),n:count(p.rows_declared)})}${p.next_row ? html` · ${t('resumes at row {r}',{r:count(p.next_row)})}` : ''}${p.rows_clipped ? html` · ${countText(p.rows_clipped,'{n} row clipped','{n} rows clipped')}` : ''}${p.gap ? html` · ${t('a gap between pages')}` : ''}${p.pages ? html` · ${countText(p.pages,'{n} page','{n} pages')}` : ''}${tb?.rows_from!=null ? html` <span class="sub-cell">${t('this page rows {a}–{b}',{a:count(tb.rows_from),b:count(tb.rows_to)})}</span>` : ''}`; };
  const foundWords=(list)=>{ const words=[]; for(const x of list || []){ const q=/^Q-([A-Z_]+):([A-Z_]+)$/.exec(String(x)), c=/^C\d+:COVERED_CANDIDATE:/.test(String(x)); if(q) words.push(t('{k} question',{k:codeWords(q[2])})); else if(c) words.push(t('covered candidate')); } return [...new Set(words)].join(' · '); };
  const whatOf=(b)=>b.table ? tableWords(b.table) : b.unit ? (b.unit.title || b.unit.region_heading || b.unit.unit_handle || '') : b.statement ? html`${codeWords(b.statement.family)} · ${codeWords(b.statement.state)}` : b.structure_family ? codeWords(b.structure_family) : foundWords(b.found_by);
  const topicsOf=(b)=>{ const list=b.topics || []; return list.length ? html`<span data-tip="${list.map(codeWords).join(' · ')}">${codeWords(list[0])}${list.length>1 ? html` <span class="sub-cell">+${count(list.length-1)}</span>` : ''}</span>` : ''; };
  const passageDelivery=(b)=>{ const d=b.delivery || {}; return html`${codeWords(d.state)}${d.whole===false ? html` · ${t('in part')}` : ''}${d.document_inspection==='PARTIAL' ? html` · ${t('document in part')}` : ''}`; };
  const issuerNames=()=>Object.fromEntries((S.view?.issuer_rows || []).map(i=>[i.entity_id,entityNames(i.tickers) || i.entity_id]));
  function viewFacts() {
    const v=viewOf(), e=evOf(); if(!e) return '';
    const iv=e.interval || {}, days=/LAST_(\d+)_DAYS/.exec(String(iv.basis || ''))?.[1];
    const interval=days ? t('Last {n} days ending at the cutoff {d}',{n:count(days),d:dayOf(iv.evidence_cutoff)}) : iv.from || iv.to ? html`${dayOf(iv.from)} — ${dayOf(iv.to)}` : t('All time to the cutoff {d}',{d:dayOf(iv.evidence_cutoff || v.source_expires_at)});
    const cells=e.coverage?.cells || [], by=(k)=>cells.filter(c=>c.delivery_state===k).length;
    const d=S.packetDelivery;
    const excerpts=!d ? html`${t('not read yet')} · ${btnAttrs(t('Read the packet'),'review-read-excerpts','','text-btn')}` : html`${t('{a} of {b} passages · part {p} of {n} read',{a:count(S.excerpts.size),b:count(d.scope_span_count ?? S.excerpts.size),p:count(d.part),n:count(d.part_count)})}${d.next_request ? html` · ${btnAttrs(t('Read the next part'),'review-read-part','','text-btn')}` : ''}`;
    return kv([[t('Packet'),html`<span class="mono">${short(S.packetTask,SHORT.id)}</span>${S.packetUnit ? html` · ${t('Unit')} <span class="mono">${S.packetUnit}</span>` : ''}${v.source_expires_at ? html` <span class="sub-cell">${t('sources expire {d}',{d:dayOf(v.source_expires_at)})}</span>` : ''}`],
      [t('Interval'),html`${interval}${iv.time_basis ? html` <span class="sub-cell">${t('by {b}',{b:codeWords(iv.time_basis)})}</span>` : ''}`],
      [t('Passages'),html`${t('{s} of {n} selected',{s:count(e.selected),n:count(e.delivered_total)})}${e.outside_interval ? html` · ${t('{n} outside the interval',{n:count(e.outside_interval)})}` : ''}${(e.boundary || []).length ? html` · ${t('{n} on the boundary',{n:count(e.boundary.length)})}` : ''}${(e.unknown_time || []).length ? html` · ${t('{n} of unknown time',{n:count(e.unknown_time.length)})}` : ''}${(e.historical_context || []).length ? html` · ${t('{n} historical context',{n:count(e.historical_context.length)})}` : ''}`],
      [t('Coverage'),html`${t('{a} complete · {b} partial · {c} pending · {d} no route · {e} no source',{a:count(by('COMPLETE')),b:count(by('PARTIAL')),c:count(by('PENDING')),d:count(by('NO_ROUTE')),e:count(by('NO_SOURCE'))})} <span class="sub-cell">${t('unfiltered')}</span>`],
      [t('Excerpts'),excerpts]]);
  }
  function viewFilter(name,value) { S.viewFilter={...S.viewFilter,[name]:String(value || '')}; S.reading=null; S.spanPage=0; writeRoute(routeUpdate()); render(); queueMicrotask(()=>void readView()); }
  function passagesTable() {
    const e=evOf(); if(!e) return '';
    const rows=e.bundles || [], q=S.spanQuery.trim().toLowerCase(), names=issuerNames(), name=(id)=>names[id] || id;
    const found=q ? rows.filter(b=>[b.span_handle,b.document_handle,b.entity_id,name(b.entity_id),b.unit?.title,b.unit?.region_heading,b.statement?.family].filter(Boolean).some(x=>String(x).toLowerCase().includes(q))) : rows;
    const PAGE=LIST_PAGE, {shown,start,page,pages}=pageOf(found,S.spanPage); S.spanPage=page;
    const toolbar=searchBar('spanQuery',t('Find a passage'),t('Find a passage…'),S.spanQuery);
    const foot=pager({total:found.length,one:'{n} entry',many:'{n} entries',page,pages,prev:['review-span-page','prev'],next:['review-span-page','next']});
    const row=(b,i)=>tr([count(start+i+1),name(b.entity_id),btnAttrs(html`<span class="mono">${b.document_handle}</span> <span class="sub-cell">${b.document_type || ''}</span>`,'review-live-item','span:'+b.span_handle,'text-btn passage-open',html`aria-pressed="${S.item==='span:'+b.span_handle}"`),timeWords(b.time),codeWords(b.method),html`<span class="owner-text">${whatOf(b)}</span>`,topicsOf(b),passageDelivery(b)]);
    return html`${rows.length>PAGE ? toolbar : ''}${found.length ? table([{label:'#',type:'num',index:true},{label:t('Issuer'),type:'text'},{label:t('Document'),type:'text',cls:'col-tight'},{label:t('Time'),type:'date',cls:'col-tight'},{label:t('Method'),type:'text',cls:'col-tight'},{label:t('What'),type:'text',absorb:true},{label:t('Topics'),type:'text',cls:'col-tight'},{label:t('Delivery'),type:'status',cls:'col-tight'}],shown.map(row),'',{report:true,countLine:false,classes:'passages-table compact'}) : emptyState(t(q ? 'No passage matches this search.' : 'No passage in this view.'),'','',q ? 'nomatch' : 'elsewhere')}${rows.length>PAGE ? foot : ''}`;
  }
  function viewGroups() {
    const e=evOf(); if(!e) return ''; const names=issuerNames(), name=(id)=>names[id] || id;
    const groups=[['boundary',t('On the boundary'),e.boundary || []],['unknown_time',t('Unknown time'),e.unknown_time || []],['historical_context',t('Historical context'),e.historical_context || []]].filter(([,,list])=>list.length);
    if(!groups.length && !e.outside_interval) return '';
    return html`<div class="card-list evidence-tree view-groups">${groups.map(([id,word,list])=>{const gid='view:'+id, open=S.ledgerOpen.has(gid);
      return html`${objectRow({name:word,why:id==='historical_context' ? t('the same text served by a later reading; not recent') : '',cls:'evidence-branch'},{key:'g:'+gid,attrs:html`data-depth="0"`,props:[btnAttrs(html`${icon('chevron')}${countText(list.length,'{n} passage','{n} passages')}`,'review-ledger-group',gid,'text-btn evidence-fold compact',html`aria-expanded="${open}"`)]})}${open ? list.map(b=>objectRow({name:html`<span class="mono">${b.span_handle}</span> · ${name(b.entity_id)}`,why:html`<span class="mono">${b.document_handle}</span> · ${timeWords(b.time)}${b.relation ? html` · ${codeWords(b.relation)}` : ''}`,to:{action:'review-live-item',value:'span:'+b.span_handle},cls:'evidence-row'},{key:'v:'+b.span_handle,selected:S.item==='span:'+b.span_handle,attrs:html`data-depth="1"`})) : ''}`;})}${e.outside_interval ? html`<p class="caption">${t('{n} passages outside the interval are not listed; the coverage counts them.',{n:count(e.outside_interval)})}</p>` : ''}</div>`;
  }
  function readingActions() {
    const v=viewOf(); if(!v) return ''; const scope=v.continuation_scope || {}, r=v.continuation_request, d=S.packetDelivery, ns=scope.next_session;
    const facts=kv([[t('Continuation'),html`${contWords(scope.state)}${ns ? html` <span class="sub-cell">${t('{w} windows · {r} reads · {b} · {p} table pages · {c} candidates',{w:count(ns.windows),r:count(ns.reads),b:bytesWords(ns.source_bytes),p:count(ns.table_pages),c:count(ns.candidates)})}</span>` : ''}`],
      ...(scope.remainders && Object.keys(scope.remainders).length ? [[t('Beyond the plan'),Object.entries(scope.remainders).map(([k,n])=>`${codeWords(String(k).toUpperCase())} ${count(n)}`).join(' · ')]] : [])]);
    const buttons=d?.next_request ? html`<div class="flow review-actions">${typedBtn(t('Read the next part'),'review-read-part','','button',current() ? '' : t('Read-only while pinned or busy'))}</div>` : ''; // `Read more` is the head's primary (round F3)
    return panel(t('Read more'),t('One bounded reading session; the owner names its scope, the reader declares the limits.'),html`${facts}${buttons}`,'','data-overview="reading-actions"');
  }
  const sameBook=(r)=>Boolean(S.selector) && Object.entries(S.selector).every(([k,v])=>r?.[k]===v);
  function continueDialog() {
    const v=viewOf(); const r=v?.continuation_request, scope=v?.continuation_scope || {}; if(!r || !current()) return;
    if(!sameBook(r)){S.error=t('This request names another book; open that book to run it.');render();return;}
    const chain=packetJson()?.litigation_matters?.reading_chain || null, declare=Array.isArray(scope.declare) ? scope.declare : [];
    const payload=requestFields(r); for(const k of declare) if(!(k in payload)) payload[k]=null;
    closeDialog();S.pending={path:Data.route('EVIDENCE_CONTINUE'),payload,revision:S.revision,kind:'execute',operation:'EVIDENCE_CONTINUE',key:S.key,role:S.role,task:S.packetTask,unit:S.packetUnit,declare};
    const ns=scope.next_session || {};
    const cost=kv([[t('One more session reads'),t('{w} windows · {r} reads · {b} · {p} table pages · {c} candidates',{w:count(ns.windows),r:count(ns.reads),b:bytesWords(ns.source_bytes),p:count(ns.table_pages),c:count(ns.candidates)})],[t('Model work'),codeWords(ns.model_work || 'NONE')],...(chain?.remaining ? [[t('Remaining in the chain'),t('{s} sessions · {w} windows',{s:count(chain.remaining.sessions),w:count(chain.remaining.windows)})]] : [])]);
    const fields=declare.map(k=>field(codeWords(String(k).toUpperCase()),'continueLimit-'+k,payload[k] ?? '','number',t(k==='session_limit' ? 'Sessions this continuation may add to the chain' : k==='window_limit' ? 'Windows the chain may read in all' : 'A limit the owner declares'),html`data-continue-limit="${k}" min="1" step="1"`));
    openDialog(t('Continue reading'),bookName(),html`<p>${t('Reads the sources further within the declared budget; a new bounded read, nothing recomputed.')}</p>${cost}${fields}${scope.scope_rule ? html`<p class="caption">${t(scope.scope_rule)}</p>` : ''}`,html`${btn(t('Continue reading'),'review-commit','','button primary')}`);
  }
  function continueLimit(name,value) { if(!S.pending || S.pending.operation!=='EVIDENCE_CONTINUE') return; const n=Number(value); S.pending.payload[name]=value!=='' && Number.isFinite(n) ? n : null; }
  // choosing a group's packet is choosing the Analyst's turn for that group, whatever the book's turn (10.9)
  function usePacket(v) { const [task,unit='']=String(v || '').split('|'); if(!task) return; S.nav+=1;S.packetTask=task;S.packetUnit=unit;S.roleChosen='analyst';S.pending=null;S.packet=null;S.bundleRev++;S.reading=null;S.spanPage=0;loadDrafts();pushRoute(routeUpdate());render();queueMicrotask(()=>void readView()); }
  function readCell(v) { const [entity,topic]=String(v || '').split(':'); S.viewFilter={entity:entity || '',topic:topic || '',days:''}; S.reading=null; S.spanPage=0; S.nav+=1; app.page='evidence-reading'; pushRoute(routeUpdate({page:'evidence-reading'})); render(); queueMicrotask(ensure); }
  function spanReading(handle) {
    const e=evOf(); const all=[...(e?.bundles || []),...(e?.boundary || []),...(e?.unknown_time || []),...(e?.historical_context || [])]; const b=all.find(x=>x.span_handle===handle);
    if(!b || !S.packetTask){ const x=spanOf(handle); if(x && x.excerpt) return html`<h3>${t('Source')}</h3>${kv([[t('Issuer'),issuerNames()[x.entity_id] || x.entity_id || ''],[t('Document'),html`<span class="mono">${x.document_handle || ''}</span> · ${x.document_type || ''}`],[t('Available'),x.available_at ? dayOf(x.available_at) : ''],[t('Verified'),t('the owner replayed this excerpt for the published review')]])}<pre class="excerpt">${x.excerpt}</pre>${(x.limitations || []).length ? html`<h3>${t('Qualifications')}</h3><ul class="fact-list">${x.limitations.map(q=>html`<li>${said(q)}</li>`)}</ul>` : ''}`; return html`<p class="caption">${t(b ? 'This passage was not delivered in the published review.' : 'This passage is not in the current view.')}</p>`; }
    const ex=excerptOf(handle), doc=docOf(b.document_handle), tm=b.time || {};
    const source=kv([[t('Issuer'),issuerNames()[b.entity_id] || b.entity_id],[t('Document'),html`<span class="mono">${b.document_handle}</span> · ${b.document_type || ''}${doc?.title ? html` · ${doc.title}` : ''}`],[t('Published'),html`${tm.published_at ? dayOf(tm.published_at) : t('not stated')} <span class="sub-cell">${t('precision {p}',{p:codeWords(tm.published_precision || 'UNKNOWN')})}</span>`],[t('Available'),tm.available_at ? dayOf(tm.available_at) : ''],[t('Accepted'),tm.accepted_at ? dayOf(tm.accepted_at) : t('not stated')],[t('Time basis'),codeWords(tm.availability_basis)],...(tm.report_period_end ? [[t('Report period end'),tm.report_period_end]] : []),[t('Characters'),html`${count(b.character_range?.[0])} – ${count(b.character_range?.[1])} · ${bytesWords(b.excerpt_bytes)}`]]);
    const excerpt=ex ? html`<pre class="excerpt">${ex.excerpt}</pre>` : html`<p class="caption">${t('The excerpt is in a part of the packet not read yet.')}</p><p class="flow">${S.packetDelivery?.next_request ? btnAttrs(t('Read the next part'),'review-read-part','','button compact') : S.packetDelivery ? '' : btnAttrs(t('Read the packet'),'review-read-excerpts','','button compact')}</p>`;
    const found=(b.found_by || []).map((x,i)=>html`${i ? ' · ' : ''}<span class="mono">${x}</span>`);
    const derived=kv([[t('Method'),codeWords(b.method)],[t('Found by'),found.length ? html`${found}` : ''],[t('Topics'),(b.topics || []).map(codeWords).join(' · ') || ''],
      ...(b.unit ? [[t('Matter'),html`${b.unit.title || ''}${b.unit.family ? html` · ${codeWords(b.unit.family)}` : ''}${b.unit.region_heading ? html` <span class="sub-cell">${b.unit.region_heading}</span>` : ''}${b.unit.named_proceeding ? html` · ${t('named proceeding')}` : ''}${(b.unit.case_numbers || []).length ? html` · ${b.unit.case_numbers.join(', ')}` : ''}`]] : []),
      ...(b.statement ? [[t('Statement'),html`${codeWords(b.statement.family)} · ${codeWords(b.statement.state)}${b.statement.reason ? html` · ${codeWords(b.statement.reason)}` : ''}${b.statement.sentence ? html` <span class="sub-cell">${b.statement.sentence}</span>` : ''} <span class="sub-cell mono">${b.statement.rule_id || ''}</span>${b.statement.section ? html` <span class="sub-cell">${[b.statement.section.part,b.statement.section.item ? t('Item {i}',{i:b.statement.section.item}) : ''].filter(Boolean).join(' · ')}</span>` : ''}${b.statement.report_period_end ? html` <span class="sub-cell">${b.statement.report_period_end}</span>` : ''}`]] : []),
      ...(b.comparison ? [[t('Comparison'),html`${codeWords(b.comparison.state)}${b.comparison.basis ? html` <span class="sub-cell">${said(b.comparison.basis)}</span>` : ''}${b.comparison.changed_characters ? html` · ${t('{n} characters changed',{n:count(b.comparison.changed_characters)})}` : ''}${b.comparison.rules_id ? html` <span class="sub-cell mono">${b.comparison.rules_id}</span>` : ''}`]] : []),
      ...(b.table ? [[t('Table'),tableWords(b.table)]] : []),
      [t('Delivery'),html`${codeWords(b.delivery?.state)}${b.delivery?.whole===false ? html` · ${t('in part')}` : ''}${b.delivery?.document_inspection ? html` · ${t('document {s}',{s:codeWords(b.delivery.document_inspection)})}` : ''}${b.delivery?.document_windows_pending ? html` · ${countText(b.delivery.document_windows_pending,'{n} window pending','{n} windows pending')}` : ''}`],
      ...((b.unresolved_references || []).length ? [[t('Unresolved references'),b.unresolved_references.map(String).join(' · ')]] : [])]);
    const quals=(b.qualifications || []).length ? html`<h3>${t('Qualifications')}</h3><ul class="fact-list">${b.qualifications.map(q=>html`<li>${q}</li>`)}</ul>` : '';
    return html`<h3>${t('Source')}</h3>${source}${excerpt}<h3>${t('Derived')}</h3>${derived}${quals}`;
  }
  /* ---- the Reading page (round F3, Linear's view): the prepared evidence as a view -- the head
   * names it and its count, the chips are its filters (a filter is a view, never the scope), the
   * map (the topic strip and the issuer x topic ledger) and the passages sit under them, the facts
   * and the distributions read in the Facts panel, `Read more` is the one lever. No chooser: the
   * packet the owner names is read; a wide book's groups are one picker; a published book that
   * names no packet reads the export's citations; else one sentence and the way in. ---- */
  const exportSpans=()=>S.exportDoc?.value?.evidence?.verified_spans || [];
  const groupWords=(r)=>{ const p=r.evidence_unit_id ? unitProgress(r.evidence_unit_id) : null; const n=p ? (p.entity_ids || []).length : 0; return (r.evidence_unit_id || short(r.task_id,SHORT.id))+(n ? ' · '+countText(n,'{n} issuer','{n} issuers') : '')+(p?.state ? ' · '+t(stateOf(String(p.state).toLowerCase()).word) : ''); };
  function readingWords() {
    const v=S.view, e=evOf(), rd=S.reading;
    if(!v || S.refusal) return {sentence:'',primary:null};
    if(!S.packetTask){
      const named=packetRequests(v);
      if(named.length) return {sentence:t('{n} prepared groups; the first is read.',{n:count(named.length)}),primary:null};
      if(exportSpans().length) return {sentence:t('The published review\'s citations; the preparation\'s own reading is not offered for this cutoff.'),primary:null};
      if(reviewPublished() || historical()) return {sentence:t('Reading the published review\'s citations.'),primary:null};
      const holds=ledgerHolds(); // U55: the ledger's groups, whatever the view names
      if(holds==='groups') return {sentence:t('{n} prepared groups, as the book\'s ledger lists them.',{n:count(ledgersHere().length)}),primary:null};
      if(holds==='reading') return {sentence:t('Reading the book\'s ledger.'),primary:null};
      return {sentence:t('No evidence is prepared for this book.'),primary:cycle().primary};
    }
    if(!e) return {sentence:rd?.status==='refused' ? t('The view was not read.') : t('Reading the prepared evidence.'),primary:null};
    const bundles=e.bundles || [], issuers=new Set(bundles.map(b=>b.entity_id).filter(Boolean)), topics=new Set(bundles.flatMap(b=>b.topics || []));
    const scope=viewOf()?.continuation_scope || {};
    const sentence=t('{p} · {i} · {t} · as of {d}',{p:countText(e.delivered_total ?? bundles.length,'{n} passage','{n} passages'),i:countText(issuers.size,'{n} issuer','{n} issuers'),t:countText(topics.size,'{n} topic','{n} topics'),d:dayOf(e.interval?.evidence_cutoff || viewOf()?.source_expires_at) || ''})+(e.outside_interval ? ' · '+t('{n} outside the interval',{n:count(e.outside_interval)}) : '')+'.';
    const primary=scope.state==='PENDING' && viewOf()?.continuation_request ? {word:t('Read more'),action:'review-continue',value:''} : null;
    return {sentence,primary};
  }
  /* The chips: issuer, topic and interval as property chips (Linear); a set chip is a view. */
  function viewChips() {
    const e=evOf(), names=issuerNames(), f=S.viewFilter;
    const issuers=[...new Set([...Object.keys(names),...(e?.coverage?.cells || []).map(c=>c.entity_id)])];
    const fields=[{name:'evidence-view-entity',label:t('Issuer'),value:f.entity,options:issuers.map(id=>[id,names[id] || id])},{name:'evidence-view-topic',label:t('Topic'),value:f.topic,options:TOPICS.map(x=>[x,codeWords(x)])},{name:'evidence-view-days',label:t('Interval'),value:f.days,options:[['30',t('Last 30 days')],['90',t('Last 90 days')],['365',t('Last 365 days')]]}];
    return html`<div class="filter-row view-chips">${filterBar(fields,{pending:S.pendingFilter,clear:'review-view-clear'})}</div>`;
  }
  function viewClear() { S.pendingFilter=''; S.viewFilter={entity:'',topic:'',days:''}; S.reading=null; S.spanPage=0; writeRoute(routeUpdate()); render(); queueMicrotask(()=>void readView()); }
  /* The Facts panel: this reading's facts, then the distributions -- by issuer, by topic, by
   * document type, by delivery -- each count a way to set the chip. */
  function readingFacts() {
    const e=evOf(); if(!e) return exportSpans().length ? [{title:t('This reading'),body:html`<p>${t('The published review\'s verified citations, read from the export; the preparation\'s own reading is not offered for this cutoff.')}</p>`}] : [];
    if(!S.packetTask) return [{title:t('This reading'),body:kv([[t('Passages'),t('{s} of {n} selected',{s:count(e.selected),n:count(e.delivered_total)})],...(e.analyses || []).map(a=>[a.unit_id || short(a.analysis_publication_hash,SHORT.hash),e.filters?.last_days ? t('Last {n} days ending at the cutoff {d}',{n:count(e.filters.last_days),d:dayOf(a.evidence_cutoff)}) : t('All time to the cutoff {d}',{d:dayOf(a.evidence_cutoff)})])])}];
    const names=issuerNames(), bundles=e.bundles || [];
    const tally=(items)=>{ const m=new Map(); for(const k of items) if(k) m.set(k,(m.get(k) || 0)+1); return [...m.entries()].sort((a,b)=>b[1]-a[1]); };
    const list=(pairs,name,word)=>pairs.length ? html`<ul class="fact-list">${pairs.map(([k,n])=>html`<li>${name ? btnAttrs(html`${word(k)} <span class="num">${count(n)}</span>`,'filter-set',name+':'+k,'text-btn') : html`${word(k)} <span class="num">${count(n)}</span>`}</li>`)}</ul>` : html`<p class="caption">${t('none in this view')}</p>`;
    return [{title:t('This reading'),body:viewFacts()},
      {title:t('By issuer'),body:list(tally(bundles.map(b=>b.entity_id)),'evidence-view-entity',(k)=>names[k] || k)},
      {title:t('By topic'),body:list(tally(bundles.flatMap(b=>b.topics || [])),'evidence-view-topic',(k)=>codeWords(k))},
      {title:t('By document type'),body:list(tally(bundles.map(b=>b.document_type)),'',(k)=>String(k))},
      {title:t('By delivery'),body:list(tally(bundles.map(b=>b.delivery?.state)),'',(k)=>codeWords(k))}];
  }
  /* The published review's citations as the reading when no packet is named (A3's fallback). */
  function citationsTable(spans) {
    const e=evOf(), handles=new Set((e?.bundles || []).map(b=>b.span_handle));
    const rows=spans.filter(x=>handles.has(x.span_handle)), names=issuerNames(), q=S.spanQuery.trim().toLowerCase();
    const found=q ? rows.filter(x=>[x.span_handle,x.document_handle,x.entity_id,names[x.entity_id],x.excerpt].filter(Boolean).some(v=>String(v).toLowerCase().includes(q))) : rows;
    const PAGE=LIST_PAGE, {shown,start,page,pages}=pageOf(found,S.spanPage); S.spanPage=page;
    const toolbar=searchBar('spanQuery',t('Find a passage'),t('Find a passage…'),S.spanQuery);
    const foot=pager({total:found.length,one:'{n} entry',many:'{n} entries',page,pages,prev:['review-span-page','prev'],next:['review-span-page','next']});
    // E3: a passage is its document over its words (two lines; whole in the pane), the issuer, the day and the state one word each
    // B2 (item 4): a passage names the findings that cite it, each the way to it (read beside the list)
    const findings=dossierOf()?.findings || reportOf()?.review?.dossier?.findings || [];
    const citing=(h)=>findings.filter(f=>[...(f.supporting_span_handles || []),...(f.contradicting_span_handles || [])].includes(h));
    const cited=(h)=>{ const list=citing(h); return list.length ? html`${list.map((f)=>btnAttrs(codeWords(f.topic),'review-live-item',f.finding_handle,'text-btn inline-link',html`data-tip="${f.summary || ''}"`))}` : html`<span class="muted">${t('none')}</span>`; }; // one topic a line (the cell's links are blocks)
    const passages=shown.map(x=>({handle:x.span_handle,document:x.document_handle || '',type:x.document_type || '',excerpt:String(x.excerpt || '').replace(/\s+/g,' '),issuer:names[x.entity_id] || x.entity_id || '',day:x.available_at ? dayOf(x.available_at) : '',state:stateLine('verified',{next:''}),cited:findings.length ? cited(x.span_handle) : null,action:'review-live-item',value:'span:'+x.span_handle,selected:S.item==='span:'+x.span_handle}));
    return html`<section class="panel" data-overview="citations" data-box="table">${sectionHead(html`${t('Citations')} <span class="num">${count(rows.length)}</span>`,t('The passages the published review verified; a filter is a view, the export stays whole.'))}<div class="panel-body">${viewChips()}<p class="caption">${t('{s} of {n} selected',{s:count(e?.selected),n:count(e?.delivered_total)})}</p>${rows.length>PAGE ? toolbar : ''}${found.length ? passageList(passages) : emptyState(t(q ? 'No passage matches this search.' : 'No passage in this view.'),'','',q ? 'nomatch' : 'elsewhere')}${found.length>PAGE ? foot : ''}${viewGroups()}</div></section>`;
  }
  function readingPage() {
    const v=S.view; if(!v || S.refusal) return html`${stateBanner(false)}`;
    if(current()) queueMicrotask(()=>void readLedgers()); // the map reads the book's ledger (U6, U55)
    const map=html`${topicStrip()}${ledgerMatrix()}`;
    if(!S.packetTask){
      const named=packetRequests(v);
      if(named.length){ // the owner's packet is the page's default, adopted quietly (no level); a wide book's first group
        const gen=S.nav, k=S.key; // the adoption belongs to this book and this navigation: a later open supersedes it
        if(typeof queueMicrotask==='function') queueMicrotask(()=>{ if(gen!==S.nav || k!==S.key || S.packetTask || !packetRequests(S.view).length) return; const r=packetRequests(S.view)[0]; S.packetTask=r.task_id; S.packetUnit=r.evidence_unit_id || ''; loadDrafts(); writeRoute(routeUpdate()); repaintMain(); });
        return html`<div data-overview="reading">${map}${skeleton('rows')}</div>`;
      }
      const published=reviewPublished() || historical();
      if(published && !S.exportDoc && !S.error && ready() && typeof queueMicrotask==='function') queueMicrotask(()=>void readExport(null,true));
      const spans=exportSpans();
      if(spans.length){
        queueMicrotask(()=>void readView());
        const rd=S.reading?.want===viewWant() ? S.reading : null;
        const list=!rd || rd.status==='reading' ? skeleton('rows') : rd.status==='refused' ? notRead(t('The view was not read'),rd.code,explain(rd.code) || rd.message) : citationsTable(spans);
        return html`<div data-overview="reading">${map}${list}</div>`;
      }
      if(S.busy==='export' || (published && !S.exportDoc && !S.error)) return html`<div data-overview="reading">${map}${skeleton('rows')}</div>`;
      const holds=ledgerHolds(); // U55: the map is the page while the ledger holds a group
      if(holds==='groups' || holds==='failed') return html`<div data-overview="reading">${map}</div>`;
      if(holds==='reading') return html`<div data-overview="reading">${map}${skeleton('rows')}</div>`;
      const p=cycle().primary;
      return html`<div data-overview="reading">${map}${emptyState(t('No evidence is prepared for this book.'),p ? typedBtn(p.word,p.action,p.value,'button primary') : '')}</div>`;
    }
    const named=packetRequests(v);
    const groups=named.length>1 ? html`<div class="flow view-pickers">${picker('viewGroup',named.map(r=>[r.task_id+'|'+(r.evidence_unit_id || ''),groupWords(r)]),{selected:S.packetTask+'|'+(S.packetUnit || ''),action:'review-use-packet',kind:'text',label:t('Group')})}</div>` : '';
    queueMicrotask(()=>void readView());
    const rd=S.reading;
    const body=!rd || rd.status==='reading' ? html`${map}${skeleton('rows')}` : rd.status==='refused' ? html`${map}${notRead(t('The view was not read'),rd.code,explain(rd.code) || rd.message)}` : html`${map}<section class="panel" data-overview="passages" data-box="table"><header class="panel-head"><div><h2>${t('Passages')} <span class="num">${count((evOf()?.bundles || []).length)}</span></h2><p class="caption">${t('One row a passage; a chip is a view of them, the map above stays whole.')}</p></div></header><div class="panel-body">${viewChips()}${passagesTable()}${viewGroups()}</div></section>${readingActions()}`; // the map is its own read: it stands whatever the view's state; the chips filter the passages, so they sit with them
    return html`<div data-overview="reading">${groups}${body}</div>`;
  }
  /* ---- the composition for the CRO (round E4): the dossier as the tables the CRO reads -- the
   * issuers with their review state, the findings with their support and the held answer's
   * disposition, the coverage's missing evidence by issuer and topic, the unresolved questions,
   * the units' publications, the routes the policy allows. Every row its owner's words. ---- */
  const REVIEW_STATE_KEY={SOURCE_MISSING:'source_missing',NO_SPANS_DELIVERED:'no_spans_delivered',NOT_REPORTED:'not_reported',CHECKS_INCOMPLETE:'checks_incomplete',EXECUTED_NO_FINDINGS:'executed_no_findings',EXECUTED_WITH_FINDINGS:'executed_with_findings'};
  // an owner's gap line reads as words: a code alone is its word, a count keeps its noun by number
  // an owner's counted words (the analysis's gap lines and the CRO's limitations are sealed text, WD2): a noun's `(s)` takes
  // the number before it, a code inside the words reads as its words; the owner's text itself is never rewritten
  const ownerWords=(s)=>{const words=String(s || '').replace(/(\d[\d,]*)(\D*?)\b([a-z]+)\(s\)/g,(all,n,mid,noun)=>n+mid+noun+(Number(n.replace(/,/g,''))===1 ? '' : 's')),read=said(words);return read!==words ? read : words.replace(/\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b/g,(c)=>codeWords(c));};
  // Only the CRO's generated limits compose this counted suffix. A participant's
  // similar words in another reader remain exact, and a detail's semicolons stay
  // inside its entry: only the next named unreviewed unit/issuer starts a new one.
  const reviewLimitWords=(s)=>{const words=String(s || ''),more=/^([\s\S]+); and (\d[\d,]*) more\.$/.exec(words),base=more ? more[1] : words,separator=I18N.zh ? '；' : '; ',read=base.split(/; (?=(?:unit [^();\n]+ \([^();\n]+\)|[^;\n]+) is not reviewed: )/).map(ownerWords).join(separator);return read+(more ? separator+t('and {n} more.',{n:more[2]}) : '');};
  const gapWords=(why)=>String(why || '').split(/;\s*/).map(part=>{const m=/^([A-Z][A-Z_]+)$/.exec(part.trim()); return m ? codeWords(m[1]) : ownerWords(part.trim());}).filter(Boolean).join(' · ');
  // the owner's gap lines as words (round E5): `<issuer> <TOPIC>: <what>` and `<issuer> is not reviewed: <CODE>`; an issuer whose every listed topic misses the same thing is one line
  // the owner's gap lines, grouped by issuer; a book of several groups prefixes each with its group (u21: ...)
  function gapLines(lines) {
    const names=issuerNames(), name=(id)=>names[id] || id, byIssuer=new Map(), out=[];
    for(const line of lines || []){ const m=/^(?:u\d+:\s*)?([A-Z0-9.\-]+)\s+([A-Z_]+):\s*(.*)$/.exec(String(line)); if(m){ if(!byIssuer.has(m[1])) byIssuer.set(m[1],[]); byIssuer.get(m[1]).push([m[2],m[3]]); continue; } const n=/^(?:u\d+:\s*)?([A-Z0-9.\-]+) is not reviewed:\s*([A-Z_]+)$/.exec(String(line)); out.push(n ? html`${name(n[1])} ${t('is not reviewed')} · ${codeWords(n[2])}` : String(line)); }
    for(const [id,list] of byIssuer){ const words=new Set(list.map(([,why])=>gapWords(why))); if(list.length>1 && words.size===1) out.unshift(html`${name(id)} · ${t('{n} topics',{n:count(list.length)})} · ${[...words][0]}`); else for(const [topic,why] of list) out.unshift(html`${name(id)} · ${codeWords(topic)} · ${gapWords(why)}`); }
    return out;
  }
  // the packet's obligation as facts: the issuers of the axis, the required checks, the sources
  function obligationFacts(facts) {
    const o=facts?.research_obligation, sc=facts?.source_coverage; if(!o && !sc) return '';
    return kv([...(o ? [[t('Issuers'),entityNames(o.issuer_axis) || (o.issuer_axis || []).join(', ') || ''],[t('Required checks'),(o.required_checks || []).map(said).join(' · ') || ''],[t('Source families'),(o.approved_source_families || []).map(codeWords).join(' · ') || '']] : []),...(sc ? [[t('Sources'),html`${countText(sc.admitted_documents,'{n} admitted document','{n} admitted documents')} · ${t('{a} of {e} issuers available',{a:count(sc.available),e:count(sc.expected)})}${sc.missing ? html` · ${countText(sc.missing,'{n} missing','{n} missing')}` : ''}${sc.failed ? html` · ${countText(sc.failed,'{n} failed','{n} failed')}` : ''} · ${codeWords(sc.status)}`]] : [])]);
  }
  /* Each page's lede; the shared boundary (not a conclusion, not clearance, not permission to
   * trade) is said where it applies. */
  const LEDES={
    'evidence-stream':'What the workspace holds for this book\'s issuers, what the last check found at the source, and what a preparation would do; reading fetches nothing.',
    evidence:'The book and its state, the eight topics and the issuer × topic ledger, four layers of progress.',
    'evidence-reading':'Every passage one prepared packet delivered, one row each; a filter is a view and the coverage stays beside it.',
    handoff:'The Analyst\'s answer and the CRO\'s assessment over what the product composed for this book.',
    report:'The research report for this exact book and evidence version, with its limits and the delivery record. A delivered report is a readback, not permission to trade.'};
  const transport=(message)=>/^local_web\.service_(unreachable|answer_unreadable)/.test(String(message || ''));
  /* Books (N2, law 123): the Evidence Home -- every saved book the desk can read, one row each,
   * opening its overview; the fact that tells two alike apart is the book's reference (law 134). */
  // Discovery reads the Evidence owner's metadata summary in History; opening verifies it.
  /* Books (N2, law 123; F1, law 136): the Evidence Home as a lobby -- every saved book one row,
   * grouped by its state (the books at work open, the published folded) or by its time; one line:
   * its reference (what tells two alike apart), its name (law 134), its facts, its day; the row
   * opens its overview (the next step is the book's page's). A book whose state is not read yet groups apart; grouped by state a row keeps its
   * mark and the head says the word (not twice). */
  function booksPage() {
    const known=new Map(Data.history().filter(r=>r.raw?.book).map(r=>[key(r.raw.book),r]));
    const items=books().map(v=>{const r=known.get(key(v.selector)),summary=r?.raw?.book_summary;return {v,r,state:summary?.state==='NOT_READ' ? 'metadata' : null,word:t('Not read yet'),kind:v.selector.experiment_task_id ? 'authored' : 'installed'};});
    const stateWords=(b)=>b.word;
    const row=(b,d)=>{const on=(k)=>d.props[k]!==false, lead=d.group==='state' ? {lead:b.state ? statusDot(b.state,stateWords(b)) : ''} : {state:b.state};
      return objectRow({...lead,name:b.v.name,ref:b.v.ref,to:{page:'evidence',extra:{review_selector:JSON.stringify(b.v.selector)}}},{key:key(b.v.selector),word:b.word,columns:[...(!lead.state ? ['state'] : []),'kind','holdings','asof'],props:[...(!lead.state ? [''] : []),on('kind') ? ['',t(b.kind==='authored' ? 'authored book' : 'installed result'),'drop'] : '',on('holdings') && b.r?.holdingsSession ? html`${t('Holdings')} ${b.r.holdingsSession}` : '',''],time:b.r?.recordedAt || ''});};
    const axes=[{key:'state',label:t('State'),group:()=>({key:'metadata',label:t('Not read yet'),rank:0,open:true})},
      {key:'time',label:t('Time'),group:(b)=>timeGroup(b.r?.recordedAt || '')}];
    const lobby=items.length ? Lobby.render('books',{items,row,axes,words:(b)=>[b.v.name,b.v.ref,stateWords(b)].join(' '),placeholder:t('Name or reference'),
      filters:[{field:'kind',label:t('Kind'),multiple:true,options:[['authored',t('authored book')],['installed',t('installed result')]],test:(b,one)=>b.kind===one}],
      properties:[['kind',t('Kind'),false],['holdings',t('Holdings session')],['asof',t('As-of')]]})
      : emptyState(html`${t('No saved book yet')}${infoMark(t('A Portfolio study saves a book; its evidence is read here.'))}`,link(t('New experiment'),'lab','button primary'),'page-empty');
    return html`<div class="es-page rc-desk">${objectHead(t('Books'),html`<p class="lede">${t('The saved books whose evidence is read here; a book opens at its overview.')}</p>`)}${lobby}</div>`;
  }
  function page() {
    if(app.page==='books') return booksPage();
    const title=t(ROUTES[app.page]?.[1] || 'Evidence'); // law 87: the page's one word
    const unresolved=S.unresolved ? noteLine(t('Task scope unresolved'),html`${t('Task')} <span class="mono">${S.unresolved.task}</span> · ${codeWords(S.unresolved.kind)} · ${S.unresolved.reason}`,'warning',btn(t('Read again'),'review-work-refresh','','button compact')) : '';
    // without a book, every review page is the one banner with the way in, never an empty desk
    const bookless=S.status==='ready' && S.view?.state==='NO_BOOK_TO_REVIEW' && !S.selector;
    const unreached=(w)=>t(transport(S.error) ? 'The service could not be reached' : w);
    const body=S.status==='empty' && !S.selector ? html`${S.unresolved ? workSection(app.page) : noteLine(t('Choose a saved book'),t('No review or default book is inferred; the owner\'s default is read when a page is opened without one.'))}` : S.status==='loading' ? skeleton('body') : S.status==='error' ? notRead(unreached('Review not read'),S.error,'',btn(t('Read again'),'review-refresh','','button compact')) : bookless ? noBook() : S.view?.state==='EVIDENCE_AUTHORITY_NOT_ADMITTED' && !S.refusal && (['evidence-reading','handoff'].includes(app.page) || (app.page==='report' && !isExperimentBook() && !isUpdateBook())) ? notSetUp() : app.page==='evidence-reading' ? (S.refusal ? stateBanner() : readingPage()) : app.page==='evidence-stream' ? (S.refusal ? stateBanner() : sourcesPage()) : app.page==='handoff' ? (S.refusal ? stateBanner() : handoffPage()) : app.page==='report' ? reportPage() : entry();
    const notice=S.error && S.status==='ready' && !S.refusal ? notRead(unreached('The owner did not accept the last step'),S.error,explain(S.error),recoveryFor(S.error)) : '';
    const withBook=(S.view && (S.view.state!=='NO_BOOK_TO_REVIEW' || S.selector)) || S.refusal;
    const v=S.view, b=v?.book || S.preview?.book || null, id=withBook ? bookName() : '';
    const study=isExperimentBook() && S.selector?.experiment_task_id ? [[t('Study'),link(t('Portfolio study'),'portfolio','inline-link',{book:S.selector.experiment_task_id})]] : []; // F3 (law 135): the book's other home, its study
    // B1 (the book plan's items 5 and 7): the book's identity and its evidence's days on every tab -- as-of, expiry and, once a review is published, its completeness and day; the publication is the chooser at the line's end
    const reviewed=withBook && Boolean(v) && (reviewPublished() || historical());
    const update=b?.update_subject;
    const headFacts=withBook ? [[t(update ? 'Formation session' : 'trading|Session'),session()],[t('Book'),b?.authority ? codeWords(b.authority) : ''],...(update ? [[t('Observed through'),update.observed_through],[t('Entry session'),update.entry_session]] : []),...study,[t('Holdings'),b?.held_count==null? '' :count(b.held_count)],[t('As-of'),v?.evidence_as_of ? dayWord(v.evidence_as_of) : ''],[t('Expires'),v?.evidence_expires_at ? dayWord(v.evidence_expires_at) : ''],...(reviewed ? [[t('Completeness'),v.review_state ? codeWords(v.review_state) : ''],[t('Published'),publishedAt() || '']] : [])] : [];
    const tools=app.page==='handoff' ? handoffTools() : app.page==='report' ? [...(S.exportDoc ? [{ic:'file',action:'review-export',value:'json',word:t('Export review JSON'),why:t('The owner\'s exact export')},{ic:'file',action:'review-export',value:'html',word:t('Export review HTML'),why:t('The owner\'s exact export')}] : []),...(deliveryOf() ? [{ic:'file',action:'review-delivery-export',value:'html',word:t('Export delivery HTML'),why:t('The composed delivery')},{ic:'file',action:'review-delivery-export',value:'json',word:t('Export delivery JSON'),why:t('The composed delivery')}] : [])] : [];
    // the lane in transition (the same book's next reading, or its report, is on its way and what is painted is the previous one): dimmed after 300 ms and not interactive, never collapsed
    const transition=(S.status==='refreshing' && !viewShown()) || (app.page==='report' && Boolean(S.exportDoc) && S.exportDoc.hash!==(S.pin || workingPublication()));
    // E1: with a book the head has no lede (the Overview's status box, each view's own summary say the state); the page's description stands only while no book is read
    const c=withBook && S.view ? (app.page==='evidence-stream' ? storeWords() : app.page==='evidence-reading' ? readingWords() : app.page==='report' ? reportWords() : cycle()) : null; // the Sources page says the store's state, the Reading page its view, the others the book's (rounds F2, F3)
    const lede=html`<p class="lede">${c && c.sentence ? html`${c.sentence}${S.view?.explanation && c.tone==='warning' && !historical() ? html` <span class="muted">${said(S.view.explanation)}</span>` : ''}` : t(LEDES[app.page] || LEDES.evidence)}</p>`;
    // N3 (law 119): a book page's sentence is the path's (i) -- its state is the top row's, the Overview's its status box
    const viewInfo=withBook && c && c.sentence && !['evidence','handoff'].includes(app.page) ? lede : '';
    return html`<div class="es-page rc-desk work-cro">${objectHead(withBook ? bookTitle() : title,withBook ? viewInfo : lede,headActions(),!withBook ? '' : headState(),tools,{object:withBook,scope:withBook ? {name:bookTitle(),href:routeUrl('evidence',S.selector ? {review_selector:JSON.stringify(S.selector)} : {}),self:app.page==='evidence'} : null,id,facts:headFacts,top:withBook ? '' : selectorBar(),subject:withBook ? readingChooser() : '',switcher:withBook ? bookSwitch() : ''})}${LiveTasks.currentGroup?.() || ''}${unresolved}${notice}${app.page==='report' ? body : html`<div class="lane-split${S.item ? ' has-reading' : ''}"><div class="lane-main"${transition ? ' aria-busy="true"' : ''}>${body}</div>${readingPaneMarkup()}</div>`}</div>`;
  }
  /* A file import belongs to the role, book, prepared Task and draft it was started for: it
   * lands only if nothing was typed there meanwhile, else it is dropped with a notice. */
  async function file(f){
    if(!f || S.busy)return;
    const refuse=(over,read)=>{S.importRefusal={name:f.name,size:over.size,max:over.max,read};followDraft(S.drafts[S.role] || '');}; // said in place, in the editor's foot
    if(f.size>4*REQUEST_LIMIT)return refuse({size:f.size,max:answerMax(S.role)},false); // far past every bound even minified: not read
    // the import belongs to the draft as it was: its edit identity is a session-unique number that
    // a later edit, or leaving and revisiting the context, replaces for good
    const asked={key:S.key,task:S.packetTask,role:S.role,edit:S.draftEdit[S.role]};
    const text=await f.text();
    if(S.key!==asked.key || S.packetTask!==asked.task){S.editorNote=t('The imported file was for another book or prepared Task; it was not applied.');render();return;}
    const over=overLimit(asked.role,text);
    if(over)return refuse(over,true); // the JSON it would send, measured exactly
    if(S.draftEdit[asked.role]!==asked.edit){S.editorNote=t('The {role} draft was edited, or left and revisited, while the file was being read; the import was not applied.',{role:asked.role==='analyst' ? t('Analyst') : t('CRO')});render();return;}
    S.importRefusal=null;S.drafts[asked.role]=text;S.draftEdit[asked.role]=++S.editSeq;saveDrafts();S.pending=null;
    S.editorNote=asked.role!==S.role ? t('The file was imported into the {role} draft it was started for.',{role:asked.role==='analyst' ? t('Analyst') : t('CRO')}) : '';
    render();
  }
  // the page follows the draft as it is typed, without a repaint: every Submit (the form's, the top row's and its folded copy)
  // held or free with its reason as its tip, and the head's Format and Clear shown only while they have a draft to act on
  function followDraft(value){
    const off=draftOff(S.role);
    for(const b of $$('[data-action="review-submit-preview"]')){ if(b.classList.contains('menu-verb')){ if(off)b.setAttribute('aria-disabled','true'); else b.removeAttribute('aria-disabled'); } else b.disabled=Boolean(off); if(off)b.dataset.tip=off; else delete b.dataset.tip; }
    for(const why of $$('#reviewSubmitReason, #reviewTopReason'))why.textContent=off;
    const status=$('#reviewAnswerStatus');if(status)status.innerHTML=answerStatus(S.role);
    for(const x of $$('[data-draft-tool]'))x.hidden=x.dataset.draftTool==='format' ? !parses(value) : !value;
  }
  function formatDraft(){ const text=S.drafts[S.role]; let value; try{value=JSON.stringify(JSON.parse(text),null,2);}catch{notify(t('Not valid JSON: nothing was formatted.'));return;} reply(value);const el=$('#reviewReply');if(el){el.value=value;CodeEditor.input(el);} }
  function reply(value){if(S.busy)return;S.importRefusal=null;S.editorNote='';S.drafts[S.role]=value;S.draftEdit[S.role]=++S.editSeq;saveDrafts();S.pending=null;followDraft(value);const slot=$('.review-answer-preview-slot');if(slot)slot.innerHTML=previewSlot(S.role);const note=$('.review-draft-note');if(note){note.textContent=S.draftsNote;note.hidden=!S.draftsNote;}else if(S.draftsNote)render();}
  function downloadDoc(doc,format,name){if(!doc)return;download(format==='html' ? doc.value.html : doc.text,name+'.'+format,format==='html' ? 'text/html' : 'application/json');}
  return {pages,ensure,page,open,refresh,leaveReads,routeContext,collectionQuery,collectionPage,inspectSources,getBundle,confirm,commit,readExport,setItem,setFilter,clearFilter,addClause,next,evidenceReading,file,reply,formatDraft,observe,watch,dismissWork,openTask,spanOf,fold:(id)=>{if(S.folded.has(id))S.folded.delete(id);else S.folded.add(id);patchMain();},
    choose,working:()=>reread(''),selectPublication:v=>reread(v),
    // choosing a prepared Task is a new context: the pending confirmation, the read packet and
    // any in-flight packet answer are superseded, the drafts of the new context are loaded
    // the reader's own context changes are navigations too: a pending intent is superseded
    prepared:(v,unit='')=>{S.nav+=1;S.packetTask=v;S.packetUnit=v ? unit : '';S.pending=null;S.packet=null;S.bundleRev++;if(S.busy==='bundle')S.busy='';loadDrafts();adoptPacketRequest(S.view);pushRoute(routeUpdate());render();},
    useTask:(v,unit='')=>{S.nav+=1;S.packetTask=v;S.packetUnit=v ? unit : '';S.pending=null;S.packet=null;S.bundleRev++;if(S.busy==='bundle')S.busy='';S.role='analyst';loadDrafts();adoptPacketRequest(S.view);app.page='handoff';pushRoute(routeUpdate({page:'handoff'}));render();return getBundle('analyst');},
    step:page=>{if(!pages.has(page))return;S.nav+=1;app.page=page;pushRoute(routeUpdate({page}));render();queueMicrotask(ensure);},
    questions:()=>{S.nav+=1;S.questionsFocus=S.nav;app.page='handoff';pushRoute(routeUpdate({page:'handoff'}));render();queueMicrotask(ensure);}, // item 8: a Report question opens the Review at the open questions
    observeRoute,
    role:v=>{S.role=v==='cro' ? 'cro' : 'analyst';S.roleChosen=S.role;S.pending=null;render();},
    source:async h=>{const ticket=S.revision;if(!S.exportDoc)await readExport();if(ticket===S.revision)setItem(h);},
    dismissConfirmation:()=>{S.pending=null;},dismissRefusal:()=>{S.lastRefusal=null;render();},settleLegacy,settleUnitless,clearDraft:()=>{S.drafts[S.role]='';S.draftEdit[S.role]=++S.editSeq;saveDrafts();S.pending=null;render();},
    reconcile:async()=>{await Promise.all([Data.refreshHistory(),refresh()]);},
    taskFinished:id=>{if(S.work?.task===id)void readWork(true);},readWork:()=>S.unresolved ? openTask(S.unresolved.task) : readWork(false),
    exportReport:f=>downloadDoc(S.exportDoc,f,'AlphaLattice-review'),exportBundle:v=>downloadDoc(v==='cro' || (!v && S.role==='cro') ? S.dossier : S.packet,'json','AlphaLattice-'+(v==='cro' || (!v && S.role==='cro') ? 'cro-dossier' : 'analyst-packet')),
    documentsPage,openAnswer,assembleDelivery,deliveryOptions,deliveryDialog,reportFacts,handoffFacts,sourcesFacts,handoffRecord,boundTasks,ledgerTopic:(v)=>{S.ledgerTopic=String(v || '');patchMain();},ledgerGroup:(v)=>{const k=String(v || '');if(S.ledgerOpen.has(k))S.ledgerOpen.delete(k);else S.ledgerOpen.add(k);patchMain();},ledgerPage:(v)=>{S.ledgerPage=Math.max(0,S.ledgerPage+(v==='prev' ? -1 : 1));patchMain();},readView,readPart,continueDialog,continueLimit,viewFilter,usePacket,readCell,readExcerpts:()=>getBundle('analyst',null,true),spanQuery:(v)=>{S.spanQuery=String(v || '');S.spanPage=0;if(typeof patchMain==='function')patchMain();},spanPage:(v)=>{S.spanPage=Math.max(0,S.spanPage+(v==='prev' ? -1 : 1));patchMain();},ledgerQuery:(v)=>{S.ledgerQuery=String(v || '');S.ledgerPage=0;if(typeof patchMain==='function')patchMain();},sourceQuery:(v)=>{S.sourceQuery=String(v || '');S.sourcePage=0;if(typeof patchMain==='function')patchMain();},sourcePage:(v)=>{S.sourcePage=Math.max(0,S.sourcePage+(v==='prev' ? -1 : 1));patchMain();},issuerQuery:(v)=>{S.issuerQuery=String(v || '');S.issuerPage=0;if(typeof patchMain==='function')patchMain();},issuerState:(v)=>{S.issuerState=['reviewed','held'].includes(v) ? v : '';S.issuerPage=0;patchMain();},issuerPage:(v)=>{S.issuerPage=Math.max(0,S.issuerPage+(v==='prev' ? -1 : 1));patchMain();},findingPage:(v)=>{S.findingPage=Math.max(0,S.findingPage+(v==='prev' ? -1 : 1));patchMain();},findingMore:(h)=>{const k=String(h || '');if(S.findingOpen.has(k))S.findingOpen.delete(k);else S.findingOpen.add(k);patchMain();},runPage:(v)=>{S.runPage=Math.max(0,S.runPage+(v==='prev' ? -1 : 1));patchMain();},runsAll:()=>{S.runsOpen=true;patchMain();},packetPage:(v)=>{S.packetPage=Math.max(0,S.packetPage+(v==='prev' ? -1 : 1));patchMain();},unitPage:(v)=>{S.unitPage=Math.max(0,S.unitPage+(v==='prev' ? -1 : 1));patchMain();},checksPage:(v)=>{S.checksPage=Math.max(0,S.checksPage+(v==='prev' ? -1 : 1));patchMain();},citePage:(v)=>{const [id,dir]=String(v || '').split('|');S.citePages.set(id,Math.max(0,(S.citePages.get(id) || 0)+(dir==='prev' ? -1 : 1)));patchMain();},execution:()=>{S.executionOpen=!S.executionShown;patchMain();},storageAgain:()=>{S.storage=null;patchMain();},issuerDocs:(v)=>{const k=String(v || '');if(S.docsOpen.has(k))S.docsOpen.delete(k);else S.docsOpen.add(k);patchMain();},cycle,storeWords,readingWords,readingFacts,viewClear,reviewWords,nameRisk,reportWords,readLedgers,exportDelivery:f=>downloadDoc(S.delivery,f,'AlphaLattice-delivery'),
    reopenDelivery:()=>{const r=S.delivery?.value?.next_requests?.reopen;if(!r)return;S.risk=r.risk_report_hash || '';S.comparison=r.right_task_id && r.right_task_id!==r.task_id ? r.right_task_id : (r.left_task_id && r.left_task_id!==r.task_id ? r.left_task_id : '');S.question=r.delivery_question || '';S.delivery=null;render();return assembleDelivery();},
    selectRisk:v=>{S.risk=v;S.delivery=null;},selectComparison:v=>{S.comparison=v;S.delivery=null;},setQuestion:v=>{S.question=v;S.delivery=null;},
    proof:()=>html`${kv([[t('Book'),bookName()],[t('Holdings date'),session()],[t('Review publication'),S.pin || workingPublication() || '']])}<p>${said(S.view?.explanation || S.refusal || '')}</p>${S.view ? kv([[t('Evidence as-of'),S.view.evidence_as_of || ''],[t('Evidence expiry'),S.view.evidence_expires_at || ''],[t('Reviewer provenance'),attributionWords(S.view.review_attribution || [],{facts:true}) || '']]) : ''}`,
    context:()=>({selector:S.selector,review_publication_hash:S.pin || workingPublication() || null,pinned:historical(),prepared_task:S.packetTask,prepared_unit:S.packetUnit,work:S.work?.task || null}),
    facts:()=>({reading:S.reading ? {status:S.reading.status,code:S.reading.code || '',selected:evOf()?.selected ?? null} : null,excerpts:S.excerpts.size,packetPart:S.packetDelivery?.part ?? null,pending:S.pending ? {operation:S.pending.operation,payload:{...S.pending.payload}} : null,ledgers:[...S.ledgers.entries()].map(([k,l])=>[k,l.status,l.code || '',l.coverage ? (l.coverage.cells || []).length : 0]),state:S.view?.state || null,refusal:S.refusal,error:S.error,busy:S.busy,pinned:historical(),publication:S.pin || workingPublication() || '',nav:S.nav,readGen:S.readGen,dossierRead:S.dossier ? S.dossier.readGen : null,taskFacts:Object.fromEntries(S.taskFacts),packetTask:S.packetTask,packetUnit:S.packetUnit,packet:S.packet ? {key:S.packet.key,task:S.packet.task,unit:S.packet.unit || ''} : null,pending:S.pending ? {path:S.pending.path,kind:S.pending.kind,operation:S.pending.operation || '',payload:S.pending.payload,task:S.pending.task,unit:S.pending.unit ?? ''} : null,finding:S.finding ? {key:S.finding.key,handle:S.finding.handle} : null,exportRequest:S.exportRequest ? S.exportRequest.request : null,unitless:S.unitless ? {id:S.unitless.id,analyst:S.unitless.analyst,cro:S.unitless.cro} : null,dossier:S.dossier ? {key:S.dossier.key} : null,role:S.role,drafts:{...S.drafts},draftsPersisted:S.draftsPersisted,draftsNote:S.draftsNote,editorNote:S.editorNote,held:Object.fromEntries([...S.held].map(([k,h])=>[k,{analyst:h.analyst,cro:h.cro}])),legacy:S.legacy ? {key:S.legacy.key,analyst:S.legacy.analyst,cro:S.legacy.cro} : null,refusalKept:S.lastRefusal ? {role:S.lastRefusal.role,kind:S.lastRefusal.kind,code:S.lastRefusal.code,fields:S.lastRefusal.fields.length} : null,work:S.work ? {task:S.work.task,kind:S.work.kind,lifecycle:S.work.view?.lifecycle || null,done:S.work.done,book:S.taskBooks.get(S.work.task) || null,parallel:workView()?.parallel || null,current_scope:S.work.view?.current_scope || null} : null,unresolved:S.unresolved,bound:Object.fromEntries([...S.bound].map(([k,b])=>[k,{prepare:[...b.prepare],analysis:[...b.analysis],review:[...b.review],imported:[...b.imported]}]))}),
    // B1: the book's tabs count their objects where the owner records them -- the documents the inventory holds, the published review's citations, the findings
    counts:()=>{ const v=S.view, inv=inventory(), d=dossierOf(), published=Boolean(v) && (reviewPublished() || historical());
      return {documents:inv ? inv.rows.reduce((a,r)=>a+r.held,0) : undefined,citations:published && Array.isArray(v.citations) ? v.citations.length : undefined,findings:Array.isArray(d?.findings) ? d.findings.length : published && Array.isArray(v.issuer_rows) ? v.issuer_rows.reduce((a,r)=>a+(Number(r.finding_count) || 0),0) : undefined}; },
    overviewFacts,pctText,pct};
})();
