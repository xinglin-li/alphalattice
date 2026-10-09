/* Workspace interaction over existing authenticated operations. No data/science authority here.
 *
 * Five pages over the workspace's owners. The welcome page is the preparation scene: before a
 * Task exists it says what one explicit preparation would do and offers the owner's preview;
 * once a Task is recorded it shows that Task the way its owners describe it, and once the input
 * is verified, the exact input and the one next step. The data page is the returning user's
 * scene: the workspace's freshness in the owner's own cutoffs, one explicit incremental update
 * (previewed, confirmed, then observed in the shared work area with maintenance's own stages
 * and units), and the way to the input versions. Issues, inputs and storage explain the data
 * owner's decisions, the immutable input versions and the bounded storage in the owners' facts.
 * Reading here starts nothing; a scene's readback and recovery view are re-read on the activity
 * cadence while its Task moves, and the page is repainted in place. */
/* The preparation's five steps as the design lists them; `marks` (stage id -> {lifecycle,
 * evidence}) is Task Control's record of each stage; without it the list is the plain roster.
 * Five drawn steps are never five equal durations. */
const PREPARATION_STEPS = ['freeze_sources', 'prepare_data', 'prepare_features', 'publish_inputs', 'verify_inputs'].map((id) => [id, STAGES[id].word, STAGES[id].note]);
const LiveWorkspace = (() => {
  const pages = new Set(['data', 'issues', 'inputs', 'storage']);
  // The preparation scene's key is `welcome` (its reads, its route param, its kept area); the
  // page that hosts it is the Home (round 74), so the scene reads `here()` where a page is meant.
  const here=()=>app.page==='overview' ? 'welcome' : app.page;
  const reads = {welcome:'/api/workspace/preparation', data:'/api/data-update',
    issues:'/api/workspace/data-issues', inputs:'/api/research-inputs', storage:'/api/workspace/storage'};
  // a request the owner composed goes to its operation's route as the session names it (U13)
  function requestStep(request, expectedOperation) {
    if(!request || request.operation!==expectedOperation || !Data.offers(expectedOperation))
      throw Error(t('Data request is not available. Refresh this page.'));
    const {operation,...payload}=request;
    return [Data.route(operation),payload];
  }
  async function previewIssue(request) {
    try { return await Data.post(...requestStep(request,'DATA_ISSUE_PREVIEW')); }
    catch(error) {
      if(error.body?.failure_code!=='feature_input.case_no_longer_pending')throw error;
      await refresh('issues');
      S.error=explain(error.body.failure_code);
      setNotice('Data issues changed; review the current cases before choosing again.');
      return null;
    }
  }
  const states = new Map();
  const S = {busy:false, pending:null, revision:0, family:'', options:new Map(), notice:'', noticePage:'', error:'', observing:null, watch:null};
  const noticeFor=(page)=>S.notice && S.noticePage===page ? S.notice : '';
  const W = LiveWorkArea, {clockText, readAge, staleFor, clockOf, shownStageOf} = W;
  const readWhen = (state) => when(new Date(state.readAt || Date.now()).toISOString()); // the last successful read, as an instant (law 133)
  /* The Task a scene tells: the route's `preparation` (welcome) or `update` (data) -- an explicit
   * choice, kept until another is chosen -- or none: a bare entry, which discovers the latest
   * Task of that kind and pins it in place so a later Task does not replace it silently. */
  const SCENE_ROUTE={welcome:'preparation',data:'update'};
  const selectedTask=(page)=>SCENE_ROUTE[page] && typeof hashParams==='function' ? (hashParams().get(SCENE_ROUTE[page]) || null) : null;
  const pinTask=(page,id)=>{ if(typeof replaceHash!=='function' || here()!==page) return false; replaceHash({[SCENE_ROUTE[page]]:id}); return true; };
  const REFUSALS={
    'workspace_preparation.task_not_found':'No preparation Task has this id in this workspace.','workspace_preparation.task_kind_mismatch':'This Task is not a workspace preparation; it has its own page.',
    'workspace_data_update.task_not_found':'No data update Task has this id in this workspace.','workspace_data_update.task_kind_mismatch':'This Task is not a data update; it has its own page.'};
  const refusalOf=(message)=>Object.keys(REFUSALS).find(code=>String(message || '').startsWith(code)) || null;
  const notConfigured=state=>String(state?.error || '').startsWith('workspace_data_update.not_configured');
  const stringify = v => v == null ? '' : typeof v === 'object' ? JSON.stringify(v) : String(v);
  const facts = rows => kv(rows.map(([k,v]) => [t(k),stringify(v)]));
  const detail = (b, title='Exact identity, scope and receipt') => html`${codeRef(t(title), b)}`;
  const said = (code) => code ? html`<span class="coded" data-tip="${code}">${codeWords(code)}</span>` : ''; // N6 (law 80): an owner's code in words, the code on hover
  const action = (label,name,value='',reason='',cls='button compact') => typedBtn(t(label),'workspace-'+name,value,cls,S.busy?t('Waiting for the product owner'):reason);
  const primary = (label,name,value='',reason='') => typedBtn(t(label),'workspace-'+name,value,'button primary',S.busy?t('Waiting for the product owner'):reason);
  const task = b => b?.task_id ? action('Inspect task','task',b.task_id) : '';
  const get = page => states.get(page)?.body;
  const readView = id => Data.readShared('/api/tasks/recovery?'+new URLSearchParams({task_id:id}));
  const dateOr=(v)=>v || '';
  function invalidate() { S.revision++; S.pending=null; S.error=''; }
  const setNotice=(text)=>{ S.notice=text || ''; S.noticePage=here(); };
  /* The owners' typed codes, in words a reader can act on. The code stays beside the text. */
  const CODES={
    'local_web.service_unreachable':'The service did not answer: nothing was read or sent. It may be stopped or restarting; read again once it answers.',
    'local_web.service_answer_unreadable':'The service answered with something other than its JSON: nothing is inferred from it. Read again once it answers.',
    'workspace_maintenance.manifest_update_consent_required':'The membership check found a changed candidate manifest (index members moved), and the data owner holds it as a pending transition: it asks for a new update plan that consents to it before any fetch. Preview an update; the transition it names is what you confirm.',
    'workspace_maintenance.initialization_consent_required':'The workspace is not initialized for research: the preparation on the Workspace page is the consent the data owner asks for before any update.',
    'workspace_data_update.not_configured':'This workspace has no maintained data yet: a daily update maintains a prepared workspace. One explicit preparation makes it usable; nothing starts by opening this page.',
    'data.truth_review_required':'A data decision is needed before this Task can continue: the data owner found an unexplained move and asks for a permitted response, which the agent takes under the first use\'s delegation or asks the person for in one line.',
    'workspace_data_update.source_access_not_admitted':'Source access is not admitted in this service: the configured provider cannot be reached from here. Nothing was fetched.',
    'data.full_history_audit_approval_required':'A full-history audit of some listings needs explicit approval: preview the update again; the owner names the listings and asks for confirmation.',
    'workspace_data_update.retry_not_due':'The owner\'s retry time has not come yet; continuing is refused until it is due.',
    'workspace_preparation.retry_not_due':'The owner\'s retry time has not come yet; resuming is refused until it is due.', // U63 (V375)
    'data.rate_limited':'The market data provider limited the requests; the work waits and resumes from every listing already fetched.',
    'data.provider_timeout':'The market data provider did not answer in time, after its retries: a listing that timed out is recorded as failed in this run while the others go on, and a universe source that timed out stops the step. No retry time is set; continue once the provider answers.', // U65 (V396): no owner defers on a timeout
    'data.provider_session_unstable':'The market data provider\'s session was unstable; the work waits and resumes from every listing already fetched.',
    'workspace_data_update.stale_plan':'The workspace moved beneath this plan; preview the update again.',
    'workspace_data_update.membership_review_required':'The membership the Panel admits changed during this update: plan again against the new membership.',
    'workspace_maintenance.cancelled_at_safe_checkpoint':'The update was cancelled at a safe checkpoint; listing work already done stays with the maintenance operation.',
    'feature_input.case_no_longer_pending':'This case is no longer pending; the owner reassessed it.',
    'feature_input.case_expired_reassess_required':'The options of this case expired; the owner must reassess it (run the update again) before a decision is confirmed.',
    'feature_input.option_changed':'The option changed since it was shown; preview the decision again.',
    'feature_input.resolution_conflict':'A different decision is already recorded for this case.',
    'feature_input.resolution_policy_changed':'The decision policy changed since this case was raised; the owner must reassess it.',
    'research_input.source_changed_preview_again':'The prepared data changed since this publication was previewed; preview it again.',
    'research_input.publication_plan_stale':'Another version was published since this preview; preview again.',
    'research_input.qualified_data_update_required':'The workspace is not research-ready; a qualified data update comes first.',
    'storage.input_pin_unavailable':'This version cannot be retained: its source files are not present.',
  };
  const explain=(code)=>code ? (CODES[code] ? t(CODES[code]) : '') : '';

  /* `quiet` is the scene's own re-read on the activity cadence: no loading banner, no full
   * render; the page is patched in place with its reading state kept. */
  async function refresh(page=here(), cursor='', quiet=false) {
    if (!reads[page]) return;
    if(!quiet) invalidate();
    const previous=states.get(page);
    const wanted=selectedTask(page);
    // What stays on the page while it reads again are the same Task's facts; a read for
    // another Task (or for the latest, after a chosen one) starts from nothing, so a failed or
    // late answer is only ever kept, marked or shown as the Task it was asked for.
    const same=!SCENE_ROUTE[page] || (previous?.requested ?? '')===(wanted || '');
    const state={body:same ? get(page) : null,view:same ? previous?.view || null : null,viewError:'',readAt:same ? previous?.readAt || 0 : 0,stale:same ? previous?.stale || null : null,loading:!quiet,error:'',requested:wanted || '',lateReads:same ? previous?.lateReads || 0 : 0};states.set(page,state);
    state.decisionStamp=previous?.decisionStamp;
    if(!quiet) render();
    try {
      const query=cursor ? {history_cursor:cursor} : wanted ? {task_id:wanted} : null;
      const body=await Data.readShared(reads[page]+(query ? '?'+new URLSearchParams(query) : ''),true);
      if (states.get(page)!==state) return;
      state.body=body;
      if(page==='welcome') {
        // Workspace discovery (the notice on other pages, the bare-entry routing) is the
        // latest preparation's story; reading an earlier Task on request changes nothing there.
        if(!body.selected || !body.latest_task_id || body.task_id===body.latest_task_id) Data.setPreparation(body);
      }
      if(page==='data' && body.inputs) { app.data=body.inputs.data_through || app.data; app.feature=body.inputs.panel_through || app.feature; }
      if(SCENE_ROUTE[page]) {
        // The recovery view of the readback's Task; if that read fails the previous view is
        // kept as the last observation and the scene says it is not current.
        let view=null, viewError='';
        if(body.task_id) { try { view=await readView(body.task_id); } catch(e) { viewError=e.message; } }
        if (states.get(page)!==state) return;
        if(body.task_id && !view && state.view?.task_id===body.task_id) { state.viewError=viewError; state.stale=notCurrent(state,viewError); }
        else { state.view=view; state.viewError=viewError; state.recovered=Boolean(state.stale); state.stale=null; state.readAt=Date.now(); }
        if(body.task_id) {
          if(!wanted && pinTask(page,body.task_id)) state.requested=body.task_id; // discovered once, then explicit (what the route now says)
          const spec=SCENES[page];
          // The answer is absorbed into this scene and Task's state: the visible area when
          // this page is shown, otherwise the kept state -- a late answer to a page the
          // reader has left never takes over the area another scene is showing.
          const area=here()===page ? W.areaFor(spec,body.task_id) : W.entryFor(spec,body.task_id);
          // A completion observed by this page (moving at the previous read, verified now) is
          // accented once, never on a first paint.
          const wasMoving=previous?.view?.task_id===body.task_id && stateMoving(previous.view.lifecycle);
          if(wasMoving && view?.lifecycle==='SUCCEEDED' && spec.completedNow(body)) area.accent=true;
          spec.absorb(body,state,area);
          W.noteTransition(view,area);
        }
        // The verified input's recorded cutoff and file presence come from the input owner.
        if(page==='welcome' && body.inputs?.length && !get('inputs')) await refresh('inputs','',true);
      }
      state.loading=false;
      // N6: the Data overview's figures read the cases and the storage beside the update, quietly --
      // on an explicit read, and again when the update's lifecycle moved (a stop raises a case)
      if(page==='data' && (!quiet || !states.get('issues') || state.view?.lifecycle!==previous?.view?.lifecycle)) { void refresh('issues','',true); if(!quiet || !states.get('storage')) void refresh('storage','',true); }
      // Data issues' next page joins the cases read before it (F2: the lobby holds every case read)
      if(page==='issues' && cursor && previous?.body?.issues) body.issues=[...previous.body.issues.filter(x=>!(body.issues || []).some(y=>y.case.case_token===x.case.case_token)),...(body.issues || [])];
      if(page==='issues')S.options=new Map([...S.options].filter(([key])=>body.issues?.some(v=>v.case.case_token===key)));
      if(page==='inputs')Data.setInputs(body);
      if(page==='inputs'&&!S.family)S.family=body.inputs?.[0]?.input_id || '';
    } catch(e) {
      if(states.get(page)!==state) return;
      state.loading=false;
      // A read that fails after a successful one keeps what was read and marks it as not
      // current, whether the scene re-read on its cadence or the person pressed Refresh:
      // `quiet` chooses how the page is drawn, never whether the retained facts are current.
      // The owners were not reached, and nothing about the Task is inferred from that.
      // A typed refusal of the chosen Task is not an outage: nothing of another Task stays
      // on the page as if it were the chosen one.
      if(SCENE_ROUTE[page] && refusalOf(e.message)) { state.body=null; state.view=null; state.stale=null; state.error=e.message; }
      else if(SCENE_ROUTE[page] && state.body) state.stale=notCurrent(state,e.message); else state.error=e.message;
    }
    // Backups have independent read authority; an inventory refusal cannot hide recovery.
    if(page==='storage' && !quiet) void readBackups(); // U48: reading makes none
    if(quiet) patchMain(); else render();
  }
  /* The page's own fact about its reads: since when the owners have not answered, how many
   * reads failed and the last refusal. It is never a Task fact and never changes one. */
  const notCurrent=(state,error)=>({since:state.readAt || Date.now(),error,failures:(state.stale?.failures || 0)+1,at:Date.now()});
  /* Called on the activity cadence: while a scene's Task moves (or its view is not read yet),
   * the readback and the recovery view are read again; nothing is read on another page. The
   * Task's projection as the activity feed and the Task Center keep it counts too: a resume or
   * a continuation confirmed elsewhere moves the Task before this scene has read it again. */
  function observe() {
    if(S.busy || S.observing) return;
    if(!SCENE_ROUTE[here()]) return settleWatched();
    const page=here()==='data' && notConfigured(states.get('data')) ? 'welcome' : here(), state=states.get(page);
    if(!state?.body || state.loading) return;
    const issueState=states.get('issues'), tokens=new Set((issueState?.body?.issues || []).map(x=>x.case.case_token)), activity=typeof LiveActivity==='undefined' || !LiveActivity.state || !LiveActivity.retained ? null : LiveActivity.state();
    const ordinal=activity?.epoch && !activity.stale && !activity.error ? Math.max(0,...LiveActivity.retained().flatMap(g=>g.items || []).filter(x=>x.schema_kind==='ProductOperationObserved' && x.payload?.operation==='DATA_ISSUE_CONFIRM' && x.payload.phase==='RETURNED' && x.payload.status==='CONFIRMED_PENDING_REVALIDATION' && tokens.has(x.payload.subject?.data_issue_case_token)).map(x=>x.ordinal)) : 0;
    const stamp=ordinal ? activity.epoch+':'+ordinal : '';
    if(stamp && issueState.decisionStamp!==stamp) {
      issueState.decisionStamp=stamp;
      const origin=here();
      S.observing=Data.refreshDecisions().then(()=>{if(here()===origin)return refresh('issues','',true);}).finally(()=>{S.observing=null;});
      return;
    }
    // A preparation admitted elsewhere can arrive after Home read an empty workspace.
    if(!state.body.task_id && !(page==='welcome' && !state.body.inputs?.length && !selectedTask(page) && Data.tasks().some(x=>x.task_id && x.task_kind==='workspace_preparation'))) return;
    const v=state.view, projection=Data.tasks().find(x=>x.task_id===state.body.task_id);
    // A projection that no longer agrees with the view (a resumed Task already finished between
    // two reads, a stage verified) is a reason to read the owners again, moving or not.
    const differs=Boolean(v && projection && (projection.lifecycle!==v.lifecycle || projection.verified_stage_count!==v.verified_stage_count));
    // The readback and the recovery view are two reads: a Task that completed between them
    // leaves a verified view beside a readback without its receipt. That is read again on
    // the cadence until the owner's readback says so too, a bounded number of times.
    const late=Boolean(v && v.lifecycle==='SUCCEEDED' && !SCENES[page].completedNow(state.body) && state.lateReads<3);
    if(v && !stateMoving(v.lifecycle) && !v.operation_running && !(projection && stateMoving(projection.lifecycle)) && !differs && !late) return;
    if(late) state.lateReads+=1;
    S.observing=refresh(page,'',true).finally(()=>{S.observing=null;});
  }

  /* A Task admitted or continued from a page without its own scene (a publication from the
   * inputs page, a continuation from the issues page): once the activity feed's projection shows
   * it settled, that page is read again once -- for a publication, its readback (the receipt)
   * first -- and the admission notice gives way to what the Task became. Nothing is read while
   * it moves, and nothing for a Task this page did not admit. Answers whether it started a read. */
  function settleWatched() {
    const watch=S.watch;
    if(!watch || here()!==watch.page) return false;
    const projection=Data.tasks().find(x=>x.task_id===watch.task);
    if(!projection || stateMoving(projection.lifecycle)) return false;
    S.watch=null;
    S.observing=(async()=>{
      if(watch.page==='inputs') { try { states.set('capture',{body:await Data.readShared('/api/research-inputs/readback?'+new URLSearchParams({task_id:watch.task}),true)}); } catch(e) { /* the versions below say what was published */ } }
      await refresh(watch.page,'',true);
      if(S.noticePage===watch.page) { S.notice=t('Task {id} {word}; the page below was read after it settled.',{id:short(watch.task),word:codeWords(projection.lifecycle).toLowerCase()}); if(here()===watch.page) patchMain(); }
    })().finally(()=>{S.observing=null;});
    return true;
  }
  /* Entering a page without its own scene again (the router says the page changed) reads the
   * owner again, quietly, behind what was read before: an update that stopped for a decision or
   * a publication that settled while another page was read is shown as it is now, without
   * pressing Refresh. A first entry reads through `page()`; the scenes keep their own rule. */
  function entered() {
    const name=here(), state=states.get(name);
    if(name==='data' && notConfigured(state)) void refresh('issues','',true);
    if(!pages.has(name) || SCENE_ROUTE[name] || !state || state.loading || S.observing) return;
    if(!settleWatched()) void refresh(name,'',true);
  }
  /* ---- confirmations: the owner's preview, said in its own terms, then one explicit confirm ---- */
  function offer(title, preview, steps, page=here(), kind='') {
    S.pending={title,preview,steps,page,revision:S.revision,kind};
    const preparing=steps.some(([p])=>p==='/api/workspace/preparation/confirm');
    const summary=preparing ? preparationSummary(preview) : kind==='update' ? updateSummary(preview) : kind==='capture' ? captureSummary(preview) : kind==='issue' || kind==='delegate' ? issueSummary(preview,kind==='delegate') : kind==='cleanup' ? cleanupSummary(preview) : kind==='pin' ? pinSummary(preview) : kind==='backup' ? backupSummary(preview) : kind==='continue' || kind==='continue-update' || kind==='continue-preparation' ? continueSummary(preview) : [['Status',preview.status],['Task',preview.task_id]];
    if(kind==='delegate') summary.push(['Task',preview.preparation_task_id]);
    if(kind==='revoke-delegation') summary.splice(0,summary.length,
      ['Grant',short(preview.grant_hash,SHORT.hash)],['Task',preview.preparation_task_id],
      ['Permitted response',optionText({option_id:preview.option_id})],['Expires',when(preview.expires_at)]);
    const CAPTIONS={cleanup:'Cleanup removes only the approved unrooted files. Metadata remains, but removed input files cannot be replayed.',update:'Confirming admits one update Task through the data owner. It may reach the configured source for the target session; saved research, published input versions and every selection stay unchanged. Nothing is downloaded before this confirmation, and an unchanged workspace is reused exactly rather than rebuilt.',capture:'Confirming admits one publication Task: the prepared data is sealed as a separate immutable version. Existing versions, configured defaults and saved research stay unchanged; selecting the new version is a further explicit step.',issue:'Confirming records your decision for this case. The data owner re-evaluates the source evidence when the update continues; a decision is not proof of qualified data.',delegate:'This Human authorization permits an external executor to confirm only the displayed case, evidence, option and target for the named blocked preparation Task. No effect is applied by issuing it; the executor remains EXTERNAL_AUTOMATION and the owner revalidates at confirmation.',pin:'Retention only decides what cleanup may release; nothing is deleted or downloaded by this change.',backup:'Backing up copies the held state outside the workspace as one new generation, sealed by its hash; nothing in the workspace changes. The owner keeps its newest generations and removes older ones.'};
    const caption=kind==='revoke-delegation' ? 'Revoking stops future delegated confirmations and continuations; recorded decisions and admitted Tasks are not undone.' : preparing ? 'Confirming admits one Task through the preparation owner. It may access the declared sources; saved research, defaults and selections stay unchanged, and nothing else starts.' : CAPTIONS[kind] || 'Only the displayed operation is sent; nothing else is planned or run.';
    openDialog(t('Workspace · explicit confirmation'),t(title),html`${kv(summary.filter(([,v])=>v!=null).map(([k,v])=>[t(k),typeof v==='string'||typeof v==='number'?String(v):v]))}${detail(preview)}<p class="caption">${t(caption)}</p>`,
      html`${action('Confirm','commit','','','button primary')}`);
  }
  /* The preparation preview in the owner's terms: scope, reuse versus new work, what may use
   * the network, the storage estimate it does not have yet, and what stays unchanged. */
  const SOURCE_MODES = {
    ACQUIRE_APPROVED_SOURCES: 'New work: the declared sources are accessed after confirmation and ten years of daily bars are acquired for every current member.',
    REUSE_QUALIFIED_LOCAL_DATA_NO_DOWNLOAD: 'Reuse: the qualified local market data and Features are reused; nothing is downloaded.',
    REUSE_CAPTURED_SOURCES_REVALIDATE_LOCAL_STATE: 'Continuation: the sources captured by the earlier Task are reused and the local state is revalidated; nothing is captured again.',
  };
  function preparationSummary(p) {
    const rows=[['Target session',p.target_session],['Work',t(SOURCE_MODES[p.source_mode] || p.source_mode || '')]];
    if(p.initial_history_years) rows.push(['Initial history',t('{n} years of daily bars before the target session',{n:p.initial_history_years})]);
    if(p.universe) rows.push(['Universe',p.universe]);
    if(p.quality) rows.push(['Quality thresholds',t('missing ratio at most {r} · at most {n} consecutive missing sessions',{r:p.quality.maximum_missing_ratio,n:p.quality.maximum_consecutive_missing_sessions})]);
    if(p.sources) rows.push(['Sources',html`${p.sources.map(v=>html`<span class="mono">${v}</span><br>`)}`]);
    rows.push(['Network',t(p.source_mode==='REUSE_QUALIFIED_LOCAL_DATA_NO_DOWNLOAD'?'None: no source is accessed.':'The declared sources, only after this confirmation. No data API key is requested.')]);
    rows.push(['Storage',t(p.candidate_count_basis==='KNOWN_AFTER_SOURCE_CAPTURE'?'Estimated by the owner after the sources are captured (the candidate count is known then); not known at this preview.':'Estimated by the owner from the retained scope.')]);
    if(p.resume_from_cancelled_task) rows.push(['Continues from',html`${t('the cancelled Task')} <span class="mono">${short(p.resume_from_cancelled_task)}</span>`]);
    rows.push(['Unchanged',t('Saved research, configured defaults and every selection. No Foundation or strategy is activated.')]);
    return rows;
  }
  /* The update preview: the target session against the cutoffs the workspace holds now, what
   * the owner will do, what may use the source, what it found due, and any change that needs
   * this confirmation (a membership transition or a full-history audit). */
  function updateSummary(p) {
    const before=p.inputs || {}, change=p.change;
    const rows=[['Target session',p.target_session],['Market data now through',dateOr(before.data_through)],['Features now through',dateOr(before.panel_through)]];
    rows.push(['Work',t(p.target_session && before.data_through && p.target_session<=before.data_through ? 'The workspace already holds the target session: the owner checks the due sources and reuses the unchanged data and Features exactly; no Task is admitted when nothing changed.' : 'Fetch the sessions after the current cutoff for every admitted member, maintain the Features and the Panel for them, publish one update receipt; no Factor research.')]);
    rows.push(['Source access',t(p.source_access==='HOST_SUPPLIED_PROVIDER' ? 'The configured provider, after this confirmation; no data API key is requested.' : 'Network access must be admitted explicitly for this service; the owner stops by name otherwise.')]);
    rows.push(['Membership sources',t(p.source_check_due ? 'Due: the current index memberships are read again before the fetch.' : 'Not due: the last membership check stands.')]);
    if(p.candidate_recheck) rows.push(['Candidate recheck',countText(p.candidate_recheck.listing_ids?.length ?? '','{n} listing due for its Feature recheck','{n} listings due for their Feature recheck')]);
    if(p.candidate_data_recheck) rows.push(['Data recheck',countText(p.candidate_data_recheck.listing_ids?.length ?? '','{n} failed candidate due for a raw retry','{n} failed candidates due for a raw retry')]);
    if(change) rows.push(['Change requiring confirmation',change.action==='UNIVERSE' ? t('Membership transition: {a}, {r}; the entrants are hydrated and qualified before they enter.',{a:countText(change.additions?.length ?? 0,'{n} addition','{n} additions'),r:countText(change.removals?.length ?? 0,'{n} removal','{n} removals')}) : t('Full-history audit of {n}: {labels}',{n:countText(change.full_history_listing_ids?.length ?? 0,'{n} listing','{n} listings'),labels:(p.audit_labels || []).join(', ') || ''})]);
    rows.push(['Unchanged',t('Published input versions, saved research and every selection. Foundation research is retained as it is.')]);
    return rows;
  }
  /* What changed between the published version and the prepared data, in the owner's source
   * fields. A rotated materializer or method identity is named as such: it is not new data. */
  const SOURCE_FIELDS={panel_through:'Features cutoff',data_revision_hash:'Market-data revision',manifest_revision:'Membership revision',panel_snapshot_hash:'Panel snapshot',outcome_watermark_hash:'Outcome watermark',outcome_recipe_hash:'Outcome method',outcome_policy_hash:'Outcome policy',materializer_hash:'Input materializer (source code)',binding_hash:'Data binding'};
  function sourceChanges(before, after) {
    return Object.keys(SOURCE_FIELDS).filter(k=>before && after && before[k]!==after[k]).map(k=>({key:k,label:SOURCE_FIELDS[k],before:before[k],after:after[k]}));
  }
  function captureSummary(p) {
    const changes=sourceChanges(p.before,p.after);
    const newData=changes.some(c=>['panel_through','data_revision_hash','panel_snapshot_hash','outcome_watermark_hash'].includes(c.key));
    const codeOnly=changes.length && !newData;
    const rows=[['Input family',p.input_id],['Previous version',mono(p.prior_binding_hash,SHORT.hash)]];
    rows.push(['What changed',changes.length ? html`${changes.map(c=>html`${t(c.label)}: ${c.key==='panel_through' ? html`${c.before} → ${c.after}` : html`${mono(c.before)} → ${mono(c.after)}`}<br>`)}` : t('Nothing recorded as changed')]);
    rows.push(['Meaning',t(codeOnly ? 'The input materializer or method identity rotated; the market data itself is unchanged. The new version re-seals the same data under the current source code.' : newData ? 'The prepared data moved: the new version seals the current Features cutoff. Existing versions stay exactly as published.' : 'An exact version already exists; nothing is re-sealed.')]);
    rows.push(['Network',t('None ({n} calls); no model is trained ({m} numerical calls).',{n:p.network_calls ?? 0,m:p.numerical_calls ?? 0})]);
    rows.push(['Unchanged',t('Configured defaults, existing versions and saved research. Selecting the new version is a further explicit step.')]);
    return rows;
  }
  const OPTION_TEXT={retain_raw_value_with_caveat:'Keep the raw value, with a recorded caveat',quarantine_listing:'Quarantine the listing until it requalifies',exclude_from_next_manifest:'Exclude the listing from the next membership',wait_then_retry:'Wait, then let the owner retry',retry_primary:'Retry the primary source with a bounded full history',escalate_for_human:'Escalate the data-truth conflict for human review'};
  const OPTION_IDS={recoverable_quarantine:'quarantine_listing',exclude_from_next_manifest:'exclude_from_next_manifest',wait_for_provider_recovery:'wait_then_retry',retain_isolated_raw_move_with_caveat:'retain_raw_value_with_caveat',bounded_full_history_retry:'retry_primary',escalate_data_truth_conflict:'escalate_for_human'};
  const optionText=(o)=>t(OPTION_TEXT[String(o?.policy_args?.action || OPTION_IDS[o?.option_id] || '').toLowerCase()] || o?.option_id || '');
  const optionNote=(o)=>{const a=o?.policy_args || {};const parts=[];if(a.recheck_after_at)parts.push(t('recheck after {t}',{t:when(a.recheck_after_at)}));if(a.retry_after_at)parts.push(t('retry after {t}',{t:when(a.retry_after_at)}));if(a.retry_after_seconds)parts.push(t('the owner retries after {n} min',{n:Math.round(Number(a.retry_after_seconds)/60)}));if(o?.disposition==='human_review')parts.push(t('recorded for human review'));return parts.join(' · ');}; // N6: what tells the responses apart, in the reader's clock
  function issueSummary(p, delegated=false) {
    const o=p.option || {};
    return [['Response',optionText(o)],[delegated ? 'Executor' : 'Applied by',delegated ? t('External automation under this exact grant') : t(o.disposition==='auto' ? 'the owner, once confirmed' : o.disposition==='human_review' ? 'a person, recorded for review' : 'the owner')],['Applied now',t(delegated ? 'No: this grant only authorizes an exact later confirmation.' : p.effect_applied ? 'Yes' : 'No: recorded as your decision; applied when the Task continues and the owner revalidates the source evidence.')],[delegated ? 'Authorized by' : 'Confirmed by',delegated ? t('You') : actorWords(p.confirmation || 'HUMAN')]];
  }
  function continueSummary(c) {
    return [['Recorded as',codeWords(c.lifecycle || c.status)],['Why it stopped',c.failure_reason?.explanation || explain(c.failure_code) || (c.failure_code ? said(c.failure_code) : '')],['Retry',c.retry_after_at ? t('Not before {time}',{time:when(c.retry_after_at)}) : t('Asks the owner to continue under the same plan; it refuses if the retry is not due.')]];
  }
  function cleanupSummary(p) {
    const targets=Object.keys(p.targets || {});
    return [['Files selected for cleanup',targets.length],['Reclaimable',bytesText(p.reclaimable_bytes)],['Retained',bytesText(p.retained_bytes)],['Unrooted versions',(p.bindings || []).length ? html`${p.bindings.map(h=>html`${mono(h,SHORT.hash)}<br>`)}` : t('none')],['Exact targets',targets.length ? html`${targets.slice(0,12).map(v=>html`<span class="mono">${v}</span><br>`)}${targets.length>12 ? t('… and {n} more, all listed in the exact plan below',{n:targets.length-12}) : ''}` : t('none: nothing is eligible')],['Kept by policy',html`${(p.limitations || []).map(v=>html`${t(LIMITATION_TEXT[v] || v)}<br>`)}`],
      ...(relinks(p).length ? [['Linked to the model store',html`${t('The retrieval model\'s copy becomes links to this machine\'s model store; nothing is lost.')}<br>${relinks(p).map(([path,bytes])=>html`<span class="mono">${path}</span> · ${bytesText(bytes)}<br>`)}`]] : [])];
  }
  const relinks=(p)=>Object.entries(p?.relinks || {}); // U44: {path: bytes} of a retained model copy the plan links to the store
  function pinSummary(v) {
    const pinned=(v.roots || []).includes('USER_PINNED');
    return [['Version',mono(v.binding_hash,SHORT.hash)],['Change',t(pinned ? 'Remove your pin: the version keeps every other root it has; without a root it becomes eligible for cleanup.' : 'Pin this version: it is retained by you and never selected for cleanup while the pin stands.')],['Referenced bytes',bytesText(v.logical_bytes)]];
  }
  async function preview(kind, value='') {
    if(S.busy)return;
    invalidate();const page=here(),ticket=S.revision;S.busy=true;render();
    try {
      let body,steps,title;
      if(kind==='prepare') {
        body=await Data.post('/api/workspace/preparation/plan',{});
        states.set('prepare',{body}); title='Confirm data preparation';
        if(body.plan_hash&&body.confirmation_available!==false&&['CONFIRMATION_REQUIRED','DEFERRED','BLOCKED','RECOVERY_REQUIRED'].includes(body.status))
          steps=[['/api/workspace/preparation/confirm',{preparation_plan_hash:body.plan_hash}]];
      } else if(kind==='reoffer') {
        // The retained preview, offered again: the owner still checks its plan hash and date.
        body=get('prepare');title='Confirm data preparation';
        if(!body?.plan_hash||body.confirmation_available===false)throw Error('Preview the preparation first.');
        steps=[['/api/workspace/preparation/confirm',{preparation_plan_hash:body.plan_hash}]];
      } else if(kind==='continue-preparation') {
        // The owner's own continuation of a waiting or stopped Task: the same plan hash, confirmed
        // again; the owner refuses a retry that is not due and re-marks the Task when it is.
        const b=get('welcome');
        if(!b?.plan_hash||!b.task_id)throw Error('No preparation Task to continue.');
        if(b.confirmation_available===false)throw Error(t('Preview the preparation first.'));
        body={status:b.status,task_id:b.task_id,plan_hash:b.plan_hash,retry_after_at:b.retry_after_at || b.progress?.retry_after_at || null,failure_code:b.failure_code};
        const resume=resumeOf(b); // U63: a deferral names its own resume
        title=resume ? 'Resume this preparation' : 'Continue this preparation Task';steps=[requestStep(resume || b.next_requests?.confirm,'WORKSPACE_PREPARE_CONFIRM')];
      } else if(kind==='update') {
        body=await Data.post('/api/data-update/plan',{});states.set('update-plan',{body});
        title=body.status==='CONFIRMATION_REQUIRED'?'Approve the change and update data':'Confirm data update';
        if(body.plan_hash&&['PLANNED','CONFIRMATION_REQUIRED'].includes(body.status)) {
          const payload={update_plan_hash:body.plan_hash};steps=[];
          if(body.status==='CONFIRMATION_REQUIRED')steps.push(['/api/data-update/confirm',payload]);
          steps.push(['/api/data-update/run',payload]);
        }
      } else if(kind==='continue-update') {
        // The owner's own continuation of a waiting or stopped update: the same plan hash, run
        // again; the owner refuses a retry that is not due and re-marks the Task when it is.
        const b=get('data');
        if(!b?.plan_hash||!b.task_id)throw Error('No data update Task to continue.');
        body={status:b.task_lifecycle,task_id:b.task_id,plan_hash:b.plan_hash,retry_after_at:b.retry_after_at || b.cycle?.retry_after_at || null,failure_code:b.cycle?.failure_code || null,lifecycle:b.task_lifecycle};
        const resume=resumeOf(b); // U63: a deferral names its own resume
        title=resume ? 'Resume this update' : 'Continue this data update';steps=[resume ? resumeStep(resume) : ['/api/data-update/run',{update_plan_hash:b.plan_hash}]];
      } else if(kind==='capture') {
        if(!S.family)throw Error('Select a published input family first.');
        body=await Data.post('/api/research-inputs/plan',{research_input_id:S.family});
        states.set('capture',{body});title='Publish a separate immutable input';
        if(body.status==='CONFIRMATION_REQUIRED')steps=[['/api/research-inputs/confirm',{research_input_plan_hash:body.plan_hash}]];
      } else if(kind==='cleanup') {
        body=await Data.post('/api/workspace/storage/plan',{});states.set('cleanup',{body});title='Confirm displayed cleanup';
        if(body.plan_hash&&(Object.keys(body.targets || {}).length || relinks(body).length))steps=[['/api/workspace/storage/confirm',{storage_plan_hash:body.plan_hash}]];
      } else if(kind==='issue') {
        const issue=get('issues')?.issues?.find(v=>v.case.case_token===value);
        const option=issue?.case.options.find(v=>v.option_id===S.options.get(value));
        if(!issue?.options_current||!option||!issue.confirmable_option_ids.includes(option.option_id))throw Error('Select a current permitted data response.');
        const request=get('issues').next_requests?.[`preview:${value}:${option.option_id}`];
        body=await previewIssue(request);if(!body)return;title='Confirm this data decision';
        steps=[requestStep(body.next_requests?.confirm,'DATA_ISSUE_CONFIRM')];
      } else if(kind==='delegate') {
        const [token,taskId]=JSON.parse(value);
        const issue=get('issues')?.issues?.find(v=>v.case.case_token===token);
        const option=issue?.case.options.find(v=>v.option_id===S.options.get(token));
        if(!issue?.options_current||!option||!issue.confirmable_option_ids.includes(option.option_id))
          throw Error('Select a current permitted data response.');
        const previewRequest=get('issues').next_requests?.[`preview:${token}:${option.option_id}`];
        const grantRequest=get('issues').next_requests?.[`delegate:${taskId}:${token}:${option.option_id}`];
        body=await previewIssue(previewRequest);if(!body)return;
        if(!grantRequest)throw Error('No blocked preparation Task is available for this grant.');
        body={...body,preparation_task_id:taskId};
        title='Authorize this exact external decision';
        steps=[requestStep(grantRequest,'DATA_ISSUE_DELEGATE')];
      } else if(kind==='revoke-delegation') {
        const grant=get('issues')?.delegations?.find(v=>v.grant_hash===value);
        if(!grant)throw Error('This grant is no longer active.');
        body=grant;title='Revoke this external decision grant';
        steps=[requestStep(grant.next_requests?.revoke,'DATA_ISSUE_REVOKE')];
      } else if(kind==='continue') {
        const c=get('issues')?.continuations?.[Number(value)];
        if(!c||!Data.offers(c.operation)||Data.route(c.operation)!==c.endpoint)throw Error('Data continuation is not available.');
        if(c.operation==='WORKSPACE_PREPARE_PLAN') {
          body=await Data.post(...requestStep(get('issues').next_requests?.[`continue:${c.task_id}`],c.operation));
          states.set('prepare',{body});
          title='Confirm preparation continuation';
          if(body.confirmation_available!==false && body.plan_hash && body.next_requests?.confirm)
            steps=[requestStep(body.next_requests.confirm,'WORKSPACE_PREPARE_CONFIRM')];
        } else {
          body=c;title='Continue the approved data task';
          steps=[requestStep(get('issues').next_requests?.[`continue:${c.task_id}`],c.operation)];
        }
      } else if(kind==='pin') {
        const input=get('storage')?.inputs?.find(v=>v.binding_hash===value);
        if(!input?.available||get('storage').status==='RECOVERY_REQUIRED')throw Error('Input retention is not currently available.');
        body=input;title='Confirm input retention change';steps=[['/api/workspace/storage/pin',{input_binding_hash:value,input_pinned:!input.roots.includes('USER_PINNED')}]];
      } else if(kind==='pin-index') { // an evidence index pinned or unpinned by its row (round E5), as the Host offers it (U4): the owner's pin route takes the index id
        const [id,act]=JSON.parse(value), ix=(get('storage')?.evidence?.indexes || []).find(v=>v.index_id===id);
        if(!ix || !(ix.available_actions || []).includes(act))throw Error('Index retention is not currently available.');
        body={...ix,claim:'EVIDENCE_INDEX_RETENTION'};title=act==='UNPIN' ? 'Unpin this evidence index' : 'Pin this evidence index';steps=[['/api/workspace/storage/pin',{input_binding_hash:id,input_pinned:act==='PIN'}]];
      } else if(kind==='rebuild-index') { // an evicted index rebuilt from its committed vectors, synchronously, by its row
        const ix=(get('storage')?.evidence?.indexes || []).find(v=>v.index_id===value);
        if(!ix)throw Error('Evidence index not found.');
        body={...ix,claim:'EVICTED_EVIDENCE_INDEX_REBUILDS_FROM_ITS_COMMITTED_VECTORS'};title='Rebuild this evidence index';steps=[['/api/workspace/storage/evidence-rebuild',{evidence_index_id:value}]];
      } else if(kind==='backup') { // U48 (V209): a generation made now, on the person's confirmation
        const bk=get('backups');
        if(!bk || states.get('backups').error)throw Error('The backups are not read; nothing was sent.');
        body=bk;title='Back up the held state now';steps=[['/api/workspace/backup',{}]];
      } else if(kind==='resume-cleanup') {
        if(!get('storage')?.pending_cleanup?.includes(value))throw Error('Approved cleanup not found.');
        // Preview the exact approved operation again; execution owner verifies its retained roots.
        body={storage_plan_hash:value,claim:'RESUME_ALREADY_APPROVED_CLEANUP'};title='Resume approved cleanup';
        steps=[['/api/workspace/storage/confirm',{storage_plan_hash:value}]];
      } else throw Error('Workspace action not available.');
      if(here()!==page||ticket!==S.revision)return;
      setNotice(kind==='cleanup'&&!Object.keys(body.targets || {}).length&&!relinks(body).length?'No eligible files to clean. Nothing was removed.':'');
      // A plan the owner refuses by name (a pending transition, a missing binding) is said
      // as attention, with the owner's code; nothing is offered.
      if(kind==='update' && !steps?.length) S.error=t('No update can be planned now ({code}): {why}',{code:body.failure_code || body.status || '',why:t(body.next_action==='RESOLVE_EXISTING_WORKSPACE_TRANSITION' ? 'the workspace has a pending membership transition that the data owner must resolve first; the preparation page shows it.' : 'the owner named no plan to confirm.')});
      S.busy=false;
      if(steps?.length)offer(title,body,steps,page,kind);
      else if(body.task_id)await LiveTasks.open(body.task_id);
    } catch(e) {
      // the press is answered even when the owner repeats a refusal already shown (the user, 2026-09-24:
      // 点prepare workspace, 什么也没发生): the toast says it was refused, the page's notice says why
      if(here()===page){S.error=e.message;notify('Action refused');}
    }
    finally {S.busy=false;render();}
  }
  async function commit() {
    const pending=S.pending;
    if(S.busy||!pending||pending.page!==here()||pending.revision!==S.revision)return;
    S.pending=null;S.busy=true;S.error='';closeDialog();render();
    try {
      let body;
      for(const [path,payload] of pending.steps)body=await Data.post(path,payload);
      const last=pending.steps.at(-1)[0], preparing=last==='/api/workspace/preparation/confirm', updating=last==='/api/data-update/run';
      setNotice(preparing || updating ? '' : body.status==='ADMITTED' && body.task_id ? t('Admitted as Task {id}; its progress is on Tasks and here once it settles.',{id:short(body.task_id)}) : body.status || body.disposition || 'Completed');
      if(last==='/api/research-inputs/confirm'){states.set('capture',{body});if(body.task_id) S.watch={task:body.task_id,page:'inputs'};}
      if(preparing)states.delete('prepare'); // admitted: the scene is the Task now, not the preview
      if(preparing && body.task_id && body.task_id!==selectedTask('welcome')) { if(typeof objectEntry==='function') objectEntry('preparation:'+body.task_id); pinTask('welcome',body.task_id); }
      if(updating) {
        states.delete('update-plan');
        if(body.status==='REUSED_EXACT') {
          // Nothing ran: the owner answered with the receipt it reuses. Said as reuse, never as a
          // rebuild; no Task is opened because none exists.
          const r=body.receipt || {};
          setNotice(t('Nothing to update: the workspace already holds {target}. The owner reused the receipt of {at} exactly; no Task was started and nothing was downloaded.',{target:r.target_session || '',at:r.completed_at ? when(r.completed_at) : ''}));
        } else if(body.task_id && here()==='data') { if(body.task_id!==selectedTask('data') && typeof objectEntry==='function') objectEntry('update:'+body.task_id); pinTask('data',body.task_id); }
        else if(body.task_id) { S.watch={task:body.task_id,page:here()}; setNotice(t('The update continues as Task {id}; the Data page shows its work.',{id:short(body.task_id)})); }
      }
      if(last==='/api/workspace/data-issues/confirm') setNotice(t(body.status==='CONFIRMED_PENDING_REVALIDATION' ? 'Your decision is recorded. It is applied when the data update continues; the owner revalidates the source evidence then.' : body.status==='ALREADY_APPLIED' ? 'This decision was already recorded; nothing changed.' : body.status || 'Completed'));
      // Storage says what it did in bytes and files, never a promise about space: the deleted
      // paths are the owner's count, and a pin is retention until the person lifts it.
      if(last==='/api/workspace/storage/confirm') setNotice(body.status==='COMPLETED' ? t(body.relinked ? 'Cleanup completed: {n} released, and the retrieval model\'s copy linked to the model store ({size}). Every retained version and the working database are as they were.' : 'Cleanup completed: {n} released. Every retained version and the working database are as they were.',{n:countText(body.deleted_paths ?? 0,'{n} file path','{n} file paths'),size:String(bytesText(Object.values(body.relinked || {}).reduce((a,b)=>a+Number(b || 0),0)))}) : t(body.status || 'Completed'));
      if(last==='/api/workspace/backup') setNotice(t('Backed up as generation {g}; nothing in the workspace changed.',{g:short(body.generation_hash,SHORT.hash)}));
      if(last==='/api/workspace/storage/pin') setNotice(body.status==='PINNED' ? t('Version {v} is retained until you unpin it.',{v:short(body.binding_hash,SHORT.hash)}) : body.status==='UNPINNED' ? t('Version {v} is no longer pinned; the owner\'s retention roots decide again.',{v:short(body.binding_hash,SHORT.hash)}) : t(body.status || 'Completed'));
      await refresh(pending.page);
      // The preparation and update scenes are their pages; a publication is named on the inputs
      // page and settles there. Nothing opens the Task Center over the page the person is reading.
    } catch(e) {S.error=e.message;}
    finally {S.busy=false;render();}
  }
  async function selectInput(value) {
    const [id,binding]=JSON.parse(value);
    if(!get('inputs'))await refresh('inputs','',true);
    const input=get('inputs')?.inputs?.find(v=>v.input_id===id)?.versions?.find(v=>v.binding_hash===binding);
    if(!input?.available)throw Error('The exact input files are unavailable.');
    // This edits a Factor draft only. It never changes a workspace default or starts PLAN/RUN.
    await LiveResearch.useInput(JSON.stringify([id,binding]));
  }

  /* ---- the preparation scene ---- */
  // S (law 88): a source's locator by its ends, whole on hover, its copy glyph beside it (WD3); its words follow
  const sourceLine = (v) => { const [index, uri] = String(v).includes(': ') ? String(v).split(': ') : [null, String(v)]; return html`${index ? html`<span class="mono">${index}</span> · ` : ''}${locatorCell(uri)}<span class="prep-source-words">${t(index ? 'current members · network, only after confirmation' : 'daily bars and corporate actions · network, only after confirmation')}</span>`; };
  /* The workspace's network control (CLI-15, OP5) in words: what decides it, its code on hover (WD2); read
   * by the preparation's refusal and by Settings, which holds the control */
  const NETWORK = {WORKSPACE_CONTROL_ALLOWED:'Allowed by this workspace\'s network control.', WORKSPACE_CONTROL_OFF:'Off by this workspace\'s network control.', DEFAULT:'Off: the workspace sets no network control, so the network stays off.'};
  const SWITCH=String(html`<code class="literal">ALPHALATTICE_NETWORK_DISABLED=1</code>`); // the operator's switch: a literal the person types
  function networkWords(n) {
    // U70 (V452): an opening the first-use goal made holds until its own end, named in the control itself (`set_by`); past
    // its end the control reads closed with no write
    if(n?.decided_by==='WORKSPACE_CONTROL' && n.set_by?.delegation){
      if(n.network_allowed) return html`<span data-tip="${n.set_by.delegation}">${t('Opened for you by your first-use goal, until {t}',{t:whenText(n.set_by.until)})}</span>`;
      if(Date.parse(n.set_by.until)<=Date.now()) return t('Closed when your first use ended');
    }
    const by=n?.decided_by || '', key=by==='WORKSPACE_CONTROL' ? (n.network_allowed ? 'WORKSPACE_CONTROL_ALLOWED' : 'WORKSPACE_CONTROL_OFF') : by;
    // V620 (U93): a hold the person cannot lift here -- the operator's offline switch, a run's offline hold -- reads in the
    // network owner's words, which name who lifts it and how; the page recreates no permission advice
    if(['OPERATOR_OFFLINE_SWITCH','RUN_HELD_OFFLINE'].includes(by) && n.detail) return html`<span data-tip="${by}">${literalWords(t(n.detail))}</span>`;
    return NETWORK[key] ? html`<span data-tip="${by}">${t(NETWORK[key])}</span>` : (n?.detail ? literalWords(t(n.detail)) : '');
  }
  const LIMITS = {CURRENT_UNIVERSE_RESEARCH_ONLY:'Current-universe research only: historical index membership is not supplied.', NO_DATA_API_KEY:'No data API key is requested or stored.', NO_FOUNDATION_OR_STRATEGY_ACTIVATION:'Preparation activates no Foundation or strategy.'};
  function scope(b) {
    const local=html`<li>${icon('lock')}<span>${t('Local: the DuckDB working copy, the Features and the immutable research inputs stay in this workspace.')}</span></li>`;
    // WD4 (the walk's R4): the sources and their access are a press away, in Facts -- the first screen before preparation
    // held some 230 words, and the preview names the network's use again when it asks for confirmation
    return factsRef(t('Sources and access'),html`<p class="caption">${t('What may use the network, what stays local, and what is not asked for.')}</p><ul class="prep-sources">${(b.sources || []).map(v=>html`<li>${icon('cloud')}<span>${sourceLine(v)}</span></li>`)}${local}${(b.limits || []).map(v=>html`<li>${icon('info')}<span>${t(LIMITS[v] || v)}</span></li>`)}</ul>`);
  }
  function planPreview(p) {
    const refused=p.confirmation_available===false;
    const access=p.source_access || {}, net=access.network_access || {};
    // a refused confirmation says its way once (ST6). Sources need the network (CLI-15): the workspace's own
    // control in Settings allows it and the preview is asked again, the Host running on; only the operator's
    // offline switch needs the Host started again, without it
    const offline=refused && net.decided_by==='OPERATOR_OFFLINE_SWITCH';
    const start=offline && access.workspace_path ? codeRef(t('Start command for this workspace'), 'python scripts/run_alphalattice.py --workspace "'+access.workspace_path+'" serve --no-browser') : '';
    const next=!refused ? undefined : offline ? raw(t('stop the idle Host and start it again without {switch}, then preview again; research execution keeps its offline policy',{switch:SWITCH})) : html`${link(html`${t('Allow network access in Settings')}${icon('arrow')}`,'settings','text-btn',{row:'networkAccess'})} ${t('then preview again; the Host keeps running')}`;
    const why=refused ? html`${t('Its sources need the network.')} ${networkWords(net)} ${t(access.verified_source_checkpoint_retained ? 'The verified source checkpoint remains recorded.' : 'No source checkpoint was captured yet.')} ${t('Nothing was admitted.')}` : '';
    return html`<section class="panel prep-preview" data-box="decision">${sectionHead(t('Preparation scope · preview'), t('Recorded by the owner as a plan; nothing has been admitted. Confirming is a separate step.'), refused ? badge('blocked', t('Confirmation unavailable')) : stateLine('planned', {word: t('Awaiting confirmation')}))}<div class="panel-body">${kv(preparationSummary(p).map(([k,v])=>[t(k),typeof v==='string'?v:v]))}${refused ? refusal({code:p.source_access_failure || '',reason:why},'warning',{word:t('Confirmation unavailable'),next,more:start}) : ''}<div class="flow">${refused ? '' : action('Confirm preparation','preview','reoffer')}${action('Preview again','preview','prepare')}</div>${detail(p)}</div></section>`;
  }
  const readiness = (b) => b.status==='LOCAL_DATA_PRESENT' ? t('Local market data is present but no research input is published. The preview reuses qualified local data without downloading; its qualification is checked there.') : t('This workspace holds no research input. One explicit preparation captures the current index members, acquires their daily history, builds Features and publishes one verified input. Opening this page starts nothing.');
  /* First use before a Task (round 74): the PLAN preview when one was offered, the workspace's
   * facts from the manifest and the preparation owner, the five stages as the run's steps to
   * come. The one action is the Home head's primary (Prepare workspace). */
  function unprepared(b, state) {
    const p=get('prepare');
    const facts=kv([[t('Workspace'),html`<strong>${Data.workspace()}</strong>${infoMark(t('the folder this service was started on; its manifest names it'))}`],[t('Profile'),html`<span class="mono">${b.profile || ''}</span>`],[t('History and cutoff'),t(p ? '{n} years of daily bars before the target session {d}' : 'Named by the preview: the owner sets the target session and the initial history there.',{n:p?.initial_history_years || '',d:p?.target_session || ''})],[t('Universe'),p?.universe || t('Current S&P 500, NASDAQ-100 and DJIA members, merged; named exactly at preview')],[t('Readiness'),readiness(b)]]);
    const steps=stepList(PREPARATION_STEPS.map(([id,name,desc])=>({id,name:t(name),mark:'PENDING',note:t(desc)})));
    return html`${p ? planPreview(p) : ''}<section class="panel">${sectionHead(t('Before preparation'),(b.limits || []).includes('NO_DATA_API_KEY') ? '' : t('No data API key is requested or stored.'),stateLine('draft',{word:t(b.status==='LOCAL_DATA_PRESENT' ? 'Local data present · no research input' : 'New workspace · not prepared')}))}<div class="panel-body">${facts}${scope(b)}</div></section>
      ${panel(t('Stages'),t('Data preparation, input publication and selecting a research input are separate actions.'),steps)}${detail(b)}`;
  }
  /* The Feature owner's own step while it runs (bound to this Task and stage): its count, or a
   * finished step said to be finished, never shown full. Shared by both scenes. */
  function featureStep(w, state, wait) {
    const label=t(stageOf(w.stage_id).word), n=Number(w.completed_units), total=Number(w.total_units), unit=t(w.unit_name);
    const reported=t('reported at {time} by the Feature owner, bound to this Task and stage',{time:when(w.updated_at)});
    if(w.status==='RUNNING') return {...workCount(n,total,unit),
      detail:html`${label}${w.current_item ? html` · ${t('current')} <span class="mono">${short(w.current_item,SHORT.id)}</span>` : ''} · ${reported}`,wait};
    if(w.status==='FAILED') return {count:html`<span class="tp-denom">${t('{label} stopped · {code}',{label,code:w.failure_code || ''})}</span>`,bar:null,detail:reported,wait};
    return {count:html`<span class="tp-denom">${t('{label} finished · {n} / {total} {unit} · the next step has not reported yet',{label,n:n.toLocaleString('en-US'),total:total.toLocaleString('en-US'),unit})}</span>`,bar:null,detail:reported,wait};
  }
  const NO_COUNT=(wait=null,verified=false)=>({count:html`<span class="tp-denom">${t('No finer count is reported for this stage')}</span>`,bar:null,detail:t(verified ? 'The owner records this stage as one unit; Task Control has verified it.' : 'The owner records this stage as one unit; its verification is the next fact.'),wait});
  const stageVerified=(v,stage)=>v.stages.some(s=>s.stage_id===stage && s.lifecycle==='VERIFIED');
  /* The current stage's own count, with its owner-defined denominator and unit, or an honest
   * "no finer count". Never a percentage across stages, never an estimate of time. */
  function work(b, v, state, shown=null) {
    const stage=shown || v.status.current_stage, p=b.progress, w=b.work_progress, age=readAge(state);
    if(stage==='prepare_data' && p?.phase==='prepare_data') {
      // The runner's counts at the last unit snapshot while it is current (they advance
      // within a chunk), otherwise the durable chunk record; both are the owner's, and the
      // caption says which boundary the number is at.
      const la=b.listing_activity, live=la && la.availability==='BOUND' && la.counts ? la.counts : null;
      const c=Number((live ? live.candidates : p.candidates) ?? 0), r=Number((live ? live.raw_ready : p.raw_ready) ?? 0);
      const boundary=live ? t('at the unit snapshot of {time} · the chunk record says {r} of {c}',{time:la.written_at ? when(la.written_at) : '',r:Number(p.raw_ready ?? 0).toLocaleString('en-US'),c:Number(p.candidates ?? 0).toLocaleString('en-US')}) : t('the chunk record');
      return {count:html`<span class="tp-number">${r.toLocaleString('en-US')}</span><span class="tp-denom"> / ${c.toLocaleString('en-US')} ${t('candidates with raw bars')}</span>`,bar:[r,c],
        detail:html`${t('{q} quality-eligible · {f} failed · candidates are the captured source membership; raw availability is not research qualification',{q:(live ? live.quality_eligible : p.quality_eligible) ?? '',f:(live ? live.failed : p.failed) ?? ''})} · ${boundary}`,wait:p.retry_after_at};
    }
    if(stage==='prepare_features' && w?.availability==='BOUND') return featureStep(w,state,p?.retry_after_at);
    if(stage==='prepare_features' && p?.phase==='prepare_features') {
      return {count:html`<span class="tp-denom">${t('No finer count is reported for this step')}</span>`,bar:null,
        detail:html`${t('Feature owner status')} ${coded(p.status)}${p.membership_revision ? html` · ${t('membership')} <span class="mono">${short(p.membership_revision,SHORT.hash)}</span>` : ''}`,wait:p.retry_after_at};
    }
    return NO_COUNT(null,stageVerified(v,stage));
  }
  function kept(v) {
    return html`<div class="tp-retained">${icon('archive')}<div><span>${t('Kept safe')}</span><strong>${v.artifact_refs.length ? countText(v.artifact_refs.length,'{n} artifact reference verified by its owner','{n} artifact references verified by their owners') : v.verified_stage_count ? t('{n} of {total} stages verified; their records are kept by Task Control',{n:v.verified_stage_count,total:v.total_stage_count}) : t('No stage verified yet; nothing to keep beyond the admitted Task')}</strong><small>${t('A lost Web connection or a closed page does not erase it; a resume re-verifies it from its evidence and runs only the unverified stage again.')}</small></div></div>`;
  }
  /* The attention state in plain language, with the consequence and only the owner's currently
   * permitted continuation. The owner's code stays in the details. */
  /* U63 (V375): a deferral's words as its owner wrote them -- the provider's, and for an update that the published
   * research inputs stand (the owner's second sentence, read on its own) -- and its resume request, the same plan again,
   * sent whole by the route the session names (U13). */
  const HELD_WORDS=' The published research inputs are unchanged, and every study keeps reading them.';
  const deferWords=(detail)=>{ const s=String(detail || ''); if(!s) return ''; const held=s.endsWith(HELD_WORDS); return html`<span class="owner-text">${t(held ? s.slice(0,-HELD_WORDS.length) : s)}${held ? html` ${t(HELD_WORDS.trim())}` : ''}</span>`; };
  const resumeOf=(b)=>{ const r=b?.next_requests?.resume; return r && Data.offers(r.operation) ? r : null; };
  const resumeStep=(r)=>[Data.route(r.operation),Object.fromEntries(Object.entries(r).filter(([k])=>k!=='operation'))];
  function attention(b, v, state) {
    const [tone,ic,text]=LiveTasks.standing(v), id=v.task_id, life=v.lifecycle;
    const truthPending=b.confirmation_available===false && b.next_requests?.issues?.operation==='DATA_ISSUES';
    const truth=b.failure_code==='data.truth_review_required' ? issuesAsking() : null;
    const reason=get('issues')?.continuations?.find(c=>c.task_id===id)?.failure_reason?.explanation;
    const shown=truth ? html`<strong>${t(truth.agent ? 'Agent is deciding data issues' : truth.decided && !truth.count ? 'Decision recorded' : 'A data decision is needed')}</strong> · ${truth.agent ? t('The agent is deciding under the first-use delegation; the issue states are shown below.') : truth.decided && !truth.count ? t('Decided · applied when the update continues') : reason || explain(b.failure_code)}` : text;
    const by=Object.fromEntries(v.actions.map(a=>[a.action,a]));
    const off=(a)=>S.busy || !a?.available ? (a?.reason || '') : '';
    let consequence='', actions='';
    const wait=b.retry_after_at || b.progress?.retry_after_at || null;
    const due=!wait || (Date.parse(wait) <= Date.parse(v.observed_at)+readAge(state)*1000);
    switch(life) {
      case 'RECOVERY_REQUIRED':
        consequence=t('Resuming is confirmed against this exact Task version and creates no new Task; the owner may stop it again and records why.');
        actions=html`${typedBtn(t('Resume this Task'),'task-recovery',id,'button primary',off(by.RECOVER))}`; break;
      case 'DEFERRED': {
        const resume=resumeOf(b);
        consequence=b.detail ? html`${deferWords(b.detail)}${wait ? html` ${t('It resumes from {time}.',{time:when(wait)})}` : ''}` : t('The owner is waiting for a condition it named; nothing is retried on its own. Continuing this Task asks the owner again under the same plan; it refuses until the retry time is due.');
        actions=html`${action(resume ? 'Resume this preparation' : 'Continue this Task','preview','continue-preparation',due ? '' : t('Not due until {time}',{time:when(wait)}))}${link(t('Data issues'),'issues','button compact')}`; break;
      }
      case 'BLOCKED':
        if(String(b.failure_code || '').startsWith('research_experiment.insufficient_listing_support:')) {
          consequence=t('This input has too few listings for research. The minimum support is enforced by the owner; a new qualified source is required before a research input can be published.');
          break;
        }
        if(b.superseded_by_task_id) {
          consequence=t('This stopped Task is historical. A successor Task continued from its retained checkpoint; inspect that exact Task for the current result.');
          actions=action('Inspect successor Task','task',b.superseded_by_task_id);
          break;
        }
        if(truthPending || (truth?.decided && !truth.count)) {
          actions=link(t('Data issues'),'issues','button compact');
          break;
        }
        consequence=b.execution_binding_changed ? t('The product changed since this Task was planned, so its plan no longer binds. A new preview continues from the sources this Task captured; this Task stays recorded as it is.') : t('The owner recorded a stop, not an interruption, and explains it above. A data decision belongs on the Data issues page. Retrying asks the owner to run the unverified stage again under the same plan; if the cause is unchanged it stops there again, and nothing recorded is undone.');
        actions=b.execution_binding_changed ? action('Preview preparation','preview','prepare') : html`${b.confirmation_available!==false && b.next_requests?.confirm?.operation==='WORKSPACE_PREPARE_CONFIRM' ? action('Retry this Task','preview','continue-preparation',due ? '' : t('Not due until {time}',{time:when(wait)})) : ''}${link(t('Data issues'),'issues','button compact')}`; break;
      case 'CANCELLED':
        consequence=t('The cancellation was acknowledged at a safe checkpoint. What was recorded before it is kept: a new preview offers to continue from the sources this Task captured, without capturing them again.');
        actions=action('Preview preparation','preview','prepare'); break;
      case 'CANCEL_REQUESTED':
        consequence=t('Requested is not acknowledged: the worker keeps running until its next safe checkpoint (a market-data chunk boundary), then stops. Nothing already committed is undone.'); break;
      case 'RUNNING': case 'QUEUED':
        if(v.liveness.status==='NOT_RECENT') consequence=t('The worker has not reported recently. It may still be running a long unit, or the local service that ran it may have stopped; this page does not decide. A Task recorded as running is not offered a resume; if the service was restarted, read again after it settles.');
        else if(v.worker_failure) consequence=t('The last command for this Task stopped outside Task Control; the Task\'s own lifecycle says where it stands.');
        break;
      default: break;
    }
    if(!consequence && !(tone==='attention')) return '';
    return html`<section class="panel prep-attention" data-box="decision"><div class="tp-callout ${tone || 'attention'}">${icon(ic)}<p>${shown}</p></div>${consequence ? html`<p class="prep-consequence">${consequence}</p>` : ''}${actions ? html`<div class="flow">${actions}</div>` : ''}</section>`;
  }
  /* The listing snapshot is folded per listing for reading: one row per listing, its
   * transitions in order, its last observation. Retention is bounded (the newest LOG_RETAIN
   * listings); rows first seen on a live read are marked as arrivals once. */
  function absorbListings(b, state, A=W.A) {
    const la=b.listing_activity;
    if(!la || !Array.isArray(la.rows)) return;
    const log=W.freshLog(la.execution_id+'|'+la.stage,la.stage,A), live=W.liveRead(state,A);
    for(const r of la.rows) {
      const rowKey=r.listing_id+'|'+r.state+'|'+r.observed_at;
      if(log.seen.has(rowKey)) continue;
      log.seen.add(rowKey);
      let row=log.rows.get(r.listing_id);
      if(!row){row={listing_id:r.listing_id,symbol:r.symbol,steps:[],at:r.observed_at,noted:r.noted_at || r.observed_at,attention:false};log.rows.delete(r.listing_id);}
      else log.rows.delete(r.listing_id); // re-insert last: the map keeps observation order
      row.steps.push(r);row.at=r.observed_at;row.noted=r.noted_at || r.observed_at;row.attention=row.attention || ['RAW_FAILED','AUDIT_FAILED','QUALITY_INELIGIBLE'].includes(r.state);
      log.rows.set(r.listing_id,row);
      if(live) A.arrived.add(r.listing_id);
    }
    W.retain(log);
    log.snapshot={observed:la.observed,retained:la.retained,dropped:la.dropped,written_at:la.written_at,age:la.age_seconds,availability:la.availability,sequence:la.sequence};
  }
  const STEP_TEXT={
    PENDING:(r)=>r.raw_through ? t('history retained through {d}; waiting to retry the provider',{d:r.raw_through}) : t('waiting for provider data'),
    RAW_READY:(r)=>r.origin==='RETAINED' ? (r.tail_acquired ? t('durable history reused, tail acquired through {d}',{d:r.raw_through || ''}) : t('durable history reused through {d}',{d:r.raw_through || ''})) : t('raw bars acquired through {d}',{d:r.raw_through || ''}),
    QUALITY_ELIGIBLE:()=>t('quality: eligible'),
    QUALITY_INELIGIBLE:(r)=>html`${t('quality: ineligible')} · ${(r.reasons || []).map(x=>codeWords(x)).join(', ') || ''}`,
    FEATURE_READY:()=>t('admitted for Features'),
    RAW_FAILED:(r)=>html`${t('failed')} · ${said(r.failure_code)}`,
    AUDIT_FAILED:(r)=>html`${t('audit failed')} · ${said(r.failure_code)}`,
  };
  const STATE_WORD={PENDING:'waiting for provider data',RAW_READY:'raw bars',QUALITY_ELIGIBLE:'eligible',QUALITY_INELIGIBLE:'ineligible',FEATURE_READY:'admitted',RAW_FAILED:'failed',AUDIT_FAILED:'audit failed'};
  const stepText=(r)=>(STEP_TEXT[r.state] || (()=>r.state))(r);
  /* One listing, one line (N6): the time, the symbol, what the owner recorded for it in order;
   * the listing's identity is the symbol's tip. */
  function listingRow(row) {
    return html`<article class="fv-log-row prep-unit${row.attention ? ' attention' : ''}" data-listing-key="${row.listing_id}"><div class="fv-log-time"><time datetime="${row.noted}">${clockOf(row.noted)}</time><span class="fv-log-dot" aria-hidden="true"></span></div><div class="fv-log-entry"><strong data-tip="${row.listing_id}">${row.symbol}</strong><span class="fv-log-object">${row.steps.map((s,i)=>html`${i ? ' → ' : ''}${stepText(s)}`)}</span></div></article>`;
  }
  /* The Feature owner's step and counters, bound to this Task; no per-listing units. */
  function featureAside(spec, w, state, statusText) {
    const bound=w && w.availability==='BOUND';
    const items=bound ? [[t('Feature owner step'),t(stageOf(w.stage_id).word)],[t('Units'),html`${Number(w.completed_units).toLocaleString('en-US')} / ${Number(w.total_units).toLocaleString('en-US')} ${t(w.unit_name)}`],[t('Current item'),w.current_item ? html`<span class="mono">${short(w.current_item,SHORT.id)}</span>` : ''],...Object.entries(w.counters || {}).map(([k,n])=>[k.replaceAll('_',' '),Number(n).toLocaleString('en-US')])] : [];
    const rail=html`<section class="ui-log-rail prep-log-rail"><div><span>${t('Feature owner')}</span><strong>${bound ? t('reported at {time}',{time:when(w.updated_at)}) : w?.availability==='NOT_CURRENT' ? t('last report is of an earlier step') : t('no report yet')}</strong></div><div><span>${t('Reading')}</span><strong>${t('Following latest')}</strong></div></section>`;
    return W.factsShell(spec,{label:t('Feature work'),title:t('Feature work'),caption:t('The Feature owner\'s own step and counters, bound to this Task; no per-listing units are reported here'),facts:bound ? kv(items) : html`<p class="caption">${statusText}</p>`,rail});
  }
  /* The stage's own recent activity: listing units for market data; the Feature owner's step
   * for Features; the stage as one unit elsewhere. Retained rows of another execution or
   * stage are history, said so, never current work. */
  function stageAside(b, v, state, shown, current) {
    const la=b.listing_activity, log=W.A.log, moving=current && v.lifecycle==='RUNNING' && v.liveness.status==='OBSERVED' && !state.stale;
    if(shown==='prepare_data') {
      const snapshot=log?.snapshot, rows=log ? [...log.rows.values()] : [];
      const history=Boolean(la && la.availability==='NOT_CURRENT') || !current;
      const latest=rows.at(-1);
      const rail=W.logRail({label:history ? t('Last retained unit') : t('Latest unit'),value:latest ? html`${latest.symbol} · ${t(STATE_WORD[latest.steps.at(-1).state] || latest.steps.at(-1).state)} · ${snapshot?.written_at ? t('snapshot of {time}',{time:when(snapshot.written_at)}) : ''}` : t('No unit reported yet')},state);
      const status=!la ? t('No listing activity has been delivered for this stage yet; the counts above are the chunk record.') : la.availability==='UNREADABLE' ? t('The listing snapshot could not be read ({cause}); the rows shown are the last valid snapshot.',{cause:la.cause || ''}) : la.availability==='UNBOUND' ? t('The listing snapshot on disk belongs to another Task; nothing of it is shown.') : snapshot ? html`${t('{r} of {o} units of this execution retained in the snapshot',{r:snapshot.retained,o:snapshot.observed})}${snapshot.dropped ? html` · ${t('{n} earlier units not retained',{n:snapshot.dropped})}` : ''}${log.rows.size>=W.LOG_RETAIN ? html` · ${t('this page keeps the latest {n} listings',{n:W.LOG_RETAIN})}` : ''}${history ? html` · ${t('retained history, not current work')}` : ''}${state.stale ? html` · ${t('last successful read')}` : ''}` : '';
      const empty=html`<p class="caption prep-log-empty">${history ? t('No listing unit was retained for this stage.') : t('Listings appear here as the data owner records each unit; a chunk of parallel fetches completes as several units at once.')}</p>`;
      return W.logShell(PREPARATION,{label:t('Recent listing activity'),title:t('Recent listing activity'),caption:t('Units the data owner recorded · newest last · a bounded snapshot, not a complete history'),history,moving,rows:rows.map(listingRow),empty,status,rail});
    }
    if(shown==='prepare_features') return featureAside(PREPARATION,b.work_progress,state,t(b.progress?.status ? 'The Feature owner reports its status only: {status}.' : 'The Feature owner has not reported a step yet.',{status:codeWords(b.progress?.status || '')}));
    return null; // the stage's retained record
  }
  const PREPARATION={key:'preparation',page:'overview',title:'Preparation work',railLabel:'Preparation stages; select to inspect, not execute',factsLabel:'Preparation facts',steps:PREPARATION_STEPS,lines:stageLines(PREPARATION_STEPS.map(([id])=>id)),fallbackLine:'Continuing the reported preparation stage.',
    oneUnit:{freeze_sources:'The sources are captured as one unit: the exact candidate list and its evidence.',publish_inputs:'The input is published as one unit from the panel.',verify_inputs:'The published input is verified as one unit by the input owner.'},
    view:()=>states.get('welcome')?.view,work,aside:stageAside,absorb:absorbListings,completedNow:(b)=>Boolean(b.inputs?.length)};
  /* The preparation while it runs or waits (round 74): the attention block, the work area, the
   * liveness line with the cancel beside it; the Home's head and the pinned run say the state. */
  function scene(b, v, state) {
    const life=v.lifecycle, id=v.task_id;
    const by=Object.fromEntries(v.actions.map(a=>[a.action,a])), cancel=by.CANCEL;
    const stale=state.stale;
    const live=v.liveness, observedNow=life==='RUNNING' && live.status==='OBSERVED' && !stale;
    const cancelControl=stateMoving(life) ? typedBtn(t('Request cancel'),'task-cancel',id,'button compact',S.busy || !cancel?.available ? (cancel?.reason || '') : '') : '';
    const liveness=html`<p class="prep-liveness"${stale ? ' data-retained="true"' : ''}>${icon(observedNow ? 'activity' : 'info')}<span>${stale ? html`${t('Last observation at {time}, not re-read since:',{time:readWhen(state)})} ` : ''}${LiveTasks.liveness(v) || t('The Task is not executing; its lifecycle, not a heartbeat, says where it stands.')}${v.operation_running ? html` · ${t('the operation has not returned')}` : ''}</span>${cancelControl}</p>`;
    return html`${disconnected(state)}${attention(b,v,state)}${W.workArea(PREPARATION,b,v,state)}${liveness}<section class="panel prep-kept">${kept(v)}<p class="caption">${t('Closing or minimizing this page does not stop the preparation: it continues while the local service runs. Stopping the service or the computer interrupts it; the recorded stages stay, and a restarted service takes the Task up from its recorded evidence.')}</p><p class="caption">${t('Stage counts are not time estimates; no rate or remaining time is inferred. Missing telemetry is named, not invented.')}</p></section>${detail({readback:b,recovery:v})}`;
  }
  // The reader's own outage is the page's fact, kept apart from the Task's: what is shown is
  // the last owner observation, nothing is signalled as fresh, and a successful read recovers.
  const disconnected=(state)=>state.stale ? html`<section class="panel prep-attention prep-disconnected" data-box="decision" data-disconnected="true"><div class="tp-callout attention">${icon('warning')}<p><strong>${t('Owners not reachable')}</strong> · <span class="mono">${state.stale.error}</span><br>${t('This page could not read the owners since {time} ({m}). What is shown is their last observation; whether the Task is still running is not known here and nothing is inferred from time. Reading continues; a successful read replaces this notice.',{time:W.sinceWhen(state),m:countText(state.stale.failures,'{n} failed read','{n} failed reads')})}</p></div></section>` : '';
  /* The viewer's explicit choices on the area, delegated to the shared work area. The way back
   * to the latest Task of the page's kind from a refused or another Task. */
  const fold=()=>W.fold(), inspect=(stageId)=>W.inspect(stageId), follow=()=>W.follow();
  function current() { const page=SCENE_ROUTE[here()] ? here() : 'welcome'; if(typeof replaceHash==='function') replaceHash({[SCENE_ROUTE[page]]:''}); return refresh(page); }
  function refusedScene(state, page) {
    const refused=refusalOf(state.error);
    const head=page==='welcome' ? '' : objectHead(t('Data'),html`<span class="mono">${short(selectedTask(page) || '',SHORT.id)}</span> · ${t('not read')}`,typedBtn(t('Open the current data state'),'workspace-current','','button compact',''));
    return html`${head}${refusal({code:refused,reason:html`${t(REFUSALS[refused])} ${t(page==='welcome' ? 'Nothing else was read or changed; the workspace\'s current preparation is one step away.' : 'Nothing else was read or changed; the workspace\'s current data state is one step away.')}`},'warning',{word:t(page==='welcome' ? 'This preparation Task cannot be shown' : 'This update Task cannot be shown'),action:page==='welcome' ? typedBtn(t('Open the current preparation'),'workspace-current','','button compact','') : ''})}`;
  }
  /* First use on the Home (round 74): the preparation as the Home's one run. Before a Task: the
   * facts and the PLAN preview; while it runs or waits: the work area; once the input is
   * verified: nothing -- the Home is the Home. No head of its own. */
  /* The owner cannot plan a preparation while the workspace itself is not readable (its manifest
   * absent or refused): the page says why, and offers no way the owner will refuse. */
  const CANNOT_PREPARE=/research_workspace\.(manifest_unreadable|manifest_refused|nonempty_manifest_absent|manifest_from_newer_build|prepared_before_renames)/;
  const prepareRefused=()=>CANNOT_PREPARE.test(String(S.error || states.get('welcome')?.error || ''));
  /* What a workspace the owner cannot prepare means, and the ways on (the user, 2026-09-24: 然后呢? 什么提示都没
   * 有): the owner's words, then what the reader can do -- restore the manifest where the service log names
   * it, or work in another workspace (the workspace menu's own two ways) -- the code small beneath. */
  function cannotPrepare() {
    const said=String(S.error || states.get('welcome')?.error || ''), code=(said.match(CANNOT_PREPARE) || [])[0];
    if(!code) return '';
    // U77 (V481): a workspace prepared before the 2026-10-02 renames reads as the door words it -- the retired spelling it
    // holds, nothing in it changed -- and its way on is a new workspace, never a restore
    if(code.endsWith('prepared_before_renames')) {
      const door=(said.match(/research_workspace\.prepared_before_renames(?::\S*)?: ([\s\S]+)$/) || [])[1] || '';
      return banner(t('This release cannot open this workspace'),html`${door ? t(door.trim()) : explainCode(code)}${infoMark(code)}`,'warning',html`<div class="flow">${btn(t('Create a workspace'),'workspace-how','create','button primary')}${btn(t('Open an existing workspace'),'workspace-how','open','button')}</div>`,'warning');
    }
    return banner(t('This workspace cannot be prepared'),html`${explainCode(code)} ${t(code.endsWith('newer_build') ? 'To research here, open it with the build that wrote it or a newer one, or work in another workspace.' : 'To research here, restore its manifest where the service log names it, or work in another workspace.')}${infoMark(code)}`,'warning',html`<div class="flow">${btn(t('Open an existing workspace'),'workspace-how','open','button primary')}${btn(t('Create a workspace'),'workspace-how','create','button')}</div>`,'warning');
  }
  function firstUse() {
    const state=states.get('welcome');
    if(!state){void refresh('welcome');return skeleton('rows');}
    const b=state.body, v=state.view;
    if(refusalOf(state.error)) return refusedScene(state,'welcome');
    const refused=prepareRefused() ? cannotPrepare() : '';
    const notices=html`${refused || (S.error||(state.error && !state.stale)?notRead(t('Workspace action needs attention'),S.error||state.error):'')}${noticeFor('welcome')?noteLine(t('Last operation'),t(noticeFor('welcome'))):''}${state.loading && !b ? skeleton('rows') : ''}`;
    if(!b) return notices;
    if(!b.task_id) return html`${b.inputs?.length ? '' : unprepared(b,state)}${notices}`;
    if(!v) return html`${notices}${notRead(t('Task not read'),state.viewError,t('The preparation owner names a Task, but its recovery view was not read. Tasks and the activity feed are unaffected; refresh to ask again.'))}`;
    if(v.lifecycle==='SUCCEEDED' && b.inputs?.length) return notices;
    return html`${scene(b,v,state)}${notices}`;
  }
  /* The next step of an unprepared or preparing workspace, discoverable from any entry page
   * (never on the scene itself, never forcing a route). Facts: the session context's
   * preparation readback and the Task's projection as the activity feed keeps it fresh. */
  function notice() {
    const p=Data.preparation();
    if(!p || p.inputs?.length || here()==='welcome') return ''; // the Home hosts the scene
    const projection=p.task_id ? Data.tasks().find(v=>v.task_id===p.task_id) : null;
    const life=projection?.lifecycle || (p.task_id ? p.status : null);
    const moving=stateMoving(life);
    const title=!p.task_id ? t('This workspace is not prepared for research yet') : moving ? t('Preparation is running · stage {i} of {n}',{i:(projection?.verified_stage_count ?? 0)+1,n:projection?.total_stage_count ?? 5}) : life==='SUCCEEDED' ? t('Preparation completed') : t('Preparation needs attention');
    const text=!p.task_id ? t('One explicit preparation makes it usable; opening pages starts nothing.') : moving ? t('It continues while the local service runs; open the scene to see where it stands.') : life==='SUCCEEDED' ? t('Open the scene to select the verified input into a draft.') : html`${t('Recorded as')} ${codeWords(life)}${projection?.latest_failure_code || p.failure_code ? html` · <span class="mono">${projection?.latest_failure_code || p.failure_code}</span>` : ''}`;
    // The projection is the activity feed's; while that feed cannot be read the notice says so
    // instead of presenting the last projection as current.
    const outage=typeof LiveActivity!=='undefined' && LiveActivity.state && LiveActivity.state().error;
    const retained=outage && p.task_id ? html` <span class="muted">${t('Last observation; the owners are not reachable now.')}</span>` : '';
    return html`<div class="banner ${p.task_id && !moving && life!=='SUCCEEDED' ? 'warning' : 'neutral'} prep-notice" role="status" data-preparation-notice="${life || 'unprepared'}"${outage ? ' data-retained="true"' : ''}>${icon(moving && !outage ? 'activity' : p.task_id ? 'task' : 'cube')}<div class="grow"><strong>${title}</strong><p>${text}${retained}</p></div>${link(t(p.task_id ? 'Open preparation' : 'Prepare workspace'),'overview','button compact',p.task_id ? {preparation:p.task_id} : {})}</div>`;
  }

  /* ---- the data page: freshness, one explicit update, the update scene ---- */
  const UPDATE_STEPS=['validate_update_request','maintain_data_feature','publish_update_receipt'].map((id)=>[id,STAGES[id].word,STAGES[id].note]); // the stage words are the one table's (status.js)
  const UPDATE_LINES={validate_update_request:'Checking that the workspace still matches the plan this Task was admitted with: the same membership revision, data revision and Panel; a moved workspace stops here by name.',maintain_data_feature:'Fetching the sessions after the current cutoff for every admitted member, one bounded cycle at a time; then quality governance, the Features and the Panel for the new sessions.',publish_update_receipt:'Sealing one receipt: the cutoffs before and after, the maintenance cycle and its effects. Published input versions are not touched.'};
  const PHASE_TEXT={preflight:'preflight: sources and membership',market_data:'market data: fetching and auditing listings',quality:'quality governance',feature:'Features and Panel',completed:'completed'};
  /* The maintenance stage's own count: the maintenance runner's units while the cycle fetches
   * (from its durable rows), the Feature owner's step while it builds, the cycle's phase
   * otherwise; every denominator is the owner's. */
  function updateWork(b, v, state, shown=null) {
    const stage=shown || v.status.current_stage, c=b.cycle, m=b.maintenance, w=b.work_progress;
    const wait=null; // N6 (law 58): a wait is said once, by the update's banner
    if(stage!=='maintain_data_feature') return NO_COUNT(null,stageVerified(v,stage));
    if(!c) return {count:html`<span class="tp-denom">${t('The maintenance cycle has not been admitted yet')}</span>`,bar:null,detail:t('The owner admits one cycle for this plan when the stage starts; its phase and units appear here.'),wait};
    const phase=t(PHASE_TEXT[c.phase] || c.phase), cycleAt=c.updated_at ? t('cycle {phase} · recorded {time}',{phase,time:when(c.updated_at)}) : t('cycle {phase}',{phase}); // N6 (laws 57, 133): the cycle's instant, never an age
    if(w?.availability==='BOUND' && w.stage_id!=='market_data_increment' && c.phase!=='completed') { const step=featureStep(w,state,wait); return {...step,detail:html`${step.detail} · ${cycleAt}`}; }
    if(c.phase==='completed' || c.status==='noop') {
      const cs=c.change_set;
      return {count:html`<span class="tp-denom">${cs ? t('{n} with new sessions · {c} with corrected history · membership {a} in, {o} out',{n:countText(cs.listings_with_new_sessions,'{n} listing','{n} listings'),c:countText(cs.listings_with_corrections,'{n} listing','{n} listings'),a:cs.membership_additions,o:cs.membership_removals}) : t('The cycle recorded no market-data change: the workspace was already current')}</span>`,bar:null,detail:cycleAt,wait};
    }
    if(m && m.availability && m.availability!=='AVAILABLE') return {count:html`<span class="tp-denom">${t(m.availability==='UNREADABLE' ? 'The maintenance units of this update could not be read' : 'The maintenance units of this update are not available')}</span>`,bar:null,detail:html`${said(m.failure_code)} · ${cycleAt}`,wait};
    if(m && m.counts) {
      // The maintenance runner's units: processed = updated or failed, over the listings the
      // cycle admitted; the count while the cycle fetches, and still the fact of what was
      // processed when a later phase (quality, Features) holds or stops.
      const n=m.counts.processed, total=m.counts.listings;
      return {count:html`<span class="tp-number">${n.toLocaleString('en-US')}</span><span class="tp-denom"> / ${total.toLocaleString('en-US')} ${t('listings processed')}</span>`,bar:[n,total],
        detail:html`${unitCounts(m.counts)} · ${cycleAt}`,wait};
    }
    return {count:html`<span class="tp-denom">${t('No finer count is reported for this step')}</span>`,bar:null,detail:cycleAt,wait};
  }
  /* The maintenance runner's durable listing units, one row per listing (its latest state),
   * newest last; a row first seen on a live read is an arrival once. */
  function absorbMaintenance(b, state, A=W.A) {
    const m=b.maintenance;
    if(!m || !Array.isArray(m.rows)) return;
    const log=W.freshLog(m.maintenance_id,'maintain_data_feature',A), live=W.liveRead(state,A);
    for(const r of m.rows) {
      const rowKey=r.listing_id+'|'+r.state+'|'+r.updated_at;
      if(log.seen.has(rowKey)) continue;
      log.seen.add(rowKey);
      log.rows.delete(r.listing_id); // re-insert last: the map keeps the runner's update order
      log.rows.set(r.listing_id,{...r,attention:r.state==='FAILED' || (r.sentinel_disposition && r.sentinel_disposition!=='ANCHORED')});
      if(live) A.arrived.add(r.listing_id);
    }
    W.retain(log);
    log.snapshot={counts:m.counts,moved:m.moved,retained:m.retained,lifecycle:m.lifecycle,as_of:m.as_of_session};
  }
  /* What one unit did, in the runner's own record: fetched new sessions (with corrections
   * beside, when any), corrected history without a new session, re-verified history with
   * nothing new, or failed. The same word names it in the row's meta and the latest-unit rail. */
  // the runner's counts that are not none, each in its words (N6: a zero tally is not shown, law 81)
  const unitCounts=(n)=>[[n.with_new_sessions,'{n} with new sessions'],[n.corrected,'{n} corrected'],[n.reverified,'{n} re-verified'],[n.failed,'{n} failed'],[n.pending,'{n} pending']].filter(([x])=>Number(x)>0).map(([x,w])=>t(w,{n:count(x)})).join(' · ') || t('nothing recorded yet');
  const unitKind=(r)=>r.state==='FAILED' ? 'failed' : r.state!=='UPDATED' ? 'pending' : r.new_sessions?.length ? 'new data' : r.restated_sessions ? 'corrected' : 're-verified';
  const unitText=(r)=>r.state==='FAILED' ? html`${t('failed')} · ${said(r.failure_code)}` : r.new_sessions?.length ? html`${pluralText(r.new_sessions.length,'{n} new session fetched through {d}','{n} new sessions fetched through {d}',{n:count(r.new_sessions.length),d:r.raw_through || r.new_sessions.at(-1)})}${r.restated_sessions ? html` · ${countText(r.restated_sessions,'{n} earlier session corrected','{n} earlier sessions corrected')}` : ''}` : r.restated_sessions ? pluralText(r.restated_sessions,'{n} earlier session corrected in retained history through {d}; no new session','{n} earlier sessions corrected in retained history through {d}; no new session',{n:count(r.restated_sessions),d:r.raw_through || ''}) : t('history re-verified through {d}; nothing new to fetch, nothing corrected',{d:r.raw_through || ''});
  /* One listing, one line (N6): the time, the symbol, what the runner recorded, the audit; the
   * listing's identity is the symbol's tip. */
  function maintenanceRow(r) {
    const extra=[r.audit_scope==='FULL' ? t('full-history audit') : r.audit_scope==='ROLLING' ? t('rolling audit') : '',r.sentinel_disposition && r.sentinel_disposition!=='ANCHORED' ? html`${t('price action')} ${said(r.sentinel_disposition)}` : '',r.attempt_count>1 ? t('attempt {n}',{n:r.attempt_count}) : ''].filter(Boolean);
    return html`<article class="fv-log-row prep-unit${r.attention ? ' attention' : ''}" data-listing-key="${r.listing_id}"><div class="fv-log-time"><time datetime="${r.updated_at}">${clockOf(r.updated_at)}</time><span class="fv-log-dot" aria-hidden="true"></span></div><div class="fv-log-entry"><strong data-tip="${r.listing_id}">${r.symbol}</strong><span class="fv-log-object">${unitText(r)}</span>${extra.map(x=>html`<span class="fv-log-by">${x}</span>`)}</div></article>`;
  }
  function updateAside(b, v, state, shown, current) {
    const c=b.cycle, m=b.maintenance, w=b.work_progress, log=W.A.log, moving=current && v.lifecycle==='RUNNING' && v.liveness.status==='OBSERVED' && !state.stale;
    if(shown!=='maintain_data_feature') return null;
    if(c && w?.availability==='BOUND' && w.stage_id!=='market_data_increment' && c.phase!=='completed') return featureAside(UPDATE,w,state,t('The Feature owner has not reported a step yet.'));
    // No units of this cycle's own: not admitted yet while it moves; none recorded once it
    // settled (a transition brings its members' data through its own path, and a cycle whose
    // data was current fetched nothing) -- the change set above is then the cycle's record.
    const unitsShell=(word,text)=>W.factsShell(UPDATE,{label:t('Recent listing activity'),title:t('Recent listing activity'),caption:t('Units the maintenance runner recorded for this update'),facts:html`<p class="caption">${text}</p>`,rail:html`<section class="ui-log-rail prep-log-rail"><div><span>${t('Maintenance units')}</span><strong>${word}</strong></div><div><span>${t('Reading')}</span><strong>${t('Following latest')}</strong></div></section>`});
    if(!m && c && (c.phase==='completed' || ['completed','noop','cancelled'].includes(c.status))) return unitsShell(t('none recorded'),t('This cycle recorded no maintenance units of its own: its market data came through another owner\'s path (a membership transition) or needed no fetch. What changed is the cycle\'s change set, read above; nothing of another update is shown here.'));
    if(!m) return featureAside(UPDATE,w,state,t(c ? 'The maintenance runner has not admitted its listing units yet ({phase}).' : 'The maintenance cycle has not been admitted yet; listing units appear as the runner records them.',{phase:t(PHASE_TEXT[c?.phase] || c?.phase || '')}));
    if(m.availability && m.availability!=='AVAILABLE') return unitsShell(t(m.availability==='UNREADABLE' ? 'unreadable' : 'not available'),html`${t(m.availability==='UNREADABLE' ? 'The maintenance units recorded for this update could not be read; nothing of another update is shown in their place.' : 'The maintenance units recorded for this update are not available in this workspace; nothing of another update is shown in their place.')} ${said(m.failure_code)}`);
    const rows=log ? [...log.rows.values()] : [], history=!current || !stateMoving(v.lifecycle);
    const latest=rows.at(-1), snapshot=log?.snapshot;
    const rail=W.logRail({label:history ? t('Last recorded unit') : t('Latest unit'),value:latest ? html`${latest.symbol} · ${t(unitKind(latest))} · ${t('recorded {time}',{time:when(latest.updated_at)})}` : t('No unit recorded yet')},state); // N6 (law 133): the unit's instant, never an age
    const status=html`${t('{m} of {l} listings moved · {r} shown',{m:snapshot?.moved ?? m.moved,l:m.counts.listings,r:snapshot?.retained ?? m.retained})}${history ? html` · ${t('recorded work of this update, not current activity')}` : ''}${state.stale ? html` · ${t('last successful read')}` : ''}`;
    const empty=html`<p class="caption prep-log-empty">${t('Listings appear here as the maintenance runner records each unit; a chunk of parallel fetches completes as several units at once.')}</p>`;
    return W.logShell(UPDATE,{label:t('Recent listing activity'),title:t('Recent listing activity'),caption:t('Units the maintenance runner recorded for this target session · newest last · one row per listing, the newest shown'),history,moving,rows:rows.map(maintenanceRow),empty,status,rail});
  }
  const UPDATE={key:'update',page:'data',title:'Update work',railLabel:'Update stages; select to inspect, not execute',factsLabel:'Update facts',steps:UPDATE_STEPS,lines:UPDATE_LINES,fallbackLine:'Continuing the reported update stage.',
    oneUnit:{validate_update_request:'The request is validated as one unit against the workspace as it stands.',publish_update_receipt:'The receipt is sealed as one unit.'},
    view:()=>states.get('data')?.view,work:updateWork,aside:updateAside,absorb:absorbMaintenance,completedNow:(b)=>b.status==='PUBLISHED'};
  /* The update's stop or wait, named once with its way on (N6, laws 57-58): what stopped it or
   * what it waits for, in words, and whether the reader must act; the one way on is the primary.
   * The owner's code is the title's tip and the Facts'. A case the data owner asks about comes
   * first -- Data issues decides and the update continues after -- whatever the lifecycle says. */
  function updateStop(b, v, state) {
    const life=v.lifecycle, id=v.task_id, c=b.cycle;
    const by=Object.fromEntries(v.actions.map(a=>[a.action,a]));
    const off=(a)=>S.busy || !a?.available ? (a?.reason || '') : '';
    const code=v.status?.latest_failure_code || c?.failure_code || v.stop?.code || '';
    const wait=b.retry_after_at || c?.retry_after_at || null;
    const due=!wait || (Date.parse(wait) <= Date.parse(v.observed_at)+readAge(state)*1000);
    const ask=issuesAsking();
    let tone='warning', ic='warning', title='', words='', way='', waits=false; // waits: the stopped update waits for the reader now (law 148: a decision whatever its tone)
    switch(life) {
      case 'RECOVERY_REQUIRED':
        title=t('The update was interrupted'); words=t('Resuming is confirmed against this exact Task version; the maintenance work already recorded is kept and verified again.');
        way=typedBtn(t('Resume this Task'),'task-recovery',id,'button primary',off(by.RECOVER)); break;
      case 'BLOCKED': case 'DEFERRED': case 'REVIEW_PENDING':
        if(ask.count || (code==='data.truth_review_required' && !ask.decided)) { title=t('A data decision is needed'); words=ask.names.length ? t('The data owner asks how to treat {names}; the update continues after you decide.',{names:ask.names.join(', ')}) : explain(code); way=link(t('Decide on Data issues'),'issues','button primary'); break; } // the owner's code says so before its cases are read
        if(life==='DEFERRED' && !due) { tone='neutral'; ic='clock'; title=t('Waiting until {time}',{time:when(wait)}); words=b.detail ? deferWords(b.detail) : t(code==='data.remediation_wait' ? 'The owner waits for the source to recover, as you decided. Nothing is needed from you until then; after it, continue the update: nothing is retried on its own.' : 'The owner waits for the condition it named. Nothing is needed from you until then; after it, continue the update: nothing is retried on its own.'); way=action(resumeOf(b) ? 'Resume this update' : 'Continue this update','preview','continue-update',t('Not due until {time}',{time:when(wait)})); break; }
        if(ask.decided) { tone='neutral'; ic='checkcircle'; waits=true; title=t('Your decision is recorded'); words=t('Continue the update: the owner checks the evidence again and applies your decision; nothing recorded is undone.'); }
        else if(life==='DEFERRED') { tone='neutral'; ic='clock'; waits=true; title=t('The wait is over'); words=html`${b.detail ? html`${deferWords(b.detail)} ` : ''}${t('Continue the update: the owner tries again under the same plan; listing work already recorded is not fetched again.')}`; }
        else { title=t('The update stopped'); words=html`${explain(code) || v.stop?.detail || ''} ${t('Retrying runs the stopped stage again under the same plan; if the cause is unchanged it stops there again.')}${causeLine(b.failure_cause || v.stop?.cause,true)}`; } // U71: what its owner saw beside the code
        way=action(life==='DEFERRED' && resumeOf(b) && !ask.decided ? 'Resume this update' : ask.decided || life==='DEFERRED' ? 'Continue this update' : 'Retry this update','preview','continue-update','','button primary'); break;
      case 'CANCEL_REQUESTED':
        tone='neutral'; ic='clock'; title=t('Stopping at the next safe checkpoint'); words=t('The worker runs until its next safe checkpoint, a maintenance cycle boundary; nothing already recorded is undone.'); break;
      case 'RUNNING': case 'QUEUED':
        if(v.liveness.status==='NOT_RECENT') { title=t('No recent heartbeat'); words=t('The worker may still be running a long unit, or the service that ran it may have stopped; this page does not decide. Read again once the service settles.'); break; }
        if(v.worker_failure) { title=t('The last command stopped outside Task Control'); words=t('The Task\'s own lifecycle says where it stands.'); break; }
        return '';
      default: return '';
    }
    return banner(html`<span${code ? html` class="coded" data-tip="${code}"` : ''}>${title}</span>`,words,tone,way,ic,tone!=='neutral' || waits);
  }
  /* The cases the data owner asks about now, from its own readback (read once, quietly, where it
   * was not read): the Data page's stop, the Home's and Tasks' way on read the same count. */
  function issuesAsking() {
    const s=states.get('issues');
    if(!s) { void refresh('issues','',true); return {count:0,decided:0,names:[],read:false}; }
    const list=s.body?.issues || [], asking=list.filter(x=>['AWAITING_CHOICE','OPTION_REFUSED'].includes(x.status));
    return {count:asking.length,agent:asking.length>0 && asking.every(issueAgent),decided:list.filter(x=>x.status==='CONFIRMED_PENDING_REVALIDATION').length,names:asking.flatMap(x=>(x.case?.listing_ids || []).map(id=>x.subjects?.[id] || short(id,SHORT.id))),read:Boolean(s.body)};
  }
  /* A held data update as the Home and Tasks name it (N6, law 58): the Data page's title and way
   * on -- a case that asks first, then a recorded decision, then the owner's code in words. */
  function stopWords(v) {
    const life=v?.lifecycle || v?.status;
    if(v?.task_kind!=='workspace_data_update' || !stateHeld(life)) return null;
    const ask=issuesAsking(), code=v.latest_failure_code || '';
    if(ask.count || (code==='data.truth_review_required' && !ask.decided)) return {title:t('A data decision is needed'),next:t('decide on Data issues'),code};
    if(ask.decided) return {title:t('Your decision is recorded'),next:t('continue the update on the Data page'),code};
    return {title:code ? codeWords(code) : t(stateOf(life).word),next:wayOn(code,life),code};
  }
  /* The membership's facts (N6): the Listings and Market sessions figures open them in the Facts
   * pane -- the bootstrap cohort and its history, the sessions, the assumption, the journal. */
  function membershipFacts(b) {
    const m=b.membership, i=b.inputs || {}, sessions=get('storage')?.budget?.research_session_count;
    const rows=m ? [[t('Membership checked'),when(i.sources_checked_at)],[t('Bootstrap cohort'),html`${countText(m.bootstrap.cohort_size,'{n} listing','{n} listings')} · <span class="numeric">${m.bootstrap.t0_session}</span>`],[t('History from'),html`<span class="numeric">${m.bootstrap.history_start}</span>`],...(sessions ? [[t('Market sessions'),count(sessions)]] : []),[t('Assumption'),t(m.bootstrap.initialization_assumption==='INITIAL_COHORT_BACKFILL_NOT_POINT_IN_TIME' ? 'The initial cohort is backfilled; it is not point-in-time membership.' : m.bootstrap.initialization_assumption)],[t('Membership journal'),m.journal_sequence ? pluralText(m.journal_sequence,'{n} event, latest effective {d}','{n} events, latest effective {d}',{n:count(m.journal_sequence),d:m.latest_effective_session || ''}) : t('no membership event since the bootstrap')]] : [[t('Membership checked'),when(i.sources_checked_at)],[t('Journal'),t('not readable')]];
    if(b.sector_reference) rows.push([t('Sector reference'),html`<span class="mono">${short(b.sector_reference.revision || JSON.stringify(b.sector_reference),SHORT.hash)}</span>`]);
    const body=html`${kv(rows)}${m ? detail({membership:m},'Membership journal and source observations') : ''}`;
    const id='facts-membership'; // one membership a page
    return {id,template:html`<template data-facts-id="${id}" data-facts-title="${t('Sources & membership')}">${body}</template>`};
  }
  /* The Data overview's figures (N6, the user's Sentry reading): the as-of, the listings, the
   * sessions, the open cases, the input versions and the storage used, each opening its page --
   * the as-of the update that set it, the listings and sessions the membership's facts. A figure
   * an owner has not answered yet is its skeleton, never a guess. */
  // Keep the inert membership facts with the figures that open them; it cannot interrupt
  // the shared stack between the following visible sections.
  function dataFigures(b, membership) {
    const i=b.inputs || {}, m=b.maintenance, store=get('storage');
    const pending=(x)=>x==null ? html`<span class="ph ph-short" aria-label="${t('Reading')}"></span>` : x;
    const listings=m?.counts?.listings ?? b.membership?.bootstrap?.cohort_size ?? store?.budget?.active_listing_count;
    const sessions=store ? (store.budget?.research_session_count != null ? count(store.budget.research_session_count) : '') : null;
    const asOf=i.panel_through || i.data_through;
    const asOfWords=i.data_through && i.panel_through && i.data_through!==i.panel_through ? t('Features through {f}; market data through {d}',{f:i.panel_through,d:i.data_through}) : t('Market data and Features through this session');
    return measureStrip(html`${figureTile(t('As of'),asOf || '',b.task_id ? {action:'workspace-task',value:b.task_id} : null,asOfWords)}${figureTile(t('Listings'),listings==null ? '' : count(listings),{action:'facts-open',value:membership.id},t('The listings the data owner maintains; the initial cohort is backfilled, not point-in-time membership.'))}${store && store.budget?.research_session_count == null ? '' : figureTile(t('Market sessions'),pending(sessions),{action:'facts-open',value:membership.id},b.membership?.bootstrap?.history_start ? t('Market sessions from {d}',{d:b.membership.bootstrap.history_start}) : '')}${membership.template}`, t('Data figures'), 'data-figures');
  }
  /* What the last update did to each listing (N6; the user, 2026-09-24: the figure, its legend and
   * the update were one update's facts in three places): the maintenance runner's own counts, said
   * on the completed update itself under its row -- a count of none is not said; the (i) says whose
   * record it is. A moving update's work has its own counts. */
  function lastUpdateCounts(b) {
    const m=b.maintenance, n=m?.counts;
    if(!n || (m.availability && m.availability!=='AVAILABLE') || !n.listings) return '';
    const parts=[[t('Listings'),n.listings],[t('New sessions fetched'),n.with_new_sessions],[t('History corrected'),n.corrected],[t('Re-verified, nothing new'),n.reverified],[t('Failed'),n.failed],[t('Pending'),n.pending]].filter(([,x])=>Number(x)>0);
    return html`<p class="data-update-counts">${parts.map(([word,x],i)=>html`${i ? ' · ' : ''}${word} <b class="num">${count(x)}</b>`)}${infoMark(t('The maintenance runner\'s own record for each listing the update admitted for {d}.',{d:m.as_of_session}))}</p>`;
  }
  const SETTLED=new Set(['SUCCEEDED','CANCELLED','FAILED']);
  /* The updates as rows (N6): the one this page reads first -- its line the change it made, the
   * stage it runs, or its target -- with its work open under it only while it runs or waits; the
   * earlier updates and preparations by day. A row opens its Task. */
  function updateRows(b, v, state) {
    const runs=Data.runsOf('update'), open=!SETTLED.has(v.lifecycle), r=b.receipt;
    const latest=runs.find(x=>x.id===b.task_id) || {id:v.task_id,kind:'workspace_data_update',name:codeWords('workspace_data_update'),state:v.lifecycle,starter:'',started:v.status?.running_since || '',finished:stateMoving(v.lifecycle) ? '' : v.status?.last_activity_at || '',current:v.status?.current_stage || '',verified:[v.verified_stage_count,v.total_stage_count],object:v};
    const target=b.maintenance?.as_of_session || r?.target_session;
    const why=v.lifecycle==='SUCCEEDED' ? (r ? dateMove(r.before.data_through,r.after.data_through) : t('its receipt is being read')) : stateMoving(v.lifecycle) && latest.current ? html`${t('stage')} ${t((UPDATE_STEPS.find(([k])=>k===latest.current) || [0,codeWords(latest.current)])[1])}` : target ? t('target {d}',{d:target}) : codeWords(v.lifecycle);
    const row=runRow({...latest,state:v.lifecycle},{why,columns:['verified'],props:[open ? html`${count(v.verified_stage_count)} / ${count(v.total_stage_count)} ${t('verified')}` : '']});
    const earlier=runs.filter(x=>x.id!==latest.id);
    return html`<div class="card-list lines slotted data-update-row">${row}${v.lifecycle==='SUCCEEDED' ? lastUpdateCounts(b) : ''}</div>${open ? updateScene(b,v,state) : ''}${earlier.length ? html`<div class="card-list lines slotted">${[...Data.groupByDay(earlier)].map(([g,rs])=>html`${groupHead(g,rs.length)}${rs.map(x=>runRow(x,{inDay:true}))}`)}</div>` : ''}`;
  }
  /* The update's work while it runs or waits (N6): its stop or wait named once, the shared work
   * area with maintenance's own stages and units, and -- while it moves -- its liveness beside the
   * cancel. A settled update is its row alone. */
  function updateScene(b, v, state) {
    const life=v.lifecycle, id=v.task_id, stale=state.stale;
    const live=v.liveness, observedNow=life==='RUNNING' && live.status==='OBSERVED' && !stale;
    const by=Object.fromEntries(v.actions.map(a=>[a.action,a])), cancel=by.CANCEL;
    const cancelControl=stateMoving(life) ? typedBtn(t('Request cancel'),'task-cancel',id,'button compact',S.busy || !cancel?.available ? (cancel?.reason || '') : '') : '';
    const liveness=stateMoving(life) ? html`<p class="prep-liveness"${stale ? ' data-retained="true"' : ''}>${icon(observedNow ? 'activity' : 'info')}<span>${stale ? html`${t('Last observation at {time}, not re-read since:',{time:readWhen(state)})} ` : ''}${LiveTasks.liveness(v) || t('No heartbeat recorded yet · liveness unknown')}${v.operation_running ? html` · ${t('the operation has not returned')}` : ''}</span>${cancelControl}</p>` : '';
    const earlier=b.selected && b.latest_task_id && b.latest_task_id!==id ? noteLine(t('An earlier update'),t('the latest is {id}',{id:short(b.latest_task_id)}),'info',action('Open the latest update','current')) : '';
    return html`<section class="prep-scene update-open" data-stack-box="box" data-scene="update" data-lifecycle="${life}">${earlier}${disconnected(state)}${updateStop(b,v,state)}${W.workArea(UPDATE,b,v,state)}${liveness}</section>`;
  }
  const SCENES={welcome:PREPARATION,data:UPDATE};
  /* ST1/WD7: the selected update owner's retained tail account, also present when coverage
   * stops publication. Counts and dates are read as recorded; recent listing activity is no source. */
  function removedMemberTails(b) {
    return html`${(b.removed_member_tails || []).map(row=>noteLine(t('{symbol} left the universe; its last {n} sessions could not be verified and are treated as missing data under the coverage rules.',{symbol:row.symbol,n:row.missing_session_count}),'','neutral'))}`;
  }
  /* The Data overview (N6; law 129: the top row's one verb, not while an update runs or waits --
   * its way on is under its row): the figures, then the updates -- the last one with its counts. */
  function data(state) {
    const b=state.body, v=state.view;
    if(refusalOf(state.error)) return refusedScene(state,'data');
    const info=html`<p class="lede">${t('The working store: one explicit update at a time; publishing an input and selecting one are separate steps.')}</p>`;
    if(notConfigured(state)) {
      // whether the workspace can be prepared is Home's owner's answer, read quietly here where it is not yet
      if(!states.get('welcome')) void refresh('welcome','',true);
      if(!states.get('issues')) void refresh('issues','',true);
      const preparation=states.get('welcome');
      return html`${objectHead(t('Data maintenance'),info)}${prepareRefused() ? cannotPrepare() : preparation?.body?.task_id ? firstUse() : !preparation?.body ? skeleton('rows') : emptyState(t('No maintained data yet'),link(t('Prepare workspace'),'overview','button primary'),'page-empty')}${issuesSection(true)}`;
    }
    const settled=!v || SETTLED.has(v.lifecycle), between=b?.next_action==='DATA_UPDATE_PLAN';
    const head=objectHead(t('Data maintenance'),info,b?.task_id && settled && !between ? action('Preview update','preview','update') : '');
    const notices=html`${S.error||(state.error && !state.stale)?notRead(t('Workspace action needs attention'),S.error||state.error):''}${noticeFor('data')?noteLine(t('Last operation'),t(noticeFor('data'))):''}${state.loading && !b ? skeleton('head') : ''}`;
    if(!b) return html`${head}${notices}`;
    const blocker=between ? banner(t('The workspace is between states'),t('The data owner asks for a new update plan before anything else; previewing one names the pending transition.'),'warning',primary('Preview update','preview','update')) : b.current_input_failure ? notRead(t('Current inputs not readable'),b.current_input_failure,t('The working store\'s current inputs could not be read; the update owner names the cause. Nothing is inferred from it.')) : '';
    const membership=membershipFacts(b);
    const updates=!b.task_id ? emptyState(t('No update recorded yet'),between ? '' : primary('Preview update','preview','update')) : v ? updateRows(b,v,state) : notRead(t('Task not read'),state.viewError,t('The update owner names a Task, but its recovery view was not read. Tasks and the activity feed are unaffected; refresh to ask again.'));
    return html`${head}${notices}${dataFigures(b,membership)}${blocker}${removedMemberTails(b)}${updates}${issuesSection()}${detail({readback:b,recovery:v},'Exact identity, scope and receipt')}`;
  }
  /* Data issues are Data maintenance's (N5 of the open-issues plan; the user, 2026-09-24): the owner's cases
   * as the lobby's groups -- a Task waiting for the reader before them, the recorded ones folded -- where it
   * lists any, else one line; an issue keeps its own page. The input versions and the storage are the
   * place's other tabs, so the strip above says neither (law 143). */
  // `onEmpty`: a page whose own empty state already stands (no maintained data yet) shows the issues only when there are some (ST8)
  function issuesSection(onEmpty=false) {
    const st=states.get('issues'), b=st?.body;
    if(!b) return st?.error && !onEmpty ? notRead(t('Data issues not read'),st.error) : '';
    const empty=!(b.issues || []).length && !(b.refused_cases || []).length && !(b.task_refusals || []).length && !(b.continuations || []).length && !(b.recorded_decisions || []).length && !(b.continued_dispositions || []).length;
    if(empty) return onEmpty ? '' : noteLine(t('No open data issue'),t('One appears when the data owner cannot resolve evidence alone.'),'neutral','','checkcircle');
    return panel(html`${t('Data issues')} <span class="num">${count((b.issues || []).length)}</span>`,t('Current cases read'),issues(b));
  }

  /* ---- the issues page (N6): the data owner's cases, each read as an issue is read -- the move as a
   * figure, the owner's checks in words, your decision among the permitted responses (in words),
   * the owner's own actions apart; the recorded decision in the form's place; an earlier decision
   * on the same listing beside it. The rules are the page's (i). ---- */
  const ISSUE_STATUS={AWAITING_CHOICE:['review_pending','Your decision is asked'],CONFIRMED_PENDING_REVALIDATION:['planned','Decided · applied when the update continues'],WAITING_FOR_RETRY:['deferred','Waiting for the owner\'s retry time'],OPTION_REFUSED:['blocked','The chosen response was refused at execution']};
  const issueAgent=issue=>['AWAITING_CHOICE','OPTION_REFUSED'].includes(issue.status) && (Data.decisions() || []).some(d=>d.kind==='DATA_ISSUE' && d.case_token===issue.case.case_token && d.waits_on==='AGENT');
  const pct=(a,b)=>{ if(!(Number.isFinite(a)&&Number.isFinite(b)&&b)) return ''; const c=(a/b-1)*100; return (c>0 ? '+' : '')+pctNumber(c); }; // the shared percentage rule, a move with its sign
  /* The owner's checks of one listing's evidence, in words: what explains the move and what does not. */
  const checksText=(e)=>{const parts=[];if(e.action_explained===false)parts.push(t('no corporate action explains it'));if(e.provider_correction_observed===false)parts.push(t('no provider correction observed'));if(e.identity_verified===true)parts.push(t('listing identity verified'));if(e.retry_exhausted)parts.push(t('retries exhausted'));if(e.range_start||e.range_end)parts.push(t('audited {a} – {b}',{a:e.range_start || '',b:e.range_end || ''}));const moves=Array.isArray(e.unexplained_moves) ? e.unexplained_moves.length : 0;if(moves>1)parts.push(countText(moves-1,'{n} more unexplained move','{n} more unexplained moves'));else if(!moves&&Array.isArray(e.unexplained_sessions)&&e.unexplained_sessions.length)parts.push(pluralText(e.unexplained_sessions.length,'{n} unexplained session: {s}','{n} unexplained sessions: {s}',{n:count(e.unexplained_sessions.length),s:e.unexplained_sessions.slice(0,4).join(', ')+(e.unexplained_sessions.length>4 ? ' …' : '')}));return parts.join(' · ') || t('evidence recorded');};
  /* The move as a figure (N6): the close before, the close, the change -- the owner's own numbers. */
  function moveFigure(e) {
    const m=Array.isArray(e.unexplained_moves) ? e.unexplained_moves[0] : null;
    if(!m) return '';
    return rail(html`${stat(t('Close on {d}',{d:m.previous_session}),Number(m.previous_close).toFixed(2))}${stat(t('Close on {d}',{d:m.session}),Number(m.close).toFixed(2))}${stat(t('Move'),pct(Number(m.close),Number(m.previous_close)))}`,'issue-move',t('The unexplained move'));
  }
  /* The decisions recorded before on the same listing (law 125's shared reference): the ones this
   * case carries, then any resolved case that named one of its listings. */
  function earlierDecisions(issue, history) {
    const ids=new Set(issue.case.listing_ids || []);
    const mine=(issue.prior_decisions || []).map(d=>({option:d.receipt?.policy_decision?.option_id || d.effect?.option_id,retry:d.effect?.retry_after_at,refused:d.effect?.failure_reasons?.length}));
    const others=history.filter(h=>h.case_token!==issue.case.case_token && (h.listing_ids || []).some(id=>ids.has(id))).map(h=>({option:h.resolution?.receipt?.policy_decision?.option_id,retry:h.resolution?.effect?.retry_after_at}));
    const all=[...mine,...others].filter(d=>d.option);
    if(!all.length) return '';
    return html`<ul class="issue-earlier" aria-label="${t('Earlier decisions on this listing')}">${all.map(d=>html`<li>${icon('history')}<span>${t('Earlier decision')}: <strong>${optionText({option_id:d.option})}</strong>${d.retry ? html` · ${t('retry after {t}',{t:when(d.retry)})}` : ''} · ${t(d.refused ? 'refused at execution' : 'ran its course')}</span></li>`)}</ul>`;
  }
  function issueBody(issue, history) {
    const c=issue.case, selected=S.options.get(c.case_token)||'', [tone,word]=ISSUE_STATUS[issue.status] || ['review_pending',issue.status], agent=issueAgent(issue);
    const resolution=issue.resolution, effect=resolution?.effect, receipt=resolution?.receipt;
    const decided=Boolean(resolution) && issue.status!=='OPTION_REFUSED';
    const asking=!decided && issue.options_current;
    const subjects=(c.listing_ids || []).map(id=>issue.subjects[id]||short(id,SHORT.id));
    const evidence=(c.evidence || []).slice(0,6);
    const ok=(o)=>issue.confirmable_option_ids.includes(o.option_id), yours=(c.options || []).filter(ok), owners=(c.options || []).filter(o=>!ok(o));
    const evidenceMarkup=html`${evidence.map(e=>html`<div class="issue-evidence-item">${evidence.length>1 ? html`<h3>${issue.subjects[e.listing_id] || short(e.listing_id,SHORT.id)}</h3>` : ''}${moveFigure(e)}<p class="issue-checks">${checksText(e)}</p></div>`)}${(c.evidence || []).length>6 ? html`<p class="issue-checks">${countText(c.evidence.length-6,'{n} more listing in the exact record','{n} more listings in the exact record')}</p>` : ''}`;
    const standing=issue.standing_quarantine?.standing?.filter(s=>s.continuation_refused?.length) || [];
    const why=html`${standing.length ? noteLine(t('Asked again although a decision stands'),t('the owner could not continue the standing quarantine: {reasons}',{reasons:standing.flatMap(s=>s.continuation_refused).map(x=>codeWords(x)).join(', ')}),'info') : ''}${effect?.failure_reasons?.length ? noteLine(t('Refused at execution'),effect.failure_reasons.map(x=>codeWords(x)).join(', '),'warning') : ''}`;
    // your decision: the permitted responses in words while it is asked; the recorded one in its place
    const optionRow=(o)=>html`<label class="issue-option${selected===o.option_id ? ' is-selected' : ''}"><input type="radio" name="issue-${c.case_token}" value="${o.option_id}" data-workspace-option="${c.case_token}"${selected===o.option_id ? ' checked' : ''}><span><strong>${optionText(o)}</strong>${optionNote(o) ? html`<small>${optionNote(o)}</small>` : ''}</span></label>`;
    const delegateActions=(get('issues')?.continuations || []).filter(v=>v.task_kind==='workspace_preparation' && v.lifecycle==='BLOCKED' && v.failure_code==='data.truth_review_required')
      .map(v=>action(t('Authorize external executor for Task {id}',{id:short(v.task_id,SHORT.id)}),'delegate',JSON.stringify([c.case_token,v.task_id]),!selected ? t('Choose a permitted response') : ''));
    const form=asking && !agent ? html`<fieldset class="issue-options"><legend>${t('Your decision')}</legend>${yours.map(optionRow)}</fieldset><div class="flow">${action('Preview decision','issue',c.case_token,!selected ? t('Choose a permitted response') : '','button primary')}${delegateActions}</div>` : agent ? noteLine(t('Agent is deciding data issues')) : '';
    const recorded=decided ? html`<p class="issue-decision">${icon('checkcircle')}<span>${t('Decision')}: <strong>${optionText((c.options || []).find(o=>o.option_id===receipt?.policy_decision?.option_id) || {option_id:receipt?.policy_decision?.option_id})}</strong>${effect?.retry_after_at ? html` · ${t('retry after {t}',{t:when(effect.retry_after_at)})}` : ''} · ${t(issue.status==='WAITING_FOR_RETRY' ? 'the owner retries then' : 'applied when the update continues')}</span></p>` : '';
    const expired=!decided && !issue.options_current ? noteLine(t('These options expired'),t('The owner must assess the case again (run the update again) before a decision can be confirmed.'),'warning') : '';
    const theirs=owners.length && asking ? html`<div class="issue-owner-actions"><p class="issue-owner-label">${t('The data owner\'s own actions')}${infoMark(t('Responses only the data owner takes; they are not confirmable here.'))}</p><ul>${owners.map(o=>html`<li>${optionText(o)}${optionNote(o) ? html` <span class="muted">· ${optionNote(o)}</span>` : ''}</li>`)}</ul></div>` : '';
    return html`<section class="panel issue-card" data-box="decision" data-issue-status="${issue.status}"><div class="panel-body">${evidenceMarkup}${earlierDecisions(issue,history)}${why}${recorded}${expired}${form}${theirs}${detail(issue,'Exact case, evidence and decision record')}</div></section>${delegationRows((get('issues')?.delegations || []).filter(g=>g.case_token===c.case_token))}`;
  }
  /* An issue is named by its listing and its kind in the owner's words (`QD000 · Unexplained move`). */
  const issueName=(subjects,code)=>`${subjects.join(', ') || t('Unnamed listing')} · ${codeWords(code)}`;
  const subjectsOf=(issue)=>(issue.case.listing_ids || []).map(id=>issue.subjects[id] || short(id,SHORT.id));
  const firstMove=(c)=>(c.evidence || [])[0]?.unexplained_moves?.[0] || null;
  /* A data issue's own page (F2; law 137, Decided 2): its listing and kind the title, its state under
   * it; the move as figures, the owner's checks, the decisions before it and yours -- the form
   * while it is asked, the record after -- then the exact record. The move's session and who decided
   * are the head's line (N2, law 126 amended); the listing, the kind and the case are its title and id. */
  function issuePage(b, token) {
    const history=b.recorded_decisions || [], standing=b.continued_dispositions || [];
    const issue=(b.issues || []).find(v=>v.case.case_token===token);
    if(issue) {
      const [tone,word]=ISSUE_STATUS[issue.status] || ['review_pending',issue.status], m=firstMove(issue.case), receipt=issue.resolution?.receipt, actor=receipt?.actor_submission;
      const delegated=receipt?.submission?.proposal?.rationale?.startsWith('first-use-goal:') || actor?.actor_id?.startsWith('first-use-goal:');
      // the way on once decided (N6): the Task that waits for the reader, offered as on the lobby -- only when no case asks first
      const asking=(b.issues || []).some(v=>['AWAITING_CHOICE','OPTION_REFUSED'].includes(v.status)), decided=(b.issues || []).some(v=>v.status==='CONFIRMED_PENDING_REVALIDATION');
      return html`${objectHead(issueName(subjectsOf(issue),issue.case.failure_code),'','',stateLine(tone,{word:t(issueAgent(issue) ? 'Agent is deciding data issues' : word),next:''}),[],{object:true,id:token,facts:[[t('Move session'),m?.session || ''],[t('Decided by'),actor?.actor_kind && !(delegated && actor.actor_kind==='HUMAN') ? actorWords(actor.actor_kind) : ''],[t('Decision'),delegated ? t('Decided under first-use delegation') : '']].filter(([,v])=>v)})}${issueBody(issue,history)}${asking ? '' : (b.continuations || []).map((c,i)=>continuationCard(c,i,decided))}`;
    }
    const names=Object.assign({},...(b.issues || []).map(x=>x.subjects || {}));
    const h=history.find(v=>v.case_token===token);
    if(h) {
      const listings=(h.listing_ids || []).map(id=>names[id] || short(id,SHORT.id)), option=h.resolution?.receipt?.policy_decision?.option_id;
      const main=html`<section class="panel issue-card" data-box="decision" data-issue-status="RECORDED"><div class="panel-body">${option ? html`<p class="issue-decision">${icon('checkcircle')}<span>${t('Your decision')}: <strong>${optionText({option_id:option})}</strong></span></p>` : ''}${detail(h,'Exact case, evidence and decision record')}</div></section>`;
      return html`${objectHead(issueName(listings,h.failure_code),'','',stateLine('recorded',{word:t('Recorded'),next:''}),[],{object:true,id:token})}${main}`;
    }
    const d=standing.find(v=>'q:'+v.listing_id===token);
    if(d) return html`${objectHead(`${d.symbol || short(d.listing_id,SHORT.id)} · ${t('standing quarantine')}`,'','',stateLine('deferred',{word:t('Standing quarantine'),next:''}),[],{object:true,id:token})}${dispositionCard(d)}`;
    return null;
  }
  /* Data issues as a lobby (F2, law 136; Decided 2): grouped by the case's lifecycle -- your
   * decision asked, refused at execution, decided (applied when the update continues), waiting
   * for the owner's retry, a standing quarantine open; the recorded folded. One line an issue: its
   * state, its case reference, its listing and kind, the move as a figure, the move's session; the
   * row opens the issue's page. Nothing the owner does not record (no severity, no found-at). */
  const LIFECYCLE={AWAITING_CHOICE:0,OPTION_REFUSED:1,CONFIRMED_PENDING_REVALIDATION:2,WAITING_FOR_RETRY:3};
  function issuesLobby(b) {
    const cases=b.issues || [], history=b.recorded_decisions || [], standing=b.continued_dispositions || [];
    const names=Object.assign({},...cases.map(x=>x.subjects || {}));
    const items=[...cases.map(x=>({kind:'case',key:x.case.case_token,status:x.status,agent:issueAgent(x),subjects:subjectsOf(x),code:x.case.failure_code,move:firstMove(x.case),more:(x.case.evidence || []).length-1})),
      ...standing.map(d=>({kind:'standing',key:'q:'+d.listing_id,subjects:[d.symbol || short(d.listing_id,SHORT.id)],d})),
      ...history.map(h=>({kind:'recorded',key:h.case_token,subjects:(h.listing_ids || []).map(id=>names[id] || short(id,SHORT.id)),code:h.failure_code,option:h.resolution?.receipt?.policy_decision?.option_id}))];
    const group=(x)=>x.kind==='standing' ? {key:'standing',label:t('Standing quarantine'),rank:4,open:true} : x.kind==='recorded' ? {key:'recorded',label:t('Recorded'),rank:5,open:false} : {key:x.status+(x.agent ? ':agent' : ''),label:t(x.agent ? 'Agent is deciding data issues' : (ISSUE_STATUS[x.status] || ['',x.status])[1]),rank:LIFECYCLE[x.status] ?? 3.5,open:true};
    const to=(x)=>({action:'workspace-issue-open',value:x.key});
    const row=(x)=>x.kind==='case' ? objectRow({lead:statusDot((ISSUE_STATUS[x.status] || ['review_pending'])[0]),ref:short(x.key,SHORT.id),name:issueName(x.subjects,x.code),to:to(x),cls:'issue-row'},{key:x.key,columns:['price-change','session','disposition'],props:[x.move ? html`<span class="num">${pct(Number(x.move.close),Number(x.move.previous_close))}</span>` : '',x.move?.session || '',x.more>0 ? countText(x.more,'{n} more listing','{n} more listings') : '']})
      : x.kind==='recorded' ? objectRow({ref:short(x.key,SHORT.id),name:issueName(x.subjects,x.code),to:to(x),cls:'issue-row'},{key:x.key,columns:['price-change','session','disposition'],props:['','',x.option ? optionText({option_id:x.option}) : '']})
      : objectRow({name:`${x.subjects[0]} · ${t('standing quarantine')}`,to:to(x),cls:'issue-row'},{key:x.key,columns:['price-change','session','disposition'],props:['','',x.d.recheck_after_at ? html`${t('Next recheck')} ${when(x.d.recheck_after_at)}` : '']});
    return Lobby.render('issues',{items,row,axes:[{key:'state',label:t('State'),group}],words:(x)=>[...x.subjects,codeWords(x.code),x.key].join(' '),placeholder:t('Listing or case'),
      foot:b.next_cursor ? Lobby.older(b.case_count!=null ? t('{n} of {total} current cases read',{n:count(cases.length),total:count(b.case_count)}) : t('More cases are not read yet'),'workspace-more',b.next_cursor,S.busy,t('Read more')) : ''});
  }
  /* A Task that waits for the reader to continue it (N6): named once, in words -- a decision
   * recorded is said as recorded, never as still needed -- with its one way on. Offered only when
   * no case asks first. */
  function continuationCard(c, i, decided) {
    const code=c.failure_code || c.failure_reason?.code || '', update=c.operation==='DATA_UPDATE_RUN';
    const verb=action(update ? 'Continue this update' : 'Continue this Task','continue',String(i),'','button primary'); // the Data page's and the Home's verb for the same Task
    if(!update && code==='data.truth_review_required' && decided) return banner(t('Decision recorded'),t('Decided · applied when the update continues'),'neutral',verb,'checkcircle',true);
    if(update && code==='data.truth_review_required' && decided) return banner(t('Your decision is recorded'),t('Continue the update: the owner checks the evidence again and applies your decision; nothing recorded is undone.'),'neutral',verb,'checkcircle',true); // law 148: the stopped update waits for the reader -- a decision
    const why=c.failure_reason?.explanation && !String(c.failure_reason.explanation).includes(code || '\0') ? c.failure_reason.explanation : explain(code) || (code ? codeWords(code) : t('The Task stopped by name.'));
    return banner(t((Data.decisions() || []).some(d=>d.kind==='STOPPED_TASK' && d.task_id===c.task_id && d.waits_on==='AGENT') ? update ? 'Data update needs attention' : 'Preparation needs attention' : update ? 'The data update waits for you to continue it' : 'The preparation waits for you to continue it'),why,'neutral',verb,'clock',true); // law 148: a decision
  }
  function dispositionCard(d) {
    const original=d.original || {}, chain=d.continuations || [];
    return panel(html`${d.symbol || short(d.listing_id,SHORT.id)} · ${t('standing quarantine')}`,t('A recheck re-examines the evidence at its due time; an unchanged anomaly continues under the original decision and is not asked again. Nothing here recovers the listing automatically.'),kv([...(d.recheck_after_at ? [[t('Next recheck'),when(d.recheck_after_at)]] : []),[t('Original decision'),html`${original.option_id ? optionText({option_id:original.option_id}) : t('recorded without a case')}${(original.reason_codes || []).length ? html` · ${original.reason_codes.map(x=>codeWords(x)).join(', ')}` : ''}`],[t('Continuations'),chain.length ? html`${chain.map(x=>html`${when(x.observed_at)} · ${t(x.rule || 'continued')} · ${countText((x.unexplained_sessions || []).length,'{n} unexplained session','{n} unexplained sessions')}<br>`)}` : t('none yet')]]));
  }
  function issues(b) {
    const cases=b.issues || [], conts=b.continuations || [], history=b.recorded_decisions || [], standing=b.continued_dispositions || [];
    const taskRefusals=(b.task_refusals || []).map(r=>refusal(r,'warning',{catalog:true,more:html`<p>${t('Task')} ${hashCell(r.task_id)}</p>`,next:prerequisiteWays(r.next_requests)}));
    const refusedCases=(b.refused_cases || []).map(r=>refusal(r,'warning',{catalog:true,more:html`<p>${t('Case')} ${hashCell(r.case_token)}</p>`,next:prerequisiteWays(r.next_requests)}));
    const asking=cases.filter(v=>['AWAITING_CHOICE','OPTION_REFUSED'].includes(v.status)).length, decided=cases.some(v=>v.status==='CONFIRMED_PENDING_REVALIDATION');
    if(!cases.length && !refusedCases.length && !taskRefusals.length && !conts.length && !history.length && !standing.length && !b.delegations?.length) return emptyState(html`${t('No open cases')}${infoMark(t('No open cases; one appears when the data owner cannot resolve evidence alone.'))}`);
    // a Task waiting for the reader is offered only when no case asks first (N6); the cases are the lobby
    // the update's own stop notice above says its continuation with the same way (law 143, one fact once): the list offers only what the page does not already say
    return html`${taskRefusals}${refusedCases}${asking ? '' : conts.map((c,i)=>c.operation==='DATA_UPDATE_RUN' ? '' : continuationCard(c,i,decided))}${cases.length || history.length || standing.length ? issuesLobby(b) : refusedCases.length || taskRefusals.length ? '' : emptyState(html`${t('No open cases')}${infoMark(t('No open cases; one appears when the data owner cannot resolve evidence alone.'))}`)}${delegationRows(b.delegations || [])}${detail({recorded_decisions:history,continued_dispositions:standing,claim:b.claim_limit},'Exact decision records')}`;
  }

  function delegationRows(grants) {
    if(!grants.length) return '';
    return panel(t('External decision grants'),'',html`${grants.map(g=>formRow({
      title:optionText({option_id:g.option_id}),
      line:html`${t('Task')} ${mono(g.preparation_task_id,SHORT.id)} · ${t('Expires')} ${when(g.expires_at)} ${factsRef(t('Exact grant and CLI request'),detail(g,'Exact grant and CLI request'))}`,
      control:action('Revoke grant','revoke-delegation',g.grant_hash),
    }))}`);
  }

  /* ---- the inputs page: immutable versions, publication and explicit selection ---- */
  const ROOT_TEXT={CURRENT_ACTIVE:'configured default',CURRENT_RESEARCH_VERSION:'latest published version',PREVIOUS_ROLLBACK:'previous version (rollback)',USER_PINNED:'pinned by you',RESEARCH_FOUNDATION:'used by Foundation research',PORTFOLIO_RESEARCH:'used by Portfolio research',IN_FLIGHT_RECOVERY:'held by a Task in flight',MODEL_TRAINING_SOURCE:'retained for model training',UNCLASSIFIED_LEGACY:'legacy, unclassified'};
  // why a version cannot be used for a draft, as the English source a held control carries (CT7)
  const heldInput=(v)=>v.unreadable ? 'This version cannot be read in this workspace' : !v.available ? 'Source files unavailable' : v.lifecycle!=='REGISTERED' && v.lifecycle!=='SUCCEEDED' ? 'Publication not complete' : '';
  const inputLifecycle=(lifecycle)=>lifecycle==='REGISTERED' ? t('registered at preparation') : lifecycle==='SUCCEEDED' ? t('published') : lifecycle ? coded(lifecycle) : '';
  function versionRow(group, v, chosen, latestEnd) {
    const key=JSON.stringify([group.input_id,v.binding_hash]), isChosen=chosen===v.binding_hash;
    const twins=(group.versions || []).filter(x=>x.start===v.start && x.end===v.end).length>1;
    // the roles as phrases: a short line breaks between them, never inside one (a CJK line would split a word)
    const role=[v.configured_default ? t('configured default') : '',inputLifecycle(v.lifecycle),v.end && latestEnd && v.end===latestEnd ? t('latest cutoff') : ''].filter(Boolean);
    const roles=role.length ? html`${role.map((w,i)=>html`${i ? ' · ' : ''}<span class="nowrap">${w}</span>`)}` : ''; // a fragment: html keeps its spans
    // a row (round 28): the recorded sessions as the title, the hash, the role and the files as
    // props, the decision at the row's end, always shown
    const decision=isChosen ? stateLine('planned',{word:t('Selected in the draft')}) : typedBtn(t('Use for a new Factor draft'),'workspace-input',key,'menu-row',heldInput(v));
    if(v.unreadable) return objectRow({lead: 'file', name: html`${t('Input version')} ${mono(v.binding_hash,SHORT.hash)}`, to: {action: 'workspace-version-open', value: v.binding_hash}, cls: 'version-row'}, {key,columns:['binding','roles','availability'],props:['',html`${coded(v.unreadable)} · ${link(t('Storage & retention'),'storage','inline-link')}`,''],actions:decision});
    return objectRow({lead: 'file', name: dateRange(v.start, v.end), to: {action: 'workspace-version-open', value: v.binding_hash}, cls: 'version-row'}, {key,selected:isChosen,columns:['binding','roles','availability'],props:[twins ? html`<span class="mono" data-tip="${v.binding_hash}">${short(v.binding_hash,SHORT.hash)}</span>` : '',roles,v.available ? '' : t('unavailable: source files missing')],actions:decision}); // N6: present is the usual state, not a fact to repeat
  }
  /* An input version's own page (F3; law 135's four levels: Data / Research inputs / <family> /
   * <version>): its dates the title, its family the directory above it (`page-directory`); what it
   * is -- its roles, its interval, its publication and lifecycle, its files -- and why it is kept,
   * the storage owner's roots, read where they are owned: Storage & retention, its other home. */
  // opened from Storage & retention it keeps that list in its path (`via`; law 135: one home, and the path follows the list it was opened from)
  function versionPage(b, hash) {
    const group=(b.inputs || []).find(g=>(g.versions || []).some(v=>v.binding_hash===hash)), v=group?.versions.find(x=>x.binding_hash===hash);
    if(!v) return null;
    const store=states.get('storage'); if(!store) void refresh('storage','',true); // why it is kept is the storage owner's, read quietly
    const kept=store?.body?.inputs?.find(x=>x.binding_hash===hash);
    const latestEnd=group.versions.map(x=>x.end).filter(Boolean).sort().at(-1), key=JSON.stringify([group.input_id,v.binding_hash]);
    const draft=typeof LiveResearch!=='undefined' && LiveResearch.context ? LiveResearch.context() : null, isChosen=draft?.input_binding_hash===v.binding_hash;
    const roles=[v.configured_default ? t('configured default') : '',inputLifecycle(v.lifecycle),v.end && v.end===latestEnd ? t('latest cutoff') : ''].filter(Boolean);
    const use=isChosen ? '' : typedBtn(t('Use for a new Factor draft'),'workspace-input',key,'button primary',heldInput(v));
    const why=kept ? (kept.roots.length ? kept.roots.map(r=>t(ROOT_TEXT[r] || r)).join(' · ') : t('no root: eligible for cleanup')) : store?.body ? t('Not in the storage inventory') : t('Reading the storage inventory');
    // N2 (law 126 amended): one section of the version's own facts in the lane's columns -- its interval is the
    // title, its family the path, its binding the head's id and its publication in the exact record
    const main=v.unreadable ? panel(t('This version'),'',html`${kv([[t('Source files'),coded(v.unreadable)],[t('Why'),html`<span class="owner-text">${v.detail || ''}</span>`],[t('Kept because'),html`${why} · ${link(t('Storage & retention'),'storage','inline-link')}`]],'kv-columns')}${detail(v)}`)
      : panel(t('This version'),'',html`${kv([[t('Roles'),html`${roles.map((w,i)=>html`${i ? ' · ' : ''}<span class="nowrap">${w}</span>`)}`],[t('Lifecycle'),inputLifecycle(v.lifecycle)],[t('Source files'),t(v.available ? 'present' : 'unavailable: source files missing')],[t('Kept because'),html`${why} · ${link(t('Storage & retention'),'storage','inline-link')}`]],'kv-columns')}${detail(v)}`);
    return html`${objectHead(v.unreadable ? html`${t('Input version')} ${mono(v.binding_hash,SHORT.hash)}` : dateRange(v.start,v.end),'',use,isChosen ? stateLine('planned',{word:t('Selected in the draft'),next:''}) : '',[],{object:true,id:v.binding_hash})}<template class="page-directory">${group.input_id}</template>${main}`;
  }
  function inputs(b) {
    const groups=b.inputs || [],capture=get('capture');
    const refused=(b.refusals || []).map(r=>refusal(r,'warning',{catalog:true,more:html`<p>${t('Recorded reference')} ${locatorCell(r.publication_hash || r.input_id || '')}</p>`,next:prerequisiteWays(r.next_requests)}));
    const draft=typeof LiveResearch!=='undefined' && LiveResearch.context ? LiveResearch.context() : null, chosen=draft?.input_binding_hash || null;
    // The versions are what this page is for; publishing another is one row after them, its
    // family and its one action side by side.
    const latestCutoff=groups.flatMap(g=>(g.versions || []).map(v=>v.end)).filter(Boolean).sort().at(-1) || Data.inputs().map(v=>v.cutoff).filter(Boolean).sort().at(-1) || '';
    const working=(typeof Data.clocks==='function' ? Data.clocks().data : app.data) || '';
    const chooser=groups.length>1 ? html`<span class="field inline-field"><span id="workspaceInputFamilyLabel" class="sr-only">${t('Input family')}</span>${picker('workspaceInputFamily',[['',t('Choose explicitly')],...groups.map(v=>[v.input_id,v.input_id])],{selected:S.family,labelId:'workspaceInputFamilyLabel'})}</span>` : '';
    const publishable=groups.length && working && (!latestCutoff || working>latestCutoff) ? banner(t('New data to publish'),t('Working data through {d}; the latest published version ends {c}.',{d:working,c:latestCutoff || ''}),'neutral',html`${chooser}${action('Preview publication','preview','capture',S.family?'':t('Choose the input family first'),'button primary')}`,'file') : '';
    const publish=html`
      ${capture ? html`<div class="issue-decision" data-capture-status="${capture.status || ''}">${icon(['SUCCEEDED','REUSED_EXACT'].includes(capture.status) ? 'check' : capture.status==='CONFIRMATION_REQUIRED' ? 'file' : 'task')}<span>${capture.status==='REUSED_EXACT' ? t('Exact version already published: {v}. Nothing was re-sealed; select it below.',{v:short(capture.binding_hash,SHORT.hash)}) : capture.status==='CONFIRMATION_REQUIRED' ? t('Previewed, not admitted: {n} of the source changed. Confirm the preview or preview again.',{n:countText(sourceChanges(capture.before,capture.after).length,'{n} field','{n} fields')}) : capture.status==='SUCCEEDED' && capture.publication ? html`${t('Published')} <span class="mono">${short(capture.publication.binding_hash,SHORT.hash)}</span> · ${t('sessions through {d}',{d:capture.publication.source?.panel_through || ''})} · ${t('Task')} <span class="mono">${short(capture.task_id)}</span>` : html`${t('Publication Task')} <span class="mono">${short(capture.task_id)}</span> · ${capture.status ? codeWords(capture.status) : ''}${capture.failure_code ? html` · <span class="mono">${capture.failure_code}</span>` : ''}`}</span>${capture.task_id ? task(capture) : ''}</div>` : ''}
`; // N6: the publishable state is the first row; the publication's own record follows it
    // F2 (law 136): the versions as a lobby grouped by their family, the latest publication first
    const items=groups.flatMap((group,i)=>{const latestEnd=group.versions.map(v=>v.end).filter(Boolean).sort().at(-1);return [...group.versions].reverse().map(v=>({group,v,latestEnd,rank:i}));});
    const groupsMarkup=items.length ? Lobby.render('inputs',{items,row:(x)=>versionRow(x.group,x.v,chosen,x.latestEnd),axes:[{key:'family',label:t('Input family'),group:(x)=>({key:x.group.input_id,label:x.group.input_id,rank:x.rank,open:true})}],cls:'version-rows',
      words:(x)=>[x.group.input_id,x.v.start,x.v.end,x.v.binding_hash,x.v.lifecycle].join(' '),placeholder:t('Family, dates or hash')}) : '';
    // N3: the version contract is the page's (i) (`INPUTS_INFO`); the draft's selection is its row's mark
    return html`${refused}${publishable}${publish}${groupsMarkup}${!groups.length && !refused.length?emptyState(t('No input family published'),link(t('Prepare workspace'),'welcome','button primary'),'page-empty'):''}`;
  }

  /* ---- the storage page: what is kept where, and what cleanup may release ---- */
  const bytesRead=(text)=>{const m=String(text || '').match(/^(.+?)\s+(B|KiB|MiB|GiB|TiB)$/);return m ? html`${m[1]} ${unit(m[2])}` : text || '';};
  const bytesText=(n)=>{ if(n==null || !Number.isFinite(Number(n))) return ''; const v=Number(n); const units=['B','KiB','MiB','GiB','TiB']; let i=0, x=v; while(x>=1024 && i<units.length-1){x/=1024;i++;} return html`${i ? x.toFixed(x>=100 ? 0 : 2) : String(v)} ${unit(units[i])}`; };
  const ROOT_LABEL={WORKING_DATABASE:['Working database','the DuckDB store the update owner maintains in place; not research input'],IMMUTABLE_INPUTS:['Immutable input versions','sealed source files and Parquet objects; shared objects stored once'],ARTIFACTS:['Feature and outcome artifacts','Panel partitions and outcome snapshots; the current Panel is protected'],RESEARCH_EXPERIMENTS:['Saved experiments','research results and Feature trials; never released by cleanup'],PROVIDER_CACHE:['Provider cache','the raw source cache and the source probe cache'],STAGING:['Staging','files being prepared'],EVIDENCE_KNOWLEDGE:['Evidence knowledge','the evidence data layer: sources, texts, vectors and indexes, detailed under Evidence'],ALTERNATIVE_EVIDENCE:['Alternative evidence','the Alternative Evidence desk\'s sealed records'],PORTFOLIO_LEDGER:['Portfolio ledger','the Portfolio research ledger\'s records'],PREPARATION:['Preparation records','a preparation\'s stage records and evidence; never released by cleanup'],HOST_RECORDS:['Host records','data update plans, change executions, valuation receipts and the Host\'s records; never released by cleanup'],AUTHORITY:['Installed authority','the Evidence and CRO packages and the retained strategies\' authority; never released by cleanup'],RETRIEVAL_MODEL:['Retrieval model','the retained retrieval recipe\'s model copy; a cleanup may link it to the machine\'s model store']};
  const LIMITATION_TEXT={EVIDENCE_INDEX_PAYLOADS_ONLY_VECTORS_BLOBS_AND_ARTIFACTS_RETAINED:'Cleanup may release an evidence index payload only; its vectors, source texts and sealed records stay.',EVICTED_EVIDENCE_INDEX_REBUILDS_FROM_ITS_COMMITTED_VECTORS:'An evicted evidence index rebuilds from its committed vectors, without the model.',CLASSIFIED_INPUT_AND_PANEL_OUTCOME_SNAPSHOTS_ONLY:'Only classified input files and Panel/outcome snapshots without a root are eligible.',RAW_HISTORY_BASE_FEATURES_AND_RECOVERY_PATCHES_RETAINED:'Raw history, base Features and recovery patches are always kept.',PANEL_GENERATIONS_PAST_THE_PREVIOUS_AND_UNPUBLISHED_CHUNKS_RELEASED:'A Panel generation no input captured is released once it is past the previous one, and the chunks no generation published with it.',A_RETAINED_RETRIEVAL_MODEL_COPY_LINKS_TO_THE_MODEL_STORE_IT_MATCHES:'A retained retrieval model\'s copy becomes links to the machine\'s model store it matches; nothing is lost.',REPORTS_AND_DECISIONS_RETAINED:'Reports and decisions are kept.',REPLAY_REQUIRES_RETAINED_INPUTS:'Replaying a result needs its input version retained.'};
  const CAPACITY_TEXT={WITHIN_BUDGET:['neutral','Within the managed budget'],CLEANUP_RECOMMENDED:['warning','Near the managed budget: a cleanup preview shows what may be released'],CAP_EXCEEDED:[TONE.attention,'Over the workspace cap: raise it in Settings or plan a cleanup'],BUDGET_AFTER_INPUT_DISCOVERY:['neutral','Budget known after an input is published']};
  /* The Evidence category (round E5, plan 4.6): what the evidence data layer keeps, physical bytes
   * each object once; the indexes with their availability and protection, pinned or rebuilt by
   * the row; the unreferenced objects; nothing here adds a book's logical bytes into a disk figure. */
  const INDEX_ACTIONS={PIN:['Pin','workspace-pin-index'],UNPIN:['Unpin','workspace-pin-index'],REBUILD:['Rebuild','workspace-rebuild-index']}; // U4: the Host's `available_actions`, in its order
  function evidencePanel(e) {
    if(!e || typeof e!=='object') return ''; // N3 (law 81): a category the owner does not report is not shown
    const sum=e.summary && typeof e.summary==='object' ? e.summary : e;
    const rows=[[t('Originals'),sum.source_object_bytes],[t('Canonical texts'),sum.source_blob_bytes],[t('Vectors'),sum.vector_payload_bytes,sum.vector_object_count!=null ? countText(sum.vector_object_count,'{n} object','{n} objects') : ''],[t('Indexes'),sum.index_payload_bytes],[t('Sealed records'),sum.sealed_artifact_bytes],[t('Staging'),sum.staging_bytes],[t('Task state'),sum.task_state_bytes]].filter(([,v])=>v!=null);
    const railMarkup=rail(html`${rows.map(([label,v,note])=>stat(label,bytesText(v),note || t('physical bytes · each object once')))}`);
    // none while an approved cleanup awaits recovery: the page's banner says why, once (ST6)
    const indexRows=(e.indexes || []).map((ix,i)=>{ const key=String(ix.availability || '').toLowerCase();
      // U80 (V560): an index that cannot be rebuilt says why, in the owner's words, where its Rebuild would stand
      const actions=html`<span class="flow">${(ix.available_actions || []).filter(a=>INDEX_ACTIONS[a]).map(a=>btnAttrs(t(INDEX_ACTIONS[a][0]),INDEX_ACTIONS[a][1],a==='REBUILD' ? ix.index_id : JSON.stringify([ix.index_id,a]),'text-btn'))}${ix.rebuild_limit ? hint(t('Cannot be rebuilt'),t(ix.rebuild_limit)) : ''}</span>`;
      return tr([count(i+1),html`<span class="mono">${short(ix.index_id,SHORT.hash)}</span>${ix.corpus_hash ? html` <span class="sub-cell mono">${short(ix.corpus_hash,SHORT.hash)}</span>` : ''}`,ix.generation_format || '',count(ix.record_count),bytesText(ix.database_bytes),html`${bytesText(ix.payload_bytes)} <span class="sub-cell">${ix.payload_proof==='PROVED' ? t('proved') : ix.payload_available===false ? t('absent') : codeWords(ix.payload_proof || '')}</span>`,stateLine(key,{next:''}),(ix.protected_by || []).length ? ix.protected_by.map(codeWords).join(' · ') : (ix.eligible ? t('eligible for cleanup') : ''),actions]); });
    const indexes=indexRows.length ? table([{label:'#',type:'num',index:true},{label:t('Index'),type:'text',cls:'col-tight'},{label:t('Format'),type:'text',cls:'col-tight'},{label:t('Records'),type:'num'},{label:t('Database'),type:'num'},{label:t('Payload'),type:'num'},{label:t('Availability'),type:'status',cls:'col-tight'},{label:t('Protected by'),type:'text',absorb:true},{label:'',type:'link'}],indexRows,'',{report:true,countLine:false,classes:'compact evidence-indexes'}) : emptyState(t('No evidence index is kept in this workspace.'));
    const un=e.unreferenced && typeof e.unreferenced==='object' ? Object.entries(e.unreferenced).filter(([,v])=>typeof v==='number' && v>0).map(([k,v])=>`${codeWords(String(k).toUpperCase())} ${count(v)}`) : [];
    return panel(t('Evidence'),t('What the evidence data layer keeps, physical bytes each object once; an index is evicted or rebuilt, never its vectors.'),html`${railMarkup}<h3>${t('Evidence indexes')} <span class="num">${count((e.indexes || []).length)}</span></h3>${indexes}${un.length ? html`<p class="caption">${t('Unreferenced')} · ${un.join(' · ')}</p>` : ''}`,'','data-storage="evidence" data-box="table"');
  }
  /* U48 (V209): the held state's backups -- the root, each kept generation (newest first), the last
   * automatic attempt after a data update; a person backs up now; a restore is the client's own
   * command onto a new directory, with no Host (V328), so the page names it and never runs it. */
  async function readBackups() {
    const state={loading:true,body:get('backups') || null,retry:false};states.set('backups',state);
    const active=()=>states.get('backups')===state && here()==='storage';
    try {
      const body=await Data.read('/api/workspace/backup',true);
      if(active()){state.body=body;state.error='';}else state.retry=true;
    } catch(e) {
      if(e?.name==='AbortError' || !active())state.retry=true;else state.error=e.message;
    } finally {
      state.loading=false;if(active())patchMain();
    }
  }
  const BACKUP_REASON={DATA_UPDATE:'After a data update',REQUEST:'On request'};
  function backupSummary(bk) {
    const newest=(bk.generations || [])[0];
    return [['Backup root',locatorCell(bk.backup_root)],['Kept generations',count((bk.generations || []).length)],['Newest',newest ? when(newest.created_at) : t('none yet')]];
  }
  function backupsPanel() {
    const s=states.get('backups');
    if(!s || (!s.loading && s.retry)) void readBackups();
    const bk=s?.body, caption=t('Copies of the held state outside the workspace: one generation after each data update and one when you ask; files are copied, tables exported, the research inputs and installed authority packages listed by digest, and everything else rebuilds from these.');
    // a refused read is said above what an earlier read kept, and holds the request: the same owner refuses it
    const make=action('Back up now','preview','backup',s?.error ? t('The backups could not be read') : bk ? '' : t('Reading the backups'));
    const refused=s?.error ? notRead(t('Backups not read'),s.error,explain(String(s.error).split(':')[0])) : '';
    if(!bk) return panel(t('Backups'),caption,refused || skeleton('rows'),make,'data-storage="backups"');
    const a=bk.last_automatic_attempt, gens=bk.generations || [], refusedGenerations=(bk.refusals || []).map(r=>refusal(r,'warning',{catalog:true,more:html`<p>${t('Generation')} ${locatorCell(r.generation_file || r.generation_hash || '')}</p>`,next:prerequisiteWays(r.next_requests)}));
    const last=!a ? t('None attempted yet; one follows each data update') : a.status==='FAILED' ? html`${t('Failed {when}',{when:when(a.at)})} · ${coded(a.failure_code)}` : a.status==='PENDING' ? t('Waiting since {when}; taken once the Host is idle',{when:when(a.at)}) : t('Backed up {when}',{when:when(a.at)});
    const failed=a?.status==='FAILED' ? banner(t('The last automatic backup failed'),t('The data update it followed stays published; back up now once the failure is resolved.'),'warning') : '';
    const {shown,start,page,pages}=pageOf(gens,S.backupPage);
    const rows=shown.map((g,i)=>tr([count(start+i+1),when(g.created_at),t(BACKUP_REASON[g.reason] || g.reason),count(g.files),count(g.tables),count(g.listed),count((g.absent || []).length),hashCell(g.generation_hash,SHORT.hash)]));
    const list=gens.length ? html`${table([{label:'#',type:'num',index:true},{label:t('Created'),type:'text'},{label:t('Reason'),type:'text',absorb:true},{label:t('Files'),type:'num'},{label:t('Tables'),type:'num'},{label:t('Listed'),type:'num'},{label:t('Not held'),type:'num'},{label:t('Generation'),type:'text',cls:'col-tight'}],rows,'',{report:true,countLine:false,classes:'compact backup-generations'})}${pager({page,pages,prev:['workspace-backups-page','prev'],next:['workspace-backups-page','next']})}` : refusedGenerations.length ? '' : emptyState(t('No backup is kept yet.'));
    const absent=gens[0]?.absent || [];
    const notHeld=absent.length ? html`<p class="caption">${t('Not held by this workspace in the newest generation')} · ${absent.map((v,i)=>html`${i ? ' · ' : ''}<span class="mono">${v}</span>`)}</p>` : '';
    const command=`alphalattice backup restore --dir <new directory> --workspace-id ${app.workspace || '<workspace id>'} --root "${bk.backup_root}"${gens[0] ? ` --generation ${gens[0].generation_hash}` : ''}`;
    // a restore is the client's own command onto a new directory, with no Host (V328): named in its code dialog, never run here
    const restore=[t('Restore'),html`${t('A command of the client, onto a new or empty directory, with no Host')}${codeRef(t('Restore command'),command,'text')}`];
    // MA2, MA4: the generations are a table's box; a warning or a refusal stands on the ground above it
    return html`${refused}${refusedGenerations}${failed}${panel(t('Backups'),caption,html`${kv([[t('Backup root'),locatorCell(bk.backup_root)],[t('Last automatic backup'),last],restore],'kv-columns')}${list}${notHeld}${detail(bk,'Exact backup listing')}`,make,gens.length ? 'data-storage="backups" data-box="table"' : 'data-storage="backups"')}`;
  }
  const turnBackups=(way)=>{S.backupPage=Math.max(0,(S.backupPage || 0)+(way==='next' ? 1 : -1));patchMain();};
  function storage(b) {
    const [tone,word]=CAPACITY_TEXT[b.capacity_status] || ['neutral',b.capacity_status || 'Capacity not reported'], roots=b.by_root || {}, shared=Math.max(0,Number(b.logical_bytes || 0)-Number(b.managed_bytes || 0));
    const capacity=tone==='neutral' && b.status!=='RECOVERY_REQUIRED' ? html`<p class="prep-liveness data-readiness-line">${icon('check')}<span>${t(word)} · ${t('Cleanup only ever removes the exact targets of an approved plan; retained versions, results and the current Panel are never released.')}</span></p>` : banner(t(word),t(b.status==='RECOVERY_REQUIRED' ? 'An approved cleanup was interrupted: resume it below before anything else changes here.' : 'Cleanup only ever removes the exact targets of an approved plan; retained versions, results and the current Panel are never released.'),tone);
    const summaryRail=rail(html`${stat(t('Managed on disk'),b.display?.managed ? bytesRead(b.display.managed.split(' (')[0]) : bytesText(b.managed_bytes),t('physical bytes · each object once'))}${stat(t('Referenced'),b.display?.logical ? bytesRead(b.display.logical.split(' (')[0]) : bytesText(b.logical_bytes),html`${bytesText(shared)} ${t('shared between versions')}`)}${stat(t('Free disk'),b.display?.free ? bytesRead(b.display.free.split(' (')[0]) : bytesText(b.free_disk_bytes),t('on this workspace\'s disk'))}${b.display?.cap ? stat(t('Managed cap'),bytesRead(b.display.cap.split(' (')[0]),b.budget ? t(b.budget?.setting?.cap_bytes === 'auto' ? 'automatic from managed data and free disk' : 'set by the operator') : '') : ''}`, 'data-readiness-strip', t('Storage summary'));
    const summary=html`${summaryRail}${capacity}`;
    const rootKeys=[...Object.keys(ROOT_LABEL).filter(k=>k in roots),...Object.keys(roots).filter(k=>!(k in ROOT_LABEL))], fileTotal=rootKeys.reduce((sum,root)=>sum+(Number(roots[root]?.files)||0),0);
    const breakdown=panel(t('What is kept where'),t('The managed roots of this workspace, from the storage owner\'s inventory; shared objects are counted once, under the first root that names them.'),table([{label:t('Root'),type:'text',absorb:true},{label:t('On disk'),type:'num',cls:'col-tight'},{label:t('Referenced'),type:'num',cls:'col-tight'},{label:t('Files'),type:'num',cls:'col-tight'}],rootKeys.map(root=>{const v=roots[root];return tr([html`${t(ROOT_LABEL[root]?.[0] || root)}${ROOT_LABEL[root]?.[1] ? infoMark(t(ROOT_LABEL[root][1])) : ''}`,bytesText(v.physical_bytes),bytesText(v.logical_bytes),count(v.files)]);}),'',{report:true,countLine:false,foot:[t('Total'),bytesText(b.managed_bytes),bytesText(b.logical_bytes),count(fileTotal)]}),'','data-box="table"'); // words first, figures last (round 60); what a root holds is its (i), so the first screen keeps few words (R4)
    // rows (round 28): the version as the title, why it is kept and its bytes as props, the
    // retention action at the row's end
    const versionName=(hash)=>{const x=Data.inputVersion(hash);return x?.cutoff ? dateRange(x.start,x.cutoff) : html`${t('Input version')} ${mono(hash,SHORT.hash)}`;}; // S (law 96): a version the inputs do not list is named by its words, its hash after
    const versions=panel(t('Retained input versions'),t('Every version and why it is kept. Pinning retains a version by your choice; unpinning leaves its other roots as they are.'),html`<div class="card-list lines slotted version-rows">${(b.inputs || []).map(v=>objectRow({lead: 'file', name: versionName(v.binding_hash), to: Data.inputVersion(v.binding_hash) ? {action: 'workspace-version-open', value: v.binding_hash} : null, cls: 'version-row'}, {key:v.binding_hash,columns:['roots','size','availability'],props:[factsRef(t('Kept because'),html`<p>${v.roots.length ? v.roots.map(r=>t(ROOT_TEXT[r] || r)).join(' · ') : t('no root: eligible for cleanup')}</p>`),bytesText(v.logical_bytes),v.available ? '' : t('unavailable')],actions:action(v.roots.includes('USER_PINNED')?'Remove pin':'Retain input','pin',v.binding_hash,!v.available||b.status==='RECOVERY_REQUIRED'?t('Retention change unavailable'): '','menu-row')}))}</div>`);
    const plan=get('cleanup');
    const planFacts=plan ? cleanupSummary(plan).filter(([k])=>!['Reclaimable','Retained','Kept by policy'].includes(k)) : [];
    const planLedger=plan ? kv([[t('Reclaimable'),bytesText(plan.reclaimable_bytes)],[t('Retained'),bytesText(plan.retained_bytes)],[html`<strong>${t('Managed total')}</strong>`,html`<strong>${bytesText(Number(plan.reclaimable_bytes||0)+Number(plan.retained_bytes||0))}</strong>`]],'num') : '';
    const resume=(b.pending_cleanup || []).map(hash=>banner(t('An approved cleanup did not finish'),t('Resume it before anything else changes here; it removes only the targets it was approved for.'),'warning',action('Resume approved cleanup','resume-cleanup',hash,'','button primary')));
    const cleanup=plan ? panel(t('Cleanup preview'),t('The exact files an approved cleanup would remove and what stays; nothing is deleted without your confirmation.'),html`${planLedger}${kv(planFacts.map(([k,v])=>[t(k),v]))}`) : '';
    // N3: what cleanup never removes is the page's (i) (`storageInfo`)
    return html`${resume}${summary}${cleanup}${breakdown}${versions}${evidencePanel(b.evidence)}${backupsPanel()}${detail(b,'Exact inventory and references')}`;
  }
  function page() {
    const name=here(),state=states.get(name);
    if(!state){void refresh(name);return skeleton('head');}
    if(SCENE_ROUTE[name]) {
      // The route names another Task than the one this state asked for (or none, where it
      // asked for one): read again -- whether the state holds a body, a refusal, nothing yet
      // or a read still in flight (a late answer to the earlier choice is discarded).
      const want=selectedTask(name) || '';
      if((state.requested ?? '')!==want) { void refresh(name); return name==='welcome' ? firstUse() : data(states.get(name)); }
      return name==='welcome' ? firstUse() : data(state);
    }
    const heads={issues:['Data issues','The data owner\'s decisions: what it asks, what it recorded, what it continues.'],inputs:['Input versions','Immutable research inputs: publish a version, select one explicitly.'],storage:['Storage','Bounded workspace storage: what is kept where, what cleanup may release.']};
    const b=state.body;
    // an issue named by the address is its own page (F2); one the owner no longer lists leaves the lobby with a line
    const token=name==='issues' ? hashParams().get('issue') || '' : name==='inputs' ? hashParams().get('version') || '' : '', own=token && b ? (name==='issues' ? issuePage(b,token) : versionPage(b,token)) : null;
    if(own) return html`${S.error||state.error?notRead(t('Workspace action needs attention'),S.error||state.error,explain(String(S.error||state.error).split(':')[0])):''}${noticeFor(name)?noteLine(t('Last operation'),t(noticeFor(name))):''}${own}`;
    // N3: the documentation a page carried in a box is its (i): the rules, the contract, the policy
    const terms=(rows)=>rows.map(([k,v])=>`${t(k)}: ${t(v)}`).join(' ');
    const info={issues:terms([['Who confirms','You, as the authorized Human; the product decides nothing on its own.'],['What is decided','One of the responses the data owner permits, for the exact evidence shown; nothing else is confirmable.'],['When it applies','When the Task continues: the owner revalidates the evidence first, then applies the decision.'],['What stays','Saved research and published input versions; a decision changes what the next Task does, never a saved object.']]),
      inputs:terms([['Identity','The binding hash of the sealed source; a materializer or method change re-seals the same data'],['Contents','Source files and Parquet objects, stored once where versions share them'],['Older versions','Retained exactly as sealed; saved research keeps the version it used'],['Selection','Explicit, in a Research Lab draft; nothing is selected for you']]),
      storage:`${t('What cleanup never removes')}: ${((get('cleanup')?.limitations) || Object.keys(LIMITATION_TEXT)).map(v=>t(LIMITATION_TEXT[v] || v)).join(' ')}`};
    const working=(typeof Data.clocks==='function' ? Data.clocks().data : app.data) || '', cutoffs=Data.inputs().map(v=>v.cutoff).filter(Boolean).sort();
    const newData=working && (!cutoffs.length || working>cutoffs.at(-1)); // the first row holds the publication then (N6)
    const headActions=html`${name==='inputs' && b?.inputs?.length && !newData ? action('Preview publication','preview','capture',S.family?'':t('Choose the input family first')) : name==='storage' && b && !(b.pending_cleanup || []).length ? action('Preview cleanup','preview','cleanup',b.status==='RECOVERY_REQUIRED' ? t('Resume the approved cleanup first') : '') : ''}`;
    return html`${objectHead(t(heads[name][0]),html`<p class="lede">${t(heads[name][1])} ${info[name]}</p>`,headActions)}
      ${S.error||state.error?notRead(t('Workspace action needs attention'),S.error||state.error,explain(String(S.error||state.error).split(':')[0])):''}
      ${noticeFor(name)?noteLine(t('Last operation'),t(noticeFor(name))):''}${token && b ? noteLine(t(name==='issues' ? 'This case is not in the owner\'s list now' : 'This version is not in the owner\'s list now'),html`<span class="mono">${short(token,SHORT.hash)}</span>`,'neutral') : ''}${state.loading?skeleton('head'):''}
      ${state.body?({issues,inputs,storage}[name])(state.body):name==='storage'?backupsPanel():''}`;
  }
  return {pages,page,firstUse,prepareRefused,refresh,preview,commit,turnBackups,selectInput,observe,notice,fold,inspect,follow,current,stopWords,networkWords,deferWords,afterPaint:(pageChanged=false)=>{W.afterPaint();if(pageChanged)entered();},bind:()=>W.bind(),selectedTask,
    task:id=>LiveTasks.open(id),more:cursor=>refresh('issues',cursor),openIssue:(token)=>{objectEntry('issue:'+token);navigate('issues',{issue:token});},openVersion:(hash)=>{const via=app.page==='storage' ? 'storage' : '';objectEntry('version:'+hash);navigate('inputs',{version:hash,via});},dismissConfirmation:()=>{S.revision++;S.pending=null;},
    changed:(name,value)=>{invalidate();if(name==='family')S.family=value;else S.options.set(name,value);render();},
    scene:(page='welcome')=>{const s=states.get(page);return s ? {body:s.body,view:s.view,loading:s.loading,readAt:s.readAt} : null;},
    area:()=>W.area()};
})();
