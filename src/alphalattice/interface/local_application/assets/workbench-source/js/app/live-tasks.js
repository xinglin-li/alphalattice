/* Read actual Task Control projections; animation never advances lifecycle.
 *
 * The selected Task is read as its recovery view (card 34): what stopped or is unknown, which
 * stages and artifact references Task Control still holds verified, which actions its owners
 * permit right now with their scope and effect, and the exact Task version those facts describe.
 * A choice is confirmed against that version and re-read at the action boundary: a Task that
 * moved meanwhile is shown again, and nothing is sent. Guanyin explains here; the owners act. */
const LiveTasks = (() => {
  // A Task kind whose result has its own page; any other kind's result is this inspector, which
  // then offers no Open result that would only reopen itself.
  const RESULT_PAGES=new Set(['workspace_data_update','workspace_preparation','research_input_capture','research_feature_materialization','research_experiment','portfolio_public_development_replay','alternative_evidence.document_intelligence','chief_risk_officer.portfolio_review']);
  // `status`: the selected Task's STATUS (its timing, who submitted it, its open incident); `guardian`: every
  // unfinished Task as Guanyin sees it (U22); `incidents`: the incident records, open first (U38); `remedy`:
  // the last remedy attempted from this page, said in place
  const S={selected:null,record:null,view:null,status:null,guardian:null,incidents:null,remedy:null,error:'',fetching:null,timer:null,confirming:null,busy:false,painted:'',settling:false,closureFailures:[]};
  const subjects=new Map(); // admission context, refreshed by receipts; never current Task state
  const shownTasks=()=>Data.tasks().filter(v=>stateMoving(v.lifecycle)).slice(0,LOBBY.shown);
  function subjectContext(id,renewed=false) {
    const held=subjects.get(id);
    if(renewed){if(held){held.dirty=true;held.view=null;}return;}
    if(!held || held.dirty && !held.pending)queueMicrotask(()=>readSubject(id));
    return (held ? held.view : S.selected===id ? S.view : null)?.subject_context;
  }
  async function readSubject(id) {
    if(subjects.get(id)?.pending || subjects.get(id)?.view && !subjects.get(id).dirty)return;
    const entry={pending:true}, navigation=Data.navigationIntent(); subjects.set(id,entry);
    try {
      const v=await Data.read('/api/tasks/recovery?'+new URLSearchParams({task_id:id}));
      if(!Data.navigationCurrent(navigation)) { if(subjects.get(id)===entry)subjects.delete(id); return; }
      if(subjects.get(id)!==entry)return;
      if(entry.dirty){entry.pending=false;paintCurrent();return;}
      Object.assign(entry,{pending:false,view:v});
    } catch(e) {
      if(!Data.navigationCurrent(navigation)) { if(subjects.get(id)===entry)subjects.delete(id); return; }
      Object.assign(entry,{pending:false,error:e.message});
    }
    paintCurrent();
  }
  const currentGroup=()=>stackSlot('currentWork',LiveViews.runningGroup(true));
  function paintCurrent() {
    const home=app.page==='overview', slot=$(home ? '#homeRunning' : '#currentWork');
    if(!slot)return;
    const content=LiveViews.runningGroup(!home);
    if(slot.innerHTML!==String(content)){const saved=preserveSurface(slot);fillStackSlot(slot,content);restoreSurface(saved);}
  }
  /* Keep polling while Task Control says the Task moves or the dispatcher says its operation
   * has not returned; the two facts are read apart and never masked over each other. */
  const alive=()=>stateMoving(S.record?.lifecycle) || (S.view ? stateMoving(S.view.lifecycle) || S.view.operation_running : false);
  // The already-read STATUS span belongs only to the exact Task the body names.
  const timedTask=(v)=>S.status?.task_id===v.task_id ? {...v,timing:S.status.timing} : v;
  // An owner's sentence keeps its meaning; only installed operation identifiers take their
  // declared display words. Unknown identifiers stay exact rather than receiving a guessed name.
  const effectWords=(action)=>t(action?.expected_effect || '').replace(/\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b/g,code=>declaredCodeWord(code) ? codeWords(code) : code);
  const STALE_AT_OWNER=new Set(['task_control.recovery_version_stale','task_control.start_version_stale']);
  /* Where the Task stands, in the owners' words (running, queued, cancellation requested or
   * acknowledged, waiting, stopped, interrupted, completed); liveness is a separate fact, never
   * a verdict: the runner's heartbeat or Task Control's durable timestamp alone (`telemetry`),
   * the age from the later of the two (`last_heartbeat_source`), never shown as refreshed. */
  function liveness(r) {
    const live=r.liveness, source=live.telemetry==='OPERATIONAL' ? (live.last_heartbeat_source==='DURABLE_TIMESTAMP' ? t('Task Control timestamp; runner heartbeat #{n} is older, at {time}',{n:live.signal_sequence,time:when(live.operational_at)}) : t('runner heartbeat #{n}',{n:live.signal_sequence})) : live.telemetry==='UNREADABLE' ? t('Task Control timestamp only; a runner heartbeat store could not be read, so a signal may be hidden') : live.telemetry==='UNBOUND' ? t('Task Control timestamp only; the recorded heartbeat is not bound to this execution') : t('Task Control timestamp only; no runner heartbeat recorded for this execution');
    return live.status==='OBSERVED' ? t('Heartbeat at {time} ({source})',{time:when(live.last_heartbeat_at),source}) : live.status==='NOT_RECENT' ? t('No heartbeat since {time} ({source}) · the worker may still be running or may have stopped; this view does not decide',{time:when(live.last_heartbeat_at),source}) : live.status==='NOT_OBSERVED' ? t('No heartbeat recorded yet · liveness unknown') : '';
  }
  function standing(r) {
    // the code once: the owner's detail often names it already
    const stop=r.stop, code=stop && !stop.detail ? coded(stop.code) : '';
    const codeLine=code ? html` · ${code}` : stop && !stop.code ? html` · ${t('no code recorded')}` : ''; // said once: the detail often names it
    const detail=stop ? html`<br><span class="coded" data-tip="${stop.code}">${t(stop.detail || '')}</span>` : ''; // the owner's sentence, with its code on hover (WD2, ST6)
    const live=r.liveness, heartbeat=liveness(r), word=t(stateOf(r.lifecycle).word);
    const running=r.operation_running ? html` · ${t('the operation has not returned')}` : '';
    // The dispatcher's own fact, apart from Task Control's: why this Host's command last stopped.
    const w=r.worker_failure, worker=w ? html`<br><span class="muted" data-tip="${w.code}">${t(STALE_AT_OWNER.has(w.code) ? 'Confirmed resume refused at Task Control' : 'Worker stop')}: ${t(w.detail)}${w.failure_type ? html` <span class="mono">${w.failure_type}</span>` : ''}</span>` : '';
    switch(r.lifecycle) {
      case 'RUNNING': return [live.status==='NOT_RECENT' ? 'attention' : '', 'activity', html`<strong>${word}</strong> · ${heartbeat}${running}${worker}`];
      case 'QUEUED': return [w ? 'attention' : '', 'clock', html`<strong>${word}</strong> · ${t('not started; nothing has run')}${running}${worker}`];
      case 'CANCEL_REQUESTED': return ['attention', 'warning', html`<strong>${word}</strong> · ${t('not yet acknowledged by the worker; it stops at its next safe checkpoint')} · ${heartbeat}${running}${worker}`];
      case 'CANCELLED': return ['', 'ban', html`<strong>${word}</strong> · ${code}${running}${detail}${worker}`];
      case 'DEFERRED': return ['attention', 'clock', html`<strong>${word}</strong>${codeLine}${running}${detail}<br>${t(stateOf('deferred').line)}${worker}`];
      case 'REVIEW_PENDING': return ['attention', 'info', html`<strong>${word}</strong>${codeLine}${running}${detail}${worker}`];
      case 'BLOCKED': return ['attention', 'ban', html`<strong>${word}</strong>${code ? html` · ${code}` : stop ? '' : html` · ${t('no code recorded')}`}${running}${detail}<br>${t(stateOf('blocked').line)}${worker}`];
      case 'RECOVERY_REQUIRED': return ['attention', 'history', html`<strong>${word}</strong>${codeLine}${running}${detail}<br>${t(stateOf('recovery_required').line)}${worker}`];
      case 'SUCCEEDED': return ['', 'checkcircle', html`<strong>${word}</strong> · ${t('its result is read through its own reader; nothing runs again')}${running}`];
      default: return ['', 'info', html`<strong>${word}</strong>${running}${detail}${worker}`];
    }
  }
  /* Task Control's own work board: each stage as recorded, with the exact evidence references
   * a verification produced. Reconnecting or losing activity rows changes none of this. */
  /* The latest instant the owners observed for a stage of this Task (an activity envelope with
   * the stage's id), read as recorded; nothing is computed from it. */
  function stageInstant(taskId, stageId) {
    if(typeof LiveActivity==='undefined' || !LiveActivity.retained) return '';
    let latest='';
    for(const g of LiveActivity.retained()) for(const item of g.items || []) if(item.task_id===taskId && item.stage_id===stageId && item.observed_at && item.observed_at>latest) latest=item.observed_at;
    return latest ? when(latest) : ''; // N3 (law 133): the reader's clock (it was the UTC clock without its zone)
  }
  /* The run's summary (round 72): the state line with the owner's times, then when it started and
   * who admitted it (the activity record's first operation for it). */
  function summary(v, r) {
    // the next action is the refusal line's (round 75) where a held run has one under the summary
    const state=S.settling ? stateLine('settling',{duration:''}) : stateLine({...timedTask(v),lifecycle:r ? r.lifecycle : v.lifecycle},{live:Boolean(r && r.lifecycle==='RUNNING' && r.liveness?.status==='OBSERVED'),next:r && stateHeld(r.lifecycle) ? '' : undefined});
    const by=LiveActivity.starterOf(v.task_id), started=v.running_since ? when(v.running_since) : '';
    return html`<p class="run-summary">${state}${started ? html`<span class="muted">${t('started')} ${started}${by ? html` ${t('by')} ${by}` : ''}</span>` : ''}${r?.operation_running ? html`<span class="muted">${t('operation returning')}</span>` : ''}</p>`;
  }
  /* The steps: Task Control's stages as the one step list -- the observed instant at the right,
   * the evidence references under a verified one, the current one's line (what is verified,
   * what comes next), a blocked one's reason. */
  function steps(r) {
    const SHOWN=8, current=r.stages.findIndex(s=>s.lifecycle!=='VERIFIED');
    const all=LiveActivity.logLines(r.task_id);
    return stepList(r.stages.map((s,i)=>{
      const refs=s.evidence.slice(0,SHOWN), more=s.evidence.length>SHOWN ? html`<em class="caption">${t('+{n} more, all held by Task Control',{n:s.evidence.length-SHOWN})}</em>` : '';
      const next=stateHeld(r.lifecycle) ? '' : r.stages[i+1] ? html` · ${t('step|Next')}: ${codeWords(r.stages[i+1].stage_id)}` : html` · ${t('nothing after this stage')}`;
      const line=i===current ? html`${t('Verified')} ${r.verified_stage_count} / ${r.total_stage_count}${next}` : '';
      // a stopped step's words are the stop box's above it, said once (ST6): the step keeps its count and its lines
      const stopped=i===current && ['BLOCKED','CANCELLED'].includes(s.lifecycle) && Boolean(r.stop?.detail);
      const note=stopped || s.lifecycle==='VERIFIED' || s.evidence.length ? '' : t('no evidence yet');
      const lines=all.filter(l=>l.stage===s.stage_id).map(logLine); // the step's own lines of the log (round 92)
      return {id:s.stage_id,name:codeWords(s.stage_id),mark:s.lifecycle,at:stageInstant(r.task_id,s.stage_id),evidence:s.evidence_count,note,line,refs,lines,current:i===current,done:s.lifecycle==='VERIFIED',under:more};
    }));
  }
  /* The receipt: what the Task declared and what it produced, each hash with Copy. */
  function receipt(v, r) {
    const hash=(h,n=SHORT.hash)=>hashCell(h,n); // a reference, the whole on hover and in the copy: never a line that rolls (WD3; the user, 2026-09-26)
    const rows=[[t('Task'),hash(v.task_id,SHORT.id)],[t('Kind'),codeWords(v.task_kind)],[t('Verified stages'),html`${count(v.verified_stage_count)} / ${count(v.total_stage_count)}`]];
    if(r) rows.push([t('Version'),hash(r.task_record_hash)],[t('Execution binding'),hash(r.execution_binding_hash)],[t('Kept safe'),r.artifact_refs.length ? html`<span class="run-step-refs">${r.artifact_refs.map(a=>refCell(a))}</span>` : r.verified_stage_count ? countText(r.verified_stage_count,'{n} verified stage; no artifact reference recorded yet','{n} verified stages; no artifact reference recorded yet') : t('No stage verified yet; nothing to keep beyond the admitted Task')]);
    else rows.push([t('Current stage'),v.current_stage ? codeWords(v.current_stage) : ''],[t('Latest refusal'),v.latest_failure_code ? html`<span class="coded" data-tip="${v.latest_failure_code}">${t(S.status?.detail || v.detail || '')}</span>${causeLine(S.status?.failure_cause,true)}` : '']); // U71: its owner's cause beside the words
    // R7: Task Control's own spans (a running Task's as of the read); U33: the agent session that submitted
    // it, as the Host's boundary read the request -- provenance, never an authority (an unknown is an empty slot)
    const tm=S.status?.timing, sb=S.status?.submitted_by, took=(s)=>s==null ? '' : durationText(Number(s)*1000);
    if(tm) rows.push([t('Waited in queue'),took(tm.queued_seconds)],[t(stateMoving(v.lifecycle) ? 'Running for' : 'Ran for'),took(tm.running_seconds)]);
    const current=S.status?.task_id===v.task_id && S.status.current_stage===v.current_stage ? S.status : null, stage=current?.timing?.stages?.find(s=>s.stage_id===current.current_stage), actual=current?.activity_timing;
    if(stage) rows.push([t('Stage updated'),when(stage.updated_at)],[t('Stage duration'),took(stage.seconds)]);
    if(actual?.stage===current?.current_stage && actual?.last_work_at) rows.push([t('Latest actual work'),when(actual.last_work_at)]);
    if(sb) rows.push([t('Submitted by'),html`${sb.vendor ? PRODUCERS[String(sb.vendor).toLowerCase()] || codeWords(sb.vendor) : ''}${sb.session ? html`${sb.vendor ? ' · ' : ''}${t('session')} ${hash(sb.session,SHORT.id)}` : ''}${sb.goal_id ? html` · ${t('goal')} ${hash(sb.goal_id,SHORT.id)}` : ''}`]);
    return kv(rows,'run-receipt');
  }
  /* The log: the exact observations, one line each, folded to `Worked for 21 s · 4 ›` (round 83). */
  function log(v) {
    const lines=LiveActivity.logLines(v.task_id), worked=durationOf(timedTask(v));
    return runLog({id:'taskLog',title:t('Log'),count:lines.length,caption:t('The exact observations, in recorded order'),lines:lines.map(logLine),empty:emptyState(t('Nothing recorded for this Task in the retained activity window.')),search:true,fold:true,open:false,summary:worked ? t('Worked for {t}',{t:worked}) : t('Log'),cls:'inspector-section'});
  }
  /* The owners' permitted actions, the next lawful one first as the primary: a verified Task's
   * result, an interrupted Task's resume, a stopped experiment's new draft (re-PLAN, through the
   * Lab). Cancel and resume confirm against this exact Task version. */
  function permitted(r,fresh) {
    const by=Object.fromEntries(r.actions.map(a=>[a.action,a]));
    const cancel=by.CANCEL, recover=by.RECOVER, replan=by.REPLAN;
    const off=(a)=>!fresh || S.busy || !a.available ? a.reason : '';
    const resumable=recover.available && !off(recover), replanNow=replan.available && r.task_kind==='research_experiment';
    // V626 (U95): every operation a re-PLAN can name has its session route, so a bound re-PLAN is always a control
    const bound=replan.available && r.next_requests?.replan && Data.offers(r.next_requests.replan.operation);
    const replanEntry=replanNow ? btn(t('Continue as a new draft'),'research-continue',r.task_id,stateHeld(r.lifecycle) && !resumable ? 'button primary' : 'button') : bound ? typedBtn(t('Declare again (re-PLAN)'),'task-replan',r.task_id,stateHeld(r.lifecycle) && !resumable ? 'button primary' : 'button',off(replan),'aria-describedby="taskActionsWhy"') : !replan.available && r.next_requests?.books ? link(t('Choose the book in History'),'history','button') : '';
    // A preparation or update Task has its own scene (the workspace page); the body names the way there.
    const sceneEntry=r.task_kind==='workspace_data_update' ? btn(t('Open update scene'),'task-update-scene',r.task_id,'button') : ['alternative_evidence.document_intelligence','chief_risk_officer.portfolio_review'].includes(r.task_kind) ? btn(t('Open the evidence review'),'task-review-scene',r.task_id,'button') : '';
    // The product's own sentences (scope, effect, reason) read through the catalog; a held action
    // keeps its reason as its title and in the folded explanation (round 58).
    const why=html`${factsRef(html`${t('What each action does')}`, html`<dl class="tp-why">${[[t('Request cancel'),cancel],[t('Resume this Task'),recover],[t('Declare again (re-PLAN)'),replan]].map(([name,a])=>html`<div><dt>${name}</dt><dd>${t(a.scope)} <small class="muted" data-tip="${a.expected_effect}">${effectWords(a)}</small>${a.available ? '' : html`<small class="muted tp-why-held">${t(a.reason)}</small>`}</dd></div>`)}</dl>`)}`;
    const held='aria-describedby="taskActionsWhy"'; // the reason is the title; the note is the fold
    // R2: a Task the installed code cannot resume shows Resume held, its reason on demand (CT7)
    const cannotResume=!recover.available && r.resume_refusal ? typedBtn(t('Resume this Task'),'task-recovery',r.task_id,'button','The code installed now cannot resume this Task; cancel it and plan the same work again.',held) : '';
    const refusedReplan=S.replanRefused?.id===r.task_id ? refusal({code:S.replanRefused.code,reason:S.replanRefused.detail ? t(S.replanRefused.detail) : explainCode(S.replanRefused.code)},'warning',{word:t('Not done')}) : '';
    return html`${refusedReplan}<div class="tp-actions"><div class="flow">${r.lifecycle==='SUCCEEDED' && RESULT_PAGES.has(r.task_kind) ? btn(t('Open result'),'task-result',r.task_id,'button primary') : ''}${replanEntry}${sceneEntry}${recover.available ? typedBtn(t('Resume this Task'),'task-recovery',r.task_id,resumable ? 'button primary' : 'button',off(recover),held) : cannotResume}${cancel.available ? typedBtn(t('Request cancel'),'task-cancel',r.task_id,'button',off(cancel),held) : ''}</div>${why}</div>`;
  }
  /* U38: the Task's open incident as Guanyin's supervisor last found it -- its code in words, the owner's
   * detail, when it was found -- with the remedies the Host offers for it, each confirmed against this Task's
   * version; the attempts recorded for it, and the last one made here, said in place (LY7). */
  const REMEDY_WORDS={RECOVER:'Resume this Task',CANCEL:'Request cancel',REPLAN:'Declare again (re-PLAN)'};
  const openIncident=(id)=>(S.incidents || []).some(x=>x.task_id===id && x.state==='OPEN');
  function incident(r) {
    const i=S.status?.incident;
    if(!i) return '';
    const attempts=(S.incidents || []).find(x=>x.key===i.key)?.attempts || [];
    const remedies=(i.remedies || []).map(a=>btn(t(REMEDY_WORDS[a] || codeWords(a)),'task-remedy',JSON.stringify([r.task_id,i.key,a]),'button'));
    const said=S.remedy && S.remedy.key===i.key ? html`<p class="caption">${S.remedy.words}</p>${S.remedy.replan && S.view?.next_requests?.replan ? html`<div class="flow">${btn(t('Declare again (re-PLAN)'),'task-replan',r.task_id,'button primary')}</div>` : ''}` : '';
    const past=attempts.length ? html`<p class="caption">${attempts.map((x,n)=>html`${n ? ' · ' : ''}${t('Attempt {n}: {action}, {outcome}',{n:x.ordinal,action:t(REMEDY_WORDS[x.action] || x.action),outcome:codeWords(x.disposition)})}`)}</p>` : '';
    const sameStop=i.detail && i.detail===r.stop?.detail, found=t('Found {time}',{time:when(i.detected_at)});
    return refusal({code:i.detail ? '' : i.code,reason:sameStop ? found : t(i.detail)},TONE.attention,{word:t('Open incident'),next:'',more:html`${sameStop ? '' : html`<p class="caption">${found}</p>`}${remedies.length ? html`<div class="flow">${remedies}</div>` : ''}${said}${past}`});
  }
  /* A remedy is confirmed against the Task version shown, as Cancel and Resume are; the Host performs its
   * own operation and records the attempt with the incident. */
  async function previewRemedy(value) {
    if(S.busy) return;
    const [id,key,remedy]=JSON.parse(value);
    S.busy=true;
    try {
      const view=await Data.read('/api/tasks/recovery?'+new URLSearchParams({task_id:id}));
      S.confirming={kind:'remedy',id,key,remedy,version:view.task_record_hash};
      openDialog(t('Task · explicit confirmation'),t(REMEDY_WORDS[remedy] || codeWords(remedy)),
        html`<div class="tp-confirm"><p>${t('The Host performs this remedy on the Task version shown ({version}) and records the attempt with the incident; a Task that moved first is shown again, and nothing is sent.',{version:short(view.task_record_hash)})}</p></div>`,
        html`${btn(t('Confirm'),'task-remedy-commit','','button primary',true)}`);
    } catch(e) { if(e.name!=='AbortError')notify(e.message); }
    finally { S.busy=false; }
  }
  async function commitRemedy() {
    if(S.busy || S.confirming?.kind!=='remedy') return;
    const {id,key,remedy,version}=S.confirming; S.confirming=null; S.busy=true; closeDialog();
    try {
      const body=await Data.post('/api/tasks/remediate',{task_id:id,incident_key:key,remedy,expected_task_hash:version});
      const a=body.attempt || {};
      S.remedy={key,replan:body.status==='REPLAN_OFFERED' && Boolean(body.next_requests?.replan),words:body.status==='REPLAN_OFFERED' ? t(body.next_requests?.replan ? 'Declare it again: preview its plan here, then confirm the work it admits.' : 'Declare it again from its draft; its owner plans the work again.') : t('Attempt {n}: {outcome}; the Task is now {state}.',{n:a.ordinal ?? '',outcome:codeWords(a.disposition || ''),state:codeWords(a.lifecycle_after || '')})};
      notify('Remedy sent');
    } catch(e) { S.remedy={key,words:e.message}; }
    finally { S.busy=false; await refresh(); }
  }
  /* V615, V627 (U91, U96): a Task's re-PLAN as its owner binds it (`next_requests.replan`: the preview's operation and
   * the retained declaration, never a default) -- previewed here through the session's route for that operation, which
   * records a plan and admits nothing. The admission the answer fills (a write it names, never a read) is a separate
   * press, sent whole, its operation choosing the route; an answer that admits nothing yet but offers the next press
   * (a person's data-change consent, then its run) offers that one; work admitted opens its Task. An answer offering
   * no admission says why in its owner's words. */
  const admissionsOf=(answer,operation)=>Object.entries(answer?.next_requests || {}).filter(([,q])=>q?.operation && q.operation!==operation && Data.offers(q.operation) && Data.posts(q.operation));
  function offerAdmissions(id,operation,answer,step) {
    S.replan={id,operation,answer};
    const admits=admissionsOf(answer,operation);
    const plan=answer.plan_hash || answer.experiment_plan_hash || answer.update_plan_hash || answer.admission_hash || '';
    const facts=kv([[t(step==='preview' ? 'Preview' : 'Answered'),codeWords(operation)],...(answer.status ? [[t('tp|State'),codeWords(answer.status)]] : []),...(plan ? [[t('Plan'),html`<span class="mono">${short(plan,SHORT.hash)}</span>`]] : []),...(answer.source_start && answer.source_end ? [[t('trading|Sessions'),t('{first} to {last}',{first:answer.source_start,last:answer.source_end})]] : [])]);
    // the owner's own words when nothing is offered: its detail, its lawful next actions (codes in words)
    const ways=admits.length ? [] : (answer.next_lawful_actions || []).map((w)=>/^[A-Z][A-Z0-9_]+$/.test(String(w)) ? codeWords(w) : t(String(w)));
    const caption=admits.length ? (step==='preview' ? 'The preview recorded a plan and admitted nothing; its work is admitted as a new Task only when you confirm it.' : 'Nothing runs yet; its work is admitted as a new Task only when you confirm the next step.') : 'The preview recorded a plan and admitted nothing.';
    openDialog(t('Task · re-PLAN'),t('Declare again (re-PLAN)'),
      html`<div class="tp-confirm">${facts}${answer.detail ? html`<p>${t(answer.detail)}</p>` : ''}${ways.length ? html`<ul class="tp-ways">${ways.map((w)=>html`<li>${w}</li>`)}</ul>` : ''}<p class="caption">${t(caption)}</p></div>`,
      html`${admits.map(([name,q])=>btn(t('Confirm: {operation}',{operation:codeWords(q.operation)}),'task-replan-commit',name,'button primary',true))}`);
  }
  async function replanPreview(id) {
    const request=S.view?.task_id===id ? S.view.next_requests?.replan : null;
    if(S.busy || !request || !Data.offers(request.operation)) return;
    const {operation,...fields}=request;
    // a re-PLAN whose operation admits its work directly (the owner says so: `admits`, the saved studies' verification)
    // is no preview: the press asks first, in the owner's words, and sends nothing until confirmed
    const action=(S.view.actions || []).find((a)=>a.action==='REPLAN');
    if(action?.admits) {
      S.replan={id,operation:'',answer:{next_requests:{replan:request}}};
      openDialog(t('Task · re-PLAN'),t('Declare again (re-PLAN)'),html`<div class="tp-confirm"><p>${t(action.scope || '')}</p><p class="caption" data-tip="${action.expected_effect}">${effectWords(action)}</p></div>`,btn(t('Confirm: {operation}',{operation:codeWords(operation)}),'task-replan-commit','replan','button primary',true));
      return;
    }
    S.busy=true; S.replanRefused=null;
    try {
      // the route's own method: a preview that reads (EVIDENCE_PREVIEW) takes its fields in the query, an object as JSON
      const answer=await (Data.posts(operation) ? Data.post(Data.route(operation),fields) : Data.readShared(Data.route(operation)+'?'+new URLSearchParams(Object.entries(fields).map(([k,v])=>[k,typeof v==='object' ? JSON.stringify(v) : String(v)]))));
      offerAdmissions(id,operation,answer,'preview');
    } catch(e) { S.replanRefused={id,code:e.body?.failure_code || String(e.message).split(':')[0],detail:e.body?.detail || ''}; }
    finally { S.busy=false; paint(); }
  }
  async function replanCommit(name) {
    const p=S.replan, request=p?.answer?.next_requests?.[name];
    if(S.busy || !request || !Data.offers(request.operation)) return;
    const {operation,...fields}=request;
    S.busy=true; S.replan=null; closeDialog();
    try {
      const b=await Data.post(Data.route(operation),fields);
      if(b.task_id) { notify('Work admitted'); open(b.task_id); }
      else if(admissionsOf(b,operation).length) offerAdmissions(p.id,operation,b,'next'); // consent given: its run is the next press
      else notify('Answered without a Task');
    } catch(e) { S.replanRefused={id:p.id,code:e.body?.failure_code || String(e.message).split(':')[0],detail:e.body?.detail || ''}; }
    finally { S.busy=false; await refresh(); }
  }
  /* The run's one body (round 72): the truth line, the summary line, the cause and the next action
   * beside it, the steps, the receipt, the log, the guardian's line. */
  function runBody() {
    const v=S.record, r=S.view;
    // Not read: the owner refused or did not answer. Its answer is shown as it came (a typed
    // code when the owner has one), never as a loading state; the list stays reachable.
    if(!v) return html`${S.error ? notRead(t('Task not read'),S.error,t('The owner did not answer for this Task; the list and the activity feed are unaffected. Refresh to ask again.'),btn(t('Refresh'),'task-refresh','','button compact')) : skeleton('body')}`;
    const fresh=!S.error;
    const [tone,ic,text]=r ? standing(r) : ['', 'info', t('Recovery view not read')];
    // The cause, once (round 75): a held or failed run opens with its refusal line -- the state,
    // the owner's stop detail and code, the next lawful action; a moving run its liveness note.
    const w=r?.worker_failure;
    // the truth line above already says the state: the stop box heads with its cause (the owner's code in
    // words, or "Why it stopped"), once, and its next step is that code's way on (the user, 2026-09-25:
    // "点 Blocked连续出现了两次"; ST6)
    const stopCode=r?.stop?.code || w?.code || '', stopKnown=Boolean(stopCode && CODE_WORDS[stopCode]);
    // U2: a blocked Task's way on is the one its recovery view offers (resume once the cause is repaired),
    // not the state table's alone; an owner's own way for its code still comes first
    const offersResume=Boolean(r?.actions?.find(a=>a.action==='RECOVER')?.available);
    const replan=r?.actions?.find(a=>a.action==='REPLAN' && a.available);
    const way=r?.lifecycle==='BLOCKED' && offersResume && !STOP_WAYS[stopCode] ? t('resume it once its cause is repaired') : replan ? effectWords(replan) : wayOn(stopCode,r?.lifecycle);
    // V507: a deferred Task's status says why in its owner's words (as the Data page says them, U63) and, until its
    // retry time, when it waits until; once the time has passed the state's own way stands
    const deferred=r?.lifecycle==='DEFERRED' ? S.status : null, until=deferred?.retry_after_at && Date.parse(deferred.retry_after_at)>(Date.parse(r.observed_at) || Date.now()) ? deferred.retry_after_at : null;
    const cause=!r ? noteLine(text,'',tone==='attention' ? 'warning' : 'neutral','',ic) : r.lifecycle==='SUCCEEDED' ? '' : stateHeld(r.lifecycle) || r.lifecycle==='CANCELLED' ? refusal({code:r.stop?.detail || w?.detail || stopKnown ? '' : stopCode,reason:(deferred?.detail ? LiveWorkspace.deferWords(deferred.detail) : '') || (r.stop?.detail ? t(r.stop.detail) : '') || (w?.detail ? t(w.detail) : '') || (r.operation_running ? t('the operation has not returned') : '')},TONE.attention,{state:r.lifecycle,word:t('Why it stopped'),next:until ? t('Waiting until {time}',{time:when(until)}) : way,more:causeLine(r.stop?.cause || S.status?.failure_cause)}) : noteLine(text,'',tone==='attention' ? 'warning' : 'neutral','',ic);
    const guardian=r ? html`<p class="caption">${t('Guanyin · G0 read-only · health {health} · no remediation attempted by this view · model facts not observed by this Host',{health:codeWords(r.health.status)})}</p>` : '';
    return html`<div class="panel-body tp-detail"><p class="tp-truth">${t(v.goal_summary)}</p>${summary(v,r)}${LiveViews.taskSuccessor(r || v)}${S.error ? notRead(t('Status uncertain'),S.error) : ''}${cause}${r ? incident(r) : ''}${r ? permitted(r,fresh) : ''}<section class="inspector-section"><h3>${t('Steps')}${r ? html` <span>${r.verified_stage_count} / ${r.total_stage_count} ${t('verified')}</span>` : ''}</h3>${r ? steps(r) : skeleton('rows')}</section><details class="reveal-details inspector-section task-receipt"><summary>${t('Receipt')}</summary>${receipt(v,r)}</details>${log(v)}${guardian}</div>`;
  }
  /* Tasks as a lobby (F2, law 136): grouped by state -- the moving and the held open, the ended
   * folded; a held state's head says its way on in the state table's words -- by time (the moving
   * first, then the day each ended) or by kind. One line a row: the dot, the goal, the facts, the
   * instant; the row opens the Task in the side panel. */
  const TASK_RANK = {running: 0, cancel_requested: 1, queued: 2, review_pending: 3, recovery_required: 4, blocked: 5, deferred: 6, succeeded: 8, cancelled: 9};
  // a row says once what needs attention: an open incident (U38), else the guardian's health when not
  // healthy (U22); a healthy or finished Task says nothing more
  const care=(r)=>{ if(openIncident(r.id)) return badge('blocked',t('Open incident')); const g=S.guardian?.get(r.id); return g?.health?.status && g.health.status!=='HEALTHY' ? codeWords(g.health.status) : ''; };
  function lobby(runs) {
    const kinds = [...new Set(runs.map((r) => r.kind))];
    // U22: an unfinished Task's progress is the guardian's (one read for every such Task); a finished one's its record's
    const row = (r, d) => earlierStopRows(r, r => { const on = (k) => d.props[k] !== false, p = S.guardian?.get(r.id)?.progress;
      const current = p ? p.current_stage : r.current, [done, total] = p ? [p.verified_stage_count, p.total_stage_count] : r.verified;
      const health = care(r), recovery = LiveViews.taskAttentionFacts(r);
      return runRow(r, {cls: 'tp-task', dot: d.group === 'state', line: false, columns: ['id', 'kind', 'stage', 'verified', 'care'], props: [on('id') ? html`<span class="mono">${short(r.id)}</span>` : '', on('kind') ? ['', codeWords(r.kind), 'drop'] : '', on('stage') && stateMoving(r.state) && current ? codeWords(current) : '', on('verified') ? html`${t('Verified stages')} · ${count(done)} / ${count(total)}` : '', health || recovery ? html`${health} ${recovery}` : '']}); });
    const byState = (r) => { const a = stateOf(r.state); return {key: a.key || 'unknown', label: t(a.word), rank: TASK_RANK[a.key] ?? 7, open: Boolean(a.moving || a.held), note: a.held && a.next ? t(a.next) : ''}; };
    const axes = [{key: 'state', label: t('State'), group: byState},
      {key: 'time', label: t('Time'), group: (r) => stateMoving(r.state) ? {key: 'moving', label: t('In progress'), rank: -1, open: true} : timeGroup(r.finished || r.started)},
      {key: 'kind', label: t('Task kind'), group: (r) => ({key: r.kind || 'unknown', label: codeWords(r.kind), rank: kinds.indexOf(r.kind), open: true})}];
    return Lobby.render('tasks', {items: runs, row, axes, cls: 'tp-task-list', words: (r) => [r, ...(r.earlierStops || [])].map(one => [one.name, codeWords(one.kind), one.id, one.starter].join(' ')).join(' '), placeholder: t('Name, kind or Task id'),
      filters: kinds.length > 1 ? [{field: 'kind', label: t('Task kind'), multiple: true, options: kinds.map((k) => [k, codeWords(k)]), test: (r, one) => r.kind === one}] : [],
      properties: [['id', t('Task id'), false], ['kind', t('Task kind')], ['stage', t('Current stage')], ['verified', t('Verified stages')]]});
  }
  /* U38: the incidents Guanyin recorded, open first, then the resolved newest first, each with its
   * attempts; paged at the tables' one size (R3). A row opens its Task, where the remedies are. */
  function incidentList() {
    const found = (x) => String(x.incident?.detected_at || '');
    const all = [...(S.incidents || [])].sort((a, b) => (a.state === 'OPEN' ? 0 : 1) - (b.state === 'OPEN' ? 0 : 1) || found(b).localeCompare(found(a)));
    if (!all.length) return '';
    const {shown, page, pages} = pageOf(all, S.incidentPage);
    const row = (x) => objectRow({lead: tile('warning', x.state === 'OPEN' ? TONE.attention : 'neutral'), name: coded(x.code || x.incident?.incident_code), why: t(x.incident?.user_safe_detail || ''), to: {action: 'task', value: x.task_id}}, {key: 'incident:' + x.key, columns: ['incident-state', 'attempts'], props: [x.state === 'OPEN' ? badge('blocked', t('Open')) : badge('succeeded', t('Resolved')), (x.attempts || []).length ? countText(x.attempts.length, '{n} attempt', '{n} attempts') : ''], time: x.incident?.detected_at ? when(x.incident.detected_at) : ''});
    return html`${groupHead(t('Incidents'), all.length)}<div class="card-list lines slotted">${shown.map(row)}</div>${pager({page, pages, prev: ['task-incident-page', 'prev'], next: ['task-incident-page', 'next']})}`;
  }
  const turnIncidents = (way) => { S.incidentPage = Math.max(0, (S.incidentPage || 0) + (way === 'next' ? 1 : -1)); patchMain(); };
  function page() {
    const runs = Data.runsOf('task');
    const refused = (Data.taskRefusals?.() || []).map(r=>refusal(r,TONE.attention,{catalog:true,more:html`<p>${t('Task')} ${hashCell(r.task_id,SHORT.id)}</p>`,next:prerequisiteWays(r.next_requests)}));
    const list = runs.length ? detailSplit(lobby(Data.groupTaskSuccessors(runs)), 'task') : refused.length ? '' : html`<section class="panel pad">${emptyState(t('No Tasks yet.'),link(t('New experiment'),'lab','button primary'),'page-empty')}</section>`;
    return html`${objectHead(t('Tasks'),t('Product-owned recorded states'),btnAttrs(icon('refresh'),'task-refresh','','icon-btn',html`aria-label="${t('Refresh task status')}" data-tip="${t('Refresh task status')}"`),'',[{ic:'team',action:'go',value:'team',word:t('Team'),why:t('The retained sessions and their exchanges')}])}${S.error ? notRead(t('Status uncertain'),S.error) : ''}${incidentList()}${refused}${list}`;
  }
  /* Repaint what a poll may have changed: the open Task's body in the inspector is replaced only
   * when its markup changed (the markup is the displayed state itself: lifecycle, liveness,
   * worker failure, permitted actions), so a reader's selection, scroll and focus stay; the Task
   * Center page is patched section by section. */
  function refreshInspector() {
    if(S.selected && Window.inspectorMode()==='task' && Window.openedBy('task',S.selected)) { // the card shows the selected Task: while another Task's record is read, the one shown stays whole (law 149: no collapse on a switch)
      const body=runBody(), markup=String(body);
      if(S.painted!==markup) { Window.setInspectorBody(body); S.painted=markup; }
    }
  }
  function paint() {
    Window.renderSide(); // the Tasks row's count (round 62)
    refreshInspector();
    const visible=$('#homeRunning') || $('#currentWork'), kept=new Set([...(visible ? shownTasks().map(v=>v.task_id) : []),S.selected]);
    for(const [id,entry] of subjects){if(!kept.has(id))subjects.delete(id);else if(entry.error)void readSubject(id);}
    if(app.page==='tasks')patchMain();else paintCurrent();
  }
  /* One read for the selected Task: the recovery view carries the same STATUS block every
   * reader receives, at the same moment, beside the owners' explanation. */
  const readView=(id)=>Data.readShared('/api/tasks/recovery?'+new URLSearchParams({task_id:id}));
  const readStatus=(id)=>Data.readShared('/api/status?'+new URLSearchParams({task_id:id}));
  // a read the page can do without (the guardian, the incidents, the selected Task's STATUS): its failure
  // leaves its facts out (ST7), never the list
  const spare=(p)=>p.catch(()=>null);
  const OPEN_WAIT=250; // ms: a record read within it opens whole; a slower one shows its skeleton, then fills in place
  const FOLLOW_WAIT=5; // s: one follow's wait at the Host, which answers as soon as the Task moves on; short, as a stopping Host joins a held read
  const CADENCE={moving:2000,failed:10000,hidden:15000}; // ms: the Task Center's timer where no follow is held -- a failed read backs off, a hidden document slows it
  async function refresh() {
    if(S.fetching) return S.fetching;
    clearTimeout(S.timer);
    const selected=S.selected, center=app.page==='tasks';
    S.fetching=(async()=>{
      try {
        const [tasks,view,status,guardian,incidents]=await Promise.all([Data.readShared('/api/tasks'), selected ? readView(selected) : null, selected ? spare(readStatus(selected)) : null,
          center ? spare(Data.readShared('/api/tasks/guardian')) : null, center || selected ? spare(Data.readShared('/api/tasks/incidents')) : null]);
        Data.setTasks(tasks.tasks, tasks.refusals || []);
        void LiveActivity.refresh(); // the same moment: what was recorded around these Tasks
        if(center) S.guardian=guardian ? new Map((guardian.tasks || []).map(e=>[e.task_id,e])) : null;
        if(center || selected) S.incidents=incidents ? incidents.incidents || [] : null;
        if(selected===S.selected) {
          const record=view ? view.status : null;
          const ended=stateMoving(S.record?.lifecycle) && !stateMoving(record?.lifecycle);
          S.record=record; S.view=view; S.status=status; S.error=''; S.settling=false;
          if(view)subjects.set(selected,{pending:false,view});
          if(ended) LiveReview.taskFinished(selected);
        }
      } catch(e) { S.error=e.message; }
    })();
    try { await S.fetching; }
    finally {
      S.fetching=null; paint();
      if(selected!==S.selected) queueMicrotask(refresh);
      // R11 (N8): a moving Task the reader has open is followed by the Host's own wait -- it answers when the Task
      // moves on (its lifecycle or a verified stage), or at the wait's end -- never by a poll. The list alone, a
      // settled Task or a failed read keep the timer; a hidden document slows it, never stops it.
      else if(alive() && !S.error && S.selected) void awaitMove(S.selected);
      else if(alive()) S.timer=setTimeout(refresh,S.error ? CADENCE.failed : document.hidden ? CADENCE.hidden : CADENCE.moving);
    }
  }
  async function awaitMove(id) {
    if(S.following===id) return;
    S.following=id;
    const began=Date.now(), held=S.status ? [S.status.lifecycle,S.status.verified_stage_count].join() : '';
    let seen=null;
    try { seen=await Data.readShared('/api/status?'+new URLSearchParams({task_id:id,wait_seconds:String(FOLLOW_WAIT)})); }
    catch(e) { /* the refresh that follows reads again and says why */ }
    finally { if(S.following===id) S.following=null; }
    if(S.selected!==id || !alive()) return;
    // an answer that moved, or one the Host held, is read at once; one given at once unmoved (a state the Host does
    // not wait on) goes back to the timer, so a follow never spins
    const moved=Boolean(seen && held) && [seen.lifecycle,seen.verified_stage_count].join()!==held;
    if(moved || Date.now()-began>=FOLLOW_WAIT*1000/2) await refresh(); else S.timer=setTimeout(refresh,CADENCE.moving); // held at least half its wait: the Host waited
  }
  const titleOf=(id, recorded=null)=>LiveViews.nameOf(Data.tasks().find(v=>v.task_id===id) || (S.record?.task_id===id ? S.record : recorded) || {task_id:id,task_kind:'',goal_summary:''}).name || id;
  /* The Task opens in the inspector (round 66): the same body from a row's Enter or Space, the
   * Home, History, a study's link or a cold address. Closing the inspector clears the selection
   * and the address; the poll stops with it. */
  function open(id, wanted=()=>true) {
    if(!id || !wanted()) return Promise.resolve(false);
    const same=S.selected===id;
    S.selected=id; if(!same) { S.record=null; S.view=null; S.status=null; S.remedy=null; S.error=''; S.painted=''; }
    Inspect.selectAddressMode('task', {task:id});
    LiveActivity.markSeen();
    Inspect.closeLens(false); closeDialog();
    // law 149 (the user, 2026-09-24: 侧栏点出来会闪烁): the record opens whole -- read first and shown
    // once, the list giving way in the same frame; a read slower than OPEN_WAIT shows its skeleton first
    const closed=({nextMode}={})=>{ if(S.selected===id) { S.selected=null; S.record=null; S.view=null; S.status=null; S.remedy=null; S.painted=''; clearTimeout(S.timer); if(hashParams().get('task')===id && (nextMode!=='record' || app.page!=='tasks')) replaceHash({task:''}); patchMain(); } };
    const show=()=>{ if(!wanted() || S.selected!==id || (Window.inspectorMode()==='task' && Window.openedBy('task',id))) return; const nextMode=Inspect.addressedMode(); if(nextMode!=='task' || hashParams().get('task')!==id) { closed({nextMode}); return; } const recorded=S.record; Window.openInspector({mode:'task', readHeader:()=>({title:titleOf(id,recorded),kind:t('Task')}), title:titleOf(id), kind:t('Task'), body:runBody(), by:['task',id], onClose:closed}); S.painted=String(runBody()); };
    if(same && S.record) { show(); return refresh(); }
    const switching=Window.inspectorMode()==='task'; // another Task's record is shown: it stays until this one is read, then this one replaces it at once
    const slow=switching ? null : setTimeout(show,OPEN_WAIT);
    const read=async()=>{ for(let i=0;i<3 && S.selected===id && !S.record;i++) await refresh(); };
    return read().then(()=>{ if(!wanted()) { closed(); return false; } show(); return S.selected===id; }).finally(()=>clearTimeout(slow));
  }
  function close(after=null) {
    if(Window.inspectorMode()==='task') Window.closeInspector(true);
    if(after) after();
  }
  /* The walk's held row is the open Task while the inspector holds one (round 58): J / K move it. */
  function follow(id) { if(Window.inspectorMode()==='task' && id && id!==S.selected) void open(id); }
  const ACTIONS={cancel:'CANCEL',recover:'RECOVER'};
  const unrecoverable = (view) => typeof view.task_record_hash === 'string' && view.attention?.task_record_hash === view.task_record_hash && view.attention?.unresolved === false && view.attention.resolution === 'UNRECOVERABLE';
  function unrecoverableActions() {
    const tasks = Data.unrecoverableTasks();
    if (!tasks.length && !S.closureFailures.length) return '';
    const failures = S.closureFailures.map(item => refusal(item, TONE.attention, {catalog: true, more: html`<div class="flow">${btn(html`${t('Task')} ${short(item.task_id, SHORT.id)}`, 'task', item.task_id, 'text-btn')}</div>`}));
    return html`<section class="inspector-section">${tasks.length ? html`<div class="flow">${btn(countText(tasks.length, 'Close {n} unrecoverable Task', 'Close {n} unrecoverable Tasks'), 'task-close-unrecoverable', '', 'button', S.busy)}</div>` : ''}${failures}</section>`;
  }
  async function previewUnrecoverable() {
    if (S.busy) return;
    const intent = Data.navigationIntent(), tasks = Data.unrecoverableTasks();
    if (!tasks.length) return;
    S.busy = true;
    let readingTask = tasks[0];
    try {
      const entries = [];
      for (const task of tasks) {
        readingTask = task;
        const view = await readView(task.task_id), action = view.actions.find(a => a.action === 'CANCEL');
        if (!Data.navigationCurrent(intent)) return;
        if (view.status.task_id === view.task_id && view.status.task_record_hash === view.task_record_hash) Data.mergeTasks([{...view.status, attention: view.attention}]);
        if (!unrecoverable(view) || !action?.available) throw Error(action?.reason || 'The task no longer admits this action. Refresh its status.');
        entries.push({id: view.task_id, version: view.task_record_hash, view, action});
      }
      S.confirming = {kind: 'unrecoverable', entries};
      openDialog(t('Task · explicit confirmation'), countText(entries.length, 'Close {n} unrecoverable Task', 'Close {n} unrecoverable Tasks'),
        html`${entries.map(({view, action}) => html`<section class="inspector-section"><h3>${LiveViews.nameOf(view.status).name}</h3>${kv([[t('Task'), hashCell(view.task_id, SHORT.id)], [t('version'), hashCell(view.task_record_hash)], [t('Scope'), t(action.scope)], [t('What changes'), effectWords(action)], [t('Why it is permitted'), t(action.reason)]])}</section>`)}<p>${t('The owner acts on exactly this Task version; if it changes before you confirm, you will be shown the new state instead.')}</p>`,
        html`${btn(t('Confirm'), 'task-commit', '', 'button primary', true)}`);
    } catch(e) {
      if (Data.navigationCurrent(intent)) S.closureFailures = [{task_id: readingTask.task_id, code: e.body?.failure_code || e.body?.refused || '', reason: e.body?.detail || e.message}];
    } finally { S.busy = false; if (Data.navigationCurrent(intent) && app.page === 'overview') patchMain(); }
  }
  async function commitUnrecoverable(entries) {
    S.confirming = null; S.busy = true; S.closureFailures = []; closeDialog();
    for (const {id, version} of entries) {
      try {
        const current = await readView(id), action = current.actions.find(a => a.action === 'CANCEL');
        if (current.task_record_hash !== version || !unrecoverable(current) || !action?.available) {
          S.closureFailures.push({task_id: id, reason: t('Not sent: the Task changed since you reviewed it and is now {state} (version {version}). Review it again.', {state: codeWords(current.lifecycle), version: short(current.task_record_hash)})});
          continue;
        }
        await Data.post('/api/cancel', {task_id: id, expected_task_hash: version});
      } catch(e) { S.closureFailures.push({task_id: id, code: e.body?.failure_code || e.body?.refused || '', reason: e.body?.detail || e.message}); }
    }
    try { await refresh(); await Data.refreshDecisions(); await Data.refreshHistory(); }
    catch(e) { S.closureFailures.push({task_id: entries[0].id, code: e.body?.failure_code || e.body?.refused || '', reason: e.body?.detail || e.message}); }
    finally { S.busy = false; paint(); if (app.page === 'overview') patchMain(); }
  }
  /* The confirmation is bound to the exact Task version and action it shows: the view is read
   * fresh, the owner's own scope and expected effect are the text, and the version travels with
   * the choice. Nothing is sent from here. */
  async function preview(kind,id) {
    if(S.busy) return;
    S.busy=true;
    try {
      const view=await Data.read('/api/tasks/recovery?'+new URLSearchParams({task_id:id}));
      const action=view.actions.find(a=>a.action===ACTIONS[kind]);
      if(!action?.available) throw Error(action?.reason || 'The task no longer admits this action. Refresh its status.');
      S.confirming={kind,id,version:view.task_record_hash,lifecycle:view.lifecycle};
      openDialog(t('Task · explicit confirmation'),t(kind==='cancel' ? 'Request safe cancellation?' : 'Resume this retained Task?'),
        html`<div class="tp-confirm"><div class="tp-confirm-object">${icon('task')}<div><strong>${LiveViews.nameOf(view.status).name}</strong><p class="tp-truth">${t(view.status.goal_summary)}</p><p><span class="mono">${id}</span> · ${codeWords(view.lifecycle)} · ${view.verified_stage_count} / ${view.total_stage_count} ${t('verified')} · ${t('version')} <span class="mono">${short(view.task_record_hash)}</span></p></div></div><dl class="tp-identity"><div><dt>${t('Scope')}</dt><dd>${t(action.scope)}</dd></div><div><dt>${t('What changes')}</dt><dd data-tip="${action.expected_effect}">${effectWords(action)}</dd></div><div><dt>${t('Why it is permitted')}</dt><dd>${t(action.reason)}</dd></div></dl><p class="tp-caption">${t('The owner acts on exactly this Task version; if it changes before you confirm, you will be shown the new state instead.')}</p></div>`,
        html`${btn(t('Confirm'),'task-commit','','button primary',true)}`);
    } catch(e) { if(e.name!=='AbortError')notify(e.message); }
    finally { S.busy=false; }
  }
  /* The action boundary: the view is read again and the choice is sent only for the version it
   * was made against; the owner refuses any other version at its own transaction. Three
   * outcomes are told apart: not sent (this reader saw the Task move first), sent and refused
   * without a change (the owner saw it move), and, for a resume, accepted for the queue but
   * refused later at Task Control (the next readback carries the dispatcher's fact). */
  async function commit() {
    if(S.busy || !S.confirming) return;
    if(S.confirming.kind === 'unrecoverable') return commitUnrecoverable(S.confirming.entries);
    const {kind,id,version}=S.confirming; S.confirming=null;S.busy=true;closeDialog();
    let renew=null, outcome='not-sent';
    try {
      const current=await readView(id);
      const action=current.actions.find(a=>a.action===ACTIONS[kind]);
      if(current.task_record_hash!==version || !action?.available) renew=current;
      else {
        try { const body=await Data.post(kind==='cancel' ? '/api/cancel' : '/api/recover',{task_id:id,expected_task_hash:version}); outcome='sent'; if(S.selected===id) { S.settling=true; if(body.lifecycle) S.record={...S.record,...body}; } }
        catch(e) { if(String(e.message).includes('local_application.confirmation_stale')) { outcome=e.body?.refused_at==='TASK_CONTROL' ? 'refused-at-owner' : 'refused-at-entry'; renew=await readView(id); } else throw e; }
      }
      await refresh();
      if(outcome==='sent' && kind==='recover' && S.selected===id && S.view && STALE_AT_OWNER.has(S.view.worker_failure?.code || '') && S.view.lifecycle!=='RUNNING' && S.view.lifecycle!=='SUCCEEDED') { outcome='refused-at-task-control'; renew=S.view; }
    } catch(e) { S.error=e.message;notify(e.message);paint(); } // said where the choice was made (the scene, not only the drawer)
    finally { S.busy=false;paint(); }
    if(renew) {
      const vars={state:renew.lifecycle,version:short(renew.task_record_hash)};
      if(outcome==='not-sent') notify('Not sent: the Task changed since you reviewed it and is now {state} (version {version}). Review it again.',vars);
      else if(outcome==='refused-at-owner' || outcome==='refused-at-task-control') notify('Sent and refused by Task Control without a change: the Task had moved to {state} (version {version}) before the owner acted. Review it again.',vars);
      else notify('Sent and refused at the operation entry without a change: the Task is now {state} (version {version}). Review it again.',vars);
      const again=renew.actions.find(a=>a.action===ACTIONS[kind]);
      if(again?.available) void preview(kind,id);
    }
  }
  /* `wanted` is an automatic opener's intent (activity follow); it is re-checked after every
   * wait and right before the move, so a pin, another follow or the reader's own navigation
   * in the meantime abandons the open (returns false) instead of moving them. */
  async function openResult(id, wanted=()=>true) {
    let intent=Data.navigationIntent();
    const current=()=>wanted() && Data.navigationCurrent(intent);
    const closeResult=()=>{close();intent=Data.navigationIntent();}; // this open owns its synchronous inspector dismissal
    const readers={'factor.screening-development':'factor','alpha.model-development':'alpha','risk.covariance-development':'risk'};
    const kind=S.record?.task_id===id ? S.record.task_kind : Data.tasks().find(v=>v.task_id===id)?.task_kind;
    if(kind==='workspace_data_update' || kind==='workspace_preparation') {
      // a preparation's result is the maintained data the Data page shows
      if(!current()) return false;
      closeResult(); navigate('data', kind==='workspace_data_update' ? {update:id} : {}); return true;
    }
    if(kind==='research_input_capture') {
      const body=await Data.readShared('/api/research-inputs/readback?'+new URLSearchParams({task_id:id}));
      if(!current()) return false;
      if(body.status!=='SUCCEEDED' || !body.input_binding_hash) throw Error(body.failure_code || body.status);
      closeResult(); LiveWorkspace.openVersion(body.input_binding_hash); return true;
    }
    if(['alternative_evidence.document_intelligence','chief_risk_officer.portfolio_review'].includes(kind)) {
      if(!current()) return false;
      closeResult();
      const opening=LiveReview.openTask(id,current); intent=Data.navigationIntent(); return opening;
    }
    if(kind==='research_feature_materialization') {
      // the build's values are its result, read as their page (LS5), never a dialog
      const body=await Data.read('/api/features/build?'+new URLSearchParams({task_id:id}));
      if(!current()) return false;
      if(body.status!=='SUCCEEDED' || !body.receipt) throw Error(body.failure_code || body.status);
      closeResult();
      return LiveFeatures.showBuild(body);
    }
    if(kind && kind!=='research_experiment') {
      // An installed replay's result is a saved Portfolio result: the owner's index names it,
      // its REPORT opens it, and the saved entry is its reader. Other kinds keep their owner.
      const report=kind==='portfolio_public_development_replay' ? await Data.installedResult(id) : null;
      if(!current()) return false;
      closeResult();
      if(report) return LiveActivity.openSavedResult(report.result_hash, current);
      // Its inspector owns the synchronous address change; later reads keep the new intent.
      const opening=open(id,current); intent=Data.navigationIntent(); return opening;
    }
    // Metadata chooses a reader, not authority. Each study's reader owns its strict
    // readback; Alpha can first paint its qualified saved summary.
    if(kind==='research_experiment') {
      let experiment=Data.experiments()?.find(v=>v.task_id===id);
      if(!experiment) { await Data.refreshExperiments(); experiment=Data.experiments()?.find(v=>v.task_id===id); }
      if(!current()) return false;
      const page=readers[experiment?.kind];
      if(page) {
        closeResult(); objectEntry('study:'+id); return LiveStudy.open(id,page);
      }
    }
    const body=await Data.read('/api/experiments/readback?'+new URLSearchParams({task_id:id}));
    if(!current()) return false;
    if(body.status!=='EXPERIMENT_PUBLISHED') throw Error(body.status);
    closeResult();
    if(body.portfolio_source) { objectEntry('book:'+id); return Data.openPortfolio(id); }
    const page=readers[body.program?.kind];
    if(page){ objectEntry('study:'+id); return LiveStudy.open(id,page,body); }
    await Data.refreshHistory();
    if(!current()) return false;
    const entry='experiment:'+id;
    if(Data.history().some(v=>v.id===entry)) LiveViews.existing(entry);
    else openDialog(t('Saved object · readback'),id,'',LiveViews.savedObjectLink(t('Find it in History'),entry));
  }
  function bind() {
    document.addEventListener('visibilitychange',()=>{if(!document.hidden&&alive())refresh();});
    window.addEventListener('pagehide',()=>clearTimeout(S.timer));
    window.addEventListener('pageshow',(e)=>{if(e.persisted&&alive())refresh();});
    LiveActivity.bind(); void LiveActivity.refresh();
  }
  return {follow,page,open,refresh,refreshInspector,currentGroup,subjectContext,preview,commit,previewUnrecoverable,unrecoverableActions,previewRemedy,commitRemedy,replanPreview,replanCommit,turnIncidents,close,openResult,bind,paintActivity:paint,standing,liveness,dismissConfirmation:()=>{S.confirming=null;},toggle:()=>navigate('tasks'),view:()=>S.view,confirming:()=>S.confirming,selected:()=>S.selected};
})();
