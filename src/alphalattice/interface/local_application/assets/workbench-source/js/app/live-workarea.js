/* The shared long-task work area: one collapsible area on the owning product page, a stage
 * rail above a stage-specific body (the shown stage's own count, meter and facts on the left,
 * its bounded recent activity on the right; stacked on narrow screens). Two real consumers
 * compose it with their own truth: the first-use preparation scene (welcome) and the daily
 * update scene (data). Each hands in a scene spec -- its stages in Task Control's order, what
 * each stage does, its own count for the shown stage and its own activity reader -- and this
 * module owns what they share: the reader's explicit choices (fold, inspect, follow) for the
 * exact Task, kept across redraw, reconnect and reload through the viewer-preference store;
 * following the current stage by default and holding an inspected stage; the automatic hold
 * of a held log when the owner moves on; the bounded activity ring with one-time arrivals;
 * the meter that moves only between two reported counts; scroll-hold on the log; and each
 * scene's reading state kept, bounded, while another scene is read, so a return finds its
 * rows, fold, inspection and held reading as they were (rehydrated from the scene's retained
 * response when nothing is kept). A response that arrives for a page not shown is absorbed
 * into that scene's kept state and never takes over the visible area. Nothing here reads an
 * owner, submits work, or infers a lifecycle. */
const LiveWorkArea = (() => {
  const LOG_RETAIN=120;
  const clockText=(s)=>{ const n=Math.max(0,Math.floor(s)); const h=Math.floor(n/3600), m=Math.floor((n%3600)/60), sec=n%60; return (h ? h+':'+String(m).padStart(2,'0') : String(m))+':'+String(sec).padStart(2,'0'); };
  const readAge=(state)=>Math.max(0,Math.round((Date.now()-(state.readAt || Date.now()))/1000));
  const staleFor=(state)=>Math.max(0,Math.round((Date.now()-(state.stale?.since || Date.now()))/1000));
  const sinceWhen=(state)=>when(new Date(state.stale?.since || Date.now()).toISOString()); // since when a read has failed, as an instant (law 133)
  const clockOf=(iso)=>{const d=new Date(iso);return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString('en-GB',{hour12:false});};
  /* The stage the area follows: the Task's current stage while it has one; for a settled
   * Task the stage it stopped at (the first not verified), or the last when all are. */
  const shownStageOf=(v)=>v.status.current_stage || (v.stages.find(s=>s.lifecycle!=='VERIFIED') || v.stages.at(-1))?.stage_id || '';
  const moving=(v,state,current=true)=>current && v?.lifecycle==='RUNNING' && v.liveness.status==='OBSERVED' && !state.stale;

  /* The area's state for the one Task it shows: the viewer's choices, the activity ring, what
   * was last painted (for the meter's move, the settle and the one-time accent), and the stage
   * the view last showed (the automatic hold is decided on the observed transition). `A` is
   * the visible scene's state; the states of the last few other scene/Task pairs are kept in
   * KEPT (bounded, oldest out) and come back whole when their scene is shown again. */
  const RETAINED_AREAS=4;
  const FIELDS=['task','spec','folded','inspect','follow','log','shown','lastMeter','arrived','accent','lastLife','currentStage','logTop','marks','lastCount','lastLatest'];
  const A={task:null,spec:null,folded:false,inspect:null,follow:true,log:null,shown:null,lastMeter:null,arrived:new Set(),accent:false,lastLife:null,currentStage:null,logTop:null,marks:null,lastCount:null,lastLatest:null};
  const KEPT=new Map();
  const keyOf=(spec,taskId)=>spec.key+'|'+taskId;
  const snapshot=(area)=>Object.fromEntries(FIELDS.map(f=>[f,area[f]]));
  /* A scene/Task pair's fresh state: the viewer's kept choices for that exact Task, from the
   * shared `workScene` preference or, for the preparation scene, the `preparationScene` one the
   * accepted first-use scene wrote before the area was shared (neither is rewritten here). */
  function fresh(spec, taskId) {
    const saved=readPreference('workScene');
    const legacy=spec.key==='preparation' ? readPreference('preparationScene') : null;
    const mine=saved && saved.task===taskId ? saved : legacy && legacy.task===taskId ? legacy : null;
    return {task:taskId,spec,folded:Boolean(mine && mine.folded),inspect:mine && typeof mine.inspect==='string' ? mine.inspect : null,follow:mine ? mine.follow!==false : true,log:null,shown:null,lastMeter:null,arrived:new Set(),accent:false,lastLife:null,currentStage:null,logTop:null};
  }
  function keep(entry) {
    KEPT.delete(keyOf(entry.spec,entry.task));
    KEPT.set(keyOf(entry.spec,entry.task),entry);
    while(KEPT.size>RETAINED_AREAS) KEPT.delete(KEPT.keys().next().value);
  }
  /* The visible area for this scene and Task: the current one, the kept one (its rows, fold,
   * inspection and held reading as they were), or a fresh one. The area shown before is kept. */
  function areaFor(spec, taskId) {
    if(A.task===taskId && A.spec===spec) return A;
    if(A.task && A.spec) keep(snapshot(A));
    const key=keyOf(spec,taskId), kept=KEPT.get(key);
    if(kept) KEPT.delete(key);
    Object.assign(A, kept || fresh(spec,taskId));
    return A;
  }
  /* The state a response is absorbed into: the visible area when it is this scene and Task,
   * otherwise that pair's kept state (created if none), which never displaces the visible one. */
  function entryFor(spec, taskId) {
    if(A.task===taskId && A.spec===spec) return A;
    const key=keyOf(spec,taskId);
    let entry=KEPT.get(key);
    if(!entry) { entry=fresh(spec,taskId); keep(entry); }
    return entry;
  }
  const remember=()=>savePreference('workScene',{task:A.task,folded:A.folded,inspect:A.inspect,follow:A.follow});
  /* Reading held on a stage's activity when the owner moves on from that stage: the stage
   * stays shown (as an inspection, with the way back). Decided on the observed transition
   * itself; a reader who returned to current afterwards is not pulled back. */
  function noteTransition(view, area=A) {
    const current=view ? shownStageOf(view) : null;
    if(current && area.currentStage && current!==area.currentStage && !area.follow && !area.inspect && area.log?.stage===area.currentStage) { area.inspect=area.currentStage; if(area===A) remember(); }
    if(current) area.currentStage=current;
  }
  /* The activity ring: rows keyed by their subject in observation order, bounded to the
   * newest LOG_RETAIN; a row first seen on a live read is an arrival once, never on the first
   * paint of an execution, a stale read or the read that recovers from one. */
  function freshLog(key, stage, area=A) {
    if(!area.log || area.log.key!==key) area.log={key,stage,rows:new Map(),seen:new Set(),primed:false,snapshot:null};
    return area.log;
  }
  const liveRead=(state, area=A)=>Boolean(area.log) && !state.stale && !state.recovered && area.log.primed;
  const retain=(log)=>logRetain(log,LOG_RETAIN);

  /* ---- shells ---- */
  function logRail(latest, state, reading=true) {
    return html`<section class="ui-log-rail prep-log-rail" aria-label="${t('Latest unit and reading state')}"><div><span>${latest.label}</span><strong class="${state.stale ? 'is-stale' : ''}">${latest.value}</strong></div>${reading ? html`<div data-reading><span>${t('Reading')}</span><strong>${A.follow ? t('Following latest') : t('Reading held')}</strong></div>` : ''}</section>`;
  }
  /* The scene's log is the one log (round 72): the shell from `runLog`, the follow control as its
   * tool, the scene's own rows as its lines. */
  function logShell(spec, {label, title, caption, history, moving, rows, empty, status, rail, controls=moving}) {
    const control=controls ? html`<button type="button" class="text-btn" data-action="workspace-follow" aria-pressed="${!A.follow}" aria-label="${A.follow ? t('Hold the current reading position') : t('Resume following new units')}">${A.follow ? t('Hold reading') : t('Resume following')}</button>` : '';
    // R14 (WD4): a log that no longer moves is its work's record -- folded to its title and status line, its units a
    // press away (a blocked update's 60 units were 800 words of the first screen); a moving one is read live
    const fold=Boolean(history) && !moving;
    return html`<aside class="fv-work prep-work" data-history="${history}" aria-label="${label}">${runLog({id:'prepListingLog',title,caption,lines:rows,empty,status,controls:control,rail,cls:'prep-log',attrs:html`data-moving="${moving}"`,...(fold ? {fold:true,open:false,summary:html`${title} · ${countText(rows.length,'{n} record','{n} records')}`} : {})})}</aside>`;
  }
  function factsShell(spec, {label, title, caption, facts, rail}) {
    return html`<aside class="fv-work prep-work" aria-label="${label}">${rail}<section class="run-log"><header class="run-log-head"><div class="panel-label"><h3>${title}</h3>${infoMark(caption)}</div></header><div class="prep-feature-facts">${facts}</div></section></aside>`;
  }
  /* The retained record of one stage: Task Control's mark and its evidence, one unit. */
  function stageRecord(spec, v, shown) {
    const s=v.stages.find(x=>x.stage_id===shown);
    const rail=html`<section class="ui-log-rail prep-log-rail"><div><span>${t('Task Control')}</span><strong>${codeWords(s?.lifecycle || 'PENDING')}</strong></div><div><span>${t('Evidence')}</span><strong>${countText(s?.evidence_count ?? 0,'{n} reference','{n} references')}</strong></div></section>`;
    // A reference in words (its owner and kind) with a short token; the exact references fold under.
    const refWords=(ref)=>{const m=String(ref).match(/^playpen:\/\/([^/]+)\/([^/]+)\/([0-9a-f]+)$/i);return m ? html`${codeWords(m[2])} <span class="sub-cell">${codeWords(m[1])} · <span class="mono">${short(m[3], SHORT.hash)}</span></span>` : html`<span class="mono line-cut" data-tip="${String(ref)}">${String(ref)}</span>`;}; // law 88: cut by the line, whole on hover, never by a count
    const facts=s?.evidence?.length ? html`${kv(s.evidence.map((ref,i)=>[t('Evidence {i}',{i:i+1}),html`<span data-tip="${ref}">${refWords(ref)}</span>`]))}${codeRef(t('Exact references'), s.evidence.join('\n'), 'text')}` : html`<p class="caption">${t('No evidence is recorded for this stage yet.')}</p>`;
    return factsShell(spec,{label:t('Stage record'),title:t('Stage record'),caption:t((spec.oneUnit || {})[shown] || 'This stage is one unit; its verification is the next fact.'),facts,rail});
  }
  /* The rail: the scene's stages through the one step list, as buttons that inspect a stage. */
  const rail=(spec, v, shown, current, moving)=>stepList(v.stages.map((s,i)=>({id:s.stage_id,name:spec.steps[i]?.[1] ? t(spec.steps[i][1]) : codeWords(s.stage_id),mark:s.lifecycle,...(s.words ? {words:s.words} : {evidence:s.evidence_count}),current:s.stage_id===current,shown:s.stage_id===shown,done:s.lifecycle==='VERIFIED'})),{rail:true,cls:'prep-rail',stepCls:'prep-step',label:t(spec.railLabel),action:'workspace-stage',moving});
  /* The shown stage's body: what it does, its own count with the owner's denominator and
   * unit (or an honest "no finer count"), the secondary facts. Inspecting a stage that is not
   * the current one shows its retained record and offers the way back. */
  function stageBody(spec, b, v, state, shown, current, moving) {
    const index=v.stages.findIndex(s=>s.stage_id===shown), step=spec.steps[index], s=v.stages[index];
    const inspecting=!v.parallel && shown!==current;
    const w=inspecting ? null : spec.work(b,v,state,shown);
    const life=v.lifecycle, since=v.status.running_since, ended=!stateMoving(life);
    const span=v.status.timing?.running_seconds;
    const elapsed=since && Number.isFinite(span) && span>=0 ? clockText(span) : null, age=readAge(state), live=v.liveness, stale=state.stale;
    const kicker=v.parallel ? html`<p class="fv-kicker prep-kicker"><span>${t('Stage totals across parallel groups')}</span></p>` : inspecting ? html`<p class="fv-kicker prep-kicker"><span>${t('Inspecting')} · ${t('stage {i} of {n}',{i:index+1,n:v.stages.length})} · ${codeWords(s?.lifecycle || 'PENDING')}</span>${typedBtn(t('Return to current'),'workspace-stage','','button compact','')}</p>` : html`<p class="fv-kicker prep-kicker"><span>${t(!ended && v.status.current_stage ? 'Current stage' : life==='SUCCEEDED' && v.stages.every(x=>x.lifecycle==='VERIFIED') ? 'Last stage' : 'Stopped at')} · ${t('stage {i} of {n}',{i:index+1,n:v.stages.length})}${!ended && v.status.current_stage ? '' : html` · ${codeWords(s?.lifecycle || 'PENDING')}`}</span></p>`;
    const heading=html`<div class="prep-stage-heading"><span class="feature-icon stage-icon" data-moving="${moving && !inspecting}" aria-hidden="true">${icon('task')}</span><div class="prep-activity-copy"><h2 id="prepStageTitle">${step ? t(step[1]) : shown ? codeWords(shown) : t('No stage reported')}${infoMark(t(spec.lines[shown] || spec.fallbackLine))}</h2></div></div>`;
    let quantity;
    if(inspecting) {
      const retained=s?.lifecycle==='VERIFIED' ? t('Verified by Task Control; its record is kept with {n}. Inspecting it runs nothing.',{n:countText(s.evidence_count || 0,'{n} evidence reference','{n} evidence references')}) : ['BLOCKED','CANCELLED'].includes(s?.lifecycle) ? t('Recorded as {state} by Task Control; the owner\'s stop is explained above.',{state:codeWords(s.lifecycle)}) : t('Not started: Task Control has not verified the stages before it yet. Nothing here can start it.');
      quantity=html`<div class="prep-activity-count"><span class="tp-denom">${retained}</span></div>`;
    } else {
      const bar=w.bar ? meter({kind:'progress',now:w.bar[0],max:w.bar[1],id:'prepMeterFill',label:t(step ? step[1] : 'Stage'),cls:'prep-meter'}) : meter({kind:'progress',indeterminate:true,label:t(step ? step[1] : 'Stage'),cls:'prep-meter'}); // one meter (law 89); an unknown extent is the dashed track
      const wait=w.wait ? html`<p class="prep-wait">${icon('clock')}<span>${t('Waiting until {time} before the next attempt (the owner\'s retry time).',{time:when(w.wait)})}</span></p>` : ''; // N6 (law 133): the reader's clock, never the owner's raw instant
      quantity=html`<div class="prep-activity-count">${w.count}</div>${bar}<p class="tp-caption">${w.detail}</p>${wait}`;
    }
    // Elapsed is the owner's recorded span. Only the ages of the heartbeat and this page's read tick; neither changes Task progress.
    const readAt=state.readAt || Date.now();
    // Under an outage no clock ticks: whether the Task still runs is not known, and a moving
    // elapsed time would say it does.
    const elapsedValue=elapsed===null ? '' : elapsed;
    const workerValue=['OBSERVED','NOT_RECENT'].includes(live.status) ? html`<span data-tick="worker" data-base="${Math.round(live.age_seconds)}" data-read="${readAt}"${stale ? '' : ' data-live="true"'}>${t('{n} s',{n:Math.round(live.age_seconds)})}</span>` : blank('stat');
    const readValue=html`<span data-tick="read" data-read="${readAt}"${stale ? '' : ' data-live="true"'}>${t('{n} s ago',{n:age})}</span>`; // the read's age, said as one (the user's phase 6 reading)
    const facts=html`<div class="stat-strip prep-strip" aria-label="${t(spec.factsLabel)}">${stat(t('Verified stages'),html`${v.verified_stage_count} / ${v.total_stage_count}`,t('verified by Task Control · not elapsed time'))}${stat(t('Elapsed'),elapsedValue,!since ? t('the Task has not started running') : ended ? t('from its start to its last recorded activity') : t('since the Task started running'))}${stat(t('Worker'),workerValue,stale ? t('at the last successful read · not re-read since {time}',{time:sinceWhen(state)}) : live.status==='OBSERVED' ? t('since the last heartbeat · observable') : live.status==='NOT_RECENT' ? t('since the last heartbeat · not recent') : live.status==='NOT_OBSERVED' ? t('no heartbeat recorded yet · liveness unknown') : t('not executing · lifecycle decides'))}${stat(t('Last read'),readValue,stale ? t('since the last successful read · {n}',{n:countText(stale.failures,'{n} failed read','{n} failed reads')}) : t('since this page read the owners'))}</div>`;
    return html`<section class="fv-main prep-main" id="prepStageBody" data-shown="${shown}" data-inspecting="${inspecting}">${kicker}${heading}${quantity}${facts}</section>`;
  }
  function foldedSummary(spec, b, v, state, current, moving) {
    if(v.parallel)return stateLine(v.status,{next:''});
    const index=v.stages.findIndex(s=>s.stage_id===current), step=spec.steps[index], w=stateMoving(v.lifecycle) ? spec.work(b,v,state,current) : null;
    const [tone]=LiveTasks.standing(v);
    return html`<span class="prep-summary"><strong>${t('stage {i} of {n}',{i:index+1,n:v.stages.length})} · ${step ? t(step[1]) : current || ''}</strong>${w?.bar ? html` · <span class="prep-summary-count">${w.bar[0].toLocaleString('en-US')} / ${w.bar[1].toLocaleString('en-US')}</span>` : ''} · <span class="prep-summary-status" data-live="${moving}">${codeWords(v.stages[index]?.lifecycle || v.lifecycle)}</span>${tone==='attention' ? html` · <span class="prep-summary-attention" role="status">${icon('warning')}${t('needs attention')}</span>` : ''}</span>`;
  }
  function parallelFacts(spec,p) {
    const scope=p.scope, packing=scope?.packing_rules_id ? t('The run packs holdings from its admitted source counts. Holdings with nothing new and carried readings are outside these groups; the preview is an estimate.') : '';
    return html`<section class="execution-record panel-body" data-parallel-groups="${p.total}"><div class="stat-strip">${stat(t('Done'),html`${p.done} / ${p.total}`)}${stat(t('Running'),p.running)}${stat(t('Waiting'),p.waiting)}${stat(t('Blocked'),p.blocked)}${stat(t('Failed'),p.failed)}${scope ? html`${stat(t('Admitted issuers'),scope.issuers_total)}${stat(t('Nothing filed'),scope.issuers_nothing_filed)}${stat(t('Carried'),scope.issuers_carried)}` : ''}</div>${packing ? html`<p class="caption">${packing}</p>` : ''}${factsRef(t('Each group'),spec.parallelTable(p.groups))}</section>`;
  }
  function workArea(spec, b, v, state) {
    const area=areaFor(spec, v.task_id), current=shownStageOf(v);
    // Nothing kept for this scene and Task (a first paint, or its state aged out): the rows
    // are rebuilt from the scene's retained response, as a paint, never as arrivals.
    if(!area.log && b) spec.absorb(b,state,area);
    const shown=area.inspect && area.inspect!==current && v.stages.some(s=>s.stage_id===area.inspect) ? area.inspect : current;
    const moves=moving(v,state);
    const head=html`<header class="prep-area-head"><div class="prep-area-title">${icon('task')}<span>${t(spec.title)}</span>${area.folded ? foldedSummary(spec,b,v,state,current,moves) : ''}</div><button type="button" class="text-btn prep-fold" data-action="workspace-fold" aria-expanded="${!area.folded}" aria-controls="prepAreaBody">${icon(area.folded ? 'chevron' : 'close')}${area.folded ? t('Expand') : t('Collapse')}</button></header>`;
    const body=area.folded ? '' : html`${v.parallel ? parallelFacts(spec,v.parallel) : ''}${rail(spec,v,shown,v.parallel ? '' : current,moves)}<div class="fv-body prep-body" id="prepAreaBody">${stageBody(spec,b,v,state,shown,current,moves)}${spec.aside(b,v,state,shown,shown===current) || stageRecord(spec,v,shown)}</div>`;
    return html`<section class="fv-surface prep-area" data-box="workspace" data-scene="${spec.key}" data-folded="${area.folded}" data-shown="${shown}" data-moving="${moves}">${head}${body}</section>`;
  }

  /* ---- the viewer's explicit choices. None of them reads, submits or changes a Task. ---- */
  function fold() { if(!A.task) return; A.folded=!A.folded; remember(); render(); }
  function inspect(stageId) {
    if(!A.task || !A.spec) return;
    // Selecting the current stage is following it, not holding it.
    const view=A.spec.view(), current=view ? shownStageOf(view) : null;
    A.inspect=stageId && stageId!==current ? stageId : null; remember(); render();
    // Returning to the current stage removes the button that had the focus: keep the keyboard
    // on the rail, at the stage now shown.
    if(!A.inspect && typeof document!=='undefined') document.querySelector('.prep-step[data-value="'+current+'"]')?.focus({preventScroll:true});
  }
  function follow() {
    A.follow=!A.follow; remember(); const el=typeof document!=='undefined' ? document.getElementById('prepListingLog') : null;
    if(A.follow && el) moveLog(el,el.scrollHeight);
    patchMain();
  }
  /* The log's position as this module knows it: what it last set (following to the end,
   * resuming) or last observed. The page's repaint restores nested scroll positions, so the
   * scroll it causes lands exactly here (clamped to the log's range when the log shrank):
   * nothing moved for the reader and no hold changes. Every other position is the reader's
   * own movement -- keyboard, wheel, touch or scrollbar, however soon after a paint -- and
   * carries the intent: away from the end holds the reading, back to the end resumes it. */
  const moveLog=(el, top)=>logMove(el,top,A);
  const heldByScroll=(el)=>logHeld(el,A,()=>{
    remember();
    if(typeof document==='undefined') return;
    const button=document.querySelector('[data-action="workspace-follow"]');
    if(button){button.textContent=A.follow ? t('Hold reading') : t('Resume following');button.setAttribute('aria-pressed',String(!A.follow));}
    for(const strong of document.querySelectorAll('.prep-log-rail>[data-reading] strong')) strong.textContent=A.follow ? t('Following latest') : t('Reading held');
  });
  /* After a paint of the area's page: the meter moves from its previous reported fraction to
   * the new one (one transition between two real counts, never an invented intermediate), rows
   * first seen on a live read light up once, the log keeps following or holds, a changed stage
   * body settles, and a completion is accented once. Reduced motion disables every one of these
   * through the shared stylesheet; nothing here changes meaning. */
  function afterPaint() {
    if(!A.spec || app.page!==A.spec.page || typeof document==='undefined') return;
    const fill=document.getElementById('prepMeterFill');
    if(fill) {
      const fraction=fill.dataset.fraction, key=A.task+'|'+(document.getElementById('prepStageBody')?.dataset.shown || '');
      const previous=A.lastMeter && A.lastMeter.key===key ? A.lastMeter.fraction : null;
      if(previous!==null && previous!==fraction && fill.dataset.painted!==fraction) {
        fill.removeAttribute('data-ui-style');
        fill.style.width=(100*Number(previous)).toFixed(3)+'%';
        void fill.offsetWidth; // the previous width is laid out before the change
        fill.style.width=(100*Number(fraction)).toFixed(3)+'%';
      }
      fill.dataset.painted=fraction;
      A.lastMeter={key,fraction};
    }
    const reduced=matchMedia('(prefers-reduced-motion:reduce)').matches;
    // A stage Task Control verified since the last paint pops once; every other paint leaves
    // the rail still. The marks are the rail's own attributes, per Task.
    const rail=document.querySelector('.prep-rail');
    if(rail) {
      const marks={}; for(const step of rail.querySelectorAll('.prep-step[data-value]')) marks[step.dataset.value]=step.dataset.state;
      if(A.marks && A.marks.task===A.task && !reduced) for(const step of rail.querySelectorAll('.prep-step[data-value]')) { const was=A.marks.by[step.dataset.value]; if(was && was!=='VERIFIED' && step.dataset.state==='VERIFIED') step.classList.add('just-verified'); }
      A.marks={task:A.task,by:marks};
    }
    const countNode=document.querySelector('#prepStageBody .prep-activity-count .tp-number');
    if(countNode) {
      const key=A.task+'|'+(document.getElementById('prepStageBody')?.dataset.shown || ''), text=countNode.textContent;
      if(A.lastCount && A.lastCount.key===key && A.lastCount.text!==text && !reduced) countNode.closest('.prep-activity-count').classList.add('prep-tick');
      A.lastCount={key,text};
    }
    const latest=document.querySelector('.prep-log-rail > div:first-child strong');
    if(latest) {
      const text=latest.textContent;
      if(A.lastLatest && A.lastLatest.task===A.task && A.lastLatest.text!==text && !reduced) latest.classList.add('just-changed');
      A.lastLatest={task:A.task,text};
    }
    const body=document.getElementById('prepStageBody');
    if(body) {
      const shown=body.dataset.shown, changed=A.shown!==shown;
      if(A.shown && changed) body.classList.add('prep-settle');
      // A narrow rail scrolls; the shown stage is brought into it when it changes, never on every paint.
      const rail=document.querySelector('.prep-rail'), step=rail?.querySelector('.prep-step[data-value="'+shown+'"]');
      if(changed && rail && step && rail.scrollWidth>rail.clientWidth+1) rail.scrollLeft=Math.max(0,step.offsetLeft-8);
      A.shown=shown;
    }
    const log=document.getElementById('prepListingLog');
    if(log) {
      if(A.arrived.size && !matchMedia('(prefers-reduced-motion:reduce)').matches) for(const row of log.querySelectorAll('[data-listing-key]')) if(A.arrived.has(row.dataset.listingKey)) row.classList.add('ui-arrived');
      if(A.follow) moveLog(log,log.scrollHeight);
    }
    A.arrived.clear();
    if(A.accent) { const done=document.querySelector('.lab-work-done > .note-line, .data-update-row > .list-row'); if(done){done.classList.add('done-accent');A.accent=false;} } // N6: the one line a finished Task became, lit once
  }
  /* Only reading ages tick between owner answers. The Task's elapsed span never extrapolates. */
  function tick() {
    if(typeof document==='undefined' || document.hidden) return;
    const now=Date.now();
    for(const el of document.querySelectorAll('[data-tick][data-live="true"]')) {
      const kind=el.dataset.tick;
      if(kind==='worker') { const read=Number(el.dataset.read), base=Number(el.dataset.base); if(Number.isFinite(read) && Number.isFinite(base)) el.textContent=t('{n} s',{n:base+Math.max(0,Math.round((now-read)/1000))}); }
      else if(kind==='read') { const read=Number(el.dataset.read); if(Number.isFinite(read)) el.textContent=t('{n} s ago',{n:Math.max(0,Math.round((now-read)/1000))}); }
    }
  }
  function bind() {
    if(typeof document==='undefined') return;
    document.addEventListener('scroll',(e)=>{ if(e.target?.id==='prepListingLog') heldByScroll(e.target); },true);
    setInterval(tick,1000);
  }
  const area=()=>({task:A.task,scene:A.spec?.key || null,folded:A.folded,inspect:A.inspect,follow:A.follow,shown:A.shown,rows:A.log ? A.log.rows.size : 0,snapshot:A.log?.snapshot || null,arrived:A.arrived.size,kept:[...KEPT.keys()],logTop:A.logTop});
  return {A,LOG_RETAIN,short,clockText,readAge,staleFor,sinceWhen,clockOf,shownStageOf,moving,areaFor,entryFor,remember,noteTransition,freshLog,liveRead,retain,logRail,logShell,factsShell,stageRecord,workArea,fold,inspect,follow,heldByScroll,afterPaint,bind,area};
})();
