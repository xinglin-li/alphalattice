// The returning user's surfaces over the owners' bodies (shaped after the tmp/qa37 recordings of
// the real update, publication, decision and storage flows): the data page's freshness and update
// scene on the shared work area, exact reuse said as reuse, the selected update Task and typed
// refusals, the issues page's decisions and stale-confirmation refusal, the input versions and
// publication preview, the storage summary and cleanup preview. No filesystem, HTTP, model or
// data work; nothing here is a claim about the owners beyond the recorded shapes.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const library=require('./workbench_library.cjs'); // the library's constants and the scripts' parameters, from the source (Q2)
const finish=library.guard('workbench_daily');
// the shared percentage and reading helpers as components.js defines them (by their markers)
const runShapes=(root)=>{const src=require('node:fs').readFileSync(require('node:path').join(root,'components.js'),'utf8');const a=src.indexOf('/* ---- run shapes (round 72)');const b=src.indexOf('/* ---- end of run shapes ---- */');return src.slice(a,b)+';globalThis.stepList=stepList;globalThis.runLog=runLog;globalThis.logLine=logLine;globalThis.observationLine=observationLine;globalThis.logMove=logMove;globalThis.logHeld=logHeld;globalThis.logRetain=logRetain;';};
const percentRule=(root)=>library.readingSource(root);
const root=process.argv[2],reads=[],posts=[],tasks=[],drafts=[],renders=[],patches=[],prefs={},factBodies=new Map();
const UPDATE='475fbace-48a9-48d5-9613-7192c8b87c64', EARLIER='23ab76b8-3671-4d6c-b752-6a480a876930', CAPTURE='4be453ef-0deb-45be-92ae-9f2503554587';
const OLD='65451da2923afcbe4a00ac53dad4845873118f38bd530031932b76e1bcad4cc2', NEW='87082400a3476d15b6da425a956abd68c0291de492fa6f6a2d70cce65cde432d';
const inputs=(data,panel)=>({data_through:data,panel_through:panel,adjusted_through:data,readiness_status:'RESEARCH_READY',sources_checked_at:'2026-08-03T23:00:50',foundation_disposition:'PRIOR_INPUTS',manifest_revision:'c8634cdd',data_revision_hash:'d810730c',panel_hash:'cc93b42d'});
const membership={bootstrap:{t0_session:'2026-07-31',history_start:'2016-08-01',cohort_size:120,cohort_hash:'3eee8d3c',derivation:'FIRST_QUALIFIED_PUBLICATION',initialization_assumption:'INITIAL_COHORT_BACKFILL_NOT_POINT_IN_TIME',admitted_at:'2026-08-02T00:00:00+00:00',record_hash:'r'},journal_sequence:0,latest_effective_session:null,recent_events:[],recent_source_observations:[{observation_hash:'o'}]};
const readback=(extra={})=>({status:'NO_UPDATE_PUBLICATION',inputs:inputs('2026-07-31','2026-07-31'),current_input_failure:null,receipt:null,membership,foundation_action:'UNCHANGED_NO_RESEARCH_OR_REVIEW',task_id:null,task_lifecycle:null,selected:false,latest_task_id:null,plan_hash:null,retry_after_at:null,next_action:null,partition_reuse:{composed_partition_count:11,reused_partition_count:0,source:'CURRENT_PANEL'},...extra});
const STAGES=['validate_update_request','maintain_data_feature','publish_update_receipt'];
const view=(id,lifecycle,stage,extra={})=>({task_id:id,task_kind:'workspace_data_update',lifecycle,operation_running:false,cancellation:'NOT_REQUESTED',stop:null,worker_failure:null,
  liveness:{status:['RUNNING','CANCEL_REQUESTED'].includes(lifecycle)?'OBSERVED':'NOT_APPLICABLE',telemetry:'OPERATIONAL',age_seconds:0.4,signal_sequence:2},verified_stage_count:stage ? STAGES.indexOf(stage) : 3,total_stage_count:3,
  stages:STAGES.map((sid,i)=>({stage_id:sid,lifecycle:!stage ? 'VERIFIED' : i<STAGES.indexOf(stage) ? 'VERIFIED' : sid===stage ? (lifecycle==='RUNNING' ? 'IN_PROGRESS' : lifecycle==='BLOCKED' ? 'BLOCKED' : 'PENDING') : 'PENDING',evidence:i<STAGES.indexOf(stage||'x') ? ['ref'] : [],evidence_count:i<STAGES.indexOf(stage||'x') ? 1 : 0})),
  artifact_refs:[],actions:[{action:'CANCEL',operation:'CANCEL',available:lifecycle==='RUNNING',reason:'r'},{action:'RECOVER',operation:'RECOVER',available:lifecycle==='RECOVERY_REQUIRED',reason:'not interrupted'},{action:'REPLAN',operation:'WORKSPACE_PREPARE_PLAN',available:true,reason:'r'}],
  status:{task_id:id,lifecycle,current_stage:stage || null,verified_stage_count:stage ? STAGES.indexOf(stage) : 3,total_stage_count:3,running_since:'2026-08-03T23:00:49+00:00',last_activity_at:'2026-08-03T23:01:46+00:00',latest_failure_code:lifecycle==='BLOCKED' ? 'data.truth_review_required' : null},task_record_hash:'b'.repeat(64),observed_at:'2026-08-03T23:01:50+00:00',...extra});
const rows=(n,from=0)=>Array.from({length:n},(_,i)=>({listing_id:'l'+(from+i),symbol:'F'+String(from+i).padStart(3,'0'),state:'UPDATED',failure_code:null,raw_through:'2026-08-03',new_sessions:['2026-08-03'],restated_sessions:0,audit_scope:'ROLLING',sentinel_disposition:'ANCHORED',attempt_count:1,updated_at:'2026-08-03T23:00:5'+(i%10)}));
const maintenance=(updated,failed=0,extraRows=[])=>{const corrected=extraRows.filter(r=>r.state==='UPDATED' && !r.new_sessions.length && r.restated_sessions).length, reverified=extraRows.filter(r=>r.state==='UPDATED' && !r.new_sessions.length && !r.restated_sessions).length; const all=updated+corrected+reverified; return {availability:'AVAILABLE',maintenance_id:'m1',cycle_id:'2b30141d',lifecycle:all+failed>=120 ? 'COMPLETED' : 'RUNNING',as_of_session:'2026-08-03',manifest_revision:'c8634cdd',age_seconds:2,counts:{listings:120,processed:all+failed,updated:all,failed,pending:120-all-failed,with_new_sessions:updated,corrected,reverified},rows:[...rows(updated),...extraRows],retained:updated+extraRows.length,moved:all+failed};};
const cycle=(phase,status,extra={})=>({cycle_id:'2b30141d',phase,status,updated_at:'2026-08-03T23:01:13+00:00',age_seconds:3,retry_after_at:null,failure_code:null,transport_workers:4,change_set:null,...extra});
const changeSet={change_set_hash:'49bfea47',listings_with_new_sessions:120,new_sessions:120,listings_with_corrections:0,restated_sessions:0,membership_additions:0,membership_removals:0,sector_revision_changed:false,receipts:361};
const receipt={plan_hash:'f412dca3',content_hash:'3f51c064f804ca1a',maintenance_request_hash:'q',cycle_id:'2b30141d',target_session:'2026-08-03',completed_at:'2026-08-03T23:01:46Z',before:inputs('2026-07-31','2026-07-31'),after:inputs('2026-08-03','2026-08-03'),child_task_refs:['95573d7a'],effect_receipts:['e1','e2']};
const bodies={
  '/api/data-update':readback(),
  '/api/data-update/plan':{status:'PLANNED',plan_hash:'f412dca318ff40f4',target_session:'2026-08-03',inputs:inputs('2026-07-31','2026-07-31'),work:'Check due sources, maintain Data/Feature, verify Panel; no Factor research.',source_access:'HOST_SUPPLIED_PROVIDER',source_check_due:true,next_action:'DATA_UPDATE_RUN',candidate_recheck:null,candidate_data_recheck:null,change:null,audit_labels:[]},
  '/api/data-update/confirm':{status:'APPROVED'},'/api/data-update/run':{status:'ADMITTED',task_id:UPDATE,lifecycle:'QUEUED',failure_code:null},
  '/api/research-inputs':{status:'AVAILABLE',inputs:[{input_id:'factor-development',versions:[{binding_hash:OLD,start:'2016-08-01',end:'2026-07-31',configured_default:true,publication_hash:null,lifecycle:'REGISTERED',available:true},{binding_hash:NEW,start:'2016-08-01',end:'2026-08-03',configured_default:false,publication_hash:'4ded46d7',lifecycle:'SUCCEEDED',available:true}]}]},
  '/api/research-inputs/plan':{status:'CONFIRMATION_REQUIRED',plan_hash:'9efbcc15a77b0e9c',input_id:'factor-development',prior_binding_hash:OLD,before:{binding_hash:'b',manifest_revision:'m',data_revision_hash:'973570bd',panel_snapshot_hash:'36b5d3db',panel_through:'2026-07-31',outcome_watermark_hash:'832a90e5',outcome_recipe_hash:'r',outcome_policy_hash:'p',materializer_hash:'mat'},after:{binding_hash:'b',manifest_revision:'m',data_revision_hash:'d810730c',panel_snapshot_hash:'cc93b42d',panel_through:'2026-08-03',outcome_watermark_hash:'a8f3227e',outcome_recipe_hash:'r',outcome_policy_hash:'p',materializer_hash:'mat'},network_calls:0,numerical_calls:0,limits:['EXPLICIT_VERSION_ONLY','DEFAULTS_AND_CU_UNCHANGED','NO_MODEL_TRAINING']},
  '/api/research-inputs/confirm':{status:'ADMITTED',task_id:CAPTURE,lifecycle:'QUEUED',failure_code:null},
  '/api/research-inputs/readback':{status:'SUCCEEDED',task_id:CAPTURE,failure_code:null,publication:{binding_hash:NEW,prior_binding_hash:OLD,anchor_binding_hash:OLD,published_at:'2026-08-03T23:04:55+00:00',receipt_hash:'4ded46d7dfcc4bb8',source:{panel_through:'2026-08-03'}}},
  '/api/workspace/preparation':{status:'RESEARCH_INPUTS_READY',inputs:[{input_id:'factor-development',binding_hash:OLD}],task_id:null,selected:false,latest_task_id:null},
  '/api/workspace/data-issues':{status:'NO_PENDING_CASE_CATALOG',issues:[],continuations:[],recorded_decisions:[],continued_dispositions:[],next_cursor:null,claim_limit:'NO_DATA_TRUTH_OR_CURRENT_READINESS_CLAIM'},
  '/api/workspace/data-issues/preview':{status:'CONFIRMATION_REQUIRED',confirmation:'HUMAN',case_token:'701f330072b97351',option:{option_id:'recoverable_quarantine',disposition:'auto',policy_args:{action:'quarantine_listing',recheck_after_at:'2026-08-04T23:00:45Z'}},next_action:'DATA_ISSUE_CONFIRM',effect_applied:false,
    next_requests:{confirm:{operation:'DATA_ISSUE_CONFIRM',data_issue_case_token:'701f330072b97351',data_issue_evidence_hash:'1587504c',data_issue_option_id:'recoverable_quarantine',data_issue_option_hash:'h2'}}},
  '/api/workspace/data-issues/confirm':{status:'CONFIRMED_PENDING_REVALIDATION',receipt_hash:'326dfabe18',case_token:'701f330072b97351',next_action:'RESUME_APPROVED_WORK',continuations:[]},
  '/api/workspace/storage':{status:'AVAILABLE',inputs:[{binding_hash:OLD,panel_snapshot_hash:'36b5d3db',roots:['CURRENT_ACTIVE','PREVIOUS_ROLLBACK'],logical_bytes:388626012,available:true},{binding_hash:NEW,panel_snapshot_hash:'cc93b42d',roots:['CURRENT_RESEARCH_VERSION'],logical_bytes:424073272,available:true}],managed_bytes:1318904021,logical_bytes:2272589016,by_root:{ARTIFACTS:{physical_bytes:343704636,logical_bytes:486467068,files:337},IMMUTABLE_INPUTS:{physical_bytes:686529110,logical_bytes:1497451673,files:101},PROVIDER_CACHE:{physical_bytes:0,logical_bytes:0,files:0},RESEARCH_EXPERIMENTS:{physical_bytes:0,logical_bytes:0,files:0},STAGING:{physical_bytes:37443,logical_bytes:37443,files:1},WORKING_DATABASE:{physical_bytes:288632832,logical_bytes:288632832,files:1}},free_disk_bytes:5518340751360,capacity_status:'WITHIN_BUDGET',display:{managed:'1.23 GiB (1,318,904,021 bytes)',free:'5,139.36 GiB (5,518,340,751,360 bytes)',logical:'2.12 GiB (2,272,589,016 bytes)',cap:'10.00 GiB (10,737,418,240 bytes)'},budget:{managed_cap_bytes:10737418240},references_hash:'h',pending_cleanup:[]},
  '/api/workspace/storage/plan':{status:'CONFIRMATION_REQUIRED',plan_hash:'77f93a79b4e3bd10',targets:{},bindings:[],retained_bytes:1318904021,reclaimable_bytes:0,limitations:['CLASSIFIED_INPUT_AND_PANEL_OUTCOME_SNAPSHOTS_ONLY','REPLAY_REQUIRES_RETAINED_INPUTS']},
  '/api/workspace/storage/confirm':{status:'COMPLETED',deleted_paths:1,plan_hash:'77f93a79b4e3bd10'},'/api/workspace/storage/pin':{status:'PINNED',binding_hash:OLD},
};
const views={};
let failed=null,failure='owner_refused',modal=null,projections=[],draft={input_id:'',input_binding_hash:''};
const c={console,URLSearchParams,app:{page:'data',data:'',feature:''},ROUTES:{data:['','Data'],issues:['','Issues'],inputs:['','Inputs'],storage:['','Storage'],welcome:['','Welcome']},
  Data:{read:async p=>{reads.push(p);const [key,query]=p.split('?');const id=new URLSearchParams(query||'').get('task_id');if(key==='/api/tasks/recovery'){if(!views[id])throw Error('local_web.task_id_invalid');return views[id];}if(key==='/api/data-update'){if(id){if(!bodies.byTask||!bodies.byTask[id])throw Error('workspace_data_update.task_not_found');return bodies.byTask[id];}if(bodies.byTask&&bodies.latest)return {...bodies.byTask[bodies.latest],selected:false};}return bodies[key]!==undefined?bodies[key]:bodies[p];},
    post:async(p,b)=>{posts.push([p,JSON.parse(JSON.stringify(b))]);if(p===failed)throw Error(failure);return bodies[p];},setInputs(b){c.Data._inputs=(b?.inputs||[]).flatMap(i=>(i.versions||[]).map(v=>({id:i.input_id,binding_hash:v.binding_hash,start:v.start||null,cutoff:v.end||null,available:v.available===true,lifecycle:v.lifecycle,pinned:v.configured_default===true})));},
    setPreparation(){},preparation:()=>null,inputVersion:h=>(c.Data._inputs||[]).find(v=>v.binding_hash===h)||null,inputs:()=>c.Data._inputs||[],workspace:()=>'daily-qa',tasks:()=>projections,runsOf:(k)=>c.Data.tasks().filter(v=>k==='task'||['workspace_data_update','workspace_preparation'].includes(v.task_kind)).map(v=>({id:v.task_id,kind:v.task_kind,name:v.task_kind,state:v.lifecycle,starter:'',started:v.running_since||'',finished:v.last_activity_at||'',current:v.current_stage||'',verified:[v.verified_stage_count,v.total_stage_count],object:v})),groupByDay:(rs)=>{const g=new Map();for(const r of rs){const k=String(r.finished||r.started||'').slice(0,10)||'Undated';if(!g.has(k))g.set(k,[]);g.get(k).push(r);}return g;},_inputs:[]},
  LiveTasks:{select:async id=>tasks.push(id),standing:v=>[['BLOCKED','DEFERRED','RECOVERY_REQUIRED'].includes(v.lifecycle) ? 'attention' : 'info','info',v.lifecycle],liveness:v=>v.liveness?.status||''},LiveResearch:{useInput:async key=>drafts.push(key),context:()=>draft},
  html:(s,...v)=>s.reduce((a,p,i)=>a+p+(Array.isArray(v[i]) ? v[i].join('') : (v[i]??'')),''),t:(s,vars)=>String(s).replace(/^[a-z0-9-]+\|/,'').replace(/\{(\w+)\}/g,(m,k)=>vars&&vars[k]!==undefined?vars[k]:m),render(){renders.push(c.app.page);},patchMain(){patches.push(c.app.page);},closeDialog(){},
  openDialog:(a,b,content)=>{modal={title:b,content};},
  emptyState:(fact,opener='')=>'EMPTY('+fact+')'+(opener||''),meter:()=>'',notRead:(title,error,words='',action='')=>`<div class="warning">${title}:${error}${words?' '+words:''}${action}</div>`,noteLine:(title,body='',tone='',action='')=>`<p class="note-line">${title}${body?' · '+body:''}${action||''}</p>`,hint:(term)=>term,factsRef:(title,body)=>{factBodies.set(title,String(body));return `<p>${title}</p>${body}`;},codeRef:(title,text)=>'<p>'+title+'</p><template>'+(typeof text==='string' ? text : JSON.stringify(text))+'</template>',refCell:(uri)=>'<span>'+uri+'</span>',hashCell:(h)=>'<span>'+(h||'—')+'</span>',readingPane:(title,kind,body,close)=>'<aside>'+title+body+'</aside>',sourceRows:(rows)=>rows.map((r)=>[r.label,r.value]),kv:rows=>rows.map(([k,v])=>k+'='+v).join(';'),panel:(a,b,content)=>'['+a+'|'+b+']'+content,rail:(items)=>String(items),groupHead:(label,count)=>'GROUP('+label+':'+count+')',listFoot:()=>'',objectRow:(s={},x)=>'ROW('+[s.name,s.why,...(((x&&x.props)||[]).filter(Boolean).map(p=>Array.isArray(p)?p[1]:p)),x&&x.actions||''].filter(Boolean).join('|')+')',separator:()=>'',stat:(a,b,n)=>'STAT('+a+':'+b+':'+(n||'')+')',figureTile:(label,value,to,note)=>'FIG('+label+':'+value+(to?'>'+(to.page||to.action):'')+')',measureStrip:(items)=>String(items),tabStrip:(label,items)=>'TABS('+items.map(x=>x.word+(x.count?'='+x.count:'')+(x.on?'*':'')).join(',')+')',refusal:(b,tone='warning',o={})=>'REFUSAL('+(tone)+':'+(o.word||o.state||'refused')+'|'+(b?.code||b?.failure_code||'')+'|'+String(b?.reason||'')+')'+(o.action||''),banner:(a,b,tone,action)=>'BANNER('+(tone||'neutral')+':'+a+'|'+b+')'+(action||''),dateRange:(a,b)=>(a||'—')+' — '+(b||'—'),dateMove:(a,b)=>(a||'—')+' → '+(b||'—'),badge:(s,l)=>'BADGE('+s+':'+(l||'')+')',stateLine:(x,o={})=>'BADGE('+(typeof x==='string'?x:(x?.lifecycle??x?.state??x?.status??''))+':'+(o.word||'')+')',statusDot:(s,l)=>'DOT('+s+')',skeleton:(shape='rows')=>'SKELETON('+shape+')',icon:()=>'',STATUS_ATTR:'',
  PREPARATION_STEPS:[['freeze_sources','Freeze sources',''],['prepare_data','Prepare market data',''],['prepare_features','Prepare Features',''],['publish_inputs','Publish research inputs',''],['verify_inputs','Verify inputs','']],
  json:(v)=>JSON.stringify(v,null,2),typedBtn:(a,b,v,cls,reason)=>`BTN(${a}:${b}:${v}:${reason})`,btn:()=>'',link:(l,page,cls,extra)=>'LINK('+l+'>'+page+(extra&&Object.keys(extra).length ? JSON.stringify(extra) : '')+')',table:(h,r)=>'TABLE['+(Array.isArray(r) ? r.join('') : r)+']',tr:cells=>'ROW('+cells.join('|')+')',picker:(id,choices,o={})=>choices.map(ch=>{const [v]=Array.isArray(ch)?ch:[ch.value];return v+(v===o.selected ? '*' : '');}).join(','),objectHead:(name,meta,actions,state,tools,o={})=>'H1:'+name+'|'+(meta||'')+'|'+(actions||'')+(state?'|'+state:'')+(o.facts||[]).map(([k,v])=>`FACT(${k}=${v})`).join('')+(o.id?`ID(${o.id})`:'')};
c.readPreference=k=>prefs[k];
// the lobby's own buttons (a group's head, `Show n more`), by their action
c.btnAttrs=(label,action,value)=>`ACT(${label}:${action}:${value})`;
c.savePreference=(k,v)=>{prefs[k]=JSON.parse(JSON.stringify(v));};
// a history entry pushed with `location.href` keeps the address (the stub's href is empty); a hash sets it
c.location={hash:'#page=data',href:''};c.history={pushState:(_a,_b,hash)=>{if(typeof hash==='string'&&hash.startsWith('#'))c.location.hash=hash;},replaceState:(_a,_b,hash)=>{if(typeof hash==='string'&&hash.startsWith('#'))c.location.hash=hash;}};
c.Data.readShared = (...args) => c.Data.read(...args);
vm.createContext(library.into(c,root));vm.runInContext(fs.readFileSync(path.join(root,'status.js'),'utf8'),c);vm.runInContext(percentRule(root),c);vm.runInContext(runShapes(root),c);
vm.runInContext(fs.readFileSync(path.join(root,'router.js'),'utf8'),c);
c.render=()=>renders.push(c.app.page);c.patchMain=()=>patches.push(c.app.page);
vm.runInContext(fs.readFileSync(path.join(root,'live-workarea.js'),'utf8')+fs.readFileSync(path.join(root,'live-workspace.js'),'utf8')+';globalThis.w=LiveWorkspace;',c);
const settle=(ms=20)=>new Promise(r=>setTimeout(r,ms));
(async()=>{
  const w=c.w;
  // ---- the data page before any update: four cutoffs kept apart, the way in, nothing posted ----
  c.Data.setInputs(bodies['/api/research-inputs']);
  await w.refresh('data');let markup=w.page();
  // N6: the Data overview -- the owners' figures, each opening its page (the as-of the update's, the listings the membership's)
  assert.ok(markup.includes('FIG(As of:2026-07-31') && markup.includes('FIG(Listings:120>facts-open)') && !markup.includes('FIG(Input versions') && !markup.includes('>issues)') && !markup.includes('>storage)'),'the strip says the data\'s own facts; the versions and the storage are the other tabs (N5): '+markup.slice(0,400));
  assert.ok(markup.includes('EMPTY(No update recorded yet)BTN(Preview update:workspace-preview:update:)'),'the returning user sees the one explicit way in, in the empty list\'s place');
  assert.equal(posts.length,0,'reading the data page posts nothing');assert.equal(c.app.data,'2026-07-31');
  draft={input_id:'factor-development',input_binding_hash:OLD}; // the draft's version is marked on Research inputs (below), never an overview figure
  // ---- preview: the owner's plan in its own terms; commit admits; the scene is this page ----
  await w.preview('update');
  assert.ok(modal && modal.title==='Confirm data update' && modal.content.includes('Target session=2026-08-03') && modal.content.includes('Market data now through=2026-07-31') && modal.content.includes('Fetch the sessions after the current cutoff') && modal.content.includes('Membership sources=Due') && modal.content.includes('Published input versions, saved research and every selection'),'the preview says the target, the cutoffs, the work and what stays');
  bodies.byTask={[UPDATE]:readback({task_id:UPDATE,task_lifecycle:'RUNNING',selected:true,latest_task_id:UPDATE,plan_hash:'f412dca3',cycle:cycle('market_data','running'),maintenance:maintenance(25),work_progress:{availability:'BOUND',stage:'maintain_data_feature',stage_id:'market_data_increment',status:'RUNNING',completed_units:25,total_units:120,unit_name:'listings',age_seconds:1,counters:{updated:25,failed:0,pending:95}}})};
  bodies.latest=UPDATE;views[UPDATE]=view(UPDATE,'RUNNING','maintain_data_feature');
  await w.commit();
  assert.deepEqual(posts.map(v=>v[0]),['/api/data-update/plan','/api/data-update/run'],'a PLANNED update needs no approval step');
  assert.equal(c.location.hash,'#page=data&update='+UPDATE,'the admitted update is pinned in the route');assert.deepEqual(tasks,[],'no Task Center opening');
  markup=w.page();
  assert.ok(markup.includes('data-scene="update"') && markup.includes('data-shown="maintain_data_feature"') && markup.includes('data-moving="true"') && markup.includes('Update work'),'the update scene on the shared work area');
  assert.ok(markup.includes('<span class="tp-number">25</span><span class="tp-denom"> / 120 listings processed</span>') && markup.includes('25 with new sessions · 95 pending') && markup.includes('cycle market data: fetching and auditing listings · recorded'),'the maintenance runner\'s own units (a zero tally is not said) and the cycle phase with its instant');
  assert.ok(markup.includes('data-listing-key="l24"') && markup.includes('1 new session fetched through 2026-08-03') && markup.includes('25 of 120 listings moved · 25 shown'),'the listing rows are the maintenance runner\'s records, one line each');
  assert.ok(markup.includes('Validate the request') && markup.includes('Maintain data and Features') && markup.includes('Publish the receipt') && !markup.includes('Freeze sources'),'maintenance\'s three stages, never initialization\'s five');
  // ---- observe: one readback of the pinned Task and one recovery view per poll; the work advances ----
  const observed=reads.length;bodies.byTask[UPDATE]={...bodies.byTask[UPDATE],maintenance:maintenance(66),cycle:cycle('market_data','running')};
  w.observe();await settle();
  assert.deepEqual(reads.slice(observed),['/api/data-update?task_id='+UPDATE,'/api/tasks/recovery?task_id='+UPDATE],'one readback of the pinned Task and one recovery view');
  assert.ok(w.page().includes('<span class="tp-number">66</span>') && w.area().rows===66 && w.area().arrived===41,'new units are arrivals once');
  // the Feature owner's step is the count once the cycle builds Features
  bodies.byTask[UPDATE]={...bodies.byTask[UPDATE],maintenance:maintenance(120),cycle:cycle('feature','running'),work_progress:{availability:'BOUND',stage:'maintain_data_feature',stage_id:'panel_year_finalize',status:'RUNNING',completed_units:7,total_units:11,unit_name:'years',age_seconds:1,counters:{}}};
  w.observe();await settle();markup=w.page();
  assert.ok(markup.includes('<span class="tp-number">7</span><span class="tp-denom"> / 11 years</span>') && markup.includes('Panel year finalization') && markup.includes('cycle Features and Panel'),'the Feature owner\'s own step, bound to this Task');
  // fold keeps a compact summary; inspecting an earlier stage holds it; the preference is this Task's
  w.fold();markup=w.page();assert.ok(markup.includes('data-folded="true"') && markup.includes('stage 2 of 3') && markup.includes('7 / 11'),'folded summary');w.fold();
  w.inspect('validate_update_request');assert.ok(w.page().includes('data-shown="validate_update_request"') && w.page().includes('Inspecting'));assert.deepEqual(prefs.workScene,{task:UPDATE,folded:false,inspect:'validate_update_request',follow:true});w.inspect('');
  // a Task that completed between the readback and the view reads: the verified view stands beside a readback without its receipt; the cadence reads again until the owner says PUBLISHED, three times at most
  views[UPDATE]=view(UPDATE,'SUCCEEDED',null);projections=[{task_id:UPDATE,task_kind:'workspace_data_update',lifecycle:'SUCCEEDED',verified_stage_count:3}];
  const lateStart=reads.length;for(let i=0;i<6;i++){w.observe();await settle();}
  assert.equal(reads.slice(lateStart).filter(r=>r.startsWith('/api/data-update?')).length,4,'the transition read, then three late reads for a completed view without its receipt, then no more');
  assert.ok(w.page().includes('its receipt is being read') && !w.page().includes('2026-07-31 → 2026-08-03') && !w.page().includes('data-scene="update"'),'meanwhile the completed update\'s row says the receipt is being read, never a cutoff the readback does not show; its work is folded (N6)');
  views[UPDATE]=view(UPDATE,'RUNNING','maintain_data_feature');projections=[{task_id:UPDATE,task_kind:'workspace_data_update',lifecycle:'RUNNING',verified_stage_count:1}];await w.refresh('data');
  // ---- completed: what changed and what was reused, in the owner's receipt; exact reuse afterwards ----
  bodies.byTask[UPDATE]={...bodies.byTask[UPDATE],status:'PUBLISHED',task_lifecycle:'SUCCEEDED',inputs:inputs('2026-08-03','2026-08-03'),receipt,partition_reuse:{composed_partition_count:1,reused_partition_count:10,source:'RECEIPT_PANEL',panel_hash:'cc93b42d'},cycle:cycle('completed','completed',{change_set:changeSet}),maintenance:maintenance(120),work_progress:{availability:'NOT_CURRENT',stage:'maintain_data_feature',stage_id:'panel_year_finalize',status:'SUCCEEDED',completed_units:11,total_units:11,unit_name:'years',age_seconds:40}};
  views[UPDATE]=view(UPDATE,'SUCCEEDED',null);projections=[{task_id:UPDATE,task_kind:'workspace_data_update',lifecycle:'SUCCEEDED',verified_stage_count:3}];
  w.observe();await settle();markup=w.page();
  assert.ok(markup.includes('2026-07-31 → 2026-08-03') && markup.includes('<p class="data-update-counts">') && !markup.includes('Listings in the last update') && markup.includes('FIG(As of:2026-08-03') && !markup.includes('data-scene="update"'),'the completed update is its row -- the change it made -- with the runner counts under it, no separate figure; its work is folded (N6, laws 72, 98; the user, 2026-09-24)');
  assert.equal(c.app.data,'2026-08-03','the shell\'s data cutoff follows the readback');
  const caught=reads.length;w.observe();await settle();assert.equal(reads.length,caught,'once the readback carries the receipt nothing more is read');
  const before=posts.length;bodies['/api/data-update/run']={status:'REUSED_EXACT',task_id:null,receipt};
  await w.preview('update');await w.commit();
  assert.deepEqual(posts.slice(before).map(v=>v[0]),['/api/data-update/plan','/api/data-update/run']);
  assert.ok(w.page().includes('Nothing to update: the workspace already holds 2026-08-03') && w.page().includes('no Task was started and nothing was downloaded'),'exact reuse is said as reuse, never as a rebuild');
  assert.equal(c.location.hash,'#page=data&update='+UPDATE,'the route keeps the Task it showed');
  // ---- the shared area across scenes: the update's rows, fold and held reading survive a preparation scene and come back whole ----
  views[UPDATE]=view(UPDATE,'RUNNING','publish_update_receipt');projections=[{task_id:UPDATE,task_kind:'workspace_data_update',lifecycle:'RUNNING',verified_stage_count:2}];await w.refresh('data'); // N6: a settled update's work is folded; the area is read while it runs (its last stage, so inspecting maintenance holds it)
  w.inspect('maintain_data_feature');w.follow();markup=w.page();
  assert.ok(w.area().rows===120 && w.area().follow===false && markup.includes('data-listing-key="l119"'),'the update shows its retained rows with the reading held');
  const PREP='b2d7c0a1-1111-4222-8333-944444444444';
  const prepRows=[{listing_id:'p1',symbol:'P001',state:'RAW_READY',origin:'ACQUIRED',raw_through:'2026-08-03',failure_code:null,reasons:[],tail_acquired:false,observed_at:'2026-08-03T23:10:01+00:00',noted_at:'2026-08-03T23:10:01+00:00'},{listing_id:'p2',symbol:'P002',state:'RAW_READY',origin:'ACQUIRED',raw_through:'2026-08-03',failure_code:null,reasons:[],tail_acquired:false,observed_at:'2026-08-03T23:10:02+00:00',noted_at:'2026-08-03T23:10:02+00:00'}];
  const prepBody={status:'RUNNING',source_mode:'ACQUIRE_DECLARED_SOURCES_AFTER_CONFIRMATION',next_action:null,execution_binding_changed:false,task_id:PREP,plan_hash:'plan-1',failure_code:null,inputs:[],profile:'us-current-index-research',sources:['yfinance'],limits:['CURRENT_UNIVERSE_RESEARCH_ONLY','NO_DATA_API_KEY','NO_FOUNDATION_OR_STRATEGY_ACTIVATION'],selected:false,latest_task_id:PREP,progress:{phase:'prepare_data',candidates:120,raw_ready:2,quality_eligible:0,failed:0,retry_after_at:null},work_progress:null,listing_activity:{availability:'BOUND',stage:'prepare_data',execution_id:'e1',sequence:2,observed:2,retained:2,dropped:0,written_at:'2026-08-03T23:10:02+00:00',age_seconds:1,rows:prepRows}};
  const PREP_STAGES=['freeze_sources','prepare_data','prepare_features','publish_inputs','verify_inputs'];
  views[PREP]={...view(PREP,'RUNNING','prepare_data'),task_kind:'workspace_preparation',verified_stage_count:1,total_stage_count:5,stages:PREP_STAGES.map((sid,i)=>({stage_id:sid,lifecycle:i<1 ? 'VERIFIED' : sid==='prepare_data' ? 'IN_PROGRESS' : 'PENDING',evidence:i<1 ? ['ref'] : [],evidence_count:i<1 ? 1 : 0})),status:{task_id:PREP,lifecycle:'RUNNING',current_stage:'prepare_data',verified_stage_count:1,total_stage_count:5,running_since:'2026-08-03T23:09:00+00:00',last_activity_at:'2026-08-03T23:10:02+00:00',latest_failure_code:null}};
  bodies['/api/workspace/preparation']=prepBody;bodies['/api/workspace/preparation?task_id='+PREP]=prepBody;
  prefs.preparationScene={task:PREP,folded:true,inspect:null,follow:true}; // the accepted first-use scene's own preference, written before the area was shared
  // a read of the update started on the data page whose answer lands after the reader left for the preparation scene
  const lateUnit={listing_id:'l-late',symbol:'F120',state:'UPDATED',failure_code:null,raw_through:'2026-08-03',new_sessions:[],restated_sessions:1,audit_scope:'ROLLING',sentinel_disposition:'ANCHORED',attempt_count:1,updated_at:'2026-08-03T23:01:20+00:00'};
  bodies.byTask[UPDATE]={...bodies.byTask[UPDATE],maintenance:maintenance(120,0,[lateUnit])};
  const realRead=c.Data.read;let releaseLate;const gate=new Promise(r=>{releaseLate=r;});
  c.Data.read=async p=>{const v=await realRead(p);if(p.startsWith('/api/data-update')){c.Data.read=realRead;await gate;}return v;};
  const late=reads.length;const lateAnswer=w.refresh('data','',true);
  c.app.page='welcome';c.location.hash='#page=welcome&preparation='+PREP;markup=w.page();await settle();markup=w.page();
  assert.ok(markup.includes('data-scene="preparation"') && markup.includes('data-folded="true"') && w.area().scene==='preparation' && w.area().rows===2 && w.area().kept.includes('update|'+UPDATE),'the preparation scene shows its own Task folded as its accepted preference says; the update\'s state is kept, not lost');
  w.fold();assert.ok(w.page().includes('data-listing-key="p1"') && w.area().folded===false,'unfolded, the preparation scene shows its own two units');
  // the late answer lands in the update's kept state, never in the visible preparation area
  releaseLate();await lateAnswer;
  assert.ok(reads.slice(late).includes('/api/data-update?task_id='+UPDATE) && w.area().scene==='preparation' && w.area().rows===2 && w.area().task===PREP && w.area().kept.includes('update|'+UPDATE),'the visible area is still the preparation scene\'s after the data page\'s late answer');
  c.app.page='data';c.location.hash='#page=data&update='+UPDATE;const returned=reads.length;markup=w.page();await settle();markup=w.page();
  assert.ok(reads.length===returned && markup.includes('data-scene="update"') && markup.includes('data-shown="maintain_data_feature"') && markup.includes('data-listing-key="l119"') && markup.includes('data-listing-key="l-late"') && w.area().rows===120 && w.area().follow===false && w.area().inspect==='maintain_data_feature' && markup.includes('Resume following'),'back on the data page without a read: the rows (with the late unit, bounded), the inspection and the held reading are as they were: '+JSON.stringify({reads:reads.length-returned,scene:markup.includes('data-scene="update"'),shown:markup.includes('data-shown="maintain_data_feature"'),l119:markup.includes('data-listing-key="l119"'),late:markup.includes('data-listing-key="l-late"'),rows:w.area().rows,follow:w.area().follow,inspect:w.area().inspect,resume:markup.includes('Resume following')}));
  w.follow();
  // nothing kept for a scene/Task (its state aged out while other preparation Tasks were read): the rows are rebuilt from the scene's retained response, as a paint
  c.app.page='welcome';
  for(let i=0;i<5;i++){const id='aged-'+i;bodies['/api/workspace/preparation']={...prepBody,task_id:id,latest_task_id:id,selected:true};views[id]={...views[PREP],task_id:id,status:{...views[PREP].status,task_id:id}};c.location.hash='#page=welcome&preparation='+id;w.page();await settle();w.page();}
  assert.ok(!w.area().kept.includes('update|'+UPDATE) && w.area().kept.length===4,'the bounded kept states let the oldest go');
  bodies['/api/workspace/preparation']=prepBody;
  c.app.page='data';c.location.hash='#page=data&update='+UPDATE;const rebuilt=reads.length, arrivedBefore=w.area().arrived;w.page();await settle();markup=w.page();
  assert.ok(reads.length===rebuilt && w.area().task===UPDATE && w.area().inspect==='maintain_data_feature' && markup.includes('data-shown="maintain_data_feature"') && markup.includes('data-listing-key="l119"') && markup.includes('data-listing-key="l-late"') && w.area().rows===120 && w.area().arrived<=arrivedBefore,'rehydrated from the retained response and the kept preference: the rows and the inspection are back without a read and without arrivals');
  w.inspect('');
  delete prefs.preparationScene;
  views[UPDATE]=view(UPDATE,'SUCCEEDED',null);projections=[{task_id:UPDATE,task_kind:'workspace_data_update',lifecycle:'SUCCEEDED',verified_stage_count:3}]; // the update settled again (its row is the receipt's change)
  // ---- the selected Task: an earlier update by route, a typed refusal, the way back ----
  const corrected={listing_id:'lc',symbol:'QC01',state:'UPDATED',failure_code:null,raw_through:'2026-08-03',new_sessions:[],restated_sessions:1,audit_scope:'ROLLING',sentinel_disposition:'ANCHORED',attempt_count:1,updated_at:'2026-08-03T23:00:58+00:00'};
  const reverified={...corrected,listing_id:'lr',symbol:'QR01',restated_sessions:0,updated_at:'2026-08-03T23:00:59+00:00'};
  const failedUnit={...corrected,listing_id:'lf',symbol:'QF01',state:'FAILED',failure_code:'data.fetch_failed',raw_through:null,restated_sessions:0,updated_at:'2026-08-03T23:01:00+00:00'};
  bodies.byTask[EARLIER]=readback({task_id:EARLIER,task_lifecycle:'BLOCKED',selected:true,latest_task_id:UPDATE,plan_hash:'8fba411d',cycle:cycle('quality','review_pending',{failure_code:'data.truth_review_required'}),maintenance:maintenance(60,1,[corrected,reverified,failedUnit])});
  views[EARLIER]=view(EARLIER,'BLOCKED','maintain_data_feature',{stop:{code:'data.truth_review_required',stage_id:'maintain_data_feature',detail:'Listings need a data decision.',recoverable:false}});
  c.location.hash='#page=data&update='+EARLIER;w.page();await settle();markup=w.page();
  assert.ok(markup.includes('An earlier update') && markup.includes('the latest is') && markup.includes('BTN(Open the latest update:workspace-current::)') && markup.includes('BANNER(warning:<span class="coded" data-tip="data.truth_review_required">A data decision is needed</span>|A data decision is needed before this Task can continue') && markup.includes('LINK(Decide on Data issues>issues)') && !markup.includes('BTN(Continue this update'),'the stop is named once with its way on: decide first; the update continues after (N6, law 58)');
  assert.ok(markup.includes('data-shown="maintain_data_feature"') && markup.includes('<span class="tp-number">63</span><span class="tp-denom"> / 120 listings processed</span>') && markup.includes('60 with new sessions · 1 corrected · 1 re-verified · 1 failed · 57 pending'),'the stopped Task\'s own work stays readable: processed counts the failure, updated units are told apart');
  assert.ok(markup.includes('1 earlier session corrected in retained history through 2026-08-03; no new session') && markup.includes('history re-verified through 2026-08-03; nothing new to fetch, nothing corrected') && markup.includes('failed · <span class="coded" data-tip="data.fetch_failed">The fetch failed</span>') && !markup.includes('>reused<'),'a corrected unit is never a reused one; each unit says what it did, a failure in words with its code on hover');
  bodies.byTask[EARLIER]={...bodies.byTask[EARLIER],maintenance:{availability:'UNAVAILABLE',failure_code:'workspace_data_update.maintenance_manifest_unavailable',membership_revision:'gone',as_of_session:'2026-08-03'}};
  await w.refresh('data');markup=w.page();
  assert.ok(markup.includes('The maintenance units of this update are not available') && markup.includes('workspace_data_update.maintenance_manifest_unavailable') && markup.includes('nothing of another update is shown in their place') && !markup.includes('data-listing-key='),'a historical record the store no longer resolves is unavailable by name, never today\'s units');
  // a settled cycle that recorded no maintenance units of its own (its members' data came through a transition): said so, with the change set as its record, never "not admitted yet"
  bodies.byTask[EARLIER]={...bodies.byTask[EARLIER],maintenance:null,cycle:cycle('completed','completed',{change_set:changeSet})};views[EARLIER]=view(EARLIER,'SUCCEEDED',null);await w.refresh('data');w.inspect('maintain_data_feature');markup=w.page();
  assert.ok(!markup.includes('data-update-counts') && !markup.includes('has not admitted its listing units yet') && !markup.includes('data-listing-key='),'a settled cycle without units of its own draws no figure and no rows of another update (law 81)');
  w.inspect('');views[EARLIER]=view(EARLIER,'BLOCKED','maintain_data_feature',{stop:{code:'data.truth_review_required',stage_id:'maintain_data_feature',detail:'Listings need a data decision.',recoverable:false}});
  bodies.byTask[EARLIER]={...bodies.byTask[EARLIER],cycle:cycle('quality','review_pending',{failure_code:'data.truth_review_required'}),maintenance:maintenance(60,1,[corrected,reverified,failedUnit])};await w.refresh('data');
  c.location.hash='#page=data&update=no-such';bodies.byTask['no-such']=undefined;w.page();await settle();markup=w.page();
  assert.ok(markup.includes('This update Task cannot be shown') && markup.includes('workspace_data_update.task_not_found') && markup.includes('BTN(Open the current data state:workspace-current::)') && !markup.includes('data-scene'),'a missing id is a typed refusal, never a fall-back');
  await w.current();assert.equal(c.location.hash,'#page=data&update='+UPDATE,'the way back discovers and pins the latest');assert.ok(w.page().includes('2026-07-31 → 2026-08-03'));
  // continuing a stopped update posts the same plan again and stays on this page
  c.location.hash='#page=data&update='+EARLIER;w.page();await settle();
  const cont=posts.length;bodies['/api/data-update/run']={status:'ADMITTED',task_id:EARLIER,lifecycle:'RECOVERY_REQUIRED',failure_code:null};
  await w.preview('continue-update');assert.ok(modal.title==='Continue this data update' && modal.content.includes('Why it stopped=A data decision is needed'));await w.commit();
  assert.deepEqual(posts.slice(cont),[['/api/data-update/run',{update_plan_hash:'8fba411d'}]]);assert.equal(c.location.hash,'#page=data&update='+EARLIER);
  // U63 (V375): an update the provider deferred says why in the owner's words, when it resumes, and Resume is the owner's
  // own request, held until its time
  const held=' The published research inputs are unchanged, and every study keeps reading them.';
  const said='The market data provider limited the requests or did not answer, which is the provider\'s state, not the workspace\'s. Every listing already fetched is kept and the work resumes from them: send the same plan again once `retry_after_at` has passed.';
  const deferral=(at)=>({task_lifecycle:'DEFERRED',detail:said+held,retry_after_at:at,next_requests:{resume:{operation:'DATA_UPDATE_RUN',update_plan_hash:'8fba411d'}},cycle:cycle('fetch','deferred',{failure_code:'data.rate_limited',retry_after_at:at})});
  views[EARLIER]=view(EARLIER,'DEFERRED',null);bodies.byTask[EARLIER]={...bodies.byTask[EARLIER],...deferral('2030-01-01T00:00:00+00:00')};
  await w.refresh('data');markup=w.page();
  assert.ok(markup.includes('Waiting until')&&markup.includes('<span class="owner-text">'+said+' The published research inputs are unchanged')&&markup.includes('BTN(Resume this update:workspace-preview:continue-update:Not due until'),'the owner\'s words, the time, Resume held until it: '+markup.slice(markup.indexOf('Waiting until')-200,markup.indexOf('Waiting until')+900));
  bodies.byTask[EARLIER]={...bodies.byTask[EARLIER],...deferral('2026-08-03T22:00:00+00:00')};await w.refresh('data');markup=w.page();
  assert.ok(markup.includes('The wait is over')&&markup.includes('BTN(Resume this update:workspace-preview:continue-update:)'),'Resume once the time has passed');
  const resumed=posts.length;await w.preview('continue-update');assert.equal(modal.title,'Resume this update');await w.commit();
  assert.deepEqual(posts.slice(resumed),[['/api/data-update/run',{update_plan_hash:'8fba411d'}]],'the owner\'s resume request, whole, by the route the session names');
  // ---- the issues page: the case in words, one permitted response, the stale refusal ----
  c.app.page='issues';c.location.hash='#page=issues';
  const issueCase={case_token:'701f330072b97351',evidence_hash:'1587504c',failure_code:'data.unexplained_raw_move',listing_ids:['7644a0b9'],manifest_revision:'m',policy_hash:'p',run_id:'r',rediagnosis_count:0,case_kind:'k',tool_names:[],
    evidence:[{listing_id:'7644a0b9',failure_code:'data.unexplained_raw_move',action_explained:false,provider_correction_observed:false,identity_verified:true,retry_exhausted:true,largest_absolute_move:0.995,range_start:'2016-08-01',range_end:'2026-08-03',reason_codes:['UNEXPLAINED_RAW_MOVE'],unexplained_sessions:['2026-08-03'],unexplained_moves:[{close:291.66941638783715,previous_close:146.16875973634797,previous_session:'2026-07-31',session:'2026-08-03'}]}],
    options:[{option_id:'bounded_full_history_retry',option_hash:'h1',disposition:'auto',policy_args:{action:'retry_primary'}},{option_id:'recoverable_quarantine',option_hash:'h2',disposition:'auto',policy_args:{action:'quarantine_listing',recheck_after_at:'2026-08-04T23:00:45Z'}},{option_id:'escalate_data_truth_conflict',option_hash:'h3',disposition:'human_review',policy_args:{action:'escalate_for_human'}}]};
  const issue=(status,resolution=null)=>({issue_hash:'i',case:issueCase,subjects:{'7644a0b9':'QD000'},standing_quarantine:null,status,options_current:true,confirmation:'HUMAN',confirmable_option_ids:['recoverable_quarantine'],resolution,prior_decisions:[],next_action:resolution ? 'RESUME_APPROVED_WORK' : 'SELECT_PERMITTED_OPTION'});
  const continuation={task_id:EARLIER,lifecycle:'BLOCKED',failure_code:'data.truth_review_required',failure_reason:{code:'data.truth_review_required',explanation:'Listings need a data decision before the update can continue.',retryable:null,recorded_at:null},operation:'DATA_UPDATE_RUN',endpoint:'/api/data-update/run',payload:{update_plan_hash:'8fba411d'}};
  // the owner's requests (3de3f325): a confirmable response's preview while the case awaits a choice, and each continuation
  const onward={['continue:'+EARLIER]:{operation:'DATA_UPDATE_RUN',update_plan_hash:'8fba411d'}};
  const choose={'preview:701f330072b97351:recoverable_quarantine':{operation:'DATA_ISSUE_PREVIEW',data_issue_case_token:'701f330072b97351',data_issue_evidence_hash:'1587504c',data_issue_option_id:'recoverable_quarantine',data_issue_option_hash:'h2'}};
  bodies['/api/workspace/data-issues']={status:'ISSUES_PENDING',issues:[issue('AWAITING_CHOICE')],continuations:[continuation],recorded_decisions:[],continued_dispositions:[],next_cursor:null,claim_limit:'c',next_requests:{...choose,...onward}};
  await w.refresh('issues');markup=w.page();
  // F2 (law 136): the lobby -- one row an issue under its lifecycle's group: the listing and its kind, the move as a figure, its session; no card, no tabs
  assert.ok(markup.includes('data-lobby="issues"') && markup.includes('<span>Your decision is asked</span><b class="num">1</b>') && markup.includes('ROW(QD000 · Unexplained move|<span class="num">+99.54%</span>|2026-08-03)') && !markup.includes('STAT(') && !markup.includes('TABS('),'the lobby lists the case as one row: '+markup);
  c.location.hash='#page=issues&issue=701f330072b97351';markup=w.page(); // the row opens the issue's own page
  // N6: the case read as an issue -- the move as figures, the owner's checks in words, the permitted responses in words, the owner's own actions apart
  assert.ok(markup.includes('H1:QD000 · Unexplained move|') && markup.includes('BADGE(review_pending:Your decision is asked)') && markup.includes('FACT(Move session=2026-08-03)') && markup.includes('ID(701f330072b97351)') && !markup.includes('Properties') && markup.includes('STAT(Close on 2026-07-31:146.17:)') && markup.includes('STAT(Close on 2026-08-03:291.67:)') && markup.includes('STAT(Move:+99.54%:)') && markup.includes('no corporate action explains it') && markup.includes('listing identity verified') && !markup.replace(/<template>[\s\S]*?<\/template>/g,'').includes('UNEXPLAINED_RAW_MOVE'),'the case is said in the owner\'s evidence: the move as figures, no code on the face');
  assert.ok(markup.includes('Quarantine the listing until it requalifies') && markup.includes('recheck after') && !markup.includes('value="bounded_full_history_retry"') && markup.includes('The data owner\'s own actions') && markup.includes('Retry the primary source with a bounded full history') && markup.includes('BTN(Preview decision:workspace-issue:701f330072b97351:Choose a permitted response)'),'only the confirmable responses are choices; the owner\'s own actions are listed apart');
  assert.ok(!markup.includes('BTN(Continue this Task'),'no continuation is offered before the case is decided (N6)');
  w.changed('701f330072b97351','recoverable_quarantine');
  assert.ok(w.page().includes('value="recoverable_quarantine" data-workspace-option="701f330072b97351" checked') && w.page().includes('BTN(Preview decision:workspace-issue:701f330072b97351:)'),'a chosen response is kept on redraw');
  const decide=posts.length;await w.preview('issue','701f330072b97351');
  assert.ok(modal.title==='Confirm this data decision' && modal.content.includes('Response=Quarantine the listing until it requalifies') && modal.content.includes('Applied now=No: recorded as your decision'),'the decision preview');
  // the stale confirmation: the owner refuses by name; nothing is admitted and the page says why
  failed='/api/workspace/data-issues/confirm';failure='feature_input.option_changed';await w.commit();failed=null;
  assert.deepEqual(posts.slice(decide).map(v=>v[0]),['/api/workspace/data-issues/preview','/api/workspace/data-issues/confirm'],'the refused confirmation admits nothing more');
  markup=w.page();assert.ok(markup.includes('feature_input.option_changed') && markup.includes('The option changed since it was shown; preview the decision again.'),'the refusal is explained, once, without a repeated admission');
  // the recorded decision, then the continuation from this page; the page re-reads once the Task settles
  bodies['/api/workspace/data-issues']={...bodies['/api/workspace/data-issues'],issues:[issue('CONFIRMED_PENDING_REVALIDATION',{receipt:{receipt_hash:'326dfabe18',policy_decision:{option_id:'recoverable_quarantine'}},effect:{}})],next_requests:onward}; // decided: the owner offers only the continuation
  await w.preview('issue','701f330072b97351');await w.commit();markup=w.page();
  assert.ok(markup.includes('Your decision is recorded. It is applied when the data update continues') && markup.includes('BADGE(planned:Decided · applied when the update continues)') && markup.includes('Your decision: <strong>Quarantine the listing until it requalifies</strong>') && !markup.includes('data-workspace-option=') && markup.includes('BANNER(neutral:Your decision is recorded|Continue the update: the owner checks the evidence again and applies your decision') && markup.includes('BTN(Continue this update:workspace-continue:0:)'),'the recorded decision replaces the form, and the continuation reads as the Data page reads it: one title, one sentence, one verb (N6, law 58)');
  const goOn=posts.length;await w.preview('continue','0');assert.ok(modal.title==='Continue the approved data task' && modal.content.includes('Recorded as=') && modal.content.includes('Why it stopped=Listings need a data decision'));
  bodies['/api/data-update/run']={status:'ADMITTED',task_id:EARLIER,lifecycle:'RECOVERY_REQUIRED',failure_code:null};await w.commit();
  assert.deepEqual(posts.slice(goOn),[['/api/data-update/run',{update_plan_hash:'8fba411d'}]]);assert.ok(w.page().includes('The update continues as Task 23ab76b8; the Data page shows its work.'));
  projections=[{task_id:EARLIER,task_kind:'workspace_data_update',lifecycle:'RUNNING',verified_stage_count:1}];const quiet=reads.length;w.observe();await settle();assert.equal(reads.length,quiet,'nothing is read while the continued Task moves');
  bodies['/api/workspace/data-issues']={status:'NO_PENDING_CASE_CATALOG',issues:[],continuations:[],recorded_decisions:[{case_token:'701f330072b97351',issue_hash:'i',listing_ids:['7644a0b9'],failure_code:'data.unexplained_raw_move',resolution:{receipt:{policy_decision:{option_id:'recoverable_quarantine'}}},claim_limit:'H'}],continued_dispositions:[],next_cursor:null,claim_limit:'c'};
  projections=[{task_id:EARLIER,task_kind:'workspace_data_update',lifecycle:'SUCCEEDED',verified_stage_count:3}];w.observe();await settle();markup=w.page();
  assert.ok(reads.slice(quiet).includes('/api/workspace/data-issues') && markup.includes('H1:7644a0b9 · Unexplained move|') && markup.includes('BADGE(recorded:Recorded)') && markup.includes('Your decision: <strong>Quarantine the listing until it requalifies</strong>'),'once the continued Task settles the page reads the owner again: the decision is history, its page says so');
  c.location.hash='#page=issues';markup=w.page();assert.ok(markup.includes('<span>Recorded</span><b class="num">1</b>') && markup.includes('ROW(7644a0b9 · Unexplained move|Quarantine the listing until it requalifies)'),'the recorded decisions are the lobby\'s folded group, one row each');
  // the returning path: a continuation confirmed here settles while another page is read; entering this page again (the router's page change) reads the owner once, and the admission notice says what the Task became
  bodies['/api/workspace/data-issues']={status:'ISSUES_PENDING',issues:[issue('CONFIRMED_PENDING_REVALIDATION',{receipt:{receipt_hash:'326dfabe18',policy_decision:{option_id:'recoverable_quarantine'}},effect:{}})],continuations:[continuation],recorded_decisions:[],continued_dispositions:[],next_cursor:null,claim_limit:'c',next_requests:onward};
  projections=[{task_id:EARLIER,task_kind:'workspace_data_update',lifecycle:'BLOCKED',verified_stage_count:1}];
  await w.refresh('issues');await w.preview('continue','0');await w.commit();
  projections=[{task_id:EARLIER,task_kind:'workspace_data_update',lifecycle:'RUNNING',verified_stage_count:1}];
  c.app.page='storage';c.location.hash='#page=storage';w.page();w.afterPaint(true);await settle();
  bodies['/api/workspace/data-issues']={status:'NO_PENDING_CASE_CATALOG',issues:[],continuations:[],recorded_decisions:[{case_token:'701f330072b97351',issue_hash:'i',listing_ids:['7644a0b9'],failure_code:'data.unexplained_raw_move',resolution:{receipt:{policy_decision:{option_id:'recoverable_quarantine'}}},claim_limit:'H'}],continued_dispositions:[],next_cursor:null,claim_limit:'c'};
  projections=[{task_id:EARLIER,task_kind:'workspace_data_update',lifecycle:'SUCCEEDED',verified_stage_count:3}];
  const back=reads.length;c.app.page='issues';c.location.hash='#page=issues';markup=w.page();w.afterPaint(true);await settle();markup=w.page();
  assert.ok(reads.slice(back).filter(r=>r==='/api/workspace/data-issues').length===1 && markup.includes('Last operation · Task 23ab76b8 completed; the page below was read after it settled.') && markup.includes('<span>Recorded</span><b class="num">1</b>') && !markup.includes('Your decision is recorded;'),'entering the page again after the continued Task settled reads the owner once and says what the Task became');
  w.afterPaint(false);w.page();await settle();assert.equal(reads.length,back+1,'a repaint of the same page reads nothing more');
  const again=reads.length;w.afterPaint(true);await settle();assert.deepEqual(reads.slice(again),['/api/workspace/data-issues'],'without a watched Task, a page entered again is still read once, quietly');
  // One unreadable case remains visible beside its healthy sibling; when every case is unreadable,
  // the issue page names those records and the Workspace/Storage routes instead of claiming empty.
  const unreadableCase={status:'REFUSED',case_token:'broken-case',failure_code:'feature_input.case_manifest_unavailable',detail:'This data issue case names a manifest that cannot be read.',next_requests:{issues:{operation:'DATA_ISSUES'},workspace:{operation:'WORKSPACE_SHOW'},backups:{operation:'WORKSPACE_BACKUPS'}}};
  const originalRefusal=c.refusal;
  c.refusal=(item,_tone,o)=>`REFUSAL(${item.failure_code}|${item.detail}|${String(o.more||'')}|${String(o.next||'')})`;
  bodies['/api/workspace/data-issues']={status:'PARTIAL',issues:[issue('AWAITING_CHOICE')],refused_cases:[unreadableCase],continuations:[],recorded_decisions:[],continued_dispositions:[],next_cursor:null,claim_limit:'c',next_requests:choose};
  await w.refresh('issues');markup=w.page();
  assert.ok(markup.includes('ROW(QD000 · Unexplained move')&&markup.includes('feature_input.case_manifest_unavailable')&&markup.includes('This data issue case names a manifest that cannot be read.')&&markup.includes('LINK(Workspace>overview)')&&markup.includes('LINK(Storage and backups>storage)'),'a partial Data Issues answer preserves the readable case and routes beside the refused one: '+markup.slice(markup.indexOf('Data issues'),markup.indexOf('Data issues')+900));
  bodies['/api/workspace/data-issues']={status:'PARTIAL',issues:[],refused_cases:[unreadableCase],continuations:[],recorded_decisions:[],continued_dispositions:[],next_cursor:null,claim_limit:'c'};
  await w.refresh('issues');markup=w.page();
  assert.ok(markup.includes('feature_input.case_manifest_unavailable')&&markup.includes('LINK(Workspace>overview)')&&markup.includes('LINK(Storage and backups>storage)')&&!markup.includes('EMPTY(No open cases)'),'all refused Data issue rows remain named with routes; the page does not infer an empty collection: '+markup.slice(markup.indexOf('Data issues'),markup.indexOf('Data issues')+800));
  c.refusal=originalRefusal;
  // ---- the inputs page: versions, roles, the draft\'s version, publication preview and reuse ----
  c.app.page='inputs';c.location.hash='#page=inputs';await w.refresh('inputs');markup=w.page();
  assert.ok(markup.includes('ROW(2016-08-01 — 2026-07-31|<span class="nowrap">configured default</span> · <span class="nowrap">registered at preparation</span>|BADGE(planned:Selected in the draft))') && markup.includes('ROW(2016-08-01 — 2026-08-03|<span class="nowrap">published</span> · <span class="nowrap">latest cutoff</span>|BTN(Use for a new Factor draft:workspace-input:'),'every version with its sessions and role (its reference only where two versions share their sessions; present is the usual state); the draft\'s version is marked as a state, not offered as an action');
  await w.preview('capture');
  assert.ok(modal.title==='Publish a separate immutable input' && modal.content.includes('Features cutoff: 2026-07-31 → 2026-08-03') && modal.content.includes('The prepared data moved: the new version seals the current Features cutoff'),'the publication preview names what changed');
  const codeOnly={...bodies['/api/research-inputs/plan'],after:{...bodies['/api/research-inputs/plan'].before,materializer_hash:'mat2'}};bodies['/api/research-inputs/plan']=codeOnly;
  await w.preview('capture');assert.ok(modal.content.includes('Input materializer (source code)') && modal.content.includes('rotated; the market data itself is unchanged'),'a materializer rotation is never described as new market data');
  bodies['/api/research-inputs/plan']={status:'REUSED_EXACT',task_id:null,input_id:'factor-development',binding_hash:NEW,next_action:'SELECT_INPUT_VERSION'};
  const reuse=posts.length;await w.preview('capture');assert.equal(posts.length,reuse+1);assert.ok(w.page().includes('Exact version already published: 87082400a347. Nothing was re-sealed; select it below.'),'exact reuse of a publication is said so');
  bodies['/api/research-inputs/plan']=codeOnly;await w.preview('capture');await w.commit();
  assert.ok(posts.at(-1)[0]==='/api/research-inputs/confirm' && w.page().includes('Publication Task <span class="mono">4be453ef</span> · Admitted') && w.page().includes('Admitted as Task 4be453ef'),'the admitted publication is named on this page');
  projections=[{task_id:CAPTURE,task_kind:'research_input_capture',lifecycle:'SUCCEEDED',verified_stage_count:1}];const pub=reads.length;w.observe();await settle();
  assert.ok(reads.slice(pub).includes('/api/research-inputs/readback?task_id='+CAPTURE) && reads.slice(pub).includes('/api/research-inputs') && w.page().includes('Published <span class="mono">87082400a347</span> · sessions through 2026-08-03'),'once the publication settles its receipt and the versions are read once');
  await w.selectInput(JSON.stringify(['factor-development',NEW]));assert.deepEqual(drafts,[JSON.stringify(['factor-development',NEW])],'selecting edits a draft only');
  // ---- the storage page: the summary in words, the roots in order, retained versions, cleanup ----
  c.app.page='storage';c.location.hash='#page=storage';await w.refresh('storage');markup=w.page();
  assert.ok(markup.includes('STAT(Managed on disk:1.23 <span class="num-unit">GiB</span>:physical bytes · each object once)') && markup.includes('STAT(Referenced:2.12 <span class="num-unit">GiB</span>:910 <span class="num-unit">MiB</span> shared between versions)') && markup.includes('Within the managed budget · Cleanup only ever removes the exact targets'),'physical and logical bytes are told apart');
  // the roots in the owner's order; what a root holds is its (i) since the walk's R4 (2026-09-30), its bytes after it
  const rootRow=(name)=>markup.indexOf('ROW('+name);
  assert.ok(rootRow('Working database')>=0 && rootRow('Working database')<rootRow('Immutable input versions') && rootRow('Immutable input versions')<rootRow('Feature and outcome artifacts'),'the roots in the order the owner lists them');
  const working=markup.slice(rootRow('Working database'),rootRow('Immutable input versions'));
  assert.ok(working.includes('the DuckDB store the update owner maintains in place; not research input') && working.includes('275 <span class="num-unit">MiB</span>') && !working.includes('text-cell'),'a root says what it holds in its (i), then its bytes: '+working.slice(0,400));
  assert.ok(markup.includes('ROW(2016-08-01 — 2026-07-31|<p>Kept because</p><p>configured default · previous version (rollback)</p>|371 <span class="num-unit">MiB</span>|BTN(Retain input:workspace-pin:'+OLD+':)') && markup.includes('<p>Kept because</p><p>latest published version</p>|404 <span class="num-unit">MiB</span>'),'every retained version, named by its sessions as Research inputs names it, says why it is kept (present is the usual state, not a fact to repeat)');
  // V634: the owner's complete root set, on both readbacks and in both languages.
  const storageBefore=bodies['/api/workspace/storage'], originalWords=c.t, words=library.words(root);
  const retentionRoots=library.codes().retention_roots || [...new Set([...fs.readFileSync(path.join(root,'../../../../../../control/product_host/storage/input_references.py'),'utf8').matchAll(/roots\[[^\n]+\]\.add\("([^"\n]+)"\)/g)].map(m=>m[1]))];
  assert.ok(retentionRoots.length,'the owner supplied retention roots');
  const reasonText={};
  for(const lang of ['en','zh']) {
    words.I18N.set(lang); c.t=words.t;
    for(const reason of retentionRoots) {
      bodies['/api/workspace/storage']={...storageBefore,inputs:[{...storageBefore.inputs[0],roots:[reason]}]};
      c.app.page='storage';c.location.hash='#page=storage';await w.refresh('storage');
      const store=w.page(), row=store.slice(store.indexOf('ROW(2016-08-01'),store.indexOf('BTN(',store.indexOf('ROW(2016-08-01')));
      const retainedReason=(factBodies.get(words.t('Kept because')) || '').replace(/<[^>]+>/g,'');
      assert.ok(retainedReason && !retainedReason.includes(reason),lang+' retention Facts name the root without leaking its code: '+reason);
      assert.ok(!row.includes(reason),lang+' retention sentence leaked '+reason);
      assert.ok(store.includes(reason),'exact inventory retains '+reason);
      c.app.page='inputs';c.location.hash='#page=inputs&version='+OLD;await w.refresh('inputs');
      const version=w.page();
      assert.ok(!version.includes(reason),lang+' input version leaked '+reason);
      if(lang==='en') reasonText[reason]=retainedReason;
      else {
        assert.notEqual(retainedReason,reasonText[reason],'the retained reason has an actual Chinese reading: '+reason);
        assert.ok(!words.I18N.untranslated().includes(reasonText[reason]),'missing retention phrase '+reasonText[reason]);
      }
      if(reason==='MODEL_TRAINING_SOURCE') assert.ok(store.includes(lang==='zh'?'为模型训练保留':'retained for model training'));
    }
  }
  c.t=originalWords;c.app.page='storage';c.location.hash='#page=storage';bodies['/api/workspace/storage']=storageBefore;await w.refresh('storage');
  const clean=posts.length;await w.preview('cleanup');assert.equal(posts.length,clean+1);assert.ok(w.page().includes('No eligible files to clean. Nothing was removed.') && w.page().includes('Reclaimable=0 <span class="num-unit">B</span>') && w.page().includes('Retained=1.23 <span class="num-unit">GiB</span>'),'a cleanup with nothing eligible removes nothing and says so');
  bodies['/api/workspace/storage/plan']={...bodies['/api/workspace/storage/plan'],targets:{'research-inputs/aaa/source/data.parquet':'d1','research-inputs/parquet/d1.parquet':'d1'},bindings:['aaa'],reclaimable_bytes:1024*1024*7};
  await w.preview('cleanup');assert.ok(modal.title==='Confirm displayed cleanup' && modal.content.includes('Files selected for cleanup=2') && modal.content.includes('Reclaimable=7.00 <span class="num-unit">MiB</span>') && modal.content.includes('research-inputs/aaa/source/data.parquet'),'the exact targets are named before any confirmation');
  const wipe=posts.length;await w.commit();assert.deepEqual(posts.slice(wipe),[['/api/workspace/storage/confirm',{storage_plan_hash:'77f93a79b4e3bd10'}]]);
  assert.ok(w.page().includes('Cleanup completed: 1 file path released. Every retained version and the working database are as they were.'),'the cleanup says what it released, in the owner\'s count');
  await w.preview('pin',OLD);assert.ok(modal.title==='Confirm input retention change' && modal.content.includes('Pin this version'));const pin=posts.length;await w.commit();assert.deepEqual(posts.slice(pin),[['/api/workspace/storage/pin',{input_binding_hash:OLD,input_pinned:true}]]);
  assert.ok(w.page().includes('Version 65451da2923a is retained until you unpin it.'),'a pin is said as retention until lifted');
  // U44 (V208): a plan whose only change links the retained model copy to the machine's store is confirmable,
  // names the copy and its bytes before the confirmation, and the confirmation says it was linked
  bodies['/api/workspace/storage/plan']={status:'CONFIRMATION_REQUIRED',plan_hash:'relinkplan',targets:{},bindings:[],retained_bytes:1024,reclaimable_bytes:4096,relinks:{'evidence-cro-authority/semantic-model':4096},limitations:['A_RETAINED_RETRIEVAL_MODEL_COPY_LINKS_TO_THE_MODEL_STORE_IT_MATCHES']};
  await w.preview('cleanup');
  assert.ok(modal.title==='Confirm displayed cleanup' && modal.content.includes('Linked to the model store') && modal.content.includes('evidence-cro-authority/semantic-model') && !w.page().includes('No eligible files to clean'),'a relink alone is a plan to confirm: '+modal.content.slice(0,400));
  assert.ok(modal.content.includes('becomes links to the machine\'s model store it matches'),'the plan\'s limitation is said in words (U40, U44)');
  bodies['/api/workspace/storage/confirm']={status:'COMPLETED',deleted_paths:0,plan_hash:'relinkplan',relinked:{'evidence-cro-authority/semantic-model':4096}};
  const relink=posts.length;await w.commit();assert.deepEqual(posts.slice(relink),[['/api/workspace/storage/confirm',{storage_plan_hash:'relinkplan'}]]);
  assert.ok(w.page().includes("the retrieval model's copy linked to the model store (4.00"),'the confirmation says the copy was linked, in the owner\'s bytes');
  // R8: a version the input owner cannot read is listed as such, with its way to Storage; it cannot be chosen
  const DEAD='d'.repeat(64);
  bodies['/api/research-inputs']={...bodies['/api/research-inputs'],inputs:[{...bodies['/api/research-inputs'].inputs[0],versions:[...bodies['/api/research-inputs'].inputs[0].versions,{binding_hash:DEAD,configured_default:false,publication_hash:'p',available:false,unreadable:'research_input.manifest_missing',detail:'This version\'s prepared input cannot be read in this workspace.',next_requests:{storage:{operation:'STORAGE_READBACK'}}}]}]};
  c.app.page='inputs';c.location.hash='#page=inputs';await w.refresh('inputs');markup=w.page();
  const dead=markup.slice(markup.indexOf('dddddddddddd')-80,markup.indexOf('dddddddddddd')+700);
  assert.ok(dead.includes('Input version') && dead.includes('Its manifest is missing') && dead.includes('LINK(Storage & retention>storage'),'the unreadable version says so, with its way to Storage: '+dead);
  assert.ok(dead.includes(':This version cannot be read in this workspace)') && !dead.includes('unavailable: source files missing'),'its use is held with the English source of its reason (CT7), never the missing-files words');
  finish();
})().catch(e=>{console.error(e);process.exitCode=1;});
