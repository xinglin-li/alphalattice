// Follow cancellation at the actual view mutation, through the real readers: the composed
// `data.js`, `live-activity.js`, `live-tasks.js` and `live-study.js` modules run as written;
// only transport (`fetch`), routing state (a fake `location`/`history`, a `navigate` that does
// what the router's does to the route) and presentation helpers are mocked. Owner responses
// are gated so a pin and a navigation can land while the final reader is still waiting, and
// nothing in the probe honours `wanted()` on the readers' behalf. The installed Portfolio
// replay is followed through its real owners (the Portfolio index, the REPORT of that exact
// result, History's INSTALLED_RESULT entry), including after the feed loses its rows.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const library=require('./workbench_library.cjs'); // the library's constants and the scripts' parameters, from the source (Q2)
const finish=library.guard('workbench_follow_readers');
const root=process.argv[2];
const runShapes=(root)=>{const src=require('node:fs').readFileSync(require('node:path').join(root,'components.js'),'utf8');const a=src.indexOf('/* ---- run shapes (round 72)');const b=src.indexOf('/* ---- end of run shapes ---- */');return src.slice(a,b)+';globalThis.stepList=stepList;globalThis.runLog=runLog;globalThis.logLine=logLine;globalThis.observationLine=observationLine;globalThis.logMove=logMove;globalThis.logHeld=logHeld;globalThis.logRetain=logRetain;';};
const toasts=[],dialogs=[],renders=[],requests=[],entries=[],urls=[],posts=[],dialogActions=[],dialogBodies=[];
const gates={};const gate=(key)=>{let release;const promise=new Promise(r=>{release=r;});gates[key]={promise,release,entered:false};return gates[key];};
const entered=async(held,label)=>{for(let i=0;i<50 && !held.entered;i++)await settled();assert.ok(held.entered,'the held request was entered: '+label);};
const settled=()=>new Promise(r=>setImmediate(r));
const hash={value:''};
const H=(n)=>String(n).repeat(64).slice(0,64);
const TASKS={
  'task-a':{task_id:'task-a',task_kind:'research_experiment',lifecycle:'RUNNING',goal_summary:'a',current_stage:'x',verified_stage_count:0,total_stage_count:2,cancel_available:true,cancel_pending:false},
  'task-b':{task_id:'task-b',task_kind:'research_experiment',lifecycle:'RUNNING',goal_summary:'b',current_stage:'x',verified_stage_count:0,total_stage_count:2,cancel_available:true,cancel_pending:false},
  'task-p':{task_id:'task-p',task_kind:'portfolio_public_development_replay',lifecycle:'RUNNING',goal_summary:'p',current_stage:'x',verified_stage_count:0,total_stage_count:4,cancel_available:true,cancel_pending:false},
  'task-q':{task_id:'task-q',task_kind:'research_experiment',lifecycle:'RUNNING',goal_summary:'q',current_stage:'x',verified_stage_count:0,total_stage_count:2,cancel_available:true,cancel_pending:false},
  'task-r':{task_id:'task-r',task_kind:'portfolio_public_development_replay',lifecycle:'RUNNING',goal_summary:'r',current_stage:'x',verified_stage_count:0,total_stage_count:1,cancel_available:true,cancel_pending:false},
  'task-s':{task_id:'task-s',task_kind:'portfolio_public_development_replay',lifecycle:'RUNNING',goal_summary:'s',current_stage:'x',verified_stage_count:0,total_stage_count:1,cancel_available:true,cancel_pending:false},
};
const readback=(task)=>({status:'EXPERIMENT_PUBLISHED',task_id:task,program:{kind:task==='task-q'?'portfolio.policy-development':'factor.screening-development'},portfolio_source:task==='task-q'?{}:null,research_input_id:'factor-development',input_binding_hash:H('b'),result:{},document:{}});
let results=[];
let history=[];
// Recovery views as the Host serves them, for the Task Center's standing line: the liveness
// source is named, and a runner heartbeat beyond the durable timestamp reads as observed.
const READ_AT=Date.parse('2026-09-15T10:01:00Z'),before=(s)=>new Date(READ_AT-s*1000).toISOString(); // the owner's instants agree with its ages at the read
const liveness=(status,telemetry,age,sequence,source=telemetry==='OPERATIONAL'?'OPERATIONAL_SIGNAL':'DURABLE_TIMESTAMP',operationalAge=telemetry==='OPERATIONAL'?age:null)=>({status,telemetry,last_heartbeat_at:before(age),last_heartbeat_source:source,operational_at:telemetry==='OPERATIONAL'?before(operationalAge):null,operational_age_seconds:operationalAge,signal_sequence:sequence,durable_at:'2026-09-15T10:00:00Z',age_seconds:age,note:'n'});
const recoveryView=(task,live)=>({kind:'TaskRecoveryView',guardian_mode:'G0_READ_ONLY',active_remediation_attempt_count:0,task_id:task,task_kind:'portfolio_public_development_replay',lifecycle:'RUNNING',operation_running:true,cancellation:'NOT_REQUESTED',stop:null,worker_failure:null,liveness:live,execution_binding_hash:H('c'),verified_stage_count:0,total_stage_count:1,stages:[{stage_id:'execute_and_publish_declared_path',lifecycle:'IN_PROGRESS',evidence:[],evidence_count:0}],artifact_refs:[],actions:[{action:'CANCEL',operation:'CANCEL',admits:false,available:true,requires_confirmation:true,scope:'s',expected_effect:'e',reason:'r'},{action:'RECOVER',operation:'RECOVER',admits:true,available:false,requires_confirmation:true,scope:'s',expected_effect:'e',reason:'The Task is RUNNING'},{action:'REPLAN',operation:'PLAN',admits:false,available:true,requires_confirmation:true,scope:'s',expected_effect:'e',reason:'r'}],health:{task_id:task,status:live.status==='OBSERVED'?'HEALTHY':'LIVENESS_STALE',observed_at:'2026-09-15T10:01:00Z',last_heartbeat_at:live.last_heartbeat_at,heartbeat_age_seconds:live.age_seconds,signal_hash:H('d')},incidents:[],model_facts:'NOT_OBSERVED',guardian:null,guardian_availability:'NOT_PUBLISHED',task_record_hash:H('e'),observed_at:'2026-09-15T10:01:00Z',view_hash:H('f'),status:{...TASKS[task],task_record_hash:H('e'),worker_failure:null}});
const recoveryViews={};
const responses={
  '/api/research-history':()=>({entries:history,next_cursor:null}),
  '/api/experiments':()=>({experiments:['task-a','task-b'].map(task_id=>({task_id,kind:'factor.screening-development'}))}),
  '/api/experiments/curation':()=>({choices:[],receipt_hash:'r',limitations:[]}),
  '/api/experiments/readback':(q)=>readback(q.get('task_id')),
  '/api/workbench/portfolio':(q)=>({subject:{task_id:q.get('task_id'),session:'2024-08-12',input_hash:H('b')},series:[],metrics:{},holdings:[],sessions:['2024-08-12']}),
  '/api/tasks':()=>({tasks:Object.values(TASKS)}),
  '/api/status':(q)=>TASKS[q.get('task_id')],
  '/api/tasks/recovery':(q)=>recoveryViews[q.get('task_id')],
  '/api/results':(q)=>{const task=q.get('task_id'),hit=results.find(r=>r.task_id===task || (r.reused_by || []).includes(task));return {results:hit?[{...hit,task_id:task}]:[]};}, // exact selected Task metadata, including a reused result
  '/api/report':(q)=>{const hit=results.find(r=>r.result_hash===q.get('result_hash'));if(!hit)return {status:'REFUSED',failure_code:'local_web.result_unknown'};return {result_hash:hit.result_hash,report_hash:H('r'),originating_task_id:hit.task_id,used_by_task_ids:[hit.task_id,...(hit.reused_by || [])]};}, // U10: every Task that used the result, the first first
};
async function fetchMock(url,init={}) {
  const [route,query]=url.split('?');requests.push(route);urls.push(url);if(init.method==='POST')posts.push([route,JSON.parse(init.body)]);
  const held=gates[route];if(held){delete gates[route];held.entered=true;await held.promise;}
  const body=responses[route]?.(new URLSearchParams(query||''));
  if(body===undefined)throw Error('probe: no response for '+route);
  return {ok:true,text:async()=>JSON.stringify(body)};
}
const el=()=>({classList:{add(){},remove(){},toggle(){}},open:false,close(){this.open=false;},show(){this.open=true;},showModal(){this.open=true;},contains:()=>false,innerHTML:'',scrollTop:0,querySelectorAll:()=>[],querySelector:()=>null,matches:()=>false,addEventListener(){},focus(){}});
// The inspector's Task body as the paint sees it (Window.openInspector below writes it): it counts
// the times its markup was actually written, so an unchanged body is proven untouched.
const region=()=>{const r=el();let markup='',writes=0;Object.defineProperty(r,'innerHTML',{get:()=>markup,set:(v)=>{markup=v;writes+=1;}});Object.defineProperty(r,'childElementCount',{get:()=>markup?1:0});r.writes=()=>writes;return r;};
const body=region(),inspector={mode:null};
const c={console,URLSearchParams,Date,Number,Math,Set,Map,Object,Array,String,Promise,JSON,Boolean,Error,Symbol,
  setTimeout:(fn,ms)=>0,clearTimeout(){},queueMicrotask:(fn)=>Promise.resolve().then(fn),setImmediate,
  window:{ALPHA_PRODUCT:true,FIXTURES:null,addEventListener(){},confirm:()=>true},
  document:{hidden:false,addEventListener(){},querySelector:()=>null,activeElement:null,body:{dataset:{page:''},classList:{add(){},remove(){},toggle(){}}}},
  location:{get hash(){return hash.value;},set hash(v){hash.value=v;}},
  history:{replaceState:(_a,_b,url)=>{hash.value=url;},pushState:(_a,_b,url)=>{hash.value=url;}},
  fetch:fetchMock,matchMedia:()=>({matches:false}),scrollTo(){},
  app:{page:'lab',book:null,session:null,input:null,yaml:'',mode:'yaml',tasks:[],inputs:[],compareOther:'',labPane:'declaration'},
  hashParams:()=>new URLSearchParams(hash.value.slice(1)),
  replaceHash:(update)=>{const q=new URLSearchParams(hash.value.slice(1));for(const [k,v] of Object.entries(update))q.set(k,v);hash.value='#'+q;},
  // The one history entry an explicit open makes (the router's `objectEntry`): recorded here.
  objectEntry:(key)=>{entries.push(key);return true;},patchMain:()=>renders.push('patch:'+c.app.page),preserveSurface:()=>({}),restoreSurface(){},
  navigate:(page)=>{const q=new URLSearchParams({page,book:c.app.book||'',session:c.app.session||'',state:'normal'});hash.value='#'+q;c.app.page=page;renders.push(page);},
  render:()=>renders.push(c.app.page),
  $:(sel)=>sel==='#dialog'?{open:false}:el(),$$:()=>[],
  html:(s,...v)=>s.reduce((a,p,i)=>a+p+(Array.isArray(v[i])?v[i].join(''):v[i]??''),''),t:(s,vars)=>String(s).replace(/\{(\w+)\}/g,(_,k)=>String(vars?.[k]??'')),
  detailSplit:(content,modes)=>'<detail-host '+modes+'>'+String(content)+'</detail-host>', // law 149: the Tasks list hosts its record
  btn:(label,action,value)=>`<${action}:${value}>`,groupHead:(label,count)=>'GROUP('+label+':'+count+')',listFoot:()=>'',displayOptions:()=>'',displayState:(name,spec)=>({group:'none',order:'',props:{}}),objectRow:(s={},x)=>'ROW('+[s.name,s.why,...(((x&&x.props)||[]).filter(Boolean).map(p=>Array.isArray(p)?p[1]:p)),x&&x.time,x&&x.actions].filter(Boolean).join('|')+')'+((x&&x.under)||''),statusDot:(s,l)=>'['+s+(l?':'+l:'')+']',btnAttrs:()=>'',typedBtn:library.stubs.empty,notRead:(title,error,words='',action='')=>`<div class="warning">${title}:${error}${words?' '+words:''}${action}</div>`,refusal:(b,tone,o={})=>'REFUSED('+(b.code||'')+': '+(b.reason||'')+')'+(o.next ? ' NEXT('+o.next+')' : ''),causeLine:()=>'',explainCode:()=>'',CODE_WORDS:{},coded:(code)=>code,noteLine:(title,body='',tone='',action='')=>`<p class="note-line">${title}${body?' · '+body:''}${action||''}</p>`,hint:(term)=>term,factsRef:(title,body)=>`<p>${title}</p>${body}`,codeRef:(title,text)=>'<p>'+title+'</p><template>'+(typeof text==='string' ? text : JSON.stringify(text))+'</template>',refCell:(uri)=>'<span>'+uri+'</span>',hashCell:(h)=>'<span>'+(h||'—')+'</span>',readingPane:(title,kind,body,close)=>'<aside>'+title+body+'</aside>',sourceRows:(rows)=>rows.map((r)=>[r.label,r.value]),badge:(tone,label)=>`[${tone}:${label}]`,stateLine:(x,o={})=>`[${(typeof x==='string'?x:(x?.lifecycle??x?.state??x?.status??''))}:${o.word??''}]`,skeleton:(shape='rows')=>'SKELETON('+shape+')',emptyState:(s,a='')=>'EMPTY('+s+')'+(a||''),banner:(a,b)=>`(${a}|${b})`,icon:()=>'',kv:()=>'',link:(label)=>String(label),objectHead:library.stubs.empty,
  openDialog:(eyebrow,heading,content,actions)=>{dialogs.push(String(heading));dialogActions.push(String(actions ?? ''));dialogBodies.push(String(content ?? ''));},closeDialog(){},notify:(m,vars)=>toasts.push(String(m).replace(/\{(\w+)\}/g,(_,k)=>String(vars?.[k]??''))),
  Navigation:{close(){}},Inspect:{closeLens(){},selectObservation(){}},Tasks:{setTop(){},map:{}},Controls:{sync(){}},
  Window:{renderSide(){},render(){},afterRender(){},syncFrame(){},syncRoutes(){},onNavigate(){},openInspector({body:m,by}){inspector.mode='task';inspector.by=by?by.join(':'):'';body.innerHTML=String(m);return true;},openedBy(a,v){return inspector.mode!==null&&inspector.by===a+':'+v;},setInspectorBody(m){body.innerHTML=String(m);},closeInspector(){inspector.mode=null;inspector.by='';return true;},inspectorOpen(){return inspector.mode!==null;},inspectorMode(){return inspector.mode;},inspectorTitle(){return '';},docked(){return true;},sideOpen(){return true;},closeSideOverlay(){},toggleSide(){},toggleGroup(){}},
  LiveViews:{nameOf:(x)=>({kind:'',subject:'',name:x?.goal_summary||x?.name||'',truth:x?.goal_summary||''}),truthOf:(x)=>x?.goal_summary||'',taskDock(){},taskSuccessor:()=>'',taskStopFacts:()=>'',taskAttentionFacts:()=>'',savedObjectLink:(label,entry)=>label+':'+entry,existing:(id)=>renders.push('existing:'+id),studyFacts:()=>({summary:'',reference:'',origin:'',interval:''}),inputState:()=>({}),cutoffText:()=>'',declaredFromReadback:()=>({})},
  LiveWorkspace:{deferWords:(d)=>'DEFER('+d+')'},LiveReview:{open:()=>{throw Error('probe: review open not expected');},taskFinished(){},pages:new Set(['evidence']),routeContext:()=>({})},
  LiveResearch:{ready(){},dirty:()=>false,inspectShared(){},routeContext:()=>({})},
  download(){},
  codeWords:(code)=>String(code ?? '—'),short:(v,n=8)=>String(v||'').slice(0,n),mono:(v,n=12)=>v?'<span class="mono">'+String(v).slice(0,n)+'</span>':'—',actorWords:(caller)=>String(caller ?? '—'),count:(n)=>String(n),when:(v)=>String(v ?? '—'),dayOf:(v)=>String(v || '').slice(0,10),pluralText:(n,one,many,args={})=>String(Number(n)===1?one:many).replace(/\{(\w+)\}/g,(m,k)=>args&&args[k]!==undefined?args[k]:m),countText:(n,one,many)=>String(n===1?one:many).replace('{n}',String(n)),
};
library.context(c, root);vm.runInContext(fs.readFileSync(path.join(root,'status.js'),'utf8'),c);vm.runInContext(runShapes(root),c);
c.defaults=()=>({});c.clone=value=>JSON.parse(JSON.stringify(value));
for(const name of ['data.js','inspect.js','live-study.js','live-tasks.js','live-activity.js'])vm.runInContext(fs.readFileSync(path.join(root,name),'utf8'),c);
vm.runInContext('globalThis.A=LiveActivity;globalThis.T=LiveTasks;globalThis.ST=LiveStudy;globalThis.D=Data;',c);
const item=(ordinal,schema,payload,extra={})=>({ordinal,observation_id:'obs-'+ordinal,occurred_at:'2026-09-14T12:00:'+String(ordinal%60).padStart(2,'0')+'Z',observed_at:'2026-09-14T12:00:00Z',schema_kind:schema,source_kind:'PRODUCT_OPERATION',source_id:'local-web:i1',source_sequence:ordinal,task_id:null,run_id:null,stage_id:null,correlation_ids:[],authority:'OPERATIONAL_ASSERTION',retention_class:'TRANSIENT_OPERATIONAL',availability:'AVAILABLE',payload,...extra});
const page=(items,tasks={})=>({workspace_id:'qa',disposition:'CONTINUED',epoch:'e1',cursor:'e1:'+(items.at(-1)?.ordinal||0),head:items.at(-1)?.ordinal||0,more:false,items,unavailable:0,tasks,observer:{status:'OK'},read_cost:{observations:items.length,elapsed_ms:1}});
const admitted=(ordinal,task,ref)=>item(ordinal,'ProductOperationObserved',{operation:'EXPERIMENT_RUN',caller:'EXTERNAL_AUTOMATION',phase:'RETURNED',operation_ref:ref,status:'ADMITTED',task_id:task,task_lifecycle:'QUEUED',subject:{}},{task_id:task,run_id:'local-web:'+task});
const evidence=(ordinal,task,kind='ResearchExecutionEvidence',hashValue='ab'+ordinal)=>item(ordinal,'ArtifactVerificationObserved',{artifact_kind:kind,artifact_hash:hashValue,availability:'AVAILABLE'},{task_id:task,run_id:'local-web:'+task,authority:'ARTIFACT_ASSERTION',source_kind:'PRODUCT_ARTIFACT'});
const projection=(task,lifecycle)=>({[task]:{...TASKS[task],lifecycle}});
(async()=>{
  const {A,ST,D,T}=c;
  hash.value='#page=lab';
  // 1. The supervisor's reproduction. The followed result reaches the real LiveStudy.open, whose
  //    readback is held; the viewer pins and navigates to History; the response is released.
  //    The page and route must stay on History: a stale reader answer rewrites nothing.
  A.absorbPage(page([admitted(1,'task-a','op1')],projection('task-a','RUNNING')));A.follow('task-a');
  assert.equal(A.state().following,'task-a');
  const studyRead=gate('/api/experiments/readback');
  A.absorbPage(page([evidence(2,'task-a')],projection('task-a','SUCCEEDED')));
  await entered(studyRead,'LiveStudy readback');
  assert.equal(requests.filter(p=>p==='/api/experiments/readback').length,1,'the metadata-known result reaches its reader with one strict read');
  assert.equal(c.app.page,'factor','the automatic open reached LiveStudy.open and navigated');
  assert.equal(c.hashParams().get('study'),'task-a');
  A.pin();c.navigate('history');
  assert.equal(c.app.page,'history');assert.equal(A.state().following,null);
  studyRead.release();
  await settled();await settled();await settled();await settled();
  assert.equal(c.app.page,'history','a stale study answer does not move the reader back');
  assert.equal(c.hashParams().get('page'),'history','nor rewrite the route');
  assert.equal(A.state().following,null);assert.equal(A.state().opening,false);
  // 2. Successful intentional navigation does not invalidate itself: the same open, undisturbed,
  //    lands on the study, the route names it, and the follow is cleared exactly once.
  hash.value='#page=lab';c.app.page='lab';
  A.absorbPage(page([admitted(3,'task-b','op3')],projection('task-b','RUNNING')));A.follow('task-b');
  A.absorbPage(page([evidence(4,'task-b')],projection('task-b','SUCCEEDED')));
  for(let i=0;i<40 && (A.state().opening || c.app.page!=='factor');i++)await settled();
  assert.equal(c.app.page,'factor');assert.equal(c.hashParams().get('study'),'task-b');assert.equal(c.hashParams().get('page'),'factor');
  assert.equal(A.state().following,null);assert.equal(A.state().opening,false);
  assert.equal(ST.routeContext('factor').study,'task-b','LiveStudy holds the opened study');
  // 3a. The installed Portfolio replay: its followed result is a saved `result:` entry that
  //     History lists as INSTALLED_RESULT and the workbench opens as its book (the Portfolio
  //     page reads an installed result; the product, 3de3f325).
  //     The History refresh is held; the viewer pins and moves on; the released answer opens nothing.
  hash.value='#page=lab';c.app.page='lab';c.app.book=null;
  history=[{entry_id:'result:'+H('9'),kind:'INSTALLED_RESULT',status:'SUCCEEDED',task_id:'task-p',recorded_at:'2026-09-14T00:00:00Z',strategy_package_id:'IW184',book:{result_hash:H('9')}}];
  results=[{result_hash:H('9'),report_hash:H('r'),program_hash:H('p'),task_id:'task-p',completed_at:'2026-09-14T00:00:00Z'}];
  A.absorbPage(page([admitted(5,'task-p','op5')],projection('task-p','RUNNING')));A.follow('task-p');
  const historyRead=gate('/api/research-history');
  A.absorbPage(page([evidence(6,'task-p','PortfolioResearchResult',H('9'))],projection('task-p','SUCCEEDED')));
  await entered(historyRead,'History refresh before the saved entry opens');
  A.pin();c.navigate('history');const routeBefore=hash.value;
  historyRead.release();await settled();await settled();await settled();await settled();
  assert.equal(c.app.page,'history');assert.equal(hash.value,routeBefore,'a stale History answer rewrites nothing');
  assert.ok(!renders.some(r=>r.startsWith('existing:')),'the abandoned open rendered no saved object');
  assert.equal(A.state().following,null);assert.equal(A.state().opening,false);
  assert.ok(!requests.includes('/api/workbench/portfolio'),'the abandoned open read no book');
  // 3b. An authored Portfolio study (a research_experiment whose readback names a Portfolio
  //     source) opens through the real Data.openPortfolio; with its book response held, the
  //     viewer pins and moves to History; the released answer must not rewrite the route.
  hash.value='#page=lab';c.app.page='lab';c.app.book=null;
  A.absorbPage(page([admitted(7,'task-q','op7')],projection('task-q','RUNNING')));A.follow('task-q');
  const bookRead=gate('/api/workbench/portfolio');
  A.absorbPage(page([evidence(8,'task-q')],projection('task-q','SUCCEEDED')));
  await entered(bookRead,'book readback');
  assert.equal(c.app.page,'portfolio','the automatic open reached Data.openPortfolio and navigated');assert.equal(c.app.book,'task-q');
  A.pin();c.navigate('history');const bookRoute=hash.value;
  bookRead.release();await settled();await settled();await settled();await settled();
  assert.equal(c.app.page,'history');assert.equal(hash.value,bookRoute,'a stale book answer rewrites nothing in the route');
  assert.equal(A.state().following,null);assert.equal(A.state().opening,false);
  // 4. Reconnect / retention loss: the followed installed replay's artifact row was never seen
  //    by this reader (the feed reset), and Task Control says SUCCEEDED. The result is found
  //    through the Portfolio owner's index and the REPORT of that exact result, then opened as
  //    the saved entry, which is its book -- not reported absent.
  hash.value='#page=lab';c.app.page='lab';c.app.book=null;renders.length=0;toasts.length=0;
  results=[{result_hash:H('7'),report_hash:H('r'),program_hash:H('p'),task_id:'task-r',completed_at:'2026-09-14T00:00:00Z'}];
  history=[{entry_id:'result:'+H('7'),kind:'INSTALLED_RESULT',status:'SUCCEEDED',task_id:'task-r',recorded_at:'2026-09-14T00:00:00Z',strategy_package_id:'IW184',book:{result_hash:H('7')}}];
  A.absorbPage(page([admitted(9,'task-r','op9')],projection('task-r','RUNNING')));A.follow('task-r');
  const before=requests.length;
  A.absorbPage({...page([],projection('task-r','SUCCEEDED')),disposition:'RESET',epoch:'e1',cursor:'e1:9',head:9});
  for(let i=0;i<40 && A.state().opening;i++)await settled();
  await settled();await settled();
  assert.deepEqual(requests.slice(before).filter(r=>r.startsWith('/api/')),['/api/results','/api/report','/api/research-history','/api/experiments','/api/workbench/portfolio'],'index, exact readback, the saved entry (History with its experiment listing), then its book');
  assert.equal(c.app.page,'portfolio','the exact saved result opened as its book');assert.equal(c.app.book,'task-r');
  assert.ok(!toasts.some(m=>m.includes('records no result')&&m.includes('task-r')),'nothing was reported absent');
  assert.equal(A.state().following,null,'the follow ended on the open');
  // 4a'. U10: a Task that reused a result the index names under its first Task opens it too -- the REPORT names
  //      every Task that used it -- and a result none names is not taken for it
  results=[{result_hash:H('6'),report_hash:H('r'),program_hash:H('p'),task_id:'task-other',completed_at:'2026-09-13T00:00:00Z'},{...results[0],reused_by:['task-reuser']}];
  requests.length=0;const selectedReadStart=urls.length;
  const reused=await D.installedResult('task-reuser');
  assert.equal(reused.result_hash,H('7'),'the reusing Task opens the result it used');
  assert.deepEqual(requests.filter(r=>r.startsWith('/api/')),['/api/results','/api/report'],'exact Task metadata, then its one REPORT');
  assert.deepEqual(urls.slice(selectedReadStart),['/api/results?task_id=task-reuser','/api/report?result_hash='+H('7')],'the owner resolves the selected reuse Task without an unfiltered scan');
  const absentReadStart=urls.length;
  assert.equal(await D.installedResult('task-nobody'),null,'a Task no REPORT names has no result');
  assert.deepEqual(urls.slice(absentReadStart),['/api/results?task_id=task-nobody'],'absent selected Task metadata triggers no REPORT read');
  results=[{...results[1],reused_by:undefined}];
  // 4b. A settled installed replay the Portfolio owner's index does not name is reported as
  //     such, by the owner, not by a History search for the wrong kind.
  results=[];history=[];
  A.absorbPage(page([admitted(10,'task-s','op10')],projection('task-s','RUNNING')));A.follow('task-s');
  A.absorbPage({...page([],projection('task-s','SUCCEEDED')),disposition:'RESET',epoch:'e1',cursor:'e1:10',head:10});
  for(let i=0;i<40 && A.state().opening;i++)await settled();
  await settled();await settled();
  assert.ok(toasts.some(m=>m.includes('The Portfolio owner records no result')),'the owner\'s answer is what is shown');
  assert.equal(A.state().following,null);
  assert.ok(!requests.some(r=>r.includes('/api/experiments/run')),'nothing here ever requests a RUN');
  // 5. The Task Center's standing line names the liveness source it was given: a runner
  //    heartbeat 0 s old beyond a 60 s durable timestamp reads as observed; the durable
  //    timestamp alone reads as not recent and says no runner heartbeat is recorded.
  recoveryViews['task-b']=recoveryView('task-b',liveness('OBSERVED','OPERATIONAL',0,2));
  await T.open('task-b');await new Promise(r=>setImmediate(r)); // the open's own activity read has settled
  let painted=body.innerHTML;
  // A poll that changes nothing the inspector shows writes nothing: the Task body keeps its
  // markup (and with it a reader's selection, scroll and disclosures).
  const bodyWrites=body.writes();
  await T.refresh();
  assert.equal(body.writes(),bodyWrites,'an unchanged Task body is not repainted');
  assert.ok(painted.includes('Heartbeat at 2026-09-15T10:01:00.000Z (runner heartbeat #2)'),'the runner signal is the source, said as its instant (law 133): '+painted.slice(painted.indexOf('Running'),painted.indexOf('Running')+120));
  assert.ok(!painted.includes('No heartbeat since'),'a fresh runner heartbeat is never read as stale');
  recoveryViews['task-b']=recoveryView('task-b',liveness('NOT_RECENT','DURABLE_ONLY',60,null));
  await T.refresh();painted=body.innerHTML;
  assert.ok(painted.includes('No heartbeat since 2026-09-15T10:00:00.000Z (Task Control timestamp only; no runner heartbeat recorded for this execution)'),'the durable-only source is named: '+painted.slice(painted.indexOf('Running'),painted.indexOf('Running')+200));
  recoveryViews['task-b']=recoveryView('task-b',liveness('NOT_RECENT','UNREADABLE',60,null));
  await T.refresh();painted=body.innerHTML;
  assert.ok(painted.includes('a runner heartbeat store could not be read, so a signal may be hidden'),'an unreadable store is said, not guessed');
  // The age follows the later timestamp and names its source: Task Control's timestamp is 0 s
  // old while the runner's bound heartbeat #5 is a minute old -- available, at its own age,
  // never shown as refreshed. With the signal the later one, it is the source, as above.
  recoveryViews['task-b']=recoveryView('task-b',liveness('OBSERVED','OPERATIONAL',0,5,'DURABLE_TIMESTAMP',60));
  await T.refresh();painted=body.innerHTML;
  assert.ok(painted.includes('Heartbeat at 2026-09-15T10:01:00.000Z (Task Control timestamp; runner heartbeat #5 is older, at 2026-09-15T10:00:00.000Z)'),'the durable timestamp is the source and the older signal keeps its own instant: '+painted.slice(painted.indexOf('Running'),painted.indexOf('Running')+160));
  assert.ok(!painted.includes('(runner heartbeat #5)'),'the older runner signal is not shown as the source');
  recoveryViews['task-b']=recoveryView('task-b',liveness('OBSERVED','OPERATIONAL',3,5,'OPERATIONAL_SIGNAL',3));
  await T.refresh();painted=body.innerHTML;
  assert.ok(painted.includes('Heartbeat at 2026-09-15T10:00:57.000Z (runner heartbeat #5)'),'the newer runner signal is the source');
  assert.equal(body.writes(),bodyWrites+4,'each changed liveness line repainted the Task body once');
  // V507: a deferred Task's stop box says why in its owner's words, from its status, and when it waits until; once
  //    the time has passed (the view's own read) the state's way stands
  recoveryViews['task-b']={...recoveryView('task-b',liveness('NOT_RECENT','DURABLE_ONLY',60,null)),lifecycle:'DEFERRED',operation_running:false,stop:{code:'data.rate_limited',detail:'Deferred.'}};
  const statusOf=responses['/api/status'],owner='The market data provider limited the requests or did not answer.';
  responses['/api/status']=(q)=>({task_id:q.get('task_id'),lifecycle:'DEFERRED',detail:owner,retry_after_at:'2026-09-15T11:00:00Z',next_requests:{read:{operation:'DATA_UPDATE_READBACK',task_id:q.get('task_id')}}});
  await T.refresh();painted=body.innerHTML;
  assert.ok(painted.includes('REFUSED(: DEFER('+owner+'))')&&painted.includes('Waiting until 2026-09-15T11:00:00'),'deferred: its owner\'s words and when it waits until: '+painted.slice(painted.indexOf('REFUSED'),painted.indexOf('REFUSED')+300));
  responses['/api/status']=(q)=>({task_id:q.get('task_id'),lifecycle:'DEFERRED',detail:owner,retry_after_at:'2026-09-15T10:00:00Z'});
  await T.refresh();painted=body.innerHTML;
  assert.ok(painted.includes('DEFER('+owner+')')&&!painted.includes('Waiting until'),'the time has passed: no wait is said');
  // V629: the inspector reads the same owner sentence as Home, and the offered re-PLAN's effect.
  const rebuiltDetail="The ledger rebuild could not recover this Task's execution or verified progress.",rebuiltNext="Re-PLAN keeps the request for a new plan or Task; its owner chooses reusable sealed evidence and work to redo.";
  const rebuilt=recoveryView('task-b',liveness('NOT_RECENT','DURABLE_ONLY',60,null));
  recoveryViews['task-b']={...rebuilt,lifecycle:'BLOCKED',operation_running:false,stop:{code:'task_control.ledger_rebuilt',detail:rebuiltDetail},actions:rebuilt.actions.map(a=>a.action==='REPLAN' ? {...a,expected_effect:rebuiltNext} : a)};
  await T.refresh();painted=body.innerHTML;
  assert.ok(painted.includes('REFUSED(: '+rebuiltDetail+') NEXT('+rebuiltNext+')'),"the owner's reason and offered effect stay together upfront, outside the action facts");
  assert.ok(painted.includes('<details class="reveal-details inspector-section task-receipt"><summary>Receipt</summary>'),'the full receipt stays available in a native disclosure');
  assert.ok(!painted.includes('REFUSED(task_control.ledger_rebuilt:')&&!painted.includes('<span class="mono">task_control.ledger_rebuilt</span>'),'a code is metadata, never the reason');
  responses['/api/status']=statusOf;
  T.close();
  // Compact stage progress is a count, not an elapsed-time estimate; completed
  // rows retain the count and badge without a full-width completed bar.
  c.LiveTeam={section:()=>''};
  D.setTasks([{...TASKS['task-b'],lifecycle:'SUCCEEDED',verified_stage_count:2,total_stage_count:2}]);
  const completedRows=T.page().split('</section>',1)[0]; // exclude the still-selected running detail
  assert.ok(completedRows.includes('Verified stages · 2 / 2'));
  assert.ok(!completedRows.includes('role="progressbar"'));
  D.setTasks([{...TASKS['task-b'],lifecycle:'RUNNING',verified_stage_count:1,total_stage_count:2}]);
  // round 72: a moving row says its verified count in words; no bar under a row
  assert.ok(T.page().includes('Verified stages · 1 / 2') && !T.page().includes('role="progressbar"'));
  // U22, U38: on the Task Center an unfinished Task's progress is the guardian's one read (not the list's
  // projection), a health other than HEALTHY is marked; the incidents are listed open first, each with its
  // attempts, and a Task with an open incident says so on its row.
  c.tile=(ic,tone)=>'<'+ic+':'+tone+'>';c.app.page='tasks';
  responses['/api/tasks/guardian']=()=>({status:'TASK_GUARDIAN',tasks:[{task_id:'task-b',task_kind:'research_experiment',lifecycle:'RUNNING',health:{status:'STALLED'},progress:{verified_stage_count:1,total_stage_count:3,current_stage:'fit_models'},preserved_stages:['prepare'],incidents:[],actions:[]}],counts:{STALLED:1}});
  responses['/api/tasks/incidents']=()=>({status:'TASK_INCIDENTS',open:1,incidents:[
    {key:'k-old',task_id:'task-a',task_kind:'research_experiment',code:'task.worker_lost',state:'RESOLVED',incident:{incident_code:'task.worker_lost',detected_at:'2026-09-14T09:00:00Z',user_safe_detail:'The worker stopped answering.'},remedies:[],attempts:[]},
    {key:'k-open',task_id:'task-b',task_kind:'research_experiment',code:'task.stalled',state:'OPEN',incident:{incident_code:'task.stalled',detected_at:'2026-09-15T09:00:00Z',user_safe_detail:'No stage moved for an hour.'},remedies:[],attempts:[{ordinal:1,action:'RECOVER',disposition:'FAILED'}]}]});
  urls.length=0;await T.refresh();await new Promise(r=>setImmediate(r));
  // R11 (N8): the moving Task the reader has open is followed by the Host's own wait, not the two-second poll
  assert.ok(urls.includes('/api/status?task_id=task-b&wait_seconds=5'),'a moving Task is followed by the Host\'s wait: '+JSON.stringify(urls));
  const center=T.page();
  assert.ok(center.includes('Verified stages · 1 / 3')&&center.includes('fit_models'),'the row reads the guardian’s progress: '+center.slice(0,600));
  assert.ok(center.includes('[blocked:Open incident]'),'a Task with an open incident says so on its row');
  const incidents=center.slice(center.indexOf('GROUP(Incidents:2)'));
  assert.ok(incidents.startsWith('GROUP(Incidents:2)'),'the incidents are a group of the Task Center');
  assert.ok(incidents.indexOf('task.stalled')<incidents.indexOf('task.worker_lost'),'open first, then the resolved');
  assert.ok(incidents.includes('No stage moved for an hour.|[blocked:Open]|1 attempt')&&incidents.includes('[succeeded:Resolved]'),'each incident with its state and attempts: '+incidents.slice(0,400));
  delete responses['/api/tasks/guardian'];delete responses['/api/tasks/incidents'];c.app.page='lab';
  // A Task that stops moving in Task Control's report is said once, through the real reader (the
  // user, 2026-09-25): a move to an ended state toasts; the same report again, a Task first seen,
  // and a Task that only moves say nothing.
  const sayings=toasts.length;
  D.setTasks([{...TASKS['task-b'],lifecycle:'SUCCEEDED',verified_stage_count:2,total_stage_count:2}]);
  assert.equal(toasts.length,sayings+1);assert.equal(toasts.at(-1),'Task completed');
  D.setTasks([{...TASKS['task-b'],lifecycle:'SUCCEEDED',verified_stage_count:2,total_stage_count:2}]);
  D.mergeTasks([{...TASKS['task-a'],lifecycle:'QUEUED'}]);D.mergeTasks([{...TASKS['task-a'],lifecycle:'RUNNING'}]);
  assert.equal(toasts.length,sayings+1,'no second saying, nothing for a Task first seen or one that only moves');
  D.mergeTasks([{...TASKS['task-a'],lifecycle:'FAILED'}]);D.mergeTasks([{...TASKS['task-a'],lifecycle:'FAILED'}]);
  assert.equal(toasts.length,sayings+2);assert.equal(toasts.at(-1),'Task failed');
  // V615 (U91): a Task's re-PLAN as its owner binds it (`next_requests.replan`) is previewed through the session's route
  // for its operation, the declaration sent whole; the preview admits nothing, and the admission it offers is a separate
  // press -- its reads are no admission
  responses['/api/session']=()=>({session_token:'tok',workspace_id:'qa',research_context:{},routes:{RESEARCH_STRATEGY_PLAN:{method:'POST',path:'/api/research-strategy/plan'},RESEARCH_STRATEGY_PREPARE:{method:'POST',path:'/api/research-strategy/prepare'},EXPERIMENTS:{method:'GET',path:'/api/experiments'},DATA_UPDATE_PLAN:{method:'POST',path:'/api/data-update/plan'},DATA_CHANGE_CONFIRM:{method:'POST',path:'/api/data-update/confirm'},DATA_UPDATE_RUN:{method:'POST',path:'/api/data-update/run'},EXPERIMENT_VERIFY_ALL:{method:'POST',path:'/api/experiments/verify-all'}}});
  responses['/api/decisions']=()=>({decisions:[]});
  await D.connect();
  const declaration={experiment:{kind:'research-strategy',components:['alpha_rebound']}};
  responses['/api/research-strategy/plan']=()=>({status:'PLANNED',plan_hash:H('p'),source_start:'2024-01-02',source_end:'2024-08-12',next_requests:{prepare:{operation:'RESEARCH_STRATEGY_PREPARE',experiment_plan_hash:H('p')},controls:{operation:'EXPERIMENTS'}}});
  responses['/api/research-strategy/prepare']=()=>({status:'ADMITTED',task_id:'task-a',lifecycle:'QUEUED'});
  const held=recoveryView('task-b',liveness('NOT_RECENT','DURABLE_ONLY',60,null));
  recoveryViews['task-b']={...held,lifecycle:'BLOCKED',operation_running:false,actions:held.actions.map(a=>a.action==='REPLAN' ? {...a,operation:'RESEARCH_STRATEGY_PLAN',available:true} : a),next_requests:{replan:{operation:'RESEARCH_STRATEGY_PLAN',experiment_document:declaration}}};
  await T.open('task-b');await settled();await T.refresh();
  const sent=posts.length,asked=dialogs.length;
  await T.replanPreview('task-b');
  assert.deepEqual(posts.slice(sent),[['/api/research-strategy/plan',{experiment_document:declaration}]],'the preview: its route, the declaration whole, no operation field');
  assert.equal(dialogs.length,asked+1,'the preview is shown');
  assert.ok(dialogActions.at(-1).includes('<task-replan-commit:prepare>')&&!dialogActions.at(-1).includes(':controls>'),'the admission it offers is the one press; a read is none: '+dialogActions.at(-1));
  assert.equal(posts.length,sent+1,'nothing admitted by the preview');
  await T.replanCommit('prepare');
  assert.deepEqual(posts.at(-1),['/api/research-strategy/prepare',{experiment_plan_hash:H('p')}],'the confirmed admission, as the preview offered it');
  assert.equal(toasts.at(-1),'Work admitted');
  await T.replanCommit('prepare');assert.equal(posts.length,sent+2,'a second press sends nothing');
  assert.equal(T.selected(),'task-a','the admitted Task opens');
  // V627 (U96): a data change's re-PLAN -- the person's consent the first press, the run its answer then offers the
  // second, each request sent whole, its operation choosing the route; nothing runs until the run is pressed
  responses['/api/data-update/plan']=()=>({status:'PLANNED',update_plan_hash:H('u'),next_requests:{confirm:{operation:'DATA_CHANGE_CONFIRM',update_plan_hash:H('u')}}});
  responses['/api/data-update/confirm']=()=>({status:'APPROVED',update_plan_hash:H('u'),next_requests:{run:{operation:'DATA_UPDATE_RUN',update_plan_hash:H('u')}}});
  responses['/api/data-update/run']=()=>({status:'ADMITTED',task_id:'task-b',lifecycle:'QUEUED'});
  const replanAs=(operation,fields)=>({...held,lifecycle:'BLOCKED',operation_running:false,actions:held.actions.map(a=>a.action==='REPLAN' ? {...a,operation,available:true} : a),next_requests:{replan:{operation,...fields}}});
  recoveryViews['task-b']=replanAs('DATA_UPDATE_PLAN',{update_plan_hash:H('o')});
  await T.open('task-b');await settled();await T.refresh();
  const change=posts.length;
  await T.replanPreview('task-b');
  assert.ok(dialogActions.at(-1).includes('<task-replan-commit:confirm>'),'the preview offers the consent its owner fills: '+dialogActions.at(-1));
  await T.replanCommit('confirm');
  assert.deepEqual(posts.slice(change),[['/api/data-update/plan',{update_plan_hash:H('o')}],['/api/data-update/confirm',{update_plan_hash:H('u')}]],'the preview, then the consent, each whole');
  assert.ok(dialogActions.at(-1).includes('<task-replan-commit:run>')&&dialogBodies.at(-1).includes('Nothing runs yet'),'the consent\'s answer offers the run as the next press');
  await T.replanCommit('run');
  assert.deepEqual(posts.at(-1),['/api/data-update/run',{update_plan_hash:H('u')}],'the run, as the answer filled it');
  assert.equal(toasts.at(-1),'Work admitted');
  // an answer that offers no admission says why in its owner's words, and offers no press
  responses['/api/data-update/plan']=()=>({status:'PLANNED',update_plan_hash:H('u'),detail:'The provider has not published the next session yet.',next_lawful_actions:['WAIT_FOR_THE_NEXT_SESSION'],next_requests:{}});
  recoveryViews['task-b']=replanAs('DATA_UPDATE_PLAN',{update_plan_hash:H('o')});
  await T.open('task-b');await settled();await T.refresh();
  await T.replanPreview('task-b');
  assert.ok(dialogActions.at(-1)===''&&dialogBodies.at(-1).includes('The provider has not published the next session yet.')&&dialogBodies.at(-1).includes('WAIT_FOR_THE_NEXT_SESSION')&&dialogBodies.at(-1).includes('The preview recorded a plan and admitted nothing.'),'no admission: the owner\'s words and its next actions, no press: '+dialogBodies.at(-1));
  // a re-PLAN whose operation admits its work directly (the owner marks it `admits`: the saved studies' verification)
  // is no preview: the press asks in the owner's words and sends nothing; the confirmation sends it whole
  responses['/api/experiments/verify-all']=()=>({status:'ADMITTED',task_id:'task-b',lifecycle:'QUEUED'});
  recoveryViews['task-b']={...held,lifecycle:'BLOCKED',operation_running:false,actions:held.actions.map(a=>a.action==='REPLAN' ? {...a,operation:'EXPERIMENT_VERIFY_ALL',admits:true,available:true,scope:'A new Task through EXPERIMENT_VERIFY_ALL.',expected_effect:'Starts a new Task.'} : a),next_requests:{replan:{operation:'EXPERIMENT_VERIFY_ALL'}}};
  await T.open('task-b');await settled();await T.refresh();
  const direct=posts.length;
  await T.replanPreview('task-b');
  assert.equal(posts.length,direct,'a direct re-PLAN sends nothing at its first press');
  assert.ok(dialogActions.at(-1).includes('<task-replan-commit:replan>')&&dialogBodies.at(-1).includes('A new Task through EXPERIMENT_VERIFY_ALL.'),'it asks first, in the owner\'s words: '+dialogBodies.at(-1));
  await T.replanCommit('replan');
  assert.deepEqual(posts.slice(direct),[['/api/experiments/verify-all',{}]],'the confirmation sends it whole');
  assert.equal(toasts.at(-1),'Work admitted');
  // a re-PLAN its owner cannot bind (an evidence Task keeps its issuer scope, not its book) is held in the owner's words,
  // its way on the book's choice in History; nothing to preview
  const words='This evidence Task retains its issuer scope, not the book selector needed to plan again. Choose the book from history, then preview its evidence preparation.';
  recoveryViews['task-b']={...held,lifecycle:'BLOCKED',operation_running:false,actions:held.actions.map(a=>a.action==='REPLAN' ? {...a,operation:'EVIDENCE_PREVIEW',available:false,reason:words} : a),next_requests:{books:{operation:'RESEARCH_HISTORY'}}};
  await T.open('task-b');await settled();await T.refresh();
  assert.ok(body.innerHTML.includes('Choose the book in History')&&body.innerHTML.includes(words)&&!body.innerHTML.includes('<task-replan:'),'held: the owner\'s words and the way on');
  {const before=posts.length;await T.replanPreview('task-b');assert.equal(posts.length,before,'a held re-PLAN previews nothing');}
  T.close();
  await settled();
  const words631=library.words(root),codes631=library.codes();
  c.t=words631.t;
  for(const lang of ['en','zh']) {
    words631.I18N.set(lang);
    for(const state of codes631.lifecycles) {
      const [,,standing]=T.standing({...recoveryViews['task-b'],lifecycle:state});
      assert.ok(standing.includes('<strong>'+words631.codeWords(state)+'</strong>'),lang+' Task standing uses the shared state word: '+state);
    }
  }
  // A succeeded Task offers Open result only where its kind has a result page; elsewhere the
  // inspector is the result, and the button would only reopen it.
  for(const [kind,offered] of [['model_training_input_preparation',false],['workspace_preparation',true]]) {
    const id='done-'+kind;TASKS[id]={...TASKS['task-b'],task_id:id,task_kind:kind,lifecycle:'SUCCEEDED'};
    recoveryViews[id]={...recoveryViews['task-b'],task_id:id,task_kind:kind,lifecycle:'SUCCEEDED',operation_running:false};
    await T.open(id);await settled();
    assert.equal(body.innerHTML.includes('task-result'),offered,kind+': Open result only where a result page exists');
  }
  finish();
})().catch(e=>{console.error(e);process.exitCode=1;});
