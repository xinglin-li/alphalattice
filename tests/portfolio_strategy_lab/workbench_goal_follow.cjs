// UIFOLLOW: real Task/Goal/activity checkpoints from the Python producer reach the
// actual data layer, router, Task inspector, study reader and published-review reader.
// Scientific readbacks below are transport fixtures, not scientific/native acceptance.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const library=require('./workbench_library.cjs'),finish=library.guard('workbench_goal_follow');
const appDir=path.resolve(process.argv[2]),fixture=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const project=path.resolve(__dirname,'../..'),build=fs.readFileSync(path.join(project,'scripts/build_local_web_ui.py'),'utf8');
const files=[...build.slice(build.indexOf('APP_FILES = ['),build.indexOf('\n]',build.indexOf('APP_FILES = ['))).matchAll(/"js\/app\/([^"\n]+)"/g)].map(m=>m[1]).filter(n=>n!=='boot.js');
// Only immutable source is shared; every case creates fresh module state and DOM.
const scripts=new Map();
const script=n=>{if(!scripts.has(n))scripts.set(n,new vm.Script(fs.readFileSync(path.join(appDir,n),'utf8'),{filename:n}));return scripts.get(n);};
const clone=v=>JSON.parse(JSON.stringify(v)),tick=()=>new Promise(r=>setImmediate(r));
async function until(test,label){for(let i=0;i<100&&!test();i++)await tick();assert.ok(test(),label);}
async function settle(){for(let i=0;i<12;i++)await tick();}
const phase=name=>fixture.checkpoints.find(v=>v.phase===name);
const scope='goal:'+fixture.goal_id;
const hash='a'.repeat(64);
function element(){return {dataset:{},style:{setProperty(){}},classList:{add(){},remove(){},toggle(){},contains(){return false;}},innerHTML:'',childElementCount:1,open:false,hidden:false,querySelector:()=>null,querySelectorAll:()=>[],addEventListener(){},removeEventListener(){},contains:()=>false,setAttribute(){},focus(){},matches:()=>false,close(){this.open=false;},showModal(){this.open=true;}};}
function context(initial,{address='#page=overview&follow='+scope,typing=false,dialogOpen=false,narrative=null,decisions=null}={}){
  let checkpoint=clone(initial),overrideNarrative=narrative,overrideDecisions=decisions;
  const requests=[],moves=[],toasts=[],gates=new Map(),storage=new Map(),main=element(),dialog=element(),live=element(),inspector=element();
  dialog.open=dialogOpen;
  const nodes=new Map([['#main',main],['#dialog',dialog],['#tpLive',live],['#inspector',inspector]]);
  const location={hash:address,search:'',href:'http://127.0.0.1/workbench.html'+address};
  const setAddress=url=>{location.hash=url.includes('#')?'#'+url.split('#')[1]:url;location.href='http://127.0.0.1/workbench.html'+location.hash;};
  const document={hidden:false,activeElement:typing?{matches:()=>true}:null,querySelector:s=>nodes.get(s)||null,querySelectorAll:()=>[],getElementById:s=>nodes.get('#'+s)||null,addEventListener(){},
    body:element(),documentElement:{...element(),lang:'en',clientWidth:900},createElement:()=>({...element(),content:{children:[]}})};
  const localStorage={getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)};
  const reviewEntry=()=>({entry_id:'review:'+fixture.publication,kind:'CRO_REVIEW',status:'REVIEW_PUBLISHED',task_id:fixture.review,recorded_at:'2026-10-08T04:00:00Z',review_publication_hash:fixture.publication,book:fixture.book});
  const listing=()=>[{task_id:fixture.factor,kind:'factor.screening-development',status:'SUCCEEDED'},{task_id:fixture.portfolio,kind:'portfolio.policy-development',status:'SUCCEEDED'}];
  const readback=id=>({status:'EXPERIMENT_PUBLISHED',task_id:id,program:{kind:id===fixture.portfolio?'portfolio.policy-development':'factor.screening-development'},research_input_id:'qa-follow-fixture',input_binding_hash:hash,
    ...(id===fixture.portfolio?{portfolio_source:{}}:{}),document:{},result:{},limitations:[]});
  async function fetch(endpoint,options={}){
    assert.ok(!options.method||options.method==='GET','following performs reads only');
    requests.push(endpoint);
    const url=new URL(endpoint,'http://127.0.0.1'),q=url.searchParams,held=gates.get(url.pathname);
    // Capture the owner's answer at request time: a late response cannot become a newer one.
    const at=clone(checkpoint),atNarrative=clone(overrideNarrative||at.narrative),atDecisions=clone(overrideDecisions??(at.phase==='review'?[]:at.decisions));
    let value;
    switch(url.pathname){
      case '/api/session':value={session_token:'fixture-only',workspace_id:'qa-goal-follow',routes:library.hostRoutes(),research_context:{inputs:{inputs:[]},tasks:{tasks:at.tasks}}};break;
      case '/api/research-history':value={entries:at.phase==='review'?[reviewEntry()]:[],next_cursor:null};break;
      case '/api/experiments':value={experiments:listing()};break;
      case '/api/goals':value=at.listing;break;
      case '/api/goals/narrative':value=atNarrative;break;
      case '/api/decisions':value={decisions:atDecisions};break;
      case '/api/tasks':value={tasks:at.tasks};break;
      case '/api/tasks/recovery':value=at.recoveries[q.get('task_id')];assert.ok(value,'producer supplied the exact Task recovery view');break;
      case '/api/status':
        if(q.has('wait_seconds'))return new Promise(()=>{}); // bounded host wait is held, never fabricated as a move
        value=at.recoveries[q.get('task_id')]?.status;break;
      case '/api/tasks/incidents':value={incidents:[]};break;
      case '/api/tasks/guardian':value={tasks:[]};break;
      case '/api/research-inputs/readback':
        assert.equal(q.get('task_id'),fixture.factor,'the input owner reads exactly the advertised Task');
        // Labelled transport fixture: routing only, not a captured-input acceptance claim.
        value={status:'SUCCEEDED',task_id:fixture.factor,input_binding_hash:hash};break;
      case '/api/experiments/readback':value=readback(q.get('task_id'));break;
      case '/api/experiments/curation':value={choices:[],receipt_hash:hash,limitations:[]};break;
      case '/api/workbench/portfolio':value={subject:{task_id:q.get('task_id'),session:fixture.book.portfolio_session,input_id:'qa-follow-fixture',input_hash:hash},series:[],metrics:{},holdings:[],sessions:[fixture.book.portfolio_session]};break;
      case '/api/evidence-cro':
        assert.equal(q.get('review_publication_hash'),fixture.publication,'the reader pins the exact published review');
        for(const [key,value]of Object.entries(fixture.book))assert.equal(q.get(key),value,'the review retains exact book field '+key);
        value={state:'REVIEW_PUBLISHED',review_publication_hash:fixture.publication,book:{formation_session:fixture.book.portfolio_session},issuer_rows:[],available_actions:[],required_actions:[],claim_limits:[],eligible_versions:[],issue_cards:[],reasons:[],citations:[],gaps:[]};break;
      case '/api/activity':value={...at.activity,items:[],tasks:{},more:false};break;
      default:throw Error('No declared fixture owner for '+endpoint);
    }
    if(held){gates.delete(url.pathname);held.entered=true;await held.promise;}
    return {ok:true,status:200,headers:{get:()=> 'application/json'},text:async()=>JSON.stringify(value)};
  }
  const c={console,URL,URLSearchParams,TextEncoder,Intl,Date,Math,Set,Map,Promise,AbortController,document,location,localStorage,sessionStorage:localStorage,fetch,
    history:{state:{alpha:1},pushState(state,_title,url){this.state=state;moves.push(url);setAddress(url);},replaceState(state,_title,url){this.state=state;setAddress(url);}},
    window:{ALPHA_PRODUCT:true,AlphaStaticMark:require(path.resolve(appDir,'../engines/static-mark.js')),innerWidth:900,innerHeight:1000,addEventListener(){},removeEventListener(){},matchMedia:()=>({matches:false,addEventListener(){}})},
    navigator:{platform:'Win32'},innerWidth:900,innerHeight:1000,matchMedia:()=>({matches:false,addEventListener(){}}),addEventListener(){},removeEventListener(){},
    requestAnimationFrame:fn=>{fn();return 1;},cancelAnimationFrame(){},queueMicrotask:fn=>Promise.resolve().then(fn),setTimeout:()=>1,clearTimeout(){},setInterval:()=>1,clearInterval(){},
    MutationObserver:class{observe(){}disconnect(){}},ResizeObserver:class{observe(){}disconnect(){}},HTMLElement:class{},
    $:s=>nodes.get(s)||null,$$:()=>[],getSelection:()=>'',scrollTo(){},clone,...library(appDir)};
  vm.createContext(c);script('../data/zh.js').runInContext(c);c.window.ALPHA_ZH_READY=true;
  for(const file of files)script(file).runInContext(c);
  vm.runInContext('globalThis.probe={Data,app,LiveActivity,LiveTasks,LiveStudy,LiveReview,LiveViews,LiveResearch,Inspect,Window,ACTIONS,PRODUCT_ACTIONS,dispatchAction,readRoute,hashParams};',c);
  const p=c.probe,shown={mode:null,by:null,body:'',opens:0,onClose:null};
  // Presentation primitives only. All route mechanics, readers and wanted boundaries stay real.
  c.render=()=>{};c.patchMain=()=>{};c.preserveSurface=()=>({});c.restoreSurface=()=>{};c.notify=m=>toasts.push(m);c.closeDialog=()=>{dialog.open=false;};c.hideToast=()=>{};
  for(const key of ['render','renderSide','renderTop','afterRender','syncFrame','syncRoutes','onNavigate','trailMark','trailAfter','markDetail'])p.Window[key]=()=>{};
  p.Window.openInspector=o=>{shown.mode=o.mode;shown.by=o.by;shown.body=String(o.body);shown.onClose=o.onClose;shown.opens++;return true;};
  p.Window.inspectorMode=()=>shown.mode;p.Window.inspectorOpen=()=>Boolean(shown.mode);
  p.Window.openedBy=(a,v)=>shown.by?.[0]===a&&shown.by?.[1]===v;
  p.Window.setInspectorBody=body=>{shown.body=String(body);};
  p.Window.closeInspector=()=>{const onClose=shown.onClose;Object.assign(shown,{mode:null,by:null,onClose:null});onClose?.();return true;};
  p.Window.closeDetail=()=>{};p.LiveResearch.ready=()=>{};
  p.readRoute();
  return {c,p,shown,dialog,requests,moves,toasts,checkpoint:name=>{checkpoint=clone(phase(name));},feed:name=>p.LiveActivity.absorbPage(clone(phase(name).activity)),
    setNarrative:value=>{overrideNarrative=value;},setDecisions:value=>{overrideDecisions=value;},
    hold:route=>{let release;const held={promise:new Promise(r=>{release=r;}),entered:false,release:()=>release()};gates.set(route,held);return held;},
    connect:async()=>{await p.Data.connect();assert.equal(p.Data.workspaceStatus,'ready',p.Data.workspaceError);await settle();}};
}
function q(ctx){return new URLSearchParams(ctx.c.location.hash.slice(1));}
function followed(ctx){assert.equal(q(ctx).get('follow'),scope,'the exact Goal remains followed');assert.equal(ctx.p.LiveActivity.state().following,scope);}
async function manual(ctx,page='history'){await ctx.p.dispatchAction('go',page);await settle();assert.equal(ctx.p.app.page,page);assert.equal(q(ctx).get('follow'),scope);assert.equal(q(ctx).get('follow_paused'),'1');assert.equal(ctx.p.LiveActivity.followPaused(),scope);}
const checks=[];
async function check(name,fn){await fn();checks.push(name);}
(async()=>{
  await check('actual checkpoints open progress, Factor, Portfolio, a person decision and published review',async()=>{
    const ctx=context(phase('running'));await ctx.connect();
    await until(()=>ctx.shown.by?.[1]===fixture.factor,'the running attributed Task inspector opens');
    assert.equal(ctx.shown.mode,'task');followed(ctx);
    const opens=ctx.shown.opens;ctx.feed('running');await settle();assert.equal(ctx.shown.opens,opens,'the same Task state does not reopen');
    ctx.checkpoint('factor');ctx.feed('factor');
    await until(()=>ctx.p.app.page==='factor'&&q(ctx).get('study')===fixture.factor,'the finished Factor opens its actual reader');
    await settle();assert.equal(ctx.p.LiveStudy.routeContext('factor').study,fixture.factor);followed(ctx);
    const readCount=ctx.requests.filter(v=>v.startsWith('/api/experiments/readback')).length;
    ctx.feed('factor');await settle();assert.equal(ctx.requests.filter(v=>v.startsWith('/api/experiments/readback')).length,readCount,'unchanged completion opens once');
    ctx.checkpoint('portfolio');ctx.feed('portfolio');
    await until(()=>ctx.p.app.page==='portfolio'&&ctx.p.app.book===fixture.portfolio,'the finished Portfolio opens its exact book');await settle();followed(ctx);
    ctx.checkpoint('decision');ctx.feed('decision');
    await until(()=>ctx.p.app.page==='data'&&q(ctx).get('page')==='issues','the actual attributed data decision opens its offered Data maintenance section');
    assert.ok(ctx.p.Data.decisions().some(v=>v.case_token===fixture.case_token&&v.waits_on==='PERSON'&&v.goal_ids.includes(fixture.goal_id)));followed(ctx);
    ctx.checkpoint('review');ctx.feed('review');
    await until(()=>q(ctx).get('review_publication')===fixture.publication,'the published CRO review opens its exact publication');await settle();
    for(const [key,value]of Object.entries(fixture.book))assert.equal(JSON.parse(q(ctx).get('review_selector'))[key],value);
    assert.ok(ctx.requests.some(v=>v.startsWith('/api/evidence-cro?')),'the actual review consumer read its pinned publication');followed(ctx);
  });
  await check('manual navigation pauses the same scope and Follow again resumes',async()=>{
    const ctx=context(phase('factor'));await ctx.connect();await until(()=>ctx.p.app.page==='factor','Factor opened');await settle();
    await manual(ctx);ctx.feed('factor');await settle();assert.equal(ctx.p.app.page,'history');
    await ctx.p.dispatchAction('activity-follow-again','');await until(()=>ctx.p.app.page==='factor','Follow again opens the same Goal result');followed(ctx);assert.ok(!q(ctx).has('follow_paused'));
  });
  await check('a late result cannot move a manually chosen page',async()=>{
    const ctx=context(phase('running'));await ctx.connect();await until(()=>ctx.shown.by?.[1]===fixture.factor,'running Task opened');
    const held=ctx.hold('/api/experiments/readback');ctx.checkpoint('factor');ctx.feed('factor');await until(()=>held.entered,'the actual final reader is held');
    await manual(ctx);const address=ctx.c.location.hash;held.release();await settle();
    assert.equal(ctx.p.app.page,'history');assert.equal(ctx.c.location.hash,address,'the late answer changes no manual address');
  });
  await check('a newer Task checkpoint rejects a held older Goal narrative',async()=>{
    const ctx=context(phase('running')),held=ctx.hold('/api/goals/narrative');
    const connecting=ctx.connect();await until(()=>held.entered,'the real running Goal narrative is held before acceptance');
    ctx.checkpoint('factor');ctx.feed('factor');await settle();held.release();await connecting;await settle();
    assert.equal(ctx.shown.opens,0,'the captured old running narrative cannot open an inspector after the finished event');
    assert.equal(ctx.p.app.page,'overview','a late old narrative opens no obsolete page');
    ctx.feed('factor');await until(()=>ctx.p.app.page==='factor'&&q(ctx).get('study')===fixture.factor,'the next existing heartbeat opens the exact finished Factor');
    followed(ctx);
  });
  for(const busy of ['typing','dialogOpen'])await check('following waits during '+busy,async()=>{
    const ctx=context(phase('factor'),{[busy]:true});await ctx.connect();ctx.feed('factor');await settle();
    assert.equal(ctx.p.app.page,'overview');assert.equal(ctx.shown.opens,0,'no inspector replaces the busy surface');
    ctx.c.document.activeElement=null;ctx.dialog.open=false;ctx.feed('factor');
    await until(()=>ctx.p.app.page==='factor','the pending checkpoint opens when the busy surface is released');
  });
  await check('only exactly attributed person decisions are opened',async()=>{
    const checkpoint=phase('decision'),stop=checkpoint.decisions.find(v=>v.task_id===fixture.blocked);
    assert.ok(stop,'the producer supplies its actual blocked decision');
    const none=clone(checkpoint.narrative);none.record.tasks=[];
    const ctx=context(checkpoint,{narrative:none,decisions:[{...stop,waits_on:'AGENT'},{...stop,goal_ids:['unrelated-goal'],waits_on:'PERSON'}]});await ctx.connect();ctx.feed('decision');await settle();
    assert.equal(ctx.shown.opens,0,'AGENT and other-Goal decisions never navigate');assert.equal(ctx.p.app.page,'overview');
    const legacy=clone(stop);delete legacy.waits_on;
    const legacyCtx=context(checkpoint,{narrative:none,decisions:[legacy]});await legacyCtx.connect();legacyCtx.feed('decision');
    await until(()=>legacyCtx.shown.by?.[1]===fixture.blocked,'an exact legacy decision with no waits_on is person-facing');
  });
  await check('successive same-Goal input-version decisions retain their exact family and cutoffs',async()=>{
    // Labelled owner-shaped transport decisions; the intervening Portfolio checkpoint is real.
    const first={kind:'INPUT_VERSION',waits_on:'PERSON',goal_ids:[fixture.goal_id],research_input_id:'qa-follow-fixture',input_through:'2024-08-12',data_through:'2024-08-13'};
    const ctx=context(phase('factor'),{decisions:[first]});let inputTransitions=0;
    const push=ctx.c.history.pushState.bind(ctx.c.history);
    ctx.c.history.pushState=(state,title,address)=>{
      const before=q(ctx).get('page'),value=push(state,title,address);
      if(before!=='inputs'&&q(ctx).get('page')==='inputs')inputTransitions++;
      return value;
    };
    await ctx.connect();
    await until(()=>ctx.p.app.page==='inputs','the first input-version decision opens its offered page');
    // objectEntry preserves the old address in History without navigating to it again.
    const inputOpens=()=>inputTransitions;
    const firstOpens=inputOpens();assert.equal(firstOpens,1);
    ctx.feed('factor');await settle();assert.equal(inputOpens(),firstOpens,'an unchanged input decision opens once');
    ctx.setDecisions([]);ctx.checkpoint('portfolio');ctx.feed('portfolio');
    await until(()=>ctx.p.app.page==='portfolio'&&ctx.p.app.book===fixture.portfolio,'clearing the first decision releases the actual Portfolio result');await settle();
    const second={...first,input_through:first.data_through,data_through:'2024-08-14'};
    ctx.setDecisions([second]);ctx.checkpoint('decision');ctx.feed('decision');
    await until(()=>ctx.p.app.page==='inputs','a newer same-Goal input decision opens even without Task, plan, publication or case identifiers');
    assert.equal(inputOpens(),firstOpens+1);followed(ctx);
    ctx.feed('decision');await settle();assert.equal(inputOpens(),firstOpens+1,'repeating the accepted new cutoff does not reopen it');
  });
  await check('a delayed exact person decision opens on the existing cadence without a new Goal or Task event',async()=>{
    const checkpoint=phase('decision'),none=clone(checkpoint.narrative);none.record.tasks=[];
    const person=checkpoint.decisions.find(v=>v.case_token===fixture.case_token);
    assert.equal(person.waits_on,'PERSON');assert.ok(person.goal_ids.includes(fixture.goal_id));
    // The initial owner answer is partial: the same real decision is temporarily absent.
    const partial=checkpoint.decisions.filter(v=>v!==person);
    const ctx=context(checkpoint,{narrative:none,decisions:partial});await ctx.connect();
    ctx.feed('decision');await settle();assert.equal(ctx.p.app.page,'overview');assert.equal(ctx.moves.length,0);
    const reads=ctx.requests.filter(v=>v.startsWith('/api/goals/narrative')).length;
    const taskState=JSON.stringify(ctx.p.Data.tasks());
    ctx.setDecisions(checkpoint.decisions);ctx.feed('decision');
    await until(()=>ctx.p.app.page==='data'&&q(ctx).get('page')==='issues','the late owner answer opens the same exact Goal decision');await settle();
    assert.equal(JSON.stringify(ctx.p.Data.tasks()),taskState,'the Task lifecycle and hashes did not change');
    assert.equal(ctx.requests.filter(v=>v.startsWith('/api/goals/narrative')).length,reads,'no narrative event was needed');
    assert.ok(ctx.p.Data.decisions().some(v=>v.case_token===person.case_token&&v.goal_ids.includes(fixture.goal_id)));followed(ctx);
    const moves=ctx.moves.length;ctx.feed('decision');await settle();assert.equal(ctx.moves.length,moves,'an unchanged accepted decision opens once');
  });
  await check('legacy un-attributed Tasks are not guessed from the activity feed',async()=>{
    const none=clone(phase('factor').narrative);none.record.tasks=[];
    const ctx=context(phase('factor'),{narrative:none,decisions:[]});await ctx.connect();ctx.feed('factor');await settle();
    assert.equal(ctx.p.app.page,'overview');assert.equal(ctx.shown.opens,0);
  });
  await check('cold latest and paused launch addresses retain their declared meaning',async()=>{
    const ctx=context(phase('factor'),{address:'#page=overview&follow=latest'});await ctx.connect();await until(()=>ctx.p.app.page==='factor','latest selects the actual active Goal');assert.equal(q(ctx).get('follow'),'latest');
    const paused=context(phase('factor'),{address:'#page=history&follow='+scope+'&follow_paused=1'});await paused.connect();paused.feed('factor');await settle();
    assert.equal(paused.p.app.page,'history');assert.equal(paused.p.LiveActivity.followPaused(),scope);assert.equal(paused.shown.opens,0);
  });
  await check('latest stays quiet with no active Goal, then follows a newly attributed Task',async()=>{
    // Labelled absence transport fixture; the later Goal/Task checkpoint and events are real.
    const empty=clone(phase('running'));empty.listing={goals:[]};empty.tasks=[];empty.decisions=[];
    empty.activity={...empty.activity,items:[],tasks:{}};
    const ctx=context(empty,{address:'#page=overview&follow=latest'});await ctx.connect();
    ctx.p.LiveActivity.absorbPage(clone(empty.activity));await settle(); // establish the owner's initial feed epoch
    const reads=ctx.requests.filter(v=>v==='/api/goals').length;
    assert.ok(reads,'latest checks the actual Goal owner even before a Goal exists');
    for(let i=0;i<2;i++){ctx.p.LiveActivity.absorbPage(clone(empty.activity));await settle();}
    assert.equal(ctx.p.app.page,'overview');assert.equal(ctx.shown.opens,0);assert.equal(ctx.moves.length,0,'an empty Goal list makes no navigation');
    assert.equal(ctx.requests.filter(v=>v==='/api/goals').length,reads,'unchanged quiet heartbeats do not reread the Goal list');
    assert.ok(!ctx.requests.some(v=>v.startsWith('/api/goals/narrative')),'no unknown Goal is opened or guessed');
    ctx.checkpoint('running');ctx.feed('running');
    await until(()=>ctx.shown.by?.[1]===fixture.factor,'new real Task events release the empty heartbeat and open their attributed Task');
    assert.equal(ctx.shown.mode,'task');assert.equal(ctx.p.LiveActivity.state().following,'latest');assert.equal(q(ctx).get('follow'),'latest');
    const opens=ctx.shown.opens;ctx.feed('running');await settle();assert.equal(ctx.shown.opens,opens,'the newly followed state opens once');
  });
  async function transportTask(kind){
    const ctx=context(phase('factor'),{address:'#page=overview&follow='+scope+'&follow_paused=1'});await ctx.connect();
    // Only the advertised transport kind changes. Recovery remains the real owner record.
    ctx.p.Data.setTasks(ctx.p.Data.tasks().map(v=>v.task_id===fixture.factor?{...v,task_kind:kind}:v));
    return ctx;
  }
  await check('an advertised data-update result opens exactly its Data update',async()=>{
    const ctx=await transportTask('workspace_data_update');
    assert.equal(await ctx.p.LiveTasks.openResult(fixture.factor),true);
    assert.equal(ctx.p.app.page,'data');assert.equal(q(ctx).get('update'),fixture.factor);
    assert.equal(ctx.dialog.open,false,'opening the result requires no invented confirmation');
    assert.equal(ctx.shown.opens,0);assert.equal(q(ctx).get('follow_paused'),'1');
  });
  await check('an advertised input capture resolves its exact Task to the binding version',async()=>{
    const ctx=await transportTask('research_input_capture');
    assert.equal(await ctx.p.LiveTasks.openResult(fixture.factor),true);
    assert.equal(ctx.p.app.page,'inputs');assert.equal(q(ctx).get('version'),hash);
    assert.deepEqual(ctx.requests.filter(v=>v.startsWith('/api/research-inputs/readback?')),['/api/research-inputs/readback?task_id='+fixture.factor]);
    assert.equal(ctx.dialog.open,false);assert.equal(ctx.shown.opens,0);
  });
  await check('an advertised evidence Task uses its real recovery without inventing a book',async()=>{
    const ctx=await transportTask('alternative_evidence.document_intelligence');
    await ctx.p.LiveTasks.openResult(fixture.factor);
    assert.equal(ctx.p.app.page,'handoff');assert.equal(q(ctx).get('work'),fixture.factor);
    assert.equal(q(ctx).get('review_selector'),null);assert.equal(q(ctx).get('review_publication'),null);
    const facts=ctx.p.LiveReview.facts();assert.equal(facts.work.book,null);
    assert.equal(facts.unresolved.view.task_kind,phase('factor').recoveries[fixture.factor].task_kind,'the original producer recovery remains authoritative');
    assert.ok(ctx.requests.some(v=>v==='/api/tasks/recovery?task_id='+fixture.factor));
    assert.equal(ctx.dialog.open,false);
  });
  await check('a late advertised evidence recovery cannot replace a manual page',async()=>{
    const ctx=await transportTask('alternative_evidence.document_intelligence'),held=ctx.hold('/api/tasks/recovery');
    const opening=ctx.p.LiveTasks.openResult(fixture.factor);await until(()=>held.entered,'the actual evidence recovery reader is held');
    await manual(ctx);const address=ctx.c.location.hash;held.release();assert.equal(await opening,false);await settle();
    assert.equal(ctx.p.app.page,'history');assert.equal(ctx.c.location.hash,address);assert.equal(ctx.dialog.open,false);
  });
  await check('an unknown no-result kind opens the ordinary original Task inspector',async()=>{
    const ctx=await transportTask('qa_unknown_transport_kind');
    assert.equal(await ctx.p.LiveTasks.openResult(fixture.factor),true);
    assert.equal(ctx.shown.mode,'task');assert.equal(ctx.shown.by?.[1],fixture.factor);
    assert.equal(q(ctx).get('task'),fixture.factor);assert.equal(ctx.dialog.open,false);
    assert.ok(ctx.shown.body.includes(fixture.factor),'the inspector holds the original exact Task');
    assert.ok(ctx.requests.some(v=>v==='/api/tasks/recovery?task_id='+fixture.factor));
  });
  console.log(JSON.stringify({holder:'workbench_goal_follow',checks:checks.length,proof:'Task checkpoints and activity/Goal attribution through actual UI consumers; labelled scientific transport fixtures',names:checks}));finish();
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
