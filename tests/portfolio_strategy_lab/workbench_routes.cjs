// TE12 / PG2: enumerate production routes, offered collection rows, object tabs and recovery.
// Primitive DOM and deterministic owner answers keep this a UI contract; the browser regression
// separately presses Reload across actual documents over the service and built bundle.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const project = path.resolve(__dirname, '../..');
const library = require(path.join(project, 'tests/portfolio_strategy_lab/workbench_library.cjs'));
const finish = library.guard('workbench_routes');
const appDir = process.argv[2] ? path.resolve(process.argv[2]) : path.join(project, 'src/alphalattice/interface/local_application/assets/workbench-source/js/app');
// Source files stay immutable during this process. Reuse bytes and compiled modules;
// every scenario still receives its own fresh VM context and fixture state.
const sourceBytes = new Map(), fixedScripts = new Map();
const source = name => {
  if(!sourceBytes.has(name))sourceBytes.set(name,fs.readFileSync(path.join(appDir,name),'utf8'));
  return sourceBytes.get(name);
};
const fixedScript = name => {
  if(!fixedScripts.has(name))fixedScripts.set(name,new vm.Script(source(name),{filename:name}));
  return fixedScripts.get(name);
};
const build = fs.readFileSync(path.join(project, 'scripts/build_local_web_ui.py'), 'utf8');
const files = [...build.slice(build.indexOf('APP_FILES = ['), build.indexOf('\n]', build.indexOf('APP_FILES = ['))).matchAll(/"js\/app\/([^"\n]+)"/g)].map(m => m[1]).filter(name => name!=='boot.js');
const failures = [], counts = {routes:0, rendered_routes:0, registered_pages:0, object_addresses:0, opener_checks:0, offered_rows:0, object_tab_pairs:0, study_shapes:0, renderer_shapes:0, recovery_documents:0};
const H = 'a'.repeat(64), ID = '11111111-2222-3333-4444-555555555555';
const clone = value => JSON.parse(JSON.stringify(value));

function makeContext({hash='#page=overview', body=null, catalog=[],historyRows=[],rawHistory=[],realData=false,inspectorDom=false} = {}) {
  const storage = new Map(), records = {pushes:[], replaces:[], reloads:0, missingData:new Set(), requests:[], readVisits:[]};
  const main = {dataset:{},innerHTML:'', childElementCount:1, querySelector(){return null;}, querySelectorAll(){return [];}, insertAdjacentHTML(_where, value){this.innerHTML += value;}};
  const live = {textContent:''}, dialog = {dataset:{},open:false, close(){this.open=false;}, showModal(){this.open=true;}};
  const inspector = {hidden:true,dataset:{},style:{setProperty(){}},innerHTML:'',querySelector:()=>null,querySelectorAll:()=>[],contains:()=>false,setAttribute(){},focus(){}};
  const selectors = new Map([['#main',main], ['#tpLive',live], ['#dialog',dialog]]);
  if(inspectorDom)selectors.set('#inspector',inspector);
  const location = {hash, search:'', href:'http://127.0.0.1/workbench.html'+hash, reload(){records.reloads++;}};
  const setAddress = address => {if(typeof address==='string'){location.hash=address.includes('#') ? '#'+address.split('#')[1] : address; location.href='http://127.0.0.1/workbench.html'+location.hash;}};
  const dataMethods = {
    workspaceStatus:'ready', workspaceError:'', manifest:{}, status:'ready', ready:false, mode:'local',
    workspace:()=> 'Routing fixture', clocks:()=>({data:'2026-08-03',feature:'2026-08-03'}), inputs:()=>[], sessions:()=>[], experiments:()=>catalog, history:()=>historyRows,
    portfolioEntries:()=>historyRows.map(row=>row.raw).filter(entry=>entry && ['portfolio.policy-development','INSTALLED_RESULT'].includes(entry.kind) && entry.status==='SUCCEEDED'),
    inputVersion:()=>({available:true,date:'2026-08-03',cutoff:'2026-08-03'}), subject:()=>null, notice:()=>'',
    actionableTasks:()=>[], unrecoverableTasks:()=>[], tasks:()=>[], runs:()=>[], runsOf:()=>[], updates:()=>[], decisions:()=>[], versions:()=>[],
    holdings:()=>[], series:()=>[], studies:()=>[], books:()=>[], recent:()=>[], riskLinks:()=>[], researchUpdate:()=>null, cpuBudget:()=>({}),
    lifecycleOf:(_id,fallback)=>fallback, declaredOf:id=>catalog.find(v=>v.task_id===id)||null,
    offers:()=>false, route:op=>library.hostRoutes()[op]?.path || '/fixture/'+op,
    read:async endpoint=>{records.requests.push(endpoint); if(typeof body==='function')return body(endpoint); if(endpoint.startsWith('/api/goals'))return body || {status:'GOAL_LIST',goals:[]}; if(endpoint.includes('curation'))return {candidates:[],curation:[],decision:null}; return body || {};},
    readShared:(...args)=>dataMethods.read(...args),
    visitPage:page=>records.readVisits.push(['visit',page]),
    leavePage:page=>records.readVisits.push(['leave',page]),
    refreshExperiments:async()=>{}, post:async()=>{throw Error('business write forbidden in routing harness');},
    registerTaskReader(){}, registerRefresh(){}, registerTeamReader(){}, subscribe(){},
  };
  // Neutral missing owner answers are recorded, not silently treated as evidence of an owner contract.
  const Data = new Proxy(dataMethods,{get(target,key){if(key in target)return target[key]; records.missingData.add(String(key)); return ()=>null;}});
  const document = {querySelector:s=>selectors.get(s)||null, querySelectorAll:()=>[], getElementById:id=>selectors.get('#'+id)||null,
    documentElement:{dataset:{},lang:'en',style:{setProperty(){}},classList:{add(){},remove(){},toggle(){}},clientWidth:1100},
    body:{dataset:{},classList:{add(){},remove(){},toggle(){},contains(){return false;}},style:{setProperty(){}},querySelectorAll(){return [];}},
    activeElement:null, addEventListener(){},createElement(){return {innerHTML:'',content:{children:[]},classList:{},style:{}};}};
  const localStorage = {getItem:key=>storage.get(key)||null, setItem:(key,value)=>storage.set(key,value)};
  const c = {console:{...console,error(){}}, URL,URLSearchParams,TextEncoder,Intl,Date,Math,Set,Map,Promise,Data,document,location,localStorage,sessionStorage:localStorage,
    history:{state:{alpha:1},pushState(state,_title,address){this.state=state;records.pushes.push(address);setAddress(address);},replaceState(state,_title,address){this.state=state;records.replaces.push(address);setAddress(address);}},
    window:{AlphaStaticMark:require(path.resolve(appDir,'../engines/static-mark.js')),innerWidth:1100,innerHeight:1000,addEventListener(){},removeEventListener(){},confirm:()=>true,matchMedia:()=>({matches:false,addEventListener(){}})},
    navigator:{platform:'Win32'},innerWidth:1100,innerHeight:1000,matchMedia:()=>({matches:false,addEventListener(){}}),
    addEventListener(){},removeEventListener(){},requestAnimationFrame:fn=>{fn();return 1;},cancelAnimationFrame(){},queueMicrotask(){},setTimeout:()=>1,clearTimeout(){},setInterval:()=>1,clearInterval(){},
    MutationObserver:class{observe(){} disconnect(){}},ResizeObserver:class{observe(){} disconnect(){}},HTMLElement:class{},
    clone, $:s=>selectors.get(s)||null,$$:()=>[],getSelection:()=>'',scrollTo(){},async fetch(endpoint,options={}){
      if(!realData)throw Error('network forbidden in routing prototype');if(options.method&&options.method!=='GET')throw Error('business write forbidden in routing harness');records.requests.push(endpoint);
      const value=endpoint.startsWith('/api/session')?{session_token:'fixture-only',workspace_id:'Routing fixture',routes:library.hostRoutes(),research_context:{inputs:{versions:[]},tasks:{tasks:[]}}}:endpoint.startsWith('/api/research-history')?{entries:rawHistory,next_cursor:null}:endpoint==='/api/experiments'?{experiments:catalog}:endpoint==='/api/decisions'?{decisions:[]}:typeof body==='function'?body(endpoint):body || {};
      const resolved = await value;
      return {ok:true,status:200,headers:{get:()=> 'application/json'},text:async()=>JSON.stringify(resolved)};
    },
    viewW:()=>1100, viewH:()=>1000, toolsMenu(){},toggleRowMenu(){},experienceSettings(){},hideToast(){},closeDialog(){},notify(){},copyText(){},
    setHeld(){},toggleDisplayMenu(){},chooseDisplay(){},openCodeRef(){},download(){},reveal(){},
    ...library(appDir)};
  library.context(c);
  // Match the product's dictionary-before-reader boot: I18N retains this object.
  // Loading a replacement dictionary afterwards silently tested English in zh mode.
  fixedScript('../data/zh.js').runInContext(c);c.window.ALPHA_ZH_READY=true;
  for(const file of files) if(realData || file!=='data.js')fixedScript(file).runInContext(c);
  vm.runInContext('globalThis.probe={Data,app,ROUTES,PAGES,PAGE_TABS,OBJECT_KEYS,Inspect,Window,LiveViews,LiveStudy,LiveGoals,LiveModels,LiveFeatureResearch,LiveTeam,LiveActivity,LiveActivation,LiveReview,LiveWorkspace,LiveFeatures,LiveResearch,LiveTasks,Settings,ACTIONS,PRODUCT_ACTIONS,readRoute,routeUrl,routeObject,routeObjectId,listUrl,navigate,objectEntry,render,renderFailure,failureCard};',c);
  const realRender = c.probe.render;
  c.render = ()=>{}; // Navigation calls retain all production route mechanics; layout itself is read explicitly below.
  c.closeDialog = ()=>{}; c.hideToast = ()=>{}; c.notify = ()=>{}; // Primitive chrome only; never replace route or reader logic.
  c.probe.Window.render=()=>{}; // Frame chrome is outside this route/reader contract.
  c.records=records;c.main=main;c.live=live;c.inspector=inspector;c.realRender=realRender;
  c.setRoute = address=>{setAddress(address);c.probe.readRoute();};
  return c;
}

async function check(name, fn) {try{await fn();}catch(error){failures.push({name,message:error.message,stack:error.stack});console.log('FAIL '+name+': '+error.stack);}}
function q(c){return new URLSearchParams(c.location.hash.slice(1));}
function resetHistory(c){c.records.pushes.length=0;c.records.replaces.length=0;}
function nonempty(markup,label){
  assert.ok(String(markup || '').trim(),label+' returns nonempty markup');
  for(const [, declared] of String(markup).matchAll(/\bdata-row-columns="([^"]*)"/g)) {
    const columns=declared.trim().split(/\s+/).filter(Boolean);
    assert.ok(columns.length<=8,label+': named facts fit the shared eight-column row: '+columns.join(', '));
  }
}
const settle=async()=>{for(let i=0;i<4;i++)await new Promise(resolve=>setImmediate(resolve));};
const decode=value=>String(value).replace(/&quot;/g,'"').replace(/&#39;|&#x27;/g,"'").replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>');
// U121 / TE12: inspect the actual shared locator markup, including its ancestry. This
// bounded source-tree control does not claim browser layout or a native clipboard read.
function markupNodes(markup) {
  const nodes=[],stack=[],voids=new Set(['area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr']);
  for(const match of String(markup).matchAll(/<\/?([a-z][\w:-]*)\b([^<>]*)>/gi)) {
    const tag=match[1].toLowerCase();
    if(match[0].startsWith('</')) { const at=stack.map(n=>n.tag).lastIndexOf(tag);if(at>=0)stack.length=at;continue; }
    const attrs=Object.fromEntries([...match[2].matchAll(/([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)].map(a=>[a[1],decode(a[2]??a[3])]));
    const node={tag,attrs,parent:stack.at(-1)||null};nodes.push(node);
    if(!voids.has(tag)&&!match[0].endsWith('/>'))stack.push(node);
  }
  return nodes;
}
function ancestors(node) { const out=[];for(let parent=node.parent;parent;parent=parent.parent)out.push(parent);return out; }
const hasClass=(node,name)=>(node.attrs.class||'').split(/\s+/).includes(name);
function assertLocators(markup,ids,label,{row=null}={}) {
  const nodes=markupNodes(markup),copies=nodes.filter(n=>n.tag==='button'&&n.attrs['data-action']==='copy-text');
  for(const copy of copies)assert.ok(!ancestors(copy).some(n=>['a','button'].includes(n.tag)&&hasClass(n,'list-row-main')),label+': copy is outside the clickable row');
  for(const id of ids) {
    const copy=copies.find(n=>n.attrs['data-value']===id&&(!row||ancestors(n).some(a=>a.attrs['data-key']===row))&&ancestors(n).some(a=>hasClass(a,'run-ref')));
    assert.ok(copy,label+': full identity has the shared copy control: '+id);
    const cell=ancestors(copy).find(n=>hasClass(n,'run-ref'));
    assert.ok(nodes.some(n=>n.attrs['data-tip']===id&&ancestors(n).includes(cell)),label+': hover holds the same full identity: '+id);
    assert.ok(copy.attrs['aria-label'],label+': copy control has accessible words');
  }
}
function localizedContext(lang,options) {
  const c=makeContext(options);
  vm.runInContext("I18N.set('"+lang+"');",c);c.probe.readRoute();return c;
}
function offered(markup,action,value){const tags=[...String(markup).matchAll(/<button\b[^>]*>/g)].map(m=>m[0]);const row=tags.find(tag=>decode(/data-action="([^"]*)"/.exec(tag)?.[1])===action&&decode(/data-value="([^"]*)"/.exec(tag)?.[1])===value);assert.ok(row,'collection offers '+action+' '+value);return row;}
function followAnchor(c,markup,predicate){const href=[...String(markup).matchAll(/<a\b[^>]*href="([^"]*)"/g)].map(m=>decode(m[1])).find(h=>predicate(new URLSearchParams(h.slice(1))));assert.ok(href,'collection offers exact object link');c.history.pushState({alpha:1},'',href);c.probe.readRoute();return href;}
async function press(c,action,value){const handler=c.probe.ACTIONS[action] || c.probe.PRODUCT_ACTIONS[action];assert.equal(typeof handler,'function','offered action '+action+' has a handler');await handler(value);await settle();}
async function connect(c){await c.probe.Data.connect();assert.equal(c.probe.Data.workspaceStatus,'ready',c.probe.Data.workspaceError);c.probe.readRoute();await settle();}
function goalBody(hash=H){return {status:'GOAL_NARRATIVE',goal_hash:hash,head_hash:hash,evidence_verification:'COMPLETE',references:[],record:{},goal:{goal_id:ID,goal_hash:hash,revision:1,parent_hash:null,state:'OPEN',recorded_at:'2026-08-03T12:00:00Z',submitted_by:'HUMAN',declaration:{kind:'RESEARCH',title:'Routing Goal',objective:'Read exact saved routing fixture',criteria:[],deliverables:[]},references:[],statements:[],submission:null,completion:null}};}
function studyBody(kind='alpha.model-development',status='EXPERIMENT_PUBLISHED',shape='development'){
  const b={status,task_id:ID,program:{kind,program_hash:H},research_input_id:'fixture-input',input_binding_hash:H,limitations:[],standing:null,
    document:{experiment:{sessions:{start:'2026-07-01',end:'2026-08-03',as_of:{session:'2026-08-03'}}},alpha:{target_recipe_id:'TARGET',model_parameters:{family:'ridge'},ordered_feature_ids:[]},factor:{factor_ids:[]},risk:{estimator:{capability:'sample-covariance',parameters:{}}}},
    alpha_source:{factor_task_id:ID,curation_receipt_hash:H},execution_preview:{fold_count:0,folds:[],split_policy:{}},result:{candidates:[],folds:[],evaluations:[],evidence_report:{items:[],hypothesis_count:0}},fold_results:[]};
  if(shape==='lifecycle'){delete b.alpha_source;b.lifecycle_research={lifecycle:{month_interval:1,anchor_month:1,training_window_sessions:20,purge_sessions:1,seeds:[1],vintage_weights:[1]},fit_call_count:0,prediction_call_count:0,formation_sessions:[]};}
  if(shape==='qualification'){delete b.alpha_source;b.alpha_qualification={status:'QUALIFIED',conclusions:[],limitations:[],qualification_hash:H};}
  return b;
}
function portfolioBody(){return {schema:'verified-portfolio-display',subject:{task_id:ID,receipt_hash:H,title:'Fixture book',input_id:'fixture-input',input_hash:H,input_date:'2026-08-03',session:'2026-08-03',support:{start:'2026-07-01',end:'2026-08-03',as_of:{session:'2026-08-03',phase:'OFFICIAL_CLOSE'}},cost_per_side:'5'},notice:'Published historical research.',seriesBasis:'index',series:[],sessions:['2026-08-03'],metrics:{},position:{session:'2026-08-03',decisionMode:'REBALANCE',holdingCount:0,cash:1},holdings:[],universe:{eligible:0,total:0,quarantine:0,definition:''},limitations:[],reviewSelector:{experiment_task_id:ID,experiment_receipt_hash:H,portfolio_session:'2026-08-03'},declaration:{top_k:1,tranches:1,exit_rank:1,weight_rule:'ew',cost_bps_per_side:'5'},source:{task_id:ID,receipt_hash:H,input_binding_hash:H,alpha_task_id:ID,candidate_id:'fixture-candidate'}};}
function reviewBody(){return {state:'EVIDENCE_AUTHORITY_NOT_ADMITTED',book:{authority:'DEVELOPMENT_RESULT',result_hash:H,formation_session:'2026-08-03',held_count:0},explanation:'Fixture has no evidence authority.',issuer_rows:[],available_actions:[],required_actions:[],claim_limits:[],eligible_versions:[],issue_cards:[],reasons:[],citations:[],gaps:[]};}

// V683: use production renderPage for visit teardown, with only the content/chrome reduced.
// The read owners, route methods, generations and navigation remain the production ones.
function lifecycleContext(options={}) {
  const c=makeContext({...options,inspectorDom:true});c.scrollX=0;c.scrollY=0;c.main.contains=()=>false;c.main.style={};
  c.probe.LiveViews.page=()=>'<section>Read lifetime fixture</section>';
  c.probe.Window.afterRender=()=>{};c.probe.Window.markDetail=()=>{};
  c.probe.Inspect.closePeek=()=>{};c.probe.Inspect.refreshFacts=()=>{};
  c.probe.Inspect.refreshRecord=()=>{};c.probe.Inspect.reopenFromAddress=()=>{};
  c.probe.LiveTasks.refreshInspector=()=>{};c.probe.LiveWorkspace.afterPaint=()=>{};
  vm.runInContext('globalThis.lifecyclePaint=renderPage;Portfolio.syncSessionButtons=()=>{};Portfolio.bindCharts=()=>{};Geometry.schedule=()=>{};',c);
  c.render=c.lifecyclePaint;c.patchMain=c.lifecyclePaint;
  c.probe.readRoute();c.render();
  return c;
}
const lifetimeGate=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};};
const readCancelled=()=>Object.assign(Error('Read cancelled'),{name:'AbortError'});


// Actual Data, Review and renderer handoff; only the offline transport is held.
// Recording selected public read calls counts subscribers even when raw GETs join.
function reviewReadContext() {
  const publication='d'.repeat(64),selector={result_hash:H};
  const projection={...reviewBody(),state:'REVIEW_PUBLISHED',review_publication_hash:publication,
    book:{authority:'DEVELOPMENT_RESULT',result_hash:H,formation_session:'2026-08-03',held_count:0}};
  const c=lifecycleContext({hash:'#page=books',realData:true,body:url=>url.startsWith('/api/evidence-cro?')?projection:{}}),p=c.probe;
  c.AbortController=AbortController;p.Data.leavePage(p.app.page);p.Data.visitPage(p.app.page);
  const fetch=c.fetch,read=p.Data.read,gates=new Map(),requests=[],selected=[],downloads=[];
  p.Data.read=(...args)=>{selected.push(args[0]);return read(...args);};
  c.download=(text,name,type)=>downloads.push({text,name,type});
  c.fetch=async(url,init={})=>{
    const held=gates.get(url);if(!held)return fetch(url,init);
    gates.delete(url);c.records.requests.push(url);
    const row={url,signal:init.signal,aborted:false};requests.push(row);held.entered=true;held.request=row;
    const aborted=()=>{row.aborted=true;};init.signal?.addEventListener('abort',aborted,{once:true});
    let value;try{value=await held.promise;}finally{init.signal?.removeEventListener('abort',aborted);}
    return {ok:true,status:200,headers:{get:()=> 'application/json'},text:async()=>JSON.stringify(value)};
  };
  const hold=(url,value)=>{
    assert.ok(!gates.has(url));let release;
    const held={entered:false,promise:new Promise(resolve=>{release=resolve;}),release:()=>release(value)};
    gates.set(url,held);return held;
  };
  return {c,p,selector,projection,publication,hold,requests,selected,downloads,
    projectionURL:'/api/evidence-cro?'+new URLSearchParams(selector),
    exportURL:'/api/evidence-cro/export?'+new URLSearchParams({...selector,review_publication_hash:publication})};
}
async function reviewReadEntered(held) {
  for(let i=0;i<20&&!held.entered;i++)await new Promise(resolve=>setImmediate(resolve));
  assert.ok(held.entered,'the actual Review reader reaches the held owner');
}
async function reviewReadCompleted(promise) {
  let end;promise.then(()=>{end=true;},error=>{end=error;});
  for(let i=0;i<5&&!end;i++)await new Promise(resolve=>setImmediate(resolve));
  assert.equal(end,true,'the cancelled page subscriber completes before its shared raw wire');await promise;
}
async function loadedReport(e) {
  await connect(e.c);await e.p.LiveReview.open(e.selector,'','evidence-stream');
  e.p.LiveReview.step('report');assert.equal(e.p.app.page,'report');
  assert.equal(e.p.LiveReview.facts().state,'REVIEW_PUBLISHED');
}

(async()=>{
  // V676 / TE12: Portfolio and its opened book share one registered reader; Home
  // and Settings consume automation. Change only outside-page owner answers:
  // no Task transition, Reload, explicit reread, navigation or business write.
  for (const site of ['Portfolio','book','Home','Settings']) for (const lang of ['en','zh']) await check('External strategy standing '+site+'/'+lang,async()=>{
    const pkg='RETURN_G6_MU_ONLY', date='2026-08-03';
    let active=false, scheduled=false, failed=false;
    const activation=()=>({status:active?'ACTIVE':'INACTIVE',book_task_id:active?ID:null,activated_at:active?'2026-08-04T12:00:00Z':null,first_forward_session:'2026-08-04',horizon:'2026-09-01',next_requests:active?{deactivate:{operation:'STRATEGY_DEACTIVATE',strategy_package_id:pkg}}:{activate:{operation:'STRATEGY_ACTIVATE',task_id:ID}}});
    const automation=()=>({status:failed?'AUTOMATION_CHECK_FAILED':scheduled?'ENABLED_SERVICE_LIFETIME':'DISABLED',settings:{package_ids:scheduled?[pkg]:[]},next_due_at:scheduled?'2026-08-05T22:00:00Z':null,runs_forward:active?[{...activation(),strategy_package_id:pkg,latest_update:failed?{task_id:'daily-task',lifecycle:'BLOCKED',target_session:date}:null}]:[],next_requests:{}});
    const body=endpoint=>{
      if(endpoint.startsWith('/api/activity'))return {disposition:'CONTINUED',epoch:'same',cursor:'same',items:[],tasks:{},more:false};
      if(endpoint.startsWith('/api/controls'))return {activation:activation()};
      if(endpoint==='/api/research-update/automation')return automation();
      if(endpoint.startsWith('/api/workbench/portfolio'))return {...portfolioBody(),subject:{...portfolioBody().subject,source_kind:'INSTALLED_RESULT',strategy_package_id:pkg,result_hash:H},standing:{activation:active?'ACTIVE':'A_PERSON_MAY_ACTIVATE',execution:'SEALED_RESULT'}};
      return {};
    };
    const c=localizedContext(lang,{realData:true,body});let patches=0;c.patchMain=()=>{patches++;};await connect(c);
    const p=c.probe, bookSite=['Portfolio','book'].includes(site);
    if(bookSite){await p.Data.openPortfolio(ID,site==='book'?date:null);await p.LiveActivation.ensure(p.Data.subject());}
    else {c.setRoute('#page='+(site==='Home'?'overview':'settings'));await p.Settings.rereadUpdate();}
    await p.LiveActivity.refresh();await settle();
    const address=c.location.hash, sealed=p.Data.raw(), beforeTasks=JSON.stringify(p.Data.tasks()), oldPatches=patches;
    const markup=()=>bookSite?String(p.LiveActivation.panel(p.Data.subject())):site==='Home'?String(p.LiveActivation.forwardRows().join('')):String(p.Settings.page());
    const inactiveMarkup=markup();
    if(bookSite)assert.ok(markup().includes('activation-confirm')&&p.Data.raw().standing.activation==='A_PERSON_MAY_ACTIVATE');
    else if(site==='Home')assert.equal(p.LiveActivation.forwardRows().length,0);
    active=true;await p.LiveActivity.refresh();await settle();
    if(bookSite){assert.ok(markup().includes('deactivate')&&!markup().includes('data-value="activate"'),'the external activation replaces the offered action');assert.equal(p.Data.raw().standing.activation,'ACTIVE');assert.equal(p.Data.raw().series,sealed.series,'sealed result not replaced');assert.equal(p.Data.raw().holdings,sealed.holdings);}
    else if(site==='Home')assert.equal(p.LiveActivation.forwardRows().length,1);
    else {assert.equal(p.Settings.dailyUpdate().value.runs_forward[0].strategy_package_id,pkg);assert.notEqual(markup(),inactiveMarkup,'Settings repaints the running strategy');}
    const unscheduledMarkup=markup();scheduled=true;await p.LiveActivity.refresh();await settle();
    if(!bookSite){assert.ok(p.Settings.dailyUpdate().value.next_due_at&&p.Settings.dailyUpdate().value.status==='ENABLED_SERVICE_LIFETIME','external schedule repaints');assert.notEqual(markup(),unscheduledMarkup);assert.ok(markup().includes(vm.runInContext("t('next {t}', {t: whenText('2026-08-05T22:00:00Z')})",c)),'visible next run reads the owner date in '+lang);}
    failed=true;await p.LiveActivity.refresh();await settle();
    if(!bookSite){assert.equal(p.Settings.dailyUpdate().value.status,'AUTOMATION_CHECK_FAILED');assert.equal(p.Settings.dailyUpdate().value.runs_forward[0].latest_update.lifecycle,'BLOCKED');assert.ok(markup().includes(vm.runInContext("t('Its last check failed')",c)),'visible daily standing reads the changed owner state in '+lang);}
    active=false;scheduled=false;failed=false;await p.LiveActivity.refresh();await settle();
    if(bookSite){assert.equal(p.Data.raw().standing.activation,'A_PERSON_MAY_ACTIVATE');assert.ok(markup().includes('data-value="activate"')&&!markup().includes('data-value="deactivate"'));}
    else assert.equal(p.Settings.dailyUpdate().value.runs_forward.length,0);
    assert.equal(JSON.stringify(p.Data.tasks()),beforeTasks,'no Task moved');assert.equal(c.location.hash,address,'same reading address');assert.equal(c.records.reloads,0);assert.equal(c.records.pushes.length,0);assert.ok(patches>oldPatches,'existing in-place patcher used');
    counts.external_strategy_readers=(counts.external_strategy_readers||0)+1;
  });
  await check('Standing exact session and stale selection',async()=>{
    let held=null, delay=false;
    const body=endpoint=>{
      if(!endpoint.startsWith('/api/workbench/portfolio'))return {};
      const query=new URL(endpoint,'http://fixture').searchParams,task=query.get('task_id'),date=query.get('portfolio_session')||'2026-08-03';
      const value={...portfolioBody(),subject:{...portfolioBody().subject,task_id:task,session:date,source_kind:'INSTALLED_RESULT',result_hash:H},standing:{activation:'ACTIVE',date}};
      return delay?new Promise(resolve=>{held=()=>resolve(value);}):value;
    };
    const c=makeContext({realData:true,body}),D=c.probe.Data;c.patchMain=()=>{};
    await D.openPortfolio(ID,'2026-08-03');await D.openPortfolio(ID,'2026-08-02');await D.refreshStanding();
    assert.equal(D.raw().standing.date,'2026-08-02','selected session, not the first loader');
    delay=true;const pending=D.refreshStanding();await settle();delay=false;await D.openPortfolio('other-book','2026-08-01');held();assert.equal(await pending,false);assert.equal(D.subject().task_id,'other-book');assert.equal(D.raw().standing.date,'2026-08-01');
  });
  await check('Superseded activation observation and single flight',async()=>{
    const c=makeContext(),D=c.probe.Data,A=c.probe.LiveActivation;
    let subject={task_id:'book-a',source_kind:'INSTALLED_RESULT',strategy_package_id:'A'},releaseA,releaseB,reads=0,standingReads=0;
    D.ready=true;D.subject=()=>subject;D.refreshStanding=async()=>{standingReads++;return true;};
    D.read=async endpoint=>{reads++;if(reads===1)return new Promise(resolve=>{releaseA=()=>resolve({activation:{status:'INACTIVE'}});});if(reads===2)return new Promise(resolve=>{releaseB=()=>resolve({activation:{status:'ACTIVE',book_task_id:'book-b'}});});return {activation:{status:'ACTIVE',book_task_id:'book-b'}};};
    c.probe.app.page='portfolio';c.patchMain=()=>{};
    const old=A.observe();await settle();await A.observe();assert.equal(reads,1,'cadence joins an unfinished observation');
    subject={task_id:'book-b',source_kind:'INSTALLED_RESULT',strategy_package_id:'B'};const newer=A.ensure(subject,true);await settle();releaseA();await old;
    assert.equal(standingReads,0,'superseded controls cannot read or clear the new book Standing');
    releaseB();await newer;await A.observe();assert.equal(standingReads,1,'new controls still cause the owner Standing read');
  });
  await check('Pending new package never borrows the earlier activation',async()=>{
    const c=makeContext(),D=c.probe.Data,A=c.probe.LiveActivation;
    let subject={task_id:'book-a',source_kind:'INSTALLED_RESULT',strategy_package_id:'A'},release;
    D.ready=true;D.subject=()=>subject;D.refreshStanding=async()=>true;
    D.read=async endpoint=>endpoint.endsWith('=A')?{activation:{status:'ACTIVE',book_task_id:'book-a',next_requests:{deactivate:{operation:'STRATEGY_DEACTIVATE',strategy_package_id:'A'}}}}:new Promise(resolve=>{release=()=>resolve({activation:{status:'INACTIVE',next_requests:{activate:{operation:'STRATEGY_ACTIVATE',task_id:'book-b'}}}});});
    c.probe.app.page='portfolio';c.patchMain=()=>{};await A.ensure(subject);
    subject={task_id:'book-b',source_kind:'INSTALLED_RESULT',strategy_package_id:'B'};const pending=A.observe();await settle();
    assert.equal(String(A.panel(subject)),'','B has no activation or Stop offer until its own controls arrive');
    release();await pending;assert.ok(String(A.panel(subject)).includes('data-value="activate"'));
  });
  const base=makeContext();
  const aliases={advanced:'settings',welcome:'overview','team-members':'team',issues:'data'};
  for(const route of Object.keys(base.probe.ROUTES)) await check('route '+route,()=>{
    const c=makeContext({hash:'#page='+route});c.probe.readRoute();
    assert.equal(c.probe.app.page,aliases[route]||route);assert.ok(c.probe.ROUTES[c.probe.app.page]);
    const address=c.probe.routeUrl(c.probe.app.page,{theme:'dark',lang:'zh',fixture:'A & B=中'});
    const params=new URLSearchParams(address.slice(1));assert.equal(params.get('fixture'),'A & B=中');assert.ok(!params.has('theme')&&!params.has('lang'));counts.routes++;
  });
  for(const route of Object.keys(base.probe.ROUTES))await check('LiveViews route '+route,async()=>{
    const c=makeContext({hash:'#page='+route});c.probe.readRoute();
    if(c.probe.LiveGoals.pages.has(c.probe.app.page))await c.probe.LiveGoals.ensure();
    const markup=String(c.probe.LiveViews.page());nonempty(markup,'LiveViews '+route);assert.ok(!markup.includes('undefined'),'no missing value in '+route);
    assert.ok(!/Not part of this build|This page is not part of this build/.test(markup),'declared route is connected: '+route);counts.rendered_routes++;
  });
  for(const [page,fn] of Object.entries(base.probe.PAGES))await check('registered PAGES '+page,()=>{
    assert.equal(typeof fn,'function',page+' has a page function');assert.ok(base.probe.ROUTES[page],page+' has a route');counts.registered_pages++;
  });
  for(const name of ['LiveGoals','LiveModels','LiveFeatureResearch','LiveTeam','LiveWorkspace','LiveReview','LiveStudy'])for(const page of base.probe[name].pages)await check('registered provider '+name+'/'+page,()=>{
    assert.ok(base.probe.ROUTES[page],name+' page has a route');assert.equal(typeof (name==='LiveTeam'?base.probe[name].section:base.probe[name].page),'function',name+' has its production page function');
  });
  // All declared object addresses are checked from their source table, independently of an individual fixture.
  for(const [page, object] of Object.entries(base.probe.OBJECT_KEYS)) for(const key of object.ids) await check('object '+page+'/'+key,()=>{
    const value=key==='feature'?H+':fixture & factor':key==='goal'?H:key==='case'?ID:'fixture & value=中';
    const c=makeContext({hash:'#'+new URLSearchParams({page,[key]:value})});c.probe.readRoute();
    assert.equal(c.probe.routeObject(),object.kind+':'+value);assert.equal(c.probe.routeObjectId(),value);
    const list=new URLSearchParams(c.probe.listUrl(page).slice(1));for(const cleared of object.clears)assert.ok(!list.has(cleared),page+' clears '+cleared);
    counts.object_addresses++;
  });
  for(const [list,set] of Object.entries(base.probe.PAGE_TABS)) if(set.param) for(const from of set.tabs) for(const to of set.tabs) await check('tabs '+from+' -> '+to,()=>{
    const values=list==='goals'?[H,ID]:[list==='books'?JSON.stringify({task_id:ID,result_hash:H,portfolio_session:'2026-08-03'}):ID];
    for(const value of values){
      const c=makeContext({hash:'#'+new URLSearchParams({page:from,[set.param]:value})});c.probe.readRoute();
      const tabs=String(c.probe.Window.pageTabs());assert.ok(tabs.includes('page-tabs'),'addressed folders have tabs');
      for(const destination of set.tabs)assert.ok(tabs.includes('data-value="'+destination+'"'),'offered tab '+destination);
      c.probe.ACTIONS.go(to);assert.equal(q(c).get(set.param),value,'tab keeps exact object param');assert.equal(q(c).get('page'),to);
      counts.object_tab_pairs++;
    }
  });
  for(const from of base.probe.PAGE_TABS.goals.tabs)for(const to of base.probe.PAGE_TABS.goals.tabs)for(const value of [H,ID])await check('legacy Goal tabs '+from+' -> '+to,async()=>{
    const c=makeContext({hash:'#'+new URLSearchParams({page:from,case:value}),body:goalBody()});c.probe.readRoute();await c.probe.LiveGoals.ensure();
    assert.equal(q(c).get('goal'),value);assert.ok(!q(c).has('case'),'legacy key is canonical before tabs render');
    assert.ok(String(c.probe.Window.pageTabs()).includes('data-value="'+to+'"'));c.probe.ACTIONS.go(to);assert.equal(q(c).get('goal'),value);nonempty(c.probe.LiveViews.page(),to);counts.object_tab_pairs++;
  });
  for(const from of base.probe.PAGE_TABS.books.tabs)for(const to of base.probe.PAGE_TABS.books.tabs)await check('review context tabs '+from+' -> '+to,async()=>{
    const selector=JSON.stringify({result_hash:H}),extra={review_selector:selector,review_publication:H,prepared_task:ID,prepared_unit:'fixture-unit'};
    const c=makeContext({hash:'#'+new URLSearchParams({page:from,...extra}),body:reviewBody()});c.probe.readRoute();c.probe.LiveReview.ensure();await settle();nonempty(c.probe.LiveViews.page(),from);c.probe.ACTIONS.go(to);for(const [key,value] of Object.entries(extra))assert.equal(q(c).get(key),value,'review tabs keep '+key);counts.object_tab_pairs++;
  });
  for(const from of base.probe.PAGE_TABS['team-sessions'].tabs)for(const to of base.probe.PAGE_TABS['team-sessions'].tabs)await check('Team context tabs '+from+' -> '+to,()=>{
    const extra={team:ID,actor:'fixture-actor',event:H},c=makeContext({hash:'#'+new URLSearchParams({page:from,...extra})});c.probe.readRoute();nonempty(c.probe.LiveViews.page(),from);c.probe.ACTIONS.go(to);for(const [key,value] of Object.entries(extra))assert.equal(q(c).get(key),value,'Team tabs keep '+key);counts.object_tab_pairs++;
  });
  // These real owner openers previously combined objectEntry with a second push from navigate.
  const openers=[
    {name:'Goal',page:'goals',key:'goal',id:H,object:'goal',call:(c,id)=>c.probe.LiveGoals.open(id)},
    {name:'Model',page:'models',key:'model',id:'fixture-model',object:'model',call:(c,id)=>c.probe.LiveModels.open(id)},
    {name:'Feature review',page:'feature-research',key:'feature',id:H+':fixture_factor',object:'feature-review',call:(c,id)=>c.probe.LiveFeatureResearch.open(id)},
    {name:'Study',page:'alpha',key:'study',id:ID,object:'study',body:studyBody(),call:(c,id)=>c.probe.PRODUCT_ACTIONS['study-open'](id)},
    {name:'Team',page:'team-sessions',key:'team',id:ID,call:(c,id)=>c.probe.LiveTeam.open(id)},
  ];
  for(const opener of openers)await check('opener '+opener.name,async()=>{
    const c=makeContext({hash:'#page='+opener.page,body:opener.body});c.probe.readRoute();resetHistory(c);await opener.call(c,opener.id);
    assert.equal(c.records.pushes.length,1,'different object creates one history entry');assert.equal(q(c).get(opener.key),opener.id);
    resetHistory(c);await opener.call(c,opener.id);assert.equal(c.records.pushes.length,0,'same object creates no history entry');counts.opener_checks+=2;
  });
  // The row is produced by each collection, its declared handler runs, and the resulting page is read through LiveViews.
  await check('offered Goal row',async()=>{
    const g={goal_id:ID,goal_hash:H,title:'Routing Goal',objective:'Read one exact fixture',kind:'RESEARCH',state:'OPEN',revision:1,reference_count:0,recorded_at:new Date().toISOString()};
    const c=makeContext({hash:'#page=goals',body:endpoint=>endpoint==='/api/goals'?{goals:[g]}:goalBody()});c.probe.readRoute();await c.probe.LiveGoals.ensure();offered(c.probe.LiveViews.page(),'goal-open',H);resetHistory(c);await press(c,'goal-open',H);await c.probe.LiveGoals.ensure();assert.equal(c.records.pushes.length,1);assert.equal(q(c).get('goal'),H);nonempty(c.probe.LiveViews.page(),'offered Goal');counts.offered_rows++;
  });
  // U191: hiding the visual brand never hides the application menu's identity.
  for(const lang of ['en','zh']) for(const mode of ['sidebar','rail']) await check('Named product menu '+mode+'/'+lang,()=>{
    const c=localizedContext(lang), side={innerHTML:'',classList:{toggle(){}},querySelector(){return null;},querySelectorAll(){return [];}};
    const select=c.document.querySelector;c.document.querySelector=selector=>selector==='#side'?side:select(selector);
    vm.runInContext("savePreference('navigation', '"+mode+"');",c);c.probe.Window.renderSide();
    const buttons=markupNodes(side.innerHTML).filter(x=>x.attrs['data-action']==='product-menu');
    assert.equal(buttons.length,1);assert.equal(buttons[0].attrs['aria-label'],'AlphaLattice');
    assert.equal(buttons[0].attrs['aria-haspopup'],'menu');
    counts.named_product_menus=(counts.named_product_menus||0)+1;
  });
  await check('offered Model row',async()=>{
    const model={model_id:'fixture-model',state:'INSTALLED',contract:{findings:[]},identity:{moves:[]}};
    const c=makeContext({hash:'#page=models',body:{models:[model]}});c.probe.readRoute();await c.probe.LiveModels.ensure();offered(c.probe.LiveViews.page(),'model-open',model.model_id);resetHistory(c);await press(c,'model-open',model.model_id);assert.equal(c.records.pushes.length,1);assert.equal(q(c).get('model'),model.model_id);nonempty(c.probe.LiveViews.page(),'offered Model');counts.offered_rows++;
  });
  await check('offered Feature review row',async()=>{
    const x={feature_plan_hash:H,factor_id:'fixture_factor',formula:'fixture formula',state:'NOT_ACTIVE',trials:[],declaration:{formula:'fixture formula',specification:{formula_ref:'fixture-reference',family:'fixture',window_sessions:1,lag_sessions:1}},contract:{status:'PASSED',goldens:[]},identity:{adds:'fixture',moves:[]},active_panel:{admitted:false,reason:'feature_extensions.not_active'},activation_effect:'ADD'};
    const c=makeContext({hash:'#page=feature-research',body:endpoint=>endpoint.includes('feature_factor_id=')?x:{factors:[x],page:1,page_count:1,total:1}});c.probe.readRoute();c.probe.LiveFeatureResearch.ensure();await settle();offered(c.probe.LiveViews.page(),'feature-research-open',H+':fixture_factor');resetHistory(c);await press(c,'feature-research-open',H+':fixture_factor');c.probe.LiveFeatureResearch.ensure();await settle();assert.equal(c.records.pushes.length,1);assert.equal(q(c).get('feature'),H+':fixture_factor');nonempty(c.probe.LiveViews.page(),'offered Feature review');counts.offered_rows++;
  });
  const ordinaryKinds={factor:'factor.screening-development',alpha:'alpha.model-development',risk:'risk.covariance-development','alpha-compare':'alpha.model-development'};
  for(const [page,kind] of Object.entries(ordinaryKinds))await check('offered study row '+page,async()=>{
    const row={task_id:ID,kind,lifecycle:'SUCCEEDED',target_recipe_id:'TARGET',model_parameters:{},factor_ids:[],input_id:'fixture-input',input_binding_hash:H};
    const c=makeContext({hash:'#page='+page,catalog:[row],body:studyBody(kind,'EXPERIMENT_BLOCKED')});c.probe.readRoute();offered(c.probe.LiveViews.page(),'study-open',ID);resetHistory(c);await press(c,'study-open',ID);assert.equal(c.records.pushes.length,1);assert.equal(q(c).get('study'),ID);assert.equal(c.probe.app.page,page);nonempty(c.probe.LiveViews.page(),'offered study '+page);counts.offered_rows++;
  });
  await check('offered Foundation row',async()=>{
    const a={admission_hash:H,factor_task_id:ID,input_id:'fixture-input',input_binding_hash:H,curation_receipt_hash:H,foundation:{foundation_hash:H,ordered_factor_ids:[],limitations:[],execution_outcome:{market_as_of:'2026-08-03'}}};
    const c=makeContext({hash:'#page=foundation',body:endpoint=>endpoint.includes('/readback')?{admission:a}:{foundations:[{admission:a,standing:'CURRENT'}]}});c.probe.readRoute();await c.probe.LiveStudy.foundations();offered(c.probe.LiveViews.page(),'study-foundation-open',H);resetHistory(c);await press(c,'study-foundation-open',H);assert.equal(c.records.pushes.length,1);assert.equal(q(c).get('foundation'),H);nonempty(c.probe.LiveViews.page(),'offered Foundation');counts.offered_rows++;
  });
  await check('offered Team session row',async()=>{
    const item={schema_kind:'ExternalActivityObserved',observation_id:H,ordinal:1,occurred_at:new Date().toISOString(),availability:'AVAILABLE',authority:'ACTOR_DECLARED',source_id:'fixture',source_sequence:1,payload:{event_kind:'NATIVE_COORDINATION_MESSAGE',producer_id:'fixture',producer_session:H,summary:'Routing fixture assignment',subject:{native_session_id:ID,native_agent_id:'fixture-pm',role:'research_lead',input_channel:'ACTOR_DECLARED',message_kind:'assignment'}}};
    const c=makeContext({hash:'#page=team-sessions',body:{items:[item],epoch:'fixture',more:false}});c.probe.readRoute();c.probe.LiveViews.page();await settle();resetHistory(c);followAnchor(c,c.probe.LiveViews.page(),p=>p.get('page')==='team'&&p.get('team')===ID);await settle();assert.equal(c.records.pushes.length,1);assert.equal(q(c).get('team'),ID);nonempty(c.probe.LiveViews.page(),'offered Team');counts.offered_rows++;
  });
  await check('offered Books review row',async()=>{
    const selector={result_hash:H},entry={id:'result:'+H,kind:'INSTALLED_RESULT',name:'Fixture book',words:'Fixture book',raw:{book:selector,book_summary:{state:'NOT_READ'}}};
    const c=makeContext({hash:'#page=books',body:reviewBody(),historyRows:[entry]});
    c.queueMicrotask=fn=>Promise.resolve().then(fn); // let any collection hydration actually start
    c.probe.readRoute();const markup=String(c.probe.LiveViews.page());await settle();
    const fullReads=()=>c.records.requests.filter(p=>p.startsWith('/api/evidence-cro?'));
    assert.equal(fullReads().length,0,'Books discovers its rows without full evidence/CRO reads');
    const href=[...markup.matchAll(/<a\b[^>]*href="([^"]*)"/g)].map(m=>decode(m[1])).find(h=>new URLSearchParams(h.slice(1)).get('review_selector')===JSON.stringify(selector));
    assert.ok(href,'Books offers exact review selector');c.history.pushState({alpha:1},'',href);c.probe.readRoute();c.probe.LiveReview.ensure();await settle();
    assert.equal(fullReads().length,1,'choosing one Books row verifies that exact selector once');
    assert.equal(new URLSearchParams(fullReads()[0].split('?')[1]).get('result_hash'),H);
    assert.equal(q(c).get('review_selector'),JSON.stringify(selector));assert.equal(c.probe.app.page,'evidence');nonempty(c.probe.LiveViews.page(),'offered Review');counts.offered_rows++;
  });
  for(const page of ['portfolio','compare','history'])await check('offered Portfolio row '+page,async()=>{
    const raw={entry_id:'experiment:'+ID,task_id:ID,kind:'portfolio.policy-development',status:'SUCCEEDED',recorded_at:new Date().toISOString(),input_id:'fixture-input',input_binding_hash:H,book:{experiment_task_id:ID,experiment_receipt_hash:H,portfolio_session:'2026-08-03'}};
    const c=makeContext({hash:'#page='+page,realData:true,rawHistory:[raw],body:portfolioBody()});await connect(c);offered(c.probe.LiveViews.page(),'history-open',raw.entry_id);resetHistory(c);await press(c,'history-open',raw.entry_id);assert.equal(c.records.pushes.length,1);assert.equal(q(c).get('book'),ID);assert.equal(c.probe.app.page,page==='compare'?'compare':'portfolio');assert.ok(c.probe.Data.ready,c.probe.Data.error);nonempty(c.probe.LiveViews.page(),'offered Portfolio '+page);counts.offered_rows++;
  });
  // U195 / P3b: discover installed books and authored studies through the real Data
  // collection, with failures and unrelated records as negative controls. No per-book
  // read is needed to list them; pressing a row reads its exact owner Task/date once.
  for(const lang of ['en','zh']) for(const page of ['portfolio','compare']) await check('Portfolio eligible collection '+page+'/'+lang,async()=>{
    const kinds=['portfolio.policy-development','INSTALLED_RESULT','portfolio.policy-development','INSTALLED_RESULT','CRO_REVIEW','unknown-future-kind'];
    const rawHistory=kinds.map((kind,i)=>({entry_id:'portfolio-entry-'+i,task_id:'portfolio-task-'+i,kind,status:i===2||i===3?'FAILED':'SUCCEEDED',recorded_at:new Date().toISOString(),input_id:'fixture-input',input_binding_hash:H,book:{portfolio_session:'2026-08-0'+(i+1),...(kind==='INSTALLED_RESULT'?{result_hash:H,strategy_package_id:'RETURN_G6_MU_ONLY'}:{experiment_receipt_hash:H})}}));
    const body=url=>{
      if(!url.startsWith('/api/workbench/portfolio?'))return {};
      const params=new URLSearchParams(url.split('?')[1]), entry=rawHistory.find(x=>x.task_id===params.get('task_id'));
      return {...portfolioBody(),subject:{...portfolioBody().subject,task_id:entry.task_id,session:entry.book.portfolio_session,source_kind:entry.kind==='INSTALLED_RESULT'?'INSTALLED_RESULT':'DEVELOPMENT_RESULT',result_hash:H}};
    };
    const c=localizedContext(lang,{hash:'#page='+page,realData:true,rawHistory,body});await connect(c);
    const eligible=c.probe.Data.portfolioEntries(), markup=String(c.probe.LiveViews.page());
    assert.equal(eligible.length,2,'successful installed and authored kinds are the owner collection');
    const offeredIds=markupNodes(markup).filter(x=>x.attrs['data-action']==='history-open').map(x=>x.attrs['data-value']).sort();
    assert.deepEqual(offeredIds,Array.from(eligible,x=>x.entry_id).sort(),'every eligible object and no failed/unrelated object is offered');
    const tabs=String(c.probe.Window.pageTabs()), portfolioTab=/<button\b[^>]*data-value="portfolio"[^>]*>([\s\S]*?)<\/button>/.exec(tabs);
    assert.ok(portfolioTab && portfolioTab[1].includes('>2<'),'existing Portfolio tab counts the same collection in '+lang);
    assert.equal(c.records.requests.filter(url=>url.startsWith('/api/workbench/portfolio?')).length,0,'listing reads no individual book');
    for(const entry of eligible){
      c.setRoute('#page='+page);resetHistory(c);const before=c.records.requests.length;
      await press(c,'history-open',entry.entry_id);
      const reads=c.records.requests.slice(before).filter(url=>url.startsWith('/api/workbench/portfolio?'));
      assert.equal(reads.length,1,'one owner read per offered book');
      const params=new URLSearchParams(reads[0].split('?')[1]);
      assert.equal(params.get('task_id'),entry.task_id);assert.equal(params.get('portfolio_session'),entry.book.portfolio_session);
      assert.equal(q(c).get('book'),entry.task_id);assert.equal(q(c).get('session'),entry.book.portfolio_session);
      assert.equal(c.probe.app.page,page);assert.equal(c.probe.Data.subject().result_hash,H);assert.ok(c.probe.Data.ready,c.probe.Data.error);
      assert.equal(c.records.pushes.length,1);nonempty(c.probe.LiveViews.page(),'eligible Portfolio readback');counts.offered_rows++;
    }
  });
  // Real Goal readback and LiveViews dispatch on every Goal folder, both route identity forms.
  // U189 / P3a: the last canonical Task mutation reads in the person's zone.
  // Missing and legacy clocks stay blank; a request/Goal clock cannot fill them.
  const previousTimezone=process.env.TZ;
  try {
    process.env.TZ='America/New_York';
    for(const lang of ['en','zh']) await check('Goal Task authoritative clock '+lang,async()=>{
      const body=goalBody();body.record.tasks=[
        {task_id:'timed-task',kind:'research_experiment',state:'SUCCEEDED',updated_at:'2024-10-06T00:30:00+00:00'},
        {task_id:'legacy-task',kind:'research_experiment',state:'SUCCEEDED'},
        {task_id:'absent-task',kind:'research_experiment',state:'BLOCKED',updated_at:null},
      ];
      const c=localizedContext(lang,{hash:'#page=goal&goal='+H,body});await c.probe.LiveGoals.ensure();
      const markup=String(c.probe.LiveGoals.page()), label=lang==='zh'?'任务记录更新时间':'Task record updated';
      const local=lang==='zh'?'2024年10月5日 20:30':'Oct 5, 2024 20:30';
      assert.ok(markup.includes(local),'actual shared clock converts the owner instant to the reader day/time');
      const clocks=markupNodes(markup).filter(x=>x.attrs['data-tip']?.startsWith(label+' · '));
      assert.equal(clocks.length,1,'no clock is fabricated for absent or legacy Task facts');
      assert.equal(clocks[0].attrs['data-tip'],label+' · 2024-10-06 00:30:00 UTC','the shared hint keeps the exact owner clock and meaning');
      assert.ok(ancestors(clocks[0]).some(x=>x.attrs['data-key']==='timed-task'),'the clock stays on its Task row');
      for(const id of ['legacy-task','absent-task'])offered(markup,'task',id);
      counts.goal_task_clocks=(counts.goal_task_clocks||0)+1;
    });
  } finally { if(previousTimezone===undefined)delete process.env.TZ;else process.env.TZ=previousTimezone; }
  for(const value of [H,ID])for(const page of base.probe.PAGE_TABS.goals.tabs)await check('Goal page '+page+'/'+(value===H?'hash':'id'),async()=>{
    const c=makeContext({hash:'#'+new URLSearchParams({page,goal:value}),body:goalBody()});c.probe.readRoute();await c.probe.LiveGoals.ensure();
    const markup=String(c.probe.LiveViews.page());nonempty(markup,page);assert.ok(markup.includes('page-tabs'),'Goal page keeps folders');
    if(page!=='goal'){
      const line=page==='goal-conversation'?'No Team exchange under this goal yet':'Not submitted yet: an agent submits it through the CLI';
      const empty=[...markup.matchAll(/<div class="section-empty[^"]*" data-empty="([^"]*)"[^>]*>([\s\S]*?)<\/div>/g)].find(row=>row[2].includes(line));
      assert.ok(empty,'empty Goal folder explains its state');assert.equal(empty[1],'elsewhere','the folder observes work done elsewhere');
      const way=offered(empty[2],'go','goal');assert.match(way,/class="[^"]*\btext-btn\b/,'empty folder offers a quiet way to Timeline');
      await press(c,'go','goal');assert.equal(q(c).get('goal'),value);assert.equal(c.probe.app.page,'goal');
    }
  });
  for(const page of base.probe.PAGE_TABS.goals.tabs)await check('bare Goal '+page,async()=>{
    const c=makeContext({hash:'#page='+page});c.probe.readRoute();await c.probe.LiveGoals.ensure();
    nonempty(c.probe.LiveViews.page(),page+' with no Goal');assert.equal(c.probe.app.page,'goals');
  });
  // Catalogue shape is the owner listing contract; exact candidate compatibility is deliberately not computed here.
  const catalog=[
    {task_id:'eligible',kind:'alpha.model-development',lifecycle:'SUCCEEDED',target_recipe_id:'TARGET',model_parameters:{}},
    {task_id:'lifecycle',kind:'alpha.model-development',lifecycle:'SUCCEEDED',component_recipe_id:'fixture-component'},
    {task_id:'qualification',kind:'alpha.model-development',lifecycle:'SUCCEEDED'},
    {task_id:'running',kind:'alpha.model-development',lifecycle:'RUNNING',target_recipe_id:'TARGET',model_parameters:{}},
    {task_id:'failed',kind:'alpha.model-development',lifecycle:'FAILED',target_recipe_id:'TARGET',model_parameters:{}},
  ];
  await check('Compare collection offered rows',()=>{
    const c=makeContext({hash:'#page=alpha-compare',catalog});c.probe.readRoute();const markup=String(c.probe.LiveViews.page());
    assert.ok(markup.includes('data-value="eligible"'));for(const row of catalog.slice(1))assert.ok(!markup.includes('data-value="'+row.task_id+'"'),'Compare does not offer '+row.task_id);
  });
  for(const shape of ['development','lifecycle','qualification'])for(const status of ['EXPERIMENT_PUBLISHED','EXPERIMENT_SUMMARY','EXPERIMENT_BLOCKED'])await check('Compare '+shape+'/'+status,async()=>{
    const c=makeContext({hash:'#'+new URLSearchParams({page:'alpha-compare',study:ID}),body:studyBody('alpha.model-development',status,shape),catalog});c.probe.readRoute();
    await c.probe.LiveStudy.open(ID,'alpha-compare');const markup=String(c.probe.LiveViews.page());nonempty(markup,shape+'/'+status);
    assert.ok(!markup.includes('undefined'),'renderer does not emit undefined');
    assert.ok(!markup.includes('data-render-failure'),'ordinary readback/refusal, not drawing failure');
    if(shape!=='development'||status!=='EXPERIMENT_PUBLISHED'){assert.ok(markup.includes('workbench.alpha_comparison_unsupported_study'),'typed unsupported Compare code');assert.ok(markup.includes('Open ordinary study'),'ordinary readback way');assert.ok(markup.includes('page=alpha')&&markup.includes('study='+ID),'ordinary way keeps exact Task');assert.ok(!/<p class="refusal-next"[^>]*>[\s\S]*?Re-PLAN/.test(markup),'ordinary readback is the offered recovery');}
    else{assert.ok(markup.includes('studyAlphaComparisonTask'),'ready candidate study renders Compare picker');assert.ok(markup.includes('data-value="eligible"'),'right selector offers same completed development row');for(const row of catalog.slice(1))assert.ok(!markup.includes('data-value="'+row.task_id+'"'),'right selector excludes '+row.task_id);}
    counts.study_shapes++;
  });
  // Enumerate normal renderer kinds from the production page-kind table, not one known Alpha instance.
  const kinds=vm.runInNewContext('('+/const kinds=(\{[^;]+\});/.exec(source('live-study.js'))[1]+')');
  for(const page of Object.keys(kinds))for(const actual of Object.keys(kinds).filter(p=>p!=='alpha-compare'))for(const status of ['EXPERIMENT_PUBLISHED','EXPERIMENT_SUMMARY','EXPERIMENT_BLOCKED'])await check('renderer '+page+'/'+actual+'/'+status,async()=>{
    const kind=kinds[actual],body=studyBody(kind,status);
    const c=makeContext({hash:'#'+new URLSearchParams({page,study:ID}),body:endpoint=>endpoint.includes('curation')?{choices:[],inputs:[],limitations:[],decisions:[]}:body});c.probe.readRoute();await c.probe.LiveStudy.open(ID,page);
    assert.equal(c.probe.app.page,page==='alpha-compare'?page:actual,'wrong-kind ordinary address follows its actual reader');const markup=String(c.probe.LiveViews.page());nonempty(markup,'declared renderer '+page);assert.ok(!markup.includes('undefined'),'renderer has no missing value');
    if(page==='alpha-compare'&&(actual!=='alpha'||status!=='EXPERIMENT_PUBLISHED')){assert.ok(markup.includes('workbench.alpha_comparison_unsupported_study'));assert.ok(markup.includes('page='+actual)&&markup.includes('study='+ID),'ordinary way names actual kind');}counts.renderer_shapes++;
  });
  // Fresh documents retry the exact route; the real Reload handler never depends on discarded VM memory.
  const failedHash='#page=alpha-compare&study='+ID;
  for(let documentNumber=0;documentNumber<2;documentNumber++)await check('recovery VM '+documentNumber,()=>{
    const c=makeContext({hash:failedHash});c.probe.readRoute();c.probe.renderFailure(new Error('fixture persistent failure'),'route class');
    const card=String(c.main.innerHTML);assert.ok(card.includes('data-action="page-reload"'));assert.ok(card.includes('page=overview'),'explicit Home recovery');
    assert.ok(/Reload retries this page/.test(card));assert.ok(!/second failure|drawn twice/.test(card),'no promise based on lost memory');
    c.probe.ACTIONS['page-reload']();assert.equal(c.records.reloads,1);assert.equal(c.location.hash,failedHash);
    c.probe.ACTIONS.go('overview');assert.equal(c.probe.app.page,'overview');assert.equal(q(c).get('page'),'overview');counts.recovery_documents++;
  });
  // The right operand may come from a retained exact route even when today's picker excludes it.
  const RHS_ID='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', GOOD_RHS_ID='99999999-8888-7777-6666-555555555555';
  const LEFT_CANDIDATE='fixture-left', RHS_CANDIDATE='fixture-right', OLD_CANDIDATE='fixture-old-right';
  const rhsCases=[];
  for(const shape of ['development','lifecycle','qualification'])for(const status of ['EXPERIMENT_PUBLISHED','EXPERIMENT_SUMMARY','EXPERIMENT_BLOCKED'])if(shape!=='development'||status!=='EXPERIMENT_PUBLISHED')rhsCases.push({name:shape+'/'+status,kind:'alpha.model-development',shape,status});
  for(const kind of ['factor.screening-development','risk.covariance-development','portfolio.policy-development','fixture.unsupported-kind'])rhsCases.push({name:'wrong-kind/'+kind,kind,shape:'development',status:'EXPERIMENT_PUBLISHED'});
  const candidateButton=markup=>[...String(markup).matchAll(/<button\b[^>]*>/g)].map(m=>m[0]).find(tag=>tag.includes('data-action="study-alpha-comparison-run"'));
  const rightOptions=markup=>{const start=String(markup).indexOf('id="studyAlphaComparisonCandidateList"'),end=String(markup).indexOf('<p class="menu-empty"',start);assert.ok(start>=0&&end>start,'right picker list exists');return String(markup).slice(start,end);};
  const heldButton=markup=>{const tag=candidateButton(markup);assert.ok(tag,'Compare action exists');assert.match(tag,/\bdisabled\b|aria-disabled="true"/,'unadmitted operand holds Compare');};
  const noReplan=markup=>{const lines=[...String(markup).matchAll(/<p class="refusal-next"[^>]*>([\s\S]*?)<\/p>/g)].map(m=>decode(m[1].replace(/<[^>]+>/g,'')));assert.ok(!lines.some(line=>/re[-\u2011\u2013 ]?plan/i.test(line)),'ordinary readback refusal never advises Re-PLAN');};
  function rhsContext(test,{retained=true,leftCandidate=LEFT_CANDIDATE,rightCandidate=OLD_CANDIDATE}={}){
    const left=studyBody();left.result.candidates=[{candidate_id:LEFT_CANDIDATE,status:'DEVELOPMENT_EVALUATED'}];
    const good=studyBody();good.task_id=GOOD_RHS_ID;good.result.candidates=[{candidate_id:RHS_CANDIDATE,status:'DEVELOPMENT_EVALUATED'}];
    const bad=studyBody(test.kind,test.status,test.shape);bad.task_id=RHS_ID;bad.result.candidates=[{candidate_id:RHS_CANDIDATE,status:'DEVELOPMENT_EVALUATED'}];
    const route={page:'alpha-compare',study:ID,alpha_left_task:ID,alpha_left_candidate:leftCandidate,alpha_right_task:retained?RHS_ID:GOOD_RHS_ID,alpha_right_candidate:retained?rightCandidate:RHS_CANDIDATE};
    const c=makeContext({hash:'#'+new URLSearchParams(route),body:endpoint=>{
      if(endpoint.includes('/alpha-compare?'))return {status:'COMPARABLE',disposition:'DESCRIPTIVE_ALPHA_COMPARISON_NO_SELECTION',left:{task_id:ID,candidate_id:LEFT_CANDIDATE},right:{task_id:RHS_ID,candidate_id:RHS_CANDIDATE},folds:[],limitations:[]};
      if(endpoint.includes('curation'))return {choices:[],inputs:[],limitations:[],decisions:[]};
      const task=new URLSearchParams(endpoint.split('?')[1]).get('task_id');return task===RHS_ID?bad:task===GOOD_RHS_ID?good:left;
    }});c.probe.readRoute();return c;
  }
  for(const test of rhsCases)for(const mode of ['retained-link','choose-after-valid']){
    const name='RHS '+mode+'/'+test.name,c=rhsContext(test,{retained:mode==='retained-link'});await c.probe.LiveStudy.open(ID,'alpha-compare');await settle();
    if(mode==='choose-after-valid'){assert.ok(rightOptions(c.probe.LiveViews.page()).includes('data-value="'+RHS_CANDIDATE+'"'),'healthy RHS was rendered first');await c.probe.LiveStudy.alphaComparisonTask(RHS_ID);await settle();}
    const markup=String(c.probe.LiveViews.page());
    await check(name+'/typed refusal',()=>{assert.ok(markup.includes('workbench.alpha_comparison_unsupported_study'));noReplan(markup);assert.ok(!rightOptions(markup).includes('data-value="'+RHS_CANDIDATE+'"'),'no old RHS candidate options');assert.ok(!markup.includes('Descriptive saved evidence'),'no previous comparison result');});
    await check(name+'/held button',()=>heldButton(markup));
    await check(name+'/run guard',async()=>{const before=c.records.requests.filter(p=>p.startsWith('/api/experiments/alpha-compare?')).length;await c.probe.LiveStudy.alphaComparisonRun();await settle();assert.equal(c.records.requests.filter(p=>p.startsWith('/api/experiments/alpha-compare?')).length,before,'unsupported RHS never asks owner to compare');});
    await check(name+'/ordinary way',async()=>{
      const taskWay=[...markup.matchAll(/<button\b[^>]*>/g)].map(m=>m[0]).some(tag=>/data-action="task"/.test(tag)&&decode(/data-value="([^"]*)"/.exec(tag)?.[1])===RHS_ID);
      if(taskWay)return;
      const ordinary=[...markup.matchAll(/<a\b[^>]*href="([^"]*)"/g)].map(m=>decode(m[1])).find(h=>{const p=new URLSearchParams(h.slice(1));return p.get('study')===RHS_ID&&['factor','alpha','risk'].includes(p.get('page'));});
      assert.ok(ordinary,'unsupported RHS has an exact Task or ordinary-study way');
      const expected={'factor.screening-development':'factor','alpha.model-development':'alpha','risk.covariance-development':'risk'}[test.kind];assert.ok(expected,'a kind without ordinary reader must offer exact Task inspection');
      const follow=rhsContext(test);follow.setRoute(ordinary);await follow.probe.LiveStudy.open(RHS_ID,q(follow).get('page'));await settle();assert.equal(follow.probe.app.page,expected,'ordinary way reaches actual owner kind');assert.equal(q(follow).get('study'),RHS_ID);nonempty(follow.probe.LiveViews.page(),'RHS ordinary readback');
    });
    counts.rhs_cases=(counts.rhs_cases||0)+1;
  }
  const valid={name:'development/EXPERIMENT_PUBLISHED',kind:'alpha.model-development',shape:'development',status:'EXPERIMENT_PUBLISHED'};
  await check('RHS valid positive',async()=>{const c=rhsContext(valid,{rightCandidate:RHS_CANDIDATE});await c.probe.LiveStudy.open(ID,'alpha-compare');await settle();const tag=candidateButton(c.probe.LiveViews.page());assert.ok(tag&&!/\bdisabled\b|aria-disabled="true"/.test(tag),'valid exact pair can compare');await c.probe.LiveStudy.alphaComparisonRun();assert.equal(c.records.requests.filter(p=>p.startsWith('/api/experiments/alpha-compare?')).length,1);counts.rhs_cases=(counts.rhs_cases||0)+1;});
  for(const side of ['left','right']){
    const c=rhsContext(valid,{leftCandidate:side==='left'?'fixture-missing-left':LEFT_CANDIDATE,rightCandidate:side==='right'?OLD_CANDIDATE:RHS_CANDIDATE});await c.probe.LiveStudy.open(ID,'alpha-compare');await settle();const markup=String(c.probe.LiveViews.page());
    await check('RHS missing '+side+' candidate/held button',()=>heldButton(markup));
    await check('RHS missing '+side+' candidate/run guard',async()=>{await c.probe.LiveStudy.alphaComparisonRun();assert.equal(c.records.requests.filter(p=>p.startsWith('/api/experiments/alpha-compare?')).length,0,'an absent candidate is never offered for comparison');});counts.rhs_cases=(counts.rhs_cases||0)+1;
  }

  // U121 (V664): every composed identity in these collection refusals uses the shared
  // full-hover/copy cell. Scripted owner answers exercise production readers and builders;
  // the real QA scene separately proves the service, browser hover and copy-control journey.
  await check('refusal locator ancestry control',()=>{
    const cell='<span class="run-ref"><span data-tip="full-id">full…</span><button data-action="copy-text" data-value="full-id" aria-label="Copy"></button></span>';
    assertLocators('<div><button class="list-row-main">Open</button>'+cell+'</div>',['full-id'],'sibling control');
    for(const tag of ['button','a'])assert.throws(()=>assertLocators('<'+tag+' class="list-row-main">'+cell+'</'+tag+'>',['full-id'],'planted nesting'),/outside the clickable row/);
    assert.throws(()=>assertLocators(cell.replace('data-value="full-id"','data-value="wrong-id"'),['full-id'],'planted wrong copy'),/full identity/);
    assert.throws(()=>assertLocators(cell.replace('data-tip="full-id"','data-tip="prefix"'),['full-id'],'planted short hover'),/hover holds/);
  });
  const missingTask='66400000-0000-4000-8000-000000000001',admissionHash='6'.repeat(64),unreadableHash='7'.repeat(64);
  const inputHash='8'.repeat(64),proseId='9'.repeat(64),ownerDetail='The owner retains this exact sentence, including '+proseId+'.';
  const recovery={workspace:{operation:'WORKSPACE_SHOW'},backups:{operation:'WORKSPACE_BACKUPS'}};
  // The unknown fixture code exercises retained prose; known codes use their owner catalogue.
  const refused=(extra)=>({status:'REFUSED',failure_code:'fixture.collection_refused',detail:ownerDetail,next_requests:recovery,...extra});
  for(const lang of ['en','zh']) {
    await check('Foundation refused identities '+lang,async()=>{
      const admission={admission_hash:admissionHash,input_id:'factor-development',input_binding_hash:inputHash,foundation:{ordered_factor_ids:['factor664'],execution_outcome:{market_as_of:'2026-08-03'}}};
      const start={start_factor:{operation:'EXPERIMENT_CONTROLS',experiment_kind:'factor.screening-development',research_input_id:'factor-development',input_binding_hash:inputHash}};
      const withSources=refused({failure_code:'task_control.task_not_found',standing:'REFUSED',admission,foundation_admission_hash:admissionHash,missing_factor_task_id:missingTask,next_requests:start});
      const withoutSources={...withSources,standing:'HISTORICAL',foundation_admission_hash:unreadableHash};delete withoutSources.admission;
      const c=localizedContext(lang,{hash:'#page=foundation',body:endpoint=>{
        if(endpoint==='/api/experiments/foundations')return {foundations:[withSources,withoutSources]};
        const hash=new URLSearchParams(endpoint.split('?')[1]).get('foundation_admission_hash');
        const body=hash===admissionHash?withSources:withoutSources,error=new Error(body.failure_code);error.body=body;throw error;
      }});
      await c.probe.LiveStudy.foundations();let markup=String(c.probe.LiveViews.page());
      const fold=offered(markup,'lobby-fold','foundation:none');
      assert.match(fold,/aria-expanded="false"/,'the admission without source metadata starts in the folded input group');
      await press(c,'lobby-fold','foundation:none');markup=String(c.probe.LiveViews.page());
      assert.notEqual(missingTask,admissionHash,'missing Task is a different identity from the admission');
      for(const hash of [admissionHash,unreadableHash]) {
        assertLocators(markup,[hash,missingTask],lang+' Foundation row '+hash,{row:hash});
        offered(markup,'study-foundation-open',hash);
      }
      for(const hash of [admissionHash,unreadableHash]) {
        await c.probe.LiveStudy.foundations(hash);markup=String(c.probe.LiveViews.page());
        assertLocators(markup,[missingTask],lang+' Foundation detail '+hash);
        const ways=[...markup.matchAll(/<a\b[^>]*href="([^"]*)"/g)].map(m=>new URLSearchParams(decode(m[1]).slice(1)));
        assert.ok(ways.some(p=>p.get('page')==='lab'&&p.get('experiment_kind')==='factor.screening-development'&&p.get('research_input')==='factor-development'&&p.get('input_binding')===inputHash),'the exact Factor re-plan route remains usable');
        assert.ok(!/data-action="study-foundation-draft"/.test(markup),'refused admission offers no Alpha draft');
      }
      counts.refusal_locator_cases=(counts.refusal_locator_cases||0)+1;
    });
    await check('History refused locator fields '+lang,()=>{
      const rows=[refused({task_id:missingTask}),refused({entry_id:'experiment:'+missingTask}),refused({result_hash:admissionHash}),refused({index_file:'D:/qa/history/unreadable-record.json'})];
      const c=localizedContext(lang,{hash:'#page=history'});c.probe.Data.historyRefusals=()=>rows;
      const markup=String(c.probe.LiveViews.page());assertLocators(markup,rows.map(r=>r.task_id||r.entry_id||r.result_hash||r.index_file),lang+' History');
      assert.ok(markup.includes(ownerDetail),'owner prose is retained verbatim');
      assert.ok(!markupNodes(markup).some(n=>n.attrs['data-action']==='copy-text'&&n.attrs['data-value']===proseId),'owner prose is not parsed into a composed identity');
      counts.refusal_locator_cases++;
    });
    await check('Input and backup refused locator fields '+lang,async()=>{
      const publicationHash='b'.repeat(64),inputId='retained-input-v664',generationHash='c'.repeat(64),generationFile='D:/qa/backup/generations/unreadable-generation.json';
      const c=localizedContext(lang,{hash:'#page=inputs',body:endpoint=>{
        if(endpoint==='/api/research-inputs')return {inputs:[],refusals:[refused({publication_hash:publicationHash}),refused({input_id:inputId})]};
        if(endpoint==='/api/workspace/storage')return {capacity_status:'CAPACITY_AVAILABLE',by_root:{},versions:[]};
        if(endpoint==='/api/workspace/backup')return {backup_root:'D:/qa/backup',generations:[],refusals:[refused({generation_hash:generationHash}),refused({generation_file:generationFile})]};
        throw Error('unexpected locator fixture read '+endpoint);
      }});
      await c.probe.LiveWorkspace.refresh('inputs');assertLocators(c.probe.LiveViews.page(),[publicationHash,inputId],lang+' Inputs');
      c.setRoute('#page=storage');await c.probe.LiveWorkspace.refresh('storage');await settle();assertLocators(c.probe.LiveViews.page(),[generationHash,generationFile],lang+' Backups');
      counts.refusal_locator_cases+=2;
    });
    await check('Feature and Model refused locators '+lang,async()=>{
      const trialId='d'.repeat(64),planId='e'.repeat(64),modelId='declared-model-v664';
      const c=localizedContext(lang,{hash:'#page=feature-research',body:endpoint=>{
        if(endpoint.startsWith(c.probe.Data.route('FEATURE_EXTENSIONS')))return {factors:[],total:0,page:1,page_count:1,damaged:[refused({feature_trial_id:trialId})],refused_plans:[refused({feature_plan_hash:planId})]};
        if(endpoint==='/api/models')return {models:[refused({model_id:modelId})]};
        throw Error('unexpected locator fixture read '+endpoint);
      }});
      c.probe.LiveFeatureResearch.ensure();await settle();assertLocators(c.probe.LiveViews.page(),[trialId,planId],lang+' Features');
      c.setRoute('#page=models');await c.probe.LiveModels.ensure();const markup=String(c.probe.LiveViews.page());
      assertLocators(markup,[modelId],lang+' Models',{row:modelId});offered(markup,'model-open',modelId);
      counts.refusal_locator_cases+=2;
    });
    await check('Each refused packet ledger locator '+lang,async()=>{
      const unit='recorded-unit-v664',group='retained-group-v664';
      const view={...reviewBody(),state:'ANALYST_PACKET_PREPARED',next_requests:{packet:{operation:'EVIDENCE_PACKET',task_id:missingTask,evidence_unit_id:unit,result_hash:H}}};
      const c=localizedContext(lang,{hash:'#page=evidence-reading',body:endpoint=>{
        if(endpoint.startsWith('/api/evidence-cro?'))return view;
        if(endpoint.startsWith(c.probe.Data.route('EVIDENCE_LEDGER')+'?'))return {status:'EVIDENCE_BOOK_LEDGER',groups:[{group_id:group,packet_task_id:missingTask,packet_unit_id:unit,refusal:'alternative_evidence.topic_coverage_absent:the recorded selection has no ledger'}],next_request:null};
        throw Error('unexpected locator fixture read '+endpoint);
      }});
      await c.probe.LiveReview.open({result_hash:H},'','evidence-reading');await c.probe.LiveReview.readLedgers();
      assertLocators(c.probe.LiveViews.page(),[missingTask,unit],lang+' per-packet ledger',{row:'ledger:'+missingTask+'|'+unit});
      counts.refusal_locator_cases++;
    });
    await check('Refused Activity and Team current Task locators '+lang,()=>{
      const session='session-v664',authored='The participant keeps this English statement.',artifactHash='f'.repeat(64);
      const fields=['spec_hash','plan_hash','experiment_plan_hash','result_hash','cached_result_hash','publication_task_id','research_input_id','strategy_package_id','candidate_hash','review_publication_hash'];
      const subject=Object.fromEntries(fields.map((key,i)=>[key,key==='publication_task_id'?'66400000-0000-4000-8000-000000000002':key==='research_input_id'?'retained-input-v664':i.toString(16).repeat(64)]));
      const observation=(ordinal,schema_kind,payload,extra={})=>({ordinal,observation_id:'locator-observation-'+ordinal,occurred_at:'2026-08-03T12:00:00Z',observed_at:'2026-08-03T12:00:00Z',schema_kind,source_kind:'PRODUCT_OPERATION',source_id:'local-web:fixture',source_sequence:ordinal,task_id:null,run_id:null,stage_id:null,correlation_ids:[],authority:'OPERATIONAL_ASSERTION',retention_class:'TRANSIENT_OPERATIONAL',availability:'AVAILABLE',payload,...extra});
      const message=observation(1,'ExternalActivityObserved',{event_kind:'NATIVE_COORDINATION_MESSAGE',producer_id:'codex-native',producer_session:'locator-scope',producer_sequence:1,summary:authored,summary_truncated:false,task_verified:false,subject:{native_session_id:session,native_agent_id:session,role:'research_lead',source_time_kind:'BRIDGE_RECEIVED',native_event_id:'locator-native-event',message_kind:'answer',message_id:'locator-message',message_sha256:'ab'.repeat(32),message_bytes:String(Buffer.byteLength(authored,'utf8')),input_channel:'ACTOR_DECLARED',reference:missingTask}},{source_kind:'EXTERNAL_CLIENT',authority:'AGENT_PROPOSAL'});
      const operation=observation(2,'ProductOperationObserved',{operation:'EXPERIMENT_RUN',caller:'EXTERNAL_AUTOMATION',phase:'RETURNED',operation_ref:'locator-operation',status:'ADMITTED',subject,task_id:missingTask},{task_id:missingTask});
      const artifact=observation(3,'ArtifactVerificationObserved',{artifact_kind:'ResearchExecutionEvidence',artifact_hash:artifactHash,availability:'AVAILABLE'},{task_id:missingTask,source_kind:'PRODUCT_ARTIFACT',authority:'ARTIFACT_ASSERTION'});
      const c=localizedContext(lang,{hash:'#page=team-evidence&team='+session});
      c.probe.LiveActivity.absorbPage({workspace_id:'qa',disposition:'TAIL',epoch:'locator-e1',cursor:'locator-e1:3',head:3,more:false,items:[message,operation,artifact],unavailable:0,tasks:{[missingTask]:refused({task_id:missingTask})},observer:{status:'OK'},read_cost:{observations:3,elapsed_ms:1}});
      const activity=String(c.probe.LiveActivity.section());assertLocators(activity,[missingTask,...Object.values(subject),artifactHash],lang+' refused Activity');
      const team=String(c.probe.LiveTeam.section()),current=team.slice(team.indexOf('class="team-current-states"'));
      assert.ok(team.includes('id="teamProductFacts"'),'real Team renderer associates the recorded product observations');
      assertLocators(current,[missingTask],lang+' Team current refused Task');
      c.setRoute('#page=team&team='+session);assert.ok(String(c.probe.LiveTeam.section()).includes(authored),'participant statement keeps its authored language');
      counts.refusal_locator_cases+=2;
    });
  }

  // V667 / TE12: every registered addressed reader survives a fresh document and
  // the real connection's completion. A newly registered mode needs its own fixture.
  const inspectorFixtures = {
    facts: {page:'goal-results',goal:H,facts:'1'},
    record: {page:'tasks',task:ID,record:'1'},
    task: {page:'tasks',task:ID},
    reference: {page:'goal-results',goal:H,reference:'fixture-reference'},
  };
  const referenceReading = (operation='REPORT',stage='PORTFOLIO',state='VERIFIED_READBACK') => ({reference:{reference_id:'fixture-reference',label:'Own referenced object',stage,request:{operation,result_hash:'b'.repeat(64),task_id:ID},identity:{report_hash:'c'.repeat(64)},reference_hash:'d'.repeat(64)},state,summary:{status:'PUBLISHED',result_hash:'b'.repeat(64),report_hash:'c'.repeat(64),readouts:{cumulative_net_wealth:1.234,cost_bps_per_side:5},limitations:[],fixture_reading:operation+' '+stage}});
  const goalWithReference = row => {const body=goalBody();body.references=[row];return body;};
  const goalReferencePath=library.hostRoutes().GOAL_REFERENCE?.path;
  assert.equal(goalReferencePath,'/api/goals/reference','selected Goal reference is a public Host route');
  const goalReferenceAnswer=row=>({status:'GOAL_REFERENCE_READBACK',goal_hash:H,reference:row,evidence_verification:'SELECTED_REFERENCE'});
  const goalAnswer=row=>endpoint=>endpoint.startsWith(goalReferencePath+'?')?goalReferenceAnswer(row):goalWithReference(row);
  const taskStatus = {task_id:ID,task_kind:'portfolio_public_development_replay',lifecycle:'SUCCEEDED',goal_summary:'Exact Task fixture',verified_stage_count:0,total_stage_count:0};
  const inspectorAnswer = endpoint => {
    if(endpoint.startsWith(goalReferencePath+'?'))return goalReferenceAnswer(referenceReading());
    if(endpoint.startsWith('/api/goals'))return goalWithReference(referenceReading());
    if(endpoint.startsWith('/api/tasks/recovery'))return {task_id:ID,status:taskStatus,stages:[],actions:['CANCEL','RECOVER','REPLAN'].map(action=>({action,available:false,reason:'The Task is SUCCEEDED',scope:'Exact Task fixture',expected_effect:'No work admitted'})),artifact_refs:[],health:{status:'HEALTHY'},incidents:[],liveness:{status:'NOT_APPLICABLE'}};
    if(endpoint.startsWith('/api/status'))return taskStatus;
    if(endpoint==='/api/tasks')return {tasks:[taskStatus]};
    if(endpoint.startsWith('/api/activity'))return {items:[],epoch:'fixture',more:false};
    return {};
  };
  for(const row of base.probe.Inspect.addressModes)await check('cold inspector '+row.mode,async()=>{
    const route=inspectorFixtures[row.mode];assert.ok(route,'every registered inspector address has a cold fixture: '+row.mode);
    const address='#'+new URLSearchParams(route);
    for(let document=0;document<2;document++){
      const c=makeContext({hash:address,realData:true,body:inspectorAnswer,inspectorDom:true});c.probe.readRoute();
      if(route.goal)await c.probe.LiveGoals.ensure();
      c.probe.Inspect.reopenFromAddress();await connect(c);await c.probe.Inspect.reopenFromAddress();await settle();
      assert.equal(c.probe.Window.inspectorMode(),row.mode,'connection keeps the addressed reading');
      for(const [key,value] of Object.entries(route))assert.equal(q(c).get(key),value,'cold reader keeps exact '+key);
      assert.ok(c.inspector.innerHTML,'cold inspector has its own body');
    }
    counts.inspector_modes=(counts.inspector_modes||0)+1;
  });
  await check('Task Record transitions',async()=>{
    const c=makeContext({hash:'#page=tasks',realData:true,body:inspectorAnswer,inspectorDom:true});await connect(c);
    await c.probe.LiveTasks.open(ID);assert.equal(c.probe.Window.inspectorMode(),'task');
    c.probe.Inspect.openRecord();assert.equal(q(c).get('task'),ID);assert.equal(q(c).get('record'),'1');
    assert.equal(c.probe.Window.inspectorMode(),'record');
    await c.probe.LiveTasks.open(ID);assert.equal(c.probe.Window.inspectorMode(),'task','Record opener mark cannot masquerade as Task mode');
    assert.ok(!q(c).has('record'));assert.equal(q(c).get('task'),ID);
    c.probe.Inspect.openRecord();c.probe.Window.closeInspector();assert.ok(!q(c).has('record'));assert.equal(q(c).get('task'),ID,'closing Record keeps its exact Task scope');
  });
  await check('pending Task cannot replace addressed Record',async()=>{
    let release;const held=new Promise(resolve=>{release=resolve;});
    const c=makeContext({hash:'#page=tasks',realData:true,body:endpoint=>endpoint.startsWith('/api/tasks/recovery')?held:inspectorAnswer(endpoint),inspectorDom:true});await connect(c);
    const opening=c.probe.LiveTasks.open(ID);c.probe.Inspect.openRecord();release(inspectorAnswer('/api/tasks/recovery'));await opening;await settle();
    assert.equal(c.probe.Window.inspectorMode(),'record');assert.equal(q(c).get('task'),ID);assert.equal(q(c).get('record'),'1');
  });
  // V668 / TE12: the public owner's admitted operation/stage table is the class,
  // including both comparison families and all four experiment stages.
  const referenceOperations=library.codes().goal_reference_operations;
  assert.ok(referenceOperations && Object.keys(referenceOperations).length,'reference kinds come from their owner');
  for(const [operation,stages] of Object.entries(referenceOperations))for(const stage of stages)await check('Goal reference '+operation+'/'+stage,async()=>{
    const row=referenceReading(operation,stage),c=makeContext({hash:'#page=goal-results&goal='+H,body:goalAnswer(row),inspectorDom:true});c.probe.readRoute();await c.probe.LiveGoals.ensure();resetHistory(c);c.records.requests.length=0;
    await c.probe.LiveGoals.evidence('fixture-reference');assert.equal(c.probe.Window.inspectorMode(),'reference');assert.equal(q(c).get('reference'),row.reference.reference_id);assert.equal(q(c).get('goal'),H);assert.ok(!q(c).has('facts'));
    assert.equal(c.records.pushes.length,1,'one browser Back returns to the exact Goal folder');
    assert.deepEqual(c.records.requests,[goalReferencePath+'?'+new URLSearchParams({goal_hash:H,goal_reference_id:row.reference.reference_id})],'the opener reads only its exact reference, with no whole Goal verification');
    assert.ok(String(c.probe.Window.detailPane('reference')).includes('Own referenced object')&&String(c.probe.Window.detailPane('reference')).includes(operation+' '+stage),'referenced object header and own reading');
    assert.ok(!String(c.probe.Window.detailPane('reference')).includes('Routing Goal'),'never the Goal Facts');
    if(operation==='REPORT')assert.ok(String(c.probe.Window.detailPane('reference')).includes('Cumulative net wealth')&&String(c.probe.Window.detailPane('reference')).includes('1.234'),'Report presents the owner readouts');
    c.setRoute('#page=goal-results&goal='+H);c.probe.Inspect.reopenFromAddress();assert.equal(c.probe.Window.inspectorMode(),null,'Back leaves the reference reader');assert.equal(q(c).get('goal'),H);
    counts.goal_reference_kinds=(counts.goal_reference_kinds||0)+1;
  });
  for(const state of ['UNAVAILABLE','VERIFIED_REFUSAL','VERIFIED_TASK_STATE'])await check('Goal reference state '+state,async()=>{
    const row=referenceReading('COMPARE','PORTFOLIO',state);if(state==='UNAVAILABLE'){delete row.summary;row.failure_code='goal.reference_changed_reattach_explicitly';}if(state==='VERIFIED_REFUSAL'){row.summary.owner_refusal={failure_code:'portfolio_research.comparison_requires_two_results',detail:'Choose two recorded results.'};row.summary.claim='Owner refused this comparison; no metrics or common scope invented.';}
    const c=makeContext({hash:'#page=goal-results&goal='+H,body:goalAnswer(row),inspectorDom:true});c.probe.readRoute();await c.probe.LiveGoals.ensure();await c.probe.LiveGoals.evidence('fixture-reference');
    assert.ok(String(c.probe.Window.detailPane('reference')).includes('fixture-reference'));assert.ok(!String(c.probe.Window.detailPane('reference')).includes('data-status="verified"'),'refusal/Task state is not a verified readback');
    if(state==='VERIFIED_REFUSAL')assert.ok(String(c.probe.Window.detailPane('reference')).includes('Choose two recorded results.')&&String(c.probe.Window.detailPane('reference')).includes('goal-reference-retry'),'comparison refusal gives its own cause and way');
    if(state==='UNAVAILABLE')assert.ok(String(c.probe.Window.detailPane('reference')).includes('could not be read at its recorded identity')&&String(c.probe.Window.detailPane('reference')).includes('goal-reference-retry'),'typed exact-reference refusal has its way on');
  });
  // The owner can return only a typed code (located_failure/OperationAnswer).
  // Enumerate that admitted class too; adding detail would hide the real shape.
  const comparisonRefusals=library.codes().goal_comparison_refusals,words=library.words(appDir);
  assert.ok(comparisonRefusals?.length,'code-only comparison refusals come from their owner');
  for(const code of comparisonRefusals)for(const lang of ['en','zh'])await check('Goal code-only refusal '+code+'/'+lang,async()=>{
    const row=referenceReading('COMPARE','PORTFOLIO','VERIFIED_REFUSAL');
    row.summary.owner_refusal={status:'REFUSED',failure_code:code};row.summary.claim='Owner refused this comparison; no metrics or common scope invented.';
    const c=makeContext({hash:'#page=goal-results&goal='+H,body:goalAnswer(row),inspectorDom:true});c.probe.readRoute();
    Object.assign(c.window.ALPHA_ZH,words.I18N.catalog);vm.runInContext('I18N.set('+JSON.stringify(lang)+')',c);words.I18N.set(lang);
    await c.probe.LiveGoals.ensure();await c.probe.LiveGoals.evidence('fixture-reference');
    const markup=String(c.probe.Window.detailPane('reference')),banner=/<div class="banner[\s\S]*?<\/div>/.exec(markup)?.[0];
    assert.ok(banner?.includes(words.t(row.summary.claim)),'code-only refusal reads its recorded owner claim in '+lang);
    assert.ok(!markup.includes(words.t('Word not declared')),'no undeclared visible cause');
    assert.ok(markup.includes('fixture-reference')&&markup.includes('goal-reference-retry'),'exact reference and integrity-read way retained');
    counts.goal_comparison_refusals=(counts.goal_comparison_refusals||0)+1;
  });
  await check('Goal missing reference and read failure',async()=>{
    for(const failure of ['missing','transport']){
      const c=makeContext({hash:'#page=goal-results&goal='+H+'&reference=missing-reference',body:async()=>{throw Error(failure==='transport'?'goal.reference_identity_unavailable':'goal.reference_not_found');},inspectorDom:true});c.probe.readRoute();await c.probe.LiveGoals.openReference('missing-reference');
      assert.deepEqual(c.records.requests,[goalReferencePath+'?'+new URLSearchParams({goal_hash:H,goal_reference_id:'missing-reference'})]);
      assert.ok(String(c.probe.Window.detailPane('reference')).includes('missing-reference')&&String(c.probe.Window.detailPane('reference')).includes('goal-reference-retry'));assert.equal(c.probe.Window.inspectorMode(),'reference');assert.ok(!q(c).has('facts'));
    }
  });

  await check('Goal selected reference exact selectors and retry',async()=>{
    const row=referenceReading(),id='report / β+?';row.reference.reference_id=id;
    const c=makeContext({hash:'#page=goal-results&goal='+H,body:goalAnswer(row),inspectorDom:true});c.probe.readRoute();await c.probe.LiveGoals.ensure();c.records.requests.length=0;
    await c.probe.LiveGoals.evidence(id);await c.probe.LiveGoals.openReference(id,{retry:true});
    const exact=goalReferencePath+'?'+new URLSearchParams({goal_hash:H,goal_reference_id:id});
    assert.deepEqual(c.records.requests,[exact,exact],'the offered ID remains exact, including punctuation and the explicit retry');
    assert.equal(q(c).get('reference'),id);assert.ok(String(c.probe.Window.detailPane('reference')).includes('Own referenced object'));
  });
  for(const mismatch of ['goal','reference'])await check('Goal selected response rejects wrong '+mismatch,async()=>{
    const row=referenceReading(),answer=goalReferenceAnswer(row);
    if(mismatch==='goal')answer.goal_hash='b'.repeat(64);else answer.reference.reference.reference_id='another-reference';
    const c=makeContext({hash:'#page=goal-results&goal='+H+'&reference=fixture-reference',body:answer,inspectorDom:true});c.probe.readRoute();await c.probe.LiveGoals.openReference('fixture-reference');
    const markup=String(c.probe.Window.detailPane('reference'));assert.ok(markup.includes('could not be read at its recorded identity')&&!markup.includes('Cumulative net wealth'),'a mismatched owner answer never becomes the selected reading');
  });

  await check('V683 Alpha Foundation label reads one exact summary',async()=>{
    const F='f'.repeat(64),full=studyBody(),summaryPath=library.hostRoutes().EXPERIMENT_FOUNDATION_SUMMARY?.path;
    assert.equal(summaryPath,'/api/experiments/foundations/summary');full.alpha_source.foundation_admission_hash=F;
    const c=lifecycleContext({hash:'#page=alpha',body:url=>url.startsWith(summaryPath+'?')?{status:'FOUNDATION_SUMMARY',foundation_admission_hash:F,ordered_factor_ids:['first','second','third','fourth','fifth'],verification:'METADATA_ONLY_SOURCE_GRAPH_NOT_CHECKED'}:full}),p=c.probe;
    await p.LiveStudy.open(ID,'alpha');p.LiveStudy.page();p.LiveStudy.page();await settle();
    const markup=String(p.LiveStudy.page()).replace(/<[^>]*>/g,'');
    assert.ok(markup.includes('5 factors · first, second, third +2'),'the Properties label keeps the ordered factor count, first names and remaining count');
    assert.deepEqual(c.records.requests.filter(url=>url.startsWith('/api/experiments/foundations')),[summaryPath+'?'+new URLSearchParams({foundation_admission_hash:F})],'Alpha label reads no full Foundation list or readback');
    assert.equal(q(c).get('study'),ID);assert.equal(p.app.page,'alpha');
  });
  for(const departure of ['page','study'])await check('V683 Alpha Foundation label discards a reply after '+departure+' leave',async()=>{
    const old=lifetimeGate(),F='f'.repeat(64),other='66666666-2222-3333-4444-555555555555';let labels=0;
    const exact='/api/experiments/foundations/summary?'+new URLSearchParams({foundation_admission_hash:F});
    const c=lifecycleContext({hash:'#page=alpha',body:url=>{if(url===exact)return ++labels===1?old.promise:{status:'FOUNDATION_SUMMARY',foundation_admission_hash:F,ordered_factor_ids:['fresh-factor']};const body=studyBody();body.task_id=new URLSearchParams(url.split('?')[1]).get('task_id');body.alpha_source.foundation_admission_hash=F;return body;}}),p=c.probe;
    await p.LiveStudy.open(ID,'alpha');p.LiveStudy.page();await settle();assert.equal(labels,1);
    if(departure==='page'){
      p.navigate('overview');old.resolve({status:'FOUNDATION_SUMMARY',foundation_admission_hash:F,ordered_factor_ids:['stale-factor']});await settle();
      assert.equal(p.app.page,'overview');p.navigate('alpha',{study:ID});p.LiveStudy.ensure();
    }else await p.LiveStudy.open(other,'alpha');
    p.LiveStudy.page();await settle();
    if(departure==='study'){old.resolve({status:'FOUNDATION_SUMMARY',foundation_admission_hash:F,ordered_factor_ids:['stale-factor']});await settle();}
    const markup=String(p.LiveStudy.page());assert.ok(markup.includes('fresh-factor')&&!markup.includes('stale-factor'),'the new selected label survives a late reply');
    assert.equal(labels,2,'the new visit/revision performs its own exact label read');
    assert.deepEqual(c.records.requests.filter(url=>url.startsWith('/api/experiments/foundations')),[exact,exact],'neither visit hydrates the Foundation collection');
    assert.equal(q(c).get('study'),departure==='page'?ID:other);counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });

  await check('V683 selected Foundation renderer ensure never starts a lobby read',async()=>{
    const standing=lifetimeGate(),collection='/api/experiments/foundations',exact='/api/experiments/foundations/readback?'+new URLSearchParams({foundation_admission_hash:H});let ensureCalls=0;
    const a={admission_hash:H,factor_task_id:ID,input_id:'fixture-input',input_binding_hash:H,curation_receipt_hash:H,foundation:{foundation_hash:H,ordered_factor_ids:['factor'],limitations:[],execution_outcome:{market_as_of:'2026-08-03'}}};
    const c=lifecycleContext({hash:'#page=foundation&foundation='+H,body:url=>{if(url===exact)return {status:'FOUNDATION_SEALED',admission:a};if(url===collection)return standing.promise;throw Error('unexpected Foundation read '+url);}}),p=c.probe,ensure=p.LiveStudy.ensure;
    // Keep production renderPage and its queued ensure; supply the real selected content.
    p.LiveViews.page=()=>p.LiveStudy.page();p.LiveStudy.ensure=(...args)=>{ensureCalls++;return ensure(...args);};
    c.queueMicrotask=fn=>Promise.resolve().then(fn);
    await p.LiveStudy.foundations(H);await settle();
    assert.ok(ensureCalls>=2,'the real renderer queued ensure before and after selected owner readback');
    assert.deepEqual(c.records.requests.filter(url=>url.startsWith(collection)),[exact,collection],'the selected full readback and one pending owner standing read are the whole demand');
    assert.equal(q(c).get('foundation'),H);assert.equal(p.LiveStudy.context()?.foundation_admission_hash,H);
    standing.resolve({foundations:[{status:'FOUNDATION_SEALED',admission:a,standing:'HISTORICAL',standing_code:'research_foundation.not_current'}]});await settle();c.render();await settle();
    const markup=String(c.main.innerHTML),draft=/<button\b[^>]*data-action="study-foundation-draft"[^>]*>/.exec(markup)?.[0];
    assert.ok(draft&&/\bdisabled\b/.test(draft),'the full standing owner still disables drafting from the historical admission');
    assert.ok(markup.includes('It reads by its recorded graph; a draft or a PLAN on it is refused.'));
    assert.deepEqual(c.records.requests.filter(url=>url.startsWith(collection)),[exact,collection],'ready selected-page renders also leave the lobby unopened');
    counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });

  for(const departure of ['abort','revision','page'])await check('V683 Foundation standing full read retries after '+departure,async()=>{
    const old=lifetimeGate(),a={admission_hash:H,factor_task_id:ID,input_id:'fixture-input',input_binding_hash:H,curation_receipt_hash:H,foundation:{foundation_hash:H,ordered_factor_ids:['factor'],limitations:[],execution_outcome:{market_as_of:'2026-08-03'}}};let collections=0,shared=0;
    const historical={foundations:[{status:'FOUNDATION_SEALED',admission:a,standing:'HISTORICAL',standing_code:'research_foundation.not_current'}]};
    const c=lifecycleContext({hash:'#page=foundation&foundation='+H,body:url=>url==='/api/experiments/foundations'?(++collections===1?old.promise:historical):{status:'FOUNDATION_SEALED',admission:a}}),p=c.probe,read=p.Data.read;
    p.Data.readShared=(url,...args)=>{if(url==='/api/experiments/foundations')shared++;return read(url,...args);};
    await p.LiveStudy.foundations(H);p.LiveStudy.page();await settle();assert.equal(collections,1);
    if(departure==='page'){
      p.navigate('overview');old.resolve({foundations:[{admission:a,standing:'CURRENT'}]});await settle();p.navigate('foundation',{foundation:H});
    }else if(departure==='revision')await p.LiveStudy.foundations(H);
    else {old.reject(readCancelled());await settle();}
    p.LiveStudy.page();await settle();
    if(departure==='revision'){old.resolve({foundations:[{admission:a,standing:'CURRENT'}]});await settle();}
    const markup=String(p.LiveStudy.page()),draft=/<button\b[^>]*data-action="study-foundation-draft"[^>]*>/.exec(markup)?.[0];
    assert.equal(collections,2,'the cancelled/obsolete full-standing holder is read again');assert.equal(shared,0,'Foundation standing belongs to its page visit');
    assert.ok(draft&&/\bdisabled\b/.test(draft),'the fresh full owner HISTORICAL answer keeps the draft disabled');
    assert.ok(markup.includes('It reads by its recorded graph; a draft or a PLAN on it is refused.'),'full owner standing is retained');
    assert.ok(!markup.includes('Read cancelled'),'cancellation is not kept as a standing refusal');
    assert.ok(c.records.requests.includes('/api/experiments/foundations/readback?'+new URLSearchParams({foundation_admission_hash:H})),'the selected admission still uses full owner readback');
    counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  for(const departure of ['abort','page'])await check('V683 Storage backup collection retries after '+departure,async()=>{
    const old=lifetimeGate(),exact='/api/workspace/backup';let backups=0,shared=0;
    const fresh={backup_root:'fixture-backup-fresh',generations:[],refusals:[]};
    const c=lifecycleContext({hash:'#page=storage',body:url=>{if(url===exact)return ++backups===1?old.promise:fresh;if(url.startsWith('/api/workspace/storage'))throw Error('local_web.service_unreachable');return {};}}),p=c.probe,read=p.Data.read;
    p.Data.readShared=(url,...args)=>{if(url===exact)shared++;return read(url,...args);};
    await p.LiveWorkspace.refresh('storage');p.LiveWorkspace.page();await settle();assert.equal(backups,1);
    if(departure==='page'){
      p.navigate('overview');old.resolve({backup_root:'fixture-backup-stale',generations:[],refusals:[]});await settle();p.navigate('storage');
    }else {old.reject(readCancelled());await settle();}
    p.LiveWorkspace.page();await settle();const markup=String(p.LiveWorkspace.page());
    assert.equal(backups,2,'an unfinished backup read never poisons the next read');assert.equal(shared,0,'the backup collection is a scoped page read');
    assert.ok(markup.includes('fixture-backup-fresh')&&!markup.includes('fixture-backup-stale')&&!markup.includes('Read cancelled'),'the retried owner collection replaces no facts with a cancellation');
    assert.deepEqual(c.records.requests.filter(url=>url===exact),[exact,exact]);assert.equal(p.app.page,'storage');
    counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });


  await check('V683 loaded Report auxiliary export leaves no successor writes',async()=>{
    const e=reviewReadContext(),{p,c}=e;await loadedReport(e);
    const old=e.hold(e.exportURL,{marker:'abandoned export',html:'obsolete HTML'});
    const exporting=p.LiveReview.readExport();await reviewReadEntered(old);
    const shared=p.Data.readShared(e.exportURL);
    p.LiveReview.step('evidence');
    const sources=e.hold(p.Data.route('EVIDENCE_PREVIEW')+'?'+new URLSearchParams(e.selector),{status:'EVIDENCE_PREVIEW',source_candidates:[]});
    const successor=p.LiveReview.inspectSources();await reviewReadEntered(sources);
    await reviewReadCompleted(exporting);
    assert.equal(p.LiveReview.facts().error,'','Report cancellation is not an Evidence refusal');
    assert.equal(p.LiveReview.facts().busy,'sources','old export finally cannot clear successor work');
    assert.equal(old.request.signal.aborted,false,'the application subscriber retains the exact export wire');
    p.LiveReview.exportReport('json');assert.equal(e.downloads.length,0);
    old.release();assert.equal((await shared).marker,'abandoned export');await settle();
    assert.equal(p.LiveReview.facts().error,'');assert.equal(p.LiveReview.facts().busy,'sources');
    p.LiveReview.exportReport('json');assert.equal(e.downloads.length,0,'late shared export is not adopted by the departed Report');
    sources.release();await successor;p.LiveReview.step('report');
    const fresh=e.hold(e.exportURL,{marker:'fresh export',html:'fresh HTML'}),reread=p.LiveReview.readExport();
    await reviewReadEntered(fresh);assert.notEqual(fresh.request.signal,old.request.signal);
    assert.equal(e.requests.filter(r=>r.url===e.exportURL).length,2,'revisiting starts a fresh owner read');
    fresh.release();await reread;p.LiveReview.exportReport('json');assert.equal(JSON.parse(e.downloads[0].text).marker,'fresh export');
    assert.equal(p.app.page,'report');assert.equal(p.LiveReview.facts().error,'');counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 current Report owner refusal remains visible',async()=>{
    const e=reviewReadContext(),{p}=e;await loadedReport(e);
    const refusal=e.hold(e.exportURL,{status:'REFUSED',refused:'fixture.report_refused',message:'Current report read refused'});
    const reading=p.LiveReview.readExport();await reviewReadEntered(refusal);refusal.release();await reading;
    assert.match(p.LiveReview.facts().error,/fixture\.report_refused/);
    assert.ok(String(p.LiveReview.page()).includes('Current report read refused'),'the current typed refusal is rendered');
    assert.equal(p.LiveReview.facts().busy,'');counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 retained Review Task retry keeps its new subscriber',async()=>{
    const e=reviewReadContext(),{p}=e;await loadedReport(e);
    const url='/api/tasks/recovery?'+new URLSearchParams({task_id:ID});
    const old=e.hold(url,{task_id:ID,task_kind:'alternative_evidence.document_intelligence',lifecycle:'RUNNING',stages:[],verified_stage_count:0,total_stage_count:1});
    const watching=p.LiveReview.watch(ID,true);await reviewReadEntered(old);const shared=p.Data.readShared(url);
    p.LiveReview.step('evidence');p.LiveReview.observe();
    assert.equal(e.selected.filter(x=>x===url).length,2,'the retained Task retries immediately after leaving');
    await reviewReadCompleted(watching);p.LiveReview.observe();
    assert.equal(e.selected.filter(x=>x===url).length,2,'old finally cannot clear the new subscriber and permit a duplicate retry');
    assert.ok(!String(p.LiveReview.page()).includes('Read cancelled'),'old catch cannot render cancellation as a Task refusal');
    assert.equal(old.request.signal.aborted,false,'the global Task reader remains active');
    old.release();assert.equal((await shared).lifecycle,'RUNNING');await settle();
    assert.equal(p.LiveReview.facts().work.lifecycle,'RUNNING','the new page subscriber adopts its owner observation');
    const refused=e.hold(url,{status:'REFUSED',refused:'fixture.task_refused',message:'Current Task read refused'});
    const current=p.LiveReview.watch(ID,true);await reviewReadEntered(refused);refused.release();await current;
    assert.ok(String(p.LiveReview.page()).includes('Current Task read refused'),'a current Task owner refusal remains visible');
    counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 Review projection owns its ticket after destination paint',async()=>{
    const e=reviewReadContext(),{p,c}=e;await connect(c);
    for(const page of ['evidence-stream','report']){
      const held=e.hold(e.projectionURL,e.projection),opening=p.LiveReview.open(e.selector,'',page);
      await reviewReadEntered(held);held.release();await opening;
      assert.equal(p.app.page,page);assert.equal(p.LiveReview.facts().state,'REVIEW_PUBLISHED');
      assert.equal(p.LiveReview.context().selector.result_hash,H);assert.equal(p.LiveReview.facts().error,'');
    }
    const held=e.hold(e.exportURL,{marker:'accepted after projection',html:''}),reading=p.LiveReview.readExport();
    await reviewReadEntered(held);held.release();await reading;
    assert.equal(p.LiveReview.facts().busy,'','accepted projection remains ready for a selected export');
    counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });

  await check('V683 Review same-selector return through renderer teardown',async()=>{
    const selector={result_hash:H},c=lifecycleContext({hash:'#page=evidence',body:reviewBody()}),p=c.probe;
    await p.LiveReview.open(selector,'','evidence');
    const fullReads=()=>c.records.requests.filter(url=>url.startsWith('/api/evidence-cro?'));
    assert.equal(fullReads().length,1);
    p.navigate('overview');p.navigate('evidence',{review_selector:JSON.stringify(selector)});
    p.LiveReview.ensure();await settle();
    assert.equal(fullReads().length,2,'the same exact book is re-opened after its earlier page visit ended');
    assert.equal(p.LiveReview.facts().busy,'');assert.equal(q(c).get('review_selector'),JSON.stringify(selector));
    counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 destination packet survives the previous Review page teardown',async()=>{
    const selector={result_hash:H},c=lifecycleContext({hash:'#page=evidence',body:{...reviewBody(),state:'ANALYST_PACKET_PREPARED'}}),p=c.probe,documents=[];
    p.Data.readDocument=async url=>{documents.push(url);const value={status:'EVIDENCE_PACKET_READY',packet:'# Analyst packet\n\n```json\n{"spans":[]}\n```\n',prepared_task_id:ID,submission_template:{operation:'EVIDENCE_ANALYSIS_SUBMIT',result_hash:H,task_id:ID,analysis_context_hash:H,packet_hash:H}};return {value,text:JSON.stringify(value)};};
    await p.LiveReview.open(selector,'','evidence');await p.LiveReview.useTask(ID);
    assert.equal(p.app.page,'handoff');assert.equal(documents.length,1,'one selected packet read follows the destination render');
    assert.equal(p.LiveReview.facts().packet?.task,ID,'the destination read is adopted, not invalidated by leaving Evidence');
    assert.equal(p.LiveReview.facts().busy,'');counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 same-page book supersession releases retained-document read',async()=>{
    const old=lifetimeGate();let documentReads=0;
    const c=lifecycleContext({hash:'#page=evidence',body:url=>url.startsWith('/api/evidence/documents')?(++documentReads===1?old.promise:{status:'EVIDENCE_DOCUMENTS',documents:[],page:1,page_count:1,total:0}):reviewBody()}),p=c.probe;
    await p.LiveReview.open({result_hash:H},'','evidence');p.LiveReview.documentsPage('1');await settle();
    await p.LiveReview.open({result_hash:'b'.repeat(64)},'','evidence');p.LiveReview.documentsPage('1');await settle();
    assert.equal(documentReads,2,'a superseded document read cannot hold the new book pending forever');
    old.resolve({status:'EVIDENCE_DOCUMENTS',documents:[],page:1,page_count:1,total:0});await settle();
    assert.equal(p.LiveReview.facts().busy,'');counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 unfinished Alpha strict read resumes after a page visit',async()=>{
    const old=lifetimeGate();let strictReads=0;
    const full={...studyBody(),receipt:{receipt_hash:H}},summary={...studyBody('alpha.model-development','EXPERIMENT_SUMMARY')};
    const c=lifecycleContext({hash:'#page=alpha',body:url=>url.startsWith('/api/experiments/summary?')?summary:url.startsWith('/api/experiments/readback?')?(++strictReads===1?old.promise:full):{}}),p=c.probe;
    const pending=p.LiveStudy.open(ID,'alpha');await settle();assert.equal(p.LiveStudy.context()?.task_id,ID,'the qualified summary was adopted before strict verification');
    p.navigate('overview');old.resolve(full);await pending;
    p.navigate('alpha',{study:ID});p.LiveStudy.ensure();await settle();
    assert.equal(strictReads,2,'returning resumes the unfinished strict read');assert.equal(p.LiveStudy.context()?.receipt_hash,H,'the full owner read replaces the summary');
    counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 unfinished Factor curation resumes after a page visit',async()=>{
    const old=lifetimeGate();let choiceReads=0;
    const choices={receipt_hash:H,choices:[],limitations:[],decisions:[{receipt_hash:H}],inputs:['fixture-input']};
    const c=lifecycleContext({hash:'#page=factor',body:url=>url.startsWith('/api/experiments/curation?')?(++choiceReads===1?old.promise:choices):studyBody('factor.screening-development')}),p=c.probe,previews=[];
    p.Data.post=async(url,payload)=>{previews.push({url,payload});return {admission:{admission_hash:H}};};
    const pending=p.LiveStudy.open(ID,'factor');await settle();p.navigate('overview');old.resolve(choices);await pending;
    p.navigate('factor',{study:ID});p.LiveStudy.ensure();await settle();
    assert.equal(choiceReads,2,'a published body does not hide its unfinished curation read');
    p.LiveStudy.chooseDecision(H);p.LiveStudy.target(JSON.stringify(['fixture-input',H]));await p.LiveStudy.previewFoundation();
    assert.equal(previews.length,1,'the restored curation can supply its saved decision');assert.equal(previews[0].payload.curation_receipt_hash,H);
    counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 fresh cross-study open owns the destination visit',async()=>{
    const other='66666666-2222-3333-4444-555555555555';
    const c=lifecycleContext({hash:'#page=factor',body:url=>{if(url.startsWith('/api/experiments/curation?'))return {receipt_hash:H,choices:[],limitations:[],decisions:[],inputs:[]};const task=new URLSearchParams(url.split('?')[1]).get('task_id');return {...studyBody(task===ID?'factor.screening-development':'alpha.model-development',url.startsWith('/api/experiments/summary?')?'EXPERIMENT_SUMMARY':'EXPERIMENT_PUBLISHED'),task_id:task,receipt:url.startsWith('/api/experiments/summary?')?undefined:{receipt_hash:H}};}}),p=c.probe;
    await p.LiveStudy.open(ID,'factor');await p.LiveStudy.open(other,'alpha');
    assert.equal(p.LiveStudy.context()?.task_id,other);assert.equal(p.LiveStudy.context()?.receipt_hash,H,'rendering Alpha does not invalidate the new open ticket');
    assert.equal(c.records.requests.filter(url=>url.startsWith('/api/experiments/readback?')&&url.includes(other)).length,1);
    counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 unfinished Lab controls reload after a page visit',async()=>{
    const old=lifetimeGate();let controlReads=0;
    const controls={status:'READY',document:{experiment:{kind:'factor.screening-development'},factor:{factor_ids:[]}},yaml:'factor draft',controls:[],research_input_id:'fixture-input',input_binding_hash:H,plan_request:{research_input_id:'fixture-input',input_binding_hash:H}};
    const route='#'+new URLSearchParams({page:'lab',research_input:'fixture-input',input_binding:H,experiment_kind:'factor.screening-development'});
    const c=lifecycleContext({hash:route,body:url=>{assert.ok(url.startsWith('/api/experiments/controls?'));return ++controlReads===1?old.promise:controls;}}),p=c.probe;
    const pending=p.LiveResearch.ready();await settle();assert.equal(p.LiveResearch.entryState(),'loading');
    p.navigate('overview');old.resolve(controls);await pending;
    c.setRoute(route);c.render();await p.LiveResearch.ready();
    assert.equal(controlReads,2,'an unfinished initialized load is retried on return');assert.equal(p.LiveResearch.entryState(),'ready','load busy was released and the new declaration adopted');
    assert.equal(p.LiveResearch.context().input_binding_hash,H);counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 completed Study write releases busy after leaving its page',async()=>{
    const answer=lifetimeGate();let writes=0,catalogReads=0;
    const choices={receipt_hash:H,choices:[],limitations:[],decisions:[],inputs:[]};
    const c=lifecycleContext({hash:'#page=factor',body:url=>url.startsWith('/api/experiments/curation?')?choices:studyBody('factor.screening-development')}),p=c.probe;
    p.Data.post=async()=>{writes++;return answer.promise;};p.Data.refreshExperiments=async()=>{catalogReads++;};
    await p.LiveStudy.open(ID,'factor');p.LiveStudy.confirm('curate');const pending=p.LiveStudy.commit();await settle();
    assert.equal(writes,1);p.navigate('overview');answer.resolve({decision:{receipt_hash:H}});await pending;
    assert.equal(p.app.page,'overview','completion does not navigate back to the page left');
    p.navigate('factor',{study:ID});await p.LiveStudy.catalog();assert.equal(catalogReads,1,'a finished write cannot leave the Study permanently busy');
    counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 cancelled comparison member is retried for the same saved pair',async()=>{
    const other='66666666-2222-3333-4444-555555555555',old=lifetimeGate();let rightReads=0;
    const body=task=>({...studyBody(),task_id:task,result:{candidates:[{candidate_id:task===ID?'left':'right',status:'DEVELOPMENT_EVALUATED'}]}});
    const c=lifecycleContext({hash:'#page=alpha-compare',body:url=>{const task=new URLSearchParams(url.split('?')[1]).get('task_id');return task===other?(++rightReads===1?old.promise:body(other)):body(ID);}}),p=c.probe;
    await p.LiveStudy.open(ID,'alpha-compare');const pending=p.LiveStudy.alphaComparisonTask(other,'right');await settle();const route=c.location.hash;
    p.navigate('overview');old.reject(readCancelled());await pending;
    c.setRoute(route);c.render();p.LiveStudy.ensure();await settle();
    assert.equal(rightReads,2,'returning to the same pair resumes its cancelled member read');
    assert.ok(!String(p.LiveStudy.page()).includes('Read cancelled'),'cancellation is not kept as an owner comparison refusal');counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  await check('V683 cancelled Team verification preserves its previous discovery',async()=>{
    const old=lifetimeGate(),ref=ID;
    const c=lifecycleContext({hash:'#page=team',body:url=>url.startsWith('/api/status?')?{task_id:ID,task_kind:'research_experiment',lifecycle:'SUCCEEDED'}:url.startsWith('/api/experiments/readback?')?old.promise:{}}),p=c.probe;
    await p.LiveTeam.resolve(ref);const previous=p.LiveTeam.resolved().get(ref);assert.equal(previous.level,'discovered');
    const pending=p.LiveTeam.verify(ref);await settle();p.navigate('overview');old.reject(readCancelled());await pending;
    assert.equal(p.LiveTeam.resolved().get(ref),previous,'cancellation keeps the same discovery without an owner failure');
    p.navigate('team');p.Data.read=async()=>({...studyBody('factor.screening-development'),task_id:ID});await p.LiveTeam.verify(ref);
    assert.equal(p.LiveTeam.resolved().get(ref).level,'verified','the cancelled verification released its busy reference');counts.read_lifetimes=(counts.read_lifetimes||0)+1;
  });
  console.log(JSON.stringify({counts,failures:failures.length}));if(failures.length)process.exitCode=1;finish();
})().catch(error=>{console.error(error);process.exitCode=1;});
