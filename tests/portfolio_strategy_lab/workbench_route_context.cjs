// PG2: enumerate production routes, offered collection rows, object tabs and recovery.
// Primitive DOM and deterministic owner answers keep this a UI contract; the browser regression
// separately presses Reload across actual documents over the service and built bundle.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const project = path.resolve(__dirname, '../..');
const library = require(path.join(project, 'tests/portfolio_strategy_lab/workbench_library.cjs'));
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
// : inspect the actual shared locator markup, including its ancestry. This
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

// : use production renderPage for visit teardown, with only the content/chrome reduced.
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


function run(name, checks) {
  const finish = library.guard(name);
  Promise.resolve().then(checks).then(() => {
    console.log(JSON.stringify({counts, failures: failures.length}));
    if (failures.length) process.exitCode = 1;
    finish();
  }).catch(error => { console.error(error); process.exitCode = 1; });
}
module.exports = {run,assert,fs,path,vm,library,appDir,source,fixedScript,files,counts,H,ID,clone,makeContext,check,q,resetHistory,nonempty,settle,decode,markupNodes,ancestors,hasClass,assertLocators,localizedContext,offered,followAnchor,press,connect,goalBody,studyBody,portfolioBody,reviewBody,lifecycleContext,lifetimeGate,readCancelled,reviewReadContext,reviewReadEntered,reviewReadCompleted,loadedReport};
