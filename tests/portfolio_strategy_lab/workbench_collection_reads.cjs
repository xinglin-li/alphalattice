// Collection/read lifetime contract over the actual Data owner. Only fetch,
// address state and presentation are fixtures; cancellation is never mocked.
// Intended final location: tests/portfolio_strategy_lab/workbench_collection_reads.cjs.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const library = require('./workbench_library.cjs');
const finish = library.guard('workbench_collection_reads');
const root = process.argv[2];
const tick = () => new Promise(resolve => setImmediate(resolve));
const outcome = promise => promise.then(value => ({value}), error => ({error}));
const abortError = () => Object.assign(new Error('Fixture transport aborted'), {name:'AbortError'});
const HASH = 'a'.repeat(64);
const TASK = '11111111-2222-4333-8444-555555555555';

function environment() {
  const requests = [], gates = new Map(), answers = new Map(), downloads = [], objectEntries = [];
  const hash = {value:'#page=books'};
  const bootstrap = new Map([
    ['/api/session', {session_token:'fixture-session', workspace_id:'collection-fixture', routes:{},
      research_context:{inputs:{inputs:[]}, tasks:{tasks:[],refusals:[]}, data_update:{}}}],
    ['/api/research-history', {entries:[],blocked_entries:[],next_cursor:null}],
    ['/api/experiments', {experiments:[],refusals:[]}],
    ['/api/decisions', {decisions:[]}],
  ]);
  function hold(url, body, {ignoreSignal=false,contentType='application/json',rawText=undefined,phase='fetch',textError=null}={}) {
    assert.ok(!gates.has(url), 'one explicit gate per request');
    let release, reject;
    const gate = {url, body, ignoreSignal, contentType, rawText, phase, textError, entered:false, bodyEntered:false,
      promise:new Promise((resolve, fail) => {release=resolve;reject=fail;})};
    gate.release = () => release();
    gate.reject = reject;
    gates.set(url, gate);
    return gate;
  }
  async function fetchMock(url, init={}) {
    const endpoint = String(url), route = endpoint.split('?')[0];
    const row = {url:endpoint, path:route, method:init.method || 'GET', signal:init.signal,
      aborted:false, abortCount:0, textReads:0};
    requests.push(row);
    const held = gates.get(endpoint);
    let body, detach=()=>{};
    if (held) {
      gates.delete(endpoint); held.entered=true; held.request=row;
      const aborted = () => {row.aborted=true;row.abortCount++;if(!held.ignoreSignal)held.reject(abortError());};
      detach = () => init.signal?.removeEventListener('abort', aborted);
      if (init.signal?.aborted) aborted();
      else init.signal?.addEventListener('abort', aborted, {once:true});
      if (held.phase !== 'body') try {await held.promise;} finally {detach();}
      body=held.body;
    } else if (answers.has(endpoint)) body=answers.get(endpoint);
    else if (bootstrap.has(route)) body=bootstrap.get(route);
    else throw Error('Unscripted owner read: '+endpoint);
    return {ok:true,status:200,headers:{get:()=>held?.contentType || 'application/json'},text:async()=>{
      row.textReads++;
      if (held?.phase === 'body') {
        held.bodyEntered=true;
        try {await held.promise;} finally {detach();}
      }
      if (held?.textError) throw held.textError;
      return held?.rawText ?? JSON.stringify(body);
    }};
  }
  const c = {console, URLSearchParams, Date, Number, Math, Set, Map, Object, Array,
    String, Promise, JSON, Boolean, Error, AbortController,
    app:{page:'books',book:null,session:null,input:null,tasks:[],inputs:[],compareOther:null},
    window:{ALPHA_PRODUCT:true,FIXTURES:null,addEventListener(){}},
    document:{body:{dataset:{page:'books'}}},
    location:{search:'',get hash(){return hash.value;}},
    hashParams:()=>new URLSearchParams(hash.value.slice(1)),
    replaceHash(update){const q=new URLSearchParams(hash.value.slice(1));for(const [k,v] of Object.entries(update))q.set(k,v);hash.value='#'+q;},
    fetch:fetchMock, render(){}, patchMain(){}, clone:value=>JSON.parse(JSON.stringify(value)),
    readPreference:()=>null, Window:{render(){},inspectorMode:()=>null},
    Inspect:{reopenFromAddress(){},resetWindow(){},selectObservation(){}},
    objectEntry:key=>{objectEntries.push(key);return true;},
    LiveViews:{existing:key=>objectEntries.push(key)},
    t:value=>String(value),btn:()=>'',alertDialog(){},notify(){},
    download:(text,name,type)=>downloads.push({text,name,type}),
    LiveResearch:{ready(){}}, LiveActivity:{setFollowing(){}},
  };
  library.context(c, root);
  vm.runInContext(fs.readFileSync(path.join(root,'data.js'),'utf8')+';globalThis.owner=Data;',c,{filename:'data.js'});
  c.owner.visitPage('books');
  const move = page => {const previous=c.app.page;c.owner.leavePage(previous);c.app.page=page;hash.value='#page='+page;c.owner.visitPage(page);};
  return {c, data:c.owner, requests, downloads, objectEntries, hold, answer:(url,body)=>answers.set(url,body), move};
}

async function entered(held) {
  for(let attempt=0;attempt<50 && !held.entered;attempt++)await tick();
  assert.ok(held.entered,'the actual owner reached the held transport');
}
async function aborted(result, label) {
  const end=await result;
  assert.equal(end.error?.name,'AbortError',label);
  assert.ok(!('value' in end),'no cancelled value was accepted');
}

async function pageLifetime() {
  for(const document of [false,true]) for(const ignoreSignal of [false,true]) {
    const e=environment(), url=document ? '/api/evidence-cro/export?result_hash='+HASH : '/api/models';
    const held=e.hold(url,{marker:'old-page'},{ignoreSignal});
    const result=outcome(document ? e.data.readDocument(url) : e.data.read(url));
    await entered(held);
    assert.ok(held.request.signal,'page GET/document reaches fetch with a signal');
    assert.equal(held.request.signal.aborted,false);
    e.move('history');
    assert.equal(held.request.signal.aborted,true,'leaving aborts the real page visit');
    assert.equal(held.request.aborted,true,'transport sees its abort event');
    if(ignoreSignal)held.release();
    await aborted(result,(document?'document':'GET')+' rejects after page leave, including an ignored signal');

    // A return to the same page is a fresh visit; no old AbortController is reused.
    e.move('books');
    const fresh=e.hold(url,{marker:'new-page'});
    const reread=document ? e.data.readDocument(url) : e.data.read(url);
    await entered(fresh);
    assert.notEqual(fresh.request.signal,held.request.signal);
    assert.equal(fresh.request.signal.aborted,false);
    fresh.release();
    const value=await reread;
    assert.equal(document ? value.value.marker : value.marker,'new-page');
  }
}

async function selectedResultLifetime() {
  const e=environment(), metadata='/api/results?'+new URLSearchParams({task_id:TASK});
  const held=e.hold(metadata,{results:[{task_id:TASK,result_hash:HASH}],refusals:[]},{ignoreSignal:true});
  const result=outcome(e.data.installedResult(TASK));
  await entered(held);e.move('history');held.release();
  await aborted(result,'leaving during selected result metadata fences the next owner read');
  assert.deepEqual(e.requests.map(r=>r.url),[metadata], 'no report read follows cancelled metadata');
}

async function selectedTaskLookup() {
  const e=environment(), metadata='/api/results?'+new URLSearchParams({task_id:TASK});
  const report='/api/report?'+new URLSearchParams({result_hash:HASH});
  e.answer(metadata,{results:[{task_id:TASK,result_hash:HASH}],refusals:[]});
  e.answer(report,{result_hash:HASH,used_by_task_ids:[TASK],marker:'exact-report'});
  const value=await e.data.installedResult(TASK);
  assert.equal(value.marker,'exact-report');
  assert.deepEqual(e.requests.map(r=>r.url),[metadata,report], 'one exact Task metadata read, then one selected report; no unfiltered scan');
  assert.ok(e.requests.every(r=>r.signal && !r.signal.aborted));

  const absent=environment();absent.answer(metadata,{results:[],refusals:[]});
  assert.equal(await absent.data.installedResult(TASK),null);
  assert.deepEqual(absent.requests.map(r=>r.url),[metadata],'an absent selected result triggers no report scan');

  const wrongTask=environment();wrongTask.answer(metadata,{results:[{task_id:'other-task',result_hash:HASH}],refusals:[]});
  await assert.rejects(wrongTask.data.installedResult(TASK),/portfolio_application\.result_readback_mismatch/);
  assert.deepEqual(wrongTask.requests.map(r=>r.url),[metadata],'metadata for another Task never triggers a report');
  for(const badReport of [{result_hash:'b'.repeat(64),used_by_task_ids:[TASK]},
    {result_hash:HASH,used_by_task_ids:['other-task']}]) {
    const wrongReport=environment();wrongReport.answer(metadata,{results:[{task_id:TASK,result_hash:HASH}],refusals:[]});
    wrongReport.answer(report,badReport);
    await assert.rejects(wrongReport.data.installedResult(TASK),/portfolio_application\.result_readback_mismatch/);
    assert.deepEqual(wrongReport.requests.map(r=>r.url),[metadata,report],'REPORT identity and Task use are checked after one exact read');
  }
}

async function sharedAndAdmittedWork() {
  const e=environment();await e.data.connect();
  assert.equal(e.data.workspaceStatus,'ready','the real connect obtains the synthetic session');
  e.requests.length=0;
  const readURL='/api/activity?limit=1', postURL='/api/experiments/run';
  const shared=e.hold(readURL,{marker:'observed'}), admitted=e.hold(postURL,{status:'ADMITTED',task_id:TASK});
  const read=e.data.readShared(readURL), post=e.data.post(postURL,{experiment_plan_hash:HASH});
  await entered(shared);await entered(admitted);
  assert.ok(shared.request.signal,'application-owned observation uses the internal wire signal');
  assert.equal(shared.request.signal.aborted,false);
  assert.equal(admitted.request.signal,undefined,'admitted POST has no page signal');
  e.move('history');
  assert.equal(shared.request.signal.aborted,false,'the application reader keeps its wire alive after page leave');
  shared.release();admitted.release();
  assert.equal((await read).marker,'observed');
  assert.equal(shared.request.signal.aborted,false,'a completed application read is not aborted');
  assert.equal((await post).task_id,TASK,'page leave does not lose the admitted Task');
  assert.deepEqual(e.requests.map(r=>[r.url,r.method]),[[readURL,'GET'],[postURL,'POST']]);
  assert.equal(e.requests.filter(r=>r.aborted).length,0);
}


// Separate follow-up regression group: selected internal Data readers share the
// visit lifetime, while Standing and application collections remain shared reads.
const SESSION = '2024-08-12';
const OTHER = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee';
const portfolioURL = (session=SESSION,scope=null) => '/api/workbench/portfolio?'+new URLSearchParams({task_id:TASK,portfolio_session:session,...(scope?{scope}:{})});
const book = (session=SESSION,installed=false) => ({subject:{task_id:TASK,session,
  source_kind:installed?'INSTALLED_RESULT':'EXPERIMENT',result_hash:HASH},series:[],holdings:[],metrics:{},standing:{marker:'old'}});
async function prime(e, {installed=false,page='portfolio'}={}) {
  e.answer(portfolioURL(),book(SESSION,installed));
  await e.data.openPortfolio(TASK,SESSION,page);
  assert.equal(e.data.subject()?.task_id,TASK,'the actual Portfolio owner accepted the fixture book');
  e.requests.length=0;
}
async function leaveSelected(e,held,result,label) {
  await entered(held);
  assert.ok(held.request.signal,label+' reaches fetch with its page signal');
  assert.equal(held.request.signal.aborted,false);
  e.move('history');const route=e.c.location.hash;
  assert.equal(held.request.signal.aborted,true,label+' signal aborts on leave');
  assert.equal(held.request.aborted,true,label+' transport observes the real abort');
  held.release();const end=await result;
  if(end.error)assert.equal(end.error.name,'AbortError',label+' only cancels, rather than failing for another reason');
  assert.equal(e.c.app.page,'history',label+' never navigates after leave');
  assert.equal(e.c.location.hash,route,label+' never rewrites the later route');
}
async function selectedInternalReads() {
  // Both initial projection and same-book date changes are selected reads.
  for(const sessionChange of [false,true]) {
    const e=environment();if(sessionChange)await prime(e);
    const session=sessionChange?'2024-08-13':SESSION;
    const held=e.hold(portfolioURL(session,sessionChange?'holdings':null),book(session),{ignoreSignal:true});
    const result=outcome(e.data.openPortfolio(TASK,session));
    await leaveSelected(e,held,result,sessionChange?'same-book date projection':'selected Portfolio projection');
    assert.equal(e.data.subject()?.session,sessionChange?SESSION:undefined,'late projection is never accepted');
  }
  {
    const e=environment();await prime(e);
    const url='/api/experiments/risk-links?'+new URLSearchParams({task_id:TASK});
    const held=e.hold(url,{links:[{marker:'late-link'}],refused_links:[]},{ignoreSignal:true});
    const result=outcome(e.data.discoverRiskLinks(TASK));
    await leaveSelected(e,held,result,'selected Risk links');
    assert.equal(e.data.riskLinks(TASK),null,'late links are never cached');
  }
  {
    const e=environment(),id='result:'+HASH;
    const url='/api/research-history?'+new URLSearchParams({history_entry_id:id});
    const held=e.hold(url,{entries:[{entry_id:id,kind:'INSTALLED_RESULT',status:'SUCCEEDED',task_id:TASK,book:{portfolio_session:SESSION}}]},{ignoreSignal:true});
    const result=outcome(e.data.openEntry(id));
    await leaveSelected(e,held,result,'exact selected History entry');
    assert.deepEqual(e.requests.map(r=>r.url),[url],'cancelled metadata starts no Portfolio projection');
    assert.deepEqual(e.objectEntries,[],'cancelled metadata creates no saved-object navigation');
    assert.equal(e.data.portfolioEntries().length,0,'late selected metadata is never retained in History');
  }
  for(const installed of [false,true]) {
    const e=environment();await prime(e,{installed,page:'compare'});
    let url;
    if(installed) {
      const otherHash='b'.repeat(64);
      e.answer('/api/results?'+new URLSearchParams({task_id:OTHER}),{results:[{task_id:OTHER,result_hash:otherHash}],refusals:[]});
      e.answer('/api/report?'+new URLSearchParams({result_hash:otherHash}),{result_hash:otherHash,used_by_task_ids:[OTHER]});
      url='/api/compare?'+new URLSearchParams({left:HASH,right:otherHash});
    } else url='/api/experiments/compare?'+new URLSearchParams({left_task_id:TASK,right_task_id:OTHER,portfolio_session:SESSION});
    const held=e.hold(url,{left:{task_id:TASK},right:{task_id:OTHER},marker:'late-comparison'},{ignoreSignal:true});
    const result=outcome(e.data.compare(OTHER));
    await leaveSelected(e,held,result,installed?'installed selected comparison':'study selected comparison');
    assert.equal(e.data.comparison(),null,'late comparison is never accepted');
    assert.notEqual(e.data.comparisonStatus,'ready','late comparison is never exportable');
    assert.equal(e.downloads.length,0);
  }
  for(const installed of [false,true]) for(const format of ['json','html']) {
    const e=environment();await prime(e,{installed});
    const url=installed ? format==='html' ? '/report?'+new URLSearchParams({result_hash:HASH})
      : '/api/report?'+new URLSearchParams({result_hash:HASH,portfolio_session:SESSION})
      : '/api/experiments/export?'+new URLSearchParams({task_id:TASK,portfolio_session:SESSION});
    const options={ignoreSignal:true,...(installed && format==='html' ? {contentType:'text/html',rawText:'<html>fixture export</html>'} : {})};
    const held=e.hold(url,{result_hash:HASH,json:'fixture export',html:'<html>fixture export</html>'},options);
    const result=outcome(e.data.exportStudy(format));
    await leaveSelected(e,held,result,(installed?'installed':'study')+' '+format+' export');
    assert.equal(e.downloads.length,0,'a late selected export never starts a download');
  }
}
async function sharedInternalReads() {
  {
    const e=environment();await prime(e,{installed:true});
    const next={...book(SESSION,true),standing:{marker:'observed'}};
    const held=e.hold(portfolioURL(),next);
    const result=e.data.refreshStanding();await entered(held);
    assert.ok(held.request.signal,'Standing observation uses the internal wire signal');
    assert.equal(held.request.signal.aborted,false);
    e.move('history');
    assert.equal(held.request.signal.aborted,false,'Standing observation survives page leave');
    held.release();
    assert.equal(await result,true);
    assert.equal(held.request.signal.aborted,false,'completed Standing observation is not aborted');
    assert.equal(e.data.raw().standing.marker,'observed','Standing still updates after navigation');
    assert.equal(held.request.aborted,false);
  }
  {
    const e=environment();await e.data.connect();e.requests.length=0;
    const history=e.hold('/api/research-history?history_limit=50',{entries:[{entry_id:'fixture',kind:'INSTALLED_RESULT',status:'SUCCEEDED'}],blocked_entries:[],next_cursor:null});
    const experiments=e.hold('/api/experiments',{experiments:[{task_id:TASK}],refusals:[]});
    const decisions=e.hold('/api/decisions',{decisions:[{kind:'fixture'}]});
    const result=Promise.all([e.data.refreshHistory(),e.data.refreshDecisions()]);
    for(const held of [history,experiments,decisions]){
      await entered(held);assert.ok(held.request.signal,'application collection uses the internal wire signal');
      assert.equal(held.request.signal.aborted,false);
    }
    e.move('history');
    for(const held of [history,experiments,decisions]){
      assert.equal(held.request.signal.aborted,false,'application collection survives page leave');held.release();
    }
    await result;
    assert.ok([history,experiments,decisions].every(held=>!held.request.signal.aborted),'completed application collections are not aborted');
    assert.equal(e.data.portfolioEntries().length,1);
    assert.equal(e.data.experiments()[0].task_id,TASK);
    assert.equal(e.data.decisions()[0].kind,'fixture');
    assert.equal(e.requests.filter(r=>r.aborted).length,0);
  }
}


// The same unfinished raw GET may serve page, application and document readers.
// Header and body gates distinguish wire sharing from sharing accepted answers.
async function bodyEntered(held) {
  for(let attempt=0;attempt<50 && !held.bodyEntered;attempt++)await tick();
  assert.ok(held.bodyEntered,'headers arrived and the actual wire is waiting for its body');
}
async function cancelledBeforeRelease(result, label) {
  let end;
  result.then(value=>{end=value;});
  for(let attempt=0;attempt<5 && !end;attempt++)await tick();
  assert.ok(end,label+' completes without waiting for the abandoned transport');
  await aborted(Promise.resolve(end),label);
}
async function sharedWireSubscribers() {
  for(const pageFirst of [true,false]) for(const phase of ['fetch','body']) {
    const e=environment(),url='/api/fixture/shared?exact=1';
    const held=e.hold(url,{marker:'shared raw body'},{ignoreSignal:true,phase});
    let page,shared;
    if(pageFirst)page=outcome(e.data.read(url));else shared=e.data.readShared(url);
    await entered(held);if(phase==='body')await bodyEntered(held);
    // The second caller joins even after headers arrived and text() has started.
    if(pageFirst)shared=e.data.readShared(url);else page=outcome(e.data.read(url));
    await tick();
    assert.equal(e.requests.length,1,'both launch orders join one exact-URL GET during '+phase);
    assert.ok(held.request.signal,'the shared wire has its own controller');
    e.move('history');
    await cancelledBeforeRelease(page,'page subscriber cancels while the application subscriber remains');
    assert.equal(held.request.signal.aborted,false,'page leave does not abort another subscriber');
    assert.equal(held.request.abortCount,0);
    held.release();
    assert.equal((await shared).marker,'shared raw body');
    assert.equal(held.request.textReads,1,'the raw body is consumed once');
    assert.equal(held.request.signal.aborted,false,'successful raw completion does not abort');
    e.answer(url,{marker:'fresh owner answer'});
    assert.equal((await e.data.readShared(url)).marker,'fresh owner answer','settled answers are never reused');
    assert.equal(e.requests.length,2);
  }
}
async function independentWireInterpretation() {
  for(const strictFirst of [true,false]) {
    const e=environment(),url='/api/fixture/unavailable';
    const held=e.hold(url,{status:'UNAVAILABLE',failure_code:'fixture.owner_unavailable',message:'fixture refusal'});
    let strict,descriptive;
    if(strictFirst){strict=outcome(e.data.read(url));descriptive=e.data.readShared(url,true);}
    else{descriptive=e.data.readShared(url,true);strict=outcome(e.data.read(url));}
    await entered(held);held.release();
    const refused=await strict,accepted=await descriptive;
    assert.equal(refused.error?.body?.failure_code,'fixture.owner_unavailable','strict caller keeps the typed refusal');
    assert.equal(accepted.failure_code,'fixture.owner_unavailable','only the permissive caller accepts unavailable metadata');
    assert.equal(e.requests.length,1);assert.equal(held.request.textReads,1);
    assert.notEqual(refused.error.body,accepted,'each caller parses its own owner body');
  }
  const e=environment(),url='/api/fixture/objects';
  const held=e.hold(url,{nested:{marker:'owner'}});
  const selected=e.data.read(url),shared=e.data.readShared(url);
  await entered(held);held.release();
  const a=await selected,b=await shared;
  assert.notEqual(a,b);assert.notEqual(a.nested,b.nested,'accepted objects are not a shared answer cache');
  a.nested.marker='caller change';assert.equal(b.nested.marker,'owner');
}
async function lastWireSubscriber() {
  for(const phase of ['fetch','body']) {
    const e=environment(),url='/api/fixture/last-subscriber';
    const held=e.hold(url,{marker:'abandoned'},{ignoreSignal:true,phase});
    const first=outcome(e.data.read(url));await entered(held);if(phase==='body')await bodyEntered(held);
    const second=outcome(e.data.read(url));await tick();
    assert.equal(e.requests.length,1);
    e.move('history');
    await cancelledBeforeRelease(first,'first subscriber rejects immediately');
    await cancelledBeforeRelease(second,'last subscriber rejects immediately');
    assert.equal(held.request.signal.aborted,true,'the last subscriber aborts an unfinished '+phase+' wire');
    assert.equal(held.request.abortCount,1,'the shared transport is aborted once');
    held.release();await tick();
  }
}
async function revisitWireCleanup() {
  const e=environment(),url='/api/fixture/revisit';
  const abandoned=e.hold(url,{marker:'old'},{ignoreSignal:true,phase:'body'});
  const old=outcome(e.data.read(url));await bodyEntered(abandoned);
  e.move('history');await cancelledBeforeRelease(old,'abandoned subscriber');
  e.move('books');
  const fresh=e.hold(url,{marker:'new'},{phase:'body'});
  const first=e.data.read(url);await bodyEntered(fresh);
  assert.equal(e.requests.length,2,'a revisit starts before the ignored old wire settles');
  assert.notEqual(fresh.request.signal,abandoned.request.signal);
  abandoned.release();await tick();
  const peer=e.data.read(url);await tick();
  assert.equal(e.requests.length,2,'late old cleanup cannot evict the still-pending replacement wire');
  fresh.release();assert.equal((await first).marker,'new');assert.equal((await peer).marker,'new');
  assert.equal(fresh.request.signal.aborted,false);
}
async function documentWireInterpretation() {
  const e=environment(),url='/api/fixture/document';
  const held=e.hold(url,null,{phase:'body',contentType:'text/html; charset=utf-8',rawText:'<article>fixture</article>'});
  const html=e.data.readDocument(url,'html');await bodyEntered(held);
  const json=outcome(e.data.readShared(url));held.release();
  const document=await html,unreadable=await json;
  assert.equal(document.value,null);assert.equal(document.text,'<article>fixture</article>');
  assert.equal(unreadable.error?.body?.refused,'local_web.service_answer_unreadable','JSON caller cannot accept an HTML document');
  assert.equal(e.requests.length,1);assert.equal(held.request.textReads,1,'HTML and JSON interpret one raw body independently');
}
async function failedTextRetry() {
  const e=environment(),url='/api/fixture/text-failure';
  const held=e.hold(url,null,{phase:'body',textError:new Error('fixture body read failed')});
  const selected=outcome(e.data.read(url));await bodyEntered(held);
  const shared=outcome(e.data.readShared(url));held.release();
  for(const result of [await selected,await shared]){
    assert.equal(result.error?.body?.refused,'local_web.service_unreachable');
    assert.equal(result.error?.body?.transport,'fixture body read failed');
  }
  assert.equal(e.requests.length,1);assert.equal(held.request.textReads,1);
  assert.equal(held.request.signal.aborted,false,'settled body failures are not cancelled');
  const fresh=e.hold(url,{marker:'retry'});
  const retried=e.data.readShared(url);await entered(fresh);fresh.release();
  assert.equal((await retried).marker,'retry','a body-read failure leaves no settled cache or poisoned pending entry');
  assert.equal(e.requests.length,2);
}
async function mutationWireIsolation() {
  const e=environment();await e.data.connect();e.requests.length=0;
  const url='/api/fixture/same-path',payload={experiment_plan_hash:HASH};
  const get=e.hold(url,{marker:'read'}),read=e.data.readShared(url);await entered(get);
  const one=e.hold(url,{status:'ADMITTED',task_id:TASK,marker:'write one'}),postOne=e.data.post(url,payload);await entered(one);
  const two=e.hold(url,{status:'ADMITTED',task_id:TASK,marker:'write two'}),postTwo=e.data.post(url,payload);await entered(two);
  assert.deepEqual(e.requests.map(r=>r.method),['GET','POST','POST'],'neither identical POSTs nor a GET on their exact path join');
  assert.ok(get.request.signal);assert.equal(one.request.signal,undefined);assert.equal(two.request.signal,undefined);
  e.move('history');assert.equal(get.request.signal.aborted,false);
  get.release();one.release();two.release();
  assert.equal((await read).marker,'read');assert.equal((await postOne).marker,'write one');assert.equal((await postTwo).marker,'write two');
  assert.ok(e.requests.every(r=>!r.aborted),'application GET and admitted writes survive leaving');
}

(async()=>{
  await pageLifetime();
  await selectedResultLifetime();
  await selectedTaskLookup();
  await sharedAndAdmittedWork();
  await selectedInternalReads();
  await sharedInternalReads();
  await sharedWireSubscribers();
  await independentWireInterpretation();
  await lastWireSubscriber();
  await revisitWireCleanup();
  await documentWireInterpretation();
  await failedTextRetry();
  await mutationWireIsolation();
  finish();
})().catch(error=>{console.error(error);process.exitCode=1;});
