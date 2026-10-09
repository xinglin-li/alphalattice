'use strict';

// goal page paints the saved narrative first and verifies evidence second.
{
const library=require('./workbench_library.cjs'),complete=library.guard("goal_page_paints_the_saved_narrative_first_and_verifies_evidence_second");
const _appDir=require('node:path').resolve(process.argv[2]),_project=require('node:path').resolve(__dirname,'../..');
const _owners=JSON.parse(process.argv[3]);
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
// a goal is an object of the Goals section; its first folder is its timeline
let route=new URLSearchParams('page=goal&goal=one');
const reads=[];
let conversationSpec=null;
// One entry per verified-readback request, in request order; each keeps its own settlers.
const verifications=[];
const finish=(index)=>verifications[index].resolve();
const fail=(index)=>verifications[index].reject();
const declaration=(hash)=>({title:'T '+hash,objective:'O?',kind:'RESEARCH',scope:'',constraints:[],
  criteria:[{criterion_id:'c1',text:'answers O'}],deliverables:[],budget:null,
  research:{purpose:'NEW_RESEARCH',comparison_design:'d',required_stages:['ALPHA','RISK']},
      parent_goal_id:null});
const reference={reference_id:'alpha',label:'Alpha',stage:'ALPHA',intent_relation:'POST_HOC',
  request:{operation:'EXPERIMENT_READBACK',task_id:'t'}};
const record={sessions:[{vendor:'codex',session_id:'s-1'}],tasks:[{task_id:'task-1',
    kind:'research_experiment',state:'SUCCEEDED'}],
  request_count:3,event_count:2,message_count:1,open_assignments:[{message_id:'m1',
      recipient_id:'analyst'}],
  conversation:[{recorded_at:'2026-09-20T11:00:00+00:00',event_kind:'NATIVE_COORDINATION_MESSAGE',
      observation_id:'obs-1',agent_vendor:'codex',
    agent_session:'s-1',agent_id:'pm',role:'research_lead',message_kind:'assignment',message_id:'m1',
        recipient_id:'analyst',reply_to:null,summary:'Read the Alpha study'}],
  session_usage:[{vendor:'codex',session_id:'s-1',participants:[{agent_id:'pm',role:'research_lead',
      models:[{model:'gpt-6',efforts:['high'],responses:4,input_tokens:10,output_tokens:5,
      cache_read_tokens:2,cache_write_tokens:1}],pin_differs:['effort']}],
    by_model:[{model:'gpt-6',responses:4,input_tokens:10,output_tokens:5,cache_read_tokens:2,
        cache_write_tokens:1}]}]};
const narrative=(hash)=>({status:'GOAL_NARRATIVE',goal_id:'g',goal_hash:hash,head_hash:hash,
    state:'OPEN',
  evidence_verification:'NOT_PERFORMED',open_choices:[],statement_context:{},
      outcome:'QUESTION_OPEN',record,
  goal:{goal_id:'g',revision:1,goal_hash:hash,parent_hash:null,state:'OPEN',statements:[],
      declaration:declaration(hash),
    references:[reference],submitted_by:'HUMAN',change_reason:'first',submission:null,
        completion:null,
    recorded_at:'2026-09-20T10:00:00+00:00',intent_registered_at:'2026-09-20T09:00:00+00:00'},
  references:[{reference,state:'SAVED_NOT_VERIFIED',next_requests:{}}]});
const verified=(hash)=>({...narrative(hash),status:'GOAL_SHOW',
  evidence_verification:'COMPLETE',gaps:[{stage:'RISK',failure_code:'goal.stage_not_evidenced'}],
  references:[{reference,state:'VERIFIED_READBACK',summary:{status:'EXPERIMENT_PUBLISHED'}}]});
const text=(value)=>Array.isArray(value)?value.join(''):String(value??'');
const c={URLSearchParams,app:{page:'goal'},render(){},closeDialog(){},openDialog(){},
  navigate(){},objectEntry(){},hashParams:()=>new URLSearchParams(route),replaceHash(){},
  routeUrl:(page,extra={})=>'#'+new URLSearchParams({page,...extra}),
  t:(k,v)=>{const word=k.replace(/^[a-z0-9-]+\|/,'');
    return v?word.replace(/\{(\w+)\}/g,(m,n)=>v[n]??m):word;},
  html:(strings,...values)=>strings.reduce(
    (out,part,i)=>out+part+(i<values.length?text(values[i]):''),''),
  panel:(title,_sub,content,action='')=>`<section>${title}${content}${action}</section>`,
  banner:(title,body,tone='',action='')=>`<div class="${tone}">${title}:${body}${action}</div>`,
  btn:(label,action,value='')=>`<button data-action="${action}"`
    +` data-value="${value}">${label}</button>`,
  icon:()=>'',
  kv:(rows)=>rows.map(r=>r.join(': ')).join(' | '),viewRail:()=>'',
  objectHead:(title,meta='',primary='',state='',_tools=[],
      details={})=>`<h1>${title}</h1>${meta}${state}${primary}`
    +`<dl>${(details.facts||[]).map(([k,v])=>k+': '+v).join(' | ')}</dl>`,
  stateLine:(code,o={})=>`<span class="state-line" data-state="${code}">${o.word||code}</span>`,
  refusal:(body,tone='',o={})=>`<div class="refusal ${tone}">${o.word||''}: ${body.reason||''}`
    +`${o.action||''}</div>`,
  objectRow:(r,s={})=>`<div class="list-row ${r.cls||''}" data-state="${r.state}"`
    +` data-key="${s.key||''}">`
    +`${s.word||''} ${r.name} ${r.why||''}${(s.props||[]).join(' ')}${s.time||''}</div>`,
  mono:(v,n)=>String(v||'').slice(0,n),actorWords:(v)=>String(v||''),badge:(s,
      l)=>`<b data-status="${s}">${l}</b>`,
  logLine:(l)=>`<div class="log-line">${l.at||''} ${l.by} ${l.words} ${l.code||''}</div>`,
  runLog:(o)=>`<section class="run-log">${o.title} ${o.count}${o.lines.join('')}</section>`,
  Inspect:{openFacts(){}},LiveReview:{open(){}},
  noteLine:(title,body='',tone='',action='')=>
    `<p class="note-line">${title}${body?' · '+body:''}${action||''}</p>`,
  hint:(term,words)=>`<span data-tip="${words||''}">${term}</span>`,
  infoMark:(words)=>`<button data-tip="${words}">${words}</button>`,
  factsRef:(title,body)=>`<p>${title}</p>${body}`,
  // a document link carries its text in a template
  codeRef:(title,text)=>`<p>${title}</p><template>${
    typeof text==='string'?text:JSON.stringify(text,null,2)}</template>`,
  // a reference is one evidence row (its state as the owner wrote it, its facts,
  // its actions)
  evidenceRow:(type,x,o={})=>`<div class="list-row evidence-row" data-kind="${type}">`
    +`${x.reference?.label||x.title||''} ${x.state||''}${(o.props||[]).join('')}`
    +`${o.actions||''}</div>`,
  citePill:(h)=>`<span class="es-cite">${h}</span>`,skeleton:()=>'',
  // the list is a lobby -- here its rows, each as the list's row function draws it
  Lobby:{render:(name,spec)=>{if(name==='goal-conversation')conversationSpec=spec;
    return `<section data-lobby="${name}">`
    +`${spec.items.map((x)=>spec.row(x,{props:{}})).join('')}${(spec.filters||[]).map(f=>'FILTER('+f.field+')').join('')}</section>`;},older:()=>''},
  short:(v,n)=>String(v||'').slice(0,n),SHORT:{id:8,hash:12},timeGroup:(at)=>({key:String(at).split('T')[0],label:'t',
      rank:0,open:true}),
  countText:(n,one,many)=>String(n)+' '+(n===1?one:many).replace('{n} ',''),
  emptyState:(s,way='')=>`<p class="section-empty">${s}</p>${way}`,when:(v)=>String(v||''),
  whenText:(v)=>String(v||'—'),codeWords:(v)=>String(v??'—'),
  picker:()=>'',count:(n)=>String(n),json:(v)=>JSON.stringify(v,null,2),
  LiveViews:{savedObjectLink:()=>''},LiveStudy:{},LiveResearch:{},
  PAGES:{},LiveActivity:{state:()=>({epoch:'fixture'}),retained:()=>[]},
  crypto:{randomUUID:()=>'x'},
  Data:{live:true,history:()=>[],read:async(url)=>{
    reads.push(url);
    const hash=new URL('http://local'+url).searchParams.get('goal_hash');
    if(url.startsWith('/api/goals/narrative?'))return narrative(hash);
    if(url.startsWith('/api/goals/show?'))return new Promise((resolve,reject)=>{
      verifications.push({hash,resolve:()=>resolve(verified(hash)),
        reject:()=>reject(Error('local_client.connection_lost_task_may_still_run'))});});
    throw Error('unexpected '+url);}}};
c.goalReferenceIntegrity=library.words(require('path').dirname(require('node:path').join(_appDir,'live-goals.js'))).goalReferenceIntegrity;
library.context(c);
vm.runInContext(fs.readFileSync(require('path').join(require('path').dirname(require('node:path').join(_appDir,'live-goals.js')),
  'live-team.js'),'utf8')+';globalThis.LiveTeam=LiveTeam;',c);
vm.runInContext(fs.readFileSync(require('node:path').join(_appDir,'live-goals.js'),'utf8')+';globalThis.live=LiveGoals;',c);
const teamScene=c.LiveTeam.scene;
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const shows=(...parts)=>{const page=String(c.live.page());return parts.every(p=>page.includes(p));};
const hides=(...parts)=>!parts.some(p=>String(c.live.page()).includes(p));
const folder=async(page)=>{c.app.page=page;route.set('page',page);};
(async()=>{
  const opening=c.live.ensure();
  await tick();await tick();
  // The narrative painted while the verified readback is still pending.
  assert.deepEqual(reads,['/api/goals/narrative?goal_hash=one','/api/goals/show?goal_hash=one']);
  // the head says what is awaited; nothing missing is claimed or denied
  // while verifying
  assert.ok(shows('T one','O?','answers O','data-state="running">Checking reference integrity',
      'data-state="pending">Open'),'the head, while verifying: '+String(c.live.page()).slice(0,
      600));
  assert.ok(hides('Missing'),'no gap is claimed or denied while verifying');
  assert.ok(hides('data-action="goal-verify"'),'no verify primary while verifying');
  await c.live.ensure();
  assert.equal(reads.length,2,'a repaint while verifying starts no second read');
  assert.deepEqual(verifications.map(v=>v.hash),['one']);
  finish(0);await opening;await tick();
  assert.ok(shows('data-state="verified">Reference integrity checked',
    'Missing · RISK','Missing: 1'),
    'the head says verified and what is missing, in words');
  assert.ok(hides('Checking reference integrity','goal.stage_not_evidenced'));
  // the timeline: the Host's facts, the messages, the notes, the sessions' usage (),
  // the page's own line
  assert.ok(shows('Host facts','codex <span class="mono">s-1</span>','task-1','Decision notes',
      'Timeline','Goal registered',
    'Revision recorded by: HUMAN',
    'Revision 1 recorded · <span class="owner-text">first</span>',
        'assignment · <span class="owner-text">Read the Alpha study</span>',
        'Reference attached · Alpha',
    'Reference integrity checked · Reference integrity only; not scientific approval.'),
    'the timeline: '+String(c.live.page()).slice(0,2000));
  assert.ok(shows('Sessions and usage','gpt-6','10 in · 5 out · 2 cache read · 1 cache written',
      'differs from its card: effort'),'the sessions\u2019 models and tokens');
  // the conversation: each exchange links to its session; the kinds a filter; the open
  // assignments counted
  await folder('goal-conversation');
  // A reply alone does not close an assignment; the page names its retained closure state.
  assert.ok(shows('Read the Alpha study','data-tip="analyst">Unidentified',
      'Open assignments · 1 assignment not closed'),
      'the conversation: '+String(c.live.page()).slice(0,1500));
  assert.ok(hides('assignment no reply names yet'),'reply presence is not closure evidence');
  // Goal's recipient cell uses Team's real hook binding, not a path-derived role.
  const recipientPath='/root/alphalattice_risk',assignment=record.conversation[0];
  const originalAssignment=JSON.stringify(assignment);
  const recipientHook=(id,overrides={})=>({ordinal:1,observation_id:'hook-'+id,
    schema_kind:'ExternalActivityObserved',payload:{event_kind:'NATIVE_SUBAGENT_START_HOOK',
      subject:{native_session_id:'s-1',native_host:'codex',native_agent_id:id,
        native_hook_event:'SubagentStart',role:'alphalattice_risk',
        native_spawn_role:'alphalattice_risk',native_agent_path:recipientPath,
        native_agent_path_basis:'CODEX_SESSION_META',...overrides}}});
  const recipientCases=[
    ['bound',[recipientHook('risk-child')],true],
    ['missing',[],false],
    ['other session',[recipientHook('risk-child',{native_session_id:'other'})],false],
    ['unselected',[recipientHook('risk-child',{native_agent_path_basis:'OTHER'})],false],
    ['other host',[recipientHook('risk-child',{native_host:'claude-code'})],false],
    ['conflicting role',[recipientHook('risk-child',{native_spawn_role:'alphalattice_cro'})],
      false],
    ['two participants',[recipientHook('risk-child'),recipientHook('other-child')],false],
    ['two paths',[recipientHook('risk-child'),
      recipientHook('risk-child',{native_agent_path:'/root/other'})],false],
  ];
  assignment.recipient_id=recipientPath;
  for(const [name,hooks,bound] of recipientCases) {
    c.LiveActivity.retained=()=>[{items:hooks}];
    assert.ok(shows('data-tip="'+recipientPath+'">'+(bound?'Risk Modeling':'Unidentified')),
      name+': only the unique observed participant labels the Goal recipient');
    assert.ok(hides('data-tip="'+recipientPath+'">'+recipientPath),
      name+': the native locator remains disclosure, never a visible role');
  }
  assignment.recipient_id='analyst';c.LiveActivity.retained=()=>[];
  assert.equal(JSON.stringify(assignment),originalAssignment,'owner assignment is preserved');
  // An accepted deliverable remains an exact owner-bound product record, with child
  // authorship and parent submission distinct. Its Task time is not the receipt clock.
  const accepted={recorded_at:'2026-09-21T11:10:00+00:00',
    occurred_at:'2026-09-20T11:04:00+00:00',agent_vendor:'codex',agent_session:'s-1',
    agent_id:'child-1',role:'alphalattice_evidence_analyst',message_kind:'answer',
    message_id:'accepted-1',observation_id:'obs-2',input_channel:'PRODUCT_ACCEPTED_ANSWER',
    source_time_kind:'TASK_ADMISSION',submitted_by:'pm',recipient_id:'pm',reply_to:'m1',
    reference:'accepted-task',bundle_reference:'exact-bundle',answer_reference:'exact-answer',
    summary:'Product accepted deliverable (ACCEPTED): retained finding.'};
  const acceptedBefore=JSON.stringify(accepted),readCount=reads.length;
  record.conversation.push(accepted);record.message_count++;
  assert.ok(shows(accepted.summary,'Accepted answer','Alternative Analyst',
    'Submitted by <span tabindex="0" data-tip="pm">Main PM',
    'product record, not a native spoken turn','Task admission time',accepted.occurred_at,
    'answers <span class="mono">m1'), 'accepted answer provenance is retained');
  assert.equal(conversationSpec.axes.find(a=>a.key==='time').group(accepted).key,
    '2026-09-20','a next-day repair groups under the displayed Task admission date');
  const filing={...accepted,message_id:'accepted-filing',source_time_kind:'PRODUCT_ACCEPTED_AT',
    summary:'Product accepted deliverable (ACCEPTED): retained generic interpretation.'};
  record.conversation.push(filing);record.message_count++;
  assert.ok(shows(filing.summary,'Answer acceptance time',filing.occurred_at));
  assert.equal(conversationSpec.axes.find(a=>a.key==='time').group(filing).key,
    '2026-09-20','a next-day generic answer repair groups under its immutable acceptance date');
  record.conversation.pop();record.message_count--;
  assert.ok(shows('Recorded '+accepted.recorded_at),'the actual record clock stays in provenance');
  assert.equal(JSON.stringify(accepted),acceptedBefore,'the exact owner record is unchanged');
  assert.equal(reads.length,readCount,'Goal rows never join another session feed');
  accepted.submitted_by='unknown-parent';
  assert.ok(shows('Submitted by <span tabindex="0" data-tip="unknown-parent">Unidentified'));
  accepted.submitted_by='pm';
  record.conversation.pop();record.message_count--;
  // the results: the references with their verification
  await folder('goal-results');
  assert.ok(shows('Alpha VERIFIED_READBACK','Recorded evidence and limitations',
      'Revision recorded by: HUMAN','Not submitted yet: an agent submits it through the CLI.'),
      'the results name the saved revision recorder while no completion is submitted');
  assert.ok(hides('Submitted by'),'saving a revision is not labelled as submitting completion');
  await folder('goal');
  // the exact revision is Facts, a document link, never a pre block
  assert.equal(c.live.hasFacts(),true);
  assert.equal(c.live.factsSections().map(s=>s.title).join(','),'Exact revision');
  const exact=String(c.live.factsSections()[0].body);
  assert.ok(exact.includes('Revision recorded by: HUMAN')
    &&exact.includes(c.t('Read the exact revision (JSON)'))&&exact.includes('"goal_id": "g"')
    &&exact.includes('"submitted_by": "HUMAN"')&&!exact.includes('<pre'),
    'the exact goal is in Facts as a document link (), never a pre block');
  // Navigating to another goal while its verification is pending, then back: the stale
  // answer for the left goal settles after the newer selection and is dropped.
  route=new URLSearchParams('page=goal&goal=two');
  const second=c.live.ensure();await tick();await tick();
  assert.deepEqual(reads.slice(2),['/api/goals/narrative?goal_hash=two',
      '/api/goals/show?goal_hash=two']);
  route=new URLSearchParams('page=goal&goal=one');
  const third=c.live.ensure();await tick();await tick();
  assert.deepEqual(verifications.map(v=>v.hash),['one','two','one']);
  finish(1);await second;await tick();
  assert.ok(shows('T one','Checking reference integrity'),
      'the newer selection is not replaced by the old verification');
  // A failed verification keeps the saved narrative and offers an explicit retry; no loop.
  fail(2);await third;await tick();
  assert.ok(shows('T one','data-state="failed">Reference integrity not verified',
    'data-action="goal-verify"'));
  await folder('goal-results');
  assert.ok(shows('Alpha SAVED_NOT_VERIFIED',
      'Reference integrity not verified: local_client.connection_lost_task_may_still_run'),
      'the references say they were not re-read');
  await folder('goal');
  const before=reads.length;await c.live.ensure();
  assert.equal(reads.length,before,'a failed verification is not retried on repaint');
  const retry=c.live.verify();await tick();
  assert.equal(reads.length,before+1);
  assert.deepEqual(verifications.map(v=>v.hash),['one','two','one','one']);
  finish(3);await retry;
  assert.ok(shows('Reference integrity checked'));
  assert.ok(hides('Reference integrity not verified'));
  // every Goal folder reads the owner's reference states; zero and unavailable
  // references cannot acquire a positive integrity mark from a completed pass.
  const ownerSource=fs.readFileSync(require('path').join(_project,
    'src/alphalattice/control/product_host/composition/goals.py'),'utf8');
  const states=[...new Set([...ownerSource.matchAll(
    /"(VERIFIED_[A-Z_]+|UNAVAILABLE|SAVED_NOT_VERIFIED)"/g)].map(m=>m[1]))];
  assert.ok(states.includes('UNAVAILABLE')&&states.includes('VERIFIED_READBACK'));
  let serial=0;
  for(const rows of [[], ...states.map(state=>[{reference,state}]),
      [{reference,state:'VERIFIED_READBACK'},{reference,state:'UNAVAILABLE'}],
      [{reference,state:'FUTURE_OWNER_STATE'}]]) {
    const hash='class-'+(++serial), body={...verified(hash),references:rows,
      goal:{...verified(hash).goal,references:rows.map(r=>r.reference)}};
    c.Data.read=async()=>body;
    route=new URLSearchParams('page=goal&goal='+hash);c.app.page='goal';await c.live.ensure();
    for(const page of ['goal','goal-conversation','goal-results']) {
      await folder(page);
      if(page==='goal') {
        const line=String(c.live.page()).match(
          /<div class="log-line">[^<]*This page ([\s\S]*?)<\/div>/)?.[1];
        assert.ok(line,'the page-owned Timeline line is present');
        const positive=rows.length>0&&rows.every(r=>r.state.startsWith('VERIFIED_'));
        assert.equal(line.includes('Reference integrity checked'),positive,
          'Timeline integrity claim follows the actual rows: '+rows.map(r=>r.state));
        if(!rows.length)assert.ok(line.includes('Nothing to check yet'),'empty Timeline');
        else assert.ok(line.includes(c.t('Reference integrity only; not scientific approval.')),
          'the Timeline names the check and its limit');
      }
      if(!rows.length)assert.ok(shows('data-state="pending">Nothing to check yet')&&
        hides('data-state="verified">'),page+' empty');
      else {
        const good=rows.every(r=>r.state.startsWith('VERIFIED_'));
        assert.ok(shows('Reference integrity only; not scientific approval.'),page+' scope');
        assert.equal(shows('data-state="verified">Reference integrity checked'),good,
          page+' '+rows.map(r=>r.state));
        if(!good)assert.ok(shows('Reference integrity not verified')||
          shows('Reference integrity not checked'),page+' nonpositive');
      }
    }
  }
  assert.ok(String(c.live.checkLine({state:'COMPLETE'})).includes('Record complete'),
    'Goal completion names its checked record');
  // The owner COMPLETE checks its record; the attributed outcome belongs to Submission.
  const closed=verified('closed');closed.goal={...closed.goal,state:'COMPLETE',
    declaration:{...closed.goal.declaration,kind:'REVIEW',research:null},
    completion:{checked_at:'2026-09-20T12:00:00+00:00',sessions:record.sessions,
      tasks:record.tasks,request_count:record.request_count},
    submission:{outcome:'ACHIEVED',summary:'A synthetic attributed outcome.',
      criteria:[{criterion_id:'c1',answer:'MET',evidence:['alpha'],note:''}],
      deliverables:[],findings:[],problems:[],follow_ups:[],files:[]}};
  c.Data.read=async()=>closed;route=new URLSearchParams('page=goal&goal=closed');
  c.app.page='goal';await c.live.ensure();
  assert.ok(shows('data-state="succeeded">Record complete')&&hides('Achieved'),
    'the positive head names record completion');
  await folder('goal-results');
  assert.ok(shows('Record complete','Submission · Achieved','A synthetic attributed outcome.'),
    'Submission retains the agent outcome separately');
  // the same saved-revision field reads as a recorder on every folder and in
  // Facts, including a later revision recorded by someone other than the completing agent.
  // Keep the submission and completion payloads while changing only the revision's author.
  const words=library.words(require('path').dirname(require('node:path').join(_appDir,'live-goals.js')));
  const english={t:c.t,I18N:c.I18N,actorWords:c.actorWords};
  Object.assign(c,{t:words.t,I18N:words.I18N,actorWords:words.actorWords});
  for(const language of ['en','zh']) {
    words.I18N.set(language);
    // Closed historical goals without retained bound messages state the limit, with
    // real folder actions. Neither same-book nor unbound Team observations are history.
    const emptyHash='unobserved-'+language,empty={...closed,goal_hash:emptyHash,
      head_hash:emptyHash,goal:{...closed.goal,goal_hash:emptyHash},
      record:{...record,conversation:[],message_count:0,open_assignments:[]}};
    const outside=[{goal_id:'other-goal',summary:'Wrong Goal message, same book'},
      {goal_id:null,summary:'Unbound proof session'}];
    c.Data.history=()=>outside;c.LiveTeam.scene=()=>{throw Error('no inferred Team join');};
    c.Data.read=async(url)=>{
      assert.equal(new URL('http://local'+url).searchParams.get('goal_hash'),emptyHash);
      return empty;};
    route=new URLSearchParams('page=goal-conversation&goal='+emptyHash);
    c.app.page='goal-conversation';await c.live.ensure();
    assert.ok(shows(words.t('No recorded conversation under this goal'),
      words.t('No Team exchange is retained with this goal\'s explicit binding. '
        +'Recorded deliverables are in Results; goal changes are in Timeline. '
        +'Unbound sessions and unobserved native history are not added here.'),
      'data-action="go" data-value="goal-results"',
      'data-action="go" data-value="goal"'),language+' historical limit and folder ways');
    assert.ok(hides(...outside.map(m=>m.summary),accepted.summary,
      words.t('No Team exchange under this goal yet.')),language+' no false old conversation');
    await folder('goal-results');assert.ok(shows('A synthetic attributed outcome.'));
    await folder('goal');assert.ok(shows(words.t('Goal registered')));
    c.LiveTeam.scene=teamScene;
    const boundHash='bound-answer-'+language,bound={...empty,goal_hash:boundHash,
      head_hash:boundHash,goal:{...empty.goal,goal_hash:boundHash},
      record:{...record,conversation:[record.conversation[0],accepted],message_count:2}};
    c.Data.read=async(url)=>{
      assert.equal(new URL('http://local'+url).searchParams.get('goal_hash'),boundHash);
      return bound;};
    route=new URLSearchParams('page=goal-conversation&goal='+boundHash);
    c.app.page='goal-conversation';await c.live.ensure();
    assert.ok(shows(accepted.summary,words.t('Accepted answer'),
      words.t('Alternative Analyst'),words.t('Submitted by'),words.t('Main PM'),
      words.t('Task admission time'),accepted.occurred_at),language+' exact bound answer');
    assert.equal(JSON.stringify(accepted),acceptedBefore,'accepted provenance remains exact');
    for(const completed of [false,true])for(const [index,actor] of
        ['HUMAN','EXTERNAL_AUTOMATION'].entries()) {
      const hash='recorder-'+language+'-'+completed+'-'+index;
      const value=completed?closed:verified(hash);
      const body={...value,goal_hash:hash,head_hash:hash,goal:{...value.goal,
        goal_hash:hash,revision:index+2,parent_hash:'prior',submitted_by:actor,
        change_reason:'Synthetic later revision saved by '+actor}};
      c.Data.read=async()=>body;
      route=new URLSearchParams('page=goal&goal='+hash);c.app.page='goal';await c.live.ensure();
      const label=language==='zh'?'修订记录者':'Revision recorded by';
      const attribution=label+': '+words.actorWords(actor);
      for(const page of ['goal','goal-conversation','goal-results']) {
        await folder(page);
        assert.ok(shows(attribution)&&hides('Submitted by','提交者'),
          language+' '+page+' names the revision recorder: '+String(c.live.page()).slice(0,600));
        const facts=String(c.live.factsSections()[0].body);
        assert.ok(facts.includes(attribution)&&facts.includes('"submitted_by": "'+actor+'"'),
          language+' Exact revision names the same recorder and keeps the owner field');
        if(page==='goal-results') {
          if(completed)assert.ok(shows(words.t('Submission'),'A synthetic attributed outcome.')&&
            hides(words.t('Not submitted yet: an agent submits it through the CLI.')),
            'the kept completion stays a submission when another actor records the revision');
          else assert.ok(shows(words.t('Not submitted yet: an agent submits it through the CLI.')),
            'a saved revision still has no completion submission');
        }
      }
      assert.equal(body.goal.submission,value.goal.submission,'the owner submission data is kept');
      assert.equal(body.goal.completion,value.goal.completion,'the owner completion data is kept');
    }
  }
  Object.assign(c,english);
  // The saved revision stays fixed while its exact Goal-bound activity record advances.
  // One shared feed observer refreshes the owner; unrelated/unbound messages do not.
  let feed={epoch:'goal-store-1',cursor:'goal-store-1:0',disposition:'CONTINUED',
    notice:'',stale:false,error:''};
  let retained=[];
  c.LiveActivity={state:()=>feed,retained:()=>[{items:retained}]};
  const liveHash='mutable-conversation';
  let liveBody={...verified(liveHash),record:{...record,conversation:[],message_count:0,
    open_assignments:[]}};
  let liveReads=0,holdVerification=false,releaseVerification=null;
  const clone=value=>JSON.parse(JSON.stringify(value));
  c.Data.read=async(url)=>{
    liveReads++;
    assert.equal(new URL('http://local'+url).searchParams.get('goal_hash'),liveHash);
    const body=clone(liveBody);
    if(url.startsWith('/api/goals/narrative?'))return {...body,
      evidence_verification:'NOT_PERFORMED',
      references:body.references.map(row=>({...row,state:'SAVED_NOT_VERIFIED'}))};
    if(holdVerification) {
      holdVerification=false;
      return new Promise(resolve=>{releaseVerification=()=>resolve(body);});
    }
    return body;
  };
  const appendBound=(ordinal,summary)=>{
    const message={...accepted,message_id:'live-'+ordinal,observation_id:'live-'+ordinal,summary};
    liveBody={...liveBody,record:{...liveBody.record,
      conversation:[...liveBody.record.conversation,message],message_count:liveBody.record.message_count+1}};
    retained.push({ordinal,schema_kind:'ExternalActivityObserved',payload:{event_kind:'NATIVE_COORDINATION_MESSAGE',subject:{goal_id:liveBody.goal.goal_id}}});
    feed={...feed,cursor:feed.epoch+':'+ordinal};
  };
  route=new URLSearchParams('page=goal-conversation&goal='+liveHash);c.app.page='goal-conversation';
  await c.live.ensure();assert.equal(liveReads,2);
  retained.push({ordinal:1,payload:{subject:{goal_id:'other-goal'}}},
    {ordinal:2,payload:{subject:{goal_id:null}}});feed={...feed,cursor:'goal-store-1:2'};
  await c.live.observe();await c.live.ensure();
  assert.equal(liveReads,2,'unrelated and unbound events do not reread the selected Goal');
  // Enumerate the public owner's read vocabulary so a new read cannot silently loop.
  for(const operation of [..._owners.goal_read_operations,
      'CASE_NARRATIVE','CASE_READBACK','CASE_EXPORT']) {
    retained.push({ordinal:20+retained.length,schema_kind:'ProductOperationObserved',
      payload:{operation,phase:'RETURNED',subject:{goal_id:liveBody.goal.goal_id}}});
    feed={...feed,cursor:feed.epoch+':'+retained.at(-1).ordinal};
  }
  await c.live.observe();await c.live.ensure();
  assert.equal(liveReads,2,'Goal reader receipts cannot create a self-refresh loop');
  appendBound(33,'Fresh exact-bound answer');await c.live.observe();
  assert.equal(liveReads,4,
    'a bound event reruns narrative and strict readback under the same hash');
  assert.ok(shows('Fresh exact-bound answer','Reference integrity checked'));
  assert.equal(liveBody.goal.revision,1,'conversation freshness creates no Goal revision');
  await c.live.observe();assert.equal(liveReads,4,'an identical activity tick does not reread');
  // An arrival while strict verification is pending survives its older answer.
  appendBound(34,'Answer before slow verification');holdVerification=true;
  const advancing=c.live.observe();await tick();
  assert.ok(releaseVerification,'the controlled strict readback started');
  assert.ok(shows('Checking reference integrity'));
  appendBound(35,'Answer during slow verification');await c.live.observe();
  releaseVerification();await advancing;await tick();await tick();
  assert.equal(liveReads,8,'the pending newer bound event gets its own complete owner pass');
  assert.ok(shows('Answer during slow verification','Reference integrity checked')&&
    hides('Checking reference integrity'));
  feed={...feed,epoch:'goal-store-2',cursor:'goal-store-2:1',disposition:'RESET'};
  retained=[{ordinal:1,payload:{subject:{goal_id:'other-goal'}}}];
  await c.live.observe();assert.equal(liveReads,10,'a changed store rereads the Goal owner');
  await c.live.observe();assert.equal(liveReads,10,'the same RESET is consumed once');
  feed={...feed,disposition:'CONTINUED'};await c.live.observe();
  assert.equal(liveReads,10,'restored continuity rearms gaps without another owner read');
  feed={...feed,cursor:'goal-store-2:9'};
  feed={...feed,disposition:'TAIL',notice:'Entries recorded between were not read.'};
  await c.live.observe();
  assert.equal(liveReads,12,'a same-store TAIL rereads after a retention gap');
  retained.push({ordinal:10,schema_kind:'ProductOperationObserved',payload:{operation:'GOAL_SHOW',
    phase:'RETURNED',subject:{goal_id:liveBody.goal.goal_id}}});
  feed={...feed,cursor:'goal-store-2:10'};
  await c.live.observe();
  assert.equal(liveReads,12,'a repeated retention boundary is consumed once');
  await c.live.ensure();assert.equal(liveReads,12,'own reads inside a TAIL gap never self-refresh');
  // Leaving invalidates mutable metadata even when activity retention loses its change.
  c.live.leaveReads();c.app.page='team';
  liveBody={...liveBody,record:{...liveBody.record,
    conversation:[{...accepted,summary:'Recorded while away'}]}};
  await c.live.observe();assert.equal(liveReads,12,'an inactive Goal reads nothing');
  c.app.page='goal-conversation';await c.live.ensure();
  assert.equal(liveReads,14);assert.ok(shows('Recorded while away','Reference integrity checked'));
  const reload={...c,app:{page:'goal-conversation'}};library.context(reload);
  vm.runInContext(fs.readFileSync(require('node:path').join(_appDir,'live-goals.js'),'utf8')+';globalThis.live=LiveGoals;',reload);
  await reload.live.ensure();
  assert.equal(liveReads,16,'a cold page reads the current owner record');
  assert.ok(String(reload.live.page()).includes('Recorded while away'));
  delete c.LiveActivity;
  // the list: one row per goal, the composer's primary in the head
  route=new URLSearchParams('page=goals');c.app.page='goals';
  c.Data.read=async(url)=>{reads.push(url);return {goals:[{goal_hash:'one',goal_id:'g',
      title:'T one',
    objective:'O?',kind:'RESEARCH',state:'OPEN',revision:1,reference_count:1,
        recorded_at:'2026-09-20T10:00:00+00:00'}],
    next_cursor:null};};
  await c.live.ensure();
  const list=String(c.live.page());
  assert.ok(list.includes('data-lobby="goals"')&&list.includes('T one')
    &&list.includes('data-action="goal-new"')&&list.includes('New goal'),
        'the list is rows under the head: '+list);
  // the list's session axis groups a goal by the agent sessions its row names
  let spec=null;const lobby=c.Lobby.render;c.Lobby.render=(n,x)=>{spec=x;return lobby(n,x);};
  c.live.page();c.Lobby.render=lobby;
  const bySession=spec.axes.find((a)=>a.key==='session').group;
  const two=bySession({sessions:[{vendor:'claude',session_id:'s-1'},
    {vendor:'codex',session_id:'s-2'}]});
  assert.ok(two.key==='claude:s-1,codex:s-2'&&two.label.includes('s-1')
    &&two.label.includes('s-2'),'a goal two sessions worked on: '+JSON.stringify(two));
  assert.equal(bySession({sessions:[]}).label,'No session yet');
  // an old research case address opens its goal
  route=new URLSearchParams('page=cases&case=one');c.app.page='cases';
  let replaced=null;c.replaceHash=(v)=>{replaced=v;};
  await c.live.ensure();
  assert.equal(c.app.page,'goal');assert.equal(JSON.stringify(replaced),
      JSON.stringify({page:'goal',goal:'one',case:''}));
  console.log('goal page probe complete');
complete();
})().catch(e=>{console.error(e);process.exitCode=1;});
}
