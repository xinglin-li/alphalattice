// The observed research team scene, read from the real activity consumer: events shaped exactly
// like the lead's producer (NATIVE_SUBAGENT_START_HOOK / NATIVE_SUBAGENT_STOP_HOOK /
// NATIVE_COORDINATION_MESSAGE) grouped by declared native session, participant and qualified
// reference; assignment, question, answer, objection and PM response told apart; the 500-character
// preview kept apart from the referenced original; product observations attached only through the
// same qualified references and rendered in recorded order with the current Task state apart;
// observation identity kept when a client-declared event id repeats (identical replay collapsed
// and named, conflicting content shown and flagged); the foreground PM only from the declared
// parent id with the research_lead role; unqualified references never correlated; discovery,
// preview readback and owner verification told apart; only the compatible artifact verification
// reads as owner-verified; the Verify answer lands on the row that asked; an installed Portfolio
// replay verified through the Portfolio owner's index and REPORT, an authored experiment through
// its readback. No HTTP, model or data work.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const library=require('./workbench_library.cjs'); // the library's constants and the scripts' parameters, from the source (Q2)
const finish=library.guard('workbench_team');
const root=process.argv[2],reads=[],hashes=[],navigations=[],renders=[],closes=[],frames=[];
const hash={value:'#page=tasks'},prefs=new Map();
const P1='1'.repeat(64),P2='2'.repeat(64),TASK='0f0f0f0f-0000-4000-8000-00000000abcd',RESULT='9'.repeat(64);
const INSTALLED='0a0a0a0a-0000-4000-8000-00000000beef',ORPHAN='0b0b0b0b-0000-4000-8000-00000000beef',RESULT2='7'.repeat(64);
let previewStatus='MISSING',reportBody=null,readbackTask=null,results=[];
const report=(p)=>{if(!reportBody)throw Error('local_web.result_unknown');const asked=new URLSearchParams(p.split('?')[1]).get('result_hash');return typeof reportBody==='function'?reportBody(asked):reportBody;};
const c={console,URLSearchParams,Date,Number,Math,Set,Map,Object,Array,String,Promise,JSON,Boolean,Error,PAGES:{},ROUTES:{team:['Research Team','Workroom'],'team-sessions':['Research Team','Sessions'],'team-evidence':['Research Team','Team evidence']}, // the page words (2026-09-21: one word per page, ROUTES owns it)
  app:{page:'tasks',book:null},document:{hidden:false,addEventListener(){},querySelector:()=>null,getElementById:()=>null},window:{addEventListener(){}},
  setTimeout:()=>0,clearTimeout(){},
  hashParams:()=>new URLSearchParams(hash.value.slice(1)),
  replaceHash:(u)=>{const q=new URLSearchParams(hash.value.slice(1));for(const [k,v] of Object.entries(u))q.set(k,v);hash.value='#'+q;hashes.push(JSON.parse(JSON.stringify(u)));},
  navigate:(page,extra)=>{navigations.push([page,extra]);c.app.page=page;const q=new URLSearchParams({page,...extra});hash.value='#'+q;},
  render:()=>renders.push(c.app.page),patchMain:()=>renders.push(c.app.page),objectEntry:()=>true,
  Data:{read:async(p)=>{reads.push(p);if(p.startsWith('/api/experiments/preview'))return {status:previewStatus,plan_hash:P1};if(p.startsWith('/api/report'))return report(p);if(p.startsWith('/api/experiments/readback'))return {status:'EXPERIMENT_PUBLISHED',program:{kind:'factor.screening-development'},task_id:readbackTask||new URLSearchParams(p.split('?')[1]).get('task_id')};if(p.startsWith('/api/results'))return {results};if(p.startsWith('/api/workbench/portfolio'))throw Error('research_experiment.task_kind_mismatch');if(p.startsWith('/api/status'))throw Error('local_web.task_unknown');throw Error('probe: no read for '+p);},
    mergeTasks(){},installedResult:async(task)=>{const results=(await c.Data.read('/api/results')).results||[],first=results.find(r=>r.task_id===task);for(const listed of first?[first]:results){const report=await c.Data.read('/api/report?'+new URLSearchParams({result_hash:listed.result_hash}));if(report.result_hash!==listed.result_hash)throw new Error('portfolio_application.result_readback_mismatch');if((report.used_by_task_ids||[]).includes(task))return report;if(first)throw new Error('portfolio_application.result_readback_mismatch');}return null;},history:()=>[{id:'result:'+RESULT,name:'Installed strategy result',raw:{status:'SUCCEEDED',kind:'INSTALLED_RESULT'},task_id:'x'},{id:'result:'+RESULT2,name:'Installed strategy result',raw:{status:'SUCCEEDED',kind:'INSTALLED_RESULT'},task_id:INSTALLED}],refreshHistory:async()=>{reads.push('/api/research-history');},openEntry(){},tasks:()=>[{task_id:TASK,lifecycle:'SUCCEEDED',task_kind:'research_experiment'},{task_id:INSTALLED,lifecycle:'SUCCEEDED',task_kind:'portfolio_public_development_replay'},{task_id:ORPHAN,lifecycle:'SUCCEEDED',task_kind:'portfolio_public_development_replay'}]},
  LiveTasks:{paintActivity(){},open(){},openResult(){},close:(after)=>closes.push(typeof after)},LiveViews:{taskDock(){},savedObjectLink:(l,e)=>l+':'+e},LiveResearch:{dirty:()=>false,inspectShared(){}},
  requestAnimationFrame:(fn)=>{frames.push(fn);},
  notify(){},openDialog(){},
  link:(label,page,cls,extra)=>`LINK(${page} ${JSON.stringify(extra||{})})${label}`,
  html:(s,...v)=>s.reduce((a,p,i)=>a+p+(Array.isArray(v[i])?v[i].join(''):v[i]??''),''),t:(s,vars)=>String(s).replace(/\{(\w+)\}/g,(_,k)=>String(vars?.[k]??'')),
  notRead:(title,error,words='',action='')=>`<div class="warning">${title}:${error}${words?' '+words:''}${action}</div>`,noteLine:(title,body='',tone='',action='')=>`<p class="note-line">${title}${body?' · '+body:''}${action||''}</p>`,hint:(term)=>term,factsRef:(title,body)=>`<p>${title}</p>${body}`,codeRef:(title,text)=>'<p>'+title+'</p><template>'+(typeof text==='string' ? text : JSON.stringify(text))+'</template>',refCell:(uri)=>'<span>'+uri+'</span>',hashCell:(h)=>'<span>'+(h||'—')+'</span>',readingPane:(title,kind,body,close)=>'<aside>'+title+body+'</aside>',sourceRows:(rows)=>rows.map((r)=>[r.label,r.value]),badge:(tone,label)=>`[${tone}:${label}]`,stateLine:(x,o={})=>`[${(typeof x==='string'?x:(x?.lifecycle??x?.state??x?.status??''))}:${o.word??''}]`,skeleton:(shape='rows')=>'SKELETON('+shape+')',emptyState:(s,a='')=>'EMPTY('+s+')'+(a||''),railTools:()=>'',groupHead:(label,count)=>'GROUP('+label+':'+count+')',listFoot:()=>'',readPreference:(k)=>prefs.get(k)??null,savePreference:(k,v)=>prefs.set(k,v),displayOptions:(name,spec)=>{(c.displays||(c.displays={}))[name]=spec;return '<display:'+name+'>';},displayState:(name,spec)=>{const saved=prefs.get('display.'+name)||{};const pick=(rows,v)=>rows?.some(([k])=>k===v)?v:(rows?.[0]?.[0]||'');const props=Object.fromEntries((spec?.properties||[]).map(([key,,on=true])=>[key,saved.props&&key in saved.props?Boolean(saved.props[key]):on]));return {show:spec?.current?.show??pick(spec?.shows,saved.show),group:pick(spec?.groupings,saved.group),order:pick(spec?.orderings,saved.order),props,density:saved.density==='compact'?'compact':'comfortable'};},setDisplay:(name,patch)=>{const next={...(prefs.get('display.'+name)||{}),...patch};prefs.set('display.'+name,next);return next;},evidenceRow:(type,x,o={})=>{const r=typeof type==='object'?type:{type,subject:String(x?.entity_id||x?.issue_handle||x?.span_handle||x?.finding_handle||x?.reference?.label||x?.title||''),state:String(x?.state||''),word:'',why:''};return '<div class="list-row evidence-row '+(o.cls||'')+'" data-kind="'+(r.type||'')+'">ROW('+[o.named?r.kind+' · '+r.subject:r.subject,'['+r.state+':'+(r.word||'')+']',o.why??r.why,...((o.brief?[]:(r.props||[])).concat(o.props||[])).filter(Boolean).map(p=>Array.isArray(p)?p[1]:p),o.time,o.actions,o.attrs].filter(Boolean).join('|')+')</div>';},citePill:(h)=>'<span class="es-cite">'+h+'</span>',objectRow:(s={},x)=>'ROW('+[s.name,s.why,...(((x&&x.props)||[]).filter(Boolean).map(p=>Array.isArray(p)?p[1]:p)),x&&x.time,x&&x.actions,x&&x.attrs].filter(Boolean).join('|')+')'+(x&&x.selected?'[shown]':'')+(s.to&&s.to.action?'<'+s.to.action+':'+s.to.value+'>':'')+(s.to&&s.to.page?'<page:'+s.to.page+':'+((s.to.extra||{}).team||'')+'>':'')+((x&&x.under)||''),statusDot:(s,l)=>'['+s+(l?':'+l:'')+']',btn:(label,action,value)=>`<${action}:${value}>{${label}}`,btnAttrs:(label,action,value,cls,attrs)=>(action==='team-actor'?`<${action}:${value}>{${label}}${attrs||''}`:''),banner:(a,b,tone,action)=>`(${a}|${b}|${action||''})`,icon:(n)=>`{${n}}`,tile:(n)=>`{${n}}`,panel:(title,sub,body)=>`<<${title}>>`+(Array.isArray(body)?body.join(''):body),kv:(rows)=>rows.map(r=>r.join(': ')).join(' | '),
  subjectChoice:(label,id,choices,o={})=>`<picker id="${id}">`+choices.map(c=>`<button data-value="${c.value}" aria-selected="${c.value===o.selected}">${c.title}<small>${c.meta||''}</small></button>`).join('')+'</picker>',objectHead:(name,meta,actions,state,tools,details={})=>`<header class="object-header"><h1${details.headingId?` id="${details.headingId}"`:''}>${name}</h1>${state||''}${meta||''}${actions||''}${(details.facts||[]).map(([k,v])=>`FACT(${k}=${v})`).join('')}${details.id?`ID(${details.id})`:''}</header>`+(details.subject?`<div class="object-subject">${details.subject}</div>`:''),
  routeUrl:(page)=>'#page='+page,tabStrip:(label,items)=>items.map(x=>(x.on?'*':'')+x.word+' ').join(''),NAV_ATTR:'',controlAttrs:()=>'',codeWords:(code)=>String(code ?? '—'),short:(v,n=8)=>String(v||'').slice(0,n),mono:(v,n=12)=>v?'<span class="mono">'+String(v).slice(0,n)+'</span>':'—',coded:(code)=>`<span class="mono">${code ?? '—'}</span>`,actorWords:(caller)=>String(caller ?? '—'),when:(value)=>String(value ?? '—'),whenText:(value)=>String(value ?? '—'),count:(n)=>String(n),pluralText:(n,one,many,args={})=>String(Number(n)===1?one:many).replace(/\{(\w+)\}/g,(m,k)=>args&&args[k]!==undefined?args[k]:m),countText:(n,one,many)=>(Number(n)===1 ? one : many).replace('{n}',String(n)),table:(headers,rows)=>'<table>'+(Array.isArray(rows)?rows.join(''):rows)+'</table>',tr:(cells)=>'<tr>'+cells.map(String).join('|')+'</tr>'};
const runShapes=(root)=>{const src=require('node:fs').readFileSync(require('node:path').join(root,'components.js'),'utf8');const a=src.indexOf('/* ---- run shapes (round 72)');const b=src.indexOf('/* ---- end of run shapes ---- */');return src.slice(a,b)+';globalThis.stepList=stepList;globalThis.runLog=runLog;globalThis.logLine=logLine;globalThis.observationLine=observationLine;globalThis.logMove=logMove;globalThis.logHeld=logHeld;globalThis.logRetain=logRetain;';};
c.Data.readShared = (...args) => c.Data.read(...args);
vm.createContext(library.into(c,root));vm.runInContext(fs.readFileSync(path.join(root,'status.js'),'utf8'),c);vm.runInContext(library.explainCodeSource(root),c);vm.runInContext(library.prerequisiteWaysSource(root),c);vm.runInContext(runShapes(root),c);library.refusal(c,root);
for(const name of ['live-activity.js','live-team.js'])vm.runInContext(fs.readFileSync(path.join(root,name),'utf8'),c);
vm.runInContext('globalThis.A=LiveActivity;globalThis.TM=LiveTeam;',c);
// The section is read as a reader pages it: the workroom (the thread and its reader), the
// participants, then the evidence view of the same retained session.
const teamSection=c.TM.section;
{const real=teamSection;c.TM.section=()=>{const was=c.app.page;c.app.page='team-evidence';const b=real();c.app.page=was;const a=real();return a+b;};} // the evidence view is painted first: a paint consumes the one-time arrival flash
// Details are progressively disclosed now: exercise the real selection action for each
// retained exchange, keeping one copy of the scene's product evidence for the old invariants.
function readScene() {
  const saved=hash.value,q=new URLSearchParams(saved.slice(1));
  const sessions=c.TM.scene().sessions,s=q.get('team')?sessions.find(s=>s.id===q.get('team')):sessions[0];
  if(!q.get('team')&&s){q.set('team',s.id);hash.value='#'+q;} // the places round (law 123): a session's pages read a chosen session
  let out=c.TM.section();
  for(const e of s?.entries.values()||[]) if(!e.replayOf&&(!q.get('actor')||e.actor===q.get('actor')||e.recipient===q.get('actor'))) {
    {const h=new URLSearchParams(hash.value.slice(1));h.delete('event');hash.value='#'+h;} c.TM.showEvent(e.id); out+=c.TM.section().match(/<div class="team-record-inline">[\s\S]*?<\/div>/)?.[0]||''; // C4: the verification opens in place
  }
  hash.value=saved; return out;
}
let ordinal=0;
const item=(schema,payload,extra={})=>{ordinal+=1;return {ordinal,observation_id:'obs-'+ordinal,occurred_at:'2026-09-14T12:'+String(Math.floor(ordinal/60)).padStart(2,'0')+':'+String(ordinal%60).padStart(2,'0')+'Z',observed_at:'2026-09-14T12:00:00Z',schema_kind:schema,source_kind:'PRODUCT_OPERATION',source_id:'local-web:i1',source_sequence:ordinal,task_id:null,run_id:null,stage_id:null,correlation_ids:[],authority:'OPERATIONAL_ASSERTION',retention_class:'TRANSIENT_OPERATIONAL',availability:'AVAILABLE',payload,...extra};};
const page=(disposition,items,tasks={})=>({workspace_id:'qa',disposition,epoch:'e1',cursor:'e1:'+ordinal,head:ordinal,more:false,items,unavailable:0,tasks,observer:{status:'OK'},read_cost:{observations:items.length,elapsed_ms:1}});
// Events shaped like the lead's producer: subject strings only, BRIDGE_RECEIVED time kind, the
// producer's own event id, EXTERNAL_CLIENT / AGENT_PROPOSAL assigned by the boundary.
let seq=0;
const external=(kind,subject,summary,opts={})=>item('ExternalActivityObserved',{event_kind:kind,producer_id:'codex-native',producer_session:'scope1',producer_sequence:++seq,summary,summary_truncated:opts.truncated===true,subject:{source_time_kind:'BRIDGE_RECEIVED',native_event_id:opts.eventId||('evt-'+seq),...subject},task_verified:Boolean(opts.task)},{source_kind:'EXTERNAL_CLIENT',authority:'AGENT_PROPOSAL',source_id:'codex-native:scope1',source_sequence:seq,task_id:opts.task||null,correlation_ids:[subject.native_session_id||'']});
const hook=(session,agent,role,event,turn,stopActive='unknown',opts={})=>external(event==='SubagentStart'?'NATIVE_SUBAGENT_START_HOOK':'NATIVE_SUBAGENT_STOP_HOOK',{native_session_id:session,native_turn_id:turn,native_agent_id:agent,role,input_channel:'CODEX_HOOK',native_hook_event:event,terminal_state:'NOT_ESTABLISHED',stop_hook_active:stopActive},`${role}: ${event} hook observed. The child may continue; this is not proof of agent exit, Task completion or publication.`,opts);
const message=(session,agent,role,kind,text,extra={},opts={})=>external('NATIVE_COORDINATION_MESSAGE',{native_session_id:session,...(agent?{native_agent_id:agent}:{}),...(role?{role}:{}),...(kind?{message_kind:kind}:{}),message_id:extra.message_id||('m-'+(seq+1)),message_sha256:'ab'.repeat(32),message_bytes:String(Buffer.byteLength(text,'utf8')),input_channel:'ACTOR_DECLARED',...(extra.reference?{reference:extra.reference}:{}),...(extra.recipient?{recipient_id:extra.recipient}:{}),...(extra.task?{task_id:extra.task}:{})},text,opts);
const op=(operation,phase,subject,extra={},itemExtra={})=>item('ProductOperationObserved',{operation,caller:'EXTERNAL_AUTOMATION',phase,operation_ref:extra.ref||('op'+ordinal),subject,...extra},itemExtra);
(async()=>{
  const {A,TM}=c;
  const S1='sess-parent-1',S2='sess-parent-2';
  // 1. One session: assignment, start hook, a truncated answer with a reference, an objection, a
  //    product refusal on that reference, a second answer on another reference, the admission,
  //    the Task Control transition and the owner-verified artifact, the PM response and a stop
  //    hook. Then a redelivered identical event, a conflicting event under the same declared id,
  //    an unknown kind, an event outside any session, a message with no agent id, and a message
  //    whose reference is ordinary text.
  const long='x'.repeat(500);
  A.absorbPage(page('TAIL',[
    message(S1,S1,'research_lead','assignment','Screen the retained factor universe on the July input and report what survives.',{recipient:'child-analyst'}),
    hook(S1,'child-analyst','alternative_analyst','SubagentStart','turn-1'),
    message(S1,'child-analyst','alternative_analyst','question','Which seed should the screen declare?',{recipient:S1}),
    message(S1,S1,'research_lead','pm_response','Use the input default; do not change the universe.',{recipient:'child-analyst'}),
    message(S1,'child-analyst','alternative_analyst','answer',long,{reference:P1,message_id:'answer-1'},{truncated:true}),
    message(S1,'child-cro','independent_cro','objection','The PLAN behind that answer is not retained; it cannot be run as stated.',{reference:P1,recipient:S1,message_id:'obj-1'},{eventId:'evt-obj'}),
    op('EXPERIMENT_RUN','REQUESTED',{experiment_plan_hash:P1},{ref:'run1'}),
    op('EXPERIMENT_RUN','RETURNED',{experiment_plan_hash:P1},{ref:'run1',status:'REFUSED',failure_code:'research_experiment.preview_required'}),
    message(S1,'child-analyst','alternative_analyst','answer','A second PLAN on the same declaration is referenced; nothing here says it fixes the first.',{reference:P2,message_id:'answer-2'}),
    op('EXPERIMENT_PLAN','RETURNED',{plan_hash:P2,research_input_id:'factor-development'},{ref:'plan2',status:'PLANNED'}),
    op('EXPERIMENT_RUN','REQUESTED',{experiment_plan_hash:P2},{ref:'run2'}),
    op('EXPERIMENT_RUN','RETURNED',{experiment_plan_hash:P2},{ref:'run2',status:'ADMITTED',task_id:TASK,task_lifecycle:'QUEUED'},{task_id:TASK,run_id:'local-web:'+TASK}),
    item('TaskControlTransition',{task_lifecycle:'SUCCEEDED',task_class:'research_experiment',disposition:'COMMAND_RETURNED',verified_prefix_count:2,total_units:2},{task_id:TASK,run_id:'local-web:'+TASK,authority:'TASK_CONTROL_ASSERTION',source_kind:'TASK_CONTROL'}),
    item('ArtifactVerificationObserved',{artifact_kind:'ResearchExecutionEvidence',artifact_hash:'e'.repeat(64),availability:'AVAILABLE'},{task_id:TASK,run_id:'local-web:'+TASK,authority:'ARTIFACT_ASSERTION',source_kind:'PRODUCT_ARTIFACT'}),
    message(S1,S1,'research_lead','pm_response','Accepted as evidence for the screen; the CRO objection stands on record.',{recipient:'child-cro'}),
    hook(S1,'child-analyst','alternative_analyst','SubagentStop','turn-1','false'),
    message(S1,'child-cro','independent_cro','objection','The PLAN behind that answer is not retained; it cannot be run as stated.',{reference:P1,recipient:S1,message_id:'obj-1'},{eventId:'evt-obj'}), // identical content, same declared id, another observation
    message(S1,'child-cro','independent_cro','objection','Withdrawn? No: the second PLAN also lacks a retained preview.',{reference:P2,recipient:S1,message_id:'obj-1'},{eventId:'evt-obj'}), // different content, same declared id
    external('NATIVE_SOMETHING_NEW',{native_session_id:S1,native_agent_id:'child-analyst'},'a kind this reader does not know'),
    external('NATIVE_TURN_STARTED',{producer:'codex'},'an event outside any declared session'),
    message(S1,null,null,null,'no agent id, no kind, no recipient, no role',{}),
    message(S1,'child-analyst','alternative_analyst','answer','See figure G6 in the retained report.',{reference:'G6',message_id:'answer-3'}),
    message(S1,'child-analyst','alternative_analyst','answer','The earlier replay result is saved under this hash.',{reference:RESULT,message_id:'answer-4'}),
    message(S1,'child-analyst','alternative_analyst','answer','The installed strategy replay is that Task.',{reference:INSTALLED,message_id:'answer-5'}),
    // Two more observations on the Task: a worker's progress row and a stage row whose payload is
    // no longer retained. Neither is an artifact verification; neither may read as verified.
    item('WorkProgressObserved',{stage_id:'screen',completed_units:'1',total_units:'2'},{task_id:TASK,run_id:'local-web:'+TASK,authority:'OPERATIONAL_ASSERTION',source_kind:'WORKER'}),
    item('TaskStageVerified',null,{task_id:TASK,run_id:'local-web:'+TASK,authority:'TASK_CONTROL_ASSERTION',source_kind:'TASK_CONTROL',availability:'EVICTED'}),
  ],{[TASK]:{task_id:TASK,task_kind:'research_experiment',lifecycle:'SUCCEEDED',verified_stage_count:2,total_stage_count:2,current_stage:'x'}}));
  const scene=TM.scene();
  assert.equal(scene.sessions.length,1);const s=scene.sessions[0];assert.equal(s.id,S1);
  assert.equal(JSON.stringify([...s.participants.keys()].sort()),JSON.stringify(['child-analyst','child-cro',S1].sort()),'an event without an agent id is not merged into anyone');
  assert.equal(s.entries.size,15,'every admitted observation keeps its identity');
  const objections=[...s.entries.values()].filter(e=>e.declaredEvent==='evt-obj');
  assert.equal(objections.length,3);
  assert.equal(objections[1].replayOf,objections[0].id,'the identical redelivery is a replay of the first observation');
  assert.equal(JSON.stringify(objections[0].replays),JSON.stringify([objections[1].id]));
  assert.equal(objections[2].replayOf,null,'conflicting content is its own entry');
  assert.equal(JSON.stringify(objections[2].conflictsWith.sort()),JSON.stringify([objections[0].id,objections[1].id].sort()));
  assert.equal(scene.unknown.length,1,'the event outside any session is kept apart');assert.equal(s.unknownKinds.length,1,'the unknown kind stays in its session, uninterpreted');
  assert.equal(JSON.stringify([...s.references].sort()),JSON.stringify([P1,P2,RESULT,INSTALLED].sort()),'G6 is not a qualified reference');
  assert.equal(JSON.stringify(s.facts.map(f=>{const p=f.item.payload||{};return [f.item.schema_kind,p.operation||p.task_lifecycle||p.artifact_kind||null,p.status||null];})),JSON.stringify([
    ['ProductOperationObserved','EXPERIMENT_RUN',null],['ProductOperationObserved','EXPERIMENT_RUN','REFUSED'],['ProductOperationObserved','EXPERIMENT_PLAN','PLANNED'],
    ['ProductOperationObserved','EXPERIMENT_RUN',null],['ProductOperationObserved','EXPERIMENT_RUN','ADMITTED'],['TaskControlTransition','SUCCEEDED',null],['ArtifactVerificationObserved','ResearchExecutionEvidence',null],
    ['WorkProgressObserved',null,null],['TaskStageVerified',null,null],
  ]),'every product observation on the references, in recorded order');
  assert.equal(JSON.stringify(s.facts[6].followed),JSON.stringify([{ref:TASK,operation:'EXPERIMENT_RUN',status:'ADMITTED',via:P2,by:'obs-12'}]),'the Task hop is labelled with the owner return that named it');assert.equal(JSON.stringify(s.facts[4].followed),'[]','the return that named the Task does not follow itself');
  assert.equal(JSON.stringify(s.facts[1].named),JSON.stringify([P1]),'the refusal names the declared hash');
  const analyst=s.participants.get('child-analyst'),cro=s.participants.get('child-cro'),lead=s.participants.get(S1);
  assert.equal(TM.participantState(analyst).label,'1 stop hook observed · terminal state not established','a stop hook is never "done"');
  assert.equal(TM.participantState(cro).label,'declared only · no host event observed','a participant with only messages is not shown as a running child');
  assert.equal(TM.participantState(lead).label,'declared only · no host event observed');
  let html=readScene();
  assert.ok(html.includes('Main PM · declared foreground conversation'),'the parent session id with research_lead is the declared foreground PM');
  assert.ok(html.includes('alternative_analyst') && html.includes('independent_cro'));
  for(const kind of ['Assignment','Question','PM response','Answer','Objection'])assert.ok(html.includes(' · '+kind),'kind told apart: '+kind);
  assert.ok(html.includes('<strong>Main PM</strong><span class="team-mention" tabindex="0" data-tip="child-analyst">→ @Alternative Analyst</span><span class="team-kind">· Assignment'),'assignment names its recipient');
  assert.ok(html.includes('[partial:the first 500 characters]'),'the preview is marked');assert.ok(html.includes('500 bytes declared'));
  assert.ok(html.includes('the full original is not held by this feed')&&!html.includes('the full original is the referenced'),'no claim that the reference holds the original');
  assert.ok(html.includes('declared · not resolved'),'a reference is a declaration until resolved');assert.ok(html.includes('<team-resolve:'+P1+'>'));
  assert.ok(html.includes('also admitted as obs-17 · identical content under the same declared event id'),'the replay names its observation instead of a delivery count');
  assert.ok(!html.includes('delivered'),'no inferred delivery counts');
  assert.ok(html.includes('Withdrawn? No: the second PLAN also lacks a retained preview.'),'the conflicting objection stays visible');
  assert.ok(html.includes('[blocked:conflicting content · same declared event id as obs-6, obs-17]'));
  assert.ok(html.includes('Terminal state')&&html.includes('NOT_ESTABLISHED')&&html.includes('Stop hook active: <span class="mono">false'));
  assert.ok(html.includes('Codex hook input · not host-authenticated')&&html.includes('Actor-declared text · not host-verified'));
  assert.ok(html.includes('agent id not declared · Message kind not declared'),'missing metadata is named, not filled');
  assert.ok(html.includes('1 event without an agent id'));
  assert.ok(html.includes('G6 <span class="muted">· not a qualified product reference</span>'),'ordinary text is shown as a declaration');
  assert.ok(html.includes('EXPERIMENT_RUN · product refusal')&&html.includes('research_experiment.preview_required'),'the refusal is its own observation');
  assert.ok(html.includes('EXPERIMENT_RUN · product admission')&&html.includes('ResearchExecutionEvidence · owner-verified artifact')&&html.includes('research_experiment · Task Control fact'));
  // the inspector's evidence (the workroom's product card repeats the last facts as rows, round 16)
  // the product evidence log, past the workroom (whose product card repeats the last facts as rows and,
  // since round 24b, follows the inspector in the side column)
  const inspected=html.split('id="teamProductFacts"')[1].split('</section>')[0]; // C4: Observations is one column, its list one section
  assert.ok(inspected.indexOf('product refusal')<inspected.indexOf('product admission')&&inspected.indexOf('product admission')<inspected.indexOf('owner-verified artifact'),'recorded order');
  assert.equal((inspected.match(/owner-verified artifact/g)||[]).length,1,'only the compatible artifact verification reads as owner-verified');
  assert.ok(html.includes('WorkProgressObserved · observation')&&html.includes('[metadata:recorded]'),'a worker progress row is an observation, labelled by its schema');
  assert.ok(html.includes('TaskStageVerified · observation')&&html.includes('[metadata:payload not retained]'),'an unretained payload is said, not filled');
  assert.ok(html.includes('WORKER · OPERATIONAL_ASSERTION')&&html.includes('artifact owner · ARTIFACT_ASSERTION')&&html.includes('data-observation="'),'every fact row keeps its source and authority; its observation id is the row key (round 76)');
  assert.ok(html.includes('· current state')&&html.includes('Task Control projection, read now'),'the current Task state is shown apart from the history');
  assert.ok(!html.includes('correct'),'no correction is inferred between the two PLANs');
  assert.ok(html.includes('NATIVE_SOMETHING_NEW (codex-native)'),'unknown kind listed by name');
  assert.ok(html.includes('1 external event without a native session id'));
  assert.ok(!html.includes('team-select')&&!html.includes('id="teamSession"'),'one session needs no chooser');
  assert.ok(!reads.some(p=>p.includes('/api/experiments/run')),'the scene never requests a RUN');
  // 2. Mismatched PM identity: research_lead on a child id is not the PM; the parent id without
  //    the role is not the PM; neither manufactures a foreground PM.
  A.absorbPage(page('CONTINUED',[message(S2,'child-x','research_lead','assignment','I claim to lead.',{recipient:'child-y'}),message(S2,S2,'alternative_analyst','answer','The parent id speaking as an analyst.',{})]));
  hash.value='#page=tasks&team='+S2;html=TM.section();
  assert.ok(!html.includes('Main PM · declared foreground conversation'),'no PM manufactured');assert.ok(html.includes('no main PM declared'));
  assert.ok(html.includes('No product observations are retained in the current activity window.'),'an empty window is stated as retention, not absence (the Evidence page says it, N5: the dock is the way there)');assert.ok(!html.includes('No product operation, Task or artifact names'));
  assert.ok(html.includes('declared research_lead, not the parent session'));assert.ok(html.includes('the parent session id without the research_lead role'));
  hash.value='#page=tasks';
  // 3. Filtering by participant is a route choice: the objection and the PM's reply to the CRO stay.
  TM.select(S1);TM.showActor('child-cro');assert.equal(hash.value.includes('actor=child-cro'),true);
  html=readScene();assert.ok(html.includes('Objection'));assert.ok(!html.includes(' · Question'),'other participants\' messages are filtered');assert.ok(html.includes('<strong>Main PM</strong><span class="team-mention" tabindex="0" data-tip="child-cro">→ @CRO</span><span class="team-kind">· PM response'));
  TM.showActor('');html=readScene();assert.ok(html.includes(' · Question'));TM.select('');
  // 4. Discovery, preview readback and verification are three answers. A hash nobody indexes is
  //    unresolved and can still be asked from the research owner (a preview state, not a
  //    verification); a Task the workbench lists is discovered, not verified, until its owner's
  //    readback answers for that exact Task; a History hit is discovered, and only the report
  //    readback verifies it; a foreign string is a declaration and is never fetched.
  TM.select(S1);await TM.resolve(P1);assert.ok(reads.some(p=>p.includes('/api/research-history')));
  assert.equal(TM.resolved().get(P1).level,'unresolved');html=readScene();assert.ok(html.includes('[metadata:unresolved]')&&html.includes('<team-verify:'+P1+'>'));
  previewStatus='EXPIRED';await TM.verify(P1);assert.equal(TM.resolved().get(P1).level,'preview');
  html=readScene();assert.ok(html.includes('[planned:preview readback · Research experiments]')&&html.includes('<research-inspect-shared:'+P1+'>'));
  assert.ok(!html.includes('[verified:verified by Research experiments]'),'a preview readback is not a result verification');
  await TM.resolve(TASK);assert.equal(TM.resolved().get(TASK).level,'discovered');
  const taskLine=TM.resolved().get(TASK);assert.ok(taskLine.note.includes('projection, not the result'));assert.equal(JSON.stringify(taskLine.open),JSON.stringify(['task',TASK]));
  assert.equal(JSON.stringify(taskLine.target),JSON.stringify({kind:'task',ref:TASK,taskKind:'research_experiment'}),'the exact target travels with the outcome');
  readbackTask='0e0e0e0e-0000-4000-8000-00000000abcd';await TM.verify(TASK);assert.equal(TM.resolved().get(TASK).level,'discovered','a readback for another Task verifies nothing');
  assert.ok(TM.resolved().get(TASK).note.includes('the owner answered for another Task'));
  readbackTask=null;await TM.verify(TASK);assert.equal(TM.resolved().get(TASK).level,'verified');assert.ok(reads.some(p=>p.includes('/api/experiments/readback?task_id='+TASK)));
  // V659: reference verification notes use the real bilingual state/kind reader.
  const referenceWords=library.words(root),originalWords={t:c.t,codeWords:c.codeWords};
  Object.assign(c,{t:referenceWords.t,codeWords:referenceWords.codeWords});
  for(const lang of ['en','zh']) {
    referenceWords.I18N.set(lang);await TM.verify(TASK);
    const note=TM.resolved().get(TASK).note;
    assert.ok(note.includes(referenceWords.codeWords('factor.screening-development')));
    assert.ok(note.includes(referenceWords.codeWords('EXPERIMENT_PUBLISHED')));
    assert.ok(!note.includes('EXPERIMENT_PUBLISHED')&&!note.includes('factor.screening-development'));
    for(const state of ['AVAILABLE','EXPIRED','INVALID','ADMITTED_AS_TASK']) {
      previewStatus=state;await TM.verify(P1);
      assert.equal(TM.resolved().get(P1).level,'preview','PLAN is a preview, not verification');
      assert.ok(TM.resolved().get(P1).note.includes(referenceWords.codeWords(state)));
      assert.ok(!TM.resolved().get(P1).note.includes(state));
    }
  }
  Object.assign(c,originalWords);previewStatus='EXPIRED';

  assert.ok(!reads.some(p=>p.includes('/api/workbench/portfolio')),'an authored experiment verifies through its readback, not the Portfolio open');
  // The bare-hash Verify flow as the button runs it: the hash is discovered as a saved result, the
  // Verify button carries the display reference, and the owner's answer lands on that same row.
  await TM.resolve(RESULT);const found=TM.resolved().get(RESULT);assert.equal(found.level,'discovered','a History hit is discovery');
  assert.equal(JSON.stringify(found.target),JSON.stringify({kind:'result',ref:RESULT}));
  html=readScene();assert.ok(html.includes('<team-verify:'+RESULT+'>'),'Verify carries the reference as written');assert.ok(!html.includes('<team-verify:result:'+RESULT+'>'));
  assert.ok(html.includes('asked as <span class="mono">result:'+RESULT.slice(0,12)+'…</span>'),'the target is shown beside the reference, a hash by its 12 (Q2)');
  await TM.verify(RESULT);assert.equal(TM.resolved().get(RESULT).level,'discovered','the owner refused: the discovery stands, nothing is verified');
  assert.equal(TM.resolved().get(RESULT).failure,'local_web.result_unknown');assert.equal(TM.resolved().has('result:'+RESULT),false,'no second row under the target');
  html=readScene();assert.ok(html.includes('owner readback failed: <span class="mono">local_web.result_unknown</span>')&&html.includes('<team-verify:'+RESULT+'>'),'the refusal is shown on the asking row and Verify is still offered');
  reportBody={result_hash:'8'.repeat(64),report_hash:'r'.repeat(64),originating_task_id:'x',used_by_task_ids:['x']};await TM.verify(RESULT);
  assert.equal(TM.resolved().get(RESULT).level,'discovered','a report for another result verifies nothing');assert.ok(TM.resolved().get(RESULT).note.includes('the owner answered for another result'));
  reportBody={result_hash:RESULT,report_hash:'r'.repeat(64),originating_task_id:'x',used_by_task_ids:['x']};await TM.verify(RESULT);
  const verified=TM.resolved().get(RESULT);assert.equal(verified.level,'verified');assert.equal(verified.failure,null);assert.equal(JSON.stringify(verified.open),JSON.stringify(['history-open','result:'+RESULT]));
  html=readScene();assert.ok(html.includes('[verified:verified by Portfolio result]')&&!html.includes('<team-verify:'+RESULT+'>')&&!html.includes('owner readback failed'));
  assert.equal(TM.resolved().has('result:'+RESULT),false,'still one row: the display reference');
  // Installed versus authored Portfolio: an installed replay Task is verified through the Portfolio
  // owner's index and the REPORT of that exact result, which must name the Task; its reader is the
  // saved result entry. A listed Task without a result stays discovered. The authored-experiment
  // route is never asked for either.
  results=[{result_hash:RESULT2,report_hash:'q'.repeat(64),program_hash:'p'.repeat(64),task_id:INSTALLED,completed_at:'2026-09-14T12:00:00Z'}];
  reportBody=(asked)=>({result_hash:asked,report_hash:'q'.repeat(64),originating_task_id:asked===RESULT2?INSTALLED:'x',used_by_task_ids:[asked===RESULT2?INSTALLED:'x']});
  await TM.resolve(INSTALLED);assert.equal(TM.resolved().get(INSTALLED).target.taskKind,'portfolio_public_development_replay');
  const asked=reads.length;await TM.verify(INSTALLED);
  const installed=TM.resolved().get(INSTALLED);assert.equal(installed.level,'verified');assert.equal(installed.owner,'Portfolio result');
  assert.equal(JSON.stringify(installed.open),JSON.stringify(['history-open','result:'+RESULT2]),'the reader is the saved result entry');
  assert.equal(JSON.stringify(reads.slice(asked)),JSON.stringify(['/api/results','/api/report?result_hash='+RESULT2]),'index, then the exact readback; no experiment readback, no Portfolio open');
  html=readScene();assert.ok(html.includes('<history-open:result:'+RESULT2+'>')&&!html.includes('<history-open:experiment:'+INSTALLED+'>'),'no experiment entry is invented for an installed Task');
  await TM.resolve(ORPHAN);await TM.verify(ORPHAN);assert.equal(TM.resolved().get(ORPHAN).level,'discovered');assert.ok(TM.resolved().get(ORPHAN).note.includes('no result is recorded for this Task'));
  reportBody=(asked)=>({result_hash:asked,report_hash:'q'.repeat(64),originating_task_id:'x',used_by_task_ids:['x']});await TM.resolve(INSTALLED);await TM.verify(INSTALLED);
  assert.equal(TM.resolved().get(INSTALLED).level,'discovered','a result whose REPORT does not name the Task verifies nothing');assert.ok(TM.resolved().get(INSTALLED).note.includes('the owner answered for another result'));
  assert.ok(!reads.some(p=>p.includes('/api/workbench/portfolio')),'the authored-Portfolio open is never asked for an installed Task');
  const before=reads.length;await TM.resolve('https://example.invalid/report.html');assert.equal(reads.length,before,'a foreign reference is never fetched or opened');
  assert.equal(TM.resolved().get('https://example.invalid/report.html').level,'declaration');
  TM.select('');assert.equal(TM.classify('G6'),'declaration');assert.equal(JSON.stringify(TM.productReferences(op('EXPERIMENT_RUN','RETURNED',{experiment_plan_hash:P2,research_input_id:'factor-development',note:'G6'},{status:'ADMITTED'}))),JSON.stringify([P2]),'typed extraction ignores ordinary text and non-identity fields');
  // 5. A second session with the same role names is another scene; selection travels in the hash
  //    and survives repaint; an activity row offers the scene of its own session.
  const S3='sess-parent-3';
  A.absorbPage(page('CONTINUED',[message(S3,S3,'research_lead','assignment','Another PM conversation.',{recipient:'child-analyst-2'}),hook(S3,'child-analyst-2','alternative_analyst','SubagentStart','turn-9')]));
  const three=TM.scene();assert.equal(three.sessions.length,3);assert.equal(three.sessions[0].id,S3,'newest session first');
  // the places round (law 123): the Sessions list is where a session is chosen -- every retained
  // session is its row, the shown one marked; the conversation shows the one chosen, no picker
  const sessionsHtml=()=>{const was=c.app.page;c.app.page='team-sessions';const h=TM.section();c.app.page=was;return h;};
  // the lobby (law 136): a session awaiting the Main PM is in the open group, a recorded one in the
  // folded group, whose head opens it -- the viewer's fold, kept
  html=sessionsHtml();assert.ok(html.includes('<page:team:'+S1+'>')&&!html.includes('<page:team:'+S3+'>')&&!html.includes('[shown]'),'the session awaiting the Main PM is open, the recorded one folded, none marked until one is chosen: '+html.slice(0,2400));
  c.Lobby.fold('sessions:recorded');html=sessionsHtml();assert.ok(html.includes('<page:team:'+S1+'>')&&html.includes('<page:team:'+S3+'>')&&prefs.get('lobby.sessions').folded.recorded===false,'the recorded group opens and stays open');
  TM.select(S3);html=sessionsHtml();assert.ok(/ROW\(Another PM conversation\|/.test(html)&&!html.includes('[shown]'),'a session is named by its title (F4: its first sentence where no rule of the verb table holds); the lobby marks none -- none is open while it is shown (the user, 2026-09-24: back from a session, its row stayed selected): '+html.slice(0,600));
  html=TM.section();assert.ok(html.includes('Another PM conversation.'),'the chosen session is shown');
  TM.select(S1);assert.ok(hash.value.includes('team='+S1));assert.ok(sessionsHtml().includes('ROW(Another PM conversation|'),'the Sessions list tells the sessions apart by their titles');html=TM.section();assert.ok(!html.includes('Another PM conversation.'));assert.ok(html.includes('Screen the retained factor universe'));
  A.absorbPage(page('CONTINUED',[message(S3,'child-analyst-2','alternative_analyst','answer','later work in the other session',{})]));
  html=TM.section();assert.ok(html.includes('Screen the retained factor universe')&&!html.includes('later work in the other session'),'a repaint with new rows keeps the selected session');
  assert.ok(A.section().includes('<team-open:'+S3+'>'),'the feed row of a native event offers its session\'s scene');
  TM.open(S3);assert.equal(JSON.stringify(navigations.at(-1)),JSON.stringify(['team',{team:S3,actor:'',event:''}]),'an explicit entry names its session and clears the exchange: never another session\'s retained one');
  assert.equal(JSON.stringify(closes),'["undefined"]','the Task inspector closes with the scene entry (round 66: no drawer, no deferred close)');
  assert.equal(frames.length,1,'the scene heading is brought into view after the page frame');frames.pop()();
  // 6. A feed reset drops the retained groups; a selection that no longer exists says so.
  TM.select(S1);A.absorbPage(page('RESET',[message(S3,S3,'research_lead','question','after the reset',{})]));
  html=TM.section();assert.ok(html.includes('Selected session not retained'));assert.ok(html.includes('<team-select:>'));
  TM.select('');html=TM.section();assert.ok(c.app.page==='team-sessions'&&html.includes('<page:team:'+S3+'>'),'a cleared selection opens Sessions, where the retained session is offered (law 123)');
  c.app.page='team';TM.select(S3);assert.ok(TM.section().includes('after the reset'));
  // 7. The independent view pins its inspector. Only fresh continuous receipts animate;
  //    neither refresh, identical redelivery nor reconnect turns history into live activity.
  assert.ok(['team','team-evidence','team-sessions'].every(p=>typeof c.PAGES[p]==='function' && c.PAGES[p]===c.PAGES.team) && !c.PAGES['team-members'],'every view of the section is the section; Participants folded into the conversation (C4)');
  const pinned=new URLSearchParams(hash.value.slice(1)).get('event');
  const fresh=message(S3,'child-analyst-2','alternative_analyst','answer','A newly received answer.',{});
  fresh.observed_at=new Date().toISOString();
  A.absorbPage(page('CONTINUED',[fresh]));
  html=TM.section();assert.ok(html.includes('just-arrived')&&html.includes('Unread'));
  assert.equal(new URLSearchParams(hash.value.slice(1)).get('event'),pinned);
  assert.ok(!TM.section().includes('just-arrived'),'ordinary rerender does not replay animation');
  TM.markSeen();A.absorbPage(page('CONTINUED',[fresh]));
  assert.ok(!TM.section().includes('team-unread'),'redelivery cannot resurrect a seen message');
  // A workspace whose research clock trails the browser's (a pinned QA scene, a skewed machine)
  // still announces: the store's commit order decides, never the stamp against Date.now().
  const trailing=message(S3,'child-analyst-2','alternative_analyst','question','Stamped by a pinned research clock.',{});
  trailing.observed_at='2026-08-05T04:00:00Z';A.absorbPage(page('CONTINUED',[trailing]));
  html=TM.section();assert.ok(html.includes('just-arrived')&&html.includes('Unread'),'a research-clock stamp is not a stale arrival');
  TM.markSeen();
  const tail=message(S3,'child-analyst-2','alternative_analyst','answer','Retained after reconnect.',{});
  tail.observed_at=new Date().toISOString();A.absorbPage(page('TAIL',[tail]));
  assert.ok(!TM.section().includes('just-arrived'));
  c.replaceHash({event:'not-retained'});assert.ok(TM.section().includes('Selected exchange is not retained'));
  // The real router preserves all old Team selectors when normalizing its former Task URL.
  const routed={URLSearchParams,PREVIEW_STATES:[],ROUTES:{tasks:[],team:[]},Data:{live:false,sessions:()=>[]},app:{inputs:[],page:'overview'},location:{hash:'#page=tasks&team=exact-session&actor=exact-actor&event=exact-event'},history:{replaceState(_a,_b,url){routed.location.hash=url;}}};
  routed.Data.readVisits = [];
  routed.Data.visitPage = page => routed.Data.readVisits.push(['visit',page]);
  routed.Data.leavePage = page => routed.Data.readVisits.push(['leave',page]);
  vm.createContext(routed);vm.runInContext(fs.readFileSync(path.join(root,'router.js'),'utf8'),routed);
  vm.runInContext('readRoute()',routed);
  assert.equal(routed.app.page,'team');assert.equal(routed.location.hash,'#page=team&team=exact-session&actor=exact-actor&event=exact-event');
  const priorRead=c.Data.read;let returnedCase=P1;
  // A case reference verifies through the owner's saved-revision read, never the evidence readback.
  c.Data.read=async p=>{if(p.startsWith('/api/goals/show?'))throw Error('team must not verify a goal through the evidence readback');return p.startsWith('/api/goals/narrative?')?{goal_hash:returnedCase,evidence_verification:'NOT_PERFORMED'}:priorRead(p);};
  await TM.resolve('case:'+P1);await TM.verify('case:'+P1);
  assert.equal(TM.resolved().get('case:'+P1).target.kind,'case');
  assert.deepEqual(Array.from(TM.resolved().get('case:'+P1).open),['goal-open',P1]);
  returnedCase=P2;await TM.verify('case:'+P1);
  assert.equal(TM.resolved().get('case:'+P1).level,'discovered');
  assert.equal(TM.resolved().get('case:'+P1).open,null,'a different revision cannot open as the requested case');
  // 8. Relationships are explicit, unambiguous associations by the established Main PM -- never a
  //    shared object, an earlier message, a declared word or an id that several exchanges declare.
  const S4='sess-parent-4',T4='0c0c0c0c-0000-4000-8000-00000000c4c4';
  A.absorbPage(page('CONTINUED',[
    message(S4,S4,'research_lead','assignment','Build the book on T4.',{recipient:'a4',reference:T4,message_id:'as-4'}),
    message(S4,S4,'research_lead','pm_response','Proceed on T4 as declared.',{recipient:'a4',reference:T4,message_id:'pm-early'}), // to the analyst, before any objection, on the same object
    message(S4,'c4','independent_cro','objection','T4 exceeds the turnover budget.',{recipient:S4,reference:T4,message_id:'obj-4'}),
    message(S4,'a4','alternative_analyst','pm_response','Declared as a PM response by the analyst.',{recipient:'c4',reference:'obj-4',message_id:'fake-pm'}),
    message(S4,'a4','alternative_analyst','answer','First message under a reused id.',{message_id:'dup-4'}),
    message(S4,'a4','alternative_analyst','answer','Second message under the same id.',{message_id:'dup-4'}),
    message(S4,S4,'research_lead','pm_response','Names an ambiguous id.',{recipient:'a4',reference:'dup-4',message_id:'pm-dup'}),
    message(S4,'c4','independent_cro','objection','A second objection with no response at all.',{recipient:S4,reference:T4,message_id:'obj-4b'}),
  ]));
  hash.value='#page=team&team='+S4;c.app.page='team';html=TM.section();
  const objRow=(id)=>html.match(new RegExp('<li class="team-exchange[^"]*" id="team-event-'+id+'"[\\s\\S]*?</li>'))?.[0]||'';
  const s4=()=>TM.scene().sessions.find(v=>v.id===S4),byId=(m)=>[...s4().entries.values()].find(e=>e.messageId===m).id; // the first observation declaring that id
  assert.ok(objRow(byId('obj-4')).includes('data-answered="false"'),'a PM message to the analyst on the same object before the objection is not its response');
  assert.ok(objRow(byId('obj-4')).includes('No recorded Main PM response · unresolved choice'));
  assert.ok(objRow(byId('obj-4')).includes('1 other message names this objection')&&objRow(byId('obj-4')).includes('<strong>Alternative Analyst</strong>')&&objRow(byId('obj-4')).includes('· PM response')&&objRow(byId('obj-4')).includes('declared pm_response, not the Main PM'),'the analyst\'s declared pm_response stays readable with its real author, not promoted');
  assert.ok(objRow(byId('pm-early')).includes('Names the same reference as')&&objRow(byId('pm-early')).includes('related, not a reply'),'a shared object is a topic relation');
  assert.ok(objRow(byId('pm-dup')).includes('declared by 2 exchanges · ambiguous, names none'),'an id several exchanges declare names none of them');
  const dupIds=[...s4().entries.values()].filter(e=>e.messageId==='dup-4').map(e=>e.id);assert.equal(dupIds.length,2);
  assert.ok(dupIds.every(id=>!objRow(byId('pm-dup')).includes('<team-event:'+id+'>')),'no last-write-wins target is offered');
  assert.ok(html.includes('2 objections awaiting the Main PM'));
  assert.ok(html.includes('Declared as a PM response by the analyst.'),'the non-PM declaration is not discarded');
  // The valid explicit reply: the established Main PM, a pm_response naming the objection's unique id.
  A.absorbPage(page('CONTINUED',[message(S4,S4,'research_lead','pm_response','Recorded: the budget stands; the book is not frozen.',{recipient:'c4',reference:'obj-4',message_id:'pm-obj-4'})]));
  html=TM.section();
  assert.ok(objRow(byId('obj-4')).includes('data-answered="true"')&&objRow(byId('obj-4')).includes('Recorded Main PM response · not a resolution; the objection stays as recorded')&&objRow(byId('obj-4')).includes('<team-event:'+byId('pm-obj-4')+'>'),'the explicit Main PM reply is recorded and read from the objection');
  assert.ok(objRow(byId('obj-4b')).includes('data-answered="false"'),'the other objection is untouched');
  assert.ok(html.includes('1 objection awaiting the Main PM'));
  assert.ok(html.includes('T4 exceeds the turnover budget.'),'a responded objection is still shown as recorded');
  assert.equal(TM.summary().unresolved,1);assert.equal(TM.summary().objections,2);assert.equal(TM.summary().session,S4,'the Overview reads the same session and the same rule');
  assert.ok(!html.includes('resolved by')&&!html.includes('accepted the correction'),'no resolution or acceptance is claimed');
  // A later message declaring the objection's id makes the id ambiguous: the association is no
  // longer attributable, and says so, rather than being settled by whichever came last.
  A.absorbPage(page('CONTINUED',[message(S4,'c4','independent_cro','objection','Reissued under the same declared id with different words.',{recipient:S4,reference:T4,message_id:'obj-4'})]));
  html=TM.section();
  assert.ok(objRow(byId('obj-4')).includes('data-answered="false"')&&objRow(byId('pm-obj-4')).includes('declared by 2 exchanges · ambiguous, names none'),'a duplicated objection id withdraws the attribution instead of picking one');
  assert.ok(html.includes('3 objections awaiting the Main PM'));
  // 9. Arrivals: a Task completion is new, not verified; only the compatible owner verification is.
  const done=item('TaskControlTransition',{task_lifecycle:'SUCCEEDED',task_class:'research_experiment',disposition:'COMMAND_RETURNED',verified_prefix_count:1,total_units:1},{task_id:T4,run_id:'local-web:'+T4,authority:'TASK_CONTROL_ASSERTION',source_kind:'TASK_CONTROL'});
  A.absorbPage(page('CONTINUED',[done]));
  html=TM.section();
  assert.ok(html.includes('New: 1 Task completion'),'a lone SUCCEEDED transition is a Task completion');
  assert.ok(!html.includes('owner-verified result')&&!html.includes('just-verified')&&!html.includes('new verified fact'),'a completion is not a verified result');
  assert.ok(!html.includes('just-verified'),'a completion alone flashes no verified fact');
  const art=item('ArtifactVerificationObserved',{artifact_kind:'ResearchExecutionEvidence',artifact_hash:'f'.repeat(64),availability:'AVAILABLE'},{task_id:T4,run_id:'local-web:'+T4,authority:'ARTIFACT_ASSERTION',source_kind:'PRODUCT_ARTIFACT'});
  A.absorbPage(page('CONTINUED',[art]));
  html=TM.section();
  assert.ok(html.includes('1 Task completion · 1 owner-verified result')&&html.includes('just-verified')&&html.includes('id="teamProductFacts"'),'the compatible owner verification is the verified arrival, said on the workroom and shown on the evidence view');
  TM.markSeen();html=TM.section();assert.ok(!html.includes('Task completion')&&!html.includes('owner-verified result'),'mark seen clears both counts');
  // 10. The question: a research case is the session's only by the Main PM's own declaration; the
  //     Case owner verifies the revision's text, never that association; several PM-declared cases
  //     are an explicit choice; a member's case is related; without a case the PM assignment stands.
  const S5='sess-parent-5',H1='3'.repeat(64),H2='4'.repeat(64),C1='case:'+H1,C2='case:'+H2;
  const narratives={[H1]:{title:'Sector-neutral refit',question:'Does the sector-neutral lane survive the causal folds?'},[H2]:{title:'Turnover budget',question:'Is the book within its turnover budget at the August holdings date?'}};
  const narrativeReads=[];
  c.Data.read=async p=>{
    if(p.startsWith('/api/goals/show?'))throw Error('team must not verify a goal through the evidence readback');
    if(p.startsWith('/api/goals/narrative?')){const h=new URLSearchParams(p.split('?')[1]).get('goal_hash');narrativeReads.push(h);const d=narratives[h];
      // the owner's narrative as it answers: the saved revision, evidence not re-verified
      return {status:'GOAL_NARRATIVE',goal_id:'11111111-1111-4111-8111-111111111111',goal_hash:h,goal:{goal_id:'11111111-1111-4111-8111-111111111111',goal_hash:h,revision:1,parent_hash:null,state:'OPEN',declaration:{title:d.title,objective:d.question,kind:'RESEARCH',scope:'The July research input.',constraints:[],criteria:[{criterion_id:'c1',text:'development IC net of cost'}],deliverables:[],budget:null,research:{purpose:'NEW_RESEARCH',comparison_design:'Causal folds.',required_stages:['FACTOR_FOUNDATION','ALPHA']},parent_goal_id:null},statements:[],references:[]},statement_context:{},head_hash:h,outcome:'QUESTION_OPEN',open_choices:[],references:[],evidence_verification:'NOT_PERFORMED',claim:'Saved revision identity is verified; references are as recorded and not re-verified against their owners. GOAL_SHOW verifies them.'};}
    return priorRead(p);};
  A.absorbPage(page('CONTINUED',[
    message(S5,S5,'research_lead','assignment','Refit on the July input.',{recipient:'a5',reference:C1,message_id:'as-5'}),
    message(S5,'a5','alternative_analyst','answer','Also relevant: the turnover case.',{recipient:S5,reference:C2,message_id:'an-5'}),
  ]));
  hash.value='#page=team&team='+S5;html=TM.section();
  assert.ok(html.includes('data-question="case-declared"')&&html.includes('Refit on the July input.')&&html.includes('<span class="mono">'+H1.slice(0,8)+'</span>'),'one PM-declared case is the candidate: the assignment of the Main PM is its question, the reference a property until verified (round 93)');
  assert.ok(html.includes('declared by Main PM · Goal not yet resolved · <span class="mono">'));
  assert.ok(html.includes('Related case references declared by members, not the session\'s question')&&html.includes('Alternative Analyst · <span class="run-ref"><span class="run-ref-kind">Goal</span><span class="mono">'+H2.slice(0,8)+'</span>'),'the member\'s case is related, never the question');
  await TM.resolve(C1);await TM.verify(C1);
  assert.deepEqual(narrativeReads,[H1],'the owner\'s narrative is read for the declared case');
  html=TM.section();
  assert.ok(html.includes('data-question="case-verified"')&&html.includes('<h1 id="teamSceneHeading">Sector-neutral refit</h1>')&&html.includes('<p class="team-question-text">Does the sector-neutral lane survive the causal folds?</p>'),'once the revision is verified the owner’s title names the session and its question is the body (F4, laws 134 and 137)');
  assert.ok(html.includes('[verified:Goal revision verified by its owner] <span class="muted">association declared by Main PM · not verified</span>'),'the revision is verified; the association stays a declaration');
  assert.ok(!html.includes('bound research case'),'no binding is claimed');
  await TM.resolve(C2);await TM.verify(C2);html=TM.section();
  assert.ok(html.includes('<h1 id="teamSceneHeading">Sector-neutral refit')&&!html.includes('<h1 id="teamSceneHeading">Turnover budget'),'a verified member-declared case is still not the question');
  A.absorbPage(page('CONTINUED',[message(S5,S5,'research_lead','question','Also decide the turnover case.',{recipient:'a5',reference:C2,message_id:'q-5'})]));
  html=TM.section();
  assert.ok(html.includes('data-question="choice"')&&html.includes('<h1 id="teamSceneHeading">2 goals declared by Main PM</h1>')&&html.includes('<span class="muted">explicit choice required · none chosen by order or recency</span>'),'two PM-declared cases are an explicit choice');
  assert.ok(!html.includes('<h1 id="teamSceneHeading">Sector-neutral refit')&&!html.includes('<h1 id="teamSceneHeading">Turnover budget'),'neither question is chosen by order');
  assert.equal((html.match(/<team-question:case:/g)||[]).length,2,'each candidate offers the choice');
  let sum=TM.summary();assert.equal(sum.session,S5);assert.equal(sum.question.kind,'choice');assert.equal(sum.question.text,'2 goals declared by Main PM');
  TM.chooseQuestion(C2);html=TM.section();
  assert.ok(html.includes('data-question="case-verified"')&&html.includes('<h1 id="teamSceneHeading">Turnover budget</h1>')&&html.includes('<p class="team-question-text">Is the book within its turnover budget at the August holdings date?</p>')&&html.includes('[ready:chosen]')&&html.includes('<team-question:'+C1+'>'),'the reader\'s choice shows that case, the other stays offered');
  sum=TM.summary();assert.equal(sum.question.kind,'case-verified');assert.equal(sum.question.text,'Is the book within its turnover budget at the August holdings date?');
  assert.equal(TM.questionSource(sum.question),'[verified:Goal revision verified by its owner] <span class="muted">association declared by Main PM · not verified</span>','the Overview says the same words');
  const S6='sess-parent-6';
  A.absorbPage(page('CONTINUED',[
    message(S6,S6,'research_lead','assignment','Screen the August universe.',{recipient:'a6',message_id:'as-6a'}),
    message(S6,S6,'research_lead','assignment','Then build the book.',{recipient:'p6',message_id:'as-6b'}),
    message(S6,'a6','alternative_analyst','answer','This is the case I mean.',{recipient:S6,reference:C1,message_id:'an-6'}),
  ]));
  hash.value='#page=team&team='+S6;html=TM.section();
  assert.ok(html.includes('data-question="assignment"')&&html.includes('<h1 id="teamSceneHeading">August screening</h1>')&&html.includes('<span class="muted">first of 2 assignments declared by Main PM · not a bound goal</span>'),'without a PM-declared case the PM assignment is the declared question, said to be the first of two');
  assert.ok(html.includes('Related case references declared by members')&&!html.includes('<h1 id="teamSceneHeading">Sector-neutral refit'),'a member\'s verified case never becomes the question');
  // 11. A global entry through the production routeUrl returns to the retained selection; explicit
  //     keys override it; "Choose a session" gives it up (Sessions, law 123); a selection the window dropped stays
  //     visibly unavailable, and the Overview says so instead of showing the newest.
  const routerCtx={URLSearchParams,PREVIEW_STATES:[],ROUTES:{tasks:[],team:[],history:[]},Data:{live:true,sessions:()=>[]},app:{inputs:[],page:'history',input:'',book:'',session:'',historyQuery:'',historyKind:'all',historyMode:'all',historySort:'source'},location:{hash:'#page=history'},history:{replaceState(_a,_b,url){routerCtx.location.hash=url;}},LiveReview:{pages:new Set(),routeContext:()=>({})},LiveStudy:{pages:new Set(),routeContext:()=>({})},LiveResearch:{routeContext:()=>({})},LiveTeam:c.TM};
  routerCtx.Data.readVisits = [];
  routerCtx.Data.visitPage = page => routerCtx.Data.readVisits.push(['visit',page]);
  routerCtx.Data.leavePage = page => routerCtx.Data.readVisits.push(['leave',page]);
  vm.createContext(routerCtx);vm.runInContext(fs.readFileSync(path.join(root,'router.js'),'utf8'),routerCtx);
  const routeUrl=vm.runInContext('routeUrl',routerCtx),keys=(url)=>Object.fromEntries(new URLSearchParams(url.slice(1)));
  hash.value='#page=team';c.app.page='team';TM.select(S4);TM.showActor('c4');TM.showEvent(byId('obj-4b'));html=TM.section();
  assert.equal(/<li class="team-exchange[^"]*is-selected[^"]*" id="team-event-([^"]+)"/.exec(html)?.[1],byId('obj-4b'),'the chosen exchange is selected on the page');
  c.app.page='history';hash.value='#page=history'; // left through the History link
  let entry=keys(routeUrl('team'));
  assert.equal(entry.team,S4);assert.equal(entry.actor,'c4');assert.equal(entry.event,byId('obj-4b'),'the navigation link carries the retained session, participant and exchange');
  hash.value=routeUrl('team');c.app.page='team';html=TM.section();
  assert.ok(html.includes('T4 exceeds the turnover budget.')&&!html.includes('Screen the August universe.'),'the entry shows the retained session, not the newest');
  assert.ok(html.includes('Showing')&&html.includes('id="team-event-'+byId('obj-4b')+'"'),'the filter and the selected exchange are back');
  entry=keys(routeUrl('team',{team:S6,actor:'',event:''}));assert.equal(entry.team,S6);assert.ok(!entry.event,'explicit keys override the retained selection (a cleared key leaves the address, 2026-09-21)');
  TM.select('');html=TM.section();entry=keys(routeUrl('team'));
  assert.ok(!entry.team&&c.app.page==='team-sessions','"Choose a session" is the one action that gives the retained selection up: the session pages open Sessions (law 123)');assert.equal(TM.summary().selected,false);
  c.app.page='team';TM.select(S4);html=TM.section();assert.equal(TM.summary().session,S4);
  A.absorbPage(page('RESET',[message(S6,'a6','alternative_analyst','answer','after another reset',{recipient:S6})]));
  html=TM.section();
  assert.ok(html.includes('Selected session not retained')&&html.includes('<team-select:>'),'a dropped selection is visibly unavailable, not replaced');
  entry=keys(routeUrl('team'));assert.equal(entry.team,S4,'the link keeps the unavailable selection until the reader chooses');
  sum=TM.summary();assert.equal(sum.missing,S4);assert.equal(sum.session,undefined,'the Overview names the missing selection instead of the newest');
  TM.select('');html=TM.section();sum=TM.summary();assert.equal(sum.session,S6);assert.equal(sum.missing,undefined);
  // 12. C2 (law 140): noise folds, the ends stay -- hooks in a row are one line with the count of the
  //     records it holds; a comment's replies keep the first and the last, the middle one line that
  //     opens in place (a follow-up to a reply flat beside them); a clamped exchange opens where it
  //     stands; a reply read quotes the exchange it names.
  const S7='sess-parent-7';
  A.absorbPage(page('CONTINUED',[
    message(S7,S7,'research_lead','assignment','Refit on the July input.',{recipient:'a7',message_id:'as-7'}),
    hook(S7,'a7','alternative_analyst','SubagentStart','turn-7'),hook(S7,'c7','independent_cro','SubagentStart','turn-7'),hook(S7,'r7','alphalattice_risk','SubagentStart','turn-7'),
    message(S7,'a7','alternative_analyst','question','Which lane is scored?',{recipient:S7,message_id:'q-7'}),
    message(S7,S7,'research_lead','pm_response','The sector-neutral lane only.',{recipient:'a7',reference:'q-7',message_id:'pm-7'}),
    message(S7,'a7','alternative_analyst','question','For August as well?',{recipient:S7,reference:'pm-7',message_id:'q-7b'}),
    message(S7,'r7','alphalattice_risk','answer','Risk reads the same lane.',{reference:'q-7',message_id:'r-7'}),
    message(S7,S7,'research_lead','pm_response','July only for now.',{recipient:'a7',reference:'q-7b',message_id:'pm-7b'}),
    message(S7,'a7','alternative_analyst','answer','y'.repeat(500),{message_id:'long-7'},{truncated:true}),
    hook(S7,'a7','alternative_analyst','SubagentStop','turn-7','false'),
  ]));
  const wasAttrs=c.btnAttrs;c.btnAttrs=(label,action,value,cls,attrs='')=>`<${action}:${value} ${attrs}>{${label}}`; // the folds' ways, drawn for this block
  hash.value='#page=team&team='+S7;c.app.page='team';html=TM.section();
  const s7=()=>TM.scene().sessions.find(v=>v.id===S7),id7=(m)=>[...s7().entries.values()].find(e=>e.messageId===m).id,hooks7=[...s7().entries.values()].filter(e=>e.kind==='hook').map(e=>e.id);
  const at=(id)=>html.indexOf('id="team-event-'+id+'"'),q7=id7('q-7'),replies7=['pm-7','q-7b','r-7','pm-7b'].map(id7),long7=id7('long-7');
  assert.ok(html.includes('<team-fold:hooks:'+hooks7[0]+' aria-expanded="false" data-fold-count="3">{<span>3 hook events · Alternative Analyst, CRO, Risk Modeling</span>'),'three hooks in a row are one line with their count');
  assert.ok(hooks7.slice(0,3).every(id=>at(id)<0)&&at(hooks7[3])>0,'the run holds its hooks folded; the lone stop hook stays its own line');
  assert.ok(at(replies7[0])>at(q7)&&at(replies7[3])>at(replies7[0])&&at(replies7[1])<0&&at(replies7[2])<0,'the comment keeps its first reply and its last, the middle folded');
  assert.ok(html.includes('<team-replies:'+q7+' aria-expanded="false" data-fold-count="2">{··· Show 2 more ···}'),'the middle is one line with the count of the replies it holds');
  assert.equal((html.match(/<ol class="team-replies">/g)||[]).length,1,'one level of replies: the follow-up to a reply sits flat beside it');
  TM.toggleReplies(q7);html=TM.section();
  assert.ok(replies7.every((id,i)=>at(id)>0&&(i===0||at(replies7[i-1])<at(id))),'opened in place, the replies keep the store\'s order');
  assert.ok(html.includes('<team-replies:'+q7+' aria-expanded="true" data-fold-count="2">{Show less}'),'the same line closes them again');
  TM.toggleReplies(q7);TM.toggleFold('hooks:'+hooks7[0]);html=TM.section();
  const runAt=html.indexOf('<team-fold:hooks:'+hooks7[0]+' aria-expanded="true"');
  assert.ok(runAt>0&&hooks7.slice(0,3).every((id,i)=>at(id)>runAt&&(i===0||at(hooks7[i-1])<at(id))),'an opened run shows its hooks under its line, in order');
  TM.toggleFold('hooks:'+hooks7[0]);TM.showEvent(hooks7[1]);html=TM.section();
  assert.ok(at(hooks7[1])>0&&html.includes('<team-event:'+hooks7[1]+' aria-expanded="true" data-fold-count="3">')&&html.includes('class="team-record-inline"'),'a run holding the hook being read opens by itself, the reading in place; its toggle closes the reading with it (C4 item 7)');
  TM.showEvent(hooks7[1]);html=TM.section();
  assert.ok(at(hooks7[1])<0&&html.includes('<li class="team-exchange team-event team-hook-run" data-kind="hooks" data-holds="'+hooks7.slice(0,3).join(' ')+'">'),'closed, the run names the records it holds');
  TM.showEvent(replies7[1]);html=TM.section();
  assert.ok(replies7.every(id=>at(id)>0),'a folded reply being read opens its run');
  assert.ok(html.includes('class="team-record-inline"')&&!html.includes('class="team-quote"'),'a reply read opens its verification in place, under the exchange it names: no quote repeats it (C4 item 7)');
  TM.showEvent(q7);html=TM.section();assert.ok(!html.includes('class="team-quote"'),'a comment quotes nothing');
  assert.ok(html.includes('<p class="team-words owner-text" data-clamp="">')&&html.includes('<team-words:'+long7+' aria-expanded="false" data-clamp-way hidden>{Show more}'),'the words carry the clamp mark; their way waits hidden until the page measures a clamp');
  assert.ok(html.includes('y'.repeat(500)+'<span class="muted"> … the feed keeps its first 500 characters</span>'),'the retained preview says where the kept words end');
  TM.toggleWords(long7);html=TM.section();
  assert.ok(html.includes('data-clamp="open"')&&html.includes('<team-words:'+long7+' aria-expanded="true" data-clamp-way >{Show less}'),'a clamped exchange opens where it stands, and closes again');
  TM.toggleWords(long7);
  // 13. C3 (law 142): the thread's Display -- Show and the density, kept per session and viewer, never
  //     an order -- the lobby's component; one member is the route's filter. Arrivals are one pill
  //     aimed at the newest, where it is: a fold that holds it is its place.
  TM.markSeen();TM.showEvent(q7);html=TM.section();
  const name7='thread.'+S7,spec7=()=>c.displays[name7],choose=(patch)=>spec7().apply(c.setDisplay(name7,patch)),actorOf=()=>new URLSearchParams(hash.value.slice(1)).get('actor')||'';
  assert.ok((html.split('<span class="team-head-tools">')[1]||'').includes('<display:'+name7+'>')&&html.includes('<span class="team-members"'),'the head carries the members (C4) and the Display');
  assert.deepEqual(spec7().shows.map(([k])=>k),['all','asks','awaiting','assignments','notes'],'Show: everything, the questions and objections, the objections awaiting the Main PM, the assignments (U25) and the decision notes');
  assert.ok(!spec7().orderings&&!spec7().groupings,'no order and no grouping: the store\'s time is the thread\'s only order');
  assert.ok(spec7().members.length===4&&spec7().members[0][1]==='Main PM','one member: every member the session declared, the Main PM first');
  choose({show:'asks'});html=TM.section();
  assert.ok(at(q7)>0&&at(id7('as-7'))<0&&at(long7)<0&&!html.includes('team-hook-run')&&at(hooks7[3])<0,'Questions and objections keeps the comments that ask with their replies; the rest and the hooks leave');
  assert.ok(html.includes('5 exchanges')&&html.includes('Showing questions and objections<display-set:'+name7+':show:all>{Show everything}'),'the head counts what is shown; the thread says what it shows and how to show everything');
  // V637: number-first quantity, independently from the label-only column.
  const countWords=library.words(root), countBefore={t:c.t,I18N:c.I18N,countText:c.countText};
  for(const lang of ['en','zh']) {
    countWords.I18N.set(lang);c.t=countWords.t;c.I18N=countWords.I18N;
    c.countText=(n,one,many)=>countWords.t(Number(n)===1?one:many,{n:String(n)});
    const counted=TM.section();
    assert.ok(counted.includes(lang==='zh'?'5 条交流':'5 exchanges'));
    assert.ok(!counted.includes('条交流 5'));
    assert.equal(countWords.t('Exchanges'),lang==='zh'?'交流':'Exchanges');
  }
  Object.assign(c,countBefore);
  hash.value='#page=team&team='+S6;html=TM.section();assert.equal(c.displays['thread.'+S6].current.show,'all','another session keeps its own Display');
  hash.value='#page=team&team='+S7;html=TM.section();assert.ok(spec7().current.show==='asks'&&prefs.get('display.'+name7).show==='asks','the session\'s Display is kept for this viewer');
  choose({show:'awaiting'});html=TM.section();assert.ok(html.includes('EMPTY(No objection awaits the Main PM.)'),'nothing awaits the Main PM: said so, nothing invented');
  const ri=spec7().members.findIndex(([,w])=>w==='Risk Modeling');choose({show:'m'+ri});html=TM.section();
  assert.ok(actorOf()==='r7'&&spec7().current.show==='m'+ri&&prefs.get('display.'+name7).show==='all'&&html.includes('Showing Risk Modeling'),'one member is the route\'s filter, read back from the route, never kept as a display choice');
  choose({show:'asks'});assert.equal(actorOf(),'','a kind clears the member filter');
  choose({density:'compact'});html=TM.section();assert.ok(html.includes('<ol class="team-thread density-compact" id="teamThread"'),'the density is the thread\'s own');
  choose({density:'comfortable',show:'all'});
  c.btnAttrs=(label,action,value,cls,attrs='')=>`<${action}:${value} ${attrs}>{${label}}`;
  const arrival=message(S7,'r7','alphalattice_risk','answer','A newly received answer on the lane.',{reference:'q-7',message_id:'r-7b'}),atFoot=message(S7,'a7','alternative_analyst','answer','And one at the foot.',{message_id:'foot-7'});
  arrival.observed_at=atFoot.observed_at=new Date().toISOString();A.absorbPage(page('CONTINUED',[arrival,atFoot]));html=TM.section();
  const foot7=id7('foot-7');
  assert.ok(html.includes('<div class="team-new-bar" data-waypoint="#team-event-'+foot7+', [data-holds~=\''+foot7+'\']" hidden><team-newest:'+foot7+' >{{arrow}<span>2 new exchanges</span>}</div>'),'arrivals are one pill aimed at the newest, hidden until the page measures its place below the view');
  const h1=hook(S7,'c7','independent_cro','SubagentStop','turn-7','false'),h2=hook(S7,'r7','alphalattice_risk','SubagentStop','turn-7','false');
  h1.observed_at=h2.observed_at=new Date().toISOString();A.absorbPage(page('CONTINUED',[h1,h2]));html=TM.section();
  const lastHook=[...s7().entries.values()].filter(e=>e.kind==='hook').at(-1).id;
  assert.ok(at(lastHook)<0&&new RegExp('data-holds="[^"]*'+lastHook+'"').test(html)&&html.includes('data-waypoint="#team-event-'+lastHook+', [data-holds~=\''+lastHook+'\']"')&&html.includes('{{arrow}<span>4 new exchanges</span>}'),'the newest folded into a run of hooks: the run is its place');
  TM.markSeen();html=TM.section();assert.ok(!html.includes('team-new-bar'),'arrivals marked seen leave no pill');
  c.btnAttrs=wasAttrs;
  // U25, U31: a usage reading is its own kind, never an unknown one, and its latest reading counts; the session names
  // the goals its rows were filed under and its usage by model; a member's tip says its models and tokens and a card
  // that differs; a decision note reads as one; a reply names the assignment it answers (reply_to)
  c.link=(label,pg,cls,extra)=>`LINK(${pg} ${JSON.stringify(extra||{})})${label}`; // the goals' links, as the builder takes them
  const S9='sess-parent-9',G9='99999999-9999-4999-8999-999999999999';
  const usage=(agent,role,counts,extra={})=>{
    const subject={native_session_id:S9,native_agent_id:agent,role,native_host:'codex',input_channel:'CODEX_SESSION_FILE',source_kind:'CODEX_SESSION_FILE',sample_time_kind:'LATEST_USAGE_RECORD_AT',last_at:'2026-09-14T12:07:00Z',model:'gpt-6',efforts:'high',pin_differs:'effort',
      responses:String(counts[0]),input_tokens:String(counts[1]),cache_read_tokens:String(counts[2]),cache_write_tokens:String(counts[3]),output_tokens:String(counts[4]),...extra};
    const observed=external('NATIVE_AGENT_USAGE',Object.fromEntries(Object.entries(subject).filter(([,value])=>value!==null&&value!==undefined&&value!=='')),'usage so far');
    delete observed.payload.subject.source_time_kind;delete observed.payload.subject.native_event_id; // usage keeps its own sample clock and canonical snapshot identity
    if(subject.native_host==='claude-code'){observed.payload.producer_id='claude-code-native';observed.source_id='claude-code-native:scope1';}
    return observed;
  };
  const assign9=message(S9,S9,'research_lead','assignment','Screen the July factors.',{recipient:'a9',message_id:'as-9'});
  const reply9=message(S9,'a9','alternative_analyst','answer','Screened.',{recipient:S9,message_id:'an-9'});
  const note9=message(S9,'a9','alternative_analyst','decision','Keep the sector-neutral lane.',{message_id:'dn-9'});
  assign9.payload.subject.goal_id=G9;reply9.payload.subject.reply_to='as-9';
  A.absorbPage(page('CONTINUED',[assign9,reply9,note9,usage('a9','alternative_analyst',[3,100,20,5,40],{input_channel:null,source_kind:null,last_at:'2026-09-14T12:06:00Z'}),usage('a9','alternative_analyst',[4,120,25,6,50])]));
  hash.value='#page=team&team='+S9;html=TM.section();
  const s9=TM.scene().sessions.find(x=>x.id===S9);
  assert.equal(s9.unknownKinds.length,0,'a usage reading is its own kind');
  const read9=[...s9.participants.get('a9').usage.values()];
  assert.ok(read9.length===1&&read9[0].input===120&&read9[0].output===50,'the latest reading of a model counts, never a sum of readings');
  assert.equal(read9[0].last_at,'2026-09-14T12:07:00Z','the sample time comes from the same latest snapshot as its counts');
  assert.equal(read9[0].source_kind,'CODEX_SESSION_FILE','the source belongs to the latest snapshot, replacing a historical unknown source');
  assert.ok(html.includes('FACT(Goals=')&&html.includes('99999999'),'the session names the goal its rows were filed under: '+html.slice(html.indexOf('FACT('),html.indexOf('FACT(')+600));
  assert.ok(!html.includes('FACT(Usage='),'the head holds no usage: the Participants folder is its one place (U51)');
  assert.ok(html.includes('gpt-6 (high) · 120 in')&&/differs from its card: [Ee]ffort/.test(html),html.slice(html.indexOf('team-actor:a9'),html.indexOf('team-actor:a9')+600)+'the member\'s tip: its models, tokens and a card that differs');
  assert.ok(html.includes('Decision')&&!html.includes('Message kind not declared'),'a decision note reads as one');
  const replyId=[...s9.entries.values()].find(e=>e.messageId==='an-9').id,assignId=[...s9.entries.values()].find(e=>e.messageId==='as-9').id;
  const at9=html.indexOf('id="team-event-'+assignId+'"'),ol9=html.indexOf('<ol class="team-replies">',at9),rp9=html.indexOf('id="team-event-'+replyId+'"',ol9);
  assert.ok(at9>=0&&ol9>at9&&rp9>ol9&&html.slice(ol9,rp9).includes('team-reply'),'a reply sits under the assignment its reply_to names');
  // U51: the session's Participants folder -- a row a member and model: its exchanges, its model and efforts, its
  // latest reading's tokens, a mark where it differs from its card; U196 keeps parent and child counts separate
  const was51={table:c.table,tr:c.tr,pageOf:c.pageOf,pager:c.pager,page:c.app.page,hint:c.hint,t:c.t,I18N:c.I18N};
  countWords.I18N.set('en');Object.assign(c,{hint:countWords.hint,t:countWords.t,I18N:countWords.I18N});
  c.table=(h,rows,note,o)=>'<table>'+h.map(x=>x.label).join('|')+' '+(Array.isArray(rows)?rows.join(''):rows)+(o&&o.foot?'<tfoot>'+o.foot.map(x=>String(x??'')).join('|')+'</tfoot>':'')+'</table>';
  c.tr=(cells)=>'<tr>'+cells.map(x=>String(x??'')).join('|')+'</tr>';
  c.pageOf=c.pageOf||((l)=>({shown:l,start:0,page:0,pages:1}));c.pager=c.pager||(()=>'');
  hash.value='#page=team-participants&team='+S9;c.app.page='team-participants';html=TM.section();
  assert.ok(html.replace(/<[^>]*>/g,'').includes('Agent|Activities|Model|Input|Output|Cache read|Cache written'),'the folder keeps compact column labels: '+html.slice(html.indexOf('<table>'),html.indexOf('<table>')+300));
  assert.ok(html.includes('<span class="hint" data-tip="All usage values are tokens. Input excludes cache reads and cache writes." tabindex="0">Input</span>'),'the existing keyboard-accessible header hint states token units and the uncached-input definition');
  const rows9=html.split('<tr>').slice(1),row9=rows9.find((r)=>r.includes('gpt-6'))||'';
  assert.ok(rows9[0].includes('Lead')&&rows9[0].includes('no usage read'),'the lead first; a member with no reading says so: '+rows9[0]);
  assert.ok(row9.includes('gpt-6')&&row9.includes('high')&&/120\D+50\D+25\D+6/.test(row9)&&row9.includes('differs from its card'),'a row of a member: its model, efforts, latest tokens and the card it differs from: '+row9);
  assert.ok(row9.includes('1 answer')&&row9.includes('1 message'),'its exchanges by kind: '+row9);
  assert.ok(!html.includes('<tfoot>')&&!html.includes('Recorded tokens so far'),'U196: a session foot cannot add parent and child counts without non-overlap evidence');
  assert.ok(row9.includes('Codex session file')&&row9.includes('sampled '),'the latest member reading names its source and source sample time: '+row9);
  countWords.I18N.set('zh');const zh51=TM.section();
  assert.ok(zh51.replace(/<[^>]*>/g,'').includes('智能体|活动|模型|输入|输出|缓存读取|缓存写入'),'the compact Chinese column labels keep the same meanings');
  assert.ok(zh51.includes('data-tip="所有用量的单位均为 token。输入不含缓存读取和缓存写入。" tabindex="0">输入</span>'),'the Chinese hint keeps the same token units and uncached-input definition');
  // U196: both hosts, direct members sharing a role, absent fields, and a later cumulative
  // snapshot are read independently. A known zero remains zero; an unknown count stays blank.
  countWords.I18N.set('en');
  for(const [host,source,label] of [['codex','CODEX_SESSION_FILE','Codex session file'],['claude-code','CLAUDE_CODE_SESSION_FILE','Claude Code session file']]){
    const sid='sess-196-'+host,childA='child-196-a-'+host,childB='child-196-b-'+host,unknown='child-196-unknown-'+host;
    const metadata={native_session_id:sid,native_host:host,input_channel:source,source_kind:null,pin_differs:''}; // the actual bridge retains source only in input_channel
    const declared196=[
      message(sid,sid,'research_lead','assignment','Read one exact member.',{recipient:childA}),
      message(sid,sid,'research_lead','assignment','Read the other exact member.',{recipient:childB}),
      usage(sid,'research_lead',[9,900,90,9,90],metadata),
    ]; // allocate the owner page in commit order before constructing the inspected child snapshot
    const raw196=usage(childA,'alternative_analyst',[1,11,1,0,12],metadata);
    assert.equal(Object.hasOwn(raw196.payload.subject,'source_kind'),false,'the current raw fixture has no duplicated source field');
    A.absorbPage(page('CONTINUED',[
      ...declared196,
      raw196,
      usage(childB,'alternative_analyst',[2,21,2,0,22],metadata),
      usage(unknown,'alphalattice_risk',[],{...metadata,model:'unknown-model',input_channel:'UNVERIFIED_SESSION_FILE',sample_time_kind:null,last_at:null}),
    ]));
    hash.value='#page=team-participants&team='+sid;c.app.page='team-participants';
    const first196=TM.scene().sessions.find(s=>s.id===sid),a196=first196.participants.get(childA),b196=first196.participants.get(childB);
    assert.equal(a196.usage.get('gpt-6').source_kind,source,'the real input_channel-only event retains its exact source');
    assert.equal(b196.usage.get('gpt-6').source_kind,source,'the same-role sibling retains its independently projected source');
    assert.equal(first196.participants.get(unknown).usage.get('unknown-model').source_kind,null,'an unadmitted channel cannot infer a source from the host');
    assert.equal(a196.usage.get('gpt-6').input,11);assert.equal(b196.usage.get('gpt-6').input,21,'members sharing a role retain their exact native ids');
    assert.equal(JSON.stringify([...first196.participants.get(unknown).usage.values()].map(u=>[u.responses,u.input,u.output,u.cacheRead,u.cacheWrite])),JSON.stringify([[null,null,null,null,null]]),'missing counters supply no measured zero');
    let shown196=TM.section(),rows196=shown196.split('<tr>').slice(1);
    assert.equal(rows196.filter(row=>row.includes('gpt-6')).length,3,'the parent and both same-role children each keep their own model row');
    assert.ok(rows196.filter(row=>row.includes('gpt-6')).every(row=>row.includes(label)&&row.includes('sampled ')),'each member names this host\'s source and sample time');
    const unknown196=rows196.find(row=>row.includes('unknown-model'))||'';
    assert.ok(unknown196.includes('Usage source not observed')&&unknown196.includes('Sample time not observed')&&/\|\|\|\|<\/tr>/.test(unknown196),'unknown counters/source/time remain visible as unknown without zeros: '+unknown196);
    assert.ok(!shown196.includes('<tfoot>')&&!shown196.includes('Recorded tokens so far'),'neither host receives a fabricated parent/child total');
    A.absorbPage(page('CONTINUED',[usage(childA,'alternative_analyst',[3,31,3,0,32],{...metadata,last_at:'2026-09-14T12:08:00Z'})]));
    const latest196=TM.scene().sessions.find(s=>s.id===sid);
    assert.equal(latest196.participants.get(childA).usage.size,1);assert.equal(latest196.participants.get(childA).usage.get('gpt-6').input,31,'a cumulative snapshot replaces the same member/model reading');
    assert.equal(latest196.participants.get(childA).usage.get('gpt-6').last_at,'2026-09-14T12:08:00Z');
    assert.equal(latest196.participants.get(childB).usage.get('gpt-6').input,21,'one member\'s new snapshot does not rewrite its same-role sibling');
    A.absorbPage(page('CONTINUED',[usage(childA,'alternative_analyst',[3,31,3,0,32],{...metadata,input_channel:null,source_kind:source,last_at:'2026-09-14T12:08:00Z'})]));
    assert.equal(TM.scene().sessions.find(s=>s.id===sid).participants.get(childA).usage.get('gpt-6').source_kind,source,'a legacy explicit source remains readable without an input channel');
    for(const child of [childA,childB]){TM.openMember(child);assert.equal(navigations.at(-1)[1].actor,child,'the member route keeps the exact child id');}
  }
  for(const value of [undefined,null,'',' ','-1','1.5','1e3','0x10','NaN','Infinity',true,Number.MAX_SAFE_INTEGER+1])assert.equal(TM.usageCount(value),'','unobserved or invalid counters cannot become zero: '+String(value));
  assert.equal(TM.usageCount('0'),'0','a recorded zero is still a zero');assert.equal(TM.usageCount('12'),'12');
  for(const metadata of [{},{source_kind:'UNKNOWN',sample_time_kind:'LATEST_USAGE_RECORD_AT',last_at:'not-a-time'},{source_kind:'CODEX_SESSION_FILE',sample_time_kind:'UNKNOWN',last_at:'2026-09-14T12:08:00Z'}]){
    assert.ok(TM.usageMetadata(metadata).includes('Sample time not observed'),'a missing, malformed or unproved time does not read as sampled');
  }
  Object.assign(c,{table:was51.table,tr:was51.tr,pageOf:was51.pageOf,pager:was51.pager,hint:was51.hint,t:was51.t,I18N:was51.I18N});c.app.page=was51.page;
  // U53: the Product record leads with the session's own work; a person's reads on the Local Web are one line, closed until
  // pressed; a group whose records differ only in time does not open
  const S53='sess-parent-53',P53='5'.repeat(64);
  const originalGroupRow=c.evidenceRow,originalGroupLink=c.link,groupReadings=[],namedLinks=[];
  c.evidenceRow=(r,x,o={})=>{if(o.cls==='team-fold-row')groupReadings.push({why:String(r.why),columns:o.columns,props:o.props});return originalGroupRow(r,x,o);};
  c.link=(label,page,cls,extra)=>{namedLinks.push({page,extra});return originalGroupLink(label,page,cls,extra);};
  A.absorbPage(page('CONTINUED',[message(S53,S53,'research_lead','assignment','Re-read the book and report its turnover.',{reference:P53,recipient:'child-53',message_id:'as-53'}),
    op('EXPERIMENT_RUN','REQUESTED',{experiment_plan_hash:P53},{ref:'run53'}),op('EXPERIMENT_RUN','RETURNED',{experiment_plan_hash:P53},{ref:'run53',status:'REFUSED',failure_code:'research_experiment.preview_required'}),
    ...[1,2,3].flatMap((i)=>[op('EVIDENCE_PREVIEW','REQUESTED',{experiment_plan_hash:P53},{ref:'hp'+i,caller:'HUMAN'}),op('EVIDENCE_PREVIEW','RETURNED',{experiment_plan_hash:P53},{ref:'hp'+i,caller:'HUMAN',status:'PREVIEWED'})])]));
  hash.value='#page=team-evidence&team='+S53;c.app.page='team-evidence';html=TM.section();
  const facts53=(html.split('id="teamProductFacts"')[1]||'').split('</section>')[0];
  assert.ok(facts53.includes('Read or asked by a person on the Local Web')&&facts53.indexOf('product refusal')>=0&&facts53.indexOf('product refusal')<facts53.indexOf('Read or asked by a person'),'the members work first, the person line after it: '+facts53.slice(0,400));
  assert.ok(!facts53.includes('EVIDENCE_PREVIEW'),'the person reads stay closed until pressed');
  assert.ok(html.includes('The product\'s records about this session\'s exact references'),'the folder names its exact-reference receipts');
  TM.toggleFold('people');html=TM.section();
  const open53=(html.split('id="teamProductFacts"')[1]||'').split('</section>')[0];
  assert.ok(open53.includes('EVIDENCE_PREVIEW'),'pressed, the person groups show');
  assert.ok(!/data-fold="group:EVIDENCE_PREVIEW/.test(open53),'a group that differs only in time does not open');
  // V682: both folded and nonopening actual groups keep reference declarations beside their
  // clipped main, with the original event target and declaring actor retained in the disclosure.
  const withNames=groupReadings.filter(row=>row.props?.[0]);
  assert.ok(withNames.length,'the labelled Team fixture reaches actual reference-bearing groups');
  for(const row of withNames) {
    assert.equal(JSON.stringify(row.columns),JSON.stringify(['names']));
    assert.ok(!row.why.includes('<a '),'no reference link nests inside a group main or clipped why');
    assert.ok(String(row.props[0]).includes('names')&&String(row.props[0]).includes('declared by Main PM'),'the original declared-reference content remains in its separate fact');
  }
  const declaration53=[...TM.scene().sessions.find(session=>session.id===S53).entries.values()].find(entry=>entry.messageId==='as-53');
  assert.ok(namedLinks.some(link=>link.page==='team'&&link.extra?.team===S53&&link.extra?.event===declaration53.id),'the actual reference link still names the declaring exchange');
  c.evidenceRow=originalGroupRow;c.link=originalGroupLink;
  TM.toggleFold('people');c.app.page=was51.page;
  // U54: the session's 产出 -- the Tasks it submitted and the goals it took, from the Host's two reads by agent session
  // (never the activity feed): each Task with its final state and its owner's readback, unavailable with the owner's
  // code; each goal with the Host's check and its deliverables, apart from the artifacts; older pages from each foot
  const wasLink54=c.link;c.link=(label,page)=>`<a href="#page=${page}">${label}</a>`;
  const S54=S53,reads54=[],ID54=(n)=>'54540000-0000-4000-8000-00000000000'+n;
  const T54=(n,extra={})=>({task_id:ID54(n),task_kind:'research_experiment',lifecycle:'SUCCEEDED',goal_summary:'Refit '+n,last_activity_at:'2026-09-30T10:0'+n+':00Z',submitted_by:{vendor:'claude',session:S54,goal_id:null},final_state:'SUCCEEDED',artifact:null,...extra});
  const G54=(n,extra={})=>({goal_id:'g-54-'+n,goal_hash:String(n).repeat(64),title:'Refit goal '+n,state:'OPEN',outcome:null,sessions:[{vendor:'claude',session_id:S54}],deliverables:null,completion:null,...extra});
  let pages54={tasks:{'':{tasks:[T54(1,{artifact:{artifact_kind:'ResearchExecutionEvidence',artifact_hash:'a'.repeat(64),availability:'AVAILABLE',failure_code:null}}),
    T54(2,{task_kind:'portfolio_run',artifact:{artifact_kind:'PortfolioResearchResult',artifact_hash:null,availability:'UNAVAILABLE',failure_code:'tasks.artifact_readback_absent'}}),
    T54(3,{lifecycle:'RUNNING',final_state:null,submitted_by:{vendor:'claude',session:S54,goal_id:'g-54'}})],next_cursor:ID54(3)},
    [ID54(3)]:{tasks:[T54(4,{task_kind:'data_update',lifecycle:'BLOCKED',final_state:'BLOCKED'})],next_cursor:null}},
    goals:{'':{goals:[G54(1,{state:'COMPLETE',outcome:'ACHIEVED',deliverables:[{deliverable_id:'d1',kind:'REPORT',description:'The refit report',reference_count:2}],completion:{checked_at:'2026-09-30T11:00:00Z'}}),G54(2)],next_cursor:null}}};
  const was54={read:c.Data.read,table:c.table,tr:c.tr,btnAttrs:c.btnAttrs,LiveGoals:c.LiveGoals,page:c.app.page},settle=()=>new Promise((r)=>setImmediate(r));
  c.LiveGoals={checkLine:(g)=>'[CHECK:'+g.state+']',outcomeWord:(o)=>o?'OUTCOME:'+o:''};
  c.Data.read=async(p)=>{const which=p.startsWith('/api/tasks?')?'tasks':p.startsWith('/api/goals?')?'goals':'';if(!which)return was54.read(p);reads54.push(p);const q=new URLSearchParams(p.split('?')[1]);
    assert.equal(q.get('agent_session'),S54,'the read names the session');assert.equal(q.get('history_limit'),String(c.LIST_PAGE),'a page of table-rows');
    const b=pages54[which][q.get('history_cursor')||''];if(b instanceof Error)throw b;return b;};
  c.table=(h,rows)=>'<table>'+h.map(x=>x.label).join('|')+' '+rows.join('')+'</table>';c.tr=(cells)=>'<tr>'+cells.map(x=>String(x??'')).join('|')+'</tr>';
  c.btnAttrs=(label,action,value)=>`<${action}:${value}>{${label}}`;
  const box54=(h,title)=>(h.split('<<'+title+'>>')[1]||'').split('<<')[0];
  hash.value='#page=team-outputs&team='+S54;c.app.page='team-outputs';
  html=TM.section();assert.ok(html.includes('SKELETON(rows)')&&reads54.length===2,'the first paint asks the Host twice, the Tasks and the goals, and waits');
  await settle();html=TM.section();
  assert.equal(reads54.length,2,'a repaint does not read again');
  assert.ok(!reads.slice(-3).some((p)=>p.includes('/api/activity')),'nothing is read from the activity feed');
  const tasks54=box54(html,'Tasks'),goals54=box54(html,'Goals');
  assert.ok(tasks54.includes('Task|State|Artifact|Goal|Last activity'),'the Tasks box: one table, a Goal column where a row names one: '+tasks54.slice(0,120));
  const rows54=tasks54.split('<tr>').slice(1).map((r)=>r.split('</tr>')[0]);
  assert.ok(rows54[0].includes('<task:'+ID54(1)+'>{Refit 1}')&&rows54[0].includes('ResearchExecutionEvidence')&&rows54[0].includes('a'.repeat(64)),'a read-back artifact: its kind and hash, the Task opens: '+rows54[0]);
  assert.ok(rows54[1].includes('Unavailable')&&rows54[1].includes('tasks.artifact_readback_absent'),'one its owner cannot read back says so, with the code: '+rows54[1]);
  assert.ok(rows54[2].includes('[RUNNING:')&&rows54[2].includes('g-54')&&!rows54[2].includes('Unavailable'),'a moving Task: its lifecycle, no artifact yet, its goal: '+rows54[2]);
  assert.ok(goals54.includes('Goal|Check|Deliverables|Checked')&&!goals54.includes('ResearchExecutionEvidence'),'the Goals box apart from the artifacts: '+goals54.slice(0,120));
  const g54=goals54.split('<tr>').slice(1).map((r)=>r.split('</tr>')[0]);
  assert.ok(g54[0].includes('Refit goal 1')&&g54[0].includes('[CHECK:COMPLETE]')&&g54[0].includes('OUTCOME:ACHIEVED')&&g54[0].includes('The refit report')&&g54[0].includes('REPORT')&&g54[0].includes('2 references')&&g54[0].includes('2026-09-30T11:00:00Z'),'a complete goal: its check, its outcome as submitted, its deliverables and when it was checked: '+g54[0]);
  assert.ok(g54[1].includes('[CHECK:OPEN]')&&!g54[1].includes('reference'),'an open goal: its check, no deliverables: '+g54[1]);
  assert.ok(tasks54.includes('Older Tasks of this session are not read yet')&&tasks54.includes('team-outputs-older')&&!('outputs' in TM.counts()),'more Tasks: the foot reads them; the tab carries no count');
  TM.outputsOlder('tasks');await settle();html=TM.section();
  assert.equal(new URLSearchParams(reads54[2].split('?')[1]).get('history_cursor'),ID54(3),'the older page by the Host cursor');
  assert.ok(reads54[2].startsWith('/api/tasks?')&&html.includes('Refit 4')&&html.includes('[BLOCKED:')&&!html.includes('Older Tasks of this session'),'appended; no foot at the last page');
  TM.leaveOutputs();html=TM.section();await settle();assert.equal(reads54.length,5,'a new visit reads both again');
  pages54.tasks={'':new Error('tasks.cursor_moved_reload')};TM.outputsRead();html=TM.section();await settle();html=TM.section();
  assert.ok(html.includes('Outputs not read')&&html.includes('tasks.cursor_moved_reload')&&html.includes('team-outputs-read')&&html.includes('Refit goal 1'),'a refusal says itself and the way to read again; the goals still show');
  pages54={tasks:{'':{tasks:[],next_cursor:null}},goals:{'':{goals:[],next_cursor:null}}};TM.outputsRead();html=TM.section();await settle();html=TM.section();
  assert.ok(html.includes('This session submitted no Task and took no goal'),'none of either: the one empty state');
  const unreadable54={task_id:ID54(5),status:'REFUSED',failure_code:'task_control.projection_unavailable',detail:'The Task projection could not be read; its state remains unknown.',next_requests:{workspace:{operation:'WORKSPACE_SHOW'},storage:{operation:'STORAGE_READBACK'}}};
  pages54={tasks:{'':{tasks:[],refusals:[unreadable54],next_cursor:null}},goals:{'':{goals:[],next_cursor:null}}};
  TM.outputsRead();html=TM.section();await settle();html=TM.section();
  const refusedTasks54=box54(html,'Tasks');
  assert.ok(!html.includes('This session submitted no Task and took no goal')&&!refusedTasks54.includes('No Task submitted by this session'),'an unreadable submitted Task is not reported absent');
  assert.ok(refusedTasks54.includes(ID54(5).slice(0,8)),'the refusal names the Task whose projection could not be read: '+refusedTasks54);
  assert.ok(refusedTasks54.includes('href="#page=overview"')&&refusedTasks54.includes('href="#page=storage"'),'the session output offers Workspace and Storage recovery');
  assert.ok(!/undefined\s*\/\s*undefined/.test(refusedTasks54),'a refusal has no fabricated stage counts');
  // U118 / MA4: the real panel and refusal builders expose mixed collections' box ancestry.
  // Legacy panel stubs above deliberately print labels only; they cannot hold a nesting rule.
  const before118={panel:c.panel,sectionHead:c.sectionHead,infoMark:c.infoMark,btnAttrs:c.btnAttrs,tr:c.tr};
  const components118=fs.readFileSync(path.join(root,'components.js'),'utf8');
  const panel118=components118.slice(components118.indexOf('function panel('),components118.indexOf('/* A banner'));
  const head118=components118.slice(components118.indexOf('function sectionHead('),components118.indexOf('/* A page of a long list'));
  assert.ok(panel118&&head118,'the actual section builders are readable');
  vm.runInContext(panel118+'\n'+head118,c);
  c.btnAttrs=(label,action,value)=>`<button data-act="${action}" data-value="${value}">${label}</button>`;
  c.tr=cells=>'<tr>'+cells.map(cell=>'<td>'+String(cell??'')+'</td>').join('')+'</tr>';
  const boxes118=markup=>{
    const root={tag:'root',parent:null},stack=[root],boxes=[];
    const voids=new Set(['area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr']);
    const tags=/<!--[^]*?-->|<\/?([a-z][\w:-]*)\b(?:"[^"]*"|'[^']*'|[^'">])*\/?>/gi;
    for(const match of markup.matchAll(tags)) {
      if(!match[1])continue;
      const tag=match[1].toLowerCase();
      if(match[0].startsWith('</')) {
        const at=stack.findLastIndex(node=>node.tag===tag);
        if(at>0)for(const node of stack.splice(at))node.end=match.index+match[0].length;
        continue;
      }
      const node={tag,parent:stack.at(-1),start:match.index,end:markup.length,
        purpose:match[0].match(/\bdata-box\s*=\s*["']([^"']+)["']/i)?.[1]||''};
      if(node.purpose)boxes.push(node);
      if(!voids.has(tag)&&!match[0].endsWith('/>'))stack.push(node);
    }
    return boxes;
  };
  const inTable118=node=>{for(let parent=node.parent;parent;parent=parent.parent)if(parent.purpose==='table')return true;return false;};
  const planted118=boxes118('<section data-box="table"><div><div data-box="decision">Refused</div></div></section>');
  assert.equal(planted118.filter(node=>node.purpose==='decision'&&inTable118(node)).length,1,'the nesting reader detects a decision below a table, through an intermediate element');
  const refusedGoal118={status:'REFUSED',goal_id:ID54(6),item:'HEAD',failure_code:'goal.head_invalid',
    detail:`Goal ${ID54(6)}'s current revision cannot be read. No prior revision is inferred.`,
    next_requests:{workspace:{operation:'WORKSPACE_SHOW'},backups:{operation:'WORKSPACE_BACKUPS'},goals:{operation:'GOAL_LIST'},schema:{operation:'GOAL_SCHEMA'}}};
  for(const location118 of ['refusals','tasks']) {
    pages54={tasks:{'':{tasks:[T54(1),...(location118==='tasks'?[unreadable54]:[])],refusals:location118==='refusals'?[unreadable54]:[],next_cursor:null}},
      goals:{'':{goals:[G54(1,{deliverables:[{deliverable_id:'d118',kind:'REPORT',description:'The readable goal report',reference_count:1}]})],refused:[refusedGoal118],next_cursor:null}}};
    TM.outputsRead();html=TM.section();await settle();html=TM.section();
    const boxes=boxes118(html),contents=node=>html.slice(node.start,node.end);
    const tableOf=name=>boxes.find(node=>node.purpose==='table'&&contents(node).includes(name));
    const refusalOf=id=>boxes.find(node=>node.purpose==='decision'&&contents(node).includes(id));
    for(const [name,id] of [['Refit 1',ID54(5)],['Refit goal 1',ID54(6)]]) {
      const table=tableOf(name),decision=refusalOf(id);
      assert.ok(table&&decision,location118+': a readable row and its named refused sibling both remain');
      assert.ok(!inTable118(decision),location118+': a refusal decision has no table ancestor');
      assert.equal(decision.parent,table.parent,location118+': the refusal stands beside its table');
      assert.ok(contents(decision).includes('href="#page=overview"')&&contents(decision).includes('href="#page=storage"'),location118+': each refusal offers Workspace and Storage recovery');
    }
    assert.ok(html.includes('The readable goal report'),'a readable Goal keeps its declared deliverable when a sibling refuses');
    assert.equal(boxes.filter(node=>node.purpose==='decision'&&inTable118(node)).length,0,'no mixed Task or Goal refusal is a decision inside a table');
  }
  Object.assign(c,before118);
  c.link=wasLink54;
  Object.assign(c,{table:was54.table,tr:was54.tr,btnAttrs:was54.btnAttrs,LiveGoals:was54.LiveGoals});c.Data.read=was54.read;TM.leaveOutputs();c.app.page=was54.page;
  // A path-addressed assignment names its recorded participant on the thread; the locator
  // remains in disclosure. No path spelling, missing basis or ambiguous binding invents one.
  const recipientSession='recipient-session',recipientPath='/root/alphalattice_evidence_analyst';
  const recipientHook=(agent,overrides={})=>{
    const row=hook(recipientSession,agent,'alphalattice_evidence_analyst','SubagentStart','recipient-turn');
    Object.assign(row.payload.subject,{native_host:'codex',native_agent_path:recipientPath,native_agent_path_basis:'CODEX_SESSION_META'},overrides);
    return row;
  };
  const recipientCases=[
    ['bound',[recipientHook('recipient-child')],true],
    ['missing',[],false],
    ['unselected',[recipientHook('recipient-child',{native_agent_path_basis:'OTHER'})],false],
    ['other host',[recipientHook('recipient-child',{native_host:'claude-code'})],false],
    ['conflicting role',[recipientHook('recipient-child',{native_spawn_role:'alphalattice_cro'})],false],
    ['two participants',[recipientHook('recipient-child'),recipientHook('another-child')],false],
    ['two paths',[recipientHook('recipient-child'),recipientHook('recipient-child',{native_agent_path:'/root/another'})],false],
    ['ambiguous plus conflicting',[recipientHook('recipient-child'),recipientHook('recipient-child',{native_agent_path:'/root/another'}),recipientHook('another-child')],false],
  ];
  const recipientWords=library.words(root),recipientBefore={t:c.t};
  c.t=recipientWords.t;
  for(const lang of ['en','zh']) {
    recipientWords.I18N.set(lang);
    for(const [name,hooks,bound] of recipientCases) {
      const assignment=message(recipientSession,recipientSession,'research_lead','assignment','The path-addressed assignment.',{recipient:recipientPath});
      A.absorbPage(page('RESET',[...hooks,assignment].sort((a,b)=>a.ordinal-b.ordinal)));
      hash.value='#page=team&team='+recipientSession;c.app.page='team';
      const rendered=TM.section(),mention=rendered.match(/class="team-mention"[^>]*>[^<]*<\/span>/)?.[0]||'';
      assert.ok(mention.includes('→ @'+recipientWords.t(bound?'Alternative Analyst':'Unidentified')),lang+' '+name+': only one bound participant supplies the visible name');
      assert.ok(!mention.includes('→ @/root/'),lang+' '+name+': no internal task path as agent name');
      assert.ok(mention.includes('data-tip="'+recipientPath+'"'),lang+' '+name+': the original recipient locator stays available');
      const retained=TM.scene().sessions.find(s=>s.id===recipientSession).entries.get(assignment.observation_id);
      assert.equal(retained.recipient,recipientPath,'rendering does not rewrite the declared recipient');
      assert.equal(retained.actor,recipientSession,'rendering does not change the assignment author');
      if(bound) {
        TM.showActor('recipient-child');
        assert.ok(TM.section().includes('The path-addressed assignment.'),'the recorded recipient filter retains its assignment');
        TM.showActor('');
      }
    }
  }
  Object.assign(c,recipientBefore);
  // The accepted answer is a product projection of the child's typed deliverable. It is
  // labelled separately from a spoken turn, with its recorded submitter and admission clock.
  const acceptedSession='66000001-0000-4000-8000-000000000001',acceptedChild='66000002-0000-4000-8000-000000000001';
  const acceptedUnknown='66000003-0000-4000-8000-000000000001',acceptedWords=library.words(root),acceptedBefore={t:c.t};
  const acceptedProse='Product accepted deliverable (ACCEPTED): The admitted evidence warrants caution.';
  c.t=acceptedWords.t;
  for(const lang of ['en','zh']) {
    acceptedWords.I18N.set(lang);
    for(const timeKind of ['TASK_ADMISSION','PRODUCT_ACCEPTED_AT'])for(const submitted of [acceptedSession,acceptedUnknown]) {
      const timeLabel=acceptedWords.t(timeKind==='PRODUCT_ACCEPTED_AT'?'Answer acceptance time':'Task admission time');
      const assignment=message(acceptedSession,acceptedSession,'research_lead','assignment','Read the prepared bundle.',{recipient:acceptedChild,message_id:'accepted-assignment'});
      const started=hook(acceptedSession,acceptedChild,'alphalattice_evidence_analyst','SubagentStart','accepted-turn');
      const answer=message(acceptedSession,acceptedChild,'alphalattice_evidence_analyst','answer',acceptedProse,{recipient:acceptedSession,reference:TASK,message_id:'accepted-deliverable'});
      answer.occurred_at='2026-09-14T11:04:00Z';answer.observed_at='2026-09-15T11:10:00Z';
      Object.assign(answer.payload.subject,{input_channel:'PRODUCT_ACCEPTED_ANSWER',source_time_kind:timeKind,submitted_by:submitted,reply_to:'accepted-assignment',authorship_basis:'HOOK'});
      A.absorbPage(page('RESET',[assignment,started,answer]));
      hash.value='#page=team&team='+acceptedSession;c.app.page='team';
      const rendered=readScene(),row=rendered.match(new RegExp('<li[^>]*id="team-event-'+answer.observation_id+'"[\\s\\S]*?</li>'))?.[0]||'';
      assert.ok(row.includes('<strong>'+acceptedWords.t('Alternative Analyst')+'</strong>'),'the accepted finding keeps its exact child role');
      assert.ok(row.includes(acceptedProse),'the typed accepted prose remains exactly as recorded');
      assert.ok(row.includes('· '+acceptedWords.t('Accepted answer')),'the kind distinguishes accepted deliverable from spoken answer');
      assert.ok(row.includes('data-tip="'+acceptedWords.t('Accepted structured answer · product record, not a native spoken turn')+'"'),'the shared tooltip explains product-derived provenance');
      assert.ok(row.includes('· '+acceptedWords.t('Submitted by')+' <span tabindex="0" data-tip="'+submitted+'">'+acceptedWords.t(submitted===acceptedSession?'Main PM':'Unidentified')+'</span>'),'submitter uses only a known recorded participant label');
      assert.ok(row.includes(' · '+timeLabel+'"'),'the time tooltip names its immutable owner clock');
      const retained=TM.scene().sessions.find(s=>s.id===acceptedSession).entries.get(answer.observation_id);
      assert.equal(retained.actor,acceptedChild,'submitter provenance never changes the child author');
      assert.equal(retained.subject.submitted_by,submitted,'the declared submitter remains exact');
      assert.equal(retained.timeKind,timeKind);
      assert.equal(retained.replyTo,'accepted-assignment','the accepted answer keeps its exact assignment reply');
      assert.ok(row.includes('team-reply'),'the accepted answer is rendered under that assignment');
      assert.ok(rendered.includes(timeLabel)&&rendered.includes(acceptedWords.t('Accepted structured answer · product record, not a native spoken turn')),'the opened record retains its time and channel provenance');
      assert.ok(rendered.includes(timeLabel+': '+answer.occurred_at),'the repaired answer timestamp is labelled with its owner clock rather than Received');
    }
  }
  Object.assign(c,acceptedBefore);
  // A nominated file and a parent relay produce no accepted output. An actual owner
  // Task row and exact returned operation receipt appear independently of citations.
  const outputBefore={read:c.Data.read,link:c.link,btnAttrs:c.btnAttrs,t:c.t,page:c.app.page,hash:hash.value},outputWords=library.words(root);
  const outputLinks=[];
  c.link=(label,page,cls,extra)=>{outputLinks.push({page,extra});return outputBefore.link(label,page,cls,extra);};
  c.btnAttrs=(label,action,value)=>`<${action}:${value}>{${label}}`;c.t=outputWords.t;
  for(const lang of ['en','zh']) {
    outputWords.I18N.set(lang);
    for(const delivered of [false,true]) {
      outputLinks.length=0;
      const assignment=message(acceptedSession,acceptedSession,'research_lead','assignment','Read the nominated bundle.',{recipient:acceptedChild,reference:P1});
      const started=hook(acceptedSession,acceptedChild,'alphalattice_evidence_analyst','SubagentStart','output-turn');
      const reply=message(acceptedSession,delivered?acceptedChild:acceptedSession,delivered?'alphalattice_evidence_analyst':'research_lead',delivered?'answer':'pm_response',delivered?acceptedProse:'The nominated answer file exists; it remains unsubmitted.',{reference:delivered?TASK:P1,recipient:acceptedSession});
      if(delivered)Object.assign(reply.payload.subject,{input_channel:'PRODUCT_ACCEPTED_ANSWER',source_time_kind:'TASK_ADMISSION',submitted_by:acceptedSession});
      const receipts=delivered?[op('AGENT_ANSWER_SUBMIT','RETURNED',{}, {task_id:TASK,status:'ACCEPTED'}),
        op('AGENT_ANSWER_SUBMIT','RETURNED',{}, {task_id:ORPHAN,status:'ACCEPTED',detail:'Foreign Task receipt'})]:[];
      A.absorbPage(page('RESET',[assignment,started,reply,...receipts]));
      c.Data.read=async(url)=>{
        const q=new URLSearchParams(url.split('?')[1]);assert.equal(q.get('agent_session'),acceptedSession);
        if(url.startsWith('/api/tasks?'))return {tasks:delivered?[{task_id:TASK,task_kind:'evidence_analyst_agent_answer',goal_summary:'Accepted evidence answer',lifecycle:'SUCCEEDED',final_state:'SUCCEEDED',artifact:null,submitted_by:{vendor:'codex',session:acceptedSession},last_activity_at:'2026-10-04T22:00:00Z'}]:[],next_cursor:null};
        if(url.startsWith('/api/goals?'))return {goals:[],next_cursor:null};
        throw Error('No citation read: '+url);};
      TM.leaveOutputs();hash.value='#page=team-outputs&team='+acceptedSession;c.app.page='team-outputs';
      TM.section();await new Promise(r=>setImmediate(r));const outputs=TM.section();
      if(delivered)assert.ok(outputs.includes('Accepted evidence answer')&&outputs.includes('<task:'+TASK+'>')&&!outputs.includes(outputWords.t('This session submitted no Task and took no goal')),'the owner-attributed accepted Task opens by its exact id');
      else assert.ok(outputs.includes(outputWords.t('This session submitted no Task and took no goal'))&&outputs.includes(outputWords.t('The Host returned no Task submitted by this session and no Goal bound to it. A written answer file or a cited reference does not create either record.')),'the unsubmitted diagnostic stays empty with its recorded limit');
      hash.value='#page=team-evidence&team='+acceptedSession;c.app.page='team-evidence';const facts=TM.section();
      if(delivered)assert.ok(facts.includes('AGENT_ANSWER_SUBMIT')&&!facts.includes('Foreign Task receipt'),'only the exact accepted Task receipt joins Product record');
      else assert.ok(facts.includes(outputWords.t('No product observations are retained in the current activity window.'))&&facts.includes(outputWords.t('This view lists retained product operation receipts for the session\'s exact references. A native Start/Stop or a completion relay does not supply one.')),'no accepted product receipt is invented from the file relay');
      assert.equal(TM.counts().observations,delivered?1:0,'foreign receipts never count for this exact session');
      if(!delivered)for(const destination of ['team','team-outputs'])assert.ok(outputLinks.some(x=>x.page===destination&&x.extra?.team===acceptedSession),'quiet ways keep the exact selected session');
    }
    // Generic accepted answers are sealed artifacts, not newly submitted scientific Tasks.
    // Older receipts can lack the refs/status; the retained accepted event is still exact.
    for(const accepted of [false,true]) {
      outputLinks.length=0;
      const assignment=message(acceptedSession,acceptedSession,'research_lead','assignment','Interpret the assigned Risk Task.',{recipient:acceptedChild,reference:P1});
      const started=hook(acceptedSession,acceptedChild,'alphalattice_risk','SubagentStart','risk-output-turn');
      const prose='Product accepted deliverable (ACCEPTED): '+('The recorded risk diagnostics support the proposed exposure. '.repeat(20));
      const answer=message(acceptedSession,acceptedChild,'alphalattice_risk','answer',prose,{recipient:acceptedSession,reference:TASK,message_id:'risk-sealed-answer'});
      answer.occurred_at='2026-10-05T01:10:00Z';
      Object.assign(answer.payload.subject,{input_channel:accepted?'PRODUCT_ACCEPTED_ANSWER':'ACTOR_DECLARED',source_time_kind:'PRODUCT_ACCEPTED_AT',authorship_basis:'HOOK',submitted_by:acceptedSession,answer_reference:RESULT,bundle_reference:P1});
      const receipt=op('AGENT_ANSWER_SUBMIT','RETURNED',accepted?{answer_reference:RESULT,bundle_reference:P1}:{},{task_id:TASK,...(accepted?{}:{status:'CORRECT'})});
      const oldReceipt=op('AGENT_BUNDLE_PREPARE','RETURNED',{}, {task_id:TASK});
      A.absorbPage(page('RESET',[assignment,started,answer,receipt,oldReceipt]));
      c.Data.read=async(url)=>{
        const q=new URLSearchParams(url.split('?')[1]);assert.equal(q.get('agent_session'),acceptedSession);
        if(url.startsWith('/api/tasks?'))return {tasks:[],next_cursor:null};
        if(url.startsWith('/api/goals?'))return {goals:[],next_cursor:null};
        throw Error('No artifact or cited Task read: '+url);};
      TM.leaveOutputs();hash.value='#page=team-outputs&team='+acceptedSession;c.app.page='team-outputs';
      TM.section();await new Promise(r=>setImmediate(r));const outputs=TM.section();
      assert.ok(!outputs.includes('<task:'+TASK+'>'),'the cited scientific Task is never a submission by this session');
      if(accepted) {
        for(const value of [outputWords.t('Risk Modeling'),outputWords.t('Main PM'),RESULT,P1,answer.occurred_at])assert.ok(outputs.includes(value),'the accepted artifact retains '+value);
        assert.ok(!outputs.includes(prose),'a long answer stays in its Conversation detail instead of stretching the Outputs table');
        assert.ok(!outputs.includes(outputWords.t('This session submitted no Task and took no goal')),'the real accepted artifact prevents a false empty Outputs');
        assert.ok(outputLinks.some(x=>x.page==='team'&&x.extra?.event===answer.observation_id&&x.extra?.team===acceptedSession),'Outputs opens the retained accepted event');
      } else assert.ok(outputs.includes(outputWords.t('This session submitted no Task and took no goal'))&&!outputs.includes(prose),'CORRECT and actor-declared text are not accepted artifacts');
      outputLinks.length=0;hash.value='#page=team-evidence&team='+acceptedSession;c.app.page='team-evidence';const facts=TM.section();
      assert.equal(TM.counts().observations,2,'accepted artifacts do not change the operation receipt count');
      if(accepted) {
        assert.ok(!facts.includes(prose)&&!facts.includes(outputWords.t('Accepted answers')),'Product record does not duplicate Outputs or the answer text');
        assert.ok(!outputLinks.some(x=>x.page==='team'&&x.extra?.event===answer.observation_id&&x.extra?.team===acceptedSession),'Product record keeps operation receipts separate from answer deliverables');
        hash.value='#page=team&team='+acceptedSession+'&event='+answer.observation_id;c.app.page='team';
        const conversation=readScene();
        assert.ok(conversation.includes(prose),'the existing Conversation link preserves the whole accepted answer');
        const detail=[...conversation.matchAll(/<div class="team-record-inline">[^]*?<\/div>/g)].map(x=>x[0]).find(x=>x.includes(outputWords.t('Answer reference')))||'';
        assert.ok(detail.includes(outputWords.t('Answer reference'))&&detail.includes(outputWords.t('Bundle reference'))&&detail.includes(RESULT)&&detail.includes(P1),'the existing Conversation record exposes exact sealed refs');
        const retained=TM.scene().sessions.find(s=>s.id===acceptedSession).facts.find(f=>f.item.observation_id===receipt.observation_id);
        assert.equal(retained.item.payload.status,undefined,'artifact display does not invent an operation status');
      } else assert.ok(!facts.includes(prose),'the correction receipt alone supplies no accepted artifact');
    }
  }
  Object.assign(c,{link:outputBefore.link,btnAttrs:outputBefore.btnAttrs,t:outputBefore.t});
  c.Data.read=outputBefore.read;c.app.page=outputBefore.page;hash.value=outputBefore.hash;TM.leaveOutputs();
  // Full accepted contributions come only from the selected observation's sealed owner.
  // The excerpt, roles and operations remain unchanged; one detail survives only its selection.
  const detailBefore={read:c.Data.read,t:c.t,page:c.app.page,hash:hash.value},detailWords=library.words(root);
  const detailReads=[],detailSession='67000001-0000-4000-8000-000000000001';
  const fullText='The sealed interpretation retains every cited limitation. '.repeat(24)+'END OF SEALED ANSWER';
  const settleDetail=()=>new Promise(r=>setImmediate(r));
  c.Data.read=(url)=>new Promise((resolve,reject)=>{
    assert.ok(url.startsWith('/api/activity/external?observation_id='),'selected details reuse the external owner');
    detailReads.push({url,resolve,reject});
  });
  const selectDetail=(event)=>{c.app.page='team';hash.value='#page=team&team='+detailSession+(event?'&event='+event.observation_id:'');return teamSection();};
  const fixtureDetail=(role)=>{
    const child='detail-'+role,assignment=message(detailSession,detailSession,'research_lead','assignment','Read the assigned bundle.',{recipient:child});
    const started=hook(detailSession,child,role,'SubagentStart','detail-turn-'+role);
    const answer=message(detailSession,child,role,'answer',fullText.slice(0,500),{recipient:detailSession,reference:TASK},{truncated:true});
    Object.assign(answer.payload.subject,{input_channel:'PRODUCT_ACCEPTED_ANSWER',source_time_kind:'PRODUCT_ACCEPTED_AT',submitted_by:detailSession,authorship_basis:'HOOK',native_host:'codex',answer_reference:RESULT,bundle_reference:P1});
    A.absorbPage(page('RESET',[assignment,started,answer]));
    return {answer,assignment,started};
  };
  const detailResponse=(answer,contribution={text:fullText,references:[TASK,P1]},epoch='e1')=>({
    workspace_id:'qa',observation_id:answer.observation_id,epoch,accepted_answer:{status:'AVAILABLE',observation_id:answer.observation_id,native_session_id:detailSession,native_agent_id:answer.payload.subject.native_agent_id,native_host:'codex',role:answer.payload.subject.role,submitted_by:detailSession,answer_reference:RESULT,bundle_reference:P1,task_id:TASK,contribution}
  });
  c.t=detailWords.t;
  const detailRoles=['alphalattice_alpha','alphalattice_cro','alphalattice_data','alphalattice_evidence_analyst','alphalattice_factor','alphalattice_portfolio','alphalattice_risk'];
  for(const lang of ['en','zh']) {
    detailWords.I18N.set(lang);
    for(const role of detailRoles) {
      const {answer,assignment}=fixtureDetail(role),before=detailReads.length;
      selectDetail(null);selectDetail(assignment);
      assert.equal(detailReads.length,before,'unselected and ordinary messages never read accepted content');
      assert.ok(selectDetail(answer).includes(detailWords.t('Loading')),'selected accepted detail declares its pending read');
      selectDetail(answer);
      assert.equal(detailReads.length,before+1,'repaints share one selected detail read');
      assert.equal(new URLSearchParams(detailReads.at(-1).url.split('?')[1]).get('observation_id'),answer.observation_id,'the read names exactly the chosen observation');
      const contribution=role==='alphalattice_cro'?{recommendation:'HOLD',findings:[{text:fullText,reference:TASK}]}:{text:fullText,references:[TASK,P1]};
      detailReads.at(-1).resolve(detailResponse(answer,contribution));await settleDetail();
      const rendered=selectDetail(answer),inline=rendered.match(/<div class="team-record-inline">[^]*?<\/div>/)?.[0]||'';
      assert.ok(inline.includes(detailWords.t('Accepted answer'))&&inline.includes(JSON.stringify(contribution.text?contribution.references:contribution)),'every role uses the same owner disclosure');
      assert.ok(inline.includes('END OF SEALED ANSWER'),'the selected detail reaches beyond the feed bound');
      if(contribution.text) {
        assert.ok(inline.includes('<p class="owner-text">'+fullText+'</p>'),'generic sealed text stays owner content in both languages');
        assert.equal(inline.split('END OF SEALED ANSWER').length-1,1,'full generic text occurs once in its detail');
      }
      assert.equal(TM.scene().sessions.find(s=>s.id===detailSession).entries.get(answer.observation_id).text.length,500,'detail does not widen the retained summary');
      assert.equal(detailReads.length,before+1,'settled repaint never rereads its own read receipt');
      selectDetail(null);assert.ok(!teamSection().includes('END OF SEALED ANSWER'),'closing removes whole contribution');
    }
    const {answer}=fixtureDetail('alphalattice_risk');selectDetail(answer);
    detailReads.at(-1).resolve(detailResponse(answer,{text:fullText,references:[]}));await settleDetail();
    const emptyReferences=selectDetail(answer).match(/<div class="team-record-inline">[^]*?<\/div>/)?.[0]||'';
    assert.ok(emptyReferences.includes('<p class="owner-text">'+fullText+'</p>'),'empty references keep the whole accepted text');
    assert.ok(!emptyReferences.includes('<p>'+detailWords.t('References')+'</p>'),'empty references have no empty disclosure');
  }
  detailWords.I18N.set('en');
  for(const key of ['observation_id','native_session_id','native_agent_id','native_host','role','submitted_by','answer_reference','bundle_reference','task_id']) {
    const {answer}=fixtureDetail('alphalattice_risk');selectDetail(answer);
    const body=detailResponse(answer);body.accepted_answer[key]='foreign-'+key;
    detailReads.at(-1).resolve(body);await settleDetail();
    const rendered=selectDetail(answer);
    assert.ok(rendered.includes('The owner answered for another accepted answer.'),'mismatched '+key+' is visible');
    assert.ok(!rendered.includes('END OF SEALED ANSWER'),'mismatched '+key+' never exposes contribution');
  }
  for(const key of ['observation_id','epoch']) {
    const {answer}=fixtureDetail('alphalattice_risk');selectDetail(answer);
    const body=detailResponse(answer);body[key]='foreign-'+key;
    detailReads.at(-1).resolve(body);await settleDetail();
    assert.ok(selectDetail(answer).includes('The owner answered for another accepted answer.'),'selected envelope checks '+key);
  }
  for(const lang of ['en','zh'])for(const failure of ['unavailable','transport','malformed']) {
    detailWords.I18N.set(lang);
    const {answer}=fixtureDetail('alphalattice_risk');selectDetail(answer);
    if(failure==='transport')detailReads.at(-1).reject(Error('selected owner read failed'));
    else {
      const body=detailResponse(answer);
      if(failure==='unavailable')body.accepted_answer={status:'UNAVAILABLE',reason:'accepted_answer.not_retained'};
      else body.accepted_answer.contribution=[];
      detailReads.at(-1).resolve(body);
    }
    await settleDetail();const rendered=selectDetail(answer);
    assert.ok(rendered.includes(failure==='unavailable'?'accepted_answer.not_retained':failure==='transport'?'selected owner read failed':detailWords.t('The owner answered for another accepted answer.')),'failed selected read remains visible');
    assert.ok(rendered.includes(detailWords.t('The full accepted answer is unavailable.')),lang+': failed selected reads have translated explanation');
    assert.ok(!rendered.includes('END OF SEALED ANSWER'),'failed read does not show contribution');
  }
  detailWords.I18N.set('en');
  const firstDetail=fixtureDetail('alphalattice_risk');selectDetail(firstDetail.answer);const staleDetail=detailReads.at(-1);
  const nextDetail=fixtureDetail('alphalattice_alpha');selectDetail(nextDetail.answer);const activeDetail=detailReads.at(-1),pendingRenders=renders.length;
  staleDetail.resolve(detailResponse(firstDetail.answer,{text:'STALE SEALED ANSWER'}));await settleDetail();
  assert.equal(renders.length,pendingRenders,'another selected event cannot be repainted by a stale detail');
  assert.ok(!selectDetail(nextDetail.answer).includes('STALE SEALED ANSWER'),'another selected event never receives stale contribution');
  activeDetail.resolve(detailResponse(nextDetail.answer));await settleDetail();
  assert.ok(selectDetail(nextDetail.answer).includes('END OF SEALED ANSWER'),'current selected detail completes after stale result');
  selectDetail(null);selectDetail(nextDetail.answer);const reopened=detailReads.at(-1);
  reopened.resolve(detailResponse(nextDetail.answer));await settleDetail();
  const epochBefore=detailReads.length,epochPage={...page('RESET',[nextDetail.assignment,nextDetail.started,nextDetail.answer]),epoch:'e2',cursor:'e2:'+ordinal};
  A.absorbPage(epochPage);selectDetail(nextDetail.answer);
  assert.equal(detailReads.length,epochBefore+1,'a changed store epoch discards the selected result');
  detailReads.at(-1).resolve(detailResponse(nextDetail.answer,{text:'NEW EPOCH ANSWER'},'e2'));await settleDetail();
  assert.ok(selectDetail(nextDetail.answer).includes('NEW EPOCH ANSWER'),'new epoch has its own exact owner result');
  A.absorbPage({...page('RESET',[nextDetail.assignment,nextDetail.started]),epoch:'e2',cursor:'e2:'+ordinal});
  assert.ok(!selectDetail(nextDetail.answer).includes('NEW EPOCH ANSWER'),'retention disappearance discards the detail');
  A.absorbPage({...epochPage,disposition:'CONTINUED'});selectDetail(nextDetail.answer);const leavingDetail=detailReads.at(-1);
  c.app.page='home';TM.refresh();const leavingRenders=renders.length;
  leavingDetail.resolve(detailResponse(nextDetail.answer,{text:'LEFT PAGE ANSWER'},'e2'));await settleDetail();
  assert.equal(renders.length,leavingRenders,'leaving the Team page stops pending detail repaint');
  selectDetail(nextDetail.answer);
  assert.notEqual(detailReads.at(-1),leavingDetail,'returning reads a fresh selection instead of an unseen settled detail');
  detailReads.at(-1).resolve(detailResponse(nextDetail.answer,{text:'RETURNED PAGE ANSWER'},'e2'));await settleDetail();
  assert.ok(selectDetail(nextDetail.answer).includes('RETURNED PAGE ANSWER'),'returning displays the new owner result');
  Object.assign(c,{t:detailBefore.t});c.Data.read=detailBefore.read;c.app.page=detailBefore.page;hash.value=detailBefore.hash;TM.refresh();
  // V631 (ST2, WD7, TE12): every owner-declared authority and lifecycle uses the real
  // shared words, in both current projections and retained records. Participant prose stays authored.
  const codes=library.codes(),words=library.words(root),session631='session-v631';
  assert.ok(codes.authorities.length && codes.lifecycles.length,'the owners supplied their whole enums');
  Object.assign(c,{t:words.t,codeWords:words.codeWords,stateLine:words.stateLine});
  const ids631=codes.lifecycles.map((_,i)=>`6100000${i}-0000-4000-8000-000000000631`);
  const authored631='The participant keeps this English statement.';
  const fixture631=codes.lifecycles.flatMap((state,i)=>[
    message(session631,session631,'research_lead','answer',authored631,{reference:ids631[i]}),
    item('TaskControlTransition',{task_class:'research_experiment',task_lifecycle:state},{task_id:ids631[i],authority:codes.authorities[i%codes.authorities.length]}),
  ]);
  const projections631=Object.fromEntries(ids631.map((id,i)=>[id,{task_id:id,task_kind:'research_experiment',lifecycle:codes.lifecycles[i],verified_stage_count:0,total_stage_count:2}]));
  c.Data.tasks=()=>Object.values(projections631);
  for(const lang of ['en','zh']) {
    words.I18N.set(lang);
    const expected=codes.lifecycles.map(words.codeWords),authorityWords=codes.authorities.map(words.codeWords);
    if(lang==='zh') {
      for(const word of [...expected,...authorityWords])assert.match(word,/[\u4e00-\u9fff]/,'every owner code has Chinese words');
      assert.deepEqual([...words.I18N.untranslated()],[],'no missing authority or state key');
    }
    for(const [i,id] of ids631.entries()) {
      await TM.resolve(id);
      assert.ok(TM.resolved().get(id).note.includes(expected[i]),lang+' discovered Task uses the shared state word: '+codes.lifecycles[i]);
    }
    for(const live of [true,false]) {
      A.absorbPage(page('RESET',fixture631,live?projections631:{}));
      hash.value='#page=team&team='+session631;
      const rendered=readScene(),records=rendered.split('id="teamProductFacts"')[1].split('</section>')[0];
      const current=[...rendered.matchAll(/class="record-bundle"[\s\S]*?<\/small>/g)].map(m=>m[0]).join('');
      assert.ok(rendered.includes(authored631),'participant words keep their language');
      for(const word of authorityWords)assert.ok(records.includes(word),'authority through shared words: '+word);
      for(const [i,state] of codes.lifecycles.entries()) {
        assert.ok(records.includes(expected[i]),'recorded state through shared words: '+state);
        assert.ok(current.includes(expected[i]),(live?'current':'last recorded')+' state through shared words: '+state);
        assert.ok(!current.includes(state),'no raw lifecycle in current-state copy: '+state);
      }
      for(const authority of codes.authorities)assert.ok(!records.includes(authority),'no raw authority: '+authority);
    }
  }
  // Labelled UI fixtures only: an actual owner return's public shape, not a native run proof.
  // The sealed content is read from its selected receipt while attribution remains unobserved.
  const productBefore={read:c.Data.read,t:c.t,page:c.app.page,hash:hash.value,tasks:c.Data.tasks};
  const productWords=library.words(root),productReads=[],productSession='68000001-0000-4000-8000-000000000001';
  const productGoal='68000002-0000-4000-8000-000000000001',productSource='local-web:'+'a'.repeat(32);
  const productText='Fixture public conclusion: exact accepted owner content. '.repeat(20)+'END OF PRODUCT ANSWER';
  const settleProduct=()=>new Promise(r=>setImmediate(r));
  c.t=productWords.t;c.Data.tasks=()=>[];
  c.Data.read=(url)=>{
    if(url.startsWith('/api/tasks?'))return Promise.resolve({tasks:[],next_cursor:null});
    if(url.startsWith('/api/goals?'))return Promise.resolve({goals:[],next_cursor:null});
    assert.ok(url.startsWith('/api/activity/external?observation_id='),'fixture selected answer uses the page-owned exact Data.read API');
    const q=new URLSearchParams(url.split('?')[1]);assert.deepEqual([...q.keys()],['observation_id'],'no broad content collection or fallback selector');
    return new Promise((resolve,reject)=>productReads.push({url,resolve,reject}));
  };
  const productFixture=(role='RISK',status='ACCEPTED')=>{
    const assignment=message(productSession,productSession,'research_lead','assignment','Fixture lead assigns the exact prepared bundle.',{recipient:'/root/fixture-unobserved-child',reference:P1});
    const relay=message(productSession,productSession,'research_lead','pm_response','Fixture lead records the public receipt; native author remains unobserved.',{reference:TASK});
    const receipt=op('AGENT_ANSWER_SUBMIT','RETURNED',{agent_role:role,agent_vendor:'codex',agent_session:productSession,goal_id:productGoal,answer_reference:RESULT,bundle_reference:P1,task_id:TASK},{status,task_id:TASK},{schema_version:1,source_id:productSource,task_id:TASK,run_id:'local-web:'+TASK});
    delete receipt.schema_version; // the real /api/activity ActivityItem projection is versionless
    A.absorbPage(page('RESET',[assignment,relay,receipt]));
    return {assignment,relay,receipt};
  };
  const selectProduct=(receipt)=>{c.app.page='team';hash.value='#page=team&team='+productSession+(receipt?'&event='+receipt.observation_id:'');return teamSection();};
  const productResponse=(receipt,contribution={text:productText,references:[TASK]},epoch='e1')=>({
    workspace_id:'qa',observation_id:receipt.observation_id,epoch,
    accepted_answer:{status:'AVAILABLE',observation_id:receipt.observation_id,record_kind:'PRODUCT_OPERATION',...receipt.payload.subject,source_kind:receipt.source_kind,source_id:receipt.source_id,authority:receipt.authority,verdict:receipt.payload.status,answer_digest:'b'.repeat(64),recorded_agent:{host:'codex',session_id:'original-sealed-session',agent_id:null,role:null,model:null,efforts:[],basis:'NOT_OBSERVED'},contribution}
  });
  const productRoles=['ALPHA','CRO','DATA','ANALYST','FACTOR','PORTFOLIO','RISK'];
  for(const lang of ['en','zh']) {
    productWords.I18N.set(lang);
    for(const [index,role] of productRoles.entries()) {
      const fixture=productFixture(role,index%2?'DONE':'ACCEPTED'),{receipt}=fixture,before=productReads.length;
      const cold=selectProduct(null);
      assert.equal(productReads.length,before,'unselected owner facts never read sealed contributions');
      assert.ok(cold.includes('data-fact="'+receipt.observation_id.slice(0,8)+'"'),'the owner receipt remains a product event');
      assert.ok(!cold.includes('END OF PRODUCT ANSWER'),'the receipt preview does not contain owner content');
      const contribution=role==='CRO'?{recommendation:'HOLD',findings:[{text:productText,reference:TASK}]}:{text:productText,references:[TASK]};
      selectProduct(receipt);selectProduct(receipt);
      assert.equal(productReads.length,before+1,'one selected receipt owns one pending contribution read');
      assert.equal(new URLSearchParams(productReads.at(-1).url.split('?')[1]).get('observation_id'),receipt.observation_id);
      productReads.at(-1).resolve(productResponse(receipt,contribution));await settleProduct();
      const shown=selectProduct(receipt);
      assert.ok(shown.includes(productWords.t('Accepted answer'))&&shown.includes('END OF PRODUCT ANSWER'),lang+' '+role+': the exact selected public conclusion opens');
      assert.ok(shown.includes(productWords.t('Native author not observed')),lang+' '+role+': no HOOK or child authorship is required or invented');
      assert.ok(!shown.includes(productWords.t('Stored native attribution')),'unobserved provenance never becomes a stored native author');
      const session=TM.scene().sessions.find(s=>s.id===productSession);
      assert.deepEqual([...session.participants.keys()],[productSession],'the product role is not a new participant');
      assert.equal(TM.counts().exchanges,2,'the accepted product fact is not a member statement');
      assert.equal(TM.counts().observations,1,'the selected answer remains the same one owner operation');
      assert.equal([...session.entries.values()].filter(e=>e.channel==='PRODUCT_ACCEPTED_ANSWER').length,0,'no native accepted artifact event is fabricated');
      assert.equal(productReads.length,before+1,'settled repaint keeps its current exact selected detail');
      TM.leaveOutputs();c.app.page='team-outputs';hash.value='#page=team-outputs&team='+productSession;teamSection();await settleProduct();
      const outputs=teamSection();
      assert.ok(!outputs.includes('END OF PRODUCT ANSWER')&&!outputs.includes('<<'+productWords.t('Accepted answers')+'>>'),'product conclusions do not become native Outputs artifacts');
      TM.leaveOutputs();
    }
  }
  productWords.I18N.set('en');
  for(const key of ['record_kind','agent_role','agent_vendor','agent_session','goal_id','answer_reference','bundle_reference','task_id','source_kind','source_id','authority','observation_id','outer_observation','epoch']) {
    const {receipt}=productFixture();selectProduct(receipt);
    const body=productResponse(receipt,{text:'FOREIGN PRODUCT ANSWER'});
    if(key==='outer_observation')body.observation_id='foreign-observation';
    else if(key==='epoch')body.epoch='foreign-epoch';
    else body.accepted_answer[key]='foreign-'+key;
    productReads.at(-1).resolve(body);await settleProduct();
    const shown=selectProduct(receipt);
    assert.ok(!shown.includes('FOREIGN PRODUCT ANSWER'),'foreign '+key+' supplies no accepted content');
    assert.ok(shown.includes(productWords.t('The full accepted answer is unavailable.')),'foreign '+key+' names the failed owner read');
  }
  for(const key of ['schema_version','source_kind','source_id','authority','status','session','task','missing_bundle']) {
    const fixture=productFixture(),receipt=JSON.parse(JSON.stringify(fixture.receipt));
    if(key==='schema_version')receipt.schema_version=2;
    else if(key==='status')receipt.payload.status='CORRECT';
    else if(key==='session')receipt.payload.subject.agent_session='foreign-session';
    else if(key==='task')receipt.task_id=productGoal;
    else if(key==='missing_bundle')delete receipt.payload.subject.bundle_reference;
    else receipt[key]='foreign-'+key;
    A.absorbPage(page('RESET',[fixture.assignment,fixture.relay,receipt]));
    const before=productReads.length;selectProduct(receipt);
    assert.equal(productReads.length,before,'a cold malformed '+key+' owner receipt never starts a content read');
  }
  const firstProduct=productFixture();selectProduct(firstProduct.receipt);const staleProduct=productReads.at(-1);
  const currentProduct=productFixture('ALPHA');selectProduct(currentProduct.receipt);const activeProduct=productReads.at(-1),productPaints=renders.length;
  staleProduct.resolve(productResponse(firstProduct.receipt,{text:'STALE PRODUCT ANSWER'}));await settleProduct();
  assert.equal(renders.length,productPaints,'a previous product selection cannot repaint the current receipt');
  assert.ok(!selectProduct(currentProduct.receipt).includes('STALE PRODUCT ANSWER'),'a previous receipt never leaks its contribution');
  activeProduct.resolve(productResponse(currentProduct.receipt));await settleProduct();
  const changedProduct={...page('RESET',[currentProduct.assignment,currentProduct.relay,currentProduct.receipt]),epoch:'product-e2',cursor:'product-e2:'+ordinal};
  A.absorbPage(changedProduct);const epochReads=productReads.length;selectProduct(currentProduct.receipt);
  assert.equal(productReads.length,epochReads+1,'a new store epoch owns a new selected product read');
  productReads.at(-1).resolve(productResponse(currentProduct.receipt,{text:'NEW PRODUCT EPOCH'},'product-e2'));await settleProduct();
  assert.ok(selectProduct(currentProduct.receipt).includes('NEW PRODUCT EPOCH'));
  selectProduct(null);selectProduct(currentProduct.receipt);const leavingProduct=productReads.at(-1);
  c.app.page='home';TM.refresh();const leftProductPaints=renders.length;
  leavingProduct.resolve(productResponse(currentProduct.receipt,{text:'LEFT PRODUCT PAGE'},'product-e2'));await settleProduct();
  assert.equal(renders.length,leftProductPaints,'a product detail stops repainting after leaving Team');
  selectProduct(currentProduct.receipt);
  assert.notEqual(productReads.at(-1),leavingProduct,'returning uses a fresh exact product selection');
  productReads.at(-1).resolve(productResponse(currentProduct.receipt,{text:productText,references:[TASK]},'product-e2'));await settleProduct();
  assert.ok(!selectProduct(currentProduct.receipt).includes('LEFT PRODUCT PAGE'));
  // Usage health is an optional observer field, not an activity row or research gate.
  const usageCode='native_bridge.lead_usage_read_failed';
  const usageCatalog=JSON.parse(fs.readFileSync(path.join(__dirname,'../../src/alphalattice/interface/local_application/refusal_words.json'),'utf8'))[usageCode];
  const stableFeed={...page('CONTINUED',[]),epoch:'product-e2',cursor:changedProduct.cursor,observer:{status:'OK',native_usage:{status:'OBSERVING',reason:null}}};
  A.absorbPage(stableFeed);TM.refresh();const beforeUsageChange=renders.length;
  A.absorbPage({...stableFeed,observer:{status:'OK',native_usage:{status:'UNAVAILABLE',reason:usageCode,detail:usageCatalog.detail,next_action:usageCatalog.next_action,body:'PRIVATE-D5-NATIVE-BODY'}}});
  TM.refresh();
  assert.equal(renders.length,beforeUsageChange+1,'native usage health changes repaint Team even when the activity cursor is unchanged');
  for(const lang of ['en','zh']) {
    productWords.I18N.set(lang);const shown=selectProduct(currentProduct.receipt),visible=shown.replace(/data-tip="[^"]*"/g,'');
    assert.ok(shown.includes(productWords.t('Usage observation unavailable'))&&shown.includes(productWords.t(usageCatalog.detail)),lang+': usage unavailability uses the exact catalog guidance');
    if(lang==='zh')assert.match(productWords.t(usageCatalog.detail),/[\u4e00-\u9fff]/,'the catalog guidance is translated');
    assert.ok(!visible.includes(usageCode)&&!shown.includes('PRIVATE-D5-NATIVE-BODY'),'safe guidance keeps the code on hover and ignores raw native body');
    assert.ok(shown.includes('END OF PRODUCT ANSWER'),'optional usage failure does not block accepted research content');
    assert.equal(TM.counts().exchanges,2);assert.equal(TM.counts().observations,1);
  }
  // Default native relays are selection candidates, never HOOK artifacts. The selected
  // owner answer binds the real correlated RETURNED receipt and keeps native authorship unknown.
  const relayText='Fixture owner content returned for the exact default relay. '.repeat(12)+'END OF RELAY OWNER ANSWER';
  const relayFixture=(host='codex',extra={})=>{
    const assignment=message(productSession,productSession,'research_lead','assignment','Fixture lead assigns the exact relay bundle.',{recipient:'fixture-exact-child',reference:extra.bundle_reference||P1});
    const relay=external('NATIVE_COORDINATION_MESSAGE',{native_session_id:productSession,native_agent_id:productSession,native_host:host,role:'research_lead',message_kind:'answer',message_id:'accepted-relay-'+ordinal,
      input_channel:'PRODUCT_ACCEPTED_ANSWER',authorship_basis:'NOT_OBSERVED',bundle_role:'RISK',submitted_by:productSession,recipient_id:productSession,source_time_kind:'PRODUCT_ACCEPTED_AT',answer_reference:RESULT,bundle_reference:P1,reference:TASK,goal_id:productGoal,...extra},'Fixture accepted relay preview only.',{truncated:true});
    delete relay.payload.subject.native_event_id;
    if(host==='claude-code'){relay.payload.producer_id='claude-code-native';relay.source_id='claude-code-native:scope1';}
    const sub=relay.payload.subject,receipt=op('AGENT_ANSWER_SUBMIT','RETURNED',{agent_role:sub.bundle_role,agent_vendor:host,agent_session:productSession,goal_id:sub.goal_id,answer_reference:sub.answer_reference,bundle_reference:sub.bundle_reference,task_id:sub.reference},
      {status:'ACCEPTED',task_id:sub.reference},{source_id:productSource,task_id:sub.reference,run_id:'local-web:'+sub.reference});
    A.absorbPage(page('RESET',[assignment,relay,receipt]));
    return {assignment,relay,receipt};
  };
  const relayResponse=({relay,receipt},text=relayText)=>{
    const body=productResponse(receipt,{text,references:[receipt.task_id]});
    body.observation_id=relay.observation_id;body.accepted_answer.observation_id=relay.observation_id;
    body.accepted_answer.owner_observation_id=receipt.observation_id;
    body.accepted_answer.recorded_agent={host:relay.payload.subject.native_host,session_id:productSession,agent_id:null,role:null,model:null,efforts:[],basis:'NOT_OBSERVED'};
    return body;
  };
  for(const lang of ['en','zh'])for(const host of ['codex','claude-code']){
    productWords.I18N.set(lang);const fixture=relayFixture(host),before=productReads.length;
    const cold=selectProduct(null);assert.equal(productReads.length,before,'an unselected default relay never reads owner content');
    assert.ok(!cold.includes('END OF RELAY OWNER ANSWER'),'the native preview does not fabricate full accepted content');
    selectProduct(fixture.relay);selectProduct(fixture.relay);
    assert.equal(productReads.length,before+1,'a selected default relay has one exact pending owner read');
    assert.equal(new URLSearchParams(productReads.at(-1).url.split('?')[1]).get('observation_id'),fixture.relay.observation_id,'the alias lookup selects the exact native observation');
    productReads.at(-1).resolve(relayResponse(fixture));await settleProduct();
    const shown=selectProduct(fixture.relay);
    assert.ok(shown.includes('END OF RELAY OWNER ANSWER')&&shown.includes(productWords.t('Native author not observed')),lang+' '+host+': full owner content opens with unobserved native attribution');
    assert.ok(!shown.includes(productWords.t('Stored native attribution')),'a default relay supplies no stored HOOK author');
    const session=TM.scene().sessions.find(s=>s.id===productSession);
    assert.deepEqual([...session.participants.keys()],[productSession],'a default relay adds no guessed child participant');
    assert.equal(TM.counts().exchanges,2);assert.equal(TM.counts().observations,1,'the correlated RETURNED envelope stays a product fact');
    assert.equal(productReads.length,before+1,'an exact settled relay repaint reuses its selected content');
  }
  productWords.I18N.set('en');
  for(const key of ['record_kind','agent_role','agent_vendor','agent_session','goal_id','answer_reference','bundle_reference','task_id','source_kind','authority','observation_id','outer_observation','epoch']){
    const fixture=relayFixture();selectProduct(fixture.relay);const body=relayResponse(fixture,'FOREIGN DEFAULT RELAY CONTENT');
    if(key==='outer_observation')body.observation_id='foreign-observation';
    else if(key==='epoch')body.epoch='foreign-epoch';
    else body.accepted_answer[key]='foreign-'+key;
    productReads.at(-1).resolve(body);await settleProduct();const shown=selectProduct(fixture.relay);
    assert.ok(!shown.includes('FOREIGN DEFAULT RELAY CONTENT'),'foreign '+key+' cannot cross the default relay request/Goal binding');
    assert.ok(shown.includes(productWords.t('The full accepted answer is unavailable.')),'a mismatched '+key+' reply keeps the failed owner read visible');
  }
  for(const key of ['actor','role','submitted_by','native_host','input_channel','authorship_basis','source_time_kind','answer_reference','bundle_reference','reference','missing_bundle_role','bundle_role','availability']){
    const fixture=relayFixture(),sub=fixture.relay.payload.subject;
    if(key==='actor')sub.native_agent_id='unproved-child';
    else if(key==='missing_bundle_role')delete sub.bundle_role;
    else if(key==='availability')fixture.relay.availability='EVICTED_BY_RETENTION';
    else sub[key]='foreign-'+key;
    A.absorbPage(page('RESET',[fixture.assignment,fixture.relay,fixture.receipt]));const before=productReads.length;
    selectProduct(fixture.relay);assert.equal(productReads.length,before,'a malformed '+key+' relay never starts a content lookup');
  }
  const absentRelay=relayFixture();selectProduct(absentRelay.relay);
  productReads.at(-1).resolve({workspace_id:'qa',observation_id:absentRelay.relay.observation_id,epoch:'e1',accepted_answer:{status:'UNAVAILABLE',reason:'native_bridge.accepted_answer_not_recorded'}});await settleProduct();
  const absentShown=selectProduct(absentRelay.relay);
  assert.ok(!absentShown.includes('END OF RELAY OWNER ANSWER')&&absentShown.includes(productWords.t('The full accepted answer is unavailable.')),'an absent actual RETURNED receipt supplies an explicit unavailable read and no accepted content');
  const earlierRelay=relayFixture();selectProduct(earlierRelay.relay);const lateRelay=productReads.at(-1);
  const nextRelay=relayFixture('codex',{goal_id:'68000003-0000-4000-8000-000000000001',answer_reference:RESULT2,bundle_reference:P2,reference:INSTALLED});
  selectProduct(nextRelay.relay);const currentRelayRead=productReads.at(-1),relayPaints=renders.length;
  lateRelay.resolve(relayResponse(earlierRelay,'LATE PREVIOUS GOAL RELAY'));await settleProduct();
  assert.equal(renders.length,relayPaints,'a previous Goal relay cannot repaint the new selected receipt');
  assert.ok(!selectProduct(nextRelay.relay).includes('LATE PREVIOUS GOAL RELAY'),'late content stays with its original request and Goal');
  currentRelayRead.resolve(relayResponse(nextRelay));await settleProduct();assert.ok(selectProduct(nextRelay.relay).includes('END OF RELAY OWNER ANSWER'));
  Object.assign(c,{t:productBefore.t});c.Data.read=productBefore.read;c.Data.tasks=productBefore.tasks;c.app.page=productBefore.page;hash.value=productBefore.hash;TM.refresh();
  finish();
})().catch(e=>{console.error(e);process.exitCode=1;});
