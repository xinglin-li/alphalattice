// Read Host pages and owner answers through Team; mock only presentation and routing.
// Usage: node workbench_team_readback.cjs <modules dir> <input json> [session] [actor].
// Input: pages/history/reads/resolve/verify; output: page, exchange readers, facts and requests.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict'),library=require('./workbench_library.cjs');
module.exports.checkSessionReadback=async({c,root,library,vm,fs,path,A,TM,reads,message,page,usage,teamSection})=>{
  const reviewSession='review-native-parent',reviewThread='047ec6ef-0000-4000-8000-000000000001',reviewRef='review-thread:'+reviewThread;
  const reviewIssue=message(reviewSession,reviewSession,'research_lead','issue','Fixture bundle issued.',{reference:reviewThread,recipient:'review-native-child'});
  reviewIssue.payload.subject.input_channel='PRODUCT_REVIEW';A.absorbPage(page('CONTINUED',[reviewIssue]));TM.select(reviewSession);
  const reviewEntry=TM.scene().sessions.find(s=>s.id===reviewSession).entries.get(reviewIssue.observation_id);
  assert.equal(reviewEntry.reference,reviewRef);assert.equal(reviewEntry.qualified.kind,'review_thread');
  const reviewReads=reads.length;await TM.resolve(reviewRef);const reviewResolved=TM.resolved().get(reviewRef);
  await TM.verify(reviewRef);assert.equal(reads.length,reviewReads);assert.equal(reviewResolved.target.kind,'review_thread');
  assert.equal(reviewResolved.route.team,reviewSession);assert.equal(reviewResolved.route.event,reviewIssue.observation_id);
  TM.showEvent(reviewIssue.observation_id);const reviewShown=teamSection();
  assert.ok(reviewShown.includes(c.t('Review thread'))&&!reviewShown.includes('Task '+reviewThread.slice(0,8)),reviewShown);
  assert.ok(reviewShown.includes('LINK(team')&&reviewShown.includes(reviewIssue.observation_id));
  const windowSession='window-native-parent',windowGoal='82000001-0000-4000-8000-000000000001';
  const windowBinding=message(windowSession,windowSession,'research_lead','session_bound','Fixture scoped binding.');
  Object.assign(windowBinding.payload.subject,{input_channel:'PRODUCT_OPERATION',goal_id:windowGoal});
  const windowIssue=message(windowSession,windowSession,'research_lead','issue','Fixture retained review.',{reference:reviewThread,recipient:'window-analyst'}),windowAnswer=message(windowSession,'window-analyst','alphalattice_evidence_analyst','answer','Fixture retained answer.',{reference:reviewThread});
  const windowCro=message(windowSession,'window-cro','alphalattice_cro','answer','Fixture CRO answer.',{reference:reviewThread});
  for(const e of [windowIssue,windowAnswer,windowCro])Object.assign(e.payload.subject,{input_channel:'PRODUCT_REVIEW',goal_id:windowGoal});
  const oldUsage=Array.from({length:256},(_,i)=>usage(windowSession,'research_lead',[i+1,100,10,0,5],{native_session_id:windowSession}));
  const latestUsage=[usage(windowSession,'research_lead',[257,101,11,0,6],{native_session_id:windowSession}),usage('window-analyst','alphalattice_evidence_analyst',[2,20,0,0,8],{native_session_id:windowSession}),usage('window-cro','alphalattice_cro',[1,10,0,0,4],{native_session_id:windowSession})];
  for(const initial of ['team-sessions','team']) {
    const calls=[],address={value:'#'+new URLSearchParams({page:initial,...(initial==='team'?{team:windowSession}:{})})};
    const fresh={...c,PAGES:{},app:{page:initial},hashParams:()=>new URLSearchParams(address.value.slice(1)),replaceHash:changes=>{const q=new URLSearchParams(address.value.slice(1));for(const [k,v] of Object.entries(changes))q.set(k,v);address.value='#'+q;},
      Data:{...c.Data,workspaceStatus:'ready',offers:()=>false,readShared:async url=>{calls.push(url);const selected=new URLSearchParams(url.split('?')[1]).get('native_session_id');return {epoch:'window-epoch',items:selected?[windowIssue,windowAnswer,windowCro]:oldUsage.slice(-200),snapshots:selected?[windowBinding,...latestUsage]:[],snapshots_more:initial==='team',more:false,oldest:null};}}};
    library.context(fresh,root);for(const name of ['status.js','live-activity.js','team-members.js','live-team.js'])vm.runInContext(fs.readFileSync(path.join(root,name),'utf8'),fresh);
    vm.runInContext('globalThis.WindowTeam=LiveTeam;globalThis.WindowActivity=LiveActivity;',fresh);
    if(initial==='team-sessions')fresh.WindowActivity.absorbPage({...page('RESET',oldUsage.slice(-200)),epoch:'window-epoch'});fresh.WindowTeam.scene();await new Promise(setImmediate);
    fresh.app.page='team';fresh.WindowTeam.select(windowSession);fresh.WindowTeam.scene();await new Promise(setImmediate);
    const windowShown=fresh.WindowTeam.scene(false,true).sessions.find(s=>s.id===windowSession);
    assert.equal(windowShown.goalScope,windowGoal);assert.equal(windowShown.entries.size,4,[...windowShown.entries.values()].map(e=>e.id).join(','));assert.equal(windowShown.participants.size,3,[...windowShown.participants.keys()].join(','));
    assert.equal(windowShown.participants.get('window-analyst').usage.get('gpt-6').output,8);
    assert.equal(calls.filter(url=>url.includes('native_session_id=')).length,1);
    for(let i=0;i<3;i++)fresh.WindowTeam.section();await new Promise(setImmediate);
    assert.equal(calls.length,initial==='team-sessions'?2:1);assert.equal(fresh.WindowTeam.section().includes(c.t('Older records are not read yet')),initial==='team');
    const words=library.words(root);Object.assign(fresh,{t:words.t,I18N:words.I18N,icon:()=>'',tile:n=>String(n)});
    for(const locale of ['en','zh']) {
      words.I18N.set(locale==='zh'?'zh-CN':'en');
      for(const status of ['OBSERVING','DELIVERED']) {
        fresh.WindowActivity.absorbPage({...page('CONTINUED',[]),epoch:'window-epoch',observer:{native_usage:{status,reason:'native_bridge.usage_catching_up',read_through:4,total_bytes:12}}});
        const rendered=fresh.WindowTeam.section();
        assert.equal(rendered.includes('data-usage-notice="CATCHING_UP"'),status==='OBSERVING');
        if(status==='OBSERVING')assert.ok(rendered.includes(words.t('Usage read through {read} of {total} bytes; the next read continues.',{read:4,total:12})));
        assert.ok(!/\{(?:read|total)\}/.test(rendered));
      }
    }
  }
};
if(require.main===module){
const [root,inputPath,team='',actor='']=process.argv.slice(2);
const input=JSON.parse(fs.readFileSync(inputPath,'utf8'));
const pages=Array.isArray(input)?input:input.pages,reads=(input.reads)||{},requested=[],history=(input.history||[]).map(e=>({id:e.entry_id,name:e.kind,task_id:e.task_id,raw:e}));
const hash={value:'#page=tasks'+(team?'&team='+encodeURIComponent(team):'')+(actor?'&actor='+encodeURIComponent(actor):'')};
const c={console,URLSearchParams,Date,Number,Math,Set,Map,Object,Array,String,Promise,JSON,Boolean,Error,PAGES:{},ROUTES:{team:['Research Team','Workroom'],'team-members':['Research Team','Participants'],'team-sessions':['Research Team','Sessions'],'team-evidence':['Research Team','Team evidence']}, // the page words (2026-09-21: one word per page, ROUTES owns it)
  app:{page:'tasks',book:null},document:{hidden:false,addEventListener(){},querySelector:()=>null,getElementById:()=>null},window:{addEventListener(){}},
  setTimeout:()=>0,clearTimeout(){},hashParams:()=>new URLSearchParams(hash.value.slice(1)),
  replaceHash:(u)=>{const q=new URLSearchParams(hash.value.slice(1));for(const [k,v] of Object.entries(u))q.set(k,v);hash.value='#'+q;},
  objectEntry:()=>true,patchMain(){},count:(n)=>String(n),when:(value)=>String(value ?? '—'),tile:(n)=>`{${n}}`,tabStrip:(label,items)=>items.map(x=>(x.on?'*':'')+x.word+' ').join(''),
  navigate(){},render(){},notify(){},openDialog(){},link:library.words(root,(page,extra)=>'#'+new URLSearchParams({page,...extra})).link,
  Data:{read:async(p)=>{requested.push(p);const key=Object.keys(reads).find(k=>p.startsWith(k));if(!key)throw Error('no answer recorded for '+p);const body=reads[key];if(body&&body.__refused)throw Error(body.__refused);return body;},mergeTasks(){},installedResult:async(task)=>{const results=(await c.Data.read('/api/results')).results||[],first=results.find(r=>r.task_id===task);for(const listed of first?[first]:results){const report=await c.Data.read('/api/report?'+new URLSearchParams({result_hash:listed.result_hash}));if(report.result_hash!==listed.result_hash)throw new Error('portfolio_application.result_readback_mismatch');if((report.used_by_task_ids||[]).includes(task))return report;if(first)throw new Error('portfolio_application.result_readback_mismatch');}return null;},history:()=>history,refreshHistory:async()=>{requested.push('/api/research-history');},openEntry(){},tasks:()=>input.tasks||[]},
  LiveTasks:{paintActivity(){},select(){},openResult(){},close(){}},requestAnimationFrame(){},LiveViews:{taskDock(){},savedObjectLink:(l,e)=>l+':'+e},LiveResearch:{dirty:()=>false,inspectShared(){}},
  html:(s,...v)=>s.reduce((a,p,i)=>a+p+(Array.isArray(v[i])?v[i].join(''):v[i]??''),''),t:(s,vars)=>String(s).replace(/\{(\w+)\}/g,(_,k)=>String(vars?.[k]??'')),
  notRead:(title,error,words='',action='')=>`<div class="warning">${title}:${error}${words?' '+words:''}${action}</div>`,noteLine:(title,body='',tone='',action='')=>`<p class="note-line">${title}${body?' · '+body:''}${action||''}</p>`,hint:(term)=>term,factsRef:(title,body)=>`<p>${title}</p>${body}`,codeRef:(title,text)=>'<p>'+title+'</p><template>'+(typeof text==='string' ? text : JSON.stringify(text))+'</template>',refCell:(uri)=>'<span>'+uri+'</span>',hashCell:(h)=>'<span>'+(h||'—')+'</span>',readingPane:(title,kind,body,close)=>'<aside>'+title+body+'</aside>',sourceRows:(rows)=>rows.map((r)=>[r.label,r.value]),badge:(tone,label)=>`[${tone}:${label}]`,stateLine:(x,o={})=>`[${(typeof x==='string'?x:(x?.lifecycle??x?.state??x?.status??''))}:${o.word??''}]`,skeleton:(shape='rows')=>'SKELETON('+shape+')',emptyState:(s,a='')=>'EMPTY('+s+')'+(a||''),railTools:()=>'',groupHead:(label,count)=>'GROUP('+label+':'+count+')',listFoot:()=>'',evidenceRow:(type,x,o={})=>{const r=typeof type==='object'?type:{type,subject:String(x?.entity_id||x?.issue_handle||x?.span_handle||x?.finding_handle||x?.reference?.label||x?.title||''),state:String(x?.state||''),word:'',why:''};return '<div class="list-row evidence-row '+(o.cls||'')+'" data-kind="'+(r.type||'')+'">ROW('+[o.named?r.kind+' · '+r.subject:r.subject,'['+r.state+':'+(r.word||'')+']',o.why??r.why,...((o.brief?[]:(r.props||[])).concat(o.props||[])).filter(Boolean).map(p=>Array.isArray(p)?p[1]:p),o.time,o.actions,o.attrs].filter(Boolean).join('|')+')</div>';},citePill:(h)=>'<span class="es-cite">'+h+'</span>',objectRow:(s={},x)=>'ROW('+[s.name,s.why,...(((x&&x.props)||[]).filter(Boolean).map(p=>Array.isArray(p)?p[1]:p)),x&&x.time,x&&x.actions].filter(Boolean).join('|')+')'+((x&&x.under)||''),statusDot:(s,l)=>'['+s+(l?':'+l:'')+']',btn:(label,action,value)=>`<${action}:${value}>{${label}}`,btnAttrs:(label,action,value,cls,attrs)=>(['team-actor','team-fold'].includes(action)?`<${action}:${value}>{${label}}${attrs||''}`:''),banner:(a,b,tone,action)=>`(${a}|${b}|${action||''})`,icon:(n)=>`{${n}}`,panel:(title,sub,body)=>`<<${title}>>`+(Array.isArray(body)?body.join(''):body),kv:(rows)=>rows.map(r=>r.join(': ')).join(' | '), // the record in place is a grid (T4): its rows as the builder lists them
  subjectChoice:library.stubs.subjectPicker,objectHead:library.stubs.headingSubject,
  routeUrl:(page)=>'#page='+page,NAV_ATTR:'',controlAttrs:()=>'',codeWords:(code)=>String(code ?? '—'),short:(v,n=8)=>String(v||'').slice(0,n),mono:(v,n=12)=>v?'<span class="mono">'+String(v).slice(0,n)+'</span>':'—',coded:(code)=>`<span class="mono">${code ?? '—'}</span>`,actorWords:(caller)=>String(caller ?? '—'),whenText:(value)=>String(value ?? '—'),pluralText:(n,one,many,args={})=>String(Number(n)===1?one:many).replace(/\{(\w+)\}/g,(m,k)=>args&&args[k]!==undefined?args[k]:m),countText:(n,one,many)=>(Number(n)===1 ? one : many).replace('{n}',String(n)),table:(headers,rows)=>'<table>'+(Array.isArray(rows)?rows.join(''):rows)+'</table>',tr:(cells)=>'<tr>'+cells.map(String).join('|')+'</tr>'};
c.Data.workspaceStatus=input.scoped_sessions?'ready':undefined;
c.Data.readShared = (url,...args) => { const id=new URLSearchParams(url.split('?')[1]).get('native_session_id');return input.scoped_sessions?.[id]?Promise.resolve(input.scoped_sessions[id]):c.Data.read(url,...args); };
library.context(c, root);for(const name of ['status.js','live-activity.js','team-members.js','live-team.js'])vm.runInContext(fs.readFileSync(path.join(root,name),'utf8'),c);
vm.runInContext('globalThis.A=LiveActivity;globalThis.TM=LiveTeam;',c);
(async()=>{
  const otherTask='11111111-2222-4333-8444-555555555555';
  if(input.committee){
    const items=pages.flatMap(p=>p.items||[]),row=structuredClone(items.find(r=>r.schema_kind==='ExternalActivityObserved'));
    row.observation_id=otherTask;row.task_id=otherTask;row.ordinal=Math.max(...items.map(r=>r.ordinal))+1;row.source_sequence=row.ordinal;row.payload.producer_sequence=row.ordinal;row.payload.summary='Additional Task';
    Object.assign(row.payload.subject,{input_channel:'ACTOR_DECLARED',message_kind:'plan',message_id:otherTask,native_event_id:otherTask,reference:otherTask});
    pages.push({...pages.at(-1),disposition:'CONTINUED',items:[row],more:false});
  }
  for(const page of pages)c.A.absorbPage(page);
  for(const ref of input.resolve||[])await c.TM.resolve(ref);for(const ref of input.verify||[])await c.TM.verify(ref);
  const scene=input.projection?null:c.TM.scene();
  if(input.projection){
    const words=library.words(root);Object.assign(c,{t:words.t,I18N:words.I18N,icon:()=>'',tile:n=>String(n)});
    const sessions=[],tr=c.tr;let rows=[];c.tr=(cells)=>{rows.push(cells.map(String));return tr(cells);};c.btnAttrs=(label,action,value)=>`<button data-action="${action}" data-value="${value}">${label}</button>`;
    const ids=Object.keys(input.scoped_sessions||{});if(!ids.length)ids.push(...new Set(pages.flatMap(p=>p.items||[]).map(e=>e.payload?.subject?.native_session_id).filter(Boolean)));
    for(const id of ids){
      hash.value='#'+new URLSearchParams({page:'team-participants',team:id});c.app.page='team-participants';c.TM.scene(false,true);await new Promise(setImmediate);
      const s=c.TM.scene(false,true).sessions.find(s=>s.id===id);if(!s)continue;rows=[];
      const participants=c.TM.section(),run=c.TM.runs([s])[0],references=[];
      for(const e of [...s.entries.values()].filter(e=>e.qualified&&!e.replayOf)) {
        if(references.some(r=>r.reference===e.reference))continue;
        if(!c.TM.resolved().has(e.reference))await c.TM.resolve(e.reference);const read=c.TM.resolved().get(e.reference),resolvedKind=read?.target?.kind||null;
        references.push({reference:e.reference,kind:['hash','history'].includes(e.qualified.kind)?resolvedKind:e.qualified.kind,resolved_kind:resolvedKind,resolved:Boolean(read?.target&&!read.failure&&read.level!=='unresolved')});
      }
      sessions.push({id:s.id,title:run.name,question:run.question,goal_scope:s.goalScope,rows,references,
        fact_ids:s.facts.map(f=>f.item.observation_id),attributed_fact_ids:s.facts.filter(f=>f.attributed).map(f=>f.item.observation_id),product_count:c.TM.counts().observations,product_more:Boolean(input.scoped_sessions?.[id]?.more),
        page_text:participants.replace(/<[^>]*>/g,''),usage_notice:/data-usage-notice="CATCHING_UP"/.test(participants)?'CATCHING_UP':null,
        unidentified:c.t('Unidentified'),html:participants,
        role_words:Object.fromEntries([...new Set([...s.participants.values()].flatMap(p=>[...p.roles]))].map(r=>[r,c.TM.roleName(r)])),
        members:[...s.participants.values()].map(p=>({id:p.id,roles:[...p.roles],name:vm.runInContext('TeamMembers',c).unique(s,p.id),usage_visible:rows.some(r=>r[0].includes(`data-value="${p.id}"`)&&r.slice(3,7).every(v=>v!=='')),usage:[...p.usage.values()].map(u=>({model:u.model,responses:u.responses,input_tokens:u.input,output_tokens:u.output,cache_read_tokens:u.cacheRead,cache_write_tokens:u.cacheWrite})),first_at:[...p.usage.values()].map(u=>u.first).filter(Number.isFinite).sort()[0]})).map(p=>({...p,first_at:p.first_at?new Date(p.first_at).toISOString():null})),
        entries:[...s.entries.values()].map(e=>({id:e.id,actor:e.actor,role:e.role,kind:e.kind,message_kind:e.messageKind,recipient:e.recipient,recipient_label:c.TM.recipientName(s,e.recipient,e)}))});
    }
    process.stdout.write(JSON.stringify({sessions}));return;
  }
  {const q=new URLSearchParams(hash.value.slice(1));if(!q.get('team')&&scene.sessions[0]){q.set('team',scene.sessions[0].id);hash.value='#'+q;}}
  if(input.committee){
    const b=reads['/api/committee'],s=scene.sessions.find(s=>s.id===b.session[1]),saved=structuredClone(b),words=library.words(root);
    hash.value='#'+new URLSearchParams({page:'team-committee',team:s.id,committee:b.update_task_id});c.app.page='team-committee';
    Object.assign(c,{I18N:words.I18N,t:words.t,codeWords:words.codeWords,declaredCodeWord:words.declaredCodeWord,stat:words.stat});let head;c.labelWords=l=>c.I18N.zh?l?.title_zh||l?.title:l?.title;c.objectHead=(...args)=>{head=args;return library.stubs.headingSubject(...args)+c.kv(args[5].facts);};
    const row=c.objectRow;c.objectRow=(s,x)=>row(s,x)+(x?.attrs||'');c.btnAttrs=(label,action,value,cls,attrs)=>`<button data-action="${action}" data-value="${value}" ${attrs||''}>${label}</button>`;c.TM.section();const sessionFacts=head[5].facts;for(const key of ['Lead','Span','Goals','Declared references'])assert(sessionFacts.some(r=>r[0]===c.t(key)));await c.TM.readCommittee(s);
    const retainedFacts=sessionFacts.filter(([key])=>key!==c.t('Declared references'));
    const checkGoals=()=>{const facts=head[5].facts.filter(([key])=>key===c.t('Goals'));assert.equal(facts.length,1);const value=String(facts[0][1]);assert(!value.includes(c.t('Goal')));const ways=[...value.matchAll(/href="#([^\"]+)"/g)].map(m=>new URLSearchParams(m[1].replaceAll('&amp;','&')));assert.deepEqual(ways.map(q=>[q.get('page'),q.get('goal')]),[...s.goals].map(id=>['goal',id]));};
    const checkHead=()=>{assert.deepEqual(head[5].facts.filter(([key])=>retainedFacts.some(([k])=>k===key)),retainedFacts);assert.equal(head[5].facts.filter(([key])=>key===c.t('Task')).length,1);const refs=String(head[5].facts.find(([key])=>key===c.t('Declared references'))[1]);assert(refs.includes(c.short(otherTask,c.SHORT.id)));assert(!refs.includes(c.short(b.update_task_id,c.SHORT.id)));checkGoals();};
    for(const [stage,messages,person_items,verdict] of [['STANCES',[],[],null],['STANCES',[saved.messages[0]],[],null],['DEBATE',saved.messages.filter(m=>!['PERSON_ANSWER','VERDICT'].includes(m.kind)),saved.person_items.map(p=>({...p,answer:null})),null],['CLOSED',saved.messages,saved.person_items,saved.verdict]]){
      Object.assign(b,{stage,messages,person_items,verdict});const floor=c.TM.section();assert(floor.includes(`data-committee-stage="${stage}"`));assert(!String(head[3]).includes(stage));assert.equal(head[5].facts[0][0],c.t(verdict?'Verdict':'Committee'));checkHead();
      for(const [role,state] of Object.entries(b.members))assert(floor.includes(`data-member="${role}" data-state="${state}"`));for(const p of b.tension_points)assert(floor.includes(`data-point="${p.alias}" data-state="${p.state}"`));for(const m of messages)assert(floor.includes(m.text));
      if(person_items.length){assert(floor.indexOf('committeeAgenda')<floor.indexOf('data-person-item=')&&floor.indexOf('data-person-item=')<floor.indexOf('committeeFloor'));for(const p of person_items){assert(floor.includes(p.question));assert.equal(floor.includes(saved.person_items.find(v=>v.id===p.id).answer),p.answer!==null);}}
      if(stage==='STANCES')for(const id of ['committeeProgress','committeeMembers'])assert(floor.includes(`id="${id}" data-stack-box="section"`));assert(!floor.includes('lobby-tools'));assert(!floor.includes('data-column="state"'));
    }
    Object.assign(b,saved);const floor=c.TM.section(),taskFact=String(head[5].facts.find(r=>r[0]===c.t('Task'))[1]);assert(taskFact.includes('data-action="task"')&&taskFact.includes(`data-value="${b.update_task_id}"`)&&!taskFact.includes('run-ref-kind'));for(const id of b.standing_dissents){const m=b.messages.find(m=>m.id===id);assert(floor.includes('data-dissent="stands"'));assert(floor.includes(`data-clamp="open">${m.text}`));}
    const report=[...floor.matchAll(/href="#([^"]+)"/g)].map(m=>new URLSearchParams(m[1].replaceAll('&amp;','&'))).find(q=>q.get('page')==='report'),answer=b.messages.find(m=>m.kind==='PERSON_ANSWER'),ruling=b.messages.find(m=>m.outcome==='REJECT');assert(answer&&ruling);
    assert(floor.includes(`data-kind="person_answer"`));assert(floor.includes(`<strong>${c.t('Person')}</strong><span class="team-kind">· ${c.t(answer.row.subject.relay_word)}</span>`));assert(floor.includes(`is-pm-response`)&&floor.includes(`id="team-event-${ruling.row.producer_session}:${ruling.id}"`));assert(floor.includes(`data-outcome="REJECT">${c.t('Rejected')}`));assert(report);assert.deepEqual(JSON.parse(report.get('review_selector')),b.review_selector);
    c.TM.chooseCommitteePoint('T2');const filtered=c.TM.section();for(const id of b.standing_dissents)assert(!filtered.includes(b.messages.find(m=>m.id===id).text));c.TM.chooseCommitteePoint('T2');
    b.tension_points=Array.from({length:c.LOBBY.shown+1},(_,i)=>({...saved.tension_points[0],alias:'T'+(i+1)}));assert(!c.TM.section().includes(`data-point="T${b.tension_points.length}"`));c.Lobby.more('committee-agenda:'+b.tension_points[0].state);assert(c.TM.section().includes(`data-point="T${b.tension_points.length}"`));b.tension_points=saved.tension_points;
    for(const page of ['team','team-participants','team-outputs','team-evidence']){c.app.page=page;const rendered=c.TM.section();assert(rendered.includes(`data-verdict="${b.verdict.outcome}"`));assert(!String(head[3]).includes(b.verdict.outcome));checkHead();}
    c.app.page='team';const folded=c.TM.section();assert(!folded.includes('data-kind="stance"'));assert(folded.includes('team-committee'));assert.equal(c.TM.counts().exchanges,1);assert(folded.includes(otherTask));
    assert.equal(requested.filter(p=>p.startsWith('/api/committee')).length,1);const retained=c.A.retained;c.A.retained=()=>[];c.app.page='team-committee';for(const m of b.messages)assert(c.TM.section().includes(m.text));vm.runInContext(fs.readFileSync(path.join(root,'live-views.js'),'utf8'),c);vm.runInContext('globalThis.V=LiveViews;',c);const facts=c.V.collaborationRows(b.update_task_id,b.review_selector.update_publication_hash,b);
    assert.equal(facts.length,1);assert.deepEqual(JSON.parse(JSON.stringify(facts[0].to)),{page:'team-committee',extra:{team:b.native_session_id,committee:b.update_task_id}});assert(String(facts[0].value).includes(c.codeWords(b.verdict.outcome)));assert.equal(c.TM.sessionsNaming(b.update_task_id,b)[0].entries,undefined);c.A.retained=retained;
    c.I18N.set('zh');const zh=c.TM.section();checkGoals();assert.equal(head[5].facts.filter(([key])=>key===c.t('Task')).length,1);for(const key of ['Person',answer.row.subject.relay_word,'Rejected','committee|Floor','committee|Received']){assert.notEqual(c.t(key),key);assert(zh.includes(c.t(key)));}assert(b.labels.title_zh);assert(zh.includes(b.labels.title_zh));c.I18N.set('en');
  }
  // Read Conversation and Evidence, then each retained exchange without its replies.
  c.app.page='team';const workroom=c.TM.section();c.app.page='team-evidence';const evidence=c.TM.section();c.app.page='team';
  const html=workroom+evidence,readers={};
  const readIn=(markup,id)=>{const at=markup.indexOf(`id="team-event-${id}"`);if(at<0)return '';const start=markup.lastIndexOf('<li',at),tag=/<(\/?)li(?=[\s>])/g;let depth=0,end=markup.length;tag.lastIndex=start;for(let m;(m=tag.exec(markup));){depth+=m[1]?-1:1;if(!depth){end=m.index+5;break;}}const li=markup.slice(start,end),cut=li.indexOf('<ol class="team-replies">');return cut<0?li:li.slice(0,cut);};
  {
    const saved=hash.value,q=new URLSearchParams(saved.slice(1)),chosenActor=q.get('actor')||'',s=q.get('team')?scene.sessions.find(v=>v.id===q.get('team')):scene.sessions[0];
    for(const e of s?[...s.entries.values()].sort((a,b)=>a.ordinal-b.ordinal):[]){
      if(e.replayOf||(chosenActor&&e.actor!==chosenActor&&e.recipient!==chosenActor))continue;
      c.TM.showEvent(e.id);
      readers[e.id]=readIn(c.TM.section(),e.id);
    }
    hash.value=saved;
  }
  const out={feed:c.A.state(),html,readers,requested,unknown:scene.unknown.length,
    resolved:Object.fromEntries([...c.TM.resolved()].map(([k,v])=>[k,{level:v.level,owner:v.owner||null,note:v.note||null,open:v.open||null,link:v.link||null,route:v.route||null,target:v.target||null,failure:v.failure||null,limit:v.limit||null}])),
    sessions:scene.sessions.map(s=>({id:s.id,producers:[...s.producers],references:[...s.references],derived:[...(s.derived||new Map())].map(([k,v])=>({ref:k,...v})),unknownKinds:s.unknownKinds.map(u=>u.item.payload.event_kind),
      participants:[...s.participants.values()].map(p=>({id:p.id,roles:[...p.roles],messages:p.messages.length,hooks:p.hooks.length})),
      entries:[...s.entries.values()].sort((a,b)=>a.ordinal-b.ordinal).map(e=>({observation:e.id,kind:e.kind,messageKind:e.messageKind||null,declaredKind:e.declaredKind||null,actor:e.actor,role:e.role,recipient:e.recipient||null,reference:e.reference||null,qualified:e.qualified?e.qualified.kind:null,truncated:e.truncated||false,bytes:e.bytes||null,textLength:(e.text||'').length,replayOf:e.replayOf,replays:e.replays||[],conflictsWith:e.conflictsWith,hookEvent:e.hookEvent||null,terminal:e.terminal||null,stopActive:e.stopActive||null,channel:e.channel,timeKind:e.timeKind,taskId:e.taskId,authority:e.authority})),
      facts:s.facts.map(f=>({observation:f.item.observation_id,schema:f.item.schema_kind,operation:f.item.payload.operation||null,phase:f.item.payload.phase||null,status:f.item.payload.status||null,failure_code:f.item.payload.failure_code||null,lifecycle:f.item.payload.task_lifecycle||null,artifact:f.item.payload.artifact_kind||null,task:f.group.task_id,named:f.named,followed:f.followed||[]}))}))};
  process.stdout.write(JSON.stringify(out));
})().catch(e=>{console.error(e);process.exitCode=1;});

}
