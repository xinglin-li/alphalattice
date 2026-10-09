// Activity feed consumer contract: cursor continuity, duplicate delivery, transport failure,
// reset, bounded catch-up and retention, hidden-document cadence, historical-versus-current
// Task state, opening by the clicked object's own kind, store change after an outage, the
// page lifecycle (pagehide / back-forward-cache pageshow) and following versus pinning. No HTTP,
// filesystem, model or data work: `Data.read` is scripted with owner-shaped pages and every
// announcement, hash change and shared-PLAN inspection is recorded.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const library=require('./workbench_library.cjs'); // the library's constants and the scripts' parameters, from the source (Q2)
const finish=library.guard('workbench_activity');
const root=process.argv[2],reads=[],toasts=[],said=[],merged=[],timers=[],opened=[],selected=[],dialogs=[],hashes=[],inspected=[],listeners={};
let pages=[],history=[],gate=null,dirty=false,dialogOpen=false,historyGate=null,openGate=null,extraHistory=[],knownTasks=[],goalObservations=0;
const settled=()=>new Promise(r=>setImmediate(r));
const on=(name,fn)=>{(listeners[name]??=[]).push(fn);},fire=(name,event={})=>{for(const fn of listeners[name]||[])fn(event);};
const item=(ordinal,schema,payload,extra={})=>({ordinal,observation_id:'obs-'+ordinal,occurred_at:'2026-09-14T12:00:'+String(ordinal%60).padStart(2,'0')+'Z',observed_at:'2026-09-14T12:00:00Z',schema_kind:schema,source_kind:'PRODUCT_OPERATION',source_id:'local-web:i1',source_sequence:ordinal,task_id:null,run_id:null,stage_id:null,correlation_ids:[],authority:'OPERATIONAL_ASSERTION',retention_class:'TRANSIENT_OPERATIONAL',availability:'AVAILABLE',payload,...extra});
const page=(disposition,epoch,items,tasks={},more=false,observer={status:'OK'})=>({workspace_id:'qa',disposition,epoch,cursor:epoch+':'+(items.at(-1)?.ordinal||0),head:items.at(-1)?.ordinal||0,more,items,unavailable:0,tasks,observer,read_cost:{observations:items.length,elapsed_ms:1}});
const projection=(task_id,lifecycle,extra={})=>({task_id,task_kind:'portfolio_public_development_replay',lifecycle,verified_stage_count:lifecycle==='SUCCEEDED'?4:1,total_stage_count:4,current_stage:'plan',worker_failure:null,worker_failure_type:null,...extra});
const percentRule=(root)=>library.readingSource(root);
const c={Window:{renderTop(){}},console,URLSearchParams,Date,Number,Math,Set,Map,Object,Array,String,Promise,JSON,Boolean,Error,
  app:{page:'portfolio',book:'book-a',session:'2024-08-12'},
  document:{hidden:false,addEventListener:on,querySelector:()=>dialogOpen?{open:true}:null},window:{addEventListener:on},
  replaceHash:(v)=>hashes.push(v),LiveResearch:{dirty:()=>dirty,inspectShared:(hash)=>inspected.push(hash)},
  setTimeout:(fn,ms)=>{timers.push(ms);return timers.length;},clearTimeout(){},
  Data:{read:async(p)=>{reads.push(p);if(gate)await gate;const next=pages.shift();if(next instanceof Error)throw next;if(!next)throw Error('probe: no page scripted for '+p);return next;},mergeTasks:(v)=>{merged.push(...v);for(const x of v){const i=knownTasks.findIndex(k=>k.task_id===x.task_id);if(i>=0)knownTasks[i]=x;else knownTasks.push(x);}},
    history:()=>history,refreshHistory:async()=>{if(historyGate)await historyGate;history=[{id:'result:9f9f9f9f00000000'},...extraHistory];},openEntry:(id)=>opened.push(id),tasks:()=>knownTasks},
  LiveTasks:{paintActivity(){},select:(id)=>selected.push('select:'+id),openResult:async(id,wanted=()=>true)=>{if(openGate)await openGate;if(!wanted())return false;opened.push('experiment:'+id);}},
  LiveViews:{taskDock(){},savedObjectLink:(label,entry)=>label+':'+entry,nameOf:(v)=>({name:'Study '+v.task_id})},
  navigate:()=>{throw Error('activity must never navigate');},link:(label,page)=>`<a href="#page=${page}">${label}</a>`,joinMarkup:(items,separator=' · ')=>items.join(separator),openDialog:(a,b)=>dialogs.push(b),
  notify:(m,vars,act=null)=>{said.push({vars,act});toasts.push(m.replace(/\{(\w+)\}/g,(_,k)=>String(vars?.[k]??'')));},
  html:(s,...v)=>s.reduce((a,p,i)=>a+p+(Array.isArray(v[i])?v[i].join(''):v[i]??''),''),t:(s,vars)=>s.replace(/\{(\w+)\}/g,(_,k)=>String(vars?.[k]??'')),
  notRead:(title,error,words='',action='')=>`<div class="warning">${title}:${error}${words?' '+words:''}${action}</div>`,noteLine:(title,body='',tone='',action='')=>`<p class="note-line">${title}${body?' · '+body:''}${action||''}</p>`,hint:(term)=>term,factsRef:(title,body)=>`<p>${title}</p>${body}`,codeRef:(title,text)=>'<p>'+title+'</p><template>'+(typeof text==='string' ? text : JSON.stringify(text))+'</template>',refCell:(uri)=>'<span>'+uri+'</span>',hashCell:(h)=>'<span>'+(h||'—')+'</span>',readingPane:(title,kind,body,close)=>'<aside>'+title+body+'</aside>',sourceRows:(rows)=>rows.map((r)=>[r.label,r.value]),badge:(tone,label)=>`[${tone}:${label}]`,stateLine:(x,o={})=>`[${(typeof x==='string'?x:(x?.lifecycle??x?.state??x?.status??''))}:${o.word??''}]`,skeleton:(shape='rows')=>'SKELETON('+shape+')',statusDot:(s,l)=>'['+s+(l?':'+l:'')+']',emptyState:(s,a='')=>'EMPTY('+s+')'+(a||''),btn:(label,action,value)=>`<${action}:${value}>`,btnAttrs:()=>'',banner:(a,b)=>`(${a}|${b})`,icon:()=>'',tile:()=>'',codeWords:(code)=>String(code ?? '—'),actorWords:(caller)=>({HUMAN:'You',EXTERNAL_AUTOMATION:'Agent · API'})[caller]||String(caller ?? '—'),when:(value)=>String(value ?? '—'),
  groupHead:(label,count)=>'GROUP('+label+':'+count+')',listFoot:()=>'',objectRow:(s={},x)=>'ROW('+[s.lead,s.name,s.why,...(((x&&x.props)||[]).filter(Boolean).map(p=>Array.isArray(p)?p[1]:p)),x&&x.time,s.to?`<${s.to.action}:${s.to.value}>`:'',x&&x.actions].filter(Boolean).join('|')+')'+((x&&x.under)||'')};
const runShapes=(root)=>{const src=require('node:fs').readFileSync(require('node:path').join(root,'components.js'),'utf8');const a=src.indexOf('/* ---- run shapes (round 72)');const b=src.indexOf('/* ---- end of run shapes ---- */');return src.slice(a,b)+';globalThis.stepList=stepList;globalThis.runLog=runLog;globalThis.logLine=logLine;globalThis.observationLine=observationLine;globalThis.logMove=logMove;globalThis.logHeld=logHeld;globalThis.logRetain=logRetain;';};
c.Data.readShared = (...args) => c.Data.read(...args);
c.LiveGoals={observe:()=>{goalObservations++;}};
library.context(c, root);vm.runInContext(fs.readFileSync(path.join(root,'status.js'),'utf8'),c);vm.runInContext(percentRule(root),c);vm.runInContext(runShapes(root),c);vm.runInContext(fs.readFileSync(path.join(root,'live-activity.js'),'utf8')+';globalThis.A=LiveActivity;',c);
(async()=>{
  const A=c.A;
  // 1. First read is the tail: nothing is announced for history the reader just joined.
  pages=[page('TAIL','e1',[item(1,'ProductOperationObserved',{operation:'PLAN',caller:'EXTERNAL_AUTOMATION',phase:'REQUESTED',operation_ref:'op1',subject:{}}),item(2,'ProductOperationObserved',{operation:'PLAN',caller:'EXTERNAL_AUTOMATION',phase:'RETURNED',operation_ref:'op1',status:'PLANNED',subject:{spec_hash:'abcdef0123456789'}})])];
  await A.refresh();
  assert.equal(goalObservations,1,'Goals observes the existing shared activity refresh');
  assert.deepEqual(toasts,[]);assert.equal(A.state().groups,1);assert.equal(A.state().cursor,'e1:2');
  assert.ok(!reads[0].includes('after='),'first read has no cursor');
  assert.ok(A.section().includes('Agent · API'),'the entry class, not a host identity (round 63: the class in words)');
  assert.ok(!A.section().includes('Codex'));
  const planned=A.groups()[0];assert.equal(A.facts(planned).state,'PLANNED');assert.ok(A.section().includes('spec abcdef01'));
  // 2. An admitted RUN is announced once, its Task joins the list, and the next read watches it.
  const asked=item(3,'ProductOperationObserved',{operation:'RUN',caller:'EXTERNAL_AUTOMATION',phase:'REQUESTED',operation_ref:'op2',subject:{}});
  const run=item(4,'ProductOperationObserved',{operation:'RUN',caller:'EXTERNAL_AUTOMATION',phase:'RETURNED',operation_ref:'op2',status:'ADMITTED',task_id:'task-1',task_lifecycle:'QUEUED',next_read:{operation:'STATUS',task_id:'task-1'},subject:{}},{task_id:'task-1',run_id:'local-web:task-1'});
  pages=[page('CONTINUED','e1',[asked,run],{'task-1':projection('task-1','QUEUED')})];
  await A.refresh();
  assert.equal(reads[1],'/api/activity?limit=200&after=e1%3A2','continues from the last cursor');
  assert.equal(A.groups()[0].items.length,2,'the REQUESTED half joins its Task once the RETURNED half names it');
  assert.deepEqual(toasts,[],'an admitted Task counts, it does not toast (round 75)');assert.equal(A.unseen(),1);
  assert.deepEqual(merged.map(v=>v.task_id),['task-1']);
  assert.equal(A.facts(A.groups()[0]).state,'QUEUED'); // the dock line retired with the dock (round 62): the unseen count is the side's badge
  // 3. A re-delivered row is not a second event and not a second toast (ordinal watermark).
  pages=[page('CONTINUED','e1',[run],{'task-1':projection('task-1','QUEUED')})];
  await A.refresh();
  assert.equal(reads[2],'/api/activity?limit=200&after=e1%3A4&watch=task-1');
  assert.equal(A.unseen(),1);assert.equal(A.state().groups,2);
  // 4. Transport failure is reported and backed off; nothing is retried, selected or navigated.
  pages=[Error('fetch failed')];
  await A.refresh();
  assert.equal(A.state().error,'fetch failed');assert.equal(timers.at(-1),10000);
  assert.ok(A.section().includes('Activity feed unreachable'));
  assert.equal(c.app.page,'portfolio');assert.equal(c.app.book,'book-a');
  // 5. Recovery continues from the same cursor; the verified result is announced, never opened.
  const done=item(5,'TaskControlTransition',{task_lifecycle:'SUCCEEDED',task_class:'portfolio_public_development_replay',disposition:'COMMAND_RETURNED',verified_prefix_count:4,total_units:4},{task_id:'task-1',run_id:'local-web:task-1',authority:'TASK_CONTROL_ASSERTION',source_kind:'TASK_CONTROL'});
  const artifact=item(6,'ArtifactVerificationObserved',{artifact_kind:'PortfolioResearchResult',artifact_hash:'9f9f9f9f00000000',availability:'AVAILABLE'},{task_id:'task-1',run_id:'local-web:task-1',authority:'ARTIFACT_ASSERTION',source_kind:'PRODUCT_ARTIFACT'});
  pages=[page('CONTINUED','e1',[done,artifact],{'task-1':projection('task-1','SUCCEEDED')})];
  await A.refresh();
  assert.equal(reads[4],'/api/activity?limit=200&after=e1%3A4&watch=task-1');
  assert.equal(A.state().error,'');assert.equal(timers.at(-1),3000);
  assert.equal(A.unseen(),2,'a verified result counts as unseen; the state line says it, not a toast');
  const group=A.groups()[0];assert.equal(group.task_id,'task-1');assert.equal(group.items.length,4);
  const verified=A.facts(group);assert.equal(verified.state,'SUCCEEDED');assert.equal(verified.basis,'current');assert.equal(verified.verified,true);
  assert.ok(A.section().includes('<activity-open:task-1>'));assert.ok(A.section().includes('Result verified by its owner'));
  assert.equal(c.app.page,'portfolio');assert.equal(c.app.book,'book-a');
  // A refused Task projection does not replace the separate verified artifact observation or
  // manufacture state/progress; its owner's Workspace and Storage recovery routes stay usable.
  const unreadableTask={task_id:'task-1',status:'REFUSED',failure_code:'task_control.projection_unavailable',detail:'The Task projection could not be read; its current state remains unknown.',next_requests:{workspace:{operation:'WORKSPACE_SHOW'},storage:{operation:'WORKSPACE_BACKUPS'}}};
  pages=[page('CONTINUED','e1',[],{'task-1':unreadableTask})];await A.refresh();
  const refusedFacts=A.facts(A.groups().find(g=>g.task_id==='task-1'));
  assert.equal(refusedFacts.state,'REFUSED');assert.equal(refusedFacts.basis,'unavailable');
  assert.equal(refusedFacts.verified,true);assert.equal(refusedFacts.artifactKind,'PortfolioResearchResult');
  assert.equal(refusedFacts.live.lifecycle,undefined);assert.equal(refusedFacts.live.verified_stage_count,undefined);
  const refusedActivity=A.section();
  assert.ok(refusedActivity.includes('Result verified by its owner')&&refusedActivity.includes(c.codeWords('PortfolioResearchResult')),'the artifact observation remains its separate verified fact');
  assert.ok(refusedActivity.includes('href="#page=overview"')&&refusedActivity.includes('href="#page=storage"'),'the refused Task offers Workspace and Storage recovery');
  assert.ok(!/undefined\s*\/\s*undefined/.test(refusedActivity),'an unreadable projection has no fabricated stage count');
  // 6. Open result goes by the clicked object's own kind and reference, never the selected Task:
  //    a Portfolio result is the saved `result:` object; an experiment goes through its readback.
  await A.open('task-1');
  assert.deepEqual(opened,['result:9f9f9f9f00000000'],'opened as a saved result after history was refreshed');
  assert.deepEqual(selected,[]);assert.equal(c.app.page,'portfolio');assert.equal(c.app.book,'book-a');
  const study=item(7,'ArtifactVerificationObserved',{artifact_kind:'ResearchExecutionEvidence',artifact_hash:'e1e1e1e1e1e1e1e1',availability:'AVAILABLE'},{task_id:'task-2',run_id:'local-web:task-2',authority:'ARTIFACT_ASSERTION',source_kind:'PRODUCT_ARTIFACT'});
  pages=[page('CONTINUED','e1',[study],{'task-2':projection('task-2','SUCCEEDED',{task_kind:'research_experiment'})})];
  await A.refresh();await A.open('task-2');
  assert.deepEqual(opened.at(-1),'experiment:task-2');
  // V659: both published artifact kinds read through the shared words in Activity and Record.
  const captionWords=library.words(root),savedWords={t:c.t,codeWords:c.codeWords};
  Object.assign(c,{t:captionWords.t,codeWords:captionWords.codeWords});
  for(const lang of ['en','zh']) {
    captionWords.I18N.set(lang);
    for(const [task,observation] of [['task-1',artifact],['task-2',study]]) {
      const {artifact_kind:kind,artifact_hash:hash}=observation.payload;
      const group=A.retained().find(g=>g.task_id===task);
      assert.equal(A.facts(group).artifactKind,kind);
      assert.equal(A.facts(group).artifactHash,hash);
      for(const markup of [A.recordOf(task),A.section()]) {
        const renderedText=markup.replace(/<[^>]*>/g,' ').replace(/\s+/g,' ').trim();
        assert.ok(renderedText.includes(captionWords.codeWords(kind)+' '+hash.slice(0,8)),lang+' artifact caption');
        assert.ok(!renderedText.includes(kind+' '+hash.slice(0,8)),lang+' no raw artifact caption');
        assert.ok(markup.includes(hash),'exact observation keeps the complete artifact ID');
      }
    }
  }
  Object.assign(c,savedWords);

  // 7. Historical facts stay distinct from current availability: a recorded RECOVERY_REQUIRED
  //    return never outranks a fresh RUNNING projection, and without a projection it is labelled
  //    as recorded, not current.
  const halted=item(8,'TaskControlTransition',{task_lifecycle:'RECOVERY_REQUIRED',task_class:'portfolio_public_development_replay',disposition:'COMMAND_RETURNED',verified_prefix_count:1,total_units:4},{task_id:'task-3',run_id:'local-web:task-3',authority:'TASK_CONTROL_ASSERTION',source_kind:'TASK_CONTROL'});
  pages=[page('CONTINUED','e1',[halted],{})];
  await A.refresh();
  let f=A.facts(A.groups()[0]);assert.equal(f.state,'RECOVERY_REQUIRED');assert.equal(f.basis,'recorded');
  assert.ok(A.section().includes('recorded at command return · not re-read'));
  pages=[page('CONTINUED','e1',[],{'task-3':projection('task-3','RUNNING')})];
  await A.refresh();
  assert.ok(reads.at(-1).includes('watch=task-3'),'a recovery-required Task is watched for change');
  f=A.facts(A.groups()[0]);assert.equal(f.state,'RUNNING');assert.equal(f.basis,'current');assert.equal(f.verified,false);
  assert.ok(!A.section().includes('<activity-open:task-3>'),'no verified result, no Open result');
  // 8. Paging is bounded per refresh: four pages, then the next tick continues sooner.
  const flood=(from,n)=>Array.from({length:n},(_,i)=>item(from+i,'ProductOperationObserved',{operation:'STORAGE_PIN',caller:'HUMAN',phase:'RETURNED',operation_ref:'op'+(from+i),status:'PINNED',subject:{}}));
  pages=[page('CONTINUED','e1',flood(9,100),{},true),page('CONTINUED','e1',flood(109,100),{},true),page('CONTINUED','e1',flood(209,100),{},true),page('CONTINUED','e1',flood(309,100),{},true),page('CONTINUED','e1',flood(409,100),{},false)];
  const before=reads.length;await A.refresh();
  assert.equal(reads.length-before,4,'at most four pages per refresh');assert.equal(pages.length,1,'the fifth page waits for the next tick');
  assert.equal(timers.at(-1),250,'still behind: continue quickly');assert.equal(A.state().cursor,'e1:408');
  // 9. Retention is bounded: at most 200 groups, Task state only for retained groups, ordinals ahead.
  assert.equal(A.state().groups,200);assert.equal(A.state().tasks,0,'settled Tasks of evicted groups are dropped');
  await A.refresh();assert.equal(A.state().cursor,'e1:508');assert.equal(timers.at(-1),3000);
  // 10. A hidden document slows polling down; it never stops it. pagehide stops it.
  c.document.hidden=true;pages=[page('CONTINUED','e1',[])];await A.refresh();assert.equal(timers.at(-1),15000);
  c.document.hidden=false;pages=[page('CONTINUED','e1',[])];await A.refresh();assert.equal(timers.at(-1),3000);
  // 11. RESET is honoured even with an unchanged epoch: retained state is dropped, the cursor
  //     continues from the fresh tail and the reader says so. The rebuilt tail is a baseline:
  //     its rows are retained facts, shown but neither announced nor counted as unseen; the
  //     next genuinely new row on an ordinary CONTINUED page announces as before.
  const toastsBeforeReset=toasts.length, unseenBeforeReset=A.unseen();
  pages=[page('RESET','e1',[item(1,'ExternalActivityObserved',{event_kind:'NATIVE_TURN_STARTED',summary:'PM asked the Analyst',producer_id:'codex'},{source_kind:'EXTERNAL_CLIENT',authority:'AGENT_PROPOSAL'})])];
  await A.refresh();
  assert.equal(A.state().epoch,'e1');assert.equal(A.state().groups,1);assert.equal(A.state().tasks,0);assert.equal(A.state().watermark,1);
  assert.ok(A.state().notice.includes('rebuilt'));assert.equal(A.state().cursor,'e1:1');
  assert.equal(toasts.length,toastsBeforeReset,'a rebuilt tail announces nothing');assert.equal(A.unseen(),unseenBeforeReset,'nor counts as unseen');
  assert.ok(A.section().includes('External client · not verified'),'the retained row is shown');
  pages=[page('CONTINUED','e1',[item(2,'ExternalActivityObserved',{event_kind:'NATIVE_TURN_ENDED',summary:'Analyst answered',producer_id:'codex'},{source_kind:'EXTERNAL_CLIENT',authority:'AGENT_PROPOSAL'})])];
  await A.refresh();
  assert.equal(A.unseen(),unseenBeforeReset+1,'what arrives after the baseline counts as unseen');
  // 12. Observer degradation and unavailability are shown as such, never as execution state.
  pages=[page('CONTINUED','e1',[],{},false,{status:'DEGRADED',recording:'OK',missing_observations:3,first_failure_at:'2026-09-14T12:00:00Z',last_failure_code:'observation.storage_busy'})];
  await A.refresh();assert.ok(A.section().includes('Activity recording degraded'));assert.ok(A.section().includes('gap remains a gap'));
  pages=[{workspace_id:'qa',disposition:'UNAVAILABLE',epoch:null,cursor:null,head:0,more:false,items:[],unavailable:0,tasks:{},observer:{status:'UNAVAILABLE',store_failure:'observation.storage_open_failed'},read_cost:{observations:0,elapsed_ms:0}}];
  await A.refresh();assert.ok(A.section().includes('Activity recording unavailable'));assert.equal(timers.at(-1),3000,'an unavailable store is still polled for recovery');
  A.markSeen();assert.equal(A.unseen(),0);assert.equal(A.state().notice,'');
  assert.equal(c.app.page,'portfolio');assert.equal(c.app.book,'book-a');
  // 13. Store change after an outage. TAIL(A,50) -> UNAVAILABLE -> TAIL(B,1): the old store's
  //     groups, watermark and cursor must not survive into store B whatever the disposition says.
  const unavailable=()=>({workspace_id:'qa',disposition:'UNAVAILABLE',epoch:null,cursor:null,head:0,more:false,items:[],unavailable:0,tasks:{},observer:{status:'UNAVAILABLE',store_failure:'observation.storage_open_failed'},read_cost:{observations:0,elapsed_ms:0}});
  pages=[page('CONTINUED','e1',[item(50,'ProductOperationObserved',{operation:'PLAN',caller:'HUMAN',phase:'RETURNED',operation_ref:'op50',status:'PLANNED',subject:{}})])];
  await A.refresh();assert.equal(A.state().watermark,50);assert.equal(A.state().cursor,'e1:50');const retained=A.state().groups;
  pages=[unavailable()];await A.refresh();
  assert.equal(A.state().disposition,'UNAVAILABLE');assert.equal(A.state().groups,retained,'retained entries stay visible');
  assert.equal(A.state().cursor,'e1:50','the cursor survives the outage for same-store continuity');assert.equal(A.state().stale,true);
  assert.ok(A.section().includes('retained from before the store became unavailable'));
  pages=[page('TAIL','e2',[item(1,'ProductOperationObserved',{operation:'RUN',caller:'HUMAN',phase:'RETURNED',operation_ref:'opB',status:'ADMITTED',task_id:'task-b',task_lifecycle:'QUEUED',subject:{}},{task_id:'task-b'})],{'task-b':projection('task-b','QUEUED')})];
  const toastsBefore=toasts.length;await A.refresh();
  assert.equal(A.state().epoch,'e2');assert.equal(A.state().watermark,1,'the new store starts its own watermark');
  assert.equal(A.state().groups,1,'store A groups were dropped');assert.equal(A.state().cursor,'e2:1');assert.equal(A.state().stale,false);
  assert.ok(A.state().notice.includes('store changed'));assert.equal(toasts.length,toastsBefore,"the new store's tail is a baseline: retained and shown, not announced");
  assert.ok(A.section().includes('task-b'),'the retained row of the new store is shown');
  assert.equal(A.facts(A.groups()[0]).state,'QUEUED');
  assert.equal(c.app.page,'portfolio');assert.equal(c.app.book,'book-a');assert.deepEqual(selected,[]);
  // Same-store continuity: an outage followed by the same store continues from the retained cursor.
  A.markSeen();pages=[unavailable()];await A.refresh();
  pages=[page('CONTINUED','e2',[item(2,'ProductOperationObserved',{operation:'CANCEL',caller:'HUMAN',phase:'RETURNED',operation_ref:'opC',status:'REFUSED',failure_code:'task.not_cancellable',subject:{}})])];
  await A.refresh();assert.ok(reads.at(-1).includes('after=e2%3A1'),'the retained cursor is presented after the outage');
  assert.equal(A.state().groups,2);assert.equal(A.state().notice,'');assert.equal(A.state().watermark,2);
  // Same store, but the feed can only offer its tail: retained entries stay, the gap is named.
  A.markSeen();pages=[unavailable()];await A.refresh();
  pages=[page('TAIL','e2',[item(200,'ProductOperationObserved',{operation:'PLAN',caller:'HUMAN',phase:'RETURNED',operation_ref:'op200',status:'PLANNED',subject:{}})])];
  await A.refresh();assert.equal(A.state().groups,3);assert.equal(A.state().watermark,200);assert.ok(A.state().notice.includes('resumed from the tail'));
  // 14. Page lifecycle: pagehide ends the loop, a back-forward-cache pageshow resumes it once,
  //     an in-flight request is joined rather than duplicated, and a real teardown stays stopped.
  A.bind();A.markSeen();
  fire('pagehide',{persisted:true});assert.equal(A.state().stopped,true);assert.equal(A.state().timer,false);
  let readsBefore=reads.length;fire('visibilitychange');fire('pageshow',{persisted:false});await Promise.resolve();
  assert.equal(reads.length,readsBefore,'a stopped page does not poll on visibility or a fresh load');
  pages=[page('CONTINUED','e2',[])];fire('pageshow',{persisted:true});
  assert.equal(A.state().stopped,false);await new Promise(r=>setImmediate(r));
  assert.equal(reads.length,readsBefore+1,'a restored page resumes with one read');assert.equal(A.state().timer,true,'and one scheduled poll');
  // In flight across hide/show: the pending request is joined, not duplicated.
  let release;gate=new Promise(r=>{release=r;});pages=[page('CONTINUED','e2',[])];
  readsBefore=reads.length;const timersBefore=timers.length;const inflight=A.refresh();
  assert.equal(A.state().fetching,true);fire('pagehide',{persisted:true});fire('pageshow',{persisted:true});
  const joined=A.refresh();await Promise.resolve();assert.equal(reads.length,readsBefore+1,'resume joins the in-flight request instead of starting another');
  release();gate=null;await inflight;await joined;
  assert.equal(reads.length,readsBefore+1,'exactly one read for the whole cycle');
  assert.equal(timers.length,timersBefore+1,'exactly one poll scheduled after it');assert.equal(A.state().stopped,false);
  // Real teardown: no restore, no polling.
  fire('pagehide',{});assert.equal(A.state().stopped,true);readsBefore=reads.length;fire('visibilitychange');await Promise.resolve();
  assert.equal(reads.length,readsBefore);assert.equal(A.state().timer,false);
  assert.equal(c.app.page,'portfolio');assert.equal(c.app.book,'book-a');assert.deepEqual(selected,[]);
  // 15. Following versus pinning. Following a Task is the viewer's explicit choice, recorded in
  //     the hash; its verified result opens only when the reader is free to be moved, announced
  //     once meanwhile. Pinning (the default) keeps the current view whatever arrives. Following
  //     a shared PLAN inspects that exact hash beside the draft and never RUNs anything.
  const admitted=(ordinal,task,ref)=>item(ordinal,'ProductOperationObserved',{operation:'EXPERIMENT_RUN',caller:'EXTERNAL_AUTOMATION',phase:'RETURNED',operation_ref:ref,status:'ADMITTED',task_id:task,task_lifecycle:'QUEUED',subject:{}},{task_id:task,run_id:'local-web:'+task});
  const evidence=(ordinal,task,kind='ResearchExecutionEvidence')=>item(ordinal,'ArtifactVerificationObserved',{artifact_kind:kind,artifact_hash:'ab'+ordinal,availability:'AVAILABLE'},{task_id:task,run_id:'local-web:'+task,authority:'ARTIFACT_ASSERTION',source_kind:'PRODUCT_ARTIFACT'});
  const experiment=(task,lifecycle)=>({[task]:projection(task,lifecycle,{task_kind:'research_experiment'})});
  A.absorbPage(page('CONTINUED','e2',[admitted(300,'task-9','op300')],experiment('task-9','QUEUED')));
  assert.ok(A.section().includes('<activity-follow:task-9>'),'an in-flight Task offers Follow');
  assert.throws(()=>A.follow('op999'),/no longer retained/);
  A.follow('task-9');
  assert.equal(A.state().following,'task-9');assert.equal(JSON.stringify(hashes.at(-1)),'{"follow":"task-9"}');
  assert.equal(toasts.at(-1),'Following Task task-9 · its verified result will open here');
  assert.ok(A.section().includes('Following Task task-9 · its verified result will open here; pin to keep the current view.'));assert.ok(A.section().includes('<activity-pin:>'),'the header and the followed row offer Pin');
  // The choice survives a feed outage and a store reset: only the viewer pins.
  A.absorbPage(unavailable());assert.equal(A.state().following,'task-9');
  A.absorbPage(page('RESET','e2',[admitted(300,'task-9','op300')],experiment('task-9','RUNNING')));assert.equal(A.state().following,'task-9');
  const openedBefore=opened.length;dirty=true;
  A.absorbPage(page('CONTINUED','e2',[evidence(301,'task-9')],experiment('task-9','SUCCEEDED')));
  assert.equal(A.state().pendingOpen,'task-9','the result waits while the draft is dirty');
  assert.equal(opened.length,openedBefore,'a dirty draft is never left for a followed result');
  assert.equal(toasts.at(-1),'Followed Task task-9 has a verified result; it opens when you finish editing');
  const toastsAfterNotice=toasts.length;
  A.absorbPage(page('CONTINUED','e2',[],experiment('task-9','SUCCEEDED')));
  assert.equal(toasts.length,toastsAfterNotice,'announced once, not on every poll');assert.equal(A.state().pendingOpen,'task-9');
  dirty=false;dialogOpen=true;A.absorbPage(page('CONTINUED','e2',[],{}));
  assert.equal(opened.length,openedBefore,'an open dialog also holds the reader');
  dialogOpen=false;A.absorbPage(page('CONTINUED','e2',[],{}));await settled();
  assert.deepEqual(opened.at(-1),'experiment:task-9','opened through the experiment readback once the reader is free');
  assert.equal(A.state().following,null);assert.equal(A.state().pendingOpen,null);assert.equal(JSON.stringify(hashes.at(-1)),'{"follow":""}');
  assert.equal(c.app.page,'portfolio');assert.equal(c.app.book,'book-a');assert.deepEqual(selected,[]);
  // Pinned: a followed Task that the viewer pins again leaves its verified result in the list.
  A.absorbPage(page('CONTINUED','e2',[admitted(302,'task-10','op302')],experiment('task-10','QUEUED')));
  A.follow('task-10');A.pin();assert.equal(A.state().following,null);assert.equal(JSON.stringify(hashes.at(-1)),'{"follow":""}');
  const pinnedOpened=opened.length;
  A.absorbPage(page('CONTINUED','e2',[evidence(303,'task-10')],experiment('task-10','SUCCEEDED')));await settled();
  assert.equal(opened.length,pinnedOpened,'pinned: nothing opens by itself');assert.equal(A.state().pendingOpen,null);
  assert.ok(!toasts.includes('Result verified · Task task-10 · open it from the activity list'),'pinned: the verified result counts as unseen and stays in the list, no toast (round 75)');
  assert.ok(A.section().includes('<activity-open:task-10>'),'the result stays one explicit click away');
  // A shared PLAN operated elsewhere: Follow inspects that exact hash, the Task follow state is untouched.
  const hash='7bf1e44881c6b21d2e7bbb95008947f7b24a35916a9b24829e8d12b8781dcb27';
  A.absorbPage(page('CONTINUED','e2',[item(304,'ProductOperationObserved',{operation:'EXPERIMENT_PLAN',caller:'EXTERNAL_AUTOMATION',phase:'REQUESTED',operation_ref:'op304',subject:{research_input_id:'factor-development'}}),item(305,'ProductOperationObserved',{operation:'EXPERIMENT_PLAN',caller:'EXTERNAL_AUTOMATION',phase:'RETURNED',operation_ref:'op304',status:'PLANNED',subject:{plan_hash:hash,research_input_id:'factor-development'}})],{}));
  assert.ok(A.section().includes('<activity-follow:op304>'),'a returned shared PLAN offers Follow');
  const hashesBefore=hashes.length;A.follow('op304');
  assert.deepEqual(inspected,[hash],'following a PLAN inspects exactly that hash');
  assert.equal(A.state().following,null);assert.equal(hashes.length,hashesBefore,'no Task follow state is invented for a PLAN');
  assert.ok(!reads.some(p=>p.includes('/api/experiments/run')),'nothing here ever requests a RUN');
  assert.equal(c.app.page,'portfolio');assert.equal(c.app.book,'book-a');assert.deepEqual(selected,[]);
  // 16. An automatic open is cancellable at its navigation boundary. The open of a followed
  //     Portfolio result waits on History; a pin, another follow or the reader's own
  //     navigation during that wait abandons it; a dialog or edit re-arms it; nothing opens
  //     twice and nothing opens under an older choice.
  const replay=(task,lifecycle)=>({[task]:projection(task,lifecycle)});
  let releaseHistory;historyGate=new Promise(r=>{releaseHistory=r;});
  A.absorbPage(page('CONTINUED','e2',[admitted(306,'task-11','op306')],replay('task-11','RUNNING')));A.follow('task-11');
  extraHistory=[{id:'result:ab307'}];
  A.absorbPage(page('CONTINUED','e2',[evidence(307,'task-11','PortfolioResearchResult')],replay('task-11','SUCCEEDED')));
  assert.equal(A.state().opening,true,'the open is in flight, waiting on History');assert.equal(A.state().pendingOpen,null);
  const beforePin=opened.length;A.pin();releaseHistory();historyGate=null;await settled();
  assert.equal(opened.length,beforePin,'pinned during the wait: the result did not open');assert.equal(A.state().following,null);assert.equal(A.state().opening,false);
  assert.ok(A.section().includes('<activity-open:task-11>'),'still one explicit click away');
  // Another follow during the wait: the old intent is abandoned, the new choice stands.
  historyGate=new Promise(r=>{releaseHistory=r;});
  A.absorbPage(page('CONTINUED','e2',[admitted(308,'task-12','op308'),admitted(309,'task-13','op309')],{...replay('task-12','RUNNING'),...replay('task-13','RUNNING')}));
  A.follow('task-12');extraHistory=[{id:'result:ab310'}];
  A.absorbPage(page('CONTINUED','e2',[evidence(310,'task-12','PortfolioResearchResult')],replay('task-12','SUCCEEDED')));
  assert.equal(A.state().opening,true);const before12=opened.length;
  A.follow('task-13');assert.equal(A.state().following,'task-13');
  releaseHistory();historyGate=null;await settled();
  assert.equal(opened.length,before12,'the abandoned open moved nothing');assert.equal(A.state().following,'task-13','the newer choice stands');assert.equal(A.state().opening,false);
  // The reader's own navigation during the wait: abandoned, follow ended, result announced.
  historyGate=new Promise(r=>{releaseHistory=r;});
  A.absorbPage(page('CONTINUED','e2',[evidence(311,'task-13','PortfolioResearchResult')],replay('task-13','SUCCEEDED')));
  assert.equal(A.state().opening,true,'waiting on History for a result it has not discovered yet');extraHistory=[{id:'result:ab310'},{id:'result:ab311'}];c.app.page='history';
  releaseHistory();historyGate=null;await settled();
  assert.equal(opened.length,before12,'navigation during the wait: nothing opened');assert.equal(A.state().following,null);assert.equal(A.state().pendingOpen,null);
  assert.equal(toasts.at(-1),'Followed Task task-13 has a verified result; open it from the activity list');
  c.app.page='portfolio';
  // A dialog that opens during the wait re-arms the pending open instead of cancelling it.
  let releaseOpen;openGate=new Promise(r=>{releaseOpen=r;});
  A.absorbPage(page('CONTINUED','e2',[admitted(312,'task-14','op312')],experiment('task-14','RUNNING')));A.follow('task-14');
  A.absorbPage(page('CONTINUED','e2',[evidence(313,'task-14')],experiment('task-14','SUCCEEDED')));
  assert.equal(A.state().opening,true);dialogOpen=true;releaseOpen();openGate=null;await settled();
  assert.equal(opened.length,before12,'a dialog at the boundary holds the open');assert.equal(A.state().following,'task-14');assert.equal(A.state().pendingOpen,'task-14');
  dialogOpen=false;A.absorbPage(page('CONTINUED','e2',[],{}));await settled();
  assert.equal(opened.at(-1),'experiment:task-14','and it opens once the reader is free');assert.equal(A.state().following,null);
  assert.equal(opened.filter(v=>v==='experiment:task-14').length,1,'opened exactly once');
  // 17. A restored follow reconciles against what is known instead of waiting for an event
  //     that will not recur: a retained verified group opens at once; a settled Task with no
  //     retained verification is asked from its owner (one readback per choice); a Task that
  //     ended without a result ends the follow; a moving Task without a group is watched.
  A.setFollowing('task-9');await settled();
  assert.equal(opened.at(-1),'experiment:task-9','restored follow of a retained verified Task opens it');assert.equal(A.state().following,null);
  knownTasks=[{task_id:'task-20',lifecycle:'SUCCEEDED',task_kind:'research_experiment'},{task_id:'task-21',lifecycle:'FAILED',task_kind:'research_experiment'},{task_id:'task-22',lifecycle:'RUNNING',task_kind:'research_experiment'}];
  A.setFollowing('task-20');await settled();
  assert.equal(opened.at(-1),'experiment:task-20','a settled Task with no retained verification is opened through its owner readback');assert.equal(A.state().following,null);
  const before21=opened.length;A.setFollowing('task-21');await settled();
  assert.equal(opened.length,before21);assert.equal(A.state().following,null);assert.equal(toasts.at(-1),'Followed Task task-21 ended '+c.codeWords('FAILED')+'; there is no result to open');
  A.setFollowing('task-22');await settled();assert.equal(A.state().following,'task-22','a moving Task waits for the feed');
  pages=[page('CONTINUED','e2',[],{'task-22':projection('task-22','RUNNING',{task_kind:'research_experiment'})})];await A.refresh();
  assert.ok(reads.at(-1).includes('watch=task-22'),'the followed Task is watched without a retained group');
  A.absorbPage(page('CONTINUED','e2',[],{'task-22':projection('task-22','SUCCEEDED',{task_kind:'research_experiment'})}));await settled();
  assert.equal(opened.at(-1),'experiment:task-22','a projection that settles is reconciled through the owner once');assert.equal(A.state().following,null);
  assert.ok(!reads.some(p=>p.includes('/api/experiments/run')),'nothing here ever requests a RUN');
  assert.equal(c.app.page,'portfolio');assert.equal(c.app.book,'book-a');assert.deepEqual(selected,[]);
  // 18. A Task that stops moving in Task Control's report (Data calls this) is said once, by its
  //     name and its state's word, with one press to open it; the followed Task says its own
  //     ending; a Task still moving says nothing. In the background a system notification follows
  //     only when the viewer turned it on and the browser granted it; the browser is asked once and
  //     a refusal keeps the choice off (the user, 2026-09-25: 任何任务结束都提示 / 切走时也能知道).
  const ended=(id,lifecycle)=>({task_id:id,lifecycle,task_kind:'research_experiment'});
  A.taskSettled(ended('task-30','SUCCEEDED'));
  assert.equal(toasts.at(-1),'Task completed');assert.equal(said.at(-1).vars.name,'Study task-30');assert.equal(JSON.stringify(said.at(-1).act),JSON.stringify({action:'task',value:'task-30',word:'Open'}));
  A.taskSettled(ended('task-31','BLOCKED'));assert.equal(toasts.at(-1),'Task blocked');
  A.taskSettled(ended('task-31','REVIEW_PENDING'));assert.equal(toasts.at(-1),'Task needs a decision');
  const quiet=toasts.length;
  A.taskSettled(ended('task-32','RUNNING'));assert.equal(toasts.length,quiet,'a Task still moving says nothing');
  knownTasks=[];A.setFollowing('task-33');await settled();
  A.taskSettled(ended('task-33','FAILED'));assert.equal(toasts.length,quiet,'the followed Task says its own ending');
  A.setFollowing(null);
  const shown=[],pressed=[];let pref=false;
  c.Notification=function(title,o){const n={title,body:o.body,tag:o.tag,close(){}};shown.push(n);return n;};c.Notification.permission='granted';
  c.readPreference=(k)=>k==='taskNotices'?pref:null;c.ACTIONS={task:(id)=>pressed.push(id)};c.window.focus=()=>{};
  c.document.hidden=true;A.taskSettled(ended('task-34','FAILED'));
  assert.equal(shown.length,0,'not turned on: the toast and the title say it');assert.equal(toasts.at(-1),'Task failed');
  pref=true;c.Notification.permission='denied';A.taskSettled(ended('task-35','FAILED'));assert.equal(shown.length,0,'refused by the browser');
  c.Notification.permission='granted';A.taskSettled(ended('task-36','SUCCEEDED'));
  assert.deepEqual([shown[0].title,shown[0].body,shown[0].tag],['Study task-36','Task completed','alphalattice-task-task-36']);
  shown[0].onclick();assert.deepEqual(pressed,['task-36'],'pressing the notification opens its Task');
  c.document.hidden=false;A.taskSettled(ended('task-37','SUCCEEDED'));assert.equal(shown.length,1,'a page in view: the toast alone');
  const kept=[];c.savePreference=(k,v)=>kept.push([k,v]);let prompts=0;
  c.Notification.permission='default';c.Notification.requestPermission=async()=>{prompts+=1;c.Notification.permission='denied';return 'denied';};
  await A.setNotices(true);assert.deepEqual(kept.at(-1),['taskNotices',false]);assert.equal(prompts,1);assert.equal(A.noticesState(),'denied');
  await A.setNotices(true);assert.equal(prompts,1,'a refusal is never prompts again behind the viewer');
  c.Notification.permission='default';c.Notification.requestPermission=async()=>{prompts+=1;c.Notification.permission='granted';return 'granted';};
  await A.setNotices(true);assert.deepEqual(kept.at(-1),['taskNotices',true]);assert.equal(A.noticesState(),'on');
  await A.setNotices(false);assert.deepEqual(kept.at(-1),['taskNotices',false]);
  delete c.Notification;assert.equal(A.noticesState(),'unsupported');
  const words631=library.words(root),codes631=library.codes();
  Object.assign(c,{t:words631.t,codeWords:words631.codeWords,notify:(key,vars)=>toasts.push(words631.t(key,vars))});
  for(const lang of ['en','zh']) {
    words631.I18N.set(lang);
    for(const state of codes631.lifecycles) {
      const task='follow-'+lang+'-'+state,before=toasts.length;
      knownTasks=[ended(task,state)];A.setFollowing(task);await settled();
      if(['BLOCKED','CANCELLED'].includes(state)) {
        assert.equal(toasts.at(-1),words631.t('Followed Task {task} ended {state}; there is no result to open',{task,state:words631.codeWords(state)}),lang+' followed Task uses the shared state word: '+state);
        assert.equal(A.state().following,null);
      } else if(state!=='SUCCEEDED') {
        assert.equal(toasts.length,before,'a moving or held Task stays followed without an ending notice');
      }
      A.setFollowing(null);
    }
  }
  finish();
})().catch(e=>{console.error(e);process.exitCode=1;});
