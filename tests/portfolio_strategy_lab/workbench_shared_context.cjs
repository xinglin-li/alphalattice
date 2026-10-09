// Shared research context in the declaration editor: a PLAN operated elsewhere is read by its
// exact hash beside the draft; a dirty draft is never replaced without an explicit adoption; the
// adoption confirmation is bound to the reviewed transaction (exact answer, draft revision,
// action) and refuses anything that moved meanwhile; a late answer to an earlier inspection is
// rejected; adopting and re-PLANning posts only PLAN, never RUN; a MISSING or admitted preview
// offers nothing to adopt or re-PLAN; a large differing block is bounded; adopting while the
// editor's own preparation (the controls read) is still pending takes that work over and
// releases its busy state, so the requested PLAN is sent. No HTTP, model or data work:
// `Data.read`/`Data.post` are scripted with owner-shaped bodies, gated where a race is the claim.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const library=require('./workbench_library.cjs'); // the library's constants and the scripts' parameters, from the source (Q2)
const finish=library.guard('workbench_shared_context');
const root=process.argv[2],reads=[],posts=[],hashes=[],dialogs=[],closes=[],toasts=[];
const pending=[];let previews={},controlsBody=null,planBody=null,controlsGate=null;
const H=(n)=>String(n).repeat(64).slice(0,64);
const doc={experiment:{kind:'factor.screening-development',determinism:{seed:1}},factor:{factor_ids:['vol_21']}};
const shared=(hash,status,extra={})=>({plan_hash:hash,status,retention:'IN_MEMORY_UNTIL_EXPIRY_OR_RESTART',retained_previews:2,preview_capacity:8,
  ...(['AVAILABLE','EXPIRED','INVALID'].includes(status) ? {caller:'EXTERNAL_AUTOMATION',previewed_at:'2026-09-14T12:00:00Z',expires_at:'2026-09-14T13:00:00Z',
    program:{kind:doc.experiment.kind},document:{...doc,experiment:{...doc.experiment,determinism:{seed:2}}},yaml:'experiment:\n  kind: factor.screening-development\n  determinism:\n    seed: 2\nfactor:\n  factor_ids:\n  - vol_21\n',
    research_input_id:'factor-development',input_binding_hash:H('b'),execution_preview:{},
    next_requests:{...(status==='AVAILABLE' ? {run:{operation:'EXPERIMENT_RUN',experiment_plan_hash:hash}} : {}),replan:{operation:'EXPERIMENT_PLAN',research_input_id:'factor-development',input_binding_hash:H('b'),experiment_document:{...doc}}}} : {}),
  ...extra});
const c={URLSearchParams,console,JSON,Object,Array,String,Number,Math,Promise,Error,Boolean,
  LiveGoals:{pages:new Set()},
  window:{},app:{page:'lab',yaml:'',mode:'yaml'},
  Data:{workspaceStatus:'ready',inputs:()=>[{id:'factor-development',binding_hash:H('b'),available:true,lifecycle:'REGISTERED',date:'2026-08-02'}],
    read:(p)=>{reads.push(p);if(p.startsWith('/api/experiments/controls'))return controlsGate ? controlsGate.then(()=>controlsBody) : Promise.resolve(controlsBody);
      const hash=new URLSearchParams(p.split('?')[1]).get('experiment_plan_hash');
      return new Promise((resolve,reject)=>pending.push({hash,resolve,reject}));},
    post:async(p,payload)=>{posts.push([p,JSON.parse(JSON.stringify(payload))]);if(p.endsWith('/plan'))return planBody;throw Error('probe: only PLAN may be posted, got '+p);},
    refreshHistory:async()=>{}},
  clone:v=>JSON.parse(JSON.stringify(v)),render:()=>{},replaceHash:(v)=>hashes.push(JSON.parse(JSON.stringify(v))),objectEntry:()=>true,
  hashParams:()=>new URLSearchParams(),
  Lab:{markDraftChanged(){},onYamlInput(e){c.app.yaml=e.target.value;}},codeEditor:(id,text,o={})=>`<textarea id="${id}" ${o.attrs||''}>${text}</textarea>`,codeMarkup:(s)=>String(s),codeEsc:(s)=>String(s),raw:(s)=>String(s),json:(v)=>JSON.stringify(v,null,2),
  LiveTasks:{select:async()=>{}},LiveViews:{cutoffText:()=>'',inputState:()=>'',researchTiming:()=>''},closeDialog:()=>closes.push(1),openDialog:(title,sub,body,actions)=>{dialogs.push({title,sub,body:String(body),actions:String(actions)});if(String(title).includes('edited draft'))queueMicrotask(()=>c.R.answerReplace('replace'));},alertDialog:(title,body,actions)=>c.openDialog(title,'',body,actions),
  notify:(m)=>toasts.push(m),t:(s,vars)=>s.replace(/\{(\w+)\}/g,(_,k)=>String(vars?.[k]??'')),
  html:(s,...v)=>s.reduce((a,p,i)=>a+p+(Array.isArray(v[i])?v[i].join(''):v[i]??''),''),kv:(rows)=>rows.map(([k,v])=>k+'='+v).join(';'),
  btn:(label,action,value)=>`<${action}:${value}>{${label}}`,codeWords:(code)=>{const w=String(code??'').replace(/[._]+/g,' ').trim();return w?w[0]+w.slice(1).toLowerCase():'—';},countText:(n,one,many)=>String(Number(n)===1?one:many).replace('{n}',String(n)),pluralText:(n,one,many,args={})=>String(Number(n)===1?one:many).replace(/\{(\w+)\}/g,(m,k)=>args&&args[k]!==undefined?args[k]:m),notRead:(title,error,words='',action='')=>`<div class="warning">${title}:${error}${words?' '+words:''}${action}</div>`,noteLine:(title,body='',tone='',action='')=>`<p class="note-line">${title}${body?' · '+body:''}${action||''}</p>`,hint:(term)=>term,factsRef:(title,body)=>`<p>${title}</p>${body}`,codeRef:(title,text)=>'<p>'+title+'</p><template>'+(typeof text==='string' ? text : JSON.stringify(text))+'</template>',refCell:(uri)=>'<span>'+uri+'</span>',hashCell:(h)=>'<span>'+(h||'—')+'</span>',readingPane:(title,kind,body,close)=>'<aside>'+title+body+'</aside>',sourceRows:(rows)=>rows.map((r)=>[r.label,r.value]),badge:(tone,label)=>`[${tone}:${label}]`,stateLine:(x,o={})=>`[${(typeof x==='string'?x:(x?.lifecycle??x?.state??x?.status??''))}:${o.word??''}]`,skeleton:(shape='rows')=>'SKELETON('+shape+')',statusDot:(s,l)=>'['+s+(l?':'+l:'')+']',emptyState:(s,a='')=>'EMPTY('+s+')'+(a||''),banner:(a,b)=>`(${a}|${b})`,mono:(v,n)=>String(v).slice(0,n||8),actorWords:(caller)=>({HUMAN:'You',EXTERNAL_AUTOMATION:'Agent · API'})[caller]||String(caller ?? '—'),
  prerequisitesPanel:(pr)=>pr?`<prerequisites ${pr.flow}>`:'', // U60: the builder components.js owns
  icon:()=>'',copyText:()=>{},tabStrip:(label,items)=>items.map(x=>(x.on?'*':'')+x.word+' ').join(''),picker:(id,choices,o={})=>choices.map(ch=>{const [v,l]=Array.isArray(ch)?ch:[ch.value,ch.title];return '('+(v===o.selected?'*':'')+l+')';}).join(''),segBtn:()=>'',typedBtn:library.stubs.empty,link:(label)=>String(label),objectHead:library.stubs.empty,propertyChip:(label,value)=>`{${label}:${value}}`,
  subjectChoice:library.stubs.subjectPicker,objectHead:library.stubs.headingSubject};
c.Data.readShared = (...args) => c.Data.read(...args);
library.context(c, root);vm.runInContext(fs.readFileSync(path.join(root,'status.js'),'utf8'),c);vm.runInContext(fs.readFileSync(path.join(root,'live-research.js'),'utf8')+';globalThis.R=LiveResearch;',c);
const view=()=>c.R.page()+c.R.reader(); // round 69: the Scope / PLAN reader is the inspector's body, read beside the page
const answer=(hash,body)=>{const i=pending.findIndex(v=>v.hash===hash);assert.ok(i>=0,'no pending inspection for '+hash.slice(0,4));const [p]=pending.splice(i,1);p.resolve(body);};
const settle=()=>new Promise(r=>setImmediate(r));
(async()=>{
  const R=c.R;
  controlsBody={status:'READY',prerequisites:{flow:'FACTOR_STUDY',needs:[],present:{FACTOR_STUDY:[]},missing:[],detail:'A Factor study needs only its research input.',next_requests:{}},template:doc,yaml:'experiment:\n  kind: factor.screening-development\n  determinism:\n    seed: 1\nfactor:\n  factor_ids:\n  - vol_21\n',input_id:'factor-development',input_binding_hash:H('b'),controls:[]};
  await R.ready();assert.equal(R.dirty(),false);const original=c.app.yaml;assert.ok(original.includes('seed: 1'));
  assert.ok(view().includes('<prerequisites FACTOR_STUDY>'),'U60: the controls answer says what the kind needs, under the work surface');
  // 1. Inspecting a shared PLAN reads exactly that hash, records it in the route, and shows it
  //    beside the draft. The draft is untouched.
  const first=R.inspectShared(H('1'));
  assert.equal(R.shared().loading,true);assert.ok(view().includes('Reading the shared PLAN'));
  assert.equal(JSON.stringify(hashes.at(-1)),JSON.stringify({page:'lab',plan:H('1')}));
  assert.equal(reads.at(-1),'/api/experiments/preview?experiment_plan_hash='+H('1'));
  answer(H('1'),shared(H('1'),'AVAILABLE'));await first;
  assert.equal(R.shared().body.status,'AVAILABLE');assert.equal(c.app.yaml,original,'reading never edits the draft');
  let page=view();assert.ok(page.includes('Shared PLAN'));assert.ok(page.includes('[planned:Available]'));
  assert.ok(page.includes('Agent · API'));assert.ok(page.includes('<research-adopt-shared:>'));assert.ok(page.includes('<research-replan-shared:>'));
  assert.ok(page.includes('reading it grants nothing'));
  assert.equal(R.routeContext().plan,H('1'));
  // 2. A dirty draft is not replaced: adopting shows the differences first and keeps the draft;
  //    only the explicit confirmation of that exact review adopts, and the route keeps naming
  //    the shared PLAN.
  R.edit({target:{id:'yamlEditor',value:original.replace('seed: 1','seed: 7')}});assert.equal(R.dirty(),true);
  assert.ok(view().includes('<research-adopt-shared:>{Inspect and adopt}'),'a dirty draft is offered inspection, not replacement');
  assert.equal(R.adoptShared(),false);assert.equal(dialogs.length,1);
  assert.ok(dialogs[0].body.includes('-     seed: 7'));assert.ok(dialogs[0].body.includes('+     seed: 2'));assert.ok(dialogs[0].body.includes('Adopting replaces 1 line of your draft with 1 line'));
  const txn1=R.adopting();assert.equal(txn1.hash,H('1'));assert.equal(txn1.action,'adopt');
  assert.ok(dialogs[0].actions.includes('<research-adopt-confirm:'+txn1.id+'>{Adopt shared declaration}'));assert.ok(!dialogs[0].actions.includes('<close:'),"the dialog closes once, by its head's x (N6, law 131)");
  assert.ok(c.app.yaml.includes('seed: 7'),'the dialog alone changes nothing');assert.equal(R.dirty(),true);
  // A confirmation for another transaction is refused outright.
  assert.throws(()=>R.confirmAdoption(txn1.id+1000),/no longer current/);assert.ok(c.app.yaml.includes('seed: 7'));
  // The reviewed transaction, confirmed: the shared declaration is now the draft.
  assert.equal(R.adoptShared(),false);const txn1b=R.adopting();
  assert.equal(R.confirmAdoption(String(txn1b.id)),true);assert.ok(closes.length>=1);
  assert.ok(c.app.yaml.includes('seed: 2'),'the shared declaration is now the draft');assert.equal(R.dirty(),false);assert.equal(R.adopting(),null);
  assert.equal(hashes.at(-1).plan,H('1'));assert.equal(R.context().input_binding_hash,H('b'));
  assert.deepEqual(posts,[],'adopting posts nothing');
  // 2b. Stale confirmations. Review A, then another inspection (B) answers before the
  //     confirmation: the original dialog's confirmation adopts nothing. Review again, edit the
  //     draft, confirm: refused too. A later answer to the same hash (new ticket) also refuses.
  R.edit({target:{id:'yamlEditor',value:c.app.yaml.replace('seed: 2','seed: 8')}});
  assert.equal(R.adoptShared(),false);const txnA=R.adopting();
  const inspectB=R.inspectShared(H('9'));assert.equal(R.adopting(),null,'a new inspection ends the review');assert.ok(closes.length>=2,'and closes its dialog');
  answer(H('9'),shared(H('9'),'AVAILABLE',{yaml:'experiment:\n  kind: factor.screening-development\n  determinism:\n    seed: 3\nfactor:\n  factor_ids:\n  - vol_21\n'}));await inspectB;
  assert.throws(()=>R.confirmAdoption(String(txnA.id)),/no longer current/);
  assert.ok(c.app.yaml.includes('seed: 8'),'unreviewed B was not adopted');assert.equal(R.dirty(),true);assert.equal(R.shared().hash,H('9'));
  assert.equal(R.adoptShared(),false);const txnB=R.adopting();assert.equal(txnB.hash,H('9'));
  R.edit({target:{id:'yamlEditor',value:c.app.yaml.replace('seed: 8','seed: 88')}});assert.equal(R.adopting(),null,'an edit ends the review');
  assert.throws(()=>R.confirmAdoption(String(txnB.id)),/no longer current/);assert.ok(c.app.yaml.includes('seed: 88'));
  assert.equal(R.adoptShared(),false);const txnC=R.adopting();
  const again=R.inspectShared(H('9'));answer(H('9'),shared(H('9'),'AVAILABLE'));await again; // same hash, new answer
  assert.throws(()=>R.confirmAdoption(String(txnC.id)),/no longer current/);assert.ok(c.app.yaml.includes('seed: 88'),'a renewed answer needs a renewed review');
  // A review whose shared answer is swapped underneath while its dialog stays open (the panel
  // state moved without ending the review) is refused at confirmation, not silently adopted.
  assert.equal(R.adoptShared(),false);const txnD=R.adopting();
  R.shared().ticket+=1; // the answer the dialog reviewed is no longer the one on the panel
  assert.throws(()=>R.confirmAdoption(String(txnD.id)),/changed since this review/);assert.ok(c.app.yaml.includes('seed: 88'));assert.equal(R.adopting(),null);
  const backTo1=R.inspectShared(H('1'));answer(H('1'),shared(H('1'),'AVAILABLE'));await backTo1;
  assert.equal(R.adoptShared(),false);R.confirmAdoption(String(R.adopting().id));assert.ok(c.app.yaml.includes('seed: 2'));assert.equal(R.dirty(),false);
  // 3. Dismissing keeps the draft (dirty or not) and clears the route's plan.
  R.edit({target:{id:'yamlEditor',value:c.app.yaml.replace('seed: 2','seed: 9')}});
  const second=R.inspectShared(H('2'));answer(H('2'),shared(H('2'),'EXPIRED'));await second;
  assert.ok(view().includes('[expired:Expired]'));assert.ok(view().includes('<research-shared-dismiss:>{Keep my draft}'));
  R.dismissShared();assert.equal(R.shared(),null);assert.ok(c.app.yaml.includes('seed: 9'));assert.equal(R.dirty(),true);
  assert.equal(JSON.stringify(hashes.at(-1)),JSON.stringify({plan:''}));
  // 4. A late answer to an earlier inspection is rejected: the newer inspection owns the panel.
  const third=R.inspectShared(H('3')),fourth=R.inspectShared(H('4'));
  answer(H('4'),shared(H('4'),'AVAILABLE'));await fourth;assert.equal(R.shared().hash,H('4'));assert.equal(R.shared().body.plan_hash,H('4'));
  answer(H('3'),shared(H('3'),'AVAILABLE'));await third;
  assert.equal(R.shared().hash,H('4'),'the earlier answer does not replace the later inspection');assert.equal(R.shared().body.plan_hash,H('4'));
  // 5. Adopt and PLAN again posts exactly one PLAN of the adopted declaration; never a RUN. A
  //    dirty draft gets the diff dialog first, and confirming that review performs the exact
  //    adoption and the one PLAN -- no further click.
  planBody={status:'PLANNED',plan_hash:H('4'),program:{kind:doc.experiment.kind},execution_preview:{},document:doc,preview:{retention:'IN_MEMORY_UNTIL_EXPIRY_OR_RESTART'},next_requests:{run:{operation:'EXPERIMENT_RUN',experiment_plan_hash:H('4')}}};
  const dialogsBefore=dialogs.length;
  assert.equal(R.replanShared(),false);assert.equal(dialogs.length,dialogsBefore+1,'dirty: the differences are shown, nothing is planned');
  assert.ok(dialogs.at(-1).actions.includes('{Adopt and PLAN again}'));assert.equal(R.adopting().action,'replan');
  assert.deepEqual(posts,[]);
  await R.confirmAdoption(String(R.adopting().id));
  assert.equal(posts.length,1,'the confirmed review adopted and planned once');assert.equal(posts[0][0],'/api/experiments/plan');
  assert.equal(posts[0][1].research_input_id,'factor-development');assert.equal(posts[0][1].input_binding_hash,H('b'));
  assert.ok(posts[0][1].experiment_yaml.includes('seed: 2'));assert.equal(posts[0][1].operation,undefined);
  assert.equal(c.app.plan.plan_hash,H('4'));assert.equal(c.app.labTask,undefined,'no Task was admitted');
  // 6. MISSING offers nothing to adopt; an admitted preview points at its Task and cannot re-PLAN.
  const fifth=R.inspectShared(H('5'));answer(H('5'),shared(H('5'),'MISSING',{next_action:'EXPERIMENT_PLAN'}));await fifth;
  page=view();assert.ok(page.includes('[metadata:Missing]'));assert.ok(page.includes('Only an explicit new PLAN recreates it'));
  assert.ok(!page.includes('<research-adopt-shared:>'));assert.ok(!page.includes('<research-replan-shared:>'));
  assert.throws(()=>R.adoptShared(),/no readable declaration/);
  const sixth=R.inspectShared(H('6'));answer(H('6'),shared(H('6'),'ADMITTED_AS_TASK',{existing_task:{task_id:'task-1',lifecycle:'SUCCEEDED',readback:{operation:'EXPERIMENT_READBACK',task_id:'task-1'}}}));await sixth;
  page=view();assert.ok(page.includes('[queued:Admitted as task]'));assert.ok(page.includes('<task:task-1>'));assert.ok(!page.includes('<research-replan-shared:>'));
  // 7. An unreadable answer is shown as such and can be dismissed; the draft is untouched.
  const seventh=R.inspectShared(H('7'));const bad=pending.find(v=>v.hash===H('7'));bad.reject(Error('local_web.plan_hash_invalid'));await seventh;
  assert.equal(R.shared().error,'local_web.plan_hash_invalid');assert.ok(view().includes('Shared PLAN unreadable'));
  assert.equal(posts.length,1,'still exactly one PLAN, no RUN');assert.ok(!posts.some(([p])=>p.endsWith('/run')));
  // 8. The comparison is bounded: an exact line diff inside a bounded middle block; a larger
  //    differing block is reported as removed-then-added without a quadratic table, and the
  //    dialog shows a bounded window of it.
  const many=(seed,n)=>'experiment:\n  kind: factor.screening-development\n'+Array.from({length:n},(_,i)=>`  k${i}: ${seed}-${i}`).join('\n')+'\nfactor:\n  factor_ids:\n  - vol_21\n';
  const small=R.diffLines(many('a',300),many('b',300));assert.equal(small.exact,true);assert.equal(small.removed,300);assert.equal(small.added,300);
  const t0=Date.now();const big=R.diffLines(many('a',4000),many('b',4000));const ms=Date.now()-t0;
  assert.equal(big.exact,false);assert.equal(big.removed,4000);assert.equal(big.added,4000);assert.ok(ms<2000,'bounded: '+ms+' ms');
  assert.equal(big.lines.filter(([k])=>k===' ').length,2+4,'common prefix and suffix are kept as context');
  const eighth=R.inspectShared(H('8'));answer(H('8'),shared(H('8'),'AVAILABLE',{yaml:many('b',4000)}));await eighth;
  R.edit({target:{id:'yamlEditor',value:many('a',4000)}});
  assert.equal(R.adoptShared(),false);const bigDialog=dialogs.at(-1);
  assert.ok(bigDialog.body.includes('Adopting replaces 4000 lines of your draft with 4000 lines'));assert.ok(bigDialog.body.includes('too large to compare line by line'));
  assert.ok(bigDialog.body.includes('more lines not shown'));assert.ok(bigDialog.body.split('\n').length<450,'the dialog shows a bounded window');
  // 9. Adoption versus pending preparation. The editor's initial controls read is held; a shared
  //    declaration is inspected and adopted meanwhile; the old controls answer is then released;
  //    a PLAN is requested. The adoption took the pending work over: the released answer changes
  //    nothing, the editor is not left busy, and the PLAN is actually sent.
  let releaseControls;controlsGate=new Promise(r=>{releaseControls=r;});
  R.forget(); // the version was read before; the kept answer would make this switch instant, and the race needs a read in flight
  const loading=R.useInput(JSON.stringify(['factor-development',H('b')])); // a fresh load, held on its controls read
  await settle(); // the edited draft was asked about in the product's dialog (answered above), then the load began
  assert.ok(dialogs.at(-1).title.includes('edited draft'),'an edited draft is asked about before it is replaced');
  assert.ok(view().includes('Waiting for the product owner'),'the editor is busy loading');
  const ninth=R.inspectShared(H('9'));answer(H('9'),shared(H('9'),'AVAILABLE',{yaml:'experiment:\n  kind: factor.screening-development\n  determinism:\n    seed: 3\nfactor:\n  factor_ids:\n  - vol_21\n'}));await ninth;
  assert.equal(R.adoptShared(),true,'a clean (loading) editor adopts at once');
  assert.ok(c.app.yaml.includes('seed: 3'));assert.ok(!view().includes('Waiting for the product owner'),'adoption released the superseded load');
  releaseControls();controlsGate=null;await loading;await settle();
  assert.ok(c.app.yaml.includes('seed: 3'),'the released controls answer does not replace the adopted declaration');
  assert.ok(!view().includes('Waiting for the product owner'),'and does not leave the editor busy');
  const postsBefore=posts.length;planBody={status:'PLANNED',plan_hash:H('9'),program:{kind:doc.experiment.kind},execution_preview:{},document:doc,preview:{retention:'IN_MEMORY_UNTIL_EXPIRY_OR_RESTART'},next_requests:{run:{operation:'EXPERIMENT_RUN',experiment_plan_hash:H('9')}}};
  assert.equal(await R.preview(),true,'the requested PLAN is sent');assert.equal(posts.length,postsBefore+1);assert.equal(posts.at(-1)[0],'/api/experiments/plan');
  assert.ok(posts.at(-1)[1].experiment_yaml.includes('seed: 3'));assert.equal(c.app.plan.plan_hash,H('9'));
  // The same race through "Adopt and PLAN again": one adoption, one PLAN, nothing dropped, and a
  // PLAN that could not be sent would be an error, never a silent success.
  controlsGate=new Promise(r=>{releaseControls=r;});
  const loading2=R.useInput(JSON.stringify(['factor-development',H('b')]));
  const tenth=R.inspectShared(H('9'));answer(H('9'),shared(H('9'),'AVAILABLE'));await tenth;
  const before2=posts.length;await R.replanShared();
  assert.equal(posts.length,before2+1,'adopt-and-PLAN-again sent its one PLAN despite the pending load');assert.equal(posts.at(-1)[0],'/api/experiments/plan');
  releaseControls();controlsGate=null;await loading2;await settle();
  assert.ok(!view().includes('Waiting for the product owner'));assert.equal(c.app.plan.plan_hash,H('9'));
  assert.ok(!posts.some(([p])=>p.endsWith('/run')),'never a RUN');
  const words631=library.words(root),codes631=library.codes();
  Object.assign(c,{t:words631.t,codeWords:words631.codeWords});
  for(const lang of ['en','zh']) {
    words631.I18N.set(lang);
    for(const [i,state] of codes631.lifecycles.entries()) {
      const hash=String(6310+(lang==='zh'?10:0)+i).padStart(64,'0'),task='task'+i;
      const reading=R.inspectShared(hash);
      answer(hash,shared(hash,'ADMITTED_AS_TASK',{existing_task:{task_id:task,lifecycle:state}}));await reading;
      assert.ok(view().includes(task+' · '+words631.codeWords(state)),lang+' admitted Task uses the shared state word: '+state);
    }
  }
  finish();
})().catch(e=>{console.error(e);process.exitCode=1;});
