// Workspace page/state/confirmation contract, and the preparation scene over the owners' bodies
// (shaped after real recordings); no filesystem mutation, HTTP, model or data work.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const library=require('./workbench_library.cjs'); // the library's constants and the scripts' parameters, from the source (Q2)
const finish=library.guard('workbench_workspace');
// the shared percentage and reading helpers as components.js defines them (by their markers)
const runShapes=(root)=>{const src=require('node:fs').readFileSync(require('node:path').join(root,'components.js'),'utf8');const a=src.indexOf('/* ---- run shapes (round 72)');const b=src.indexOf('/* ---- end of run shapes ---- */');return src.slice(a,b)+';globalThis.stepList=stepList;globalThis.runLog=runLog;globalThis.logLine=logLine;globalThis.observationLine=observationLine;globalThis.logMove=logMove;globalThis.logHeld=logHeld;globalThis.logRetain=logRetain;';};
const percentRule=(root)=>{const src=require('node:fs').readFileSync(require('node:path').join(root,'components.js'),'utf8');const a=src.indexOf('/* ---- percentages for reading');const b=src.indexOf('const pctFraction');const e=src.indexOf('\n',b);return src.slice(a,e+1)+';globalThis.pctNumber=pctNumber;globalThis.pctText=pctText;globalThis.pctFraction=pctFraction;globalThis.short=short;globalThis.mono=mono;globalThis.roundDecimalText=roundDecimalText;globalThis.CODE_WORDS=CODE_WORDS;globalThis.codeWords=codeWords;globalThis.coded=coded;globalThis.methodWords=methodWords;globalThis.countText=countText;globalThis.pluralText=pluralText;globalThis.pluralText=pluralText;globalThis.whenText=whenText;';};
const root=process.argv[2],reads=[],posts=[],tasks=[],drafts=[];
const bodies={
  '/api/data-update':{status:'NO_UPDATE_PUBLICATION',inputs:{data_through:'2026-08-03',panel_through:'2026-08-03'}},
  '/api/data-update/plan':{status:'CONFIRMATION_REQUIRED',plan_hash:'plan',change:{additions:['one'],removals:[]}},
  '/api/data-update/confirm':{status:'APPROVED'},'/api/data-update/run':{status:'QUEUED',task_id:'data-task'},
  '/api/research-inputs':{inputs:[{input_id:'family',versions:[{binding_hash:'binding',available:true,end:'2026-08-03'}]}]},
  '/api/research-inputs/plan':{status:'REUSED_EXACT',input_id:'family',binding_hash:'binding'},
  '/api/workspace/preparation':{status:'SUCCEEDED',inputs:[]},
  '/api/workspace/data-issues':{issues:[{case:{case_token:'case',evidence_hash:'evidence',listing_ids:['one'],options:[{option_id:'quarantine',option_hash:'option'}]},subjects:{one:'ONE'},options_current:true,confirmable_option_ids:['quarantine']}],continuations:[],
    next_requests:{'preview:case:quarantine':{operation:'DATA_ISSUE_PREVIEW',data_issue_case_token:'case',data_issue_evidence_hash:'evidence',data_issue_option_id:'quarantine',data_issue_option_hash:'option'}}},
  '/api/workspace/data-issues/preview':{status:'PREVIEW',claim:'NO_DATA_TRUTH_OR_CURRENT_READINESS_CLAIM',
    next_requests:{confirm:{operation:'DATA_ISSUE_CONFIRM',data_issue_case_token:'case',data_issue_evidence_hash:'evidence',data_issue_option_id:'quarantine',data_issue_option_hash:'option'}}},
  '/api/workspace/data-issues/confirm':{status:'DECISION_RECORDED'},
  '/api/workspace/storage':{status:'AVAILABLE',display:{},inputs:[],pending_cleanup:[]},
  '/api/workspace/storage/plan':{status:'CONFIRMATION_REQUIRED',plan_hash:'cleanup',targets:{'old.parquet':'hash'}},
  '/api/workspace/storage/confirm':{status:'COMPLETED'},
};
let delayed=null,failed=null,modal=null;
const c={console,URLSearchParams,app:{page:'data'},ROUTES:{data:['','Data'],issues:['','Issues'],inputs:['','Inputs'],storage:['','Storage']},
  Data:{read:async p=>{reads.push(p);const key=p.split('?')[0];return bodies[key]!==undefined?bodies[key]:bodies[p];},post:async(p,b)=>{posts.push([p,JSON.parse(JSON.stringify(b))]);if(p===failed)throw Error('owner_refused');if(delayed?.path===p)await delayed.wait;return bodies[p];},setInputs(){},
    setPreparation(b){discovered.push(b?.task_id||null);discovery.current=b;},preparation:()=>discovery.current,inputVersion:()=>null,inputs:()=>[],workspace:()=>'ws',tasks:()=>[],runsOf:(k)=>c.Data.tasks().filter(v=>k==='task'||['workspace_data_update','workspace_preparation'].includes(v.task_kind)).map(v=>({id:v.task_id,kind:v.task_kind,name:v.task_kind,state:v.lifecycle,starter:'',started:v.running_since||'',finished:v.last_activity_at||'',current:v.current_stage||'',verified:[v.verified_stage_count,v.total_stage_count],object:v})),groupByDay:(rs)=>{const g=new Map();for(const r of rs){const k=String(r.finished||r.started||'').slice(0,10)||'Undated';if(!g.has(k))g.set(k,[]);g.get(k).push(r);}return g;},},
  LiveTasks:{select:async id=>tasks.push(id),standing:v=>['attention','info',v.lifecycle],liveness:v=>v.liveness?.status||''},LiveResearch:{useInput:async key=>drafts.push(key)},
  html:(s,...v)=>s.reduce((a,p,i)=>a+p+(v[i]??''),''),t:(s,vars)=>s.replace(/^[a-z0-9-]+\|/,'').replace(/\{(\w+)\}/g,(m,k)=>vars&&vars[k]!==undefined?vars[k]:m),render(){renders.push(c.app.page);},patchMain(){patches.push(c.app.page);},closeDialog(){},notify:(m)=>toasts.push(m),
  openDialog:(a,b,content)=>{modal={title:b,content};},raw:(v)=>String(v),esc:(v)=>String(v),json:(v)=>JSON.stringify(v,null,2),btnAttrs:(label)=>String(label),
  meter:()=>'',notRead:(title,error,words='',action='')=>`<div class="warning">${title}:${error}${words?' '+words:''}${action}</div>`,noteLine:(title,body='',tone='',action='')=>`<p class="note-line">${title}${body?' · '+body:''}${action||''}</p>`,hint:(term)=>term,factsRef:(title,body)=>`<p>${title}</p>${body}`,codeRef:(title,text)=>'<p>'+title+'</p><template>'+(typeof text==='string' ? text : JSON.stringify(text))+'</template>',refCell:(uri)=>'<span>'+uri+'</span>',hashCell:(h)=>'<span>'+(h||'—')+'</span>',readingPane:(title,kind,body,close)=>'<aside>'+title+body+'</aside>',sourceRows:(rows)=>rows.map((r)=>[r.label,r.value]),kv:rows=>JSON.stringify(rows),panel:(a,b,content)=>a+b+content,rail:(items)=>String(items),separator:()=>'',stat:(a,b,n)=>a+':'+b+':'+(n||''),figureTile:(label,value,to)=>'FIG('+label+':'+value+(to?'>'+(to.page||to.action):'')+')',measureStrip:(items)=>String(items),tabStrip:(label,items)=>'TABS('+items.map(x=>x.word+(x.count?'='+x.count:'')+(x.on?'*':'')).join(',')+')',refusal:(b,tone='warning',o={})=>'REFUSAL('+(tone)+':'+(o.word||o.state||'refused')+'|'+(b?.code||b?.failure_code||'')+'|'+String(b?.detail||b?.reason||'')+'|next:'+(o.next||'')+'|more:'+(o.more||'')+')'+(o.action||''),banner:(a,b,tone,action='')=>a+b+(action||''),dateRange:(a,b)=>(a||'—')+' — '+(b||'—'),dateMove:(a,b)=>(a||'—')+' → '+(b||'—'),badge:(s,l)=>l||s,stateLine:(x,o={})=>o.word||(typeof x==='string'?x:(x?.lifecycle??x?.state??x?.status??'')),statusDot:(s,l)=>l||s,skeleton:(shape='rows')=>'SKELETON('+shape+')',emptyState:(s,a='')=>'EMPTY('+s+')'+(a||''),icon:()=>'',STATUS_ATTR:'',
  PREPARATION_STEPS:[['freeze_sources','Freeze sources',''],['prepare_data','Prepare market data',''],['prepare_features','Prepare Features',''],['publish_inputs','Publish research inputs',''],['verify_inputs','Verify inputs','']],
  typedBtn:(a,b,v,cls,reason)=>`${a}:${b}:${v}:${reason}`,btn:()=>'',link:(l,page='',cls='',extra={})=>'LINK:'+l+'@'+page+(extra.row?'#row='+extra.row:''),table:()=>'',tr:()=>'',picker:(id,choices,o={})=>choices.map(ch=>{const [v,l]=Array.isArray(ch)?ch:[ch.value,ch.title];return '('+(v===o.selected?'*':'')+l+')';}).join(''),objectHead:(name,meta,actions)=>'H1:'+name+'|'+(meta||'')+'|'+(actions||'')};
const renders=[],patches=[],toasts=[],prefs={},discovered=[],discovery={current:null};c.readPreference=k=>prefs[k];c.savePreference=(k,v)=>{prefs[k]=JSON.parse(JSON.stringify(v));};
c.Data.readShared = (...args) => c.Data.read(...args);
vm.createContext(library.into(c,root));vm.runInContext(fs.readFileSync(path.join(root,'status.js'),'utf8'),c);vm.runInContext(percentRule(root),c);vm.runInContext(runShapes(root),c);vm.runInContext(fs.readFileSync(path.join(root,'live-workarea.js'),'utf8')+fs.readFileSync(path.join(root,'live-workspace.js'),'utf8')+';globalThis.w=LiveWorkspace;globalThis.W=LiveWorkArea;',c);
const builderContext={...c,window:{},document:{documentElement:{}}};
library.into(builderContext,root);
vm.runInContext(fs.readFileSync(path.join(root,'components.js'),'utf8'),builderContext);
const publicBuilders=vm.runInContext('({measureStrip,objectRow,stackSlot,fillStackSlot})',builderContext);
c.objectRow=publicBuilders.objectRow;
(async()=>{
  const w=c.w;
  if(process.argv[3]==='--blocked-preparation') {
    const fixture=JSON.parse(fs.readFileSync(0,'utf8'));
    let owner=fixture,decisions=owner.decisions.decisions,decisionReads=0;
    let featureStatus=null;const words=c.t;c.t=(key,vars)=>{if(key==='The Feature owner reports its status only: {status}.')featureStatus=vars.status;return words(key,vars);};
    const banners=[],banner=c.banner;c.banner=(...args)=>{banners.push(args);return banner(...args);};
    c.Data.decisions=()=>decisions;
    c.Data.refreshDecisions=async()=>{decisionReads++;decisions=owner.decisions.decisions;};
    c.LiveActivity={state:()=>owner.activity,retained:()=>[{items:owner.activity.items}]};
    c.Data.read=async p=>{
      reads.push(p);
      const key=p.split('?')[0];
      if(key==='/api/data-update') {const error=Error(owner.dataUpdate.failure_code);error.body=owner.dataUpdate;throw error;}
      return ({'/api/workspace/preparation':owner.preparation,'/api/workspace/data-issues':owner.issues,'/api/tasks/recovery':owner.recovery})[key];
    };
    c.app.page='data'; await w.refresh();
    await Promise.all([w.refresh('welcome','',true),w.refresh('issues','',true)]);
    let page=String(w.page());
    assert.equal(owner.preparation.status,'BLOCKED');
    assert.ok(!['No maintained data yet','Prepare workspace'].some(key=>page.includes(c.t(key))),'the real blocked preparation replaces the empty preparation offer');
    assert.ok(page.includes(c.t('Agent is deciding data issues')),'the exact Pending decision says the agent decides');
    assert.equal(featureStatus,c.codeWords(owner.preparation.progress.status),'the visible Feature status field uses its declared word; exact codes stay in Facts');
    for(const issue of owner.issues.issues) for(const symbol of Object.values(issue.subjects)) assert.ok(page.includes(symbol),'each owner issue remains readable');
    assert.ok(page.includes(c.link(c.t('Data issues'),'issues'))&&!['Your decision is asked','Retry this Task','Continue this preparation Task','Resume this Task'].some(key=>page.includes(c.t(key))),'the owner issues route remains, without asking a person or inventing a retry');
    const before=reads.length; banners.length=0;owner=fixture.decided; w.observe(); await new Promise(r=>setImmediate(r));
    assert.deepEqual(reads.slice(before),['/api/workspace/data-issues'],'the exact accepted case event refreshes its issue once');
    assert.equal(decisionReads,1,'the paused page refreshes canonical pending decisions from the accepted event');
    page=String(w.page());assert.ok(page.includes(c.t('Decided · applied when the update continues')),'the applied owner decision replaces the pending state without Reload');
    assert.ok(!page.includes(c.t('Agent is deciding data issues')),'a resolved case is no longer shown as being decided');
    assert.ok(banners.some(([title])=>title===c.t('Decision recorded'))&&!banners.some(([title])=>title===c.t('A data decision is needed')),'a current banner records the choice instead of requesting it; the exact prior refusal remains in Facts');
    const reason=owner.issues.continuations.find(x=>x.task_id===owner.preparation.task_id).failure_reason.explanation;assert.ok(banners.every(([,body])=>!String(body).includes(reason)),'a current banner never asks for the recorded decision again; the original stop remains in Facts');
    const settled=reads.length;w.observe();await new Promise(r=>setImmediate(r));assert.equal(reads.length,settled,'no settled issue polling');assert.equal(decisionReads,1,'the event is consumed once');
    await w.refresh('welcome','',true);assert.ok(!String(w.page()).includes(c.t('Retry this Task')),'a cold read keeps the exact continuation offer after the decision');
    assert.equal(posts.length,0,'rendering never admits work');
    const vocabulary=library.words(root),head=c.objectHead,receipt=owner.issues.issues[0].resolution.receipt,submission=receipt.actor_submission,proposal=receipt.submission.proposal,rationale=proposal.rationale;let facts=[];c.objectHead=(...args)=>{facts=args[5]?.facts || [];return head(...args);};
    c.hashParams=()=>new URLSearchParams({issue:owner.issues.issues[0].case.case_token});c.app.page='issues';
    vocabulary.I18N.set('en');const authority='Decided under first-use delegation',english=vocabulary.t(authority);c.actorWords=vocabulary.actorWords;
    for(const lang of ['en','zh']) {
      vocabulary.I18N.set(lang);c.t=vocabulary.t;w.page();
      assert.deepEqual(Array.from(facts.find(([key])=>key===c.t('Decision')) || []),[c.t('Decision'),c.t(authority)],'the real recorded case names its exact delegation authority in '+lang);
      if(lang==='zh')assert.notEqual(c.t(authority),english,'the delegation authority has a Chinese key');
      assert.deepEqual(Array.from(facts.find(([key])=>key===c.t('Decided by')) || []),[c.t('Decided by'),c.actorWords(submission.actor_kind)],'the actual agent is named beside its authority');
      proposal.rationale='ordinary decision';w.page();assert.ok(!facts.some(([key])=>key===c.t('Decision')),'an ordinary decision has no delegation authority');assert.deepEqual(Array.from(facts.find(([key])=>key===c.t('Decided by')) || []),[c.t('Decided by'),c.actorWords(submission.actor_kind)],'an ordinary actor keeps its recorded actor word');proposal.rationale=rationale;
    }
    finish();return;
  }
  c.Data.decisions=()=>null;
  // V661 recovery: the first Storage read has no retained inventory. Backups are an
  // independent owner, so its exact restore command stays reachable after this refusal.
  {
    const read=c.Data.read,table=c.table,tr=c.tr,panel=c.panel,oldPage=c.app.page,oldWorkspace=c.app.workspace;
    const root='Z:/Test Fixture/AlphaLattice/backups/cold-storage',generation='a'.repeat(64),task='66100000-0000-4000-8000-000000000001';
    const inventoryMessage='task_control.database_authority_unreadable: Part of the Task Control authority could not be read.';
    const backupMessage='research_workspace.manifest_unreadable: the backup owner could not read its manifest';
    const oldBackup=bodies['/api/workspace/backup'];let backupRefused=false;
    bodies['/api/workspace/backup']={status:'READ',backup_root:root,last_automatic_attempt:null,generations:[{generation_hash:generation,created_at:'2026-09-29T20:00:00+00:00',reason:'REQUEST',files:12,tables:9,listed:3,absent:[]}]};
    c.table=(_headers,rows)=>'<table>'+rows.join('')+'</table>';c.tr=cells=>'<tr>'+cells.map(x=>String(x ?? '')).join(' | ')+'</tr>';
    c.panel=(a,b,content,action='')=>a+b+content+action;c.app.page='storage';c.app.workspace='cold-storage';
    c.Data.read=async p=>{
      if(p==='/api/workspace/storage'){reads.push(p);const error=Error(inventoryMessage);error.body={status:'REFUSED',failure_code:'task_control.database_authority_unreadable',refusals:[{task_id:task,status:'REFUSED',failure_code:'task_control.database_authority_unreadable'}],detail:inventoryMessage,next_requests:{backups:{operation:'WORKSPACE_BACKUPS'}}};throw error;}
      if(p==='/api/workspace/backup'&&backupRefused){reads.push(p);throw Error(backupMessage);}
      return read(p);
    };
    try {
      const sent=posts.length,before=reads.length;await w.refresh('storage');await new Promise(r=>setImmediate(r));
      assert.deepEqual(reads.slice(before),['/api/workspace/storage','/api/workspace/backup'],'cold inventory refusal still asks the independent Backups owner once');
      assert.ok(!w.scene('storage').body,'refused cold inventory invents no readable inventory');
      let page=String(w.page()).replace(/\s+/g,' ');
      assert.ok(page.includes(inventoryMessage),'the exact inventory refusal remains beside Backups');
      assert.ok(page.includes(root)&&page.includes(generation),'the readable Backups owner retains its exact root and generation');
      assert.ok(!/Managed on disk|What is kept where|Retained input versions|Cleanup preview|No eligible files to clean/.test(page),'no inventory figures, retained-version list or cleanup result is fabricated');
      assert.ok(page.includes('Back up now:workspace-preview:backup:')&&!page.includes('Back up now:workspace-preview:backup:The backups could not be read'),'the readable owner still offers its backup action');
      const restoreAt=page.indexOf('Restore command'),start=page.indexOf('<template>',restoreAt),end=page.indexOf('</template>',start);
      assert.ok(restoreAt>=0&&start>=0&&end>start,'the recovery route remains the client restore command');
      assert.equal(JSON.parse('"'+page.slice(start+'<template>'.length,end)+'"'),'alphalattice backup restore --dir <new directory> --workspace-id cold-storage --root "'+root+'" --generation '+generation,'restore still names the exact independently read root and generation');
      backupRefused=true;const failedRead=reads.length;await w.refresh('storage');await new Promise(r=>setImmediate(r));page=String(w.page()).replace(/\s+/g,' ');
      assert.deepEqual(reads.slice(failedRead),['/api/workspace/storage','/api/workspace/backup'],'a failed Backups owner remains a separate read');
      assert.ok(page.includes(inventoryMessage)&&page.includes('Backups not read')&&page.includes(backupMessage),'each owner refusal stays local and exact');
      assert.ok(page.includes(root)&&page.includes(generation)&&page.includes('Back up now:workspace-preview:backup:The backups could not be read'),'earlier Backups facts remain readable while its mutating action is held by the real read failure');
      assert.equal(posts.length,sent,'all recovery reads and renders send no mutation');
    } finally {
      c.Data.read=read;c.table=table;c.tr=tr;c.panel=panel;c.app.page=oldPage;c.app.workspace=oldWorkspace;
      if(oldBackup===undefined)delete bodies['/api/workspace/backup'];else bodies['/api/workspace/backup']=oldBackup;
    }
  }
  await w.refresh();assert.equal(posts.length,0);assert.ok(w.page().includes('2026-08-03'));
  // The Data figures carry their own inert Facts resource, without a separate page-stack child.
  {
    const update=bodies['/api/data-update'],issues=bodies['/api/workspace/data-issues'],
      storage=bodies['/api/workspace/storage'],figureTile=c.figureTile,measureStrip=c.measureStrip;
    const ownerStrip=publicBuilders.measureStrip,tiles=[],strips=[];
    // A public readback carrier keeps its body and identity without another painted surface.
    const body='<section class="panel">Recorded section</section>';
    for(const kind of ['box','section','field']) {
      const carrier=String(publicBuilders.stackSlot('recordedSlot',body,kind));
      assert.ok(carrier.includes('id="recordedSlot"')&&carrier.includes(body));
      assert.ok(carrier.includes('data-stack-box="'+kind+'"'));
      assert.ok(!carrier.includes('data-box='),'the carrier supplies no material');
    }
    const empty=String(publicBuilders.stackSlot('emptySlot',''));
    assert.ok(empty.includes('id="emptySlot"')&&!empty.includes('data-stack-box'),
      'an absent readback has no stack marker or invented surface');
    const slot={innerHTML:'',dataset:{},removeAttribute(name){
      assert.equal(name,'data-stack-box');delete this.dataset.stackBox;
    }};
    publicBuilders.fillStackSlot(slot,body);
    assert.equal(slot.innerHTML,body);assert.equal(slot.dataset.stackBox,'section');
    publicBuilders.fillStackSlot(slot,'');
    assert.equal(slot.innerHTML,'');assert.ok(!Object.hasOwn(slot.dataset,'stackBox'),
      'a partial read that removes the section removes its spacing relation too');
    publicBuilders.fillStackSlot(slot,body,'box');
    assert.equal(slot.innerHTML,body);assert.equal(slot.dataset.stackBox,'box');
    const membership={bootstrap:{cohort_size:3,t0_session:'2026-08-03',history_start:'2025-08-04',
      initialization_assumption:'INITIAL_COHORT_BACKFILL_NOT_POINT_IN_TIME'},journal_sequence:2,latest_effective_session:'2026-08-03'};
    bodies['/api/data-update']={...update,membership};
    bodies['/api/workspace/data-issues']={...issues,issues:issues.issues.map(issue=>({
      ...issue,status:'AWAITING_CHOICE',
      case:{...issue.case,failure_code:'data.truth_review_required',evidence:[]},
    }))};
    bodies['/api/workspace/storage']={...storage,budget:{...storage.budget,
      active_listing_count:3,research_session_count:252}};
    c.figureTile=(label,value,to,note)=>{tiles.push({label,to});return figureTile(label,value,to,note);};
    c.measureStrip=(items,label,cls)=>{
      const markup=String(ownerStrip(items,label,cls));strips.push({items:String(items),markup,label,cls});return markup;
    };
    try {
      const sent=posts.length;await w.refresh('data');
      await w.refresh('issues','',true);await w.refresh('storage','',true);
      const page=String(w.page());
      const figure=strips.filter(strip=>strip.label==='Data figures'&&strip.cls==='data-figures');
      assert.equal(figure.length,1,'the public Data producer composes one figure container');
      const strip=figure[0],resource='<template data-facts-id="facts-membership"';
      assert.ok(strip.markup.startsWith('<div class="es-measures-box">')&&strip.markup.endsWith('</div>'),
        'the existing measure-strip builder supplies the figure container');
      assert.ok(strip.items.includes(resource)&&strip.markup.includes(resource),
        'membership Facts are inside the figure container that opens them');
      assert.equal(page.split(resource).length-1,1,'the public page publishes exactly one membership Facts resource');
      assert.ok(page.includes(strip.markup)&&!page.replace(strip.markup,'').includes(resource),
        'there is no orphan membership template outside the figures');
      assert.ok(strip.items.includes('data-facts-title="Sources & membership"'),
        'the Facts title is retained');
      const payloadAt=strip.items.indexOf('<template>',strip.items.indexOf(resource)),
        payloadEnd=strip.items.indexOf('</template>',payloadAt);
      assert.ok(payloadAt>=0&&payloadEnd>payloadAt,'the Facts resource carries its own JSON payload');
      assert.deepEqual(JSON.parse(strip.items.slice(payloadAt+'<template>'.length,payloadEnd)),
        {membership},'the complete owner membership payload is retained');
      for(const label of ['Listings','Market sessions']) {
        const offered=tiles.filter(tile=>tile.label===label);
        assert.equal(offered.length,1,'the figure keeps its membership Facts action: '+label);
        assert.deepEqual(JSON.parse(JSON.stringify(offered[0].to)),{action:'facts-open',value:'facts-membership'},
          'the figure still opens the same membership Facts resource');
      }
      assert.equal(posts.length,sent,'placing the inert Facts resource sends no mutation');
    } finally {
      bodies['/api/data-update']=update;bodies['/api/workspace/data-issues']=issues;
      bodies['/api/workspace/storage']=storage;c.figureTile=figureTile;c.measureStrip=measureStrip;
      await w.refresh('data');
    }
  }
  await w.preview('update');assert.equal(posts.length,1);assert.ok(modal.content.includes('additions'));
  w.dismissConfirmation();await w.commit();assert.equal(posts.length,1);
  let release;delayed={path:'/api/data-update/plan',wait:new Promise(r=>{release=r;})};modal=null;
  const old=w.preview('update');await Promise.resolve();w.dismissConfirmation();release();await old;delayed=null;
  assert.equal(modal,null,'late preview must not revive dismissed confirmation');
  await w.preview('update');const before=posts.length,readsBeforeRun=reads.length;await Promise.all([w.commit(),w.commit()]);
  assert.deepEqual(posts.slice(before).map(v=>v[0]),['/api/data-update/confirm','/api/data-update/run']);
  // the admitted update is this page's scene: the page reads the owner again for it; nothing opens the Task Center
  assert.deepEqual(tasks,[],'the update scene is the data page, not the Task Center');assert.ok(reads.slice(readsBeforeRun).includes('/api/data-update'),'the page re-reads the update owner after admission');
  failed='/api/data-update/confirm';await w.preview('update');const blocked=posts.length;await w.commit();failed=null;
  assert.deepEqual(posts.slice(blocked).map(v=>v[0]),['/api/data-update/confirm'],'failed approval must not run data work');
  // a refused preview is answered (the user, 2026-09-24: 点prepare workspace, 什么也没发生): a toast; the page's
  // notice says why. (No page render here: a scene page read before router.js loads leaves vm's lookup of
  // hashParams cached as absent, and the later preparation scene would lose its Task.)
  failed='/api/data-update/plan';await w.preview('update');failed=null;
  assert.equal(toasts.at(-1),'Action refused','the refused press is answered');
  c.app.page='issues';await w.refresh();w.changed('case','quarantine');await w.preview('issue','case');
  const payload={data_issue_case_token:'case',data_issue_evidence_hash:'evidence',data_issue_option_id:'quarantine',data_issue_option_hash:'option'};
  assert.deepEqual(posts.at(-1),['/api/workspace/data-issues/preview',payload]);
  w.changed('case','');const cancelled=posts.length;await w.commit();assert.equal(posts.length,cancelled);
  w.changed('case','quarantine');await w.preview('issue','case');await w.commit();assert.deepEqual(posts.at(-1),['/api/workspace/data-issues/confirm',payload]);
  // the owner names each request; one naming an operation off the client's fixed routes is refused before anything is sent
  const issues=bodies['/api/workspace/data-issues'],offered=issues.next_requests;
  issues.next_requests={'preview:case:quarantine':{...offered['preview:case:quarantine'],operation:'DATA_UPDATE_RUN'}};await w.refresh();
  const foreign=posts.length;w.changed('case','quarantine');await w.preview('issue','case');assert.equal(posts.length,foreign,'a request off the fixed routes is refused locally');
  issues.next_requests=offered;await w.refresh();
  bodies['/api/workspace/data-issues'].continuations=[{operation:'DATA_UPDATE_RUN',endpoint:'https://foreign.example/run',payload:{}}];
  const unsafe=posts.length;await w.preview('continue','0');assert.equal(posts.length,unsafe,'foreign continuation is refused locally');
  c.app.page='inputs';await w.refresh();await w.preview('capture');const afterCapture=posts.length;
  await w.commit();assert.equal(posts.length,afterCapture,'exact input reuse needs no confirm/write');
  await w.selectInput(JSON.stringify(['family','binding']));assert.deepEqual(drafts,[JSON.stringify(['family','binding'])]);
  c.app.page='storage';await w.refresh();await w.preview('cleanup');c.app.page='data';const moved=posts.length;await w.commit();assert.equal(posts.length,moved,'a different page cannot confirm stale cleanup');
  c.app.page='storage';bodies['/api/workspace/storage/plan'].targets={};await w.preview('cleanup');
  assert.ok(w.page().includes('No eligible files to clean. Nothing was removed.'));const noTargets=posts.length;await w.commit();assert.equal(posts.length,noTargets);
  assert.ok(!posts.some(([p])=>p.includes('/experiments/run')));
  // Quick Open uses navigate()/pushState rather than a hashchange event.
  c.Data.live=true;c.LiveWorkspace=w;c.Navigation={close(){}};c.hideToast=()=>{};c.scrollTo=()=>{};
  c.THEMES=[];c.location={hash:'',href:''};c.history={pushState:(_a,_b,hash)=>{if(typeof hash==='string')c.location.hash=hash;},replaceState:(_a,_b,hash)=>{if(typeof hash==='string')c.location.hash=hash;}};
  vm.runInContext(fs.readFileSync(path.join(root,'router.js'),'utf8'),c);
  c.render=()=>{};c.readRoute=()=>{};c.routeUrl=page=>'#page='+page;
  c.app.page='data';await w.preview('update');const navigated=posts.length;
  c.navigate('inputs');c.navigate('data');await w.commit();
  assert.equal(posts.length,navigated,'programmatic navigation must also invalidate confirmation');

  // ---- the preparation scene: readback + recovery view -> one page, read-only, patched in place ----
  c.patchMain=()=>patches.push(c.app.page);c.render=()=>renders.push(c.app.page); // the router probe above installed the real ones
  const TASK='1d8ea0bf-f50e-4e65-974c-9552b3851430';
  const view=(lifecycle,stage,extra={})=>({task_id:TASK,task_kind:'workspace_preparation',lifecycle,operation_running:false,cancellation:lifecycle==='CANCELLED'?'ACKNOWLEDGED':'NOT_REQUESTED',stop:null,worker_failure:null,
    liveness:{status:['RUNNING','CANCEL_REQUESTED'].includes(lifecycle)?'OBSERVED':'NOT_APPLICABLE',telemetry:'OPERATIONAL',age_seconds:0.7,signal_sequence:3},verified_stage_count:['freeze_sources','prepare_data','prepare_features','publish_inputs','verify_inputs'].indexOf(stage),total_stage_count:5,
    stages:['freeze_sources','prepare_data','prepare_features','publish_inputs','verify_inputs'].map((id,i,all)=>({stage_id:id,lifecycle:i<all.indexOf(stage)?'VERIFIED':id===stage?(lifecycle==='RUNNING'?'IN_PROGRESS':lifecycle==='CANCELLED'?'CANCELLED':lifecycle==='BLOCKED'?'BLOCKED':'PENDING'):'PENDING',evidence:i<all.indexOf(stage)?['ref']:[],evidence_count:i<all.indexOf(stage)?1:0})),
    artifact_refs:[],actions:[{action:'CANCEL',operation:'CANCEL',available:lifecycle==='RUNNING',reason:'r'},{action:'RECOVER',operation:'RECOVER',available:lifecycle==='RECOVERY_REQUIRED',reason:'not interrupted'},{action:'REPLAN',operation:'WORKSPACE_PREPARE_PLAN',available:true,reason:'r'}],
    status:{task_id:TASK,lifecycle,current_stage:stage,verified_stage_count:1,total_stage_count:5,running_since:'2026-09-15T21:07:24+00:00',last_activity_at:'2026-09-15T21:10:21+00:00'},task_record_hash:'a'.repeat(64),observed_at:'2026-09-15T21:17:02+00:00',...extra});
  const readback=(status,extra={})=>({status,source_mode:'ACQUIRE_DECLARED_SOURCES_AFTER_CONFIRMATION',next_action:status==='INITIALIZATION_REQUIRED'?'WORKSPACE_PREPARE_PLAN':null,execution_binding_changed:false,task_id:null,plan_hash:null,failure_code:null,inputs:[],profile:'us-current-index-research',sources:['yfinance','SP500: https://example.test/sp500'],limits:['CURRENT_UNIVERSE_RESEARCH_ONLY','NO_DATA_API_KEY','NO_FOUNDATION_OR_STRATEGY_ACTIVATION'],progress:null,work_progress:null,...extra});
  const page=async()=>{await w.refresh('welcome');return w.page();};
  c.app.page='welcome';
  // U77 (V481): a workspace prepared before the renames is refused by name; the first use reads the door's words, the
  // spelling it holds carried, its way on a new workspace -- never a restore of its manifest
  const door="This workspace was prepared before the 2026-10-02 renames, which this release does not read: it holds `dynamic_panel_v1`, a spelling the renames retired. Nothing in it is changed; start a new workspace, as the guide says, and prepare its data and research there.";
  const firstRead=c.Data.read;c.Data.read=async p=>{reads.push(p);throw Error('research_workspace.prepared_before_renames:dynamic_panel_v1: '+door);};
  await w.refresh('welcome');let renamed=w.page();
  assert.ok(w.prepareRefused()&&renamed.includes('This release cannot open this workspace')&&renamed.includes('it holds `dynamic_panel_v1`')&&!renamed.includes('restore its manifest')&&!renamed.includes('newer one'),'renamed: the door words, a new workspace: '+renamed.slice(renamed.indexOf('This release'),renamed.indexOf('This release')+400));
  c.Data.read=firstRead;
  // unprepared: the scope, the sources, no Task; reading posts nothing
  bodies['/api/workspace/preparation']=readback('INITIALIZATION_REQUIRED');
  const unpreparedPosts=posts.length,unpreparedReads=reads.length;
  let markup=await page();
  assert.ok(markup.includes('Before preparation') && markup.includes('No data API key is requested or stored.') && markup.includes('https://example.test/sp500'),'the unprepared scene is the Home\'s first-use section: the facts, the sources and the missing key');
  assert.ok(markup.includes('run-steps') && !markup.includes('data-state='),'the plain stage roster (the one step list) before a Task exists');
  assert.equal(posts.length,unpreparedPosts);assert.deepEqual(reads.slice(unpreparedReads),['/api/workspace/preparation'],'one readback, no recovery view without a Task');
  // An externally completed preparation is discovered from its projection, without reload.
  {
    const originalTasks=c.Data.tasks,originalHash=c.location.hash,empty=bodies['/api/workspace/preparation'],coldReads=reads.length,coldPosts=posts.length;
    c.app.page='overview';c.location.hash='#page=overview&task='+TASK;w.observe();await new Promise(r=>setImmediate(r));
    c.Data.tasks=()=>[{task_id:'research',task_kind:'research_experiment',lifecycle:'SUCCEEDED'}];
    w.observe();await new Promise(r=>setImmediate(r));
    assert.equal(reads.length,coldReads,'no preparation read without a preparation projection');
    const completed=view('SUCCEEDED','verify_inputs',{verified_stage_count:5});
    c.Data.tasks=()=>[completed];
    bodies['/api/workspace/preparation']=readback('SUCCEEDED',{task_id:TASK,inputs:[{input_id:'family',binding_hash:'binding'}]});
    bodies['/api/tasks/recovery']=completed;
    c.app.page='history';w.observe();await new Promise(r=>setImmediate(r));
    assert.equal(reads.length,coldReads,'the initial Home is not read from another page');
    c.app.page='overview';w.observe();w.observe();await new Promise(r=>setImmediate(r));
    assert.deepEqual(reads.slice(coldReads),['/api/workspace/preparation','/api/tasks/recovery?task_id='+TASK],'one current owner read discovers the external Task and reads its exact recovery view');
    assert.equal(discovery.current.task_id,TASK);assert.equal(discovery.current.inputs.length,1);
    const address=new URLSearchParams(c.location.hash.slice(1));
    assert.equal(address.get('page'),'overview');assert.equal(address.get('task'),TASK);assert.equal(address.get('preparation'),TASK,'the owner-discovered Task is pinned without changing Home or its inspector');
    assert.equal(String(w.firstUse()),'','the verified input removes the initial preparation scene from Home');
    w.observe();await new Promise(r=>setImmediate(r));
    assert.equal(reads.length,coldReads+2,'a verified completion is not polled again');
    assert.equal(posts.length,coldPosts,'discovery posts no PLAN, confirmation or other mutation');
    c.Data.tasks=originalTasks;c.location.hash=originalHash;c.app.page='welcome';bodies['/api/workspace/preparation']=empty;
    await w.refresh('welcome');
  }
  // preview: the owner's plan is offered with its scope; a dismissed preview stays and is re-offered without a second PLAN
  bodies['/api/workspace/preparation/plan']={status:'CONFIRMATION_REQUIRED',confirmation_available:true,plan_hash:'plan-1',target_session:'2026-07-31',initial_history_years:10,source_mode:'ACQUIRE_APPROVED_SOURCES',candidate_count:null,candidate_count_basis:'KNOWN_AFTER_SOURCE_CAPTURE',universe:'current S&P 500 union NASDAQ-100 union DJIA',quality:{maximum_missing_ratio:0.02,maximum_consecutive_missing_sessions:20},sources:['yfinance'],next_action:'WORKSPACE_PREPARE_CONFIRM',limits:[]};
  modal=null;await w.preview('prepare');
  assert.deepEqual(posts.at(-1),['/api/workspace/preparation/plan',{}]);
  assert.ok(modal.content.includes('2026-07-31') && modal.content.includes('New work') && modal.content.includes('not known at this preview') && modal.content.includes('plan-1'),'the confirmation names the scope, reuse versus new work and the storage estimate honestly');
  w.dismissConfirmation();markup=w.page();
  assert.ok(markup.includes('Preparation scope · preview') && markup.includes('Confirm preparation:workspace-preview:reoffer'),'the retained preview stays on the page with its own confirm');
  modal=null;const plans=posts.length;await w.preview('reoffer');assert.equal(posts.length,plans,'re-offering the retained preview plans nothing again');assert.ok(modal.content.includes('plan-1'));
  // a preview refused its sources (CLI-15): the workspace's network control in Settings, then preview again, no
  // restart; only the operator's offline switch names a start command, the root entry without the switch
  const netRefused=(by,allowed=false)=>({...bodies['/api/workspace/preparation/plan'],confirmation_available:false,source_access_failure:'workspace_preparation.source_access_not_admitted',
    source_access:{status:'NETWORK_CONTROL_REQUIRED',network_access:{status:'NETWORK_ACCESS',network_allowed:allowed,decided_by:by,detail:'owner words'},authorization_required:true,restart_only_when_idle:true,workspace_path:'C:/qa/ws',verified_source_checkpoint_retained:false,network_requests:'UNKNOWN_UNTIL_EXECUTION'}});
  const confirmable=bodies['/api/workspace/preparation/plan'];
  bodies['/api/workspace/preparation/plan']=netRefused('DEFAULT');modal=null;await w.preview('prepare');markup=w.page();
  assert.equal(modal,null,'a refused preview offers no confirmation');
  assert.ok(markup.includes('Its sources need the network.') && markup.includes('data-tip="DEFAULT"') && markup.includes('Off: the workspace sets no network control, so the network stays off.'),'the refusal says what decides the network, its code on hover');
  assert.ok(markup.includes('LINK:Allow network access in Settings@settings#row=networkAccess') && markup.includes('then preview again; the Host keeps running'),'its way on is the workspace control in Settings, then a new preview');
  assert.ok(!markup.includes('serve --no-browser') && !markup.includes('Start command'),'no restart while the workspace control can allow it');
  assert.ok(markup.includes('Preview again:workspace-preview:prepare') && !markup.includes('Confirm preparation:'),'preview again stays; confirm is not offered');
  bodies['/api/workspace/preparation/plan']=netRefused('WORKSPACE_CONTROL');await w.preview('prepare');markup=w.page();
  assert.ok(markup.includes("Off by this workspace's network control.") &&markup.includes('LINK:Allow network access in Settings'),'a control set off points to the same control');
  bodies['/api/workspace/preparation/plan']=netRefused('OPERATOR_OFFLINE_SWITCH');await w.preview('prepare');markup=w.page();
  assert.ok(markup.includes('Start command for this workspace') && markup.includes('python scripts/run_alphalattice.py --workspace "C:/qa/ws" serve --no-browser'),'the offline switch alone names the start command, the root entry');
  assert.ok(markup.includes('start it again without <code class="literal">ALPHALATTICE_NETWORK_DISABLED=1</code>') && !markup.includes('LINK:Allow network access'),'it is started again without the switch, a literal to type; the held control is not offered');
  for(const old of ['run_local_research','NETWORK_DISABLED=0','required_policy'])assert.ok(!markup.includes(old),'the older launcher and its switch are never named: '+old);
  // V620 (U93): a hold the person cannot lift here reads in the network owner's words (who lifts it, how); the page
  // recreates no permission advice for it
  assert.ok(markup.includes('<span data-tip="OPERATOR_OFFLINE_SWITCH">owner words</span>'),'the operator\'s switch in its owner\'s words');
  bodies['/api/workspace/preparation/plan']=netRefused('RUN_HELD_OFFLINE');await w.preview('prepare');markup=w.page();
  assert.ok(markup.includes('<span data-tip="RUN_HELD_OFFLINE">owner words</span>'),'a run\'s offline hold in its owner\'s words');
  const switchSaid=String(w.networkWords({network_allowed:false,decided_by:'OPERATOR_OFFLINE_SWITCH',detail:'ALPHALATTICE_NETWORK_DISABLED=1 keeps this process offline.'}));
  assert.ok(switchSaid.includes('<code class="literal">ALPHALATTICE_NETWORK_DISABLED=1</code> keeps this process offline.'),'the switch the owner names is the literal a person types: '+switchSaid);
  // U70 (V452): an opening the first-use goal made holds until its own time, named in the control (`set_by`); once it
  // lapsed the control says the first use ended; the operator's switch still decides first
  const LATER=new Date(Date.now()+3600000).toISOString(),EARLIER=new Date(Date.now()-3600000).toISOString();
  const delegated=(allowed,until,by='WORKSPACE_CONTROL')=>String(w.networkWords({network_allowed:allowed,decided_by:by,set_by:{delegation:'first-use-goal:g1',until}}));
  assert.ok(delegated(true,LATER).startsWith('<span data-tip="first-use-goal:g1">Opened for you by your first-use goal, until ')&&!delegated(true,LATER).includes('{t}'),'opened for the person until its end: '+delegated(true,LATER));
  assert.equal(delegated(false,EARLIER),'Closed when your first use ended','past its end, the control says the first use ended');
  assert.ok(!delegated(false,LATER).includes('first use'),'closed before its end: its control words');
  assert.ok(!delegated(false,EARLIER,'OPERATOR_OFFLINE_SWITCH').includes('first use'),'the operator switch decides first');
  bodies['/api/workspace/preparation/plan']=confirmable;modal=null;await w.preview('prepare');assert.ok(modal?.content.includes('plan-1'),'an allowed preview offers its confirmation again');
  const explicitPlans=posts.filter(([p])=>p==='/api/workspace/preparation/plan').length; // every PLAN so far was a press
  // confirm: one POST; the page becomes the Task's scene (readback + recovery view), not the drawer
  bodies['/api/workspace/preparation/confirm']={status:'ADMITTED',task_id:TASK,lifecycle:'QUEUED'};
  bodies['/api/workspace/preparation']=readback('RUNNING',{task_id:TASK,plan_hash:'plan-1',progress:{phase:'prepare_data',candidates:120,raw_ready:25,quality_eligible:25,failed:0,retry_after_at:null}});
  bodies['/api/tasks/recovery']=view('RUNNING','prepare_data');
  const drawerOpens=tasks.length;await w.commit();
  assert.deepEqual(posts.at(-1),['/api/workspace/preparation/confirm',{preparation_plan_hash:'plan-1'}]);
  assert.equal(tasks.length,drawerOpens,'the admitted preparation stays on its scene; the Task Center is not opened over it');
  markup=w.page();
  assert.ok(markup.includes('25') && markup.includes('candidates with raw bars') && markup.includes('25 quality-eligible · 0 failed'),'the data stage shows its own counts with their denominator');
  assert.ok(markup.includes('data-action="workspace-stage" data-value="freeze_sources" data-state="VERIFIED"') && markup.includes('data-value="prepare_data" data-state="IN_PROGRESS"') && markup.includes('aria-current="step"'),'the stage rail carries Task Control\'s marks and the current stage');
  assert.ok(!markup.includes('run-steps'),'the plain roster is not drawn beside the rail');
  assert.ok(markup.includes('Request cancel:task-cancel:'+TASK+':'),'cancel is offered while the Task moves');
  assert.ok(!markup.includes('Preparation scope · preview'),'an admitted preview is not shown as pending');
  // the finer count of the Feature stage is shown only when bound to this Task and stage
  bodies['/api/workspace/preparation']=readback('RUNNING',{task_id:TASK,plan_hash:'plan-1',progress:{phase:'prepare_features',status:'PREPARING_FEATURE_CLOSURE'},work_progress:{availability:'BOUND',stage:'prepare_features',execution_id:'e1',age_seconds:1.3,stage_id:'base_feature_materialization',status:'RUNNING',completed_units:63,total_units:120,unit_name:'listings',current_item:'6f5cd63e-8230',counters:{},failure_code:null}});
  bodies['/api/tasks/recovery']=view('RUNNING','prepare_features');
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('63') && markup.includes('/ 120 listings') && markup.includes('Feature materialization') && markup.includes('bound to this Task and stage'),'a bound Feature step shows its count and unit');
  bodies['/api/workspace/preparation'].work_progress={...bodies['/api/workspace/preparation'].work_progress,status:'SUCCEEDED',completed_units:120};
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('Feature materialization finished · 120 / 120 listings · the next step has not reported yet') && !markup.includes('aria-valuenow="120"'),'a finished step is said to be finished, never shown as the stage\'s progress');
  bodies['/api/workspace/preparation'].work_progress={...bodies['/api/workspace/preparation'].work_progress,availability:'NOT_CURRENT'};
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('No finer count is reported for this step') && !markup.includes('/ 120 listings'),'an unbound observation is not shown as the current step');
  // the scene's re-read on the activity cadence: reads only, patched in place; nothing elsewhere or once the Task has stopped
  const observed=reads.length,paintsBefore=patches.length,rendersBefore=renders.length;
  w.observe();await new Promise(r=>setTimeout(r,20));
  assert.deepEqual(reads.slice(observed),['/api/workspace/preparation?task_id='+TASK,'/api/tasks/recovery?task_id='+TASK],'one readback (of the pinned Task) and one recovery view per observation');
  assert.equal(patches.length,paintsBefore+1,'the observed scene is patched in place');assert.equal(renders.length,rendersBefore,'not fully rendered');
  assert.equal(posts.filter(([p])=>p==='/api/workspace/preparation/plan').length,explicitPlans,'no PLAN in the polling path');
  c.app.page='history';const away=reads.length;w.observe();await new Promise(r=>setTimeout(r,20));assert.equal(reads.length,away,'nothing is read for the scene from another page');c.app.page='welcome';
  // transport loss after a valid RUNNING answer: the last observation stays, named as such; no
  // fresh-progress signal, no heartbeat claimed as current, the outage is the page's own fact;
  // a successful read recovers, and no Task lifecycle is changed by the browser's clock
  bodies['/api/workspace/preparation']=readback('RUNNING',{task_id:TASK,plan_hash:'plan-1',progress:{phase:'prepare_data',candidates:120,raw_ready:50,quality_eligible:50,failed:0,retry_after_at:null}});
  bodies['/api/tasks/recovery']=view('RUNNING','prepare_data');
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('data-live="true"') && markup.includes('data-moving="true"') && !markup.includes('data-disconnected'),'a fresh RUNNING read signals live work');
  const realRead=c.Data.read;c.Data.read=async p=>{reads.push(p);throw Error('TypeError: Failed to fetch');};
  w.observe();await new Promise(r=>setTimeout(r,20));markup=w.page();
  assert.ok(markup.includes('data-disconnected="true"') && markup.includes('Failed to fetch'),'the outage is shown as the page\'s own fact');
  assert.ok(!markup.includes('data-live="true"') && !markup.includes('data-moving="true"'),'nothing is signalled as fresh progress');
  assert.ok(markup.includes('Last observation at') && markup.includes('not re-read since') && markup.includes('data-retained="true"'),'the heartbeat sentence is named a retained observation');
  assert.ok(markup.includes('at the last successful read · not re-read since') && markup.includes('1 failed read'),'the read facts are truthful');
  assert.ok(markup.includes('50') && markup.includes('candidates with raw bars'),'the last owner observation is kept, not blanked');
  assert.ok(!markup.includes('Workspace action needs attention'),'a lost read is not a workspace action failure');
  w.observe();await new Promise(r=>setTimeout(r,20));assert.ok(w.page().includes('2 failed reads'),'reading continues on the cadence while disconnected');
  c.Data.read=realRead;w.observe();await new Promise(r=>setTimeout(r,20));markup=w.page();
  assert.ok(markup.includes('data-live="true"') && !markup.includes('data-disconnected'),'a successful read recovers the live scene');
  assert.equal(posts.filter(([p])=>p==='/api/cancel'||p==='/api/recover').length,0,'no Task lifecycle is touched by the reader');
  // the person's own Refresh (the actual action, not the cadence) failing after a valid answer
  // has the same meaning: `quiet` chose how the page is drawn, not whether the facts are current
  c.Data.read=async p=>{reads.push(p);throw Error('TypeError: Failed to fetch');};
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('data-disconnected="true"') && !markup.includes('data-live="true"') && !markup.includes('data-moving="true"'),'a failed manual refresh marks the retained scene stale');
  assert.ok(markup.includes('50') && markup.includes('candidates with raw bars') && markup.includes('1 failed read') && markup.includes('data-retained="true"'),'the counts are kept and the failure is counted');
  assert.ok(!markup.includes('Workspace action needs attention'),'a lost manual read is not a workspace action failure');
  c.Data.read=realRead;await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('data-live="true"') && !markup.includes('data-disconnected'),'a successful manual read recovers the live scene');
  assert.equal(posts.filter(([p])=>p==='/api/cancel'||p==='/api/recover').length,0,'no Task lifecycle is touched by a manual read either');
  // interrupted: the owner's resume is the one offered action; stopped by its owner: retry under the same plan or the data issues
  bodies['/api/workspace/preparation']=readback('RECOVERY_REQUIRED',{task_id:TASK,plan_hash:'plan-1',failure_code:'TASK_EXECUTION_INTERRUPTED'});
  bodies['/api/tasks/recovery']=view('RECOVERY_REQUIRED','publish_inputs',{stop:{code:'TASK_EXECUTION_INTERRUPTED',stage_id:'publish_inputs',detail:'The worker stopped before the current stage was verified.',recoverable:true}});
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('Resume this Task:task-recovery:'+TASK+':') && !markup.includes('Request cancel:'),'an interrupted Task offers the owner\'s resume and nothing to cancel');
  // a resume confirmed in the Task Center moves the Task before this scene has read it again: the
  // projection the activity feed keeps says so, and the scene re-reads once on the cadence
  const idle=reads.length;w.observe();await new Promise(r=>setTimeout(r,20));assert.equal(reads.length,idle,'a stopped Task is not re-read while nothing says it moved');
  c.Data.tasks=()=>[{task_id:TASK,lifecycle:'RUNNING',verified_stage_count:3,total_stage_count:5}];
  w.observe();await new Promise(r=>setTimeout(r,20));assert.equal(reads.length,idle+2,'a moving projection re-reads the scene');c.Data.tasks=()=>[];
  // a resumed Task that already finished between two reads: the projection disagrees with the
  // view without moving, and the scene still re-reads (it was left on "interrupted" once)
  bodies['/api/workspace/preparation']=readback('SUCCEEDED',{task_id:TASK,plan_hash:'plan-1',inputs:[{input_id:'family',binding_hash:'binding'}]});
  bodies['/api/tasks/recovery']=view('SUCCEEDED','verify_inputs',{verified_stage_count:5,artifact_refs:['playpen://x']});
  c.Data.tasks=()=>[{task_id:TASK,lifecycle:'SUCCEEDED',verified_stage_count:5,total_stage_count:5}];
  const settled=reads.length;w.observe();await new Promise(r=>setTimeout(r,20));
  assert.equal(reads.length,settled+2,'a projection that disagrees with the view re-reads the scene');
  assert.ok(!w.page().includes('prep-area') && discovery.current.inputs.length===1,'and the completion is reached: the Home is the Home');c.Data.tasks=()=>[];
  // ---- the work area: real listing units bound to the execution, follow / hold, inspection
  // held across re-reads, one collapse, the choices kept per exact Task; no post from any of it
  const unit=(symbol,state,at,extra={})=>({listing_id:symbol.toLowerCase()+'-0000-listing',symbol,state,observed_at:at,origin:state==='RAW_READY'?'ACQUIRED':state==='RAW_FAILED'?null:'LOCAL',raw_through:state==='RAW_READY'?'2026-09-15':null,failure_code:state==='RAW_FAILED'?'data.provider_fetch_failed':null,reasons:[],tail_acquired:false,...extra});
  const activity=(rows,extra={})=>({availability:'BOUND',stage:'prepare_data',execution_id:'e1',sequence:1,observed:rows.length,retained:rows.length,dropped:0,written_at:'2026-09-15T21:07:30+00:00',age_seconds:0.4,rows,...extra});
  const firstRows=[unit('AAA','RAW_READY','2026-09-15T21:07:29+00:00'),unit('AAA','QUALITY_ELIGIBLE','2026-09-15T21:07:29+00:00'),unit('AAA','FEATURE_READY','2026-09-15T21:07:29+00:00'),unit('BBB','RAW_FAILED','2026-09-15T21:07:30+00:00')];
  {
    const words=library.words(root),prior={t:c.t,I18N:c.I18N};Object.assign(c,{t:words.t,I18N:words.I18N});
    bodies['/api/workspace/preparation']=readback('DEFERRED',{task_id:TASK,plan_hash:'plan-1',progress:{phase:'prepare_data'},listing_activity:activity([unit('AAA','PENDING','2026-09-15T21:07:30+00:00',{raw_through:'2026-09-10'})],{execution_id:'lagging-provider'})});
    bodies['/api/tasks/recovery']=view('DEFERRED','prepare_data');await w.refresh('welcome');
    for(const lang of ['en','zh']) {
      words.I18N.set(lang);const rendered=String(w.page()),step=words.t('history retained through {d}; waiting to retry the provider',{d:'2026-09-10'}),standing=words.t('waiting for provider data');
      assert.ok(rendered.includes(step)&&rendered.includes(standing),'pending retained-history row and latest-unit rail have their own words in '+lang);
      if(lang==='zh')assert.ok(/[\u3400-\u9fff]/.test(step)&&/[\u3400-\u9fff]/.test(standing),'both pending meanings are Chinese');
    }
    Object.assign(c,prior);
  }
  bodies['/api/workspace/preparation']=readback('RUNNING',{task_id:TASK,plan_hash:'plan-1',progress:{phase:'prepare_data',candidates:4,raw_ready:0,quality_eligible:0,failed:0,retry_after_at:null},listing_activity:activity(firstRows)});
  bodies['/api/tasks/recovery']=view('RUNNING','prepare_data');
  const lifecyclePosts=()=>posts.filter(([p])=>p==='/api/cancel'||p==='/api/recover'||p.endsWith('/confirm')||p.endsWith('/plan')).length;
  const postsBeforeArea=lifecyclePosts();
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('class="fv-surface prep-area"') && markup.includes('data-folded="false"') && markup.includes('data-shown="prepare_data"'),'one work area, expanded, following the current stage');
  assert.ok(markup.includes('tp-number">0</span>') && markup.includes('> / 4 candidates with raw bars'),'the denominator is shown before the first chunk moves the count');
  assert.ok(markup.includes('data-listing-key="aaa-0000-listing"') && markup.includes('raw bars acquired through 2026-09-15') && markup.includes('quality: eligible') && markup.includes('admitted for Features') && (markup.match(/data-listing-key=/g) || []).length===2,'a listing is one row with its recorded path (the stub joins fragments with commas)');
  assert.ok(markup.includes('prep-unit attention') && markup.includes('failed · <span class="coded" data-tip="data.provider_fetch_failed">'),'the refused unit is shown as recorded, its code in words with the code on hover (N6, law 80)');
  assert.ok(markup.includes('4 of 4 units of this execution retained in the snapshot') && !markup.includes('not retained'),'the snapshot is labelled with its bounds');
  assert.equal(w.area().rows,2,'rows are folded per listing');assert.equal(w.area().arrived,0,'the first paint of an execution announces nothing');
  // more units on a live read: only the new listing is an arrival; a listing seen before is not
  bodies['/api/workspace/preparation'].listing_activity=activity([...firstRows,unit('CCC','RAW_READY','2026-09-15T21:07:33+00:00'),unit('CCC','QUALITY_ELIGIBLE','2026-09-15T21:07:33+00:00')],{sequence:2,observed:7,retained:6,dropped:1});
  w.observe();await new Promise(r=>setTimeout(r,20));markup=w.page();
  assert.equal(w.area().rows,3);assert.equal(w.area().arrived,1,'the newly seen listing is the one arrival');
  assert.ok(markup.includes('6 of 7 units of this execution retained in the snapshot · 1 earlier units not retained'),'retention gaps are disclosed, never reconstructed');
  // a read that fails keeps the rows and announces nothing on the read that recovers
  c.Data.read=async p=>{reads.push(p);throw Error('TypeError: Failed to fetch');};
  w.observe();await new Promise(r=>setTimeout(r,20));markup=w.page();
  assert.ok(markup.includes('data-listing-key="ccc-0000-listing"') && markup.includes('last successful read') && markup.includes('data-disconnected="true"'),'rows are kept through an outage and labelled as the last successful read');
  c.Data.read=realRead;
  bodies['/api/workspace/preparation'].listing_activity=activity([...firstRows,unit('CCC','RAW_READY','2026-09-15T21:07:33+00:00'),unit('CCC','QUALITY_ELIGIBLE','2026-09-15T21:07:33+00:00'),unit('DDD','RAW_READY','2026-09-15T21:07:36+00:00')],{sequence:3,observed:8,retained:7,dropped:1});
  w.observe();await new Promise(r=>setTimeout(r,20));
  assert.equal(w.area().rows,4);assert.equal(w.area().arrived,1,'the read that recovers from an outage is a baseline: the earlier arrival is still pending its paint, the new row is not announced');
  // hold / follow is the reader's choice; the rail shows it
  w.follow();markup=w.page();assert.ok(markup.includes('Resume following') && markup.includes('Reading held') && w.area().follow===false,'reading is held on request');
  w.follow();assert.ok(w.page().includes('Hold reading') && w.area().follow===true,'and following resumes');
  // the reader's scroll versus the page's own: the position this module set or last saw is not
  // the reader's movement when a repaint restores it (nor when the restoration clamps to a shorter
  // log); any other position is the reader's, however soon after a paint, and carries the intent
  const W=c.W;const log={scrollTop:0,clientHeight:450,scrollHeight:2250};
  W.A.logTop=null;log.scrollTop=1800;W.heldByScroll(log);assert.ok(W.A.follow===true && W.A.logTop===1800,'the first observed position at the end keeps following');
  W.heldByScroll(log);assert.ok(W.A.follow===true,'a restoration to the known end changes nothing');
  log.scrollTop=1625;W.heldByScroll(log);assert.ok(W.A.follow===false && W.A.logTop===1625 && prefs.workScene.follow===false,'a real move away from the end holds the reading at once');
  W.heldByScroll(log);assert.ok(W.A.follow===false,'the repaint restoring 1625 keeps the hold');
  log.scrollHeight=2000;log.scrollTop=1550;W.heldByScroll(log);assert.ok(W.A.follow===false && W.A.logTop===1550,'a restoration clamped to a shorter log is not the reader either');
  log.scrollHeight=2250;log.scrollTop=1800;W.heldByScroll(log);assert.ok(W.A.follow===true,'the reader back at the end resumes following');
  log.scrollTop=1000;W.heldByScroll(log);assert.equal(W.A.follow,false);
  c.document={getElementById:()=>log};w.follow();delete c.document;
  assert.ok(W.A.follow===true && log.scrollTop===2250 && W.A.logTop===2250,'resuming by the button moves the log to the end as this module\'s own move');
  log.scrollTop=1800;W.heldByScroll(log);assert.ok(W.A.follow===true && W.A.logTop===1800,'the clamped end the browser reports for that move is not the reader either');
  assert.ok(w.page().includes('Hold reading') && w.area().follow===true);
  // inspecting an earlier stage holds it across a re-read and offers the way back; it runs nothing
  w.inspect('freeze_sources');markup=w.page();
  assert.ok(markup.includes('data-shown="freeze_sources"') && markup.includes('Inspecting') && markup.includes('Return to current:workspace-stage::') && markup.includes('data-inspecting="true"'),'the inspected stage is shown, held, with the way back');
  assert.ok(markup.includes('Verified by Task Control; its record is kept with 1 evidence reference. Inspecting it runs nothing.'),'the retained record of the inspected stage');
  assert.deepEqual(prefs.workScene,{task:TASK,folded:false,inspect:'freeze_sources',follow:true},'the choice is kept for this exact Task');
  w.observe();await new Promise(r=>setTimeout(r,20));assert.ok(w.page().includes('data-shown="freeze_sources"'),'a re-read does not pull the reader away');
  w.inspect('');assert.ok(w.page().includes('data-shown="prepare_data"') && w.page().includes('Current stage'),'return to current follows again');
  w.inspect('prepare_data');assert.equal(w.area().inspect,null,'selecting the current stage is following it');
  // folding leaves a compact summary with the stage, the count and the status; a reload of the
  // same Task keeps it; another Task starts expanded and following
  w.fold();markup=w.page();
  assert.ok(markup.includes('data-folded="true"') && markup.includes('class="prep-summary"') && markup.includes('stage 2 of 5') && markup.includes('0 / 4') && !markup.includes('id="prepListingLog"'),'folded: the summary, not the body');
  assert.deepEqual(prefs.workScene,{task:TASK,folded:true,inspect:null,follow:true});
  // the discovered Task was pinned in the route: a latest B answering a bare read never
  // replaces it here (the read asks for A by id); B is shown only when chosen (the route)
  assert.equal(c.location.hash,'#preparation='+TASK,'the discovered preparation is pinned in place');
  const OTHER='0000000-other-task';
  const readbackB=readback('RUNNING',{task_id:OTHER,plan_hash:'plan-2',progress:{phase:'prepare_data',candidates:4,raw_ready:0,quality_eligible:0,failed:0,retry_after_at:null}});
  const readbackA=bodies['/api/workspace/preparation'];
  const byTask={[TASK]:readbackA,[OTHER]:readbackB};
  const viewByTask={[OTHER]:{...view('RUNNING','prepare_data'),task_id:OTHER}};
  const settle=(ms=20)=>new Promise(r=>setTimeout(r,ms));
  const readByTask=(over={})=>async p=>{reads.push(p);const [key,query]=p.split('?');const id=new URLSearchParams(query||'').get('task_id');
    if(over[key]) return over[key](id);
    if(key==='/api/workspace/preparation'){ if(id && !byTask[id]) throw Error('workspace_preparation.task_not_found'); return id ? byTask[id] : bodies[key]; }
    if(key==='/api/tasks/recovery') return viewByTask[id] || bodies[key];
    return bodies[key]!==undefined?bodies[key]:bodies[p];};
  c.Data.read=async p=>{reads.push(p);const [key,query]=p.split('?');const id=new URLSearchParams(query||'').get('task_id');if(key==='/api/workspace/preparation')return byTask[id] || readbackB;if(key==='/api/tasks/recovery')return id===OTHER ? {...view('RUNNING','prepare_data'),task_id:OTHER} : bodies[key];return bodies[key]!==undefined?bodies[key]:bodies[p];};
  await w.refresh('welcome');assert.ok(w.area().task===TASK && w.page().includes('data-folded="true"'),'a fresh read keeps the pinned A folded as chosen; the latest B is not substituted');
  c.location.hash='#page=welcome&preparation='+OTHER;
  await w.refresh('welcome');assert.ok(w.page().includes('data-folded="false"') && w.area().task===OTHER && w.area().rows===0,'the chosen B: expanded, following, no rows carried over');
  c.location.hash='#page=welcome&preparation='+TASK;
  bodies['/api/workspace/preparation']=readback('RUNNING',{task_id:TASK,plan_hash:'plan-1',progress:{phase:'prepare_features',status:'PREPARING_FEATURE_CLOSURE'},listing_activity:activity(firstRows,{availability:'NOT_CURRENT'})});
  byTask[TASK]=bodies['/api/workspace/preparation'];
  bodies['/api/tasks/recovery']=view('RUNNING','prepare_features');
  await w.refresh('welcome');assert.ok(w.page().includes('data-folded="true"') && w.area().task===TASK,'the same Task returns folded, as chosen');
  w.fold();
  // a refused selection is said so, with the way to the current preparation; nothing else read
  c.location.hash='#page=welcome&preparation=no-such-task';
  c.Data.read=readByTask();
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('This preparation Task cannot be shown') && markup.includes('workspace_preparation.task_not_found') && markup.includes('Open the current preparation:workspace-current') && !markup.includes('prep-area'),'a missing Task is a typed refusal, never a fall-back to latest');
  c.location.hash='#page=welcome&preparation='+TASK;
  const beforeReturn=reads.length;markup=w.page();
  assert.ok(markup.includes('SKELETON(') && !markup.includes('This preparation Task cannot be shown') && !markup.includes('prep-area'),'the page entry starts the read of the chosen Task itself and shows nothing of the refusal meanwhile');
  await settle();
  assert.equal(reads[beforeReturn],'/api/workspace/preparation?task_id='+TASK,'the chosen Task is read by the page entry after a refusal, once');
  assert.ok(w.page().includes('prep-area') && w.page().includes('data-shown="prepare_features"') && w.scene().body.task_id===TASK,'A again, exactly, through the page entry');
  assert.equal(reads.slice(beforeReturn).filter(v=>v.startsWith('/api/workspace/preparation')).length,1,'one read per choice');
  // B chosen, then A chosen while B is still being read: the late answer for B is discarded
  let releaseB;const gateB=new Promise(r=>{releaseB=r;});
  c.Data.read=readByTask({'/api/workspace/preparation':async id=>{if(id===OTHER){await gateB;return readbackB;}return byTask[id];}});
  c.location.hash='#page=welcome&preparation='+OTHER;markup=w.page();await settle(5);
  assert.ok(markup.includes('SKELETON(') && !markup.includes('prep-area') && w.scene().body===null,'while B is read nothing of A stays on the page as if it were B');
  c.location.hash='#page=welcome&preparation='+TASK;w.page();await settle();
  assert.ok(w.area().task===TASK && w.scene().body.task_id===TASK,'A is read and shown');
  releaseB();await settle();
  assert.ok(w.area().task===TASK && w.scene().body.task_id===TASK && w.page().includes('data-shown="prepare_features"') && !w.scene().loading,'the late answer for B did not replace the chosen A');
  // B chosen while A is shown and the owners do not answer: the failure is B's, said as such;
  // A's facts are not kept on the page marked as B's last observation
  c.Data.read=readByTask({'/api/workspace/preparation':async id=>{if(id===OTHER) throw Error('TypeError: Failed to fetch');return byTask[id];}});
  c.location.hash='#page=welcome&preparation='+OTHER;w.page();await settle();
  markup=w.page();
  assert.ok(w.scene().body===null && markup.includes('TypeError: Failed to fetch') && !markup.includes('prep-area') && !markup.includes('Last observation') && !markup.includes('could not read the owners'),'a failed read for B shows B\'s failure, never A\'s facts as stale');
  c.location.hash='#page=welcome&preparation='+TASK;w.page();await settle();
  assert.ok(w.scene().body.task_id===TASK && w.page().includes('prep-area') && !w.page().includes('Failed to fetch'),'and A reads normally again');
  c.Data.read=realRead;
  // the earlier stage's rows are history once the stage moved on: shown on inspection, said so
  w.inspect('prepare_data');markup=w.page();
  assert.ok(markup.includes('data-history="true"') && markup.includes('retained history, not current work') && markup.includes('Last retained unit'),'retained rows of an earlier stage are history, never current work');
  assert.ok(/<details class="run-log reveal-details prep-log" data-moving="false">/.test(markup) && markup.includes('run-log-fold'),'a log that no longer moves is folded to its one line, its units a press away (R14): '+(markup.match(/<(details|section) class="run-log[^>]*>/) || [''])[0]);
  w.inspect('');
  // reading held on the data log while the owner advances: the data stage stays shown as an
  // inspection with the way back, the hold survives a reload of the same Task, and returning
  // to current follows the stage again without touching the hold
  bodies['/api/workspace/preparation']=readback('RUNNING',{task_id:TASK,plan_hash:'plan-1',progress:{phase:'prepare_data',candidates:4,raw_ready:0,quality_eligible:0,failed:0,retry_after_at:null},listing_activity:activity(firstRows)});
  byTask[TASK]=bodies['/api/workspace/preparation'];bodies['/api/tasks/recovery']=view('RUNNING','prepare_data');
  await w.refresh('welcome');w.follow();assert.equal(w.area().follow,false);
  assert.deepEqual(prefs.workScene,{task:TASK,folded:false,inspect:null,follow:false},'the hold is remembered');
  bodies['/api/workspace/preparation']=readback('RUNNING',{task_id:TASK,plan_hash:'plan-1',progress:{phase:'prepare_features',status:'PREPARING_FEATURE_CLOSURE'},listing_activity:activity(firstRows,{availability:'NOT_CURRENT'})});
  byTask[TASK]=bodies['/api/workspace/preparation'];bodies['/api/tasks/recovery']=view('RUNNING','prepare_features');
  w.observe();await new Promise(r=>setTimeout(r,20));markup=w.page();
  assert.ok(markup.includes('data-shown="prepare_data"') && markup.includes('Inspecting') && markup.includes('Return to current:workspace-stage::') && markup.includes('id="prepListingLog"'),'the held log stays shown when the owner advances, with the way back');
  assert.deepEqual(prefs.workScene,{task:TASK,folded:false,inspect:'prepare_data',follow:false});
  const reloaded=vm.createContext({...c});vm.runInContext(fs.readFileSync(path.join(root,'status.js'),'utf8')+fs.readFileSync(path.join(root,'router.js'),'utf8')+fs.readFileSync(path.join(root,'live-workarea.js'),'utf8')+fs.readFileSync(path.join(root,'live-workspace.js'),'utf8')+';globalThis.w=LiveWorkspace;',reloaded); // a reload: the router and the page again, over the kept route and preference
  reloaded.patchMain=()=>patches.push(c.app.page);reloaded.render=()=>renders.push(c.app.page);
  await reloaded.w.refresh('welcome');markup=reloaded.w.page();
  assert.ok(reloaded.w.area().follow===false && markup.includes('data-shown="prepare_data"') && markup.includes('Resume following'),'after a reload of the same Task the hold and the held stage are still there');
  reloaded.w.inspect('');assert.ok(reloaded.w.page().includes('data-shown="prepare_features"') && reloaded.w.area().follow===false,'return to current follows the stage; the hold on the log is the reader\'s to lift');
  reloaded.w.observe();await settle();
  assert.ok(reloaded.w.page().includes('data-shown="prepare_features"') && reloaded.w.area().inspect===null && reloaded.w.area().follow===false,'the next poll of the same stage stays on it: the hold is not re-armed');
  assert.deepEqual(prefs.workScene,{task:TASK,folded:false,inspect:null,follow:false},'the log-follow preference is the reader\'s, kept apart');
  w.inspect('');await w.refresh('welcome');w.observe();await settle();
  assert.ok(w.page().includes('data-shown="prepare_features"') && w.area().inspect===null,'the same in the page that observed the transition');
  w.follow();
  // workspace discovery is the latest Task's story: with B verified as the latest, reading the
  // cancelled A on request changes nothing that other pages say about the workspace
  const readbackDone={...readback('SUCCEEDED',{task_id:OTHER,plan_hash:'plan-2',inputs:[{input_id:'family',binding_hash:'binding'}]}),selected:false,latest_task_id:OTHER};
  const readbackOld={...readback('CANCELLED',{task_id:TASK,plan_hash:'plan-1',failure_code:'TASK_CANCELLED_AT_SAFE_CHECKPOINT'}),selected:true,latest_task_id:OTHER};
  byTask[TASK]=readbackOld;byTask[OTHER]={...readbackDone,selected:true};viewByTask[OTHER]={...view('SUCCEEDED','verify_inputs'),task_id:OTHER};viewByTask[TASK]=view('CANCELLED','prepare_data');
  bodies['/api/workspace/preparation']=readbackDone;c.Data.read=readByTask();discovered.length=0;
  c.location.hash='#page=welcome';await w.refresh('welcome');
  assert.deepEqual(discovered,[OTHER],'a bare read discovers the latest');
  assert.equal(c.location.hash,'#page=welcome&preparation='+OTHER,'and pins it');
  c.location.hash='#page=welcome&preparation='+TASK;w.page();await settle();
  markup=w.page();
  assert.ok(markup.includes('Cancelled') && markup.includes('data-shown="prepare_data"') && w.scene().body.task_id===TASK,'the cancelled A is shown exactly, at the stage it stopped');
  assert.deepEqual(discovered,[OTHER],'reading a historical Task on request is not a workspace-state update');
  c.app.page='history';assert.equal(String(w.notice()),'','the workspace is still prepared (B) on other pages');c.app.page='welcome';
  c.location.hash='#page=welcome&preparation='+OTHER;w.page();await settle();
  assert.deepEqual(discovered,[OTHER,OTHER],'the latest read on request is the workspace\'s story too');
  assert.equal(lifecyclePosts(),postsBeforeArea,'inspection is read-only');
  delete viewByTask[TASK];c.location.hash='#page=welcome&preparation='+TASK;c.Data.read=realRead;
  assert.equal(lifecyclePosts(),postsBeforeArea,'no reading, folding, inspection or following posts anything');
  bodies['/api/workspace/preparation']=readback('RECOVERY_REQUIRED',{task_id:TASK,plan_hash:'plan-1',failure_code:'TASK_EXECUTION_INTERRUPTED'});
  bodies['/api/tasks/recovery']=view('RECOVERY_REQUIRED','publish_inputs',{stop:{code:'TASK_EXECUTION_INTERRUPTED',stage_id:'publish_inputs',detail:'The worker stopped before the current stage was verified.',recoverable:true}});
  await w.refresh('welcome');
  // Pending truth review follows the preparation owner's read request before any mutation.
  const pendingTruth=(extra={})=>readback('BLOCKED',{task_id:TASK,plan_hash:'plan-1',failure_code:'data.truth_review_required',execution_binding_changed:false,next_action:'DATA_ISSUES',confirmation_available:false,next_requests:{issues:{operation:'DATA_ISSUES'}},...extra});
  bodies['/api/workspace/preparation']=pendingTruth({execution_binding_changed:true});
  bodies['/api/tasks/recovery']=view('BLOCKED','prepare_features',{stop:{code:'data.truth_review_required',stage_id:'prepare_features',detail:'Listings need a data decision.',recoverable:false}});
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('LINK:Data issues@issues') && !markup.includes('Retry this Task') && !markup.includes('Preview preparation:workspace-preview:prepare'),'pending choices precede re-planning even when the execution binding changed');
  let beforeContinuation=posts.length;
  modal=null;await w.preview('continue-preparation');
  assert.equal(modal,null,'a stale continue press opens no confirmation when the owner withholds it');
  assert.equal(posts.length,beforeContinuation,'a stale continue press mutates nothing');
  assert.ok(w.page().includes('Preview the preparation first.'),'the withheld continuation answers in the existing words');
  bodies['/api/workspace/preparation']=pendingTruth({superseded_by_task_id:OTHER,next_action:'WORKSPACE_PREPARE_READBACK',next_requests:{successor:{operation:'WORKSPACE_PREPARE_READBACK',task_id:OTHER}}});
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('Inspect successor Task:workspace-task:'+OTHER) && !markup.includes('Retry this Task'),'the owner\'s successor remains the way on for a historical stopped Task');
  // A recorded decision restores only the exact confirmation the owner offers.
  bodies['/api/workspace/preparation']=readback('BLOCKED',{task_id:TASK,plan_hash:'plan-1',failure_code:'data.truth_review_required',execution_binding_changed:false,confirmation_available:null,next_requests:{confirm:{operation:'WORKSPACE_PREPARE_CONFIRM',preparation_plan_hash:'plan-1'}}});
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('Blocked') && markup.includes('Retry this Task:workspace-preview:continue-preparation:') && markup.includes('LINK:Data issues') && !markup.includes('Resume this Task:task-recovery'),'a stopped Task retries only after the owner offers its confirm');
  modal=null;await w.preview('continue-preparation');assert.ok(modal && modal.content.includes('plan-1'),'the continuation is confirmed explicitly');
  bodies['/api/workspace/preparation/confirm']={status:'ADMITTED',task_id:TASK,lifecycle:'RECOVERY_REQUIRED'};await w.commit();
  assert.deepEqual(posts.at(-1),['/api/workspace/preparation/confirm',{preparation_plan_hash:'plan-1'}],'the continuation is the owner\'s own confirmation under the same plan');
  bodies['/api/workspace/preparation']=readback('BLOCKED',{task_id:TASK,plan_hash:'plan-1',failure_code:'catalog.replace_blocked',execution_binding_changed:false,next_requests:{}});
  bodies['/api/tasks/recovery']=view('BLOCKED','prepare_features',{stop:{code:'catalog.replace_blocked',stage_id:'prepare_features',detail:'Storage did not replace the retained receipt.',recoverable:false}});
  await w.refresh('welcome');markup=w.page();
  assert.ok(!markup.includes('Retry this Task'),'a plan hash alone invents no confirmation');
  beforeContinuation=posts.length;modal=null;await w.preview('continue-preparation');
  assert.equal(modal,null,'an absent owner request opens no confirmation');
  assert.equal(posts.length,beforeContinuation,'an absent owner request sends no assembled mutation');
  // U63 (V375): a preparation the provider deferred says why in the owner's words and when it resumes; Resume is the owner's
  // own request, held until its time
  const said='The market data provider limited the requests or did not answer, which is the provider\'s state, not the workspace\'s. Every listing already fetched is kept and the work resumes from them: send the same plan again once `retry_after_at` has passed.';
  const deferred=(at)=>readback('DEFERRED',{task_id:TASK,plan_hash:'plan-1',failure_code:'data.rate_limited',progress:{phase:'prepare_data',candidates:120,raw_ready:60,quality_eligible:60,failed:0,retry_after_at:at},detail:said,retry_after_at:at,next_requests:{resume:{operation:'WORKSPACE_PREPARE_CONFIRM',preparation_plan_hash:'plan-1'}}});
  bodies['/api/workspace/preparation']=deferred('2030-01-01T00:00:00+00:00');
  bodies['/api/tasks/recovery']=view('DEFERRED','prepare_data',{stop:{code:'data.rate_limited',stage_id:'prepare_data',detail:'Deferred.',recoverable:false}});
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('<span class="owner-text">'+said+'</span>')&&markup.includes('It resumes from ')&&markup.includes('Resume this preparation:workspace-preview:continue-preparation:Not due until'),'the owner\'s words, the time, Resume held until it: '+markup.slice(markup.indexOf('owner-text')-100,markup.indexOf('owner-text')+700));
  bodies['/api/workspace/preparation']=deferred('2026-01-01T00:00:00+00:00');await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('Resume this preparation:workspace-preview:continue-preparation:'+'BTN-END')===false&&/Resume this preparation:workspace-preview:continue-preparation:(?!Not due)/.test(markup),'Resume once the time has passed');
  modal=null;await w.preview('continue-preparation');assert.ok(modal&&modal.title.includes('Resume this preparation'),'the resume is confirmed explicitly');
  bodies['/api/workspace/preparation/confirm']={status:'ADMITTED',task_id:TASK,lifecycle:'RUNNING'};await w.commit();
  assert.deepEqual(posts.at(-1),['/api/workspace/preparation/confirm',{preparation_plan_hash:'plan-1'}],'the owner\'s resume request, whole');
  bodies['/api/tasks/recovery']=view('BLOCKED','prepare_features',{stop:{code:'data.truth_review_required',stage_id:'prepare_features',detail:'Listings need a data decision.',recoverable:false}}); // the stopped Task the next case reads
  bodies['/api/workspace/preparation']=readback('BLOCKED',{task_id:TASK,plan_hash:'plan-1',failure_code:'x',execution_binding_changed:true,next_action:'WORKSPACE_PREPARE_PLAN',confirmation_available:false});
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('Preview preparation:workspace-preview:prepare') && !markup.includes('Retry this Task'),'a Task planned under an older product is re-planned, not retried');
  bodies['/api/workspace/preparation']=readback('CANCELLED',{task_id:TASK,plan_hash:'plan-1',failure_code:'TASK_CANCELLED_AT_SAFE_CHECKPOINT'});
  bodies['/api/tasks/recovery']=view('CANCELLED','prepare_data',{stop:{code:'TASK_CANCELLED_AT_SAFE_CHECKPOINT',stage_id:'prepare_data',detail:'Acknowledged.',recoverable:false}});
  await w.refresh('welcome');markup=w.page();
  assert.ok(markup.includes('Cancelled') && markup.includes('Preview preparation:workspace-preview:prepare') && markup.includes('from its start to its last recorded activity'),'a cancelled Task offers a new preview; its elapsed time stops at its last activity');
  const stopped=reads.length;w.observe();await new Promise(r=>setTimeout(r,20));assert.equal(reads.length,stopped,'a stopped Task is not re-read on the cadence');
  // completed: the exact input and the one next step, which edits a draft and runs nothing
  // (the input owner's listing read earlier on the inputs page names this version)
  bodies['/api/workspace/preparation']=readback('SUCCEEDED',{task_id:TASK,plan_hash:'plan-1',inputs:[{input_id:'family',binding_hash:'binding'}]});
  bodies['/api/tasks/recovery']=view('SUCCEEDED','verify_inputs',{verified_stage_count:5,artifact_refs:['playpen://x']});
  await w.refresh('welcome');markup=w.page();
  assert.ok(!markup.includes('prep-area') && !markup.includes('Request cancel:'),'the completion leaves the Home the Home; the verified input is the Inputs page\'s');
  const draftPosts=posts.length;await w.selectInput(JSON.stringify(['family','binding']));
  assert.deepEqual(drafts.at(-1),JSON.stringify(['family','binding']));assert.equal(posts.length,draftPosts,'selecting the input into a draft posts nothing');
  // the entry notice names the workspace's state from the session context and the Task projection
  c.Data.preparation=()=>readback('INITIALIZATION_REQUIRED');c.app.page='history';
  assert.ok(String(w.notice()).includes('not prepared for research yet') && String(w.notice()).includes('LINK:Prepare workspace'));
  c.Data.preparation=()=>readback('RUNNING',{task_id:TASK});c.Data.tasks=()=>[{task_id:TASK,lifecycle:'RUNNING',verified_stage_count:1,total_stage_count:5}];
  assert.ok(String(w.notice()).includes('Preparation is running · stage 2 of 5'),'the notice follows the live projection');
  c.Data.tasks=()=>[{task_id:TASK,lifecycle:'RECOVERY_REQUIRED',verified_stage_count:3,total_stage_count:5,latest_failure_code:'TASK_EXECUTION_INTERRUPTED'}];
  assert.ok(String(w.notice()).includes('Preparation needs attention') && String(w.notice()).includes('TASK_EXECUTION_INTERRUPTED'));
  c.Data.preparation=()=>readback('SUCCEEDED',{task_id:TASK,inputs:[{input_id:'f',binding_hash:'b'}]});assert.equal(String(w.notice()),'','a prepared workspace has no notice');
  c.app.page='welcome';c.Data.preparation=()=>readback('INITIALIZATION_REQUIRED');assert.equal(String(w.notice()),'','never on the scene itself');
  // --- round E5: the storage page's Evidence category, from the readback's `evidence` block
  {
    const evidence={summary:{source_object_bytes:1048576,source_blob_bytes:524288,vector_payload_bytes:262144,vector_object_count:12,index_payload_bytes:131072,sealed_artifact_bytes:65536,staging_bytes:0,task_state_bytes:4096},
      indexes:[{index_id:'1111aaaa2222bbbb3333cccc',generation_format:'hybrid-v2',corpus_hash:'c'.repeat(64),database_bytes:131072,availability:'AVAILABLE',payload_bytes:131072,payload_available:true,payload_proof:'PROVED',record_count:340,protected_by:['ACTIVE_LEASE'],eligible:false,available_actions:['PIN']},{index_id:'4444dddd5555eeee6666ffff',generation_format:'hybrid-v2',corpus_hash:'d'.repeat(64),database_bytes:0,availability:'EVICTED_BY_RETENTION',payload_bytes:0,payload_available:false,payload_proof:'storage.evidence_index_payload_absent',record_count:0,protected_by:[],eligible:false,available_actions:['REBUILD']},{index_id:'7777aaaa8888bbbb9999cccc',generation_format:'legacy-v1',corpus_hash:'e'.repeat(64),database_bytes:0,availability:'EVICTED_BY_RETENTION',payload_bytes:0,payload_available:false,payload_proof:'storage.evidence_index_payload_absent',record_count:0,protected_by:[],eligible:false,available_actions:[],rebuild_limit:'This legacy index has no committed vector payload and cannot be rebuilt. Start a new Evidence preparation.'}],
      unreferenced:{source_objects:2,indexes:0,vector_objects:1},pinned_index_ids:[]};
    bodies['/api/workspace/storage']={status:'AVAILABLE',display:{},inputs:[],pending_cleanup:[],by_root:{},managed_bytes:2097152,logical_bytes:2097152,free_disk_bytes:1e9,capacity_status:'WITHIN_BUDGET',budget:{},evidence};
    const tableStub=c.table, trStub=c.tr; c.table=(headers,rows)=>'<table>'+headers.map(h=>h.label).join(' | ')+' '+(Array.isArray(rows) ? rows.join('') : String(rows))+'</table>'; c.tr=(cells)=>'<tr>'+cells.map(x=>String(x ?? '')).join(' | ')+'</tr>';
    c.app.page='storage';await w.refresh();let page=String(w.page()).replace(/\s+/g,' ');
    assert.ok(page.includes('Originals')&&/1\.00[^K]*MiB/.test(page)&&page.includes('Vectors')&&page.includes('12 objects')&&page.includes('Sealed records'),'the Evidence category reads its summary as physical bytes: '+page.slice(page.indexOf('Originals')-40,page.indexOf('Originals')+300));
    assert.ok(page.includes('1111aaaa2222')&&/[Aa]vailable/.test(page)&&page.includes('Active lease')&&page.includes('proved'),'an available index names its protection and its proof: '+page.slice(page.indexOf('1111aaaa'),page.indexOf('1111aaaa')+500));
    assert.ok(page.includes('4444dddd5555')&&/[Ee]victed/.test(page)&&page.includes('Rebuild'),'an evicted index offers Rebuild: '+page.slice(page.indexOf('4444dddd'),page.indexOf('4444dddd')+500));
    // U4 (V199): each row offers the actions the Host names for it, in its order, and nothing of the page's own
    const rowOf=(id)=>page.slice(page.indexOf(id),page.indexOf('</tr>',page.indexOf(id)));
    assert.ok(rowOf('1111aaaa2222').includes('Pin')&&!rowOf('1111aaaa2222').includes('Rebuild')&&!rowOf('1111aaaa2222').includes('Unpin'),'the available index: Pin alone: '+rowOf('1111aaaa2222'));
    assert.ok(!rowOf('4444dddd5555').includes('Pin'),'the evicted index is offered no pin');
    // U80 (V560): a legacy index that cannot be rebuilt says so where its Rebuild would stand, the owner's words its hover
    assert.ok(rowOf('7777aaaa8888').includes('Cannot be rebuilt')&&!rowOf('7777aaaa8888').includes('Rebuild'),'a legacy index says it cannot be rebuilt: '+rowOf('7777aaaa8888'));
    c.table=tableStub;c.tr=trStub;
    assert.ok(page.includes('Unreferenced')&&page.includes('Source objects 2')&&page.includes('Vector objects 1')&&!page.includes('Indexes 0'),'the unreferenced objects are counted by kind, zeros left out: '+page.slice(page.indexOf('Unreferenced'),page.indexOf('Unreferenced')+200));
    posts.length=0;await w.preview('rebuild-index','4444dddd5555eeee6666ffff');
    assert.ok(modal&&String(modal.title).includes('Rebuild this evidence index'),'the rebuild is confirmed as the row\'s own request: '+String(modal&&modal.title));
    bodies['/api/workspace/storage/evidence-rebuild']={status:'REBUILT',receipt_hash:'r'.repeat(64)};await w.commit();
    assert.deepEqual(posts.at(-1),['/api/workspace/storage/evidence-rebuild',{evidence_index_id:'4444dddd5555eeee6666ffff'}],'the rebuild names the exact index and nothing else');
    await w.preview('pin-index',JSON.stringify(['4444dddd5555eeee6666ffff','PIN']));
    assert.ok(!modal || !String(modal.title).includes('Pin this evidence index'),'a pin the Host does not offer is not confirmed');
    await w.preview('pin-index',JSON.stringify(['1111aaaa2222bbbb3333cccc','PIN']));
    bodies['/api/workspace/storage/pin']={status:'PINNED'};await w.commit();
    assert.deepEqual(posts.at(-1),['/api/workspace/storage/pin',{input_binding_hash:'1111aaaa2222bbbb3333cccc',input_pinned:true}],'a pin names the index through the owner\'s pin route');
    // while an approved cleanup awaits recovery the Host offers no index action, and the rows offer none
    bodies['/api/workspace/storage']={...bodies['/api/workspace/storage'],pending_cleanup:['p'.repeat(64)],status:'RECOVERY_REQUIRED',evidence:{...evidence,indexes:evidence.indexes.map(ix=>({...ix,available_actions:[]}))}};
    c.table=(headers,rows)=>'<table>'+(Array.isArray(rows) ? rows.join('') : String(rows))+'</table>';c.tr=(cells)=>'<tr>'+cells.map(x=>String(x ?? '')).join(' | ')+'</tr>';
    await w.refresh();page=String(w.page()).replace(/\s+/g,' ');c.table=tableStub;c.tr=trStub;
    const indexes=page.slice(page.indexOf('1111aaaa2222'),page.indexOf('</table>',page.indexOf('1111aaaa2222')));
    assert.ok(!/\bPin\b|Rebuild|Unpin/.test(indexes),'no index action while recovery is required: '+indexes);
    bodies['/api/workspace/storage']={status:'AVAILABLE',display:{},inputs:[],pending_cleanup:[]};c.app.page='storage';await w.refresh();page=String(w.page()).replace(/\s+/g,' ');
    assert.ok(!page.includes('data-storage="evidence"'),'a readback without the category draws no Evidence section (N3, law 81)');
  }
  // --- U48 (V209): the held state's backups beside the inventory -- read on an explicit read of Storage,
  // made only on the person's confirmation, restored by the client's own command (named, never run)
  {
    const gen=(h,reason,at)=>({generation_hash:h.repeat(64),created_at:at,reason,files:12,tables:9,listed:3,absent:['market-data.duckdb/universe_history']});
    const root='Z:/Test Fixture/AlphaLattice/backups/ws-1';
    bodies['/api/workspace/backup']={status:'READ',backup_root:root,generation_hash:null,
      last_automatic_attempt:{status:'FAILED',failure_code:'workspace_backup.object_changed_while_copied',at:'2026-09-29T21:00:00+00:00'},
      generations:[gen('a','DATA_UPDATE','2026-09-29T20:00:00+00:00'),gen('b','REQUEST','2026-09-28T20:00:00+00:00')]};
    const tableStub=c.table, trStub=c.tr, panelStub=c.panel;c.panel=(a,b,content,action='')=>a+b+content+action;
    c.table=(headers,rows)=>'<table>'+headers.map(h=>h.label).join(' | ')+' '+rows.join('')+'</table>';
    c.tr=(cells)=>'<tr>'+cells.map(x=>String(x ?? '')).join(' | ')+'</tr>';
    c.app.workspace='ws-1';reads.length=0;posts.length=0;c.app.page='storage';
    await w.refresh();await new Promise(r=>setImmediate(r));
    assert.ok(reads.includes('/api/workspace/backup'),'an explicit read of Storage reads the backups too');
    assert.equal(posts.length,0,'reading the backups makes none');
    let page=String(w.page()).replace(/\s+/g,' ');
    const box=page.slice(page.indexOf('Backups'));
    assert.ok(box.includes(root)&&box.includes('Last automatic backup')&&box.includes('A file changed while it was copied'),
      'the root and the failed automatic attempt, its code in words: '+box.slice(0,600));
    assert.ok(page.includes('The last automatic backup failed'),'a failed automatic attempt is said as a warning');
    assert.ok(box.includes('After a data update')&&box.includes('On request')&&box.includes('aaaaaaaaaaaa')&&box.includes('bbbbbbbbbbbb'),
      'each kept generation, newest first, with its reason and hash: '+box.slice(0,900));
    assert.ok(box.indexOf('aaaaaaaaaaaa')<box.indexOf('bbbbbbbbbbbb'),'the owner\'s order: newest first');
    assert.ok(box.includes('Not held by this workspace in the newest generation')&&box.includes('market-data.duckdb/universe_history'),
      'what the workspace does not hold, named');
    const restoreLabel=box.indexOf('Restore command'),templateStart=box.indexOf('<template>',restoreLabel),templateEnd=box.indexOf('</template>',templateStart);
    assert.ok(box.includes('A command of the client, onto a new or empty directory, with no Host')&&restoreLabel>=0&&templateStart>=0&&templateEnd>templateStart,
      'the page labels the client restore command in its own code template: '+box.slice(box.indexOf('Restore command'),box.indexOf('Restore command')+300));
    const encodedRestoreCommand=box.slice(templateStart+'<template>'.length,templateEnd),restoreCommand=JSON.parse('"'+encodedRestoreCommand+'"');
    const expectedRestoreCommand='alphalattice backup restore --dir <new directory> --workspace-id ws-1 --root "'+root+'" --generation '+gen('a','DATA_UPDATE','2026-09-29T20:00:00+00:00').generation_hash;
    assert.equal(restoreCommand,expectedRestoreCommand,
      'the encoded template decodes to the exact CLI-supported command with quoted root and newest full generation hash');
    assert.ok(box.includes('Back up now:workspace-preview:backup:'),'back up now is offered once the backups are read');
    await w.preview('backup');
    assert.equal(posts.length,0,'the preview sends nothing');
    assert.ok(modal&&String(modal.title).includes('Back up the held state now')&&modal.content.includes('Kept generations')&&modal.content.includes(root),
      'the confirmation names the root and the kept generations: '+String(modal&&modal.content).slice(0,400));
    bodies['/api/workspace/backup']={...bodies['/api/workspace/backup'],status:'BACKED_UP',generation_hash:'c'.repeat(64),
      generations:[gen('c','REQUEST','2026-09-30T08:00:00+00:00'),...bodies['/api/workspace/backup'].generations]};
    await w.commit();await new Promise(r=>setImmediate(r));
    assert.deepEqual(posts.at(-1),['/api/workspace/backup',{}],'the confirmation sends the one backup request, the kept count the owner\'s');
    page=String(w.page()).replace(/\s+/g,' ');
    assert.ok(page.includes('Backed up as generation')&&page.includes('cccccccccccc'),'the answer is said in place, and the new generation listed: '+page.slice(0,400));
    // a refused read holds the button with its reason and says the refusal
    const read=c.Data.read;c.Data.read=async(p)=>{if(p==='/api/workspace/backup')throw Error('research_workspace.manifest_unreadable: no manifest');return read(p);};
    await w.refresh();await new Promise(r=>setImmediate(r));c.Data.read=read;
    page=String(w.page()).replace(/\s+/g,' ');
    assert.ok(page.includes('Backups not read')&&page.includes('research_workspace.manifest_unreadable')&&page.includes('Back up now:workspace-preview:backup:The backups could not be read'),
      'a refused read is said, and back up now is held: '+page.slice(page.indexOf('Backups'),page.indexOf('Backups')+400));
    const unreadable={status:'REFUSED',generation_hash:'d'.repeat(64),failure_code:'workspace_backup.object_changed_while_copied',detail:'This retained backup generation cannot be read.',next_requests:{backups:{operation:'WORKSPACE_BACKUPS'},workspace:{operation:'WORKSPACE_SHOW'}}};
    bodies['/api/workspace/backup']={status:'READ',backup_root:root,last_automatic_attempt:null,generations:[gen('a','DATA_UPDATE','2026-09-29T20:00:00+00:00')],refusals:[unreadable]};
    await w.refresh();await new Promise(r=>setImmediate(r));page=String(w.page()).replace(/\s+/g,' ');
    assert.ok(page.includes('aaaaaaaaaaaa')&&page.includes('workspace_backup.object_changed_while_copied')&&page.includes('This retained backup generation cannot be read.')&&page.includes('LINK:Storage and backups@storage')&&page.includes('LINK:Workspace@overview'),'a partial backup listing preserves its readable generation beside the refused one and both routes: '+page.slice(page.indexOf('REFUSAL('),page.indexOf('REFUSAL(')+700));
    bodies['/api/workspace/backup']={status:'READ',backup_root:root,last_automatic_attempt:null,generations:[],refusals:[unreadable]};
    await w.refresh();await new Promise(r=>setImmediate(r));page=String(w.page()).replace(/\s+/g,' ');
    assert.ok(page.includes('workspace_backup.object_changed_while_copied')&&page.includes('This retained backup generation cannot be read.')&&page.includes('LINK:Storage and backups@storage')&&!page.includes('No backup is kept yet.'),'an all-refused backup listing is not presented as empty: '+page.slice(page.indexOf('REFUSAL('),page.indexOf('REFUSAL(')+650));
    c.table=tableStub;c.tr=trStub;c.panel=panelStub;
  }
  // U13: an owner-composed request goes to the route the session names; one the session does not name is refused by
  // its code and nothing is sent to a guessed path
  {
    c.app.page='issues';await w.refresh();w.changed('case','quarantine');
    const offers=c.Data.offers,sent=posts.length;c.Data.offers=(op)=>op!=='DATA_ISSUE_PREVIEW'&&offers(op);
    await w.preview('issue','case');c.Data.offers=offers;
    assert.equal(posts.length,sent,'no route named, nothing sent');
    assert.equal(toasts.at(-1),'Action refused','the refused press is answered');
    await w.preview('issue','case');assert.deepEqual(posts.at(-1)[0],'/api/workspace/data-issues/preview','the route the session names');
  }
  finish();
})().catch(e=>{console.error(e);process.exitCode=1;});
