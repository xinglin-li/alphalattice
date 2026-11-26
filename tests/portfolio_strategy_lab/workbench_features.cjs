// A late answer does not take the page (the product lead's regression of 2026-09-25): the Features
// page's PLAN and build are sent to their owners and complete there, but an answer that arrives after
// the reader left -- a link to Data, Back, or Back then Forward to the same composer -- moves nothing:
// no saved PLAN is opened over the page the reader is on, no Task is opened, and a toast says what
// the owner did. Staying on the page, the answer still opens the saved PLAN and the build's Task.
// `live-features.js` runs as written; the owners' answers are held so the reader can move while a
// request is in flight; the router is the product's rule for this page (it ends the visit of the page
// it leaves, `renderPage`), checked against `router.js`. No HTTP, model or data work.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const library=require('./workbench_library.cjs'); // the library's constants and the scripts' parameters, from the source (Q2)
const finish=library.guard('workbench_features');
const root=process.argv[2];
const read=(name)=>fs.readFileSync(path.join(root,name),'utf8');
const H=(n)=>String(n).repeat(64).slice(0,64), B=H('b');
const settled=()=>new Promise(r=>setImmediate(r));
const gates={};
const gate=(route)=>{let release,refuse;const promise=new Promise((r,j)=>{release=r;refuse=j;});gates[route]={promise,release,refuse,entered:false};return gates[route];};
const entered=async(held,label)=>{for(let i=0;i<50 && !held.entered;i++)await settled();assert.ok(held.entered,'the held request was entered: '+label);};
const posts=[],toasts=[],opened=[],moves=[];

// the router's part in this: the address, one history entry per navigation, Back and Forward along
// them, and the paint that ends the visit of the page it leaves
const routerText=read('router.js');
assert.match(routerText,/if \(pageChanged && lastRenderedPage === 'features'\) LiveFeatures\.leave\(\);/,'renderPage ends the Features visit when it paints another page');
let entries=['page=overview'],at=0,last=null;
const place=()=>new URLSearchParams(entries[at]);
function render(){
  if(last!==c.app.page && last==='features') c.F.leave();
  last=c.app.page;
  if(c.app.page==='features') c.F.page();
}
function navigate(page,extra={}){
  const q=new URLSearchParams({page});
  for(const [k,v] of Object.entries(extra)) if(v!=='' && v!=null) q.set(k,String(v));
  entries=entries.slice(0,at+1).concat(q.toString()); at=entries.length-1;
  moves.push(q.toString()); c.app.page=page; render();
}
const back=()=>{at-=1;c.app.page=place().get('page');render();};
const forward=()=>{at+=1;c.app.page=place().get('page');render();};

// the owners: controls and readback answer at once; PLAN and build are held
const controls={template:{input_binding_hash:B,base_revision_hash:H('r')},input_binding_hash:B,definitions:[],
  registered_formulas:[{factor_id:'mom_20',formula_ref:'MOM',required_fields:['provider_adjusted_close'],return_convention:'log'}],
  registered_kernels:[{formula_ref:'MOM'}],source_input_factor_count:3,local_definition_factor_count:0,source_period:{start:'2020-01-02',end:'2026-07-31'}};
const planBody=(hash)=>{
  const document={parent_plan_hash:null,reason:'walk',edits:[{operation:'CREATE',factor_id:'mom_20',specification:{factor_id:'mom_20'}}]};
  return {plan_hash:hash,input_binding_hash:B,yaml:JSON.stringify(document),document,
  candidate_revision:{revision_hash:H('c'),features:[{factor_id:'mom_20'}]},work:{base_compute_factor_ids:['mom_20']},required_fields:[],delta:{},
  trial_baseline:'A completed Alpha study handed off from a Factor study on this input, or the Portfolio study built on one.',
  next_requests:{build:{feature_output:'RAW_VALUES'}}};
};
const Data={
  read:async(p)=>{
    if(p.startsWith('/api/features/controls')) return controls;
    if(p.startsWith('/api/features/readback')) return planBody(new URLSearchParams(p.split('?')[1]).get('feature_plan_hash'));
    throw Error('probe: unexpected read '+p);
  },
  post:(p)=>{posts.push(p);const held=gates[p];if(!held)throw Error('probe: an ungated post '+p);held.entered=true;return held.promise;},
  inputVersion:()=>({id:'factor-development'}),
};
const html=(s,...v)=>s.reduce((a,p,i)=>a+p+(Array.isArray(v[i])?v[i].join(''):v[i]??''),'');
const c={console,URLSearchParams,JSON,Object,Array,String,Number,Math,Promise,Error,Boolean,Set,Map,
  app:{page:'overview'},PAGES:{},Data,hashParams:place,navigate,render,patchMain:()=>{if(c.app.page==='features')c.F.page();},
  LiveTasks:{open:async(id)=>{opened.push(id);navigate('tasks',{task:id});}},LiveViews:{cutoffText:()=>'',inputState:()=>''},
  notify:(m)=>toasts.push(String(m)),download:()=>{},t:(s,vars)=>String(s).replace(/\{(\w+)\}/g,(_,k)=>String(vars?.[k]??'')),
  html,raw:(s)=>String(s),mono:(v)=>String(v),count:(n)=>String(n),countText:(n,one,many)=>String(n===1?one:many).replace('{n}',String(n)),
  icon:()=>'',infoMark:()=>'',kv:(rows)=>rows.map(([k,v])=>k+'='+v).join(';'),panel:(title,note,body)=>'<panel>'+title+body+'</panel>',
  codeRef:(title)=>'<code-ref>'+title+'</code-ref>',refusal:(r)=>'<refusal>'+(r.message||'')+'</refusal>',
  typedBtn:(label,action,value,cls,off)=>`<${action}${off?` held="${off}"`:''}>${label}</${action}>`,btn:(label,action)=>`<${action}>${label}</${action}>`,
  segBtn:(label,action,value)=>`<${action}:${value}>`,subjectChoice:(label,id)=>`<picker id="${id}">`,link:(label)=>String(label),pageWord:(p)=>p,
  objectHead:(name,meta,actions)=>`<h1>${name}</h1>${meta||''}${actions||''}`,stateLine:(s)=>`[${s}]`,emptyState:(s)=>'EMPTY('+s+')',
  skeleton:(shape)=>'SKELETON('+shape+')',sectionHead:(s)=>String(s),table:()=>'<table></table>',tr:()=>'<tr></tr>'};
library.context(c, root);
vm.runInContext(read('live-features.js')+';globalThis.F=LiveFeatures;',c);
const shown=()=>c.F.page(), here=()=>entries[at];

(async()=>{
  // 0. Staying on the page: the saved PLAN opens as its page, and the build opens its Task.
  navigate('features',{input_binding:B}); await settled(); c.F.select('mom_20');
  let held=gate('/api/features/plan'), pending=c.F.plan();
  await entered(held,'PLAN while staying');
  held.release(planBody(H('1'))); await pending; await settled();
  assert.equal(place().get('feature_plan'),H('1'),'staying, the saved PLAN is the page');
  assert.ok(String(shown()).includes('A trial runs against=<span class="owner-text">A completed Alpha study handed off'),'V354: the PLAN says what its trial runs against, in the owner’s words');
  held=gate('/api/features/build'); pending=c.F.build();
  await entered(held,'build while staying');
  held.release({task_id:'task-stay'}); await pending;
  assert.deepEqual(opened,['task-stay'],'staying, the build opens its Task');
  assert.equal(c.app.page,'tasks');

  // 1. PLAN, then a link to Data: the PLAN is saved by its owner; nothing opens over Data.
  navigate('features',{input_binding:B}); await settled(); c.F.select('mom_20');
  held=gate('/api/features/plan'); pending=c.F.plan();
  await entered(held,'PLAN before leaving for Data');
  assert.ok(shown().includes('held="Waiting for the product owner"'),'the PLAN is in flight');
  navigate('data');
  const atData=moves.length;
  held.release(planBody(H('2'))); await pending; await settled();
  assert.equal(c.app.page,'data','the reader stays on Data');
  assert.equal(place().get('page'),'data');
  assert.ok(moves.slice(atData).every(m=>!m.includes('feature_plan')),'no saved PLAN is opened over Data: '+moves.slice(atData).join(' | '));
  assert.ok(toasts.at(-1).includes('saved while you were elsewhere'),'a toast says the PLAN was saved: '+toasts.at(-1));

  // 2. Build, then Back: the Task is admitted by its owner; it is not opened over the page Back reached.
  navigate('features',{feature_plan:H('2')}); await settled();
  assert.ok(shown().includes('<feature-build>'),'the saved PLAN offers its build');
  held=gate('/api/features/build'); pending=c.F.build();
  await entered(held,'build before Back');
  back();
  assert.equal(c.app.page,'data','Back left the page');
  const beforeAnswer=opened.length;
  held.release({task_id:'task-late'}); await pending; await settled();
  assert.equal(opened.length,beforeAnswer,'no Task is opened after Back');
  assert.equal(c.app.page,'data'); assert.equal(place().get('page'),'data');
  assert.ok(toasts.at(-1).includes('admitted while you were elsewhere'),'a toast says the Task was admitted: '+toasts.at(-1));

  // 3. PLAN, Back, then Forward to the same composer before the answer: a new visit. The editor is
  //    free again with the change kept, and the late answer does not replace it with the PLAN.
  navigate('features',{input_binding:B}); await settled(); c.F.select('mom_20');
  const composer=here();
  held=gate('/api/features/plan'); pending=c.F.plan();
  await entered(held,'PLAN before Back and Forward');
  back(); assert.notEqual(c.app.page,'features','Back left the page');
  forward(); assert.equal(here(),composer,'Forward returned to the same composer');
  await settled();
  const returned=shown();
  assert.ok(returned.includes('<feature-plan>') && !returned.includes('held='),'a new visit: the editor is free and the chosen formula kept: '+returned.slice(0,200));
  const beforeLate=moves.length;
  held.release(planBody(H('3'))); await pending; await settled();
  assert.equal(here(),composer,'the composer stays the page');
  assert.equal(moves.length,beforeLate,'nothing navigated');
  assert.ok(toasts.at(-1).includes('saved while you were elsewhere'));

  // 4. A refusal that arrives after the reader left is said by a toast; the page is untouched.
  held=gate('/api/features/plan'); pending=c.F.plan();
  await entered(held,'PLAN before a refusal');
  navigate('data');
  held.refuse(Error('feature.plan_refused')); await pending; await settled();
  assert.equal(c.app.page,'data');
  assert.ok(toasts.at(-1).includes('did not complete while you were elsewhere'),'a toast says it did not complete: '+toasts.at(-1));

  // Every request was sent once and left to its owner: none was withdrawn or repeated.
  assert.deepEqual(posts,['/api/features/plan','/api/features/build','/api/features/plan','/api/features/build','/api/features/plan','/api/features/plan']);
  finish();
})().catch(e=>{console.error(e);process.exitCode=1;});

// feature input prerequisite keeps its elsewhere route.
{
const library=require('./workbench_library.cjs'),complete=library.guard("feature_input_prerequisite_keeps_its_elsewhere_route");
const _appDir=require('node:path').resolve(process.argv[2]),_project=require('node:path').resolve(__dirname,'../..');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {babelParse,traverse}=require(
 require('node:path').join(_project,'third_party/playwright/node_modules/playwright/lib/transform/babelBundle.js'));
const dir=require('node:path').dirname(_appDir)+'/';
function fn(file,name){const source=fs.readFileSync(dir+file,'utf8');let found;
 traverse(babelParse(source,file,false),{FunctionDeclaration(p){
  if(p.node.id?.name===name)found=source.slice(p.node.start,p.node.end);}});
 assert.ok(found,name);return found;}
const c={window:{},document:{documentElement:{}},console,
 S:{operation:'CREATE',controls:null,busy:false,spec:null},operations:{CREATE:'Create'},
 html:(parts,...values)=>parts.map((s,i)=>s+(values[i]??'')).join(''),
 segBtn:()=>'',typedBtn:()=>'',objectHead:()=>'',stateLine:()=>'',
 pageWord:()=> 'New experiment',
 link:(word,page,cls)=>'<a class="'+cls+'" href="#page='+page+'">'+word+'</a>',
 notice:()=>'',skeleton:()=>'<div class="skeleton"></div>'};
library.context(c);vm.runInContext(fs.readFileSync(dir+'data/zh.js','utf8'),c);
vm.runInContext(fs.readFileSync(dir+'app/i18n.js','utf8')+';globalThis.I18N=I18N;',c);
c.t=c.I18N.t;
vm.runInContext(fn('app/components.js','emptyState')+'\n'+fn('app/live-features.js','composeView')+
 ';globalThis.read=composeView;',c);
for(const lang of ['en','zh']){c.I18N.set(lang);const read=c.read({binding:null});
 assert.ok(read.includes('data-empty="elsewhere"'),'input selection happens elsewhere');
 assert.ok(read.includes('href="#page=lab"'),'its existing prerequisite route stays available');
 assert.ok(read.includes(c.t('Select an input')));
 assert.ok(!read.includes('class="button primary"'));
 assert.ok(!c.read({binding:'held-input'}).includes('section-empty'),
  'bound editors retain their own body');}
assert.deepEqual(Array.from(c.I18N.untranslated()),[]);
complete();
}
