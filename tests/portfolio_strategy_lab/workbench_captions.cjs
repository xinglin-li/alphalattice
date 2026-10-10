const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const library=require('./workbench_library.cjs');
const {babelParse,traverse}=require('../../third_party/playwright/node_modules/playwright/lib/transform/babelBundle.js');
const [appDir,serialized]=process.argv.slice(2), kinds=JSON.parse(serialized).receipt_kinds, words=library.words(appDir);
assert.ok(kinds.length>1,'the owners declare artifact kinds');
const hash='a'.repeat(64);
for(const lang of ['en','zh']) {
 words.I18N.set(lang);
 const superseded=words.stateOf('alternative_evidence_superseded');
 assert.equal(superseded.key,'alternative_evidence_superseded');assert.equal(superseded.word,'Superseded');
 assert.ok(superseded.line && superseded.next);words.t(superseded.line);words.t(superseded.next);
 assert.match(String(words.signed(-0.001,'pp')).replace(/<[^>]+>/g,''),/^0\.00 /);
 assert.match(String(words.signed(-0.00034)),/−0\.00034/);
 for(const kind of kinds)for(const suffix of ['', '.json?spans=synthetic']) {
  const uri=kind==='Reference'?'playpen://workspace-preparation/synthetic-task/verify_inputs':`semantic://synthetic-owner/${kind}/${hash}${suffix}`;
  const html=String(words.refCell(uri)), label=html.match(/class="run-ref-kind">([^<]*)</)[1];
  assert.ok(label,'receipt caption '+kind);
  if(lang==='zh')assert.match(label,/[\u3400-\u9fff]/,'Chinese receipt caption '+kind);
  assert.ok(html.includes(uri),'copy control retains the exact reference');
 }
 assert.equal(words.I18N.untranslated().length,0,'every receipt kind keyed');
 const before=words.I18N.untranslated().length;
 const owner='A synthetic participant wrote these arbitrary words.';
 assert.equal(words.codeWords(owner),owner,'full prose is never humanized as a code');
 assert.equal(words.I18N.untranslated().length,before,'free prose is no missing UI key');
 const fixed='No benchmark provenance is available.';
 assert.equal(words.codeWords(fixed),words.said(fixed),'complete fixed owner sentence follows its original spelling');
 const template=words.t('Not due until {time}',{time:'2099-01-01 12:34'});
 assert.equal(words.t('Not due until 2099-01-01 12:34'),template,'filled disabled reason uses the same template');
 const detail='The requested Task was not found in this workspace.';
 const html=String(words.notRead(words.t('Task not read'),'task_control.task_not_found: '+detail));
 assert.ok(html.includes('task_control.task_not_found'),'refusal retains exact code');
 assert.ok(html.includes(words.said(detail)),'refusal words have their own boundary');
 assert.deepEqual({...words.refusalParts({message:'local_web.transport',body:{refused:'local_web.transport'}})},{code:'local_web.transport',detail:''});
 assert.deepEqual({...words.refusalParts({message:'local_web.session_renewal_failed: synthetic renewal',body:{refused:'old.refusal',detail:'old detail'}})},{code:'local_web.session_renewal_failed',detail:'synthetic renewal'});
 const section=String(words.evidenceRow('section',{title:'Synthetic section',document_handle:'synthetic-document',start_line:3,end_line:7}));
 assert.ok(section.includes(words.t('lines {from}–{to}',{from:3,to:7})),'line range keeps the full quantity sentence');
 if(lang==='zh')assert.ok(!section.includes('行 3'),'number precedes its measure word');
}
// Every opener participates in the shared refresh contract, even one no scene happens to open.
const openers=[];
for(const file of fs.readdirSync(appDir).filter(x=>x.endsWith('.js'))) {
 const src=fs.readFileSync(path.join(appDir,file),'utf8');
 traverse(babelParse(src,file,false),{CallExpression(p){
  const callee=p.node.callee;
  if(callee.type!=='MemberExpression'||callee.object.name!=='Window'||callee.property.name!=='openInspector')return;
  const arg=p.node.arguments[0];assert.equal(arg.type,'ObjectExpression',file+' explicit header contract');
  const header=arg.properties.find(x=>x.key?.name==='readHeader');
  const headerValue=header?.value.type==='Identifier'?p.scope.getBinding(header.value.name)?.path.node.init:header?.value;
  assert.ok(headerValue&&['ArrowFunctionExpression','FunctionExpression'].includes(headerValue.type),file+':'+p.node.loc.start.line+' lazy locale header');
  const c={...library(appDir),t:words.t,h:{ticker:'SYNTH'},id:'synthetic',recorded:{task_id:'synthetic'},b:{fold_index:0},
   row:{id:'synthetic',name:'Factor screening',kind:'Factor screening',summary:'2 factors',raw:{kind:'factor.screening-development',status:'SUCCEEDED'}},
   Data:{history:()=>[{id:'synthetic',name:'Factor screening',kind:'Factor screening',summary:words.t('{n} factors',{n:2}),raw:{kind:'factor.screening-development',status:'SUCCEEDED'}}]},
   words:x=>x,titleOf:()=>words.t('Research experiment'),foldLabel:i=>words.t('Fold {n}',{n:i+1}),codeWords:words.codeWords,reading:{row:{reference:{label:'synthetic',request:{operation:'REPORT'}}}},
   panelHeader:on=>({title:words.t('Tasks'),kind:words.t('Research'),tabs:[{key:'facts',word:words.t('Facts'),on:on==='facts'},{key:'record',word:words.t('Record'),on:on==='record'}]})};
  library.context(c);let read;
  if(file==='live-views.js') {
   // History's header closes over its owner's naming helpers. Invoke the real opener
   // rather than extracting its expression and substituting a second naming rule.
   let opened;
   c.Window={openInspector:value=>{opened=value;}};
   c.html=(parts,...values)=>parts.reduce((text,part,i)=>text+part+(values[i]??''),'');
   c.kv=()=>'';c.btn=()=>'';c.codeRef=()=>''; // body presentation does not decide the header
   vm.runInContext(src,c);
   vm.runInContext('LiveViews.existing("synthetic")',c);
   assert.ok(opened&&typeof opened.readHeader==='function','the actual History opener supplies its callback');
   read=opened.readHeader;
  } else read=vm.runInContext('('+src.slice(headerValue.start,headerValue.end)+')',c);
  words.I18N.set('en');const en=read();words.I18N.set('zh');const zh=read();
  assert.match(zh.kind,/[\u3400-\u9fff]/,file+' kind changes language');
  if(['pages-portfolio.js','live-study.js'].includes(file)&&zh.title==='SYNTH'||zh.title==='synthetic')assert.equal(zh.title,en.title,'exact object name retained');
  else assert.notEqual(zh.title,en.title,file+' composed title changes language');
  if(zh.tabs)assert.ok(zh.tabs.every(tab=>/[\u3400-\u9fff]/.test(tab.word)),file+' tabs change language');
  words.I18N.set('en');assert.equal(read().title,en.title,file+' round-trip title');
  openers.push(file+':'+p.node.loc.start.line);
 }});
}
assert.ok(openers.length,'all inspector openers enumerated');
console.log('receipt kinds '+kinds.length+'; inspector opener contracts '+openers.length+'; caption boundaries passed');

// every fixed storage cap control word reads in chinese.
{
const library=require('./workbench_library.cjs'),complete=library.guard("every_fixed_storage_cap_control_word_reads_in_chinese");
const _appDir=require('node:path').resolve(process.argv[2]),_project=require('node:path').resolve(__dirname,'../..');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const kit=path.join(_project,'third_party/playwright/node_modules/playwright/lib/transform');
const {babelParse,traverse}=require(path.join(kit,'babelBundle.js'));
const app=path.join(_project,'src/alphalattice/interface/local_application/assets',
 'workbench-source/js/app');
const source=fs.readFileSync(path.join(app,'pages-settings.js'),'utf8');
const words=new Set();
traverse(babelParse(source,'pages-settings.js',false),{
 FunctionDeclaration(p){if(p.node.id?.name!=='storageCapRow')return;
 p.traverse({CallExpression(q){if(q.node.callee.name==='t'&&q.node.arguments[0]?.type==='StringLiteral')words.add(q.node.arguments[0].value);}});
}});
assert.ok(words.has('Set')&&words.has('Storage cap'));
const reader=library.words(_appDir);reader.I18N.set('zh');
for(const key of words)assert.notEqual(reader.t(key),key,key);
assert.deepEqual(Array.from(reader.I18N.untranslated()),[]);
complete();
}

// submission words name submissions in page labels and chinese.
{
const library=require('./workbench_library.cjs'),complete=library.guard("submission_words_name_submissions_in_page_labels_and_chinese");
const _appDir=require('node:path').resolve(process.argv[2]),_project=require('node:path').resolve(__dirname,'../..');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict');
const project=_project,app=path.join(project,
 'src/alphalattice/interface/local_application/assets/workbench-source/js/app');
const {babelParse,traverse}=require(path.join(project,
 'third_party/playwright/node_modules/playwright/lib/transform/babelBundle.js'));
const c={window:{}};library.context(c);
vm.runInContext(fs.readFileSync(path.join(app,'../data/zh.js'),'utf8'),c);
const catalog=c.window.ALPHA_ZH;
const submission=/\b(?:re)?submit(?:ted|ting|s)?\b|\b(?:re)?submissions?\b/i;
function checkTranslation(key,zh) {
 if(zh.includes('提交'))assert.ok(submission.test(key),
  'Chinese submission words need a submission: '+key+' -> '+zh);
}
for(const [key,zh] of Object.entries(catalog))checkTranslation(key,zh);
for(const key of ['Admitted task','Revision recorded by','Rebuild from committed vectors'])
 assert.throws(()=>checkTranslation(key,'已提交'),/need a submission/);
checkTranslation('Submitted by','提交者');
// Resolve a row's saved-revision value through its declared local name. This distinguishes
// Goal.submitted_by from Task Control's status.submitted_by without renaming either field.
const property=p=>p.node.computed?p.node.property.value:p.node.property.name;
function isGoal(p,seen=new Set()) {
 if(!p?.node||seen.has(p.node))return false;
 seen=new Set(seen).add(p.node);
 if(p.isIdentifier()) {
  const binding=p.scope.getBinding(p.node.name);
  return binding?.path.isVariableDeclarator()&&isGoal(binding.path.get('init'),seen);
 }
 return (p.isMemberExpression()||p.isOptionalMemberExpression())&&property(p)==='goal';
}
function checkRows(source,filename) {
 const keys=new Set(),ast=babelParse(source,filename,false);
 traverse(ast,{
  StringLiteral(p) {if(Object.hasOwn(catalog,p.node.value)&&submission.test(p.node.value))
   keys.add(p.node.value);},
  ArrayExpression(p) {
   const cells=p.get('elements');if(cells.length<2||!cells[0]?.node)return;
   let submits=false;
   cells[0].traverse({StringLiteral(q){if(submission.test(q.node.value))submits=true;}});
   if(cells[0].isStringLiteral())submits=submission.test(cells[0].node.value);
   if(!submits)return;
   for(const cell of cells.slice(1))if(cell?.node) {
    const checkValue=q=>{
     assert.ok(!(property(q)==='submitted_by'&&isGoal(q.get('object'))),
      filename+': a saved revision recorder cannot be labelled as submission');
    };
    if(cell.isMemberExpression()||cell.isOptionalMemberExpression())checkValue(cell);
    cell.traverse({'MemberExpression|OptionalMemberExpression':checkValue});
   }
  }
 });
 return keys;
}
assert.throws(()=>checkRows("const g=b.goal; kv([[t('Submitted by'),actorWords(g.submitted_by)]]);",
 'revision-mutation.js'),/saved revision recorder/);
checkRows("const sb=S.status.submitted_by; kv([[t('Submitted by'),sb.session]]);",
 'true-task-submission.js');
const keys=new Set();
for(const file of fs.readdirSync(app).filter(f=>f.endsWith('.js')))
 for(const key of checkRows(fs.readFileSync(path.join(app,file),'utf8'),file))keys.add(key);
assert.ok(keys.has('Submission')&&keys.has('Submitted by')&&keys.has('Submit the Analyst answer'),
 'the census reaches Goal completion, Task attribution and external answer submission');
console.log('submission page sentence census complete: '+JSON.stringify([...keys].sort()));
complete();
}

// every chinese quantity template places its numbers before measure words.
{
const library=require('./workbench_library.cjs'),complete=library.guard("every_chinese_quantity_template_places_its_numbers_before_measure_words");
const _appDir=require('node:path').resolve(process.argv[2]);
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=require('node:path').dirname(_appDir),c={window:{},document:{documentElement:{}}};library.context(c);
vm.runInContext(fs.readFileSync(root+'/data/zh.js','utf8'),c);
vm.runInContext(fs.readFileSync(root+'/app/i18n.js','utf8')+';globalThis.words=I18N;',c);
c.words.set('zh');
const measures='个月|毫秒|分钟|小时|个|项|次|条|份|家|位|只|组|场|页|行|列|年|秒|天|折';
const grammar=new RegExp(String.raw`\{(\w+)\}\s*(`+measures+')','g');
const quantityKeys=JSON.parse(process.argv[3]).quantity_keys;
assert.ok(Array.isArray(quantityKeys),'the producer supplies quantity keys');
const counts=new Set(quantityKeys);
const backwards=new RegExp('^(?:'+measures+')[^0-9{}]*\\{\\w+\\}');
for(const bad of ['条交流 {n}','个对象 {n}','行 {from}\u2013{to}'])
 assert.ok(backwards.test(bad),'quantity mutation is rejected');
for(const key of counts)assert.ok(Object.hasOwn(c.window.ALPHA_ZH,key),'count key missing: '+key);
let checked=0;
for(const [key,zh] of Object.entries(c.window.ALPHA_ZH)) {
 const quantities=[...zh.matchAll(grammar)];
 const full=counts.has(key);if(!quantities.length&&!full)continue;
 if(full)assert.ok(!backwards.test(zh),key+' reversed quantity template');
 const slots=[...new Set([...zh.matchAll(/\{(\w+)\}/g)].map(m=>m[1]))];
 for(const base of [0,1,2,1234]) {
  const args=Object.fromEntries(slots.map((slot,i)=>[slot,String(base+i)]));
  const read=c.words.t(key,args);
  for(const match of quantities)assert.ok(
   read.includes(args[match[1]]+match[2])||read.includes(args[match[1]]+' '+match[2]),
   key+' quantity order');
 }
 checked++;
}
assert.ok(checked);assert.equal(c.words.untranslated().length,0);
console.log('all quantity templates: '+checked);
complete();
}

// Every owner refusal uses the catalog renderer without losing its facts.
{
const assert=require('node:assert/strict'),fs=require('node:fs');
const path=require('node:path'),vm=require('node:vm');
const root=require('node:path').resolve(__dirname,'../..'),appRoot=process.argv[2],owners=JSON.parse(process.argv[3]);
const wordReader=require('./workbench_library.cjs').words(appRoot);wordReader.I18N.set('zh');
const rows=JSON.parse(fs.readFileSync(owners.refusal_rows_file,'utf8')).map(({code,detail})=>({code,detail,chinese:wordReader.t(detail),subjects:Object.fromEntries([...new Set(detail.match(/SUBJECT-7f3a(?:-\d+)?/g)||[])].map(marker=>[marker,detail.split(marker).length-1])),literals:[...detail.matchAll(/`([^`]+)`/g)].map(m=>m[1])}));
const appDir=path.join(root,'src/alphalattice/interface/local_application/assets',
  'workbench-source/js/app');
const library=require(path.join(root,'tests/portfolio_strategy_lab/workbench_library.cjs'));
const c={console,window:{},document:{documentElement:{}},app:{},ROUTES:{},...library(appDir)};
library.context(c);
for(const name of ['html.js','../data/zh.js','i18n.js','status.js','icons.js','components.js'])
  vm.runInContext(fs.readFileSync(path.join(appDir,name),'utf8'),c);
const {I18N,t,refusal,CODE_LINES}=vm.runInContext('({I18N,t,refusal,CODE_LINES})',c);
I18N.set('zh');
const decode=s=>s.replace(/<[^>]*>/g,'').replace(/&(amp|lt|gt|quot|#39);/g,
  (_,e)=>({amp:'&',lt:'<',gt:'>',quot:'"','#39':"'"}[e]));
const cause=markup=>decode(markup.match(/<p>([\s\S]*?)<\/p>/)[1]);
function subjects(text,row) {
  for(const [marker,count] of Object.entries(row.subjects))
    assert.equal(text.split(marker).length-1,count,'preserve distinct subject '+marker);
  for(const literal of row.literals)
    assert.ok(text.includes(literal),'preserve literal '+literal);
}
function translated(markup,row) {
  assert.ok(markup.includes('data-status="refused"'),row.code+': keep the refusal state');
  assert.ok(markup.includes('data-tip="'+row.code+'"'),row.code+': keep the exact code');
  assert.ok(cause(markup).startsWith(row.chinese),row.code+': read the actual catalog explanation');
  subjects(cause(markup),row);
}
const visited=new Set();let readings=0;
for(const row of rows) {
  const held=Object.hasOwn(CODE_LINES,row.code),before=CODE_LINES[row.code];
  try {
    for(const listed of [false,true]) {
      // A page listing a code must not replace the owner's actual catalog sentence.
      if(listed)CODE_LINES[row.code]=rows.find(other=>other.detail!==row.detail).detail;
      else delete CODE_LINES[row.code];
      translated(String(refusal({code:row.code,detail:row.detail},'warning',{catalog:true})),row);
      readings++;
    }
    visited.add(row.code);
  } finally {if(held)CODE_LINES[row.code]=before;else delete CODE_LINES[row.code];}
}
const target=rows.find(row=>row.code==='task_control.database_authority_unreadable');
const pair=rows.find(row=>Object.keys(row.subjects).length>1);
assert.ok(pair,'composed refusals must retain distinct identities');
const dropped=Object.keys(pair.subjects)[1];
const pairMarkup=String(refusal({code:pair.code,detail:pair.detail},'warning',{catalog:true}));
assert.throws(()=>subjects(cause(pairMarkup).replaceAll(dropped,''),pair),
  /preserve distinct subject/);
assert.ok(target,'the unreadable Task recovery must stay in the catalog');
const held=Object.hasOwn(CODE_LINES,target.code),before=CODE_LINES[target.code];
try {
  delete CODE_LINES[target.code];
  const targetBody={failure_code:target.code,detail:target.detail};
  const markup=String(refusal(targetBody,'warning',{catalog:true}));
  translated(markup,target);
  const command='alphalattice backup restore --dir <new directory> '+
    '--generation <verified generation hash> --workspace-id <workspace id> --root <backup root>';
  assert.ok(cause(markup).includes(command),'the full recovery command remains literal');
  for(const name of ['new directory','verified generation hash','workspace id','backup root'])
    assert.ok(markup.includes('&lt;'+name+'&gt;'),'metavariables remain escaped text');
  // Independent former-cause control: the same real renderer with the old listing gate
  // fails the exact same translated-guidance assertion, without rewriting product source.
  const oldGate=(body,o)=>refusal(body,'warning',
    {...o,catalog:o.catalog&&Boolean(CODE_LINES[body.failure_code])});
  assert.throws(()=>translated(String(oldGate(targetBody,{catalog:true})),target),
    /read the actual catalog explanation/);
  CODE_LINES[target.code]=rows[0].detail;
  for(const options of [{catalog:false},{}])
    assert.ok(cause(String(refusal({failure_code:target.code,detail:target.detail},'warning',options))).startsWith(target.detail),
      'a participant or owner narrative is not catalog text merely because its words have a key');
  CODE_LINES[target.code]=target.detail;
  translated(String(refusal({failure_code:target.code},'warning',{catalog:true})),target);
} finally {if(held)CODE_LINES[target.code]=before;else delete CODE_LINES[target.code];}
assert.deepEqual([...visited].sort(),[...new Set(rows.map(r=>r.code))].sort());
assert.equal(readings,2*rows.length);assert.deepEqual(Array.from(I18N.untranslated()),[]);
}
