// Owner-derived receipt kinds and the runtime boundaries found by the Phase 7 census.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const library=require('./workbench_library.cjs');
const {babelParse,traverse}=require('../../third_party/playwright/node_modules/playwright/lib/transform/babelBundle.js');
const [appDir,serialized]=process.argv.slice(2), kinds=JSON.parse(serialized), words=library.words(appDir);
assert.ok(kinds.length>1,'the owners declare artifact kinds');
const hash='a'.repeat(64);
for(const lang of ['en','zh']) {
 words.I18N.set(lang);
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
  vm.createContext(c);let read;
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
