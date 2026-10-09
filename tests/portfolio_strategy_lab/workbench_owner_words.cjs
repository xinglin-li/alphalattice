// V659 / TE12: owner declarations drive the vocabulary boundary, not a scene's few values.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const library=require('./workbench_library.cjs'),finish=library.guard('workbench_owner_words');
const appDir=process.argv[2],groups=JSON.parse(process.argv[3]),details=JSON.parse(process.argv[4]),w=library.words(appDir);
const missing=[];
const cause={exception_type:'DataTargetSessionLag',step:'prepare_data',detail:'At 2026-10-07, 0 of 10 listings have a market bar; the baseline needs at least 5. Latest common bar: 2026-09-10.'};
w.I18N.set('en');const englishCause=String(w.causeLine(cause));w.I18N.set('zh');const chineseCause=String(w.causeLine(cause));
assert.ok(chineseCause!==englishCause&&/[\u3400-\u9fff]/.test(chineseCause)&&!chineseCause.includes(cause.detail),'generated stop detail reads through its catalog key');
assert.deepEqual(chineseCause.match(/\d+/g)?.sort(),englishCause.match(/\d+/g)?.sort(),'translated stop keeps every owner date and count');
for(const [owner,codes] of Object.entries(groups)) {
  assert.ok(codes.length,'nonempty owner declaration: '+owner);
  for(const code of codes)if(owner==='actor kinds' ? !w.ACTORS[code] : !w.declaredCodeWord(code))missing.push(`${owner}: ${code}`);
}
assert.deepEqual(missing,[],'every consumed closed owner value has explicit words');
// A declaration without its word fails the same census, rather than quietly escaping it.
for(const code of groups['Task incident codes']) {
  const saved=w.CODE_WORDS[code];delete w.CODE_WORDS[code];
  assert.equal(w.declaredCodeWord(code),'','missing incident word is not humanized: '+code);
  w.CODE_WORDS[code]=saved;
}
for(const lang of ['en','zh']) {
  w.I18N.set(lang);
  for(const [owner,codes] of Object.entries(groups))for(const code of codes) {
    const word=owner==='actor kinds' ? w.actorWords(code) : w.codeWords(code);
    assert.notEqual(word,w.t('Word not declared'),code);
    assert.notEqual(word,w.t('Unidentified'),code);
    assert.ok(word,code);
  }
  for(const code of ['NEW_OWNER_STATE','owner.new_relation','NewOwnerState','u01_prepare_features','u0000_analyze_evidence','u01_UNKNOWN']) {
    assert.equal(w.codeWords(code),w.t('Word not declared'),'unknown is not humanized: '+code);
    assert.equal(w.t(w.stateOf(code).word),w.t('Word not declared'));
  }
  for(const stage of groups['coverage stages']) {
    assert.equal(w.codeWords('u01_'+stage),w.codeWords(stage));
    assert.equal(w.codeWords('u123_'+stage),w.codeWords(stage));
    assert.equal(w.t(w.stageOf('u01_'+stage).word),w.codeWords(stage));
  }
  const reference={reference_id:'synthetic',label:'Owner reference',stage:'PORTFOLIO',request:{operation:'REPORT'}};
  for(const state of groups['Goal reference states'])for(const intent_relation of groups['GoalReference.intent_relation']) {
    const row=String(w.evidenceRow('reference',{state,reference:{...reference,intent_relation}}));
    assert.ok(row.includes(w.codeWords(state))&&row.includes(w.codeWords(intent_relation)),state+'/'+intent_relation);
  }
  const unavailable=String(w.evidenceRow('reference',{state:'UNAVAILABLE',failure_code:'owner.reference_not_read',reference}));
  assert.ok(unavailable.includes(w.codeWords('UNAVAILABLE')),'failure facts do not replace the reference state');
  assert.ok(unavailable.includes('owner.reference_not_read'),'the exact failure remains a fact');
  assert.deepEqual(Array.from(w.I18N.untranslated()),[],'every declared code word has a zh key');
}
// Every word table participates, even a branch this particular owner matrix does not render.
w.I18N.set('zh');
for(const [word,variants] of [['Case',['case']],['Human review required',['human review required']],['gaps',['gap','gap(s)']]]) {
  assert.notEqual(w.t(word),word,'the canonical noun has a Chinese key: '+word);
  for(const variant of variants)assert.equal(w.t(variant),w.t(word),'one declared noun translation: '+variant);
  w.I18N.set('en');for(const variant of variants)assert.equal(w.t(variant),variant,'English spelling stays owned');w.I18N.set('zh');
}
for(const word of [...Object.values(w.CODE_WORDS),...Object.values(w.STATES).map(s=>s.word),...Object.values(w.STAGES).map(s=>s.word),...Object.values(w.ACTORS)])w.t(word);
assert.deepEqual(Array.from(w.I18N.untranslated()),[],'all word declarations are keyed');
const {babelParse,traverse}=require('../../third_party/playwright/node_modules/playwright/lib/transform/babelBundle.js');
for(const file of fs.readdirSync(appDir).filter(f=>f.endsWith('.js'))) {
  const filename=path.join(appDir,file),ast=babelParse(fs.readFileSync(filename,'utf8'),filename,false);
  traverse(ast,{CallExpression(p){assert.notEqual(p.node.callee.name,'wordsOf','retired humanization caller in '+file);}});
}
(async()=>{
  // Both current and resolved incidents pass through the actual Task list and shared builders.
  const codes=groups['Task incident codes'],future='task_runtime.future_reason';
  const incidents=codes.flatMap(code=>['OPEN','RESOLVED'].map(state=>({
    key:code+':'+state,task_id:'synthetic-task',code,state,attempts:[],
    incident:{incident_code:code,detected_at:'2026-10-05T12:00:00Z',user_safe_detail:details[code]}
  })));
  incidents.push({key:'future',task_id:'synthetic-task',state:'RESOLVED',attempts:[],incident:{incident_code:future}});
  let reads=0;
  const c={...library(appDir),...w,console,app:{page:'tasks'},clearTimeout,setTimeout,
    Data:{readShared:async url=>{reads++;if(url==='/api/tasks')return {tasks:[]};
      if(url==='/api/tasks/guardian')return {tasks:[]};if(url==='/api/tasks/incidents')return {incidents};
      throw Error('unexpected owner read '+url);},setTasks(){},runsOf:()=>[],taskRefusals:()=>[]},
    LiveActivity:{refresh(){}},Window:{renderSide(){},inspectorMode:()=>''},patchMain(){},stateMoving:()=>false,
    objectHead:()=>'',btnAttrs:()=>'',icon:()=>'',groupHead:()=>'',pager:()=>'',tile:()=>'',
    emptyState:()=>'',link:()=>'',when:x=>x,pageOf:rows=>({shown:rows,page:0,pages:1})};
  vm.createContext(c);
  vm.runInContext(fs.readFileSync(path.join(appDir,'live-tasks.js'),'utf8')+';globalThis.tasks=LiveTasks;',c);
  await c.tasks.refresh();const initialReads=reads;
  for(const lang of ['en','zh','en']) {
    w.I18N.set(lang);const page=String(c.tasks.page());
    for(const code of codes)for(const state of ['OPEN','RESOLVED']) {
      const key='incident:'+code+':'+state,start=page.indexOf(`data-key="${key}"`);
      assert.ok(start>=0,'actual incident row '+key);
      const next=page.indexOf('data-key="',start+10),row=page.slice(start,next<0?undefined:next);
      assert.ok(row.includes(`data-tip="${code}"`),'exact owner code '+code);
      assert.ok(row.includes(w.codeWords(code)),'declared incident label '+code);
      assert.ok(row.includes(String(w.html`${w.t(details[code])}`)),'owner explanation follows locale '+code);
      assert.ok(row.includes(w.t(state==='OPEN'?'Open':'Resolved')),'incident state '+state);
      assert.ok(row.includes('data-action="task"')&&row.includes('data-value="synthetic-task"'),'exact Task opener');
      if(lang==='zh')assert.notEqual(w.t(details[code]),details[code],'owner explanation has a Chinese key');
    }
    assert.ok(page.includes(`data-tip="${future}"`),'future incident retains its exact owner code');
    assert.ok(page.includes(w.t('Word not declared')),'future incident has no guessed label');
    const uri='semantic://synthetic/owner_future_kind/'+'a'.repeat(64),receipt=String(w.refCell(uri));
    assert.ok(receipt.includes(w.t('Word not declared')),'future receipt has no guessed label');
    assert.ok(receipt.includes(`data-tip="${uri}"`)&&receipt.includes(`data-value="${uri}"`),'future receipt keeps exact tip and copy');
    assert.equal(reads,initialReads,'locale change uses the admitted records without another read');
    assert.deepEqual(Array.from(w.I18N.untranslated()),[],'actual incident words are bilingual');
  }
  finish();
})().catch(e=>{console.error(e);process.exitCode=1;});
