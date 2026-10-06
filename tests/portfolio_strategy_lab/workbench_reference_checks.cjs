// TE12 / ST1: completed owner passes only mark nonempty, checked reference sets.
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const library = require('./workbench_library.cjs');
const finish = library.guard('workbench_reference_checks');
const appDir = process.argv[2], words = library.words(appDir);
// A new direct COMPLETE/PASS reader must join the semantic review. Dynamic owner
// fields (source snapshots, model/standing contracts) have separate nonempty bounds;
// this syntax inventory does not claim to trace their whole dataflow.
const {babelParse,traverse}=require('../../third_party/playwright/node_modules/playwright/lib/transform/babelBundle.js');
const passCodes=new Set(['COMPLETE','PASS','PASSED']),readers=new Set();
function ownerOf(p) {
  for(let q=p;q;q=q.parentPath) {
    if(q.isFunctionDeclaration()&&q.node.id)return q.node.id.name;
    if(q.isFunction()&&q.parentPath?.isVariableDeclarator()&&q.parentPath.node.id.type==='Identifier')return q.parentPath.node.id.name;
    if(q.isVariableDeclarator()&&q.node.id.type==='Identifier'&&['ObjectExpression','ArrayExpression'].includes(q.node.init?.type))return q.node.id.name;
  }
  return '<script>';
}
for(const file of fs.readdirSync(appDir).filter(n=>n.endsWith('.js'))) {
  const filename=path.join(appDir,file),ast=babelParse(fs.readFileSync(filename,'utf8'),filename,false);
  const keep=(p,code)=>{if(passCodes.has(code))readers.add(`${file}:${ownerOf(p)}:${code}`);};
  traverse(ast,{StringLiteral(p){keep(p,p.node.value);},Identifier(p){keep(p,p.node.name);}});
}
const meanings={
  'components.js:CODE_WORDS:COMPLETE':'code wording',
  'components.js:goalReferenceIntegrity:COMPLETE':'nonempty reference integrity',
  'live-feature-research.js:contract:PASSED':'required golden examples',
  'live-feature-research.js:held:PASSED':'formula contract admission',
  'live-goals.js:GOAL_STATES:COMPLETE':'record completion',
  'live-goals.js:RANK:COMPLETE':'listing order',
  'live-goals.js:passComplete:COMPLETE':'owner pass ended',
  'live-review.js:CELL_STATE:COMPLETE':'planned delivery',
  'live-review.js:contWords:COMPLETE':'continuation completion',
  'live-review.js:cycle:COMPLETE':'delivery counts',
  'live-review.js:viewFacts:COMPLETE':'delivery facts'
};
assert.deepEqual([...readers].sort(),Object.keys(meanings).sort(),
  'every COMPLETE/PASS reader has a reviewed scope');
const owner = fs.readFileSync(path.join(__dirname, '../../src/alphalattice/control/product_host/composition/goals.py'), 'utf8');
const states = [...new Set([...owner.matchAll(/"(VERIFIED_[A-Z_]+|UNAVAILABLE|SAVED_NOT_VERIFIED)"/g)].map(m => m[1]))];
assert.ok(states.includes('UNAVAILABLE') && states.includes('VERIFIED_READBACK'));
// Enumerate the adapter table itself: a new reader of the reference pass joins this check.
const adapters = Object.entries(words.EVIDENCE).filter(([, f]) => /evidence_verification|goalReferenceIntegrity/.test(String(f)));
assert.ok(adapters.length, 'reference-pass adapters exist');
const variants = [[], ...states.map(state => [{state}]),
  [{state:'VERIFIED_READBACK'}, {state:'UNAVAILABLE'}], [{state:'FUTURE_OWNER_STATE'}]];
for (const lang of ['en', 'zh']) {
  words.I18N.set(lang);
  for (const evidence_verification of ['NOT_PERFORMED', 'COMPLETE']) for (const references of variants) {
    const body = {references, evidence_verification};
    for (const [name, read] of [['Goal', words.goalReferenceIntegrity], ...adapters]) {
      const shown = read(body);
      const positive = evidence_verification === 'COMPLETE' && references.length > 0 && references.every(r => r.state.startsWith('VERIFIED_'));
      assert.equal(shown.state === 'verified', positive, `${lang}/${name}/${evidence_verification}/${JSON.stringify(references)}`);
      if (!references.length) assert.equal(shown.word, words.t('Nothing to check yet'));
      if (positive) {
        assert.equal(shown.word, words.t('Reference integrity checked'));
        assert.equal(shown.limit, words.t('Reference integrity only; not scientific approval.'));
      }
    }
  }
  const reference={reference_id:'synthetic',label:'Synthetic reference',stage:'PORTFOLIO',request:{operation:'REPORT'}};
  const scoped={VERIFIED_READBACK:'Owner readback checked',VERIFIED_TASK_STATE:'Task state checked',VERIFIED_REFUSAL:'Refusal checked'};
  for(const state of states.filter(s=>s.startsWith('VERIFIED_'))) {
    assert.ok(scoped[state], 'every owner-declared positive row names what it checks: '+state);
    const row=String(words.evidenceRow('reference',{state,reference}));
    assert.ok(row.includes(words.t(scoped[state])), 'scoped reference row: '+state);
    if(lang==='zh')assert.match(words.t(scoped[state]),/[\u3400-\u9fff]/);
  }
  const contract=fs.readFileSync(path.join(__dirname,'../../src/alphalattice/interface/local_application/goals.py'),'utf8');
  const declared=contract.match(/intent_relation: Literal\[([\s\S]*?)\]/);
  assert.ok(declared,'the owner declares reference registration relations');
  for(const [,intent_relation] of declared[1].matchAll(/"([A-Z_]+)"/g)) {
    const row=String(words.evidenceRow('reference',{state:'VERIFIED_READBACK',reference:{...reference,intent_relation}}));
    assert.ok(!row.includes(intent_relation),'relation is product words');
    if(lang==='zh')assert.ok(!/Question recorded before task admission|Post hoc|No execution time proof/.test(row),'registration relation is Chinese');
  }
  for(const [body,zero] of [[{reference_count:0},true],[{reference_count:1},false],[{},false]])
    for(const [name,read] of [['Goal',words.goalReferenceIntegrity],...adapters]) {
      const shown=read({...body,evidence_verification:'COMPLETE'});
      assert.notEqual(shown.state,'verified',name+' count-only is no verification');
      assert.equal(shown.word,words.t(zero?'Nothing to check yet':'Reference integrity not checked'));
    }
  assert.equal(words.I18N.untranslated().length, 0, 'every reference word has a Chinese key');
}
finish();
