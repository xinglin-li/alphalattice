// The review delivery reads owner groups without interpreting rounded weights.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const finish=require('./workbench_library.cjs').guard('review positions');
const c={t:x=>x,said:x=>x,sectionHead:x=>x,emptyState:x=>x,html:(p,...v)=>p.reduce((s,x,i)=>s+x+(v[i]??''),'')};
vm.createContext(c);vm.runInContext(fs.readFileSync(path.join(process.argv[2],'review-positions.js'),'utf8')+';globalThis.reader=ReviewPositions;',c);
const held=[{name:'Held',weight:'0.00%'}],exits=[{name:'Exit',weight:'0.00%'}],calls=[];
const table=(id,rows)=>{calls.push([id,rows]);return id;};
const page=Array.from(c.reader.render({position_rows:held,exit_rows:exits},table,()=>''),String).join('');
assert.deepEqual(calls,[['delivery-positions',held],['delivery-exits',exits]]);
assert(page.includes('Exits (not held)'));assert.equal(c.reader.render({},table,()=>''),'');
assert(String(c.reader.render({position_rows:[] ,exit_rows:[]},table,()=>'' )).includes('No current holdings'));
assert(String(c.reader.render({position_rows:held},table,()=>'' )).includes('Positions and changes'));
assert(String(c.reader.render({position_rows:[],exit_rows:[{name:'Exit',basis:'known-basis'}]},table,()=>'' )).includes('known-basis'));
finish();
