// The Plugins page renders the Host's list and a run form read from a capability's input schema: a scalar
// parameter, or an array of scalars, is a field of its own and any other parameter goes in the JSON field;
// the run posts what the form held and shows the receipt. `live-plugins.js` and the product's markup helper
// (`html.js`) run as written against a held list. As in the live page, each paint replaces the form's controls
// with empty ones, so a value counts only if the run read it before repainting. No HTTP, model or data work.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const library=require('./workbench_library.cjs');
const finish=library.guard('workbench_plugins');
const root=process.argv[2];
const settled=()=>new Promise(r=>setImmediate(r));
const hostile='x" onmouseover="alert(1)';
const schema={type:'object',required:['window','prices','adjusted'],properties:{
  window:{type:'integer'},prices:{type:'array',items:{type:'number'}},label:{type:'string'},
  adjusted:{type:'boolean'},shape:{type:'object'},[hostile]:{type:'string'}}};
const listed={status:'AVAILABLE',capabilities:[{capability:'fixture.window',plugin:'fixture.window',effect_mode:'STAGING_WRITE',workspace:false,state:'VERIFIED',input_schema:schema}]};
const receipt={invocation_id:'0f0e',idempotency_key:'blake3:'+'a'.repeat(64),status:'ok',output:{values:[1,2]},wall_ms:12,peak_bytes:0};
const posts=[];
let address=new URLSearchParams('page=plugins'), painted='', controls=[];
const unescape=(v)=>v.replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&');
// Each paint makes the form's controls anew and empty, read from the markup it painted.
const repaint=()=>{painted=String(c.P.page());controls=[...painted.matchAll(/<(?:input|select) id="[^"]*" class="ui-field"(?: type="\w+")? data-plugin-param="([^"]*)" data-kind="(\w+)" data-item="(\w*)"/g)]
  .map(([,name,kind,item])=>({dataset:{pluginParam:unescape(name),kind,item},value:''}));};
const typeInto=(values)=>{for(const el of controls) if(el.dataset.pluginParam in values) el.value=values[el.dataset.pluginParam];};
const c={console,URLSearchParams,JSON,Object,Array,String,Number,Math,Promise,Error,Boolean,Set,Map,WeakMap,window:{},
  app:{page:'plugins'},hashParams:()=>address,render:repaint,
  navigate:(page,extra={})=>{address=new URLSearchParams({page,...extra});c.app.page=page;repaint();},
  Data:{read:async(p)=>{assert.equal(p,'/api/plugins');return listed;},post:async(p,body)=>{posts.push([p,body]);return {status:'COMPLETED',receipt};}},
  $:(s)=>s==='#plugin-params'?{value:'{"shape":{"k":1}}'}:null,$$:(s)=>s==='[data-plugin-param]'?controls:[],
  t:(s,vars)=>String(s).replace(/\{(\w+)\}/g,(_,k)=>String(vars?.[k]??'')),
  objectRow:(o,d)=>`<row ${d.key}>${o.name}${d.word}${d.props.join('|')}</row>`,objectHead:(name,meta,actions)=>`<h1>${name}</h1>${meta||''}${actions||''}`,
  panel:(title,note,body)=>`<panel ${title}>${body}</panel>`,kv:(rows)=>rows.map(([k,v])=>k+'='+v).join(';'),codeRef:(title)=>`<code-ref>${title}</code-ref>`,
  codeWords:(s)=>String(s),hashCell:(h)=>String(h),durationText:(ms)=>ms+' ms',notRead:(title,detail)=>`<refused>${title}:${detail}</refused>`,
  explain:()=>'',explainCode:()=>'',btn:(label,action)=>`<${action}>`,typedBtn:(label,action,value,cls,off)=>`<${action}${off?' held':''}>`,
  skeleton:(s)=>'SKELETON('+s+')',emptyState:(s)=>'EMPTY('+s+')'};
// The stubs return trusted fragments, as the product's components do.
const trusted=['objectRow','objectHead','panel','kv','codeRef','codeWords','hashCell','durationText','notRead','explain','explainCode','btn','typedBtn','skeleton','emptyState'];
for(const name of trusted){const stub=c[name];c[name]=(...a)=>c.raw(String(stub(...a)));}
library.context(c, root);
vm.runInContext(fs.readFileSync(path.join(root,'html.js'),'utf8')+';globalThis.html=html;globalThis.raw=raw;',c);
vm.runInContext(fs.readFileSync(path.join(root,'live-plugins.js'),'utf8')+';globalThis.P=LivePlugins;',c);

(async()=>{
  await c.P.ensure(); await settled();
  assert.match(painted,/<row fixture\.window>.*VERIFIED.*STAGING_WRITE\|Installed<\/row>/,'the list names each capability with its state and effect: '+painted);
  c.navigate('plugins',{capability:'fixture.window'});
  const fields=controls.map(f=>[f.dataset.pluginParam,f.dataset.kind,f.dataset.item]);
  assert.deepEqual(fields,[['window','integer',''],['prices','array','number'],['label','string',''],['adjusted','boolean',''],[hostile,'string','']],'each scalar or scalar-array parameter is its own field');
  assert.ok(!painted.includes(hostile) && painted.includes('data-plugin-param="x&quot; onmouseover=&quot;alert(1)"'),'a schema name stays inside its quoted, escaped attribute');
  assert.match(painted,/plugins\.label\.other_params: <span class="mono">shape<\/span>/,'any other parameter goes in the JSON field');
  assert.match(painted,/<span class="mono">window<\/span> \*/,'a required parameter is marked');
  assert.match(painted,/<select id="plugin-adjusted"[^>]*><option value=""><\/option><option value="true">true<\/option><option value="false">false<\/option><\/select>/,'a boolean is true, false or not given');
  typeInto({window:'20',prices:'1.5, 2',adjusted:'false'});
  await c.P.run();
  assert.deepEqual(JSON.parse(JSON.stringify(posts)),[['/api/plugins/run',{capability:'fixture.window',params:{shape:{k:1},window:20,prices:[1.5,2],adjusted:false}}]],'the run posts the form as it was typed, false included, read before the page repainted');
  assert.match(painted,/<panel Receipt>Status=ok;plugins\.label\.idempotency_key=blake3:a{64};Run time=12 ms/,'the receipt is shown: '+painted);
  finish();
})().catch(e=>{console.error(e);process.exitCode=1;});
