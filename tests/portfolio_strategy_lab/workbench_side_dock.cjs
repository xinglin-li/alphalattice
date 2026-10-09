// Production routes and owner readers; each scenario has an isolated VM.
const {assert,fs,path,vm,library,appDir,source,fixedScript,files,counts,H,ID,clone,makeContext,check,q,resetHistory,nonempty,settle,decode,markupNodes,ancestors,hasClass,assertLocators,localizedContext,offered,followAnchor,press,connect,goalBody,studyBody,portfolioBody,reviewBody,lifecycleContext,lifetimeGate,readCancelled,reviewReadContext,reviewReadEntered,reviewReadCompleted,loadedReport,run} = require('./workbench_route_context.cjs');
run('workbench_side_dock', async () => {
  // Hiding the visual brand never hides the application menu's identity.
  for(const lang of ['en','zh']) for(const mode of ['sidebar','rail']) await check('Named product menu '+mode+'/'+lang,()=>{
    const c=localizedContext(lang), side={innerHTML:'',classList:{toggle(){}},querySelector(){return null;},querySelectorAll(){return [];}};
    const select=c.document.querySelector;c.document.querySelector=selector=>selector==='#side'?side:select(selector);
    vm.runInContext("savePreference('navigation', '"+mode+"');",c);c.probe.Window.renderSide();
    const buttons=markupNodes(side.innerHTML).filter(x=>x.attrs['data-action']==='product-menu');
    assert.equal(buttons.length,1);assert.equal(buttons[0].attrs['aria-label'],'AlphaLattice');
    assert.equal(buttons[0].attrs['aria-haspopup'],'menu');
    counts.named_product_menus=(counts.named_product_menus||0)+1;
  });
  // The Tasks row's numbers: a stop that needs the person is the warning counter, a running Task the neutral one.
  await check('Tasks row counts what needs you apart from what runs',()=>{
    const c=localizedContext('en'), side={innerHTML:'',classList:{toggle(){}},querySelector(){return null;},querySelectorAll(){return [];}};
    const select=c.document.querySelector;c.document.querySelector=selector=>selector==='#side'?side:select(selector);
    vm.runInContext("savePreference('navigation', 'sidebar');",c);
    const tasks=[{task_id:'run',lifecycle:'RUNNING'},{task_id:'queued',lifecycle:'QUEUED'},{task_id:'stop',lifecycle:'BLOCKED'}];
    c.probe.Data.tasks=()=>tasks;c.probe.Data.actionableTasks=()=>tasks.filter(x=>x.task_id==='stop');
    c.probe.Window.renderSide();
    const row=String(side.innerHTML).match(/data-page="tasks"[\s\S]*?<\/a>/)?.[0]||'';
    assert.match(row,/class="side-badge num is-attention"[^>]*>1<\/b>/,'the stop that needs the person is the warning count');
    assert.match(row,/class="side-badge num">2<\/b>/,'the running and queued Tasks are the neutral count');
    counts.task_row_counts=(counts.task_row_counts||0)+1;
  });

});
