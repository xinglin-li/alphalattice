const assert = require('node:assert/strict');
const library = require('./workbench_library.cjs');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const appDir = process.argv[2];
const app = {page:'portfolio',tab:'holdings',session:'2024-08-12',holdingsSort:'listing-asc',holdingsQuery:'',holdingsPage:0,chart:'indexed'};
const raw = {
  subject:{task_id:'book',title:'Book',input_date:'2024-08-01',cost_per_side:'5',support:{start:'2024-08-05',end:'2024-08-12'}},
  position:{holdingCount:113},
  holdings_basis:{basis:'HISTORICAL_REPLAY',comparison_basis:'PRECEDING_FORMATION',preceding_formation_session:'2024-07-29',formation_session:'2024-08-12'},
  holdings:[{listing_id:'A',ticker:'AAA',name:'Alpha',target:2,executed:99,current_weight:0.015,preceding_weight:0.01,weight_change:0.005,disposition:'INCREASED',sector:null,mapped:null},...Array.from({length:128},(_,i)=>({listing_id:`exit-${i}`,ticker:`EXIT${i}`,current_weight:0,weight_change:-0.01,sector:null,mapped:null}))],
  forward_holdings:{status:'RECORDED_FORWARD_HOLDINGS',available:true,basis:'CONDITIONAL_ESTIMATE',comparison_basis:'FORMATION_CLOSE_ESTIMATE',formation_session:'2024-08-19',entry_session:'2024-08-20',holding_count:1,holdings:[{listing_id:'A',ticker:'AAA',name:'Alpha',target:null,executed:null,current_weight:0.025,preceding_weight:0.02,weight_change:0.005,disposition:null,sector:null,mapped:null}]},
  forward_performance:{available:true,status:'INSUFFICIENT_REALIZED_OBSERVATIONS',cost_bps_per_side:5,selected_window_metrics:{},selected_window_metric_provenance:{observation_count:0,selected_start:null,selected_end:null}},
  metrics:{},series:[],sessions:['2024-08-05','2024-08-12'],notice:'Owner readback',
};
const Data = {
  raw:()=>raw, subject:()=>raw.subject, sessions:()=>raw.sessions, history:()=>[], notice:()=>raw.notice,
  holdings:()=>raw.holdings, series:()=>raw.series, metrics:()=>({}), performanceMode:()=>mode,
  rollingPerformance:()=>null, forwardPerformance:()=>raw.forward_performance,
  portfolioReading:()=>false, portfolioSession:()=>app.session,
};
let mode='historical', headDetails;
const html = (parts,...values) => parts.reduce((out,part,index)=>out+part+(index<values.length?String(values[index] ?? ''):''),'');
const t = (word,args={}) => String(word).replace(/\{([^}]+)\}/g,(_,key)=>args[key] ?? '');
const button = (label,action,value='',cls='') => `<button class="${cls}" data-action="${action}" data-value="${value}">${label}</button>`;
const tabStrip = (label,items,cls='') => `<div class="tabs ${cls}"><div class="tabs-list" role="tablist" aria-label="${label}">${items.map(item=>`<button role="tab" aria-selected="${item.on?'true':'false'}" data-action="${item.action}" data-value="${item.value}">${item.word}${item.count||''}</button>`).join('')}</div></div>`;
const table = (headers,rows,_note,options={}) => `<table><thead><tr>${headers.map(h=>`<th>${h.label}</th>`).join('')}</tr></thead><tbody>${rows}</tbody><tfoot>${options.foot||''}</tfoot></table>`;
const c = {
  app,Data,raw,html,t,tabStrip,PAGES:{},ACTIONS:{},ON_INPUT:{},ON_CHANGE:{},
  queueMicrotask:()=>{},window:{ALPHA_PRODUCT:true},
  btn:(label,action,value,cls)=>button(label,action,value,cls),
  btnAttrs:(label,action,value,cls)=>button(label,action,value,cls),
  segBtn:(label,action,value,pressed)=>button(label,action,value,pressed?'on':'off'),
  icon:(name)=>`<i>${name}</i>`, picker:(id)=>`<select id="${id}"></select>`,
  objectHead:(name,_meta,_actions,_state,_tools,details)=>{headDetails=details;return `<header class="object-head">${name}</header>`;},
  badge:()=>'', codeWords:(value)=>value, link:()=>'',
  LiveViews:{bookWords:()=>'',bookTools:()=>'',researchTiming:()=>'',portfolioDetails:()=>'<div>Diagnostics</div>',metricAbsence:()=>null},
  LiveActivation:{ensure:()=>{},panel:()=>''},
  Data,
  displayState:(_name)=>({order:'listing-asc',group:'none',props:{}}), setDisplay:()=>{}, displayOptions:()=>'',
  searchBar:()=>'<div class="search"></div>', detailSplit:(body)=>body,
  table,pager:()=>'', pageOf:(values)=>({shown:values,start:0,page:0,pages:1}), stateLine:()=>'',
  num:(value,kind)=>kind==='percent'&&value!=null?`${Number(value).toFixed(1)}%`:String(value??''),
  signed:(value,kind)=>`${value>0?'+':value<0?'−':''}${Math.abs(Number(value)).toFixed(2)} ${kind==='pp'?'pp':''}`.trim(),
  fmt:(value)=>String(value??''), count:(value)=>String(value??''), dateRange:(a,b)=>a&&b?`${a} — ${b}`:'', when:()=>'',
  stat:(label,value,note)=>`<div class="stat"><span>${label}</span><b>${value}</b><small>${note}</small></div>`,
  hint:(term,note)=>`${term} ${note}`, emptyState:(line,way='')=>`<div class="section-empty" data-empty="elsewhere"><p>${line}</p>${way}</div>`,
  joinMarkup:(parts,separator='')=>parts.join(separator), sectionHead:()=>'', kv:()=>'',
  LEGEND:{fold:8,shown:6},
  figureBox:(title,drawing,options)=>`<section class="${options.cls}" data-box="figure">${title}${options.info}${drawing}</section>`,
  meter:(options)=>`<div class="composition">${options.slices.map(([label,value])=>`${label}:${value}`).join('|')}</div>`, infoMark:(note)=>`<span class="source">${note}</span>`,
  METRIC_STATS:undefined,
  standingPanel:()=>'',
  stackSlot:(id,content)=>`<div id="${id}">${content}</div>`,
  Inspect:{observationDock:()=>'',window:()=>[0,0],presetsFor:()=>[],setWindow:()=>{}},
  chartHead:()=>'<div class="chart-headline"></div>', chart:()=>'<svg></svg>',
};
library.context(c);
vm.runInContext(fs.readFileSync(path.join(appDir,'pages-portfolio.js'),'utf8'),c,{filename:'pages-portfolio.js'});

const render = () => c.PAGES.portfolio();
let markup=render();
assert.equal(headDetails.subject,undefined,'the book head does not own the holdings date picker');
assert.match(markup,/Holdings113/,'the Historical badge uses the owner current holding count, not the 129-row replay union');
assert.equal(raw.holdings.length,129,'the fixture contains exit rows in addition to the owner holding count');
assert.ok(markup.indexOf('class="tabs portfolio-tabs"') < markup.indexOf('class="holdings-toolbar"'));
assert.ok(markup.indexOf('class="holdings-toolbar"') < markup.indexOf('id="portfolioDetail"'));
assert.match(markup,/Replay weight/);
assert.match(markup,/Change \(pp\)/);
assert.doesNotMatch(markup,/Execution weight|<th>Target weight/);
assert.match(markup,/1\.5%/);
assert.match(markup,/\+0\.50 pp/);
assert.match(markup,/Historical replay weights at 2024-08-12/);
assert.match(markup,/not actual trades/);
assert.ok(markup.indexOf('class="tabs performance-tabs"') < markup.indexOf('data-box="figure"'),'Historical source tabs sit outside the chart figure');
assert.doesNotMatch(markup.slice(markup.indexOf('class="tabs performance-tabs"'), markup.indexOf('data-box="figure"')),/data-box=/);

mode='forward';
markup=render();
assert.match(markup,/class="tabs performance-tabs"/);
assert.match(markup,/aria-selected="true" data-action="portfolio-performance" data-value="forward"/);
assert.match(markup,/Proposed weight/);
assert.match(markup,/2\.5%/);
assert.doesNotMatch(markup,/1\.5%|Execution weight|<th>Target weight/);
assert.match(markup,/Conditional estimated weights at 2024-08-19; they are not actual fills/);
assert.match(markup,/Change compares formation-close estimates/);
const emptyPerformance=markup.slice(markup.indexOf('<div id="portfolioPerformance">'),markup.indexOf('<div id="bookActivation">'));
assert.match(emptyPerformance,/data-empty="elsewhere"/,'zero observations retain an ordinary owner absence state after the source tabs');
assert.match(emptyPerformance,/Fewer than two settled outcomes were recorded/);
assert.match(emptyPerformance,/0 observations/,'the scope retains the owner-reported observation count');
assert.doesNotMatch(emptyPerformance,/Underlying values|data-box=|metric-basis|performance-surface|chart-under/,'zero observations do not fabricate a performance figure or its controls');
assert.doesNotMatch(markup,/chart-toolbar|chart-headline|<svg|stat-strip rail five/);
assert.doesNotMatch(markup,/Realized window\s+—/);
assert.doesNotMatch(markup,/data-action="chart-data"/);
assert.doesNotMatch(markup,/class="chart-under"><span>/);
assert.doesNotMatch(markup,/holdingsSession|Holdings session/,'Forward does not offer the historical holdings-date picker');
assert.match(markup,/Formation 2024-08-19 · Entry 2024-08-20/,'Forward shows its own read-only formation and entry dates');
assert.match(markup,/Holdings1/,'Forward badge uses the owner holding count');
assert.doesNotMatch(markup,/performance-surface/,'the zero-observation Forward view has no performance box wrapper');

// Both modes use the existing figure/meter builder, with their own owner weights, sealed
// classification source and dates. Changing the selected mode never borrows Historical sectors.
raw.holdings[0].sector='Historical only';
raw.sectors={status:'UNAVAILABLE'};
raw.forward_holdings.holdings[0].sector='Technology';
raw.forward_holdings.holdings.push(
  {listing_id:'B',ticker:'BBB',current_weight:0.015,sector:'Technology'},
  {listing_id:'C',ticker:'CCC',current_weight:0.05,sector:'Health Care'},
);
raw.forward_holdings.holding_count=3;
raw.forward_holdings.sectors={status:'BOOK_EXECUTION',sector_revision:'forward-sealed-sector'};
markup=render();
assert.match(markup,/class="sector-exposure" data-box="figure"/);
assert.match(markup,/Technology:4\|Health Care:5|Health Care:5\|Technology:4/,'the Forward composition sums Forward owner weights by sector');
assert.match(markup,/book weight at 2024-08-19 · 2 sectors/,'a proposal uses its own formation date');
assert.match(markup,/Sector map sealed with its Panel/,'Forward uses its own sealed sector source status');
assert.doesNotMatch(markup,/Historical only|Sector source not resolved/);
assert.ok(markup.indexOf('class="sector-exposure"') < markup.indexOf('class="holdings-box"'),'sector exposure is above Forward holdings');
assert.equal((markup.match(/class="sector-exposure"/g)||[]).length,1,'Forward has one sector figure');
mode='historical';
markup=render();
assert.match(markup,/Historical only:1\.5/);
assert.match(markup,/book weight at 2024-08-12 · 1 sectors/);
assert.match(markup,/Sector source not resolved/,'Historical uses its own sector source status');
assert.doesNotMatch(markup,/Technology:4|Health Care:5/);
mode='forward';
raw.forward_holdings.basis='OBSERVED_RESEARCH_ENTRY';
raw.forward_holdings.comparison_basis='OBSERVED_PRETRADE_WEIGHTS';
raw.forward_holdings.entry_session='2024-08-20';
markup=render();
assert.match(markup,/Research-entry weight/);
assert.match(markup,/Observed research-entry weights come from daily-bar QA, not actual fills/);
assert.match(markup,/Change compares the observed pretrade weights/);
assert.match(markup,/book weight at 2024-08-20 · 2 sectors/,'observed entry weights use the publication entry date');
for(const holding of raw.forward_holdings.holdings)holding.sector=null;
raw.forward_holdings.sectors={status:'UNAVAILABLE'};
markup=render();
assert.doesNotMatch(markup,/class="sector-exposure"|Historical only/,'an unclassified Forward publication cannot borrow Historical sectors');

// A realized publication with observations retains the normal chart and its reading controls.
raw.series=[{date:'2024-08-20',daily:0.1},{date:'2024-08-21',daily:0.2}];
raw.forward_performance.selected_window_metric_provenance={observation_count:2,selected_start:'2024-08-20',selected_end:'2024-08-21'};
markup=render();
assert.match(markup,/Forward realized performance|chart-toolbar|id="performanceChart"/);
assert.match(markup,/2 observations/);
assert.match(markup,/data-action="chart-data"/);

// U158/F16: use the production Portfolio.page -> performancePanel producer and the
// actual empty-state, bounded-code and en/zh readers. These are markup conditions;
// the parent browser walk owns viewport geometry and painted theme verification.
const portfolioSource=fs.readFileSync(path.join(appDir,'pages-portfolio.js'),'utf8');
const componentsSource=fs.readFileSync(path.join(appDir,'components.js'),'utf8');
const viewsSource=fs.readFileSync(path.join(appDir,'live-views.js'),'utf8');
const strategyFactsFrom=viewsSource.indexOf('  function studyFacts('),strategyFactsTo=viewsSource.indexOf('  /* A history date',strategyFactsFrom);
assert.ok(strategyFactsFrom>=0&&strategyFactsTo>strategyFactsFrom,'the actual shared study naming reader remains available');
const ownerLabels=Object.fromEntries(JSON.parse(fs.readFileSync(path.resolve(appDir,'../../../..','labels.json'),'utf8')).labels.map(v=>[v.id,v]));
const emptyFrom=componentsSource.indexOf('function emptyState(');
const emptyTo=componentsSource.indexOf('/* A refusal or a failure',emptyFrom);
const wordsFrom=componentsSource.indexOf('const labelOf =');
const wordsTo=componentsSource.indexOf('/* A feature by its name',wordsFrom);
const codedSource=componentsSource.match(/^const coded = .*$/m)?.[0];
assert.ok(emptyFrom>=0 && emptyTo>emptyFrom && wordsFrom>=0 && wordsTo>wordsFrom && codedSource,'the actual display helpers remain available');
const headFrom=componentsSource.indexOf('function contextFacts(');
const headTo=componentsSource.indexOf('/* Attributes a builder is handed',headFrom);
assert.ok(headFrom>=0 && headTo>headFrom,'the actual object-head facts builder remains available');
const inputHeads=process.argv[3] ? JSON.parse(process.argv[3]).portfolio_heads : null;
assert.ok(Array.isArray(inputHeads)&&inputHeads.length===3,'the shared producer supplies every owner input-head case');
const oldBranch="    if (noForwardObservations) return html`<div id=\"portfolioPerformance\">${controls}</div>`;";
const branchAnchor='    const noForwardObservations=forward && Data.series().length===0;';
assert.ok(portfolioSource.includes(branchAnchor),'the negative control reinserts the exact former tabs-only branch');
const oldPortfolioSource=portfolioSource.replace(branchAnchor,branchAnchor+'\n'+oldBranch);
const historicalRows=[{date:'2024-08-05',daily:99},{date:'2024-08-12',daily:98}];
function performanceReader(source,locale,theme,width) {
  const owner=JSON.parse(JSON.stringify(raw));
  let selectedMode='forward';
  const view={...c,app:{...app},PAGES:{},ACTIONS:{},ON_INPUT:{},ON_CHANGE:{},
    innerWidth:width,window:{ALPHA_PRODUCT:true},document:{documentElement:{lang:'en'},body:{dataset:{theme}}},
    raw:(markup)=>markup,railTools:()=>'',Inspect:{...c.Inspect,hasFacts:()=>false},
    CODE_WORDS:{},STATES:{},stageWord:()=>null,LABELS:ownerLabels,
    LiveViews:{...c.LiveViews},
    Data:{...Data,raw:()=>owner,subject:()=>owner.subject,forwardPerformance:()=>owner.forward_performance,
      performanceMode:()=>selectedMode,series:()=>selectedMode==='forward' ? owner.forward_performance?.available===true ? owner.forward_performance.series || [] : [] : historicalRows},
  };
  library.context(view);
  vm.runInContext(fs.readFileSync(path.join(appDir,'..','data','zh.js'),'utf8'),view,{filename:'zh.js'});
  vm.runInContext(fs.readFileSync(path.join(appDir,'i18n.js'),'utf8'),view,{filename:'i18n.js'});
  vm.runInContext(componentsSource.slice(emptyFrom,emptyTo)+componentsSource.slice(wordsFrom,wordsTo)+codedSource+componentsSource.slice(headFrom,headTo),view,{filename:'components-performance-readers.js'});
  vm.runInContext(viewsSource.slice(strategyFactsFrom,strategyFactsTo)+'\nglobalThis.strategyFacts=studyFacts;',view,{filename:'shared-study-naming.js'});
  view.LiveViews.studyFacts=view.strategyFacts;
  vm.runInContext('globalThis.translate=t;globalThis.chooseLocale=I18N.set;',view);
  view.chooseLocale(locale);
  vm.runInContext(source,view,{filename:'pages-portfolio.js'});
  const fragment=()=>{
    const page=String(view.PAGES.portfolio()),start=page.indexOf('<div id="portfolioPerformance">'),end=page.indexOf('<div id="bookActivation">');
    assert.ok(start>=0 && end>start,'the actual performance producer precedes Activation');
    return page.slice(start,end);
  };
  return {owner,fragment,page:()=>String(view.PAGES.portfolio()),studyFacts:view.strategyFacts,translate:view.translate,setMode:(next)=>{selectedMode=next;}};
}
const zeroOwner=()=>({available:true,status:'RECORDED_REALIZED_FORWARD_WINDOW',cost_bps_per_side:'5',selected_window_metrics:{},
  selected_window_metric_provenance:{status:'INSUFFICIENT_REALIZED_OBSERVATIONS',selected_start:null,selected_end:null,observation_count:0,observed_through:'2024-08-21',published_at:'2026-10-05T12:00:00Z'},series:[]});
function assertEmptyPerformance(reader,reason) {
  const fragment=reader.fragment();
  assert.match(fragment,/aria-selected="true" data-action="portfolio-performance" data-value="forward"/);
  assert.match(fragment,/data-empty="elsewhere"/,'the selected zero-observation tab explains its owner absence');
  assert.ok(fragment.includes(reader.translate(reason).replace(/[.。]\s*$/,'')),'the exact typed owner absence reads in the selected language');
  assert.doesNotMatch(fragment,/performanceChart|chart-toolbar|chart-data|chart-under|stat-strip|data-box=/,'no chart, metrics or chart controls are fabricated');
  assert.doesNotMatch(fragment,/2024-08-05|2024-08-12|99|98/,'an empty Forward window cannot borrow Historical observations');
  return fragment;
}
let performanceConditions=0;
for(const locale of ['en','zh-CN'])for(const theme of ['light','dark'])for(const width of [1470,900]) {
  const reader=performanceReader(portfolioSource,locale,theme,width),translate=reader.translate;
  const originalSubject=reader.owner.subject;
  const installedLabel=Object.values(ownerLabels).find(v=>v.kind==='strategy');
  for(const packageId of ['QA_UNLABELLED_STRATEGY',installedLabel.id]) {
    reader.owner.subject={...originalSubject,source_kind:'INSTALLED_RESULT',title:packageId};
    const page=reader.page(),head=page.slice(0,page.indexOf('<div id="portfolioPerformance">'));
    const word=packageId===installedLabel.id ? locale==='zh-CN' ? installedLabel.title_zh : installedLabel.title : translate('Installed strategy result');
    assert.ok(head.includes(word),'the installed head uses the shared declared title or naming absence in either language');
    assert.ok(!head.includes(translate('Word not declared')),'an unknown package code is not the installed book title');
    assert.equal(reader.owner.subject.title,packageId,'rendering preserves the exact owner package code for its recorded facts');
    assert.equal(reader.owner.subject.task_id,originalSubject.task_id,'naming preserves the exact saved object');
    const review=reader.studyFacts(null,{kind:'CRO_REVIEW',strategy_package_id:packageId});
    assert.equal(review.words,word,'a published review retains the same declared or unnamed installed-book title');
    assert.equal(review.summary,packageId,'the review retains the exact owner package code as metadata');
  }
  const authoredReview=reader.studyFacts(null,{kind:'CRO_REVIEW',book:{experiment_task_id:'authored-book'}});
  assert.equal(authoredReview.words,'','a review with no strategy package retains its existing naming fallback');
  assert.equal(authoredReview.summary,'','an authored review does not invent an installed strategy identifier');
  reader.owner.subject={...originalSubject,title:null};
  assert.ok(reader.page().includes(translate('Portfolio study')),'an authored unnamed book retains its existing title fallback');
  reader.owner.subject=originalSubject;
  if(inputHeads) {
    // The real installed-book projection is read through Portfolio.page and its shared
    // head builder. Unknown input names stay absent; a recorded name keeps its spelling.
    for(const [index,subject] of inputHeads.entries()) {
      reader.owner.subject=subject;
      const page=reader.page(),head=page.slice(0,page.indexOf('<div id="portfolioPerformance">'));
      const inputKey=`<span class="context-key">${translate('Input')}</span>`;
      if(index===0) {
        assert.doesNotMatch(head,/Input not recorded|Input Input/,'an absent owner input is not a fabricated name or repeated placeholder');
        assert.ok(!head.includes(inputKey),'an unknown input leaves no labelled placeholder in either language');
      } else {
        assert.ok(head.includes(inputKey),'a known input retains its translated property label');
        assert.ok(head.includes(index===1 ? 'recorded-input' : 'Bound historical input'),'the owner input name or bound-input distinction stays intact');
      }
    }
    reader.owner.subject=JSON.parse(JSON.stringify(raw.subject));
  }
  reader.owner.forward_performance=zeroOwner();
  let fragment=assertEmptyPerformance(reader,'Fewer than two settled outcomes were recorded');
  assert.ok(fragment.includes(translate('{n} observations',{n:'0'})));
  assert.ok(fragment.includes(translate('Net of {c} bps per side',{c:'5'})));
  assert.ok(fragment.includes(translate('Observed through {date}; published {published}',{date:'2024-08-21',published:'2026-10-05T12:00:00Z'})),'the empty state retains the actual owner dates');
  assert.ok(fragment.indexOf('class="tabs performance-tabs"')<fragment.indexOf('data-empty="elsewhere"'));
  assert.doesNotMatch(fragment,/Realized window\s+—|已实现窗口\s+—/,'missing window endpoints leave no invented range');

  reader.owner.forward_performance={available:false,status:'NO_REALIZED_FORWARD_PUBLICATION'};
  fragment=assertEmptyPerformance(reader,'No realized forward publication is available for this book and cost lane.');
  assert.doesNotMatch(fragment,/2024-|2026-|\d+ observations|\d+ 个观测/,'a missing publication invents neither dates nor a count');

  reader.owner.forward_performance=zeroOwner();
  reader.owner.forward_performance.selected_window_metric_provenance.status='UNDECLARED_REALIZED_OWNER_STATUS';
  fragment=reader.fragment();
  assert.match(fragment,/data-empty="elsewhere"/);
  assert.match(fragment,/data-tip="UNDECLARED_REALIZED_OWNER_STATUS"/,'an unknown status retains the exact owner code');
  assert.ok(fragment.includes(translate('Word not declared')),'the unknown code uses the existing bounded display in either language');
  assert.doesNotMatch(fragment,/Fewer than two settled outcomes were recorded|记录的已结算结果少于两个|performanceChart|chart-toolbar|stat-strip/);

  reader.owner.forward_performance={...zeroOwner(),selected_window_metric_provenance:{status:'DERIVED_FROM_SEALED_REALIZED_NET_RETURN_PATH',selected_start:'2024-08-20',selected_end:'2024-08-21',observation_count:2},series:[{date:'2024-08-20',daily:0.1},{date:'2024-08-21',daily:0.2}]};
  fragment=reader.fragment();
  assert.match(fragment,/id="performanceChart"/);
  assert.match(fragment,/data-action="chart-data"/);
  assert.ok(fragment.includes(translate('Realized window {from} — {to} · {n} observations · {c} bps per side',{from:'2024-08-20',to:'2024-08-21',n:'2',c:'5'})));
  assert.doesNotMatch(fragment,/data-empty=|2024-08-05|2024-08-12/);

  reader.setMode('historical');
  fragment=reader.fragment();
  assert.match(fragment,/aria-selected="true" data-action="portfolio-performance" data-value="historical"/);
  assert.match(fragment,/stat-strip rail five|id="performanceChart"/);
  assert.ok(fragment.includes(translate('Whole report {from} — {to} · net of {c} bps per side',{from:'2024-08-05',to:'2024-08-12',c:'5'})));
  assert.doesNotMatch(fragment,/data-empty=|2024-08-20|2024-08-21/,'Historical retains its own axis after Forward selection');

  const former=performanceReader(oldPortfolioSource,locale,theme,width);
  former.owner.forward_performance=zeroOwner();
  assert.throws(()=>assertEmptyPerformance(former,'Fewer than two settled outcomes were recorded'),{code:'ERR_ASSERTION'},'the former controls-only branch fails the same absence assertion');
  performanceConditions++;
}
console.log(`portfolio empty performance: ${performanceConditions} locale/theme/width markup conditions with old-branch negative controls`);
console.log('portfolio layout and holdings basis assertions passed');

// Exercise the production open/read path: only a same-book, changed-session Historical request
// may use scope=holdings, and the retained rolling/forward facts require exact saved-result identity.
async function transportAssertions() {
  const requests=[];
  const overlays={rolling_performance:{head_hash:'head-a'},forward_holdings:{status:'forward-a'}};
  const body=(session,resultHash,extra={})=>({subject:{task_id:'book',session,result_hash:resultHash},...extra});
  const queue=[body('2024-08-12','result-a',overlays),body('2024-08-19','result-a'),body('2024-08-26','result-b'),body('2024-08-26','result-b',{rolling_performance:{head_hash:'head-b'}}),body('2024-08-26','result-b',{forward_performance:{available:false}}),body('2024-08-26','result-b',{rolling_performance:{head_hash:'head-historical'}}),body('2024-08-26','result-b'),body('2024-09-02','result-b',{rolling_performance:{head_hash:'head-c'}}),{subject:{task_id:'other-book',session:'2024-10-01',result_hash:'result-b'},rolling_performance:{head_hash:'wrong-task'}},body('2024-10-01','result-b',{rolling_performance:{head_hash:'head-d'}})];
  const transport={
    app:{page:'portfolio',book:null,session:null,compareOther:null},
    document:{body:{dataset:{page:''}},visibilityState:'visible'},
    window:{},
    AbortController,
    fetch:async url=>{requests.push(String(url));const value=queue.shift();return {ok:true,status:200,text:async()=>JSON.stringify(value),headers:{get:()=> 'application/json'}};},
    replaceHash:()=>{},hashParams:()=>new URLSearchParams(),refresh:()=>{},render:()=>{},
    notify:()=>{},Window:{inspectorMode:()=>'',closeInspector:()=>{}},
    Inspect:{reopenFromAddress:()=>{},resetWindow:()=>{},selectObservation:()=>{},},
    LiveResearch:{ready:()=>{}},
    Portfolio:{refreshSessionSurface:()=>{},refreshPerformanceSurface:()=>{}},
    scrollX:0,scrollY:0,URLSearchParams,console,
  };
  library.context(transport);
  vm.runInContext(fs.readFileSync(path.join(appDir,'data.js'),'utf8'),transport,{filename:'data.js'});
  await vm.runInContext(`Data.openPortfolio('book','2024-08-12')`,transport);
  assert.equal(new URL(requests[0],'http://local').searchParams.has('scope'),false,'first open is a full owner read');
  await vm.runInContext(`Data.openPortfolio('book','2024-08-19')`,transport);
  assert.equal(new URL(requests[1],'http://local').searchParams.get('scope'),'holdings','same-book changed-session Historical read is scoped');
  assert.equal(vm.runInContext('Data.rollingPerformance().head_hash',transport),'head-a','rolling facts are retained only for the exact same result');
  assert.equal(vm.runInContext('Data.forwardPerformance()',transport),null,'no unrelated forward performance is fabricated from the holdings response');
  await vm.runInContext(`Data.openPortfolio('book','2024-08-26')`,transport);
  assert.equal(new URL(requests[2],'http://local').searchParams.get('scope'),'holdings','a later changed-session Historical read begins scoped');
  assert.equal(new URL(requests[3],'http://local').searchParams.has('scope'),false,'a changed result identity forces a full read');
  assert.equal(vm.runInContext('Data.rollingPerformance().head_hash',transport),'head-b','the full response replaces stale rolling facts after identity changes');
  await vm.runInContext(`Data.openPortfolio('book','2024-08-26','portfolio',null,null,'forward')`,transport);
  assert.equal(new URL(requests[4],'http://local').searchParams.has('scope'),false,'switching to Forward always performs a full read');
  assert.equal(new URL(requests[4],'http://local').searchParams.get('performance'),'latest');
  await vm.runInContext(`Data.openPortfolio('book','2024-08-26','portfolio',null,null,'historical')`,transport);
  assert.equal(new URL(requests[5],'http://local').searchParams.has('scope'),false,'switching back to Historical performs a full read');
  await vm.runInContext(`Data.openPortfolio('book','2024-09-02')`,transport);
  assert.equal(new URL(requests[6],'http://local').searchParams.get('scope'),'holdings','the next Historical session starts scoped');
  assert.equal(new URL(requests[7],'http://local').searchParams.has('scope'),false,'a scoped response for the wrong requested session forces a full read even when task and result hashes match');
  assert.equal(vm.runInContext('Data.subject().session',transport),'2024-09-02','the full fallback supplies the requested session');
  assert.equal(vm.runInContext('Data.rollingPerformance().head_hash',transport),'head-c','the wrong-session response cannot retain or replace the previous overlay');
  await vm.runInContext(`Data.openPortfolio('book','2024-10-01')`,transport);
  assert.equal(new URL(requests[8],'http://local').searchParams.get('scope'),'holdings','the next changed session begins with the scoped read');
  assert.equal(new URL(requests[9],'http://local').searchParams.has('scope'),false,'a scoped response for the wrong task forces a full read despite matching result hash and requested session');
  assert.equal(vm.runInContext('Data.subject().task_id',transport),'book','the full fallback supplies the requested book');
  assert.equal(vm.runInContext('Data.subject().session',transport),'2024-10-01','the full fallback supplies the requested session');
  assert.equal(vm.runInContext('Data.rollingPerformance().head_hash',transport),'head-d','the wrong-task response cannot contribute an overlay');
  console.log('portfolio scoped-read identity assertions passed');
}

// Different Portfolio views of the same task and date have separate open identities and
// wire URLs. A later Forward open may supersede a pending Historical read, but it must not
// join that read or let its answer overwrite the active Forward result.
async function overlappingPerformanceReads() {
  const requests=[], gates=new Map();
  const body=(session,resultHash,extra={})=>({subject:{task_id:'book',session,result_hash:resultHash},...extra});
  const transport={
    app:{page:'portfolio',book:null,session:null,compareOther:null},
    document:{body:{dataset:{page:''}},visibilityState:'visible'},window:{},AbortController,
    fetch:async (url) => {
      const key=String(url);requests.push(key);
      const gate=gates.get(key);
      if(gate)return {ok:true,status:200,text:async()=>JSON.stringify(await gate.promise),headers:{get:()=> 'application/json'}};
      return {ok:true,status:200,text:async()=>JSON.stringify(body('2024-08-12','result-a',{rolling_performance:{head_hash:'head-a'}})),headers:{get:()=> 'application/json'}};
    },
    replaceHash:()=>{},hashParams:()=>new URLSearchParams(),refresh:()=>{},render:()=>{},notify:()=>{},
    Window:{inspectorMode:()=>'',closeInspector:()=>{}},Inspect:{reopenFromAddress:()=>{},resetWindow:()=>{},selectObservation:()=>{}},
    LiveResearch:{ready:()=>{}},Portfolio:{refreshSessionSurface:()=>{},refreshPerformanceSurface:()=>{}},
    scrollX:0,scrollY:0,URLSearchParams,console,
  };
  library.context(transport);
  vm.runInContext(fs.readFileSync(path.join(appDir,'data.js'),'utf8'),transport,{filename:'data.js'});
  await vm.runInContext(`Data.openPortfolio('book','2024-08-12')`,transport);
  let releaseHistorical,releaseForward;
  const historicalURL='/api/workbench/portfolio?'+new URLSearchParams({task_id:'book',portfolio_session:'2024-08-19',scope:'holdings'});
  const forwardURL='/api/workbench/portfolio?'+new URLSearchParams({task_id:'book',portfolio_session:'2024-08-19',performance:'latest'});
  gates.set(historicalURL,{promise:new Promise(resolve=>{releaseHistorical=resolve;})});
  gates.set(forwardURL,{promise:new Promise(resolve=>{releaseForward=resolve;})});
  const historical=vm.runInContext(`Data.openPortfolio('book','2024-08-19')`,transport);
  const forward=vm.runInContext(`Data.openPortfolio('book','2024-08-19','portfolio',null,null,'forward')`,transport);
  await Promise.resolve();await Promise.resolve();
  assert.ok(requests.includes(historicalURL),'the Historical read uses its scoped owner URL');
  assert.ok(requests.includes(forwardURL),'the overlapping Forward read uses its distinct full owner URL');
  assert.equal(requests.length,3,'the Forward view does not deduplicate onto the in-flight Historical open');
  releaseHistorical(body('2024-08-19','result-a',{rolling_performance:{head_hash:'late-historical'}}));
  releaseForward(body('2024-08-19','result-a',{rolling_performance:{head_hash:'head-a'},forward_holdings:{status:'forward-current'},forward_performance:{available:true}}));
  await Promise.all([historical,forward]);
  assert.equal(vm.runInContext('Data.performanceMode()',transport),'forward','the later requested view remains active');
  assert.equal(vm.runInContext('Data.subject().session',transport),'2024-08-19');
  assert.equal(vm.runInContext('Data.raw().forward_holdings.status',transport),'forward-current');
  assert.equal(vm.runInContext('Data.rollingPerformance().head_hash',transport),'head-a','the superseded Historical answer cannot overwrite the prior source');
}
// U194: the public chart binding installs the real resize/RAF path. The DOM and event
// ports retain the painted Historical chart while the current reader requests Forward;
// neither the private navigator function nor its data/window inputs are monkeypatched.
function navigatorFixture(source=portfolioSource) {
  const drawnRows=[
    {date:'2024-08-05',value:100,benchmark:100,daily:0,benchmarkDaily:0},
    {date:'2024-08-12',value:101,benchmark:100,daily:1,benchmarkDaily:0},
    {date:'2024-08-19',value:102,benchmark:100,daily:1,benchmarkDaily:0},
  ];
  let currentRows=drawnRows, navMounted=true, plotMounted=true, ticket=0;
  const frames=new Map(), timers=new Map(), resizeListeners=[];
  const node=(dataset={})=>({dataset,style:{},attributes:{},textContent:'',listeners:new Map(),
    classList:{add(){},remove(){}},
    addEventListener(kind,listener){this.listeners.set(kind,listener);},
    setAttribute(key,value){this.attributes[key]=String(value);},
    getBoundingClientRect(){return {left:16,right:916,width:900};},
  });
  const rail=node(), win=node(), startShade=node(), endShade=node(), start=node(), end=node(), readings=node();
  const handles=[node({handle:'start'}),node({handle:'end'})];
  const children=new Map([
    ['[data-rail]',rail],['[data-window]',win],['[data-shade-start]',startShade],
    ['[data-shade-end]',endShade],['[data-window-start]',start],['[data-window-end]',end],['[data-readings]',readings],
  ]);
  const nav=node();
  nav.querySelector=selector=>children.get(selector);
  nav.querySelectorAll=selector=>selector==='[data-handle]'?handles:[];
  const plot=node({chartId:'portfolio',chart:'indexed',width:'1000',left:'36',right:'36',top:'20',bottom:'40',height:'320',min:'90',max:'110'});
  plot.getBoundingClientRect=()=>({left:64,right:864,width:800});
  const tip=node(), cross=node(), marker=node(), wrap=node();
  wrap.querySelector=selector=>({'[data-tooltip]':tip,'[data-cross]':cross,'[data-marker]':marker}[selector]);
  plot.closest=selector=>selector==='[data-chart-wrap]'?wrap:selector==='svg'?{querySelector:()=>cross}:null;
  const range=node({value:'all'}), charts=new Map([['portfolio',{
    rows:drawnRows,window:[1,2],kind:'indexed',lines:[{key:'value',daily:'daily',cls:'series',label:'Study'}],
  }]]);
  const view={...c,app:{...app,page:'portfolio'},PAGES:{},ACTIONS:{},ON_INPUT:{},ON_CHANGE:{},
    Data:{...Data,series:()=>currentRows},CHART_ROWS:charts,fitCharts:()=>0,
    Inspect:{...c.Inspect,observation:1,window:n=>[0,Math.max(0,n-1)],
      syncObservation(){},readingsMarkup:(rows,a,b)=>rows.slice(a,b+1).map(row=>row.date).join('|'),
      presetActive:(_key,rows,a,b)=>a===0&&b===rows.length-1,
    },
    $:selector=>selector==='[data-navigator]'?navMounted?nav:null
      :selector==='#performanceChart [data-chart]'||selector==='#performanceChart'?plotMounted?plot:null:null,
    $$:selector=>selector==='[data-chart]'?plotMounted?[plot]:[]
      :selector==='.performance-surface [data-action="chart-range"]'?[range]:[],
    addEventListener:(kind,listener)=>{if(kind==='resize')resizeListeners.push(listener);},
    requestAnimationFrame:callback=>{const id=++ticket;frames.set(id,callback);return id;},
    cancelAnimationFrame:id=>frames.delete(id),
    setTimeout:callback=>{const id=++ticket;timers.set(id,callback);return id;},
    clearTimeout:id=>timers.delete(id),
  };
  library.context(view);
  vm.runInContext(source,view,{filename:'pages-portfolio.js'});
  vm.runInContext('Portfolio.bindCharts()',view);
  const snapshot=()=>JSON.parse(JSON.stringify({start:start.textContent,end:end.textContent,
    handles:handles.map(handle=>handle.attributes),window:win.style,rail:rail.style,
    startShade:startShade.style,endShade:endShade.style,readings:readings.innerHTML,range:range.attributes}));
  return {charts,drawnRows,snapshot,setCurrent:rows=>{currentRows=rows;},
    mount:(withNav,withPlot=withNav)=>{navMounted=withNav;plotMounted=withPlot;},
    queueResize:()=>resizeListeners.forEach(listener=>listener({type:'resize'})),
    flushFrames:()=>{const pending=[...frames.values()];frames.clear();for(const callback of pending)callback();},
  };
}
const navigatorReader=navigatorFixture(), paintedNavigator=navigatorReader.snapshot();
assert.equal(paintedNavigator.start,'2024-08-12');
assert.equal(paintedNavigator.end,'2024-08-19');
assert.equal(paintedNavigator.window.left,'50.000%');
assert.equal(paintedNavigator.window.width,'50.000%');
assert.equal(paintedNavigator.readings,'2024-08-12|2024-08-19');
for(const currentRows of [[],[{date:'2026-10-05',value:200},{date:'2026-10-06',value:201}]]) {
  navigatorReader.setCurrent(currentRows);
  navigatorReader.queueResize();
  assert.doesNotThrow(()=>navigatorReader.flushFrames(),'a pending reader cannot break the mounted chart resize');
  assert.deepEqual(navigatorReader.snapshot(),paintedNavigator,'dates, window, readings and range all keep the painted chart snapshot');
}
const acceptedRows=[{date:'2026-10-01',value:100},{date:'2026-10-02',value:101}];
navigatorReader.charts.set('portfolio',{rows:acceptedRows,window:[0,1]});
navigatorReader.queueResize();navigatorReader.flushFrames();
assert.equal(navigatorReader.snapshot().start,'2026-10-01','the accepted mounted chart supplies the new first date');
assert.equal(navigatorReader.snapshot().end,'2026-10-02');
assert.equal(navigatorReader.snapshot().window.width,'100.000%');
assert.equal(navigatorReader.snapshot().readings,'2026-10-01|2026-10-02');
navigatorReader.charts.set('portfolio',{rows:[acceptedRows[0]],window:[0,0]});
navigatorReader.queueResize();navigatorReader.flushFrames();
assert.equal(navigatorReader.snapshot().start,navigatorReader.snapshot().end,'a single drawn observation is a valid zero-span window');
assert.equal(navigatorReader.snapshot().window.width,'0.000%');
const lastValidNavigator=navigatorReader.snapshot();
for(const invalid of [undefined,{rows:[],window:[0,0]},
  {rows:acceptedRows,window:[-1,1]},{rows:acceptedRows,window:[0,2]},
  {rows:acceptedRows,window:[1,0]},{rows:acceptedRows,window:[0.5,1]},
  {rows:acceptedRows},{rows:[{},acceptedRows[1]],window:[0,1]},
  {rows:[acceptedRows[0],{}],window:[0,1]}]) {
  if(invalid)navigatorReader.charts.set('portfolio',invalid);else navigatorReader.charts.delete('portfolio');
  navigatorReader.queueResize();
  assert.doesNotThrow(()=>navigatorReader.flushFrames(),'an absent or invalid drawn window has no readable navigator dates');
  assert.deepEqual(navigatorReader.snapshot(),lastValidNavigator,'an invalid snapshot cannot partly rewrite the navigator');
}
navigatorReader.charts.set('portfolio',{rows:acceptedRows,window:[0,1]});
navigatorReader.queueResize();navigatorReader.mount(false);
assert.doesNotThrow(()=>navigatorReader.flushFrames(),'a queued resize cannot operate on an unmounted navigator');
assert.deepEqual(navigatorReader.snapshot(),lastValidNavigator);
navigatorReader.mount(true,false);navigatorReader.queueResize();
assert.doesNotThrow(()=>navigatorReader.flushFrames(),'a navigator without its mounted plot cannot borrow a stale chart registration');
assert.deepEqual(navigatorReader.snapshot(),lastValidNavigator);
const mountedRowsClause=`    // A pending view read keeps the prior chart mounted. Its navigator measures the rows
    // and window that chart drew, as fitCharts does, until the accepted surface replaces it.
    const rows = drawn?.rows, [a, b] = drawn?.window || [];
    if (!Array.isArray(rows) || !rows.length || !Number.isInteger(a) || !Number.isInteger(b)
      || a < 0 || b < a || b >= rows.length || !rows[a]?.date || !rows[b]?.date) return;
    const n = rows.length;`;
assert.ok(portfolioSource.includes(mountedRowsClause),'the negative control restores only the former live-series layout binding');
const formerNavigator=navigatorFixture(portfolioSource.replace(mountedRowsClause,
  '    const rows = Data.series(), n = rows.length, [a, b] = Inspect.window(n);'));
formerNavigator.setCurrent([]);formerNavigator.queueResize();
assert.throws(()=>formerNavigator.flushFrames(),/Cannot read properties of undefined \(reading 'date'\)/,
  'the former live-series binding reproduces the reported error while the Historical navigator remains mounted');
console.log('portfolio navigator: mounted/pending series, accepted window, invalid endpoints and queued unmount with former-binding negative control');

async function runPortfolioTransportAssertions() {
  let watchdog;
  try {
    await Promise.race([
      (async()=>{await transportAssertions();await overlappingPerformanceReads();})(),
      new Promise((_,reject)=>{watchdog=setTimeout(()=>reject(new Error('Portfolio owner transport harness did not settle')),5000);}),
    ]);
  } finally { if(watchdog)clearTimeout(watchdog); }
}
{
  const {makeContext,connect,settle,clone,markupNodes}=require('./workbench_route_context.cjs');
  const finish=library.guard('portfolio layout');
const H=c=>c.repeat(64),SESSIONS=['2024-08-05','2024-08-12'];
const view = (task, session) => ({schema: 'verified-portfolio-display',
  subject: {task_id: task, result_hash: H('c'), source_kind: 'INSTALLED_RESULT', receipt_hash: H(task.at(-1)), title: 'Portfolio · 40 / 3 · ew', input_id: 'factor-development', input_hash: 'hash-a', input_date: '2026-08-03', session, support: {start: '2019-08-05', end: '2024-08-12', as_of: {session: '2026-08-03', phase: 'OFFICIAL_CLOSE'}}, cost_per_side: '5'},
  notice: 'Published historical research; not independent validation, current advice or trade authority.', seriesBasis: 'index',
  series: [{date: SESSIONS[0], daily: 0.1, benchmarkDaily: null, value: 100.1, benchmark: null}, {date: SESSIONS[1], daily: -0.1, benchmarkDaily: null, value: 100, benchmark: null}], sessions: SESSIONS,
  metrics: {total: 1, annual: 2, vol: 3, drawdown: -1, sharpe: 0.5, sortino: 0.6, informationRatio: null, turnover: 12.5, costBps: 4},
  metricProvenance: {status: 'ARCHIVED_BASE', report_hash: H('x'), window_hash: H('y')},
  holdings_basis: {basis:'HISTORICAL_REPLAY',comparison_basis:'PRECEDING_FORMATION',preceding_formation_session:'2024-07-29',formation_session:session,entry_session:null,observed_through:session},
  holdings: [{ticker: 'ABC', name: 'abc-one', listing_id: 'L-1', sector: null, mapped: null, target: 2.0, executed: 1.5,current_weight:0.015,preceding_weight:0.01,weight_change:0.005,disposition:'INCREASED'}, {ticker: 'DEF', name: 'def-two', listing_id: 'L-2', sector: null, mapped: null, target: 2.5, executed: 2.4,current_weight:0.024,preceding_weight:0.025,weight_change:-0.001,disposition:'REDUCED'}],
  universe: {eligible: 457, total: 466, quarantine: 9, definition: 'notice'}, limitations: ['DEVELOPMENT_EVIDENCE_ONLY'], reviewSelector: null,
  declaration: {top_k: 40, tranches: 3, exit_rank: 70, weight_rule: 'ew', cost_bps_per_side: '5', unavailable_return_policy: 'quarantine_listings'},
  source: {task_id: task, receipt_hash: H(task.at(-1)), input_binding_hash: 'hash-a', alpha_task_id: null, candidate_id: 'cand-' + task, target_recipe_id: 'TARGET', foundation_admission_hash: null, origin_task_id: null}});
async function projectionAssertions() {
 const rollingReads=[],forwardReads=[];let forwardScenario='recorded';
 const body=endpoint=>{
  if(!endpoint.startsWith('/api/workbench/portfolio'))return {};
  const u=new URL(endpoint,'http://localhost'),task=u.searchParams.get('task_id');
  if(!['book-b','book-c'].includes(task))return {};
    const body=view(task,u.searchParams.get('portfolio_session') || SESSIONS[1]);
    if (task === 'book-b' && u.searchParams.get('performance') !== 'latest' && u.searchParams.get('scope') !== 'holdings') {
      rollingReads.push({task, session: u.searchParams.get('portfolio_session') || SESSIONS[1]});
      const revision=rollingReads.length;
      body.rolling_performance={head_hash:H(String(revision%9+1)),base_report_hash:H('r'),base_result_hash:H('s'),head_storage:'SEALED',formation_range:{start:SESSIONS[0],end:'2024-08-19',base_end:SESSIONS[1]},outcomes_observed_through:`2024-08-${String(12+revision).padStart(2,'0')}`,claim:'DEVELOPMENT_EVIDENCE_WITH_POST_OBSERVED_QA_EXTENSION',execution_basis:'DAILY_BAR_QA_NOT_VERIFIED_VENUE_EXECUTION',cost_bps_per_side:5,metrics:{cumulative_return:0.12,annualized_return:0.1+revision*0.01,annualized_volatility:0.2,maximum_drawdown:-0.08,sharpe:0.7,sortino:0.9},metric_absences:{},curve:[{formation_session:SESSIONS[0],net_simple_return:0.01+revision*0.001,benchmark_simple_return:0.002,value:100.1,benchmark:100.2},{formation_session:SESSIONS[1],net_simple_return:-0.02,value:98.098,benchmark:100.1,benchmarkDaily:-0.001},{formation_session:'2024-08-19',net_simple_return:0.03,value:101.04094,benchmark_simple_return:null,benchmark:null}],open_positions:[{status:'ENTRY_SETTLED_OUTCOME_PENDING',formation_session:'2024-08-19',entry_session:'2024-08-20',holding_end_session:'2024-08-26'},{status:'PROPOSAL_NOT_YET_SETTLED',formation_session:'2024-08-26',entry_session:'2024-08-27',holding_end_session:'2024-09-02'}],observation_count:10+revision,outcome_count:revision};
    }
    if(u.searchParams.get('performance')==='latest') {
      forwardReads.push(task);
      const revision=forwardReads.length;
      if(task==='book-b') body.forward_holdings={status:'RECORDED_FORWARD_HOLDINGS',available:true,basis:'CONDITIONAL_ESTIMATE',comparison_basis:'FORMATION_CLOSE_ESTIMATE',formation_session:'2024-08-19',entry_session:'2024-08-20',holding_end_session:'2024-08-26',observed_through:'2024-08-20',publication_hash:H('4'),position_hash:H('5'),checkpoint_hash:H('6'),source_book_task_id:task,strategy_package_id:'pkg',cash:null,holdings:[{ticker:'ABC',name:'abc-one',listing_id:'L-1',sector:null,mapped:null,target:null,executed:null,current_weight:0.025,preceding_weight:0.02,weight_change:0.005,disposition:null}]};
      body.forward_performance=task==='book-b' && forwardScenario==='zero' ? {available:true,status:'INSUFFICIENT_REALIZED_OBSERVATIONS',cost_bps_per_side:'5',selected_window_metrics:{},selected_window_metric_absences:Object.fromEntries(['cumulative_return','annualized_return','annualized_volatility','maximum_drawdown','sharpe','sortino'].map((key)=>[key,{reason:'INSUFFICIENT_REALIZED_OBSERVATIONS',detail:'At least two settled one-session outcomes are needed for these metrics. Published proposals are not realized returns.'}])),selected_window_metric_provenance:{status:'INSUFFICIENT_REALIZED_OBSERVATIONS',selected_start:null,selected_end:null,observation_count:0,return_unit:'FRACTION',source_book_task_id:task,publication_hash:H('8'),window_hash:H('7'),claim:'POST_OBSERVED_QA_NOT_TIMELY_ADVICE',execution_basis:'DAILY_BAR_QA_NOT_VERIFIED_VENUE_EXECUTION'},series:[]} : task==='book-b' ? {available:true,status:'RECORDED_REALIZED_FORWARD_WINDOW',cost_bps_per_side:'5',selected_window_metrics:{annualized_return:0.2+Math.max(0,revision-1)*0.01,annualized_volatility:0.1,maximum_drawdown:-0.05,sharpe:1.25,sortino:1.5},selected_window_metric_absences:{},selected_window_metric_provenance:{status:'RECORDED_REALIZED_FORWARD_WINDOW',selected_start:SESSIONS[0],selected_end:SESSIONS[1],observation_count:2,return_axis:'CONTIGUOUS_ONE_SESSION_ENTRY_OPEN_TO_HOLDING_END_OPEN',return_unit:'FRACTION',annualization_sessions_per_year:252,volatility_degrees_of_freedom:1,sharpe_cash_return_per_session:0,sortino_downside_threshold:0,strategy_package_id:'pkg',strategy_package_hash:H('a'),checkpoint_hash:H('b'),source_book_task_id:task,publication_hash:H(String(revision%9+1)),history_prefix_hash:H('d'),settlement_path_hash:H('e'),source_snapshot_hash:H('f'),observed_through:SESSIONS[1],published_at:'2026-10-04T16:00:00Z',claim:'POST_OBSERVED_QA_NOT_TIMELY_ADVICE',execution_basis:'DAILY_BAR_QA_NOT_VERIFIED_VENUE_EXECUTION',window_hash:H('1'),net_return_field:'net_simple_return',cost_assumption_hash:H('2'),cost_bps_per_side:'5',platform_one_way_cost_bps:'5'},series:[{session:SESSIONS[0],formation_session:SESSIONS[0],entry_session:SESSIONS[0],holding_end_session:SESSIONS[0],net_simple_return:0.01+Math.max(0,revision-1)*0.001},{session:SESSIONS[1],formation_session:SESSIONS[0],entry_session:SESSIONS[1],holding_end_session:SESSIONS[1],net_simple_return:-0.02}]} : {available:false,status:'NO_REALIZED_FORWARD_PUBLICATION'};
    }

  return body;
 };
 const c=makeContext({hash:'#page=overview',realData:true,body});
 const {Data:D,app,LiveViews:V}=c.probe;c.AbortController=AbortController;
 await connect(c);c.setRoute('#page=portfolio&book=book-b');
 await D.openPortfolio('book-b',SESSIONS[1]);app.tab='holdings';
 const read=expression=>clone(vm.runInContext(expression,c));
 const render=()=>String(c.probe.PAGES.portfolio());
 const text=markup=>markup.replace(/<[^>]*>/g,'');
 const performance=()=>{const markup=render();return markup.slice(markup.indexOf('id="portfolioPerformance"'),markup.indexOf('id="bookActivation"'));};
 const selectedSource=()=>[...performance().matchAll(/<button([^>]*)>(.*?)<\/button>/gs)].filter(([,attrs])=>attrs.includes('role="tab"')&&attrs.includes('aria-selected="true"')).map(([,attrs,body])=>text(body));
 const hasClass=(node,name)=>node.attrs.class?.split(' ').includes(name);
 const performanceNodes=()=>markupNodes(performance());
 const holdingCells=()=>[...render().match(/<tr data-holding-row="L-1"[^]*?<\/tr>/)[0].matchAll(/<td class="num[^"]*">(.*?)<\/td>/g)].map(([,value])=>text(value));
 const word=(key,args={})=>{c.fixtureKey=key;c.fixtureArgs=args;return vm.runInContext('t(fixtureKey,fixtureArgs)',c);};
 const factValues=()=>{const page=render(),start=page.indexOf('metric-basis'),text=page.slice(start,page.indexOf('</div>',start));return [...text.matchAll(/<span class="context-key">([^<]*)<\/span>\s*<b[^>]*>(.*?)<\/b>/g)].map(([,label,value])=>({label,value:value.replace(/<[^>]*>/g,"" )}));};
 assert.equal(D.metrics().annual,(0.1+rollingReads.length*0.01)*100);
 assert.deepEqual(selectedSource(),['Historical report']);
 assert.deepEqual(read('({annual:Data.raw().metrics.annual,result:Data.raw().subject.result_hash,rolling:Data.raw().rolling_performance.base_result_hash})'),{annual:2,result:H('c'),rolling:H('s')});
 assert.equal(D.series().length,3);
 assert.deepEqual(clone(D.series().map(({date,daily,value,benchmark,benchmarkDaily})=>({date,daily,value,benchmark,benchmarkDaily}))),[{date:SESSIONS[0],daily:0.011*100,value:100.1,benchmark:100.2,benchmarkDaily:0.2},{date:SESSIONS[1],daily:-2,value:98.098,benchmark:100.1,benchmarkDaily:-0.001},{date:'2024-08-19',daily:3,value:101.04094,benchmark:null,benchmarkDaily:null}]);
 assert.deepEqual(clone(D.raw().holdings_basis),view('book-b',SESSIONS[1]).holdings_basis);
 assert.deepEqual(holdingCells().slice(1),['1.5%','+0.50 pp']);
 const firstRollingDate=D.raw().rolling_performance.outcomes_observed_through,rollingFirstReads=rollingReads.length;
 assert.deepEqual(factValues(),[{label:'Formation range',value:'2024-08-05 — 2024-08-19'},{label:'Outcomes through',value:firstRollingDate},{label:'Cost',value:'5 bps per side'}]);
 const proof=()=>String(V.proofBody('annual'));
 let markup=render(),nodes=markupNodes(markup);
 assert.equal(nodes.filter(node=>node.attrs.class?.split(' ').includes('rolling-open-periods')).length,1);
 for(const key of ['Entry settled; outcome pending','Proposal not yet settled'])assert.ok(markup.includes(word(key)));
 assert.match(text(markup),/Open periods.*Entry settled; outcome pending.*Formation\s*2024-08-19.*Entry\s*2024-08-20.*Holding end\s*2024-08-26/s);
 assert.match(text(markup),/Proposal not yet settled.*Formation\s*2024-08-26.*Entry\s*2024-08-27.*Holding end\s*2024-09-02/s);
 assert.equal(nodes.filter(node=>node.attrs.class?.split(' ').includes('rolling-provenance')).length,0);
 assert.ok(!markup.includes(H('r').slice(0,12))&&!markup.includes('Head storage'));
 assert.ok(proof().includes(word('Owner-reported metric for rolling outcomes')));
 assert.match(text(proof()),/Rolling head.*Base report.*Base result/s);
 assert.ok(proof().includes(firstRollingDate)&&!proof().includes(H('x').slice(0,12)));
 for(const key of ['Development evidence extended with post-observed QA outcomes','Daily-bar QA; venue execution is not verified'])assert.ok(proof().includes(word(key)));
 vm.runInContext('I18N.set("zh")',c);
 assert.deepEqual(factValues(), [
    {label:'形成日范围',value:'2024-08-05 — 2024-08-19'},
    {label:'结果观测截至',value:firstRollingDate},
    {label:'成本',value:'5 每边 bps'},
  ]);
 assert.match(render().replace(/<[^>]*>/g,''),/未结束的期间.*已结算.*形成日 2024-08-19/s);
 assert.match(proof().replace(/<[^>]*>/g,''),/滚动结果指标由所有者报告/);
 assert.match(proof().replace(/<[^>]*>/g,''),/开发证据.*事后观测 QA/);
 assert.match(proof().replace(/<[^>]*>/g,''),/日线 QA.*未经交易场所执行验证/);
 for(const [key,value] of [
 ["The daily checkpoint's source book has no readable publishing Task. Read the saved studies to confirm the Task; if its retained record is missing, restore a verified workspace backup and read again.","每日检查点的来源账本没有可读取的发布任务。请读取已保存的研究以确认该任务；如果保留记录缺失，请恢复经过验证的工作区备份后重新读取。"],
 ["This book's declared cost has no sealed daily return lane. Daily updates record 5 or 10 bps per side; read a book with one of those declared costs, or report the unavailable lane. Do not substitute another cost.","此账本声明的成本没有已封存的日收益档位。每日更新记录每侧 5 或 10 个基点；请读取声明了其中一种成本的账本，或报告该档位不可用。请勿替换为其他成本。"],
 ["The rolling report's sealed base, update, or link does not match its declared identity or date axis, so no performance values are shown. Restore a verified workspace backup, then read the saved study again.","滚动报告的已封存基准、更新或链接与声明的身份或日期轴不符，因此不显示绩效数值。请恢复经过验证的工作区备份，然后重新读取已保存的研究。"]])assert.equal(word(key),value);
 vm.runInContext('I18N.set("en")',c);
 const firstMetric=D.metrics().annual;
 D.setTasks([{task_id:'portfolio-history-refresh-probe',lifecycle:'RUNNING'}]);D.setTasks([{task_id:'portfolio-history-refresh-probe',lifecycle:'SUCCEEDED'}]);await settle();
 assert.ok(rollingReads.length>rollingFirstReads&&D.metrics().annual>firstMetric);
 assert.equal(D.subject().task_id,'book-b');assert.equal(D.subject().session,SESSIONS[1]);assert.equal(D.performanceMode(),'historical');
 assert.notEqual(D.raw().rolling_performance.outcomes_observed_through,firstRollingDate);
 assert.equal(D.metrics().annual,(0.1+rollingReads.length*0.01)*100);
 assert.equal(markupNodes(render()).filter(node=>node.attrs.class?.split(' ').includes('stat')&&node.parent?.attrs.class?.split(' ').includes('five')).length,5);
 await D.openPortfolio('book-b',SESSIONS[1],'portfolio',null,null,'forward');
 assert.equal(D.performanceMode(),'forward');assert.equal(D.forwardPerformance().available,true);
 assert.ok(c.records.requests.some(endpoint=>new URL(endpoint,'http://local').searchParams.get('performance')==='latest'));
 assert.equal(D.metrics().annual,20);
 assert.deepEqual(selectedSource(),['Forward realized']);
 assert.deepEqual(read('({raw:Data.series()[0].net_simple_return,display:Data.series()[0].daily})'),{raw:0.01,display:1});
 assert.equal(D.raw().forward_holdings.holdings[0].current_weight,0.025);
 assert.deepEqual(holdingCells().slice(1),['2.5%','+0.50 pp']);
 assert.match(text(performance()),new RegExp(SESSIONS[0]+'.*'+SESSIONS[1]));
 assert.match(text(performance()),/Not timely advice|事后观测 QA/);
 assert.equal(performanceNodes().filter(node=>hasClass(node,'stat-strip')&&hasClass(node,'five')).length,0);
 assert.equal(performanceNodes().filter(node=>hasClass(node,'rolling-provenance')).length,0);
 assert.equal(markupNodes(render()).filter(node=>node.attrs['data-navigator']).length,0);
 const firstForwardReads=forwardReads.length;
 D.setTasks([{task_id:'portfolio-refresh-probe',lifecycle:'RUNNING'}]);D.setTasks([{task_id:'portfolio-refresh-probe',lifecycle:'SUCCEEDED'}]);await settle();
 assert.ok(forwardReads.length>firstForwardReads);
 assert.equal(performanceNodes().filter(node=>hasClass(node,'stat-strip')&&hasClass(node,'five')).length,0);
 assert.ok(Math.abs(D.metrics().annual-(0.2+Math.max(0,forwardReads.length-1)*0.01)*100)<1e-8);
 assert.ok(Math.abs(D.series()[0].daily-(0.01+Math.max(0,forwardReads.length-1)*0.001)*100)<1e-8);
 forwardScenario='zero';await D.reload();
 assert.equal(D.forwardPerformance().status,'INSUFFICIENT_REALIZED_OBSERVATIONS');assert.equal(D.series().length,0);
 assert.deepEqual(read('({metrics:Data.metrics(),rows:Data.series().length,historyAnnual:Data.raw().metrics.annual})'),{metrics:{},rows:0,historyAnnual:2});
 const empty=text(performance());
 assert.match(empty,/Fewer than two settled outcomes were recorded/);
 assert.match(empty,/0 observations.*Net of 5 bps per side/);
 assert.doesNotMatch(empty,/Not timely advice|No published observations|Realized window|2024-08-05|2024-08-12/);
 assert.equal(performanceNodes().filter(node=>hasClass(node,'performance-surface')||node.attrs['data-box']).length,0);
 assert.equal(performanceNodes().filter(node=>hasClass(node,'stat-strip')&&hasClass(node,'five')).length,0);
 assert.equal(performanceNodes().filter(node=>node.attrs['data-empty']==='elsewhere').length,1);
 for(const name of ['chart-toolbar','chart-headline','chart-under'])assert.equal(performanceNodes().filter(node=>hasClass(node,name)).length,0);
 assert.equal(performanceNodes().filter(node=>node.tag==='svg'&&(()=>{for(let p=node.parent;p;p=p.parent)if(p.attrs.id==='performanceChart')return true;return false;})()).length,0);
 assert.match(D.forwardPerformance().selected_window_metric_absences.annualized_return.detail,/At least two settled one-session outcomes/);
 assert.equal(performanceNodes().filter(node=>node.attrs['data-action']==='chart-data').length,0);
 const beforeHistoricalSelection=c.records.requests.length;
 await D.openPortfolio('book-b',SESSIONS[1],'portfolio',null,null,'historical');
 assert.equal(D.metrics().annual,(0.1+rollingReads.length*0.01)*100);
 assert.equal(D.raw().metrics.annual,2);
 assert.equal(performanceNodes().filter(node=>hasClass(node,'stat')&&hasClass(node.parent,'five')).length,5);
 assert.ok(c.records.requests.slice(beforeHistoricalSelection).map(endpoint=>new URL(endpoint,'http://local')).some(u=>u.pathname==='/api/workbench/portfolio'&&u.searchParams.get('task_id')==='book-b'&&u.searchParams.get('portfolio_session')===SESSIONS[1]&&u.searchParams.get('performance')!=='latest'));
 assert.equal(D.performanceMode(),'historical');
 await D.openPortfolio('book-c',SESSIONS[1],'portfolio',null,null,'forward');
 assert.equal(D.forwardPerformance().status,'NO_REALIZED_FORWARD_PUBLICATION');
 assert.deepEqual(read('({mode:Data.performanceMode(),metrics:Data.metrics(),historyAnnual:Data.raw().metrics.annual})'),{mode:'forward',metrics:{},historyAnnual:2});
}
async function coverageAssertions() {
 const cv={mapping:'100.000%',reviewed_ending_weight:'44.000%',reviewed_absolute_change:'0.000%',selected_issuers:'100.000%'};
 let projection={explanation:'',review_attribution:[],available_actions:[],required_actions:[],issuer_rows:[],issue_cards:[],citations:[],claim_limits:[],reasons:[],rule_ids:[],gaps:[],eligible_versions:[],scope_coverage:{attainable_below_minimum:false,book_positions:2,unmapped_positions:0,minimum_required_weight:'60.000%',whole_book_reviewed_weight:'44.000%'},coverage:{...cv,nothing_filed_ending_weight:'56.000%',unreached_ending_weight:'0.000%',accounted_ending_weight:'100.000%',nothing_filed_window_days:30},task_id:null,task_lifecycle:null,book:{result_hash:H('r'),formation_session:SESSIONS[1],held_count:2},evidence_as_of:'2026-10-02T16:00:00Z',evidence_expires_at:'2026-10-03T16:00:00Z',review_publication_hash:H('p'),review_state:'COMPLETE',disposition:'NO_MATERIAL_OBJECTION',policy_version:'v1',state:'REVIEW_PUBLISHED'};
 const c=makeContext({hash:'#page=books',realData:true,body:url=>url.startsWith('/api/evidence-cro?')?projection:{}}),review=c.probe.LiveReview;
 c.AbortController=AbortController;await connect(c);await review.open({result_hash:H('r')});
 const render=()=>String(review.page()),text=markup=>markup.replace(/<[^>]*>/g,''),classes=node=>node.attrs.class?.split(' ')||[];
 const word=key=>{c.fixtureKey=key;return vm.runInContext('t(fixtureKey)',c);};
 const under=(node,cls)=>{for(let p=node.parent;p;p=p.parent)if(classes(p).includes(cls))return true;return false;};
 let markup=render(),words=text(markup);
 assert.ok(words.includes(word('at or above the floor')));
 for(const value of [/44(?:\.000)?%/,/56(?:\.000)?%/,/0(?:\.000)?%/,/100(?:\.000)?%/])assert.match(words,value);
 assert.equal(markupNodes(markup).filter(node=>node.tag==='i'&&under(node,'meter-bar')&&under(node,'coverage-meter')).length,1);
 projection={...projection,coverage:cv};await review.refresh();markup=render();words=text(markup);
 assert.ok(words.includes(word('Whole-book coverage accounting was not recorded; the floor comparison is unavailable.')));
 assert.match(words,/44(?:\.000)?%/);assert.match(words,/floor comparison is unavailable/);
 assert.equal(markupNodes(markup).filter(node=>classes(node).includes('coverage-meter')).length,0);
}
  (async()=>{await runPortfolioTransportAssertions();await projectionAssertions();await coverageAssertions();finish();})()
    .catch(error=>{console.error(error);process.exitCode=1;});
}
