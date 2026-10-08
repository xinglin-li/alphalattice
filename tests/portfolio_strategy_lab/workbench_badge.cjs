// Real durable Task records through the served bundle; no response or attention fixtures.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
const finish = require('./workbench_library.cjs').guard('workbench badge');
const [url, launch, flag, input, ...extra] = process.argv.slice(2);
const mode = flag?.startsWith('--mode=') ? flag.slice(7) : '';
assert.ok(url && launch && input && !extra.length && ['badge', 'pair', 'walk'].includes(mode),
  'workbench_badge requires URL launch --mode=badge|pair|walk receipt-or-output');
const ids = mode === 'walk' ? {out: input} : JSON.parse(fs.readFileSync(input, 'utf8'));
fs.mkdirSync(ids.out, {recursive: true});
const base = url.replace(/\/$/, '');
let browser;
const seenKeys = new Set();

async function reading(page, place, language = 'en', theme = 'light') {
  await page.goto(base + '/workbench.html#' + new URLSearchParams({page: place, lang: language, theme}));
  await page.waitForFunction(() => window.AlphaLattice?.data.workspaceStatus === 'ready', null,
    {timeout: 30000});
  await page.evaluate(({language, theme}) => {
    window.AlphaLattice.setTheme(theme);
    window.AlphaLattice.setLocale(language);
  }, {language, theme});
  await page.waitForFunction(p => window.AlphaLattice.state().page === p &&
    document.querySelector('#main')?.textContent.trim() && !document.querySelector('[data-render-failure]'),
  place, {timeout: 15000});
  await page.evaluate(async () => {
    await window.AlphaLattice.data.refreshDecisions();
    await window.AlphaLattice.data.refreshHistory();
  });
  await page.waitForFunction(({language, theme}) => document.documentElement.lang === language &&
    document.documentElement.dataset.theme === theme, {language, theme}, {timeout: 15000});
  assert.equal(await page.evaluate(() => innerWidth), 900, 'the actual viewport is 900 px');
  const visible = await page.locator('body').innerText();
  if(visible.includes('Word not declared')) {
    fs.writeFileSync(path.join(ids.out, 'undeclared-words.txt'), visible);
    await screenshot(page, 'undeclared-words');
  }
  assert.ok(!visible.includes('Word not declared'),
    'no undeclared product word is accepted as a readable UI');
  for (const key of await page.evaluate(() => window.AlphaLattice.i18nKeys())) seenKeys.add(key);
}

async function screenshot(page, name) {
  await page.screenshot({path: path.join(ids.out, name + '.png'), fullPage: true});
}

async function badgeCount(page, expected) {
  // At 900px the default dock is its rail; the public navigation toggle reveals its counts.
  if (await page.locator('#side.side-folded').count())
    await page.locator('#top [data-action="side-toggle"]').click();
  try {
    await page.waitForFunction(n => {
      const badge = document.querySelector('#side [data-page="tasks"] .side-badge');
      return n ? badge?.textContent.trim() === String(n) : !badge;
    }, expected, {timeout: 15000});
  } catch (error) {
    const diagnostic = await page.evaluate(() => ({
      tasks: window.AlphaLattice.data.tasks(), decisions: window.AlphaLattice.data.decisions(),
      actionable: window.AlphaLattice.data.actionableTasks(), side: document.querySelector('#side')?.outerHTML
    }));
    fs.writeFileSync(path.join(ids.out, 'badge-diagnostic.json'), JSON.stringify(diagnostic, null, 2));
    throw error;
  }
}

async function groupedPair(page, place) {
  await reading(page, place);
  const lobby = page.locator(`#main [data-lobby="${place}"]`);
  await lobby.waitFor({state: 'visible'});
  while (await lobby.locator('[data-action="lobby-fold"][aria-expanded="false"]').count())
    await lobby.locator('[data-action="lobby-fold"][aria-expanded="false"]').first().click();
  const sourceKey = place === 'history' ? 'task:' + ids.source : ids.source;
  const successorKey = place === 'history' ? 'task:' + ids.successor : ids.successor;
  const front = lobby.locator(`[data-key="${successorKey}"]`);
  const earlier = lobby.locator(`[data-key="${sourceKey}"]`);
  assert.equal(await front.count(), 1, place + ': one canonical successor front');
  await front.waitFor({state: 'visible'});
  assert.equal(await earlier.count(), 1, place + ': one retained earlier stop');
  assert.equal(await earlier.isVisible(), false, place + ': earlier stop begins folded');
  const fold = earlier.locator('xpath=ancestor::details[1]');
  assert.equal(await fold.count(), 1, place + ': earlier stop is in its disclosure');
  await fold.locator(':scope > summary').click();
  await earlier.waitFor({state: 'visible'});
  assert.match(await earlier.textContent(), /Superseded by/);
  await screenshot(page, place + '-earlier-stop');
  if (mode === 'badge') {
    const peerKey = place === 'history' ? 'task:' + ids.peer : ids.peer;
    const peer = lobby.locator(`[data-key="${peerKey}"]`);
    assert.equal(await peer.count(), 1, place + ': the agent-owned stop stays listed');
    await peer.waitFor({state: 'visible'});
    const read = page.waitForResponse(response => {
      const address = new URL(response.url());
      return address.pathname === '/api/tasks/recovery' && address.searchParams.get('task_id') === ids.peer;
    });
    await peer.locator('.list-row-main').click();
    const view = await (await read).json();
    assert.equal(view.task_id, ids.peer, place + ': the retained row opens its exact Task');
    assert.equal(view.task_record_hash, ids.peer_hash);
    await page.waitForFunction(id => new URLSearchParams(location.hash.slice(1)).get('task') === id &&
      document.querySelector('[data-mode="task"] .tp-detail'), ids.peer, {timeout: 15000});
  }
}

(async () => {
  browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 900, height: 900}});
  const errors = [];
  const writes = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {
    if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/cancel')
      writes.push(request.postDataJSON());
  });
  const entered = await page.context().request.get(launch, {maxRedirects: 0});
  assert.equal(entered.status(), 303, 'real launch URL supplies the browser session');
  if (mode === 'walk') {
    for (const language of ['en', 'zh-CN']) for (const theme of ['light', 'dark'])
      for (const place of ['overview', 'tasks', 'history']) {
        await reading(page, place, language, theme);
        await screenshot(page, `${place}-${language}-${theme}-900`);
      }
  } else {
    await reading(page, 'overview');
    await badgeCount(page, 0);
    assert.equal(await page.evaluate(() => window.AlphaLattice.data.actionableTasks().length), 0,
      'agent-owned stops do not count as a person decision');
    const tasks = await page.evaluate(() => window.AlphaLattice.data.read('/api/tasks'));
    const source = tasks.tasks.find(task => task.task_id === ids.source);
    const successor = tasks.tasks.find(task => task.task_id === ids.successor);
    assert.equal(source.lifecycle, 'BLOCKED');
    assert.equal(successor.lifecycle, 'SUCCEEDED');
    assert.equal(source.attention.unresolved, false);
    assert.equal(source.attention.successor_task_id, ids.successor);
    assert.equal(source.attention.successor_task_hash, successor.task_record_hash);
    const activity = await page.evaluate(({source, successor}) => window.AlphaLattice.data.read(
      '/api/activity?' + new URLSearchParams({watch: [source, successor].join(',')})), ids);
    for (const task of [source, successor])
      assert.equal(activity.tasks[task.task_id]?.task_record_hash, task.task_record_hash,
        'the real activity projection carries its canonical Task version');
    const afterActivity = await page.evaluate(({rows, id}) => {
      const data = window.AlphaLattice.data;
      data.mergeTasks(Object.values(rows));
      return data.taskSuccessor(data.taskOf(id));
    }, {rows: activity.tasks, id: ids.source});
    assert.equal(afterActivity?.successor_task_id, ids.successor,
      'the real activity answer keeps same-version successor attention in the built consumer');
    const decisions = await page.evaluate(() => window.AlphaLattice.data.read('/api/decisions'));
    assert.equal(decisions.counts.STOPPED_TASK || 0, mode === 'badge' ? 1 : 0,
      'the owner kind inventory retains the agent-owned stop');
    assert.ok(!decisions.decisions.some(row => row.task_id === ids.source));
    if (mode === 'badge') {
      const peerDecision = decisions.decisions.find(row => row.kind === 'STOPPED_TASK' && row.task_id === ids.peer);
      assert.equal(peerDecision?.waits_on, 'AGENT');
      assert.ok(!decisions.decisions.some(row => row.task_id === ids.rebuilt));
      const needs = page.locator('#main .home-groups .group-head').filter({hasText: 'Needs a decision'});
      assert.equal(await needs.count(), 0, 'nothing in this fixture waits on the person');
      assert.equal(await page.locator(`#main .home-groups [data-key="${ids.peer}"]`).count(), 0);
      assert.equal(await page.locator(`#main .home-groups [data-key="${ids.rebuilt}"]`).count(), 0);
    }
    await screenshot(page, 'overview-900');
    await groupedPair(page, 'tasks');
    await groupedPair(page, 'history');
    if (mode === 'badge') {
      await reading(page, 'overview');
      const stale = await page.evaluate(async id => {
        try { return await window.AlphaLattice.data.post('/api/cancel',
          {task_id: id, expected_task_hash: '0'.repeat(64)}); }
        catch (error) { return error.body; }
      }, ids.rebuilt);
      assert.equal(stale.failure_code, 'local_application.confirmation_stale');
      assert.equal(stale.lifecycle, 'BLOCKED');
      await page.locator('#main [data-action="task-close-unrecoverable"]').click();
      const confirm = page.locator('[data-action="task-commit"]');
      await confirm.waitFor({state: 'visible'});
      const preview = await page.evaluate(id => window.AlphaLattice.data.read(
        '/api/tasks/recovery?task_id=' + id), ids.rebuilt);
      assert.equal(preview.task_record_hash, ids.rebuilt_hash);
      const sent = page.waitForResponse(response => response.request().method() === 'POST' &&
        new URL(response.url()).pathname === '/api/cancel');
      await confirm.click();
      const response = await sent;
      const closed = await response.json();
      assert.equal(closed.lifecycle, 'CANCELLED');
      assert.equal(closed.latest_failure_code, 'task_control.ledger_rebuilt');
      assert.equal(writes.at(-1).task_id, ids.rebuilt);
      assert.equal(writes.at(-1).expected_task_hash, ids.rebuilt_hash);
      await page.waitForFunction(() => !document.querySelector('#main [data-action="task-close-unrecoverable"]'),
        null, {timeout: 15000});
      await badgeCount(page, 0);
      await screenshot(page, 'overview-closed-900');
    }
  }
  assert.deepEqual(errors, [], 'the served bundle has no page errors');
  fs.writeFileSync(path.join(ids.out, 'i18n-keys.json'), JSON.stringify([...seenKeys].sort(), null, 2));
  console.log('workbench badge ' + mode + ' complete');
  finish();
})().catch(error => {console.error(error.stack || error); process.exitCode = 1;})
  .finally(async () => {if (browser) await browser.close();});
