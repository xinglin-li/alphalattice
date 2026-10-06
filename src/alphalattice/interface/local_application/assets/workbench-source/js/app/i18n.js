/* Localization: one catalog (English source text → Chinese), one lookup, resolved while rendering.
 *
 * Every user-facing string in the templates goes through t(). The English text is the key; the
 * Chinese translation lives in data/zh.js. A key may carry a short context prefix ("tp|Queued")
 * when the same English phrase is translated differently in another part of the product; in the
 * English locale the prefix is simply stripped. Composite strings (counts, dates, elapsed times)
 * fall back to a bounded set of structural templates. Machine identifiers, YAML, JSON and fixture
 * data never pass through t().
 */
const I18N = (() => {
  // One catalog object for the product's life: the dictionary (`workbench.zh.js`, round 96) is its
  // own file, loaded only for a zh reader, and fills this object whether it loads before the app
  // (the prelude asked for it) or after (the reader switched language).
  const catalog = window.ALPHA_ZH || (window.ALPHA_ZH = {});
  const missing = new Set();
  const seen = new Set(); // every key t() was asked for, in either locale (catalog coverage tooling)
  let locale = 'en';
  const CONTEXT = /^[a-z0-9-]+\|/;
  const lookup = (s) => (Object.hasOwn(catalog, s) ? catalog[s] : templated(s));

  /* Structural templates: exact shapes only, never arbitrary substring replacement. */
  const TEMPLATES = [
    // V671: complete Evidence owner grammar; generated components recurse, authored subjects stay exact.
    [/^(\d[\d,]*) of (\d[\d,]*) finding(?:\(s\)|s)? were read by the reviewer and not named as a risk\.$/, (m) => t("{n} of {total} finding(s) were read by the reviewer and not named as a risk.", {n: m[1], total: m[2]})],
    [/^filing index read at the cutoff: (\d[\d,]*) filing(?:\(s\)|s)? in the last (\d[\d,]*) days, every one read earlier; no new filing$/, (m) => t("filing index read at the cutoff: {n} filing(s) in the last {days} days, every one read earlier; no new filing", {n: m[1], days: m[2]})],
    [/^filing index read at the cutoff: (\d[\d,]*) new filing(?:\(s\)|s)? in the last (\d[\d,]*) days$/, (m) => t("filing index read at the cutoff: {n} new filing(s) in the last {days} days", {n: m[1], days: m[2]})],
    [/^filing index read at the cutoff: (\d[\d,]*) new filing(?:\(s\)|s)? in the last (\d[\d,]*) days, (\d[\d,]*) read earlier$/, (m) => t("filing index read at the cutoff: {n} new filing(s) in the last {days} days, {earlier} read earlier", {n: m[1], days: m[2], earlier: m[3]})],
    [/^filing index read at the cutoff: (\d[\d,]*) new filing(?:\(s\)|s)? in the last (\d[\d,]*) days, (\d[\d,]*) beyond the capacity$/, (m) => t("filing index read at the cutoff: {n} new filing(s) in the last {days} days, {beyond} beyond the capacity", {n: m[1], days: m[2], beyond: m[3]})],
    [/^filing index read at the cutoff: (\d[\d,]*) new filing(?:\(s\)|s)? in the last (\d[\d,]*) days, (\d[\d,]*) read earlier, (\d[\d,]*) beyond the capacity$/, (m) => t("filing index read at the cutoff: {n} new filing(s) in the last {days} days, {earlier} read earlier, {beyond} beyond the capacity", {n: m[1], days: m[2], earlier: m[3], beyond: m[4]})],
    [/^([\s\S]+) Managed \(Provider\-backed\) analysis and review need an admitted credential; none is admitted here, so only the native entries above are offered\.$/, (m) => said(m[1]) + ' ' + t("Managed (Provider-backed) analysis and review need an admitted credential; none is admitted here, so only the native entries above are offered.")],
    [/^No admitted Alternative Evidence analysis covers this book's issuers yet\. Prepare the admitted sources into an analyst packet and submit an analysis against it; a review without evidence would have nothing to read\. ([\s\S]+)$/, (m) => t("No admitted Alternative Evidence analysis covers this book's issuers yet. Prepare the admitted sources into an analyst packet and submit an analysis against it; a review without evidence would have nothing to read.") + ' ' + said(m[1])],
    [/^([\s\S]+) It was sealed under an earlier CRO review policy; it reads as recorded\.$/, (m) => said(m[1]) + ' ' + t("It was sealed under an earlier CRO review policy; it reads as recorded.")],
    [/^([\s\S]+) The review published on this dossier reads back by its handle; this build no longer gives its recommendation, so it is not the current answer\.$/, (m) => said(m[1]) + ' ' + t("The review published on this dossier reads back by its handle; this build no longer gives its recommendation, so it is not the current answer.")],
    [/^([\s\S]+) The published review reads back by its handle; it answers its sealed dossier, not this newly compiled dossier\.$/, (m) => said(m[1]) + ' ' + t("The published review reads back by its handle; it answers its sealed dossier, not this newly compiled dossier.")],
    [/^([\s\S]+) The review published against it remains readable by its handle\.$/, (m) => said(m[1]) + ' ' + t("The review published against it remains readable by its handle.")],
    [/^([\s\S]+) (\d[\d,]*) of (\d[\d,]*) units have a current analysis; the review reads those exactly and names the rest as not reviewed\.$/, (m) => said(m[1]) + ' ' + t("{ready} of {total} units have a current analysis; the review reads those exactly and names the rest as not reviewed.", {ready: m[2], total: m[3]})],
    [/^([\s\S]+) (\d[\d,]*) of (\d[\d,]*) units have a current analysis; current coverage gaps remain named\.$/, (m) => said(m[1]) + ' ' + t("{ready} of {total} units have a current analysis; current coverage gaps remain named.", {ready: m[2], total: m[3]})],
    [/^Position basis: ([^;\n]+); decision formation: (\d{4}-\d{2}-\d{2})\.$/, (m) => t('Position basis: {basis}; decision formation: {date}.', {basis: lookup(m[1]), date: m[2]})],
    [/^(\d[\d,]*) of (\d[\d,]*) (unit|units) failed to prepare, each with its code and words\.(?: (\d[\d,]*) failed for too few source documents, which preparing again under the installed sources does not change; the ways on are the person's \(`source_ways`\)\.| (Under official acquisition, preparing again retries the filings that could not be obtained\.))?(?: (Preparing again retries the others and keeps every unit that completed\.))?$/, (m) => [t(m[3] === 'unit' ? '{failed} of {total} unit failed to prepare, each with its code and words.' : '{failed} of {total} units failed to prepare, each with its code and words.', {failed: m[1], total: m[2]}), m[4] ? t("{n} failed for too few source documents, which preparing again under the installed sources does not change; the ways on are the person's (`source_ways`).", {n: m[4]}) : '', m[5] ? t(m[5]) : '', m[6] ? t(m[6]) : ''].filter(Boolean).join(' ')],
    [/^([\s\S]+) The unit failed with `([^`\n]+)`: ([\s\S]+)$/, (m) => t('{words} The unit failed with `{code}`: {detail}', {words: said(m[1]), code: m[2], detail: said(m[3])})],
    [/^([\s\S]+) Without a source: ([^\n]+)\.$/, (m) => said(m[1]) + ' ' + t('Without a source: {issuers}.', {issuers: m[2]})],
    [/^The reviewer's last answer had (\d[\d,]*) item(?:\(s\)|s)? the Host could not admit after two corrections; they are not part of this review\. ([\s\S]*?)(?: And (\d[\d,]*) more\.)?$/, (m) => t("The reviewer's last answer had {n} item(s) the Host could not admit after two corrections; they are not part of this review. {items}{more}", {n: m[1], items: said(m[2]), more: m[3] ? ' ' + t('And {n} more.', {n: m[3]}) : ''})],
    [/^Item (\d[\d,]*): ([\s\S]+)$/, (m) => t('Item {item}: {text}', {item: m[1], text: said(m[2])})],
    [/^unit ([^()\n]+) \(([^()\n]+)\) is not reviewed: ([\s\S]+)$/, (m) => t('unit {unit} ({issuers}) is not reviewed: {reason}', {unit: m[1], issuers: m[2], reason: said(m[3])})],
    [/^([^;\n]+) is not reviewed: ([\s\S]+)$/, (m) => t('{issuer} is not reviewed: {reason}', {issuer: m[1], reason: said(m[2])})],
    [/^Not due until (.+)$/, (m) => t('Not due until {time}', {time: m[1]})],
    // V599, V620: a research update stopped for the network names each provider need it has, joined by "; ", then its
    // owner's way on (a closed workspace, the operator's switch, a run's hold, allowed now, unlocated) --
    // each read through its own key (a subject that is no need is carried as written), so the sentence reads whole
    [/^The research update needs (.+?)\. (.+)$/, (m) => t('The research update needs {subject}. ' + m[2], {subject: m[1].split('; ').map((need) => lookup(need)).join('；')})], // U93 (V620): every network way on
    // a Task's stop at an owner's code Task Control does not word (task_recovery.stop_detail's last sentence)
    // U73 (LS1, V459): the activation refusals a component or the book's support fills (plain_refusals.py's {subject})
    [/^A strategy runs forward from its book's last formation, decided from its first: run the book over its whole support \((.+)\) and activate that run\.$/, (m) => '策略从其账本的最后一个形成日向前运行，由第一个形成日起决定：请在完整区间（' + m[1].replace(' to ', ' 至 ') + '）上运行该账本，并激活那次运行。'],
    [/^The Alpha study behind the strategy's (.+) component does not read back; `study show` verifies it\.$/, (m) => '该策略 ' + m[1] + ' 组件背后的 Alpha 研究无法读回；`study show` 会验证它。'],
    [/^The training input the (.+) component's Alpha study read is no longer in this workspace: prepare and install the strategy again from current studies\.$/, (m) => m[1] + ' 组件的 Alpha 研究读取的训练输入已不在这个工作区：请用当前的研究重新准备并安装该策略。'],
    [/^The (.+) component's training input is not the one its study read, or ends before the book's next formation: prepare and install the strategy again\.$/, (m) => m[1] + ' 组件的训练输入不是其研究读取的那一个，或在账本的下一个形成日之前就已结束：请重新准备并安装该策略。'],
    [/^The (.+) component's models do not reach the book's next formation: prepare and install the strategy again from current studies\.$/, (m) => m[1] + ' 组件的模型到不了账本的下一个形成日：请用当前的研究重新准备并安装该策略。'],
    [/^Switch or create workspace · (.+)$/, (m) => '切换或创建工作区 · ' + m[1]],
    // UX2-02: a CRO assessment without the reviewer's summary (oversight submissions.py), read through `said`
    [/^(\d+) risks? named in the evidence read\.$/, (m) => '在已读证据中指出了 ' + m[1] + ' 项风险。'],
    // UX2-01: the packet's generated gap fragments (analysis contracts.py and routing.py), pluralized by the page
    [/^(\d[\d,]*) of (\d[\d,]*) unit needs? unread$/, (m) => m[2] + ' 个单元需求中有 ' + m[1] + ' 个未读'],
    [/^(\d[\d,]*) tables? not dealt a view$/, (m) => m[1] + ' 个表格未获得视图'],
    [/^(\d[\d,]*) tables? delivered in part$/, (m) => m[1] + ' 个表格仅部分交付'],
    [/^(\d[\d,]*) tables? whose page progress is unknown$/, (m) => m[1] + ' 个表格的页面进度未知'],
    [/^(\d[\d,]*) tables? unrenderable \(REPRESENTATION_GAP\)$/, (m) => t('{n} table(s) unrenderable (Representation gap)', {n: m[1]})],
    [/^(\d[\d,]*) tables? unrenderable \((.+)\)$/, (m) => m[1] + ' 个表格无法呈现（' + m[2] + '）'],
    [/^(\d[\d,]*) returned candidates? unread$/, (m) => m[1] + ' 个返回的候选未读'],
    [/^(\d[\d,]*) tables? in routed regions without a retained original$/, (m) => '路由区域中有 ' + m[1] + ' 个表格没有保留的原件'],
    [/^(\d[\d,]*) residual questions? skipped by the pair budget \((\d[\d,]*) of (\d[\d,]*) pairs spent\)$/, (m) => m[1] + ' 个余量问题因配对预算而跳过（已用 ' + m[2] + ' / ' + m[3] + ' 对）'],
    [/^(\d[\d,]*) more cell gap lines? in the routing record$/, (m) => '路由记录中还有 ' + m[1] + ' 行单元格缺口'],
    // V618: the Task queue's reason for its places (task_control.queue.waiting_places), read on Settings
    [/^auto: one waiting place per (\d+) of the (\d+) processors$/, (m) => t('auto: one waiting place per {k} of the {n} processors', {k: m[1], n: m[2]})],
    [/^set to (\d+)$/, (m) => t('set to {n}', {n: m[1]})],
    [/^Selected immutable input (\d{4}-\d{2}-\d{2}) and all saved studies\.$/, (m) => '已选不可变输入 ' + m[1] + ' 与全部历史研究保持不变。'],
    [/^One immutable input from current Data \/ Features through (\d{4}-\d{2}-\d{2})\.$/, (m) => '基于当前数据与特征创建截止至 ' + m[1] + ' 的不可变输入。'],
    [/^OBSERVATION (\d+) \/ (\d+)$/, (m) => `观测 ${m[1]} / ${m[2]}`],
    [/^(.+) · unchanged$/, (m) => m[1] + ' · 保持不变'],
    [/^(\d+) local destinations$/, (m) => m[1] + ' 个本地入口'],
    [/^(\d+) edits?$/, (m) => m[1] + ' 项变更'],
    [/^Factor development · (.+)$/, (m) => '因子开发 · ' + m[1]],
    [/^Input · (.+)$/, (m) => '输入 · ' + m[1]],
    [/^Holdings · (.+)$/, (m) => '持仓日期 · ' + m[1]],
    [/^Book · (.+)$/, (m) => 'Book · ' + m[1]],
    // U61 (V347): a result's time statements, generated by temporal_statement.py -- one template per mark value, its
    // values filled in; each is matched whole and read through its catalog line with placeholders
    [/^The Panel records no T0: every session holds the cohort of (\d+) listings it was built with\.$/, (m) => t('The Panel records no T0: every session holds the cohort of {n} listings it was built with.', {n: m[1]})],
    [/^The window starts (\d{4}-\d{2}-\d{2}), before T0 \((\d{4}-\d{2}-\d{2})\): until T0 every session holds the initial cohort of (\d+) listings\.$/, (m) => t('The window starts {start}, before T0 ({t0}): until T0 every session holds the initial cohort of {n} listings.', {start: m[1], t0: m[2], n: m[3]})],
    [/^The window starts (\d{4}-\d{2}-\d{2}), before T0 \((\d{4}-\d{2}-\d{2})\): until T0 every session holds the initial cohort\.$/, (m) => t('The window starts {start}, before T0 ({t0}): until T0 every session holds the initial cohort.', {start: m[1], t0: m[2]})],
    [/^The window starts at or after T0 \((\d{4}-\d{2}-\d{2})\), but its inputs reach back to (\d{4}-\d{2}-\d{2}), and before T0 every session holds the initial cohort of (\d+) listings\.$/, (m) => t('The window starts at or after T0 ({t0}), but its inputs reach back to {data}, and before T0 every session holds the initial cohort of {n} listings.', {t0: m[1], data: m[2], n: m[3]})],
    [/^The window starts at or after T0 \((\d{4}-\d{2}-\d{2})\), but its inputs reach back to (\d{4}-\d{2}-\d{2}), and before T0 every session holds the initial cohort\.$/, (m) => t('The window starts at or after T0 ({t0}), but its inputs reach back to {data}, and before T0 every session holds the initial cohort.', {t0: m[1], data: m[2]})],
    [/^The window and its inputs start at or after T0 \((\d{4}-\d{2}-\d{2})\): membership follows the entries and exits observed since\.$/, (m) => t('The window and its inputs start at or after T0 ({t0}): membership follows the entries and exits observed since.', {t0: m[1]})],
    [/^Every session uses the Sector classification observed on (\d{4}-\d{2}-\d{2})(, the sessions after T0 included)?; it is not point in time\.$/, (m) => t(m[2] ? 'Every session uses the Sector classification observed on {date}, the sessions after T0 included; it is not point in time.' : 'Every session uses the Sector classification observed on {date}; it is not point in time.', {date: m[1]})],
    [/^Each listing uses the Sector classification first recorded for it on every session before its first reclassification, which is not point in time; (\d+) reclassification\(s\) since (\d{4}-\d{2}-\d{2}) each apply from the session whose data update observed it, and no published session moved\.$/, (m) => t('Each listing uses the Sector classification first recorded for it on every session before its first reclassification, which is not point in time; {n} reclassification(s) since {date} each apply from the session whose data update observed it, and no published session moved.', {n: m[1], date: m[2]})],
    [/^Each listing uses the Sector classification first recorded for it on every session before its first reclassification, which is not point in time; (\d+) reclassification\(s\) since the rule's start each apply from the session whose data update observed it, and no published session moved\.$/, (m) => t("Each listing uses the Sector classification first recorded for it on every session before its first reclassification, which is not point in time; {n} reclassification(s) since the rule's start each apply from the session whose data update observed it, and no published session moved.", {n: m[1]})],
    [/^The (Sector treatment|price basis) `([^`]+)` has no installed statement; its mark is recorded above\.$/, (m) => t('The {kind} `{value}` has no installed statement; its mark is recorded above.', {kind: t(m[1]), value: m[2]})],
    // The Task drawer's permitted actions (round 24): the product's own sentences, interpolated by
    // task_recovery.py; each is matched whole and read through its catalog line with placeholders.
    [/^The Task has ended \((\w+)\); there is nothing to cancel\.$/, (m) => t('The Task has ended ({state}); there is nothing to cancel.', {state: t(m[1].replaceAll('_', ' '))})],
    [/^This Task only \((\w+)\)\. Verified stages, their artifacts and every published result stay exactly as recorded; nothing is deleted or recomputed\.$/, (m) => t('This Task only ({short}). Verified stages, their artifacts and every published result stay exactly as recorded; nothing is deleted or recomputed.', {short: m[1]})],
    [/^This same queued Task \((\w+)\); no new Task and no new declaration\. Nothing has run; it is handed back to the worker and starts once the workspace's active place is free\.$/, (m) => t("This same queued Task ({short}); no new Task and no new declaration. Nothing has run; it is handed back to the worker and starts once the workspace's active place is free.", {short: m[1]})],
    [/^This same Task \((\w+)\); no new Task and no new declaration\. Its (\d+) verified stage\(s\) and their artifacts are kept and re-verified from their evidence; only the unverified stage runs again\.$/, (m) => pluralText(m[2], 'This same Task ({short}); no new Task and no new declaration. Its {n} verified stage and its artifacts are kept and re-verified from their evidence; only the unverified stage runs again.', 'This same Task ({short}); no new Task and no new declaration. Its {n} verified stages and their artifacts are kept and re-verified from their evidence; only the unverified stage runs again.', {short: m[1], n: m[2]})],
    [/^The Task is (\w+); only an interrupted or parked queued Task can be resumed\.$/, (m) => t('The Task is {state}; only an interrupted or parked queued Task can be resumed.', {state: t(m[1].replaceAll('_', ' '))})],
    [/^This service has no recovery command for (\S+)\.$/, (m) => t('This service has no recovery command for {kind}.', {kind: m[1]})],
    [/^A new preview through (\w+), which records a plan and admits nothing\. This Task stays exactly as recorded: nothing here is changed, cancelled or deleted\.$/, (m) => t('A new preview through {preview}, which records a plan and admits nothing. This Task stays exactly as recorded: nothing here is changed, cancelled or deleted.', {preview: m[1]})],
    [/^(\w+) admits a Task and runs its actor; it is that owner's own confirmed admission, not a re-plan\. This Task stays exactly as recorded\.$/, (m) => t("{admitting} admits a Task and runs its actor; it is that owner's own confirmed admission, not a re-plan. This Task stays exactly as recorded.", {admitting: m[1]})],
    [/^Nothing is admitted until (\w+) is confirmed with that owner(, which runs an analyst or reviewer actor)?\. An identical declaration is reused exactly once this Task has succeeded \(no second Task or publication\); a changed one admits a new Task beside this one; while this Task runs or waits, the owner answers with its own disposition\.$/, (m) => t(m[2] ? 'Nothing is admitted until {admitting} is confirmed with that owner, which runs an analyst or reviewer actor. An identical declaration is reused exactly once this Task has succeeded (no second Task or publication); a changed one admits a new Task beside this one; while this Task runs or waits, the owner answers with its own disposition.' : 'Nothing is admitted until {admitting} is confirmed with that owner. An identical declaration is reused exactly once this Task has succeeded (no second Task or publication); a changed one admits a new Task beside this one; while this Task runs or waits, the owner answers with its own disposition.', {admitting: m[1]})],
    [/^Only (\w+), confirmed with its owner, admits work; it runs an actor and may publish\. No confirmation-free re-plan exists for this kind\.$/, (m) => t('Only {admitting}, confirmed with its owner, admits work; it runs an actor and may publish. No confirmation-free re-plan exists for this kind.', {admitting: m[1]})],
    [/^(\S+) is previewed through (\w+) and admitted through (\w+), each with that owner's own confirmation\.$/, (m) => t("{kind} is previewed through {preview} and admitted through {admitting}, each with that owner's own confirmation.", {kind: m[1], preview: m[2], admitting: m[3]})],
    [/^No confirmation-free preview exists for (\S+); (\w+) admits\.$/, (m) => t('No confirmation-free preview exists for {kind}; {admitting} admits.', {kind: m[1], admitting: m[2]})],
    // V573: a kind its owner admits directly, without a plan preview -- its re-plan is a new Task through that owner
    [/^A new Task through (\w+)\. This Task stays exactly as recorded: nothing here is changed, cancelled or deleted\.$/, (m) => t('A new Task through {admitting}. This Task stays exactly as recorded: nothing here is changed, cancelled or deleted.', {admitting: m[1]})],
    [/^Starts a new Task through (\w+); it does not resume or change this Task\.$/, (m) => t('Starts a new Task through {admitting}; it does not resume or change this Task.', {admitting: m[1]})],
    [/^The owner admits this Task kind directly through (\w+), without a plan preview\.$/, (m) => t('The owner admits this Task kind directly through {admitting}, without a plan preview.', {admitting: m[1]})],
    [/^Elapsed (.+)$/, (m) => '已用时 ' + m[1].replace(' · Retry ', ' · 重试 ').replace(' · illustrative', ' · 示例').replace('Stored artifact', '已保存制品').replace('shared application operation', '共用产品操作')],
    [/^(\d+) \/ (\d+) stages · stage-count fixture, not byte progress$/, (m) => `${m[1]} / ${m[2]} 个阶段 · 阶段计数示例，非字节进度`],
    [/^(\d+) \/ (\d+) named stages$/, (m) => `${m[1]} / ${m[2]} 个明确阶段`],
    [/^(.+) · (fixture|illustrative|synthetic)$/, (m) => m[1] + ' · ' + (m[2] === 'synthetic' ? '合成数据' : '示例')],
    [/^(.+) · (Running|Deferred|Succeeded|Cancelled|Ready|Blocked)$/, (m) => lookup(m[1]) + ' · ' + lookup(m[2])],
    [/^Running · (.+)$/, (m) => '运行中 · ' + lookup(m[1])],
    [/^(.+) · (Screen factors|Prepare market data|Replay history|Completed)$/, (m) => lookup(m[1]) + ' · ' + lookup(m[2])],
    [/^(\d+) fixture tasks · progress, waiting and recovery are separate\.$/, (m) => `${m[1]} 个示例任务 · 进度、等待与恢复分别展示。`],
    [/^EXISTING_PRODUCT_CAPABILITY · (.+)$/, (m) => 'EXISTING_PRODUCT_CAPABILITY · ' + lookup(m[1])],
    [/^REQUIRES_PRODUCT_IMPLEMENTATION · (.+)$/, (m) => 'REQUIRES_PRODUCT_IMPLEMENTATION · ' + lookup(m[1])],
    [/^(\d+) held · (\d+) exited$/, (m) => m[1] + ' 个持仓 · ' + m[2] + ' 个退出仓位'],
    [/^(.+) · stage progress$/, (m) => lookup(m[1]) + ' · 阶段进度'],
    [/^Workspace: (.+)$/, (m) => '工作区：' + m[1]],
    [/^(\d+) held reports?$/, (m) => '保留 ' + m[1] + ' 条报告'],
    [/^Reading held · (\d+) new reports? waiting$/, (m) => '阅读已固定 · 等待 ' + m[1] + ' 条新报告'],
    [/^(\d+) reports waiting$/, (m) => m[1] + ' 条报告待查看'],
    [/^(\d+) retained reports? · sample events$/, (m) => m[1] + ' 条保留报告 · 示例事件'],
    [/^Revised packet · r(\d+)$/, (m) => '修订材料包 · r' + m[1]],
    [/^Candidate association · (.+)$/, (m) => '候选关联 · ' + m[1]],
    [/^Response · (.+)$/, (m) => '退回回应 · ' + m[1]],
    [/^(\w+) \/ Northstar Asia-Pacific multi-source (Portfolio evidence comparison|cross-sectional factor development) — observation window and retained input (\d+)$/, (m) => `${m[1]} / Northstar 亚太多源${m[2] === 'Portfolio evidence comparison' ? '组合证据对照' : '截面因子开发'} · 观测区间与保留输入 ${m[3]}`],
  ];
  /* U76: a catalog key that holds an owner's `{subject}` (the code's suffix a refusal's words carry, as the door
   * fills it) matches the sentence the owner filled; its subject is carried into the Chinese as written. */
  let subjects = null, subjectsOf = 0;
  function subjectTemplates() {
    const keys = Object.keys(catalog);
    if (subjects && subjectsOf === keys.length) return subjects;
    subjectsOf = keys.length;
    subjects = keys.filter((k) => k.includes('{subject}') && !CONTEXT.test(k)).map((k) => [new RegExp('^' + k.split('{subject}').map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('(.*?)') + '$'), catalog[k]]);
    return subjects;
  }
  function templated(s) {
    for (const [re, fn] of TEMPLATES) {
      const m = s.match(re);
      if (m) return fn(m);
    }
    for (const [re, zh] of subjectTemplates()) {
      const m = s.match(re);
      if (m) { let i = 0; return zh.replace(/\{subject\}/g, () => m[++i] ?? ''); }
    }
    return s;
  }

  function interpolate(text, vars) {
    if (!vars) return text;
    return text.replace(/\{(\w+)\}/g, (match, name) => (vars[name] !== undefined ? vars[name] : match));
  }

  function t(key, vars) {
    seen.add(key.trim());
    const en = interpolate(key.replace(CONTEXT, ''), vars);
    if (locale !== 'zh-CN') return en;
    // Each line of a multi-line phrase is its own catalog entry; surrounding whitespace is kept.
    if (en.includes('<br>')) return en.split('<br>').map((line) => t(line)).join('<br>');
    const lead = en.match(/^\s*/)[0];
    const trail = en.match(/\s*$/)[0];
    const core = en.trim();
    const raw = key.trim();
    const plain = raw.replace(CONTEXT, '');
    let zh;
    if (Object.hasOwn(catalog, raw)) zh = interpolate(catalog[raw], vars);
    else if (Object.hasOwn(catalog, plain)) zh = interpolate(catalog[plain], vars);
    else if (Object.hasOwn(catalog, core)) zh = catalog[core];
    else {
      zh = templated(core);
      if (zh === core && /[A-Za-z]{2}/.test(core) && !/[\u3400-\u9fff]/.test(core)) missing.add(key);
    }
    return lead + zh + trail;
  }

  /* An owner's text that may be one of its generated sentences (UX2-01, UX2-02: the packet's gap fragments, a CRO
   * assessment without the reviewer's own summary): in Chinese, the catalog's words when it holds the sentence or a
   * template matches it, else the text as written -- an author's own words are no missing key. */
  function said(text) {
    const s = String(text ?? '');
    if (locale !== 'zh-CN') return s;
    const core = s.trim();
    return Object.hasOwn(catalog, core) ? catalog[core] : templated(core);
  }

  /* Choose between two full sentences without a catalog entry (used for plural forms). */
  const plural = (count, one, many, vars) => t(count === 1 ? one : many, Object.assign({n: count}, vars));

  function set(next) {
    locale = ['zh', 'cn', 'zh-CN'].includes(next) ? 'zh-CN' : 'en';
    document.documentElement.lang = locale;
    return locale;
  }
  /* The dictionary for a locale, loaded once (round 96). Returns null when nothing has to load
   * (English, or the dictionary already complete), else a promise that resolves once
   * `workbench.zh.js` has run -- whether the prelude's tag is already in flight or a tag is
   * added here. A failed load resolves too: the product paints in English rather than not at all. */
  let loading = null;
  function ensure(next) {
    const zh = ['zh', 'cn', 'zh-CN'].includes(next);
    if (!zh || window.ALPHA_ZH_READY) return null;
    if (loading) return loading;
    loading = new Promise((resolve) => {
      let tag = document.querySelector('script[data-alpha-zh]');
      if (!tag) {
        tag = document.createElement('script');
        tag.src = (window.ALPHA_ASSETS && window.ALPHA_ASSETS.zh) || '/workbench.zh.js';
        tag.async = false;
        tag.setAttribute('data-alpha-zh', 'loading');
        document.head.appendChild(tag);
      }
      const done = () => { tag.setAttribute('data-alpha-zh', window.ALPHA_ZH_READY ? 'ready' : 'failed'); loading = null; resolve(Boolean(window.ALPHA_ZH_READY)); };
      if (window.ALPHA_ZH_READY) done();
      else { tag.addEventListener('load', done, {once: true}); tag.addEventListener('error', done, {once: true}); }
    });
    return loading;
  }

  return {
    t,
    said,
    plural,
    set,
    ensure,
    get locale() {
      return locale;
    },
    get zh() {
      return locale === 'zh-CN';
    },
    catalog,
    untranslated: () => [...missing],
    keys: () => [...seen],
  };
})();
const t = I18N.t;
const said = I18N.said;
