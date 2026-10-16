/* A Session's members: those that took part in it -- a member speaking by
 * key or by hook, or a child whose first event falls within the Session's span, from its first to
 * its last record here (its product operations and its exchanges). Every other child of the same
 * native session is one line with its summed usage: never a row or a chip, and never hidden. An
 * unnamed thread reads as the shortest prefix of its id, eight characters at least, that no other
 * id on the page shares (as Git's short hashes). */
const HOST_ROLE = 'alphalattice_host';
const TeamMembers = (() => {
  const unique = (s, id, n = 8) => {
    while (n < id.length && [...s.participants.keys()].some((o) => o !== id && o.startsWith(id.slice(0, n)))) n++;
    return id.slice(0, n);
  };
  const rows = (s, aliases = {}) => [...s.participants.values()].flatMap((pt) => {
    const roles = [...new Set([...pt.roles].map((role) => aliases[role] || role))].filter((role) => role !== HOST_ROLE);
    if (!roles.length && pt.roles.has(HOST_ROLE)) return [];
    return (roles.length ? roles : [null]).map((role, i) => ({...pt, roles: new Set([...pt.roles].filter((r) => (aliases[r] || r) === role)), displayRole: role, sharedUsage: i > 0 && pt.usage.size > 0}));
  });
  const countOf = (s, aliases = {}) => rows(s, aliases).filter((pt) => !s.goalScope || [...pt.messages, ...pt.hooks].some((e) => e.subject.goal_id === s.goalScope && (!pt.displayRole || (aliases[e.role] || e.role) === pt.displayRole))).length;
  const activities = (pt, all, aliases = {}, kinds = {}) => {
    const mine = all.filter((e) => e.kind === 'message' && e.actor === pt.id && (!pt.displayRole || (aliases[e.role] || e.role) === pt.displayRole)), by = new Map();
    for (const e of mine) by.set(e.messageKind || '', (by.get(e.messageKind || '') || 0) + 1);
    return {mine, latest: mine.at(-1), made: [...by].map(([k, n]) => kinds[k] ? `${t(kinds[k][0])} × ${count(n)}` : countText(n, '{n} message', '{n} messages')).join(' · ')};
  };
  /* A member's own Session -- its own participant, never a child, holds a committee or review role
   * and sent that thread's message or answer -- reads as its part in that day's committee or
   * review, in declared words, never a goal's stored text. A lead (research_lead) never does. */
  const PARTS = [['COMMITTEE_SUBMIT', '{role}, committee of {date}', ['alphalattice_alpha', 'alphalattice_risk', 'alphalattice_cro']], ['AGENT_ANSWER_SUBMIT', '{role}, review of {date}', ['alphalattice_evidence_analyst', 'alphalattice_cro']]];
  const memberTitle = (s, all, aliases) => {
    const own = [...new Set([...s.participants.get(s.id)?.roles || []].map((r) => aliases[r] || r))], ops = new Set(s.facts.map((f) => f.item.payload?.operation));
    const d = new Date(all.map((e) => e.at).filter(Boolean).sort()[0] || ''), date = [d.getFullYear(), d.getMonth() + 1, d.getDate()].map((n) => String(n).padStart(2, '0')).join('-');
    const [, part, roles] = own.includes('research_lead') || isNaN(d) ? [] : PARTS.find(([op, , held]) => ops.has(op) && own.some((r) => held.includes(r))) || [];
    return part ? t(part, {role: own.filter((r) => roles.includes(r)).map(LiveTeam.roleName).join('/'), date}) : '';
  };
  function externalItems(X) {
    const merged = new Map(X.items);
    for (const g of LiveActivity.retained()) for (const item of g.items) if (item.schema_kind === 'ExternalActivityObserved') merged.set(item.observation_id, item);
    const usage = new Map(), usageKey = item => { const sub = item.payload?.subject || {}; return [sub.native_session_id, sub.native_agent_id, sub.model].join('|'); };
    for (const item of merged.values()) if (item.payload?.event_kind === 'NATIVE_AGENT_USAGE') { const key = usageKey(item), held = usage.get(key); if (!held || held.ordinal < item.ordinal) usage.set(key, item); }
    const items = [...merged.values()].filter(item => {
      if (item.payload?.event_kind !== 'NATIVE_AGENT_USAGE' || usage.get(usageKey(item)) === item) return true;
      X.items.delete(item.observation_id); return false;
    }).sort((a, b) => a.ordinal - b.ordinal);
    if (items.length > X.keep) { for (const item of items.splice(0, items.length - X.keep)) X.items.delete(item.observation_id); }
    return items;
  }
  function prune(sessions) {
    for (const s of sessions) {
      const times = [...s.facts.map((f) => f.item.occurred_at), ...[...s.entries.values()].map((e) => e.at)];
      const at = times.map((x) => Date.parse(x || '')).filter((x) => x).sort((a, b) => a - b);
      s.others = {count: 0, input: 0, output: 0};
      for (const [id, pt] of s.participants) {
        const first = Math.min(...[...pt.usage.values()].map((u) => u.first || Infinity));
        if (id === s.id || pt.messages.length || pt.hooks.length || (first >= at[0] && first <= at.at(-1))) continue;
        for (const u of pt.usage.values()) { s.others.input += u.input || 0; s.others.output += u.output || 0; }
        s.others.count++;
        s.participants.delete(id);
      }
    }
  }
  const othersLine = (s) => {
    const o = s.others || {};
    if (!o.count) return '';
    const words = [t('{n} in', {n: count(o.input)}), t('{n} out', {n: count(o.output)})].join(' · ');
    return noteLine(countText(o.count, '{n} other thread of this native session', '{n} other threads of this native session'), words, 'neutral');
  };
  return {unique, rows, countOf, activities, memberTitle, externalItems, prune, othersLine};
})();
