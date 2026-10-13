/* The committee page agenda, consumed by LiveTeam; no state or transport. */
const CommitteeAgenda = (() => {
  function render(b, selected) {
    const points = Lobby.render('committee-agenda', {items: b.tension_points, tools: false, columns: [{key: 'dissent', label: t('Dissent stands')}], axes: [{key: 'state', label: t('State'), group: p => ({key: p.state, label: p.state === 'BELOW_AGENDA' ? t('Below the agenda line, no stance required') : codeWords(p.state), open: true})}], row: p => objectRow({lead: 'evidence', name: p.alias, to: {action: 'team-committee-point', value: p.alias}, why: p.word ? t(p.word, {...p.words, ...(p.words?.reason ? {reason: declaredCodeWord(p.words.reason) ? codeWords(p.words.reason) : t(p.words.reason)} : {})}) : html`<span class="owner-text">${p.text}</span>`}, {key: p.alias, selected: selected === p.alias, columns: ['dissent'], props: [p.dissent_stands ? ['', stateLine('review_pending', {word: t('Dissent stands'), next: ''})] : ''], attrs: html`data-point="${p.alias}" data-state="${p.state}"`})});
    return html`${sectionHead(t('Agenda'))}${b.agenda ? html`<p class="caption" data-agenda-summary>${t('{n} agenda points; {below} below the agenda line, no stance required.', {n: b.agenda.aliases.length, below: b.agenda.below_count})} ${t(b.agenda.ranking)}${b.agenda.raised.length ? html` ${t('Raised by PM: {points}', {points: b.agenda.raised.join(', ')})}` : ''}</p>` : ''}${points}`;
  }
  return {render};
})();
