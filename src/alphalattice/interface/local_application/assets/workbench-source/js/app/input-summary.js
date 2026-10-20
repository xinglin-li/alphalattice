/* Research-input preview rows; LiveWorkspace renders this owner-supplied disclosure. */
const WorkspaceInputSummary = (() => {
  /* What changed between the published version and the prepared data, in the owner's source
   * fields. A rotated materializer or method identity is named as such: it is not new data. */
  const SOURCE_FIELDS={panel_through:'Features cutoff',data_revision_hash:'Market-data revision',manifest_revision:'Membership revision',panel_snapshot_hash:'Panel snapshot',outcome_watermark_hash:'Outcome watermark',outcome_recipe_hash:'Outcome method',outcome_policy_hash:'Outcome policy',materializer_hash:'Input materializer (source code)',binding_hash:'Data binding'};
  function sourceChanges(before, after) {
    return Object.keys(SOURCE_FIELDS).filter(k=>before && after && before[k]!==after[k]).map(k=>({key:k,label:SOURCE_FIELDS[k],before:before[k],after:after[k]}));
  }
  function captureSummary(p) {
    const changes=sourceChanges(p.before,p.after);
    const newData=changes.some(c=>['panel_through','data_revision_hash','panel_snapshot_hash','outcome_watermark_hash'].includes(c.key));
    const codeOnly=changes.length && !newData;
    const rows=[['Input family',p.input_id],['Previous version',mono(p.prior_binding_hash,SHORT.hash)]];
    for(const [key,label] of [['database_snapshot_bytes','Database snapshot (estimate)']]) if(p.summary?.[key]!=null) rows.push([label,html`${(Number(p.summary[key])/1e9).toFixed(3)} GB (${count(p.summary[key])} B)`]);
    if(p.summary?.write_policy) rows.push(['Storage',t(p.summary.write_policy)]);
    rows.push(['What changed',changes.length ? html`${changes.map(c=>html`${t(c.label)}: ${c.key==='panel_through' ? html`${c.before} → ${c.after}` : html`${mono(c.before)} → ${mono(c.after)}`}<br>`)}` : t('Nothing recorded as changed')]);
    rows.push(['Meaning',t(codeOnly ? 'The input materializer or method identity rotated; the market data itself is unchanged. The new version re-seals the same data under the current source code.' : newData ? 'The prepared data moved: the new version seals the current Features cutoff. Existing versions stay exactly as published.' : 'An exact version already exists; nothing is re-sealed.')]);
    rows.push(['Network',t('None ({n} calls); no model is trained ({m} numerical calls).',{n:p.network_calls ?? 0,m:p.numerical_calls ?? 0})]);
    rows.push(['Unchanged',t('Configured defaults, existing versions and saved research. Selecting the new version is a further explicit step.')]);
    return rows;
  }
  return {sourceChanges, captureSummary};
})();
