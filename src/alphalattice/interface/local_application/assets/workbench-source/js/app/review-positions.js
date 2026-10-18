/* Review delivery positions: LiveReview supplies the owner groups and collection reader.
   No position state is inferred from rounded display values. */
const ReviewPositions = (() => {
  function render(p, collection, percent) {
    if(!Array.isArray(p.position_rows)) return '';
    const groups = Object.hasOwn(p,'exit_rows')
      ? [['Holdings','positions',p.position_rows],['Exits (not held)','exits',p.exit_rows]]
      : [['Positions and changes','positions',p.position_rows]];
    const columns = [{label:t('Listing'),type:'text',absorb:true},
      {label:t('Weight'),unit:'%',type:'num'}, {label:t('Change'),unit:'bps',type:'num'}];
    const cells = r => [r.name || r.listing_id,
      num(percent(r.weight).replace(/%$/,'') || null,'ratio',2),
      Number.isFinite(Number(r.change)) ? signed(r.change,'ratio',2) : t(r.change || '')];
    const basis = (p.position_rows[0] || p.exit_rows?.[0])?.basis;
    const sections = groups.filter(([,key,rows])=>key==='positions' || rows?.length).map(([label,key,rows])=>
      html`${sectionHead(t(label))}${rows?.length
        ? collection('delivery-'+key,rows,columns,cells)
        : emptyState(t('No holdings for this date'))}`);
    return html`${basis ? html`<p class="caption">${said(basis)}</p>` : ''}${sections}`;
  }
  return {render};
})();
