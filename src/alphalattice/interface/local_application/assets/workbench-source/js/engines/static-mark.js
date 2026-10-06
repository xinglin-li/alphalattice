/* Static identity is 20 edges; the independently frozen animated Silk Flow stays 28. */
(function (G) {
  'use strict';
  const nodes = Object.freeze(
    Array.from({length: 8}, (_, i) =>
      Object.freeze({
        id: i + 1,
        x: Number((16 + 14 * Math.cos(-Math.PI / 2 + (i * Math.PI) / 4)).toFixed(5)),
        y: Number((16 + 14 * Math.sin(-Math.PI / 2 + (i * Math.PI) / 4)).toFixed(5)),
      }),
    ),
  );
  const edges = [];
  for (let i = 0; i < 8; i++)
    for (let j = i + 1; j < 8; j++) {
      const step = Math.min(j - i, 8 - j + i);
      if (step !== 2) edges.push(Object.freeze({i: i + 1, j: j + 1, step}));
    }
  Object.freeze(edges);
  function audit() {
    const counts = {1: 0, 2: 0, 3: 0, 4: 0},
      degrees = Array(8).fill(0);
    for (const e of edges) {
      counts[e.step]++;
      degrees[e.i - 1]++;
      degrees[e.j - 1]++;
    }
    const unique = new Set(edges.map((e) => e.i + '-' + e.j)).size;
    return {
      nodes: 8,
      edges: edges.length,
      unique_edges: unique,
      perimeter: counts[1],
      star: counts[3],
      diameters: counts[4],
      forbidden: counts[2],
      degrees,
      valid:
        edges.length === 20 &&
        unique === 20 &&
        counts[2] === 0 &&
        counts[1] === 8 &&
        counts[3] === 8 &&
        counts[4] === 4 &&
        degrees.every((x) => x === 5),
    };
  }
  if (!audit().valid) throw new Error('Static mark geometry failed.');
  function svg() {
    return (
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" data-mark-spec="octagram-20-v1" aria-hidden="true">' +
      edges
        .map((e) => {
          const a = nodes[e.i - 1],
            b = nodes[e.j - 1];
          return `<line data-edge="${e.i}-${e.j}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" opacity="${e.step === 1 ? 1 : 0.6}"/>`;
        })
        .join('') +
      '</svg>'
    );
  }
  G.AlphaStaticMark = Object.freeze({nodes, edges, svg, audit});
  if (typeof module !== 'undefined' && module.exports) module.exports = G.AlphaStaticMark;
})(typeof window !== 'undefined' ? window : globalThis);
