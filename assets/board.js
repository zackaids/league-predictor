// Generic bracket board. Renders columns -> boxes -> matches -> rows from a plain
// data object and reports clicks back to Python. All text goes through
// textContent: team names and notes are data, never markup.
//
// data = { columns: [{title, subtitle?, center?, boxes: [Box]}], edges: [{from, to, kind}] }
// Box   = { title?, subtitle?, matches?: [Match], likely?: [Row], likely_caption?, action?, note? }
// Match = { id, tag, stake?, rows: [Row, Row], footer?, pickable, decided? }
// Row   = { team, short, league, p?, state: open|picked|won|lost, score?, link? }
// Events: pick {match, team} | action {id} | team {name}
export default function (component) {
  const { data, setTriggerValue, parentElement } = component;

  const h = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  };
  const fire = (name, payload) => setTriggerValue(name, { ...payload, n: Math.random() });
  const pct = (p) => (p === null || p === undefined ? '' : p >= 0.995 ? '100%' : p < 0.0005 ? '0%' : p < 0.0995 ? (p * 100).toFixed(1) + '%' : Math.round(p * 100) + '%');

  let board = parentElement.querySelector('.board');
  if (!board) {
    board = h('div', 'board');
    parentElement.appendChild(board);
  }
  board.textContent = '';

  // Dark mode follows Streamlit's theme (its background color), not the OS.
  const bg = getComputedStyle(board).getPropertyValue('--st-background-color').trim();
  const lum = (s) => {
    let m = s.match(/^#([0-9a-f]{6})$/i), c;
    if (m) c = [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16));
    else if ((m = s.match(/rgba?\(([^)]+)\)/))) c = m[1].split(',').slice(0, 3).map((x) => parseFloat(x));
    return c ? (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255 : null;
  };
  const l = lum(bg);
  const dark = l === null ? window.matchMedia('(prefers-color-scheme: dark)').matches : l < 0.5;
  board.dataset.mode = dark ? 'dark' : 'light';

  const rowInner = (r) => {
    const dot = h('span', 'dot');
    const short = h('span', 'short', r.short || r.team);
    const name = h('span', 'name', r.team);
    name.appendChild(h('i', null, r.league || ''));
    const mark = r.state === 'won' ? '✓ ' : r.state === 'picked' ? '→ ' : '';
    const odds = h('span', 'odds', mark + (r.score ? r.score : pct(r.p)));
    return [dot, short, name, odds];
  };

  const rowEl = (match, r) => {
    const wrap = h('div', 'rw');
    const clickable = match && match.pickable && r.state !== 'won' && r.state !== 'lost';
    const main = h(clickable ? 'button' : 'div', 'row');
    main.dataset.state = r.state || 'open';
    main.dataset.lg = r.league || '';
    main.style.setProperty('--p', r.p === null || r.p === undefined ? 0 : r.p);
    rowInner(r).forEach((e) => main.appendChild(e));
    if (r.p !== null && r.p !== undefined && r.state !== 'won' && r.state !== 'lost') main.appendChild(h('span', 'bar'));
    if (clickable) {
      main.type = 'button';
      main.setAttribute('aria-pressed', r.state === 'picked' ? 'true' : 'false');
      main.setAttribute('aria-label', `${r.team}, ${pct(r.p)} to win. ${r.state === 'picked' ? 'Remove this pick' : 'Pick as winner'}`);
      main.onclick = () => fire('pick', { match: match.id, team: r.team });
    }
    wrap.appendChild(main);
    if (r.link) {
      const go = h('button', 'go', '↗');
      go.type = 'button';
      go.setAttribute('aria-label', `Open ${r.team} team page`);
      go.onclick = () => fire('team', { name: r.team });
      wrap.appendChild(go);
    }
    return wrap;
  };

  const matchEl = (m) => {
    const card = h('div', 'match' + (m.decided ? ' decided' : ''));
    card.dataset.id = m.id;
    const head = h('div', 'head');
    head.appendChild(h('b', null, m.tag || ''));
    head.appendChild(h('span', null, m.stake || ''));
    card.appendChild(head);
    m.rows.forEach((r) => card.appendChild(rowEl(m, r)));
    if (m.footer) card.appendChild(h('div', 'foot', m.footer));
    return card;
  };

  (data.columns || []).forEach((col) => {
    const c = h('div', 'col' + (col.center ? ' center' : ''));
    const title = h('h3', null, col.title);
    if (col.subtitle) title.appendChild(h('small', null, col.subtitle));
    c.appendChild(title);
    const body = h('div', 'colbody');
    c.appendChild(body);
    (col.boxes || []).forEach((b) => {
      const box = h('div', 'box');
      if (b.title) {
        const t = h('div', 'btitle', b.title);
        if (b.subtitle) t.appendChild(h('small', null, b.subtitle));
        box.appendChild(t);
      }
      (b.matches || []).forEach((m) => box.appendChild(matchEl(m)));
      if (b.likely && b.likely.length) {
        const lk = h('div', 'likely');
        b.likely.forEach((r) => lk.appendChild(rowEl(null, { ...r, state: 'open' })));
        if (b.likely_caption) lk.appendChild(h('div', 'cap', b.likely_caption));
        box.appendChild(lk);
      }
      if (b.action) {
        const a = h('button', 'action', b.action.label);
        a.type = 'button';
        a.disabled = !!b.action.disabled;
        if (!a.disabled) a.onclick = () => fire('action', { id: b.action.id });
        box.appendChild(a);
        if (b.action.hint) box.appendChild(h('div', 'hint', b.action.hint));
      }
      if (b.note) box.appendChild(h('div', 'note', b.note));
      body.appendChild(box);
    });
    board.appendChild(c);
  });

  // Connector lines, measured from the laid-out cards.
  const NS = 'http://www.w3.org/2000/svg';
  let svg = null;
  const drawEdges = () => {
    if (svg) svg.remove();
    if (!data.edges || !data.edges.length) return;
    svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('class', 'edges');
    svg.setAttribute('width', board.scrollWidth);
    svg.setAttribute('height', board.scrollHeight);
    const origin = board.getBoundingClientRect();
    const rel = (e) => {
      const r = e.getBoundingClientRect();
      return { l: r.left - origin.left + board.scrollLeft, r: r.right - origin.left + board.scrollLeft,
               y: r.top - origin.top + board.scrollTop + r.height / 2 };
    };
    data.edges.forEach((edge) => {
      const a = board.querySelector(`[data-id="${CSS.escape(edge.from)}"]`);
      const b = board.querySelector(`[data-id="${CSS.escape(edge.to)}"]`);
      if (!a || !b) return;
      const p = rel(a), q = rel(b), mid = (p.r + q.l) / 2;
      const path = document.createElementNS(NS, 'path');
      path.setAttribute('d', `M${p.r} ${p.y} H${mid} V${q.y} H${q.l}`);
      if (edge.kind === 'lose') path.setAttribute('class', 'lose');
      svg.appendChild(path);
    });
    board.appendChild(svg);
  };
  requestAnimationFrame(drawEdges);
  const ro = new ResizeObserver(() => requestAnimationFrame(drawEdges));
  ro.observe(board);
  return () => ro.disconnect();
}
