// Faster drop-in for makefun-diff 1.0.0.
//
// Same answer as makefun-diff's diffPatch on every input, by the rule in its
// header: lines cut after every "\n" only, longest common prefix then suffix,
// then Myers's greedy search (deletion preferred on ties, no edge pruning,
// every equal run followed as far as it goes), then hunks of up to 3 context
// lines with runs at most 6 kept lines apart sharing a hunk.
//
// Speed comes from (a) interning lines to integer ids so the search compares
// ints in typed arrays, (b) a flat V-trace (one Int32Array, no per-step path
// objects) with exact backtracking that recomputes each step's pre-snake
// point from the trace, (c) sentinel-terminated id arrays so the snakes need
// no bounds checks, and (d) a single string join for the patch.
//
// Lines are kept as the raw split pieces: line i of a text is pieces[i], and
// it carries a "\n" unless it is the text's last line and the text does not
// end with one. Two lines are equal only when their pieces match and their
// newline-ness matches.

const FORMAT = 'makefun-diff/1';
const CONTEXT = 3;

/**
 * The edit runs of the middle slices (integer line ids, each array carrying
 * a distinct sentinel id past its end so a snake stops there), in order:
 * [{ kind: 0 keep | 1 delete | 2 insert, count }].
 * Reproduces makefun-diff's editRuns exactly, including its tie-break.
 */
function editRuns(A, B) {
  const n = A.length - 1;
  const m = B.length - 1;
  if (n === 0 && m === 0) return [];
  const max = n + m;
  // Flat V-trace: slice for depth d starts at flat index d*d (since the
  // slices hold 1, 3, 5, ... entries), diagonal k at slice offset k + d.
  // Only even offsets of a slice are written (k has d's parity); the rest
  // are never read. -1 marks a diagonal no path reached.
  let flat = new Int32Array(1 << 24);
  const ensure = (need) => {
    if (need <= flat.length) return;
    let cap = flat.length;
    while (cap < need) cap *= 2;
    const grown = new Int32Array(cap);
    grown.set(flat);
    flat = grown;
  };

  // Depth 0: the snake from (0, 0). The sentinels stop it at either end.
  let x = 0;
  while (A[x] === B[x]) x++;
  flat[0] = x;
  let D = 0;
  let endK = 0;
  if (x < n || x < m) {
    let found = false;
    for (let d = 1; d <= max && !found; d++) {
      const prev = (d - 1) * (d - 1);
      ensure(d * d + 2 * d + 1);
      const woff = 2 * d - 1; // write index for diagonal k is p + woff
      let p = prev; // flat index of previous depth's diagonal k+1
      for (let k = -d; k <= d; k += 2, p += 2) {
        // Previous depth's diagonals k-1 and k+1, at flat[p - 2] and
        // flat[p]. Either is out of that depth's range at the loop's
        // edges, where no path reached it (-1).
        const x1 = k > -d + 1 ? flat[p - 2] : -1;
        const x2 = k < d - 1 ? flat[p] : -1;
        const canDel = x1 >= 0 && x1 < n;
        const canIns = x2 >= 0 && x2 - k - 1 < m;
        let xx;
        if (!canDel && !canIns) {
          flat[p + woff] = -1;
          continue;
        }
        if (canDel && (!canIns || x1 + 1 >= x2)) xx = x1 + 1;
        else xx = x2;
        let y = xx - k;
        while (A[xx] === B[y]) {
          xx++;
          y++;
        }
        flat[p + woff] = xx;
        if (xx >= n && y >= m) {
          D = d;
          endK = k;
          found = true;
          break;
        }
      }
    }
    if (!found) throw new Error('makefun-diff: the search ended without reaching both ends');
  }

  // Backtrack from (n, m) at depth D, diagonal endK. The pre-snake point of
  // each step is recomputed from the trace, so the keep counts are exact.
  const rev = [];
  let cx = n;
  let k = endK;
  for (let d = D; d >= 1; d--) {
    const prevBase = (d - 1) * (d - 1);
    const pm = d - 1; // trace[d-1] covers diagonals [-(d-1), d-1]
    const j1 = k - 1 + pm;
    const j2 = k + 1 + pm;
    const x1 = j1 >= 0 && j1 <= 2 * pm ? flat[prevBase + j1] : -1;
    const x2 = j2 >= 0 && j2 <= 2 * pm ? flat[prevBase + j2] : -1;
    const canDel = x1 >= 0 && x1 < n;
    const canIns = x2 >= 0 && x2 - (k + 1) < m;
    if (canDel && (!canIns || x1 + 1 >= x2)) {
      // Deletion: pre-snake point (x1 + 1, x1 - (k - 1)).
      const keep = cx - (x1 + 1);
      if (keep > 0) rev.push(0, keep);
      rev.push(1, 1);
      cx = x1;
      k = k - 1;
    } else {
      // Insertion: pre-snake point (x2, x2 - k).
      const keep = cx - x2;
      if (keep > 0) rev.push(0, keep);
      rev.push(2, 1);
      cx = x2;
      k = k + 1;
    }
  }
  // Depth 0 snake.
  if (cx > 0) rev.push(0, cx);
  // Reverse into forward order, merging adjacent runs of one kind.
  const runs = [];
  for (let i = rev.length - 2; i >= 0; i -= 2) {
    const kind = rev[i];
    const count = rev[i + 1];
    const last = runs.length ? runs[runs.length - 1] : null;
    if (last && last.kind === kind) last.count += count;
    else runs.push({ kind, count });
  }
  return runs;
}

/** make.fun's own patch of input.old into input.new. */
export function run(input) {
  if (input === null || typeof input !== 'object' || typeof input.old !== 'string' || typeof input.new !== 'string') {
    throw new TypeError('makefun-diff: the input is { old, new }, two strings');
  }
  const pa = input.old.split('\n');
  const pb = input.new.split('\n');
  const la = pa.length - 1;
  const lb = pb.length - 1;
  const n = pa[la] === '' ? la : pa.length;
  const m = pb[lb] === '' ? lb : pb.length;
  // endsNL: the text's last line carries a "\n" (false when the text is empty).
  const endsNLa = n > 0 && pa[la] === '';
  const endsNLb = m > 0 && pb[lb] === '';
  // The common prefix, then the common suffix of what is left. Lines are
  // equal only when their pieces match and their newline-ness matches.
  let prefix = 0;
  while (prefix < n && prefix < m && pa[prefix] === pb[prefix]
    && (prefix < n - 1 || endsNLa) === (prefix < m - 1 || endsNLb)) prefix++;
  let suffix = 0;
  while (suffix < n - prefix && suffix < m - prefix
    && pa[n - 1 - suffix] === pb[m - 1 - suffix]
    && (n - 1 - suffix < n - 1 || endsNLa) === (m - 1 - suffix < m - 1 || endsNLb)) suffix++;
  // Intern the middle lines to integer ids. Lines with a "\n" share one map;
  // a last line with no "\n" goes in a second map, so it never shares an id
  // with a line of the same text that has one.
  const idsNL = new Map();
  const idsNoNL = new Map();
  let nextId = 0;
  const nA = n - prefix - suffix;
  const nB = m - prefix - suffix;
  const A = new Int32Array(nA + 1);
  const B = new Int32Array(nB + 1);
  for (let i = 0; i < nA; i++) {
    const gi = prefix + i;
    const noNL = gi === n - 1 && !endsNLa;
    const map = noNL ? idsNoNL : idsNL;
    const s = pa[gi];
    let id = map.get(s);
    if (id === undefined) {
      id = nextId++;
      map.set(s, id);
    }
    A[i] = id;
  }
  for (let j = 0; j < nB; j++) {
    const gj = prefix + j;
    const noNL = gj === m - 1 && !endsNLb;
    const map = noNL ? idsNoNL : idsNL;
    const s = pb[gj];
    let id = map.get(s);
    if (id === undefined) {
      id = nextId++;
      map.set(s, id);
    }
    B[j] = id;
  }
  // Distinct sentinels past each end, so the snakes need no bounds checks.
  A[nA] = nextId++;
  B[nB] = nextId++;
  const mid = editRuns(A, B);
  const runs = [];
  if (prefix > 0) runs.push({ kind: 0, count: prefix });
  for (const r of mid) runs.push(r);
  if (suffix > 0) runs.push({ kind: 0, count: suffix });

  // The change runs: each a stretch of deletions and insertions between two kept lines.
  const changes = [];
  let o = 0;
  let p = 0;
  let deleted = 0;
  let inserted = 0;
  for (const r of runs) {
    if (r.kind === 0) {
      o += r.count;
      p += r.count;
      continue;
    }
    let c = changes.length ? changes[changes.length - 1] : undefined;
    if (c === undefined || c.o + c.dels !== o || c.p + c.ins !== p) {
      c = { o, p, dels: 0, ins: 0 };
      changes.push(c);
    }
    if (r.kind === 1) {
      c.dels += r.count;
      o += r.count;
      deleted += r.count;
    } else {
      c.ins += r.count;
      p += r.count;
      inserted += r.count;
    }
  }

  // The hunks: change runs at most 2 * CONTEXT kept lines apart share one.
  const hunks = [];
  for (const c of changes) {
    const h = hunks.length ? hunks[hunks.length - 1] : undefined;
    const prev = h ? h[h.length - 1] : undefined;
    if (prev !== undefined && c.o - (prev.o + prev.dels) <= 2 * CONTEXT) h.push(c);
    else hunks.push([c]);
  }
  // A line as the patch writes it: its op, the line, a newline, and "%" on a
  // line of its own when it had none.
  const written = (op, parts, endsNL, count, i) =>
    (i < count - 1 || endsNL) ? op + parts[i] + '\n' : op + parts[i] + '\n%\n';
  const out = [`${FORMAT} old=${n} new=${m} hunks=${hunks.length} deleted=${deleted} inserted=${inserted}\n`];
  for (const h of hunks) {
    const first = h[0];
    const last = h[h.length - 1];
    const oldStart = Math.max(0, first.o - CONTEXT);
    const newStart = first.p - (first.o - oldStart);
    const after = Math.min(CONTEXT, n - (last.o + last.dels));
    const oldEnd = last.o + last.dels + after;
    const newEnd = last.p + last.ins + after;
    out.push(`@ ${oldStart + 1} ${oldEnd - oldStart} ${newStart + 1} ${newEnd - newStart}\n`);
    let at = oldStart;
    for (const c of h) {
      for (; at < c.o; at++) out.push(written('=', pa, endsNLa, n, at));
      for (let i = 0; i < c.dels; i++) out.push(written('-', pa, endsNLa, n, c.o + i));
      for (let i = 0; i < c.ins; i++) out.push(written('+', pb, endsNLb, m, c.p + i));
      at = c.o + c.dels;
    }
    for (; at < oldEnd; at++) out.push(written('=', pa, endsNLa, n, at));
  }
  return out.join('');
}
