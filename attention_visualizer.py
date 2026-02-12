"""
Interactive Multi-Layer Attention Connectivity Graph for Jupyter Notebook.
Uses inline HTML/SVG/JS — no matplotlib needed.

Usage (in Jupyter):
    import numpy as np
    from attention_visualizer import show_attention_graph

    reaction_ids = ['PFK', 'PFL', ...]   # list of N short reaction IDs
    reaction_names = { 'PFK': 'Phosphofructokinase', ... }
    attention_tensors = [np.random.rand(4, 8, 150, 150) for _ in range(3)]
    input_bounds = np.random.rand(4, 150)

    show_attention_graph(reaction_ids, reaction_names, attention_tensors, input_bounds)
"""

import json
import random
import numpy as np
from IPython.display import display, HTML

_PALETTE = [
    ('#2a9d8f', '#1d7068', '#3fc0b0'),
    ('#c47a53', '#8e5535', '#dba07a'),
    ('#7b68ae', '#554889', '#9d8dcf'),
    ('#c97b84', '#9e545d', '#e4a5ad'),
    ('#8a9a5b', '#647040', '#a8bb78'),
    ('#6a7fdb', '#4a5aab', '#90a0ef'),
    ('#b07aa1', '#875c7d', '#cf9ec2'),
    ('#c4a35a', '#9a7d3a', '#dfbf7a'),
    ('#56b6a6', '#3d8f82', '#78d4c5'),
    ('#d4816b', '#a85d4a', '#eba591'),
    ('#6b9e8a', '#4c7a67', '#8ec2ac'),
    ('#a0855b', '#7a6340', '#c0a57b'),
]


def show_attention_graph(reaction_ids, reaction_names_dict, attention_tensors, input_bounds,
                         node_h=8, gap=1):
    L = len(attention_tensors)
    B, H, N, _ = attention_tensors[0].shape
    assert N == len(reaction_ids)
    input_bounds = np.array(input_bounds, dtype=float)
    assert input_bounds.shape == (B, N)

    attn_data = [np.round(t, 6).tolist() for t in attention_tensors]
    bounds_data = np.round(input_bounds, 6).tolist()
    full_names = [reaction_names_dict.get(rid, rid) for rid in reaction_ids]

    uid = f"av{random.randint(100000, 999999)}"
    ids_json = json.dumps(reaction_ids)
    fullnames_json = json.dumps(full_names)
    attn_json = json.dumps(attn_data)
    bounds_json = json.dumps(bounds_data)

    num_cols = L + 1
    col_colors = [_PALETTE[i % len(_PALETTE)] for i in range(num_cols)]

    nameBlockX = 5
    nameBlockW = 200
    guideGap = 25
    nodeW = 22
    colSpacing = 175
    sliderGroupW = 155

    col0_x = nameBlockX + nameBlockW + guideGap
    colX = [col0_x + i * colSpacing for i in range(L + 1)]
    total_width = colX[-1] + nodeW + 30

    topPad = 50
    botPad = 15
    svg_height = topPad + N * (node_h + gap) + botPad
    sliderAreaH = 80

    bandX = nameBlockX
    bandW = colX[-1] + nodeW - nameBlockX

    col_css = ""
    for c, (fill, stroke, hover) in enumerate(col_colors):
        col_css += f"""
        .{uid}-ncol{c} {{ fill:{fill}; stroke:{stroke}; stroke-width:0.3; cursor:pointer; }}
        .{uid}-ncol{c}:hover {{ fill:{hover}; }}"""

    search_html = f"""
    <div id="{uid}-search-wrap" style="position:absolute; left:{nameBlockX}px; top:0px; width:{nameBlockW}px; z-index:10;">
      <div style="position:relative;">
        <input id="{uid}-search" type="text" placeholder="Search reaction..."
               autocomplete="off"
               style="width:100%; box-sizing:border-box; padding:3px 24px 3px 8px;
                      font-size:11px; border:1px solid #bbb; border-radius:4px;
                      font-family:monospace; outline:none;">
        <span id="{uid}-search-clear"
              style="position:absolute; right:6px; top:50%; transform:translateY(-50%);
                     cursor:pointer; font-size:14px; color:#999; display:none;
                     user-select:none; line-height:1;">&times;</span>
        <div id="{uid}-dropdown"
             style="position:absolute; left:0; right:0; top:100%; max-height:180px;
                    overflow-y:auto; background:#fff; border:1px solid #bbb;
                    border-top:none; border-radius:0 0 4px 4px; display:none;
                    box-shadow:0 3px 8px rgba(0,0,0,0.15); z-index:20;">
        </div>
      </div>
    </div>
    """

    batch_slider_html = f"""
    <div style="position:absolute; left:{nameBlockX}px; top:24px; width:{nameBlockW}px;
                font-size:11px; color:#555; background:#f5f5f5; border:1px solid #ddd;
                border-radius:4px; padding:4px 8px; box-sizing:border-box;">
      <div style="display:flex; align-items:center; gap:4px;">
        <span style="font-weight:bold; min-width:50px;">Batch:</span>
        <input id="{uid}-batch" type="range" min="0" max="{B - 1}" value="0"
               style="flex:1; height:14px; cursor:pointer;">
        <span id="{uid}-bval" style="min-width:20px; text-align:right; font-weight:bold;">0</span>
      </div>
    </div>
    """

    layer_sliders_html = ""
    for l in range(L):
        center = (colX[l] + nodeW + colX[l + 1]) / 2.0
        left = center - sliderGroupW / 2.0
        layer_sliders_html += f"""
        <div style="position:absolute; left:{left:.0f}px; top:6px; width:{sliderGroupW}px;
                     font-size:10px; color:#555; background:#f5f5f5; border:1px solid #ddd;
                     border-radius:4px; padding:3px 6px; box-sizing:border-box;">
          <div style="font-weight:bold; text-align:center; margin-bottom:2px; color:#333;">Layer {l}</div>
          <div style="display:flex; align-items:center; gap:2px;">
            <span style="min-width:14px;">H:</span>
            <input id="{uid}-head-{l}" type="range" min="0" max="{H - 1}" value="0"
                   style="flex:1; height:12px; cursor:pointer;" data-layer="{l}">
            <span id="{uid}-hval-{l}" style="min-width:18px; text-align:right;">0</span>
          </div>
          <div style="display:flex; align-items:center; gap:2px;">
            <span style="min-width:14px;">T:</span>
            <input id="{uid}-thresh-{l}" type="range" min="0" max="1000" value="950"
                   style="flex:1; height:12px; cursor:pointer;" data-layer="{l}">
            <span id="{uid}-tval-{l}" style="min-width:50px; text-align:right; font-size:9px;"></span>
          </div>
        </div>
        """

    html = f"""
<div id="{uid}-wrapper" style="width:{total_width}px; font-family:sans-serif;">
  <div style="position:relative; height:{sliderAreaH}px; overflow:visible;">
    {search_html}
    {batch_slider_html}
    {layer_sliders_html}
  </div>
  <div id="{uid}-scroll" style="width:{total_width}px; overflow-y:auto; border:1px solid #ccc; border-radius:0 0 6px 6px; background:#fafafa;">
    <svg id="{uid}-svg" width="{total_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
      <style>
        {col_css}
        .{uid}-hl {{ fill:none; stroke:#ff69b4; stroke-width:2; pointer-events:none; rx:2; ry:2; }}
        .{uid}-conn {{ stroke:#ff69b4; pointer-events:none; }}
        .{uid}-tl {{ stroke:#aaa; pointer-events:none; }}
        .{uid}-tt {{ pointer-events:none; }}
        .{uid}-vl {{ pointer-events:none; font-size:9px; fill:#c0458a; font-weight:bold; }}
        .{uid}-nbg {{ fill:#1a1a1a; rx:1; ry:1; }}
        .{uid}-nred {{ fill:#cc2222; rx:1; ry:1; pointer-events:none; }}
        .{uid}-nbar {{ fill:transparent; stroke:none; cursor:pointer; rx:1; ry:1; }}
        .{uid}-nbar:hover {{ stroke:#ffcc00; stroke-width:1.2; }}
        .{uid}-ntxt {{ fill:#fff; font-size:{max(node_h - 2, 5)}px; font-family:monospace; pointer-events:none; }}
        .{uid}-guide {{ stroke:#aaa; stroke-width:0.5; stroke-dasharray:2,2; pointer-events:none; }}
        .{uid}-rowband {{ fill:#d0d0d0; opacity:0.35; pointer-events:none; }}
        .{uid}-pinband {{ fill:#ffe066; opacity:0.38; pointer-events:none; }}
      </style>
    </svg>
  </div>
</div>

<script>
(function() {{
  const uid = "{uid}";
  const ids = {ids_json};
  const fullNames = {fullnames_json};
  const attnData = {attn_json};
  const boundsData = {bounds_json};
  const L = {L}, B = {B}, H = {H}, N = {N};
  const nodeH = {node_h}, nodeW = {nodeW}, gap = {gap};
  const nameBlockX = {nameBlockX}, nameBlockW = {nameBlockW};
  const topPad = {topPad};
  const colX = {json.dumps(colX)};
  const bandX = {bandX}, bandW = {bandW};
  const MAX_TL = 600;

  const svgEl = document.getElementById(uid + '-svg');
  const scrollEl = document.getElementById(uid + '-scroll');
  const ns = 'http://www.w3.org/2000/svg';

  const yPos = [];
  for (let i = 0; i < N; i++) yPos.push(topPad + i * (nodeH + gap));
  function cy(i) {{ return yPos[i] + nodeH / 2; }}

  function elId(id) {{ return document.getElementById(uid + '-' + id); }}
  function curBatch() {{ return parseInt(elId('batch').value); }}

  // ---- SVG groups ----
  const pinBandGroup = document.createElementNS(ns, 'g'); svgEl.appendChild(pinBandGroup);
  const bandGroup    = document.createElementNS(ns, 'g'); svgEl.appendChild(bandGroup);
  const threshGroups = [];
  for (let l = 0; l < L; l++) {{
    const g = document.createElementNS(ns, 'g'); svgEl.appendChild(g); threshGroups.push(g);
  }}
  const connGroup = document.createElementNS(ns, 'g'); svgEl.appendChild(connGroup);
  const nodeGroup = document.createElementNS(ns, 'g'); svgEl.appendChild(nodeGroup);

  // ---- Hover row band ----
  const rowBand = document.createElementNS(ns, 'rect');
  rowBand.classList.add(uid + '-rowband');
  rowBand.setAttribute('x', bandX); rowBand.setAttribute('width', bandW);
  rowBand.setAttribute('height', nodeH); rowBand.setAttribute('visibility', 'hidden');
  bandGroup.appendChild(rowBand);
  let activeRowIdx = -1;
  function showRowBand(idx) {{
    if (idx === activeRowIdx) return;
    activeRowIdx = idx;
    rowBand.setAttribute('y', yPos[idx]); rowBand.setAttribute('visibility', 'visible');
  }}
  function hideRowBand() {{
    if (activeRowIdx < 0) return;
    activeRowIdx = -1; rowBand.setAttribute('visibility', 'hidden');
  }}

  // ---- Pinned search/click band ----
  let pinnedIdx = -1, pinnedRect = null;
  const searchInput = elId('search');
  const clearBtn = elId('search-clear');

  function pinRow(idx) {{
    unpinRow(); pinnedIdx = idx;
    pinnedRect = document.createElementNS(ns, 'rect');
    pinnedRect.classList.add(uid + '-pinband');
    pinnedRect.setAttribute('x', bandX); pinnedRect.setAttribute('y', yPos[idx]);
    pinnedRect.setAttribute('width', bandW); pinnedRect.setAttribute('height', nodeH);
    pinBandGroup.appendChild(pinnedRect);
    // Sync search box
    searchInput.value = ids[idx];
    clearBtn.style.display = 'inline';
  }}
  function unpinRow() {{
    if (pinnedRect) {{ pinnedRect.remove(); pinnedRect = null; }} pinnedIdx = -1;
  }}

  // ---- Name bars ----
  const redRects = [];
  for (let i = 0; i < N; i++) {{
    const y = yPos[i];
    const bg = document.createElementNS(ns, 'rect');
    bg.setAttribute('x', nameBlockX); bg.setAttribute('y', y);
    bg.setAttribute('width', nameBlockW); bg.setAttribute('height', nodeH);
    bg.classList.add(uid + '-nbg'); nodeGroup.appendChild(bg);

    const red = document.createElementNS(ns, 'rect');
    red.setAttribute('x', nameBlockX); red.setAttribute('y', y);
    red.setAttribute('width', 0); red.setAttribute('height', nodeH);
    red.classList.add(uid + '-nred'); nodeGroup.appendChild(red); redRects.push(red);

    const txt = document.createElementNS(ns, 'text');
    txt.setAttribute('x', nameBlockX + 3); txt.setAttribute('y', y + nodeH - 1.5);
    txt.classList.add(uid + '-ntxt'); txt.textContent = ids[i]; nodeGroup.appendChild(txt);

    const hit = document.createElementNS(ns, 'rect');
    hit.setAttribute('x', nameBlockX); hit.setAttribute('y', y);
    hit.setAttribute('width', nameBlockW); hit.setAttribute('height', nodeH);
    hit.classList.add(uid + '-nbar'); hit.dataset.nbar = i; nodeGroup.appendChild(hit);

    const gl = document.createElementNS(ns, 'line');
    gl.setAttribute('x1', nameBlockX + nameBlockW); gl.setAttribute('y1', cy(i));
    gl.setAttribute('x2', colX[0]); gl.setAttribute('y2', cy(i));
    gl.classList.add(uid + '-guide'); nodeGroup.appendChild(gl);
  }}

  function updateRedFills() {{
    const b = curBatch(); const bv = boundsData[b];
    let mx = -Infinity;
    for (let i = 0; i < N; i++) if (Math.abs(bv[i]) > mx) mx = Math.abs(bv[i]);
    if (mx <= 0) mx = 1;
    for (let i = 0; i < N; i++) redRects[i].setAttribute('width', (Math.abs(bv[i]) / mx) * nameBlockW);
  }}
  updateRedFills();

  // ---- Node columns ----
  for (let col = 0; col <= L; col++) {{
    for (let i = 0; i < N; i++) {{
      const r = document.createElementNS(ns, 'rect');
      r.setAttribute('x', colX[col]); r.setAttribute('y', yPos[i]);
      r.setAttribute('width', nodeW); r.setAttribute('height', nodeH);
      r.setAttribute('rx', 1); r.setAttribute('ry', 1);
      r.classList.add(uid + '-ncol' + col);
      r.dataset.col = col; r.dataset.idx = i; nodeGroup.appendChild(r);
    }}
  }}

  // ---- Tooltip ----
  const ttRect = document.createElementNS(ns, 'rect');
  ttRect.classList.add(uid + '-tt');
  ttRect.setAttribute('rx', 4); ttRect.setAttribute('ry', 4);
  ttRect.setAttribute('fill', '#ffffdd'); ttRect.setAttribute('stroke', '#999');
  ttRect.setAttribute('stroke-width', 0.8); ttRect.setAttribute('visibility', 'hidden');
  svgEl.appendChild(ttRect);
  const ttText1 = document.createElementNS(ns, 'text');
  ttText1.classList.add(uid + '-tt');
  ttText1.setAttribute('font-size', '11'); ttText1.setAttribute('fill', '#333');
  ttText1.setAttribute('visibility', 'hidden'); svgEl.appendChild(ttText1);
  const ttText2 = document.createElementNS(ns, 'text');
  ttText2.classList.add(uid + '-tt');
  ttText2.setAttribute('font-size', '10'); ttText2.setAttribute('fill', '#888');
  ttText2.setAttribute('visibility', 'hidden'); svgEl.appendChild(ttText2);

  function showTooltip(x, y, anchor, line1, line2) {{
    ttText1.setAttribute('x', x); ttText1.setAttribute('y', y);
    ttText1.setAttribute('text-anchor', anchor);
    ttText1.textContent = line1; ttText1.setAttribute('visibility', 'visible');
    if (line2 !== null) {{
      ttText2.setAttribute('x', x); ttText2.setAttribute('y', y + 13);
      ttText2.setAttribute('text-anchor', anchor);
      ttText2.textContent = line2; ttText2.setAttribute('visibility', 'visible');
    }} else {{ ttText2.setAttribute('visibility', 'hidden'); }}
    const b1 = ttText1.getBBox();
    let bx = b1.x, by = b1.y, bw = b1.width, bh = b1.height;
    if (line2 !== null) {{
      const b2 = ttText2.getBBox();
      bx = Math.min(bx, b2.x); by = Math.min(by, b2.y);
      bw = Math.max(b1.x + b1.width, b2.x + b2.width) - bx;
      bh = (b2.y + b2.height) - by;
    }}
    ttRect.setAttribute('x', bx - 5); ttRect.setAttribute('y', by - 3);
    ttRect.setAttribute('width', bw + 10); ttRect.setAttribute('height', bh + 6);
    ttRect.setAttribute('visibility', 'visible');
  }}
  function hideTooltip() {{
    ttText1.setAttribute('visibility', 'hidden');
    ttText2.setAttribute('visibility', 'hidden');
    ttRect.setAttribute('visibility', 'hidden');
  }}

  // ---- Threshold ----
  function getMatrix(l) {{
    return attnData[l][curBatch()][parseInt(elId('head-' + l).value)];
  }}
  function updateThreshLines(l) {{
    const g = threshGroups[l];
    while (g.firstChild) g.removeChild(g.firstChild);
    const m = getMatrix(l);
    let mn = Infinity, mx = -Infinity;
    for (let r = 0; r < N; r++)
      for (let c = 0; c < N; c++) {{
        const v = Math.abs(m[r][c]);
        if (v < mn) mn = v; if (v > mx) mx = v;
      }}
    const sv = parseInt(elId('thresh-' + l).value);
    const thresh = mn + (sv / 1000) * (mx - mn);
    elId('tval-' + l).textContent = thresh.toExponential(2);
    const entries = [];
    for (let r = 0; r < N; r++)
      for (let c = 0; c < N; c++) {{
        const v = Math.abs(m[r][c]);
        if (v >= thresh) entries.push({{ r, c, v }});
      }}
    entries.sort((a, b) => b.v - a.v);
    const drawn = entries.slice(0, MAX_TL);
    const range = mx - thresh;
    const x1 = colX[l] + nodeW, x2 = colX[l + 1];
    for (const e of drawn) {{
      let alpha = 0.08;
      if (range > 0) alpha = 0.05 + ((e.v - thresh) / range) * 0.45;
      const line = document.createElementNS(ns, 'line');
      line.setAttribute('x1', x1); line.setAttribute('y1', cy(e.r));
      line.setAttribute('x2', x2); line.setAttribute('y2', cy(e.c));
      line.setAttribute('stroke-width', 0.8);
      line.setAttribute('stroke-opacity', alpha);
      line.classList.add(uid + '-tl'); g.appendChild(line);
    }}
  }}
  function updateAllThresh() {{ for (let l = 0; l < L; l++) updateThreshLines(l); }}

  // ============================================================
  //  CHAIN
  // ============================================================
  let chain = [];
  let hlRects = [];
  let segments = [];

  function clearChain() {{
    hlRects.forEach(r => r.remove());
    segments.forEach(s => s.forEach(a => a.remove()));
    hlRects = []; segments = []; chain = [];
  }}

  function makeHL(col, idx) {{
    const pad = 3;
    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('x', colX[col] - pad);
    rect.setAttribute('y', yPos[idx] - pad);
    rect.setAttribute('width', nodeW + pad * 2);
    rect.setAttribute('height', nodeH + pad * 2);
    rect.classList.add(uid + '-hl');
    svgEl.appendChild(rect);
    return rect;
  }}

  function makeSegment(keyCol, keyIdx, queryIdx) {{
    const l = keyCol;
    const m = getMatrix(l);
    const val = m[keyIdx][queryIdx];
    let rmin = Infinity, rmax = -Infinity;
    for (let c = 0; c < N; c++) {{
      const v = Math.abs(m[keyIdx][c]);
      if (v < rmin) rmin = v; if (v > rmax) rmax = v;
    }}
    let normed = 0.5;
    if (rmax > rmin) normed = (Math.abs(val) - rmin) / (rmax - rmin);
    const lw = 0.3 + normed * 7.7;
    const alpha = 0.2 + normed * 0.45;
    const arts = [];

    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', colX[l] + nodeW); line.setAttribute('y1', cy(keyIdx));
    line.setAttribute('x2', colX[l + 1]); line.setAttribute('y2', cy(queryIdx));
    line.setAttribute('stroke-width', lw); line.setAttribute('stroke-opacity', alpha);
    line.classList.add(uid + '-conn');
    connGroup.appendChild(line); arts.push(line);

    const midx = (colX[l] + nodeW + colX[l + 1]) / 2;
    const midy = (cy(keyIdx) + cy(queryIdx)) / 2;
    const lbl = document.createElementNS(ns, 'text');
    lbl.setAttribute('x', midx); lbl.setAttribute('y', midy - 4);
    lbl.setAttribute('text-anchor', 'middle');
    lbl.classList.add(uid + '-vl');
    lbl.textContent = val.toExponential(3);
    svgEl.appendChild(lbl); arts.push(lbl);
    return arts;
  }}

  function redrawSegmentAt(i) {{
    if (i < 0 || i >= segments.length) return;
    segments[i].forEach(a => a.remove());
    segments[i] = makeSegment(chain[i].col, chain[i].idx, chain[i + 1].idx);
  }}

  function redrawAllSegments() {{
    for (let i = 0; i < segments.length; i++) {{
      segments[i].forEach(a => a.remove());
      segments[i] = makeSegment(chain[i].col, chain[i].idx, chain[i + 1].idx);
    }}
  }}

  function redrawSegmentsForLayer(l) {{
    for (let i = 0; i < segments.length; i++) {{
      if (chain[i].col === l) redrawSegmentAt(i);
    }}
  }}

  function extend(col, idx) {{
    const prev = chain[chain.length - 1];
    chain.push({{ col, idx }});
    hlRects.push(makeHL(col, idx));
    segments.push(makeSegment(prev.col, prev.idx, idx));
  }}

  function truncateFrom(pos) {{
    while (hlRects.length > pos) hlRects.pop().remove();
    const segCut = Math.max(0, pos - 1);
    while (segments.length > segCut) segments.pop().forEach(a => a.remove());
    chain.length = pos;
  }}

  function replaceAt(pos, col, idx) {{
    hlRects[pos].remove();
    hlRects[pos] = makeHL(col, idx);
    if (pos > 0) {{
      segments[pos - 1].forEach(a => a.remove());
      segments[pos - 1] = makeSegment(chain[pos - 1].col, chain[pos - 1].idx, idx);
    }}
    if (pos < chain.length - 1) {{
      segments[pos].forEach(a => a.remove());
      segments[pos] = makeSegment(col, idx, chain[pos + 1].idx);
    }}
    chain[pos] = {{ col, idx }};
  }}

  // ---- Slider wiring ----
  elId('batch').addEventListener('input', function() {{
    elId('bval').textContent = this.value;
    updateRedFills(); updateAllThresh(); redrawAllSegments();
  }});
  for (let l = 0; l < L; l++) {{
    (function(layer) {{
      elId('head-' + layer).addEventListener('input', function() {{
        elId('hval-' + layer).textContent = this.value;
        updateThreshLines(layer); redrawSegmentsForLayer(layer);
      }});
      elId('thresh-' + layer).addEventListener('input', function() {{
        updateThreshLines(layer);
      }});
    }})(l);
  }}
  updateAllThresh();

  // ---- Click handler (nodes) ----
  svgEl.addEventListener('click', function(e) {{
    const t = e.target;

    // --- Name bar click: pin/unpin row + sync search ---
    if (t.dataset && t.dataset.nbar !== undefined) {{
      const idx = parseInt(t.dataset.nbar);
      if (pinnedIdx === idx) {{
        // Same bar clicked → unpin and clear search
        unpinRow();
        searchInput.value = '';
        clearBtn.style.display = 'none';
      }} else {{
        pinRow(idx);  // pinRow also sets searchInput.value
      }}
      return;
    }}

    // --- Node column click: chain logic ---
    if (!t.dataset || t.dataset.col === undefined) return;
    const col = parseInt(t.dataset.col);
    const idx = parseInt(t.dataset.idx);

    if (chain.length === 0) {{
      chain.push({{ col, idx }});
      hlRects.push(makeHL(col, idx));
      return;
    }}

    const startCol = chain[0].col;
    const pos = col - startCol;

    if (pos === chain.length && chain[chain.length - 1].col < L) {{
      extend(col, idx); return;
    }}

    if (pos >= 0 && pos < chain.length) {{
      if (chain[pos].idx === idx) {{
        truncateFrom(pos);
      }} else {{
        replaceAt(pos, col, idx);
      }}
      return;
    }}

    clearChain();
    chain.push({{ col, idx }});
    hlRects.push(makeHL(col, idx));
  }});

  // ---- Hover ----
  svgEl.addEventListener('mousemove', function(e) {{
    const t = e.target;
    if (t.dataset && t.dataset.nbar !== undefined) {{
      const idx = parseInt(t.dataset.nbar);
      showRowBand(idx);
      const rx = parseFloat(t.getAttribute('x'));
      const ry = parseFloat(t.getAttribute('y'));
      const bv = boundsData[curBatch()][idx];
      showTooltip(rx + nameBlockW + 8, ry + nodeH / 2, 'start',
                  fullNames[idx], 'Bound: ' + bv.toExponential(3));
      return;
    }}
    if (t.dataset && t.dataset.col !== undefined) {{
      const idx = parseInt(t.dataset.idx);
      showRowBand(idx);
      const rx = parseFloat(t.getAttribute('x'));
      const ry = parseFloat(t.getAttribute('y'));
      showTooltip(rx + nodeW + 8, ry + nodeH / 2 + 4, 'start', fullNames[idx], null);
      return;
    }}
    hideRowBand(); hideTooltip();
  }});

  // ============================================================
  //  SEARCH BAR
  // ============================================================
  const dropdown = elId('dropdown');
  const idsLower = ids.map(s => s.toLowerCase());
  const fullNamesLower = fullNames.map(s => s.toLowerCase());

  function getMatches(query) {{
    if (!query) return [];
    const q = query.toLowerCase();
    const scored = [];
    for (let i = 0; i < N; i++) {{
      let score = -1;
      if (idsLower[i].startsWith(q)) score = 3;
      else if (idsLower[i].includes(q)) score = 2;
      else if (fullNamesLower[i].includes(q)) score = 1;
      if (score > 0) scored.push({{ idx: i, score }});
    }}
    scored.sort((a, b) => b.score - a.score || a.idx - b.idx);
    return scored.slice(0, 12);
  }}

  function renderDropdown(matches) {{
    dropdown.innerHTML = '';
    if (matches.length === 0) {{ dropdown.style.display = 'none'; return; }}
    dropdown.style.display = 'block';
    for (const m of matches) {{
      const div = document.createElement('div');
      div.style.cssText = 'padding:4px 8px; cursor:pointer; font-size:11px; font-family:monospace; border-bottom:1px solid #eee; color:#333;';
      div.innerHTML = '<b style="color:#000;">' + ids[m.idx] + '</b> <span style="color:#888; font-size:10px;">' + fullNames[m.idx] + '</span>';
      div.dataset.idx = m.idx;
      div.addEventListener('mouseenter', function() {{ this.style.background = '#e8f0fe'; }});
      div.addEventListener('mouseleave', function() {{ this.style.background = ''; }});
      div.addEventListener('mousedown', function(ev) {{
        ev.preventDefault();
        selectResult(parseInt(this.dataset.idx));
      }});
      dropdown.appendChild(div);
    }}
  }}

  function selectResult(idx) {{
    dropdown.style.display = 'none';
    pinRow(idx);  // also sets search box text + clear btn
    // Scroll SVG
    const rowTop = yPos[idx];
    scrollEl.scrollTop = Math.max(0, rowTop - scrollEl.clientHeight / 2 + nodeH / 2);
  }}

  function clearSearch() {{
    searchInput.value = ''; dropdown.style.display = 'none';
    clearBtn.style.display = 'none'; unpinRow();
  }}

  searchInput.addEventListener('input', function() {{
    const q = this.value.trim();
    clearBtn.style.display = q ? 'inline' : 'none';
    if (!q) unpinRow();
    renderDropdown(getMatches(q));
  }});
  searchInput.addEventListener('focus', function() {{
    const q = this.value.trim();
    if (q) renderDropdown(getMatches(q));
  }});
  searchInput.addEventListener('blur', function() {{
    setTimeout(function() {{ dropdown.style.display = 'none'; }}, 150);
  }});
  searchInput.addEventListener('keydown', function(ev) {{
    const items = dropdown.querySelectorAll('div[data-idx]');
    if (items.length === 0) return;
    let activeItem = dropdown.querySelector('.kb-active');
    let ai = -1;
    items.forEach((it, i) => {{ if (it.classList.contains('kb-active')) ai = i; }});
    if (ev.key === 'ArrowDown') {{
      ev.preventDefault();
      if (activeItem) {{ activeItem.classList.remove('kb-active'); activeItem.style.background = ''; }}
      ai = (ai + 1) % items.length;
      items[ai].classList.add('kb-active'); items[ai].style.background = '#e8f0fe';
    }} else if (ev.key === 'ArrowUp') {{
      ev.preventDefault();
      if (activeItem) {{ activeItem.classList.remove('kb-active'); activeItem.style.background = ''; }}
      ai = ai <= 0 ? items.length - 1 : ai - 1;
      items[ai].classList.add('kb-active'); items[ai].style.background = '#e8f0fe';
    }} else if (ev.key === 'Enter') {{
      ev.preventDefault();
      if (ai >= 0) selectResult(parseInt(items[ai].dataset.idx));
      else if (items.length > 0) selectResult(parseInt(items[0].dataset.idx));
    }} else if (ev.key === 'Escape') {{
      dropdown.style.display = 'none'; searchInput.blur();
    }}
  }});
  clearBtn.addEventListener('click', clearSearch);

}})();
</script>
"""
    display(HTML(html))
