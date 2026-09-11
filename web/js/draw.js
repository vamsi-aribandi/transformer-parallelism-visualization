"use strict";
/* SVG builders. Decks and weight rects are redrawn from tensor metadata with
   the same geometry as src/tpviz/mobjects/tensor_mobject.py (centi-units). */

const NS = "http://www.w3.org/2000/svg";
const DIM_LEN = { D: 150, F: 230, H: 120, T: 100, S: 90, B: 100, E: 100 };
const N_SLABS = 4, SLAB_DX = 9, SLAB_DY = 7;

function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(NS, tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(el);
  return el;
}

function installGlyphs() {
  const defs = document.getElementById("glyph-defs");
  for (const gid in DOC.glyphs) {
    svgEl("path", { id: gid, d: DOC.glyphs[gid], fill: "currentColor" }, defs);
  }
}

/* Equation group in pt-space scaled to `heightPx`, centered at origin unless
   anchor === "nw" (top-left at origin). Returns the <g>. */
function eqGroup(eqKey, heightPx, anchor = "c") {
  const eq = DOC.eq[eqKey];
  const g = document.createElementNS(NS, "g");
  const s = heightPx / eq.h;
  const ox = anchor === "c" ? -eq.w * s / 2 : 0;
  const oy = anchor === "c" ? -eq.h * s / 2 : 0;
  g.setAttribute("transform", `translate(${ox},${oy}) scale(${s})`);
  for (const [gid, x, y] of eq.u) {
    svgEl("use", { href: `#${gid}`, x, y }, g);
  }
  for (const [x, y, w, h] of eq.r || []) {
    svgEl("rect", { x, y, width: w, height: h, fill: "currentColor" }, g);
  }
  return g;
}

/* Standalone inline <svg> for an equation (program panel, tooltips, chips). */
function eqSvg(eqKey, heightPx) {
  const eq = DOC.eq[eqKey];
  const s = heightPx / eq.h;
  const svg = svgEl("svg", {
    viewBox: `0 0 ${eq.w} ${eq.h}`,
    width: (eq.w * s).toFixed(1),
    height: heightPx,
  });
  const g = svgEl("g", {}, svg);
  for (const [gid, x, y] of eq.u) svgEl("use", { href: `#${gid}`, x, y, fill: "currentColor" }, g);
  for (const [x, y, w, h] of eq.r || []) svgEl("rect", { x, y, width: w, height: h, fill: "currentColor" }, g);
  return svg;
}

function ghostRect(g, w, h, cx, cy) {
  svgEl("rect", {
    x: cx - w / 2, y: cy - h / 2, width: w, height: h,
    fill: "none", stroke: PAL.muted, "stroke-opacity": 0.45,
    "stroke-width": 1.4, "stroke-dasharray": "6 5",
  }, g);
}

/* ------------------------------------------------------------------ deck */
function drawDeck(spec) {
  const t = spec.tensor;
  const g = document.createElementNS(NS, "g");
  const widthDim = t.dims[t.dims.length - 1];
  const fullW = DIM_LEN[widthDim] || 150, fullH = DIM_LEN.T;
  const hasB = t.dims.includes("B"), hasT = t.dims.includes("T");
  const [fb, ob] = hasB ? shardFrac(t, "B") : [1, 0];
  const [ft, ot] = hasT ? shardFrac(t, "T") : [1, 0];
  const [fw, ow] = shardFrac(t, widthDim);
  const fill = PAL.fill[t.kind], stroke = PAL.stroke[t.kind];
  const partial = t.partial && t.partial.length;
  const opacity = partial ? 0.3 : 0.85;

  let own, total;
  if (hasB) {
    const nOwn = Math.max(1, Math.round(N_SLABS * fb));
    const start = Math.min(Math.round(N_SLABS * ob), N_SLABS - nOwn);
    own = new Set(Array.from({ length: nOwn }, (_, i) => start + i));
    total = fb < 1 ? N_SLABS : Math.max(...own) + 1;
  } else { own = new Set([0]); total = 1; }
  const frontOwn = Math.min(...own);

  const solidW = fullW * fw, solidH = fullH * ft;
  for (let i = total - 1; i >= 0; i--) {
    const dx = i * SLAB_DX, dy = -i * SLAB_DY; // y-down svg: manim up-shift = -y
    if (!own.has(i)) { ghostRect(g, fullW, fullH, dx, dy); continue; }
    const fade = i !== frontOwn ? 0.55 : 1;
    // shard offset inside the full extent (top-left anchored)
    const sx = dx - fullW / 2 + fullW * ow + solidW / 2;
    const sy = dy - fullH / 2 + fullH * ot + solidH / 2;
    if ((ft < 1 || fw < 1) && i === frontOwn) ghostRect(g, fullW, fullH, dx, dy);
    const attrs = {
      x: sx - solidW / 2, y: sy - solidH / 2, width: solidW, height: solidH,
      fill, "fill-opacity": opacity * fade,
      stroke, "stroke-opacity": partial ? 0 : fade, "stroke-width": 1.6,
    };
    svgEl("rect", attrs, g);
    if (partial) {
      svgEl("rect", {
        x: sx - solidW / 2, y: sy - solidH / 2, width: solidW, height: solidH,
        fill: "none", stroke, "stroke-opacity": fade, "stroke-width": 1.6,
        "stroke-dasharray": "7 5",
      }, g);
    }
  }
  return fitToSpec(g, spec);
}

/* ----------------------------------------------------------- weight rect */
function drawWeightRect(spec) {
  const t = spec.tensor;
  const g = document.createElementNS(NS, "g");
  const d0 = t.dims[t.dims.length - 2], d1 = t.dims[t.dims.length - 1];
  const fullH = (DIM_LEN[d0] || 100) * 0.75, fullW = (DIM_LEN[d1] || 100) * 0.75;
  const [f0, o0] = shardFrac(t, d0);
  const [f1, o1] = shardFrac(t, d1);
  const fill = PAL.fill[t.kind] || PAL.fill.weight;
  const stroke = PAL.stroke[t.kind] || PAL.stroke.weight;
  const partial = t.partial && t.partial.length;
  const solidW = fullW * f1, solidH = fullH * f0;
  let sx = 0, sy = 0;
  if (f0 < 1 || f1 < 1) {
    ghostRect(g, fullW, fullH, 0, 0);
    sx = -fullW / 2 + fullW * o1 + solidW / 2;
    sy = -fullH / 2 + fullH * o0 + solidH / 2;
  }
  svgEl("rect", {
    x: sx - solidW / 2, y: sy - solidH / 2, width: solidW, height: solidH,
    fill, "fill-opacity": partial ? 0.3 : 0.85, stroke, "stroke-width": 1.8,
    ...(partial ? { "stroke-dasharray": "6 4" } : {}),
  }, g);
  // hatch texture
  const step = 18;
  for (let cx = -solidW / 2 + step; cx < solidW / 2 + solidH; cx += step) {
    const x1 = Math.max(-solidW / 2, cx - solidH), y1 = Math.min(solidH / 2, -solidH / 2 + (cx - x1));
    const x2 = Math.min(solidW / 2, cx), y2 = -solidH / 2 + (cx - x2);
    if (x1 < x2) svgEl("line", {
      x1: sx + x1, y1: sy + y1, x2: sx + x2, y2: sy + y2,
      stroke, "stroke-opacity": 0.25, "stroke-width": 0.8,
    }, g);
  }
  return fitToSpec(g, spec);
}

/* Scale a natural-size group so its bbox matches the recorded spawn size. */
function fitToSpec(g, spec) {
  const wrap = document.createElementNS(NS, "g");
  wrap.appendChild(g);
  // natural bbox is computed after insertion; do it lazily on first layout
  wrap.dataset.fitW = spec.w;
  wrap.dataset.fitH = spec.h;
  return wrap;
}

function applyFit(wrap) {
  const inner = wrap.firstChild;
  const bb = inner.getBBox();
  if (bb.width < 1e-3) return;
  const s = Math.min(+wrap.dataset.fitW / bb.width, +wrap.dataset.fitH / bb.height);
  const cx = bb.x + bb.width / 2, cy = bb.y + bb.height / 2;
  inner.setAttribute("transform", `scale(${s.toFixed(5)}) translate(${(-cx).toFixed(2)},${(-cy).toFixed(2)})`);
}

/* ------------------------------------------------------------ primitives */
function drawObject(spec) {
  switch (spec.c) {
    case "deck": return drawDeck(spec);
    case "wrect": return drawWeightRect(spec);
    case "eq": {
      const g = document.createElementNS(NS, "g");
      g.appendChild(eqGroup(spec.eq, spec.h));
      g.style.color = spec.color || PAL.text;
      return g;
    }
    case "text": {
      const g = document.createElementNS(NS, "g");
      const el = svgEl("text", {
        "text-anchor": "middle", "dominant-baseline": "central",
        "font-size": (spec.h * 1.05).toFixed(1),
        fill: spec.color || PAL.text,
        "font-weight": spec.bold ? 700 : 500,
      }, g);
      el.textContent = spec.s;
      return g;
    }
    case "rect": {
      const g = document.createElementNS(NS, "g");
      svgEl("rect", {
        x: -spec.w / 2, y: -spec.h / 2, width: spec.w, height: spec.h,
        rx: (spec.r || 0) * 100,
        fill: spec.fill, "fill-opacity": spec.fo,
        stroke: spec.stroke, "stroke-opacity": spec.so, "stroke-width": spec.sw,
      }, g);
      return g;
    }
    case "dashrect": {
      const g = document.createElementNS(NS, "g");
      svgEl("rect", {
        x: -spec.w / 2, y: -spec.h / 2, width: spec.w, height: spec.h,
        fill: "none", stroke: spec.stroke, "stroke-width": spec.sw,
        "stroke-dasharray": "7 5",
      }, g);
      return g;
    }
    case "token": case "dot": {
      const g = document.createElementNS(NS, "g");
      if (spec.c === "dot") svgEl("circle", { r: spec.w / 2, fill: spec.fill }, g);
      else svgEl("rect", {
        x: -spec.w / 2, y: -spec.h / 2, width: spec.w, height: spec.h,
        fill: spec.fill, "fill-opacity": spec.fo,
        stroke: spec.stroke, "stroke-width": spec.sw,
      }, g);
      return g;
    }
    case "line": {
      const g = document.createElementNS(NS, "g");
      svgEl("line", {
        x1: -spec.dx * 100 / 2, y1: spec.dy * 100 / 2,
        x2: spec.dx * 100 / 2, y2: -spec.dy * 100 / 2,
        stroke: spec.stroke, "stroke-width": spec.sw,
        ...(spec.dashed ? { "stroke-dasharray": "5 5" } : {}),
      }, g);
      return g;
    }
    case "arc": {
      const g = document.createElementNS(NS, "g");
      svgEl("circle", {
        r: spec.w / 2, fill: "none", stroke: spec.stroke, "stroke-width": spec.sw,
        "stroke-dasharray": "4 4",
      }, g);
      return g;
    }
    default: return document.createElementNS(NS, "g");
  }
}
