"use strict";
/* <tpviz-figure> — an embeddable, discrete-step visualization of one
   parallelism strategy. Scrubbing teleports between exact step states;
   animation happens only in play mode (short tweens per recorded beat,
   snapping to the step's exact end state afterwards).

   The canvas is a deliberately dark panel (like a screen) in both page
   themes; article chrome and typography follow the page theme. */

(() => {
  const DOC = JSON.parse(document.getElementById("tpviz-data").textContent);
  const NS = "http://www.w3.org/2000/svg";
  const FRAME_W = DOC.frame[0], FRAME_H = DOC.frame[1];
  const SX = (v) => v + FRAME_W / 2;
  const SY = (v) => FRAME_H / 2 - v;
  const VIEWBOX = "0 116 1422 600";

  /* the canvas's own (theme-independent) palette — mirrors style.py */
  const C = {
    fill: { activation: "#f59e0b", weight: "#3b82f6", kv: "#14b8a6", grad: "#f43f5e" },
    stroke: { activation: "#fcd34d", weight: "#93c5fd", kv: "#5eead4", grad: "#fda4af" },
    muted: "#9ca3af",
  };

  function svgEl(tag, attrs, parent) {
    const el = document.createElementNS(NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(el);
    return el;
  }

  /* ---------------------------------------------------------- equations */
  let glyphsInstalled = false;
  function installGlyphs() {
    if (glyphsInstalled) return;
    glyphsInstalled = true;
    const host = document.createElementNS(NS, "svg");
    host.setAttribute("width", 0);
    host.setAttribute("height", 0);
    host.style.position = "absolute";
    const defs = svgEl("defs", {}, host);
    for (const gid in DOC.glyphs) {
      svgEl("path", { id: `tpv-${gid}`, d: DOC.glyphs[gid], fill: "currentColor" }, defs);
    }
    document.body.prepend(host);
  }

  function eqInto(parent, eqKey) {
    const eq = DOC.eq[eqKey];
    for (const [gid, x, y] of eq.u) svgEl("use", { href: `#tpv-${gid}`, x, y }, parent);
    for (const [x, y, w, h] of eq.r || []) {
      svgEl("rect", { x, y, width: w, height: h, fill: "currentColor" }, parent);
    }
  }

  function eqSvg(eqKey, heightPx) {
    const eq = DOC.eq[eqKey];
    const s = heightPx / eq.h;
    const svg = svgEl("svg", {
      class: "tpv-eq",
      viewBox: `0 0 ${eq.w} ${eq.h}`,
      width: (eq.w * s).toFixed(1),
      height: heightPx,
      "aria-hidden": "true",
    });
    eqInto(svgEl("g", {}, svg), eqKey);
    return svg;
  }

  function eqGroup(eqKey, height) {
    const eq = DOC.eq[eqKey];
    const g = document.createElementNS(NS, "g");
    const s = height / eq.h;
    g.setAttribute("transform", `translate(${-eq.w * s / 2},${-eq.h * s / 2}) scale(${s})`);
    eqInto(g, eqKey);
    return g;
  }

  /* --------------------------------------------- tensor + primitive draw */
  const DIM_LEN = { D: 150, F: 230, H: 120, T: 100, S: 90, B: 100, E: 100 };
  const N_SLABS = 4, SLAB_DX = 9, SLAB_DY = 7;

  function axisCoord(mesh, axis, device) {
    let rem = device;
    for (const name of Object.keys(mesh).reverse()) {
      const size = mesh[name], coord = rem % size;
      if (name === axis) return coord;
      rem = Math.floor(rem / size);
    }
    return 0;
  }

  function shardFrac(t, dim) {
    const ax = t.shard[dim];
    if (!ax || !t.sharded) return [1, 0];
    const n = t.mesh[ax] || 1;
    return [1 / n, axisCoord(t.mesh, ax, t.device) / n];
  }

  function ghostRect(g, w, h, cx, cy) {
    svgEl("rect", {
      x: cx - w / 2, y: cy - h / 2, width: w, height: h,
      fill: "none", stroke: C.muted, "stroke-opacity": 0.45,
      "stroke-width": 1.4, "stroke-dasharray": "6 5",
    }, g);
  }

  function drawDeck(spec) {
    const t = spec.tensor;
    const g = document.createElementNS(NS, "g");
    const widthDim = t.dims[t.dims.length - 1];
    const fullW = DIM_LEN[widthDim] || 150, fullH = DIM_LEN.T;
    const hasB = t.dims.includes("B"), hasT = t.dims.includes("T");
    const [fb, ob] = hasB ? shardFrac(t, "B") : [1, 0];
    const [ft, ot] = hasT ? shardFrac(t, "T") : [1, 0];
    const [fw, ow] = shardFrac(t, widthDim);
    const fill = C.fill[t.kind], stroke = C.stroke[t.kind];
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
      const dx = i * SLAB_DX, dy = -i * SLAB_DY;
      if (!own.has(i)) { ghostRect(g, fullW, fullH, dx, dy); continue; }
      const fade = i !== frontOwn ? 0.55 : 1;
      const sx = dx - fullW / 2 + fullW * ow + solidW / 2;
      const sy = dy - fullH / 2 + fullH * ot + solidH / 2;
      if ((ft < 1 || fw < 1) && i === frontOwn) ghostRect(g, fullW, fullH, dx, dy);
      svgEl("rect", {
        x: sx - solidW / 2, y: sy - solidH / 2, width: solidW, height: solidH,
        fill, "fill-opacity": opacity * fade,
        stroke, "stroke-opacity": partial ? 0 : fade, "stroke-width": 1.6,
      }, g);
      if (partial) svgEl("rect", {
        x: sx - solidW / 2, y: sy - solidH / 2, width: solidW, height: solidH,
        fill: "none", stroke, "stroke-opacity": fade, "stroke-width": 1.6,
        "stroke-dasharray": "7 5",
      }, g);
    }
    return g;
  }

  function drawWeightRect(spec) {
    const t = spec.tensor;
    const g = document.createElementNS(NS, "g");
    const d0 = t.dims[t.dims.length - 2], d1 = t.dims[t.dims.length - 1];
    const fullH = (DIM_LEN[d0] || 100) * 0.75, fullW = (DIM_LEN[d1] || 100) * 0.75;
    const [f0, o0] = shardFrac(t, d0);
    const [f1, o1] = shardFrac(t, d1);
    const fill = C.fill[t.kind] || C.fill.weight;
    const stroke = C.stroke[t.kind] || C.stroke.weight;
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
    const step = 18;
    for (let cx = -solidW / 2 + step; cx < solidW / 2 + solidH; cx += step) {
      const x1 = Math.max(-solidW / 2, cx - solidH);
      const y1 = Math.min(solidH / 2, -solidH / 2 + (cx - x1));
      const x2 = Math.min(solidW / 2, cx);
      const y2 = -solidH / 2 + (cx - x2);
      if (x1 < x2) svgEl("line", {
        x1: sx + x1, y1: sy + y1, x2: sx + x2, y2: sy + y2,
        stroke, "stroke-opacity": 0.25, "stroke-width": 0.8,
      }, g);
    }
    return g;
  }

  function drawObject(spec) {
    let inner;
    switch (spec.c) {
      case "deck": inner = drawDeck(spec); break;
      case "wrect": inner = drawWeightRect(spec); break;
      case "eq": {
        inner = document.createElementNS(NS, "g");
        inner.appendChild(eqGroup(spec.eq, spec.h));
        inner.style.color = spec.color || "#e5e7eb";
        break;
      }
      case "text": {
        inner = document.createElementNS(NS, "g");
        const el = svgEl("text", {
          "text-anchor": "middle", "dominant-baseline": "central",
          "font-size": (spec.h * 1.05).toFixed(1),
          fill: spec.color || "#e5e7eb",
          "font-weight": spec.bold ? 700 : 500,
        }, inner);
        el.textContent = spec.s;
        break;
      }
      case "rect":
        inner = document.createElementNS(NS, "g");
        svgEl("rect", {
          x: -spec.w / 2, y: -spec.h / 2, width: spec.w, height: spec.h,
          rx: (spec.r || 0) * 100,
          fill: spec.fill, "fill-opacity": spec.fo,
          stroke: spec.stroke, "stroke-opacity": spec.so, "stroke-width": spec.sw,
        }, inner);
        break;
      case "dashrect":
        inner = document.createElementNS(NS, "g");
        svgEl("rect", {
          x: -spec.w / 2, y: -spec.h / 2, width: spec.w, height: spec.h,
          fill: "none", stroke: spec.stroke, "stroke-width": spec.sw,
          "stroke-dasharray": "7 5",
        }, inner);
        break;
      case "token": case "dot":
        inner = document.createElementNS(NS, "g");
        if (spec.c === "dot") svgEl("circle", { r: spec.w / 2, fill: spec.fill }, inner);
        else svgEl("rect", {
          x: -spec.w / 2, y: -spec.h / 2, width: spec.w, height: spec.h,
          fill: spec.fill, "fill-opacity": spec.fo,
          stroke: spec.stroke, "stroke-width": spec.sw,
        }, inner);
        break;
      case "line":
        inner = document.createElementNS(NS, "g");
        svgEl("line", {
          x1: -spec.dx * 100 / 2, y1: spec.dy * 100 / 2,
          x2: spec.dx * 100 / 2, y2: -spec.dy * 100 / 2,
          stroke: spec.stroke, "stroke-width": spec.sw,
          ...(spec.dashed ? { "stroke-dasharray": "5 5" } : {}),
        }, inner);
        break;
      default:
        inner = document.createElementNS(NS, "g");
        if (spec.c === "arc") svgEl("circle", {
          r: spec.w / 2, fill: "none", stroke: spec.stroke,
          "stroke-width": spec.sw, "stroke-dasharray": "4 4",
        }, inner);
    }
    const wrap = document.createElementNS(NS, "g");
    wrap.appendChild(inner);
    if (spec.c === "deck" || spec.c === "wrect") {
      wrap.dataset.fitW = spec.w;
      wrap.dataset.fitH = spec.h;
    }
    return wrap;
  }

  function applyFit(wrap) {
    if (!wrap.dataset.fitW) return;
    const inner = wrap.firstChild;
    const bb = inner.getBBox();
    if (bb.width < 1e-3) return;
    const s = Math.min(+wrap.dataset.fitW / bb.width, +wrap.dataset.fitH / bb.height);
    inner.setAttribute(
      "transform",
      `scale(${s.toFixed(5)}) translate(${(-(bb.x + bb.width / 2)).toFixed(2)},${(-(bb.y + bb.height / 2)).toFixed(2)})`
    );
  }

  /* --------------------------------------------------------- the figure */
  const PHASE_TITLE = { attn: "Attention", mlp: "MLP", moe: "MoE MLP", pipeline: "Pipeline" };
  const ease = (u) => u * u * (3 - 2 * u);

  class TPVizFigure extends HTMLElement {
    connectedCallback() {
      installGlyphs();
      const strategy = this.getAttribute("strategy") || "tp";
      this.tl = DOC.timelines[`${strategy}_train`] || DOC.timelines[this.getAttribute("timeline")];
      if (!this.tl) { this.textContent = "figure data missing"; return; }
      this.meta = (DOC.meta.strategies || {})[strategy] || {};
      this.expanded = false;
      this.playing = false;
      this._token = 0;

      this.fwdCount = this.tl.steps.filter((s) => !s.bwd).length;
      this.buildDOM();
      this.precompute();
      this.show(-1); // base state
      // QA / deep-link params: #step=12&bwd=1
      const p = new URLSearchParams(location.hash.slice(1));
      if (p.get("bwd") === "1") this.toggleBackward();
      if (p.get("step") !== null && p.get("step") !== undefined && p.get("step") !== "") {
        const k = Math.min(parseInt(p.get("step"), 10), this.limit());
        if (k >= -1) this.show(k);
      }
    }

    /* ---- precompute absolute state + aliveness per step boundary ---- */
    precompute() {
      const n = this.tl.objects.length;
      const mk = () => ({ st: new Array(n).fill(null), alive: new Uint8Array(n) });
      const cur = mk();
      for (const row of this.tl.base) {
        cur.st[row[0]] = row.slice(1);
        cur.alive[row[0]] = 1;
      }
      this.states = [ { st: cur.st.slice(), alive: cur.alive.slice() } ];
      for (const step of this.tl.steps) {
        for (const row of step.d) cur.st[row[0]] = row.slice(1);
        for (const i of step.in) cur.alive[i] = 1;
        for (const i of step.gone) cur.alive[i] = 0;
        this.states.push({ st: cur.st.slice(), alive: cur.alive.slice() });
      }
    }

    buildDOM() {
      this.classList.add("tpv");
      const head = document.createElement("div");
      head.className = "tpv-head";
      head.innerHTML = `
        <div class="tpv-controls">
          <button class="tpv-btn tpv-prev" title="previous step">‹</button>
          <button class="tpv-btn tpv-play" title="play from here">▶</button>
          <button class="tpv-btn tpv-next" title="next step">›</button>
          <span class="tpv-pos"></span>
        </div>
        <div class="tpv-chips">
          <span class="tpv-mesh"></span>
          <span class="tpv-chip" title="collectives so far">⇄ <b class="tpv-coll">0</b></span>
          <span class="tpv-chip" title="matmuls so far">× <b class="tpv-mm">0</b></span>
        </div>`;
      this.appendChild(head);
      if (this.meta.mesh) head.querySelector(".tpv-mesh").appendChild(eqSvg(this.meta.mesh, 14));

      const wrap = document.createElement("div");
      wrap.className = "tpv-canvas-wrap";
      this.svg = svgEl("svg", { viewBox: VIEWBOX, class: "tpv-canvas", preserveAspectRatio: "xMidYMid meet" }, wrap);
      this.appendChild(wrap);

      this.scene = svgEl("g", {}, this.svg);
      this.els = this.tl.objects
        .map((spec, i) => ({ spec, i }))
        .sort((a, b) => (a.spec.z - b.spec.z) || (a.i - b.i))
        .reduce((acc, { spec, i }) => {
          const el = drawObject(spec);
          el.style.display = "none";
          if (spec.tip !== undefined) {
            el.classList.add("tpv-hit");
            el.dataset.tip = spec.tip;
            el.dataset.tex = spec.tex;
          }
          this.scene.appendChild(el);
          acc[i] = el;
          return acc;
        }, new Array(this.tl.objects.length));
      this.els.forEach(applyFit);

      // stepper
      this.stepper = document.createElement("div");
      this.stepper.className = "tpv-stepper";
      this.segs = this.tl.steps.map((st, i) => {
        const seg = document.createElement("button");
        seg.className = "tpv-seg" + (st.bwd ? " bwd" : "");
        seg.title = `${st.bwd ? "◀ " : ""}L${st.layer} · ${PHASE_TITLE[st.phase] || st.phase} · ${st.kind}`;
        seg.addEventListener("click", () => { this.stop(); this.show(i); });
        this.stepper.appendChild(seg);
        return seg;
      });
      this.appendChild(this.stepper);

      // current-step readout
      this.readout = document.createElement("div");
      this.readout.className = "tpv-readout";
      this.appendChild(this.readout);

      // backward reveal
      this.bwdToggle = document.createElement("button");
      this.bwdToggle.className = "tpv-bwd-toggle";
      const nb = this.tl.steps.length - this.fwdCount;
      const s = this.tl.summary;
      this.bwdToggle.innerHTML =
        `<span class="chev">▸</span> <span class="tpv-bwd-label">Continue into the backward pass</span> ` +
        `<span class="tpv-bwd-meta">${nb} steps · ${s.mm.backward} matmuls (2× forward)</span>`;
      this.bwdToggle.addEventListener("click", () => this.toggleBackward());
      this.appendChild(this.bwdToggle);

      // program list
      const details = document.createElement("details");
      details.className = "tpv-program";
      details.innerHTML = "<summary>program</summary>";
      this.programList = document.createElement("div");
      this.programList.className = "tpv-program-list";
      details.appendChild(this.programList);
      this.appendChild(details);
      this.buildProgram();

      head.querySelector(".tpv-play").addEventListener("click", () => this.playing ? this.stop() : this.play());
      head.querySelector(".tpv-prev").addEventListener("click", () => { this.stop(); this.show(Math.max(-1, this.cur - 1)); });
      head.querySelector(".tpv-next").addEventListener("click", () => { this.stop(); this.advance(); });
      this.posEl = head.querySelector(".tpv-pos");
      this.collEl = head.querySelector(".tpv-coll");
      this.mmEl = head.querySelector(".tpv-mm");
      this.playBtn = head.querySelector(".tpv-play");

      this.tabIndex = 0;
      this.addEventListener("keydown", (ev) => {
        if (ev.key === "ArrowRight") { ev.preventDefault(); this.stop(); this.advance(); }
        else if (ev.key === "ArrowLeft") { ev.preventDefault(); this.stop(); this.show(Math.max(-1, this.cur - 1)); }
        else if (ev.code === "Space") { ev.preventDefault(); this.playing ? this.stop() : this.play(); }
      });
      this.initTooltip(wrap);
      this.applyExpansion();
    }

    buildProgram() {
      let sectionKey = "", entries = null;
      this.progEntries = [];
      this.tl.steps.forEach((st, i) => {
        const key = `${st.bwd ? "b" : "f"}|${st.layer}|${st.phase}`;
        if (key !== sectionKey) {
          sectionKey = key;
          const head = document.createElement("div");
          head.className = `tpv-prog-sec ${st.bwd ? "bwd" : "fwd"}`;
          head.textContent = st.phase === "pipeline"
            ? `${st.bwd ? "◀ backward · " : ""}pipeline`
            : `${st.bwd ? "◀ backward · " : ""}layer ${st.layer} · ${(PHASE_TITLE[st.phase] || st.phase).toLowerCase()}`;
          head.dataset.bwd = st.bwd ? "1" : "0";
          this.programList.appendChild(head);
        }
        const e = document.createElement("button");
        e.className = "tpv-prog-entry" + (st.bwd ? " bwd" : "");
        e.dataset.bwd = st.bwd ? "1" : "0";
        if (st.kind === "SaveActivation") {
          const chip = document.createElement("span");
          chip.className = "tpv-save";
          chip.textContent = "save";
          e.appendChild(chip);
        }
        if (st.eq) e.appendChild(eqSvg(st.eq, 12));
        else e.appendChild(document.createTextNode(st.kind));
        e.addEventListener("click", () => { this.stop(); this.show(i); });
        this.programList.appendChild(e);
        this.progEntries.push(e);
      });
    }

    limit() { return this.expanded ? this.tl.steps.length - 1 : this.fwdCount - 1; }

    toggleBackward() {
      this.expanded = !this.expanded;
      this.applyExpansion();
      if (!this.expanded && this.cur > this.limit()) { this.stop(); this.show(this.limit()); }
      if (this.expanded) { this.stop(); this.show(this.fwdCount - 1); }
    }

    applyExpansion() {
      this.classList.toggle("tpv-expanded", this.expanded);
      this.bwdToggle.querySelector(".chev").textContent = this.expanded ? "▾" : "▸";
      this.bwdToggle.querySelector(".tpv-bwd-label").textContent =
        this.expanded ? "Backward pass shown — collapse" : "Continue into the backward pass";
      this.segs.forEach((seg, i) => {
        seg.style.display = this.tl.steps[i].bwd && !this.expanded ? "none" : "";
      });
      this.programList.querySelectorAll('[data-bwd="1"]').forEach((el) => {
        el.style.display = this.expanded ? "" : "none";
      });
    }

    /* ------------------------------------------------------ state apply */
    applyState(i, obj, st, visible) {
      const el = this.els[obj];
      if (!visible) { el.style.display = "none"; return; }
      el.style.display = "";
      const [x, y, w, o] = st;
      el.setAttribute("transform", `translate(${SX(x).toFixed(1)},${SY(y).toFixed(1)})${w !== 100 ? ` scale(${(w / 100).toFixed(4)})` : ""}`);
      el.setAttribute("opacity", (o / 100).toFixed(3));
    }

    show(i) {
      this.cur = i;
      const s = this.states[i + 1];
      for (let k = 0; k < this.els.length; k++) {
        this.applyState(i, k, s.st[k] || [0, 0, 100, 0], !!s.alive[k] && !!s.st[k]);
      }
      this.updateChrome();
    }

    updateChrome() {
      const i = this.cur;
      const lim = this.limit();
      this.segs.forEach((seg, k) => {
        seg.classList.toggle("done", k < i);
        seg.classList.toggle("current", k === i);
      });
      this.progEntries.forEach((e, k) => {
        e.classList.toggle("current", k === i);
        e.classList.toggle("done", k < i);
      });
      const cur = this.progEntries[i];
      if (cur && this.querySelector(".tpv-program").open) {
        cur.scrollIntoView({ block: "nearest" });
      }
      const st = i >= 0 ? this.tl.steps[i] : null;
      this.posEl.textContent = i < 0 ? `0 / ${lim + 1}` : `${i + 1} / ${lim + 1}`;
      this.collEl.textContent = st ? st.coll : 0;
      this.mmEl.textContent = st ? st.mm : 0;
      this.playBtn.textContent = this.playing ? "❚❚" : "▶";
      this.renderReadout(st);
    }

    renderReadout(st) {
      this.readout.textContent = "";
      if (!st) {
        this.readout.innerHTML = `<span class="tpv-cap">Initial placement — press play, step with ‹ ›, or click any segment above.</span>`;
        return;
      }
      const line = document.createElement("div");
      line.className = "tpv-ro-eq" + (st.bwd ? " bwd" : "");
      if (st.kind === "SaveActivation") {
        const chip = document.createElement("span");
        chip.className = "tpv-save";
        chip.textContent = "save";
        line.appendChild(chip);
      }
      if (st.eq) line.appendChild(eqSvg(st.eq, 17));
      else line.appendChild(document.createTextNode(st.kind));
      this.readout.appendChild(line);
      if (st.note) {
        const note = document.createElement("div");
        note.className = "tpv-ro-note";
        note.appendChild(eqSvg(st.note, 12));
        this.readout.appendChild(note);
      }
      if (st.cap) {
        const cap = document.createElement("div");
        cap.className = "tpv-cap";
        cap.textContent = st.cap;
        this.readout.appendChild(cap);
      }
    }

    /* ------------------------------------------------------- play mode */
    advance() {
      if (this.cur < this.limit()) this.show(this.cur + 1);
    }

    async play() {
      if (this.playing) return;
      this.playing = true;
      const token = ++this._token;
      this.updateChrome();
      while (this.playing && this._token === token && this.cur < this.limit()) {
        const next = this.cur + 1;
        await this.playStep(next, token);
        if (this._token !== token) return;
        this.show(next); // exact snap
        await new Promise((r) => setTimeout(r, 260));
      }
      if (this._token === token) { this.playing = false; this.updateChrome(); }
    }

    stop() {
      this.playing = false;
      this._token++;
      if (this.playBtn) this.updateChrome();
    }

    async playStep(i, token) {
      const step = this.tl.steps[i];
      const from = this.states[i]; // state at previous boundary
      const live = new Map();
      for (const beat of step.beats) {
        if (this._token !== token) return;
        const moves = [];
        for (const row of beat.ch) {
          const [obj, x, y, w, o] = row;
          const el = this.els[obj];
          const prev = live.get(obj) || from.st[obj];
          const spawning = beat.in.includes(obj) || !from.alive[obj] && !live.has(obj);
          const start = spawning || !prev ? [x, y, w, 0] : prev;
          el.style.display = "";
          moves.push({ obj, el, start, end: [x, y, w, o] });
          live.set(obj, [x, y, w, o]);
        }
        const outs = beat.out.map((obj) => this.els[obj]);
        await this.tween(moves, outs, beat.d, token);
        for (const el of outs) el.style.display = "none";
      }
    }

    tween(moves, outs, dur, token) {
      return new Promise((resolve) => {
        const t0 = performance.now();
        const frame = (now) => {
          if (this._token !== token) return resolve();
          const u = Math.min(1, (now - t0) / dur);
          const p = ease(u);
          for (const m of moves) {
            const v = m.start.map((a, k) => a + (m.end[k] - a) * p);
            this.applyState(0, m.obj, v, true);
          }
          for (const el of outs) el.setAttribute("opacity", (1 - p).toFixed(3));
          if (u < 1) requestAnimationFrame(frame);
          else resolve();
        };
        requestAnimationFrame(frame);
      });
    }

    /* -------------------------------------------------------- tooltips */
    initTooltip(wrap) {
      const tip = document.createElement("div");
      tip.className = "tpv-tooltip";
      tip.hidden = true;
      wrap.appendChild(tip);
      const KIND = { activation: "act", weight: "wt", kv: "kv", grad: "grad" };

      const show = (target, ev) => {
        const data = DOC.tooltips[+target.dataset.tip];
        if (!data) return;
        tip.textContent = "";
        const head = document.createElement("div");
        head.className = "tpv-tt-head";
        if (target.dataset.tex) head.appendChild(eqSvg(target.dataset.tex, 14));
        const chip = document.createElement("span");
        chip.className = `tpv-kind tpv-kind-${KIND[data.kind] || "act"}`;
        chip.textContent = data.kind;
        head.appendChild(chip);
        tip.appendChild(head);
        const rows = [
          ["global", data.global.join(" × "), data.memGlobal],
          ["this device", data.local.join(" × "), data.memLocal],
        ];
        const tbl = document.createElement("table");
        for (const [k, shape, mem] of rows) {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>${k}</td><td>${shape}</td><td>${mem}</td>`;
          tbl.appendChild(tr);
        }
        tip.appendChild(tbl);
        const sd = document.createElement("div");
        sd.className = "tpv-tt-shard";
        sd.textContent = data.shardDesc;
        tip.appendChild(sd);
        if (data.partial) {
          const pw = document.createElement("div");
          pw.className = "tpv-tt-partial";
          pw.textContent = `unreduced partial sum over ${data.partial.join(", ")}`;
          tip.appendChild(pw);
        }
        tip.hidden = false;
        move(ev);
      };
      const move = (ev) => {
        const r = wrap.getBoundingClientRect();
        let x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 14;
        if (x + tip.offsetWidth > r.width - 6) x = ev.clientX - r.left - tip.offsetWidth - 14;
        if (y + tip.offsetHeight > r.height - 6) y = ev.clientY - r.top - tip.offsetHeight - 14;
        tip.style.left = `${x}px`;
        tip.style.top = `${y}px`;
      };
      this.svg.addEventListener("pointerover", (ev) => {
        const t = ev.target.closest(".tpv-hit");
        if (t) show(t, ev);
      });
      this.svg.addEventListener("pointermove", (ev) => { if (!tip.hidden) move(ev); });
      this.svg.addEventListener("pointerout", (ev) => {
        if (!ev.relatedTarget?.closest?.(".tpv-hit")) tip.hidden = true;
      });
    }
  }

  customElements.define("tpviz-figure", TPVizFigure);

  /* ------------------------------------------------------------- theme */
  const root = document.documentElement;
  const hashTheme = new URLSearchParams(location.hash.slice(1)).get("theme");
  let saved = null;
  try { saved = localStorage.getItem("tpviz-theme"); } catch (e) {}
  if (hashTheme) root.dataset.theme = hashTheme;
  else if (saved) root.dataset.theme = saved;
  document.addEventListener("click", (ev) => {
    if (!ev.target.closest("#theme-toggle")) return;
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tpviz-theme", root.dataset.theme); } catch (e) {}
  });
})();
