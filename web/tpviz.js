"use strict";
/* <tpviz-figure> — an embeddable, discrete-step visualization of one
   parallelism strategy. The program listing sits beside the canvas and is the
   narration: it auto-scrolls, highlights the current operation, and for
   backward steps cites (and links) the forward operation it differentiates.

   Scrubbing teleports between exact step states; animation happens only in
   play mode. The canvas follows the page theme via --cv-* variables. */

(() => {
  const DOC = JSON.parse(document.getElementById("tpviz-data").textContent);
  const NS = "http://www.w3.org/2000/svg";
  const FRAME_W = DOC.frame[0], FRAME_H = DOC.frame[1];
  const SX = (v) => v + FRAME_W / 2;
  const SY = (v) => FRAME_H / 2 - v;
  const VIEWBOX = "0 116 1422 600";

  const TOKENS = new Set([
    "text", "muted", "accent", "comm", "good", "act", "actS", "wt", "wtS",
    "kv", "kvS", "grad", "gradS", "boxFill", "boxStroke", "surf", "surf2",
    "dev0", "dev1", "dev2", "dev3", "exp0", "exp1", "exp2", "exp3",
  ]);
  const col = (v) => (TOKENS.has(v) ? `var(--cv-${v})` : v);

  function svgEl(tag, attrs, parent) {
    const el = document.createElementNS(NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(el);
    return el;
  }

  /* ------------------------------------------------ canvas equations (svg) */
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

  function eqGroup(eqKey, height) {
    const eq = DOC.eq[eqKey];
    const g = document.createElementNS(NS, "g");
    const s = height / eq.h;
    g.setAttribute("transform", `translate(${-eq.w * s / 2},${-eq.h * s / 2}) scale(${s})`);
    for (const [gid, x, y] of eq.u) svgEl("use", { href: `#tpv-${gid}`, x, y }, g);
    for (const [x, y, w, h] of eq.r || []) {
      svgEl("rect", { x, y, width: w, height: h, fill: "currentColor" }, g);
    }
    return g;
  }

  function eqSvg(eqKey, heightPx) {
    const eq = DOC.eq[eqKey];
    const s = heightPx / eq.h;
    const svg = svgEl("svg", {
      viewBox: `0 0 ${eq.w} ${eq.h}`,
      width: (eq.w * s).toFixed(1),
      height: heightPx,
      "aria-hidden": "true",
    });
    const g = svgEl("g", {}, svg);
    for (const [gid, x, y] of eq.u) svgEl("use", { href: `#tpv-${gid}`, x, y, fill: "currentColor" }, g);
    for (const [x, y, w, h] of eq.r || []) svgEl("rect", { x, y, width: w, height: h, fill: "currentColor" }, g);
    return svg;
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
      fill: "none", stroke: "var(--cv-muted)", "stroke-opacity": 0.5,
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
    const kk = t.kind === "kv" ? "kv" : t.kind === "grad" ? "grad" : "act";
    const fill = `var(--cv-${kk})`;
    const stroke = `var(--cv-${kk}S)`;
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
    const grad = t.kind === "grad";
    const fill = `var(--cv-${grad ? "grad" : "wt"})`;
    const stroke = `var(--cv-${grad ? "gradS" : "wtS"})`;
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
        inner.style.color = col(spec.color) || "var(--cv-text)";
        break;
      }
      case "text": {
        inner = document.createElementNS(NS, "g");
        const el = svgEl("text", {
          "text-anchor": "middle", "dominant-baseline": "central",
          "font-size": (spec.h * 1.05).toFixed(1),
          fill: col(spec.color) || "var(--cv-text)",
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
          fill: col(spec.fill), "fill-opacity": spec.fo,
          stroke: col(spec.stroke), "stroke-opacity": spec.so, "stroke-width": spec.sw,
        }, inner);
        break;
      case "dashrect":
        inner = document.createElementNS(NS, "g");
        svgEl("rect", {
          x: -spec.w / 2, y: -spec.h / 2, width: spec.w, height: spec.h,
          fill: "none", stroke: col(spec.stroke), "stroke-width": spec.sw,
          "stroke-dasharray": "7 5",
        }, inner);
        break;
      case "token": case "dot":
        inner = document.createElementNS(NS, "g");
        if (spec.c === "dot") svgEl("circle", { r: spec.w / 2, fill: col(spec.fill) }, inner);
        else svgEl("rect", {
          x: -spec.w / 2, y: -spec.h / 2, width: spec.w, height: spec.h,
          fill: col(spec.fill), "fill-opacity": spec.fo,
          stroke: col(spec.stroke), "stroke-width": spec.sw,
        }, inner);
        break;
      case "line":
        inner = document.createElementNS(NS, "g");
        svgEl("line", {
          x1: -spec.dx * 100 / 2, y1: spec.dy * 100 / 2,
          x2: spec.dx * 100 / 2, y2: -spec.dy * 100 / 2,
          stroke: col(spec.stroke), "stroke-width": spec.sw,
          ...(spec.dashed ? { "stroke-dasharray": "5 5" } : {}),
        }, inner);
        break;
      default:
        inner = document.createElementNS(NS, "g");
        if (spec.c === "arc") svgEl("circle", {
          r: spec.w / 2, fill: "none", stroke: col(spec.stroke),
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
      this.tl = DOC.timelines[`${strategy}_train`];
      if (!this.tl) { this.textContent = "figure data missing"; return; }
      this.meta = (DOC.meta.strategies || {})[strategy] || {};
      this.mode = "fwd";
      this.playing = false;
      this.speed = 1; // recorded pacing is "2x"; default plays at half that
      this._token = 0;
      this.saveObjs = new Set(this.tl.saveObjs || []);

      this.buildDOM();
      this.precompute();
      this.setMode("fwd", true);

      const p = new URLSearchParams(location.hash.slice(1));
      if (p.get("mode") === "train") this.setMode("train");
      if (p.get("step")) {
        const k = Math.min(parseInt(p.get("step"), 10), this.visible.length - 1);
        if (k >= -1) this.show(k);
      }
    }

    precompute() {
      const n = this.tl.objects.length;
      const cur = { st: new Array(n).fill(null), alive: new Uint8Array(n) };
      for (const row of this.tl.base) {
        cur.st[row[0]] = row.slice(1);
        cur.alive[row[0]] = 1;
      }
      this.states = [{ st: cur.st.slice(), alive: cur.alive.slice() }];
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
        <div class="tpv-modes" role="tablist">
          <button data-mode="fwd" class="active" role="tab">Forward</button>
          <button data-mode="train" role="tab">+ Backward</button>
        </div>
        <label class="tpv-speed" title="play speed">
          <input type="range" min="0.1" max="2" step="0.1" value="1">
          <span class="tpv-speed-val">1.0×</span>
        </label>
        <span class="tpv-mesh meq"></span>`;
      this.appendChild(head);
      if (this.meta.meshh) head.querySelector(".tpv-mesh").innerHTML = this.meta.meshh;
      head.querySelectorAll(".tpv-modes button").forEach((b) =>
        b.addEventListener("click", () => this.setMode(b.dataset.mode))
      );

      const wrap = document.createElement("div");
      wrap.className = "tpv-canvas-wrap";
      this.svg = svgEl("svg", { viewBox: VIEWBOX, class: "tpv-canvas", preserveAspectRatio: "xMidYMid meet" }, wrap);
      this.appendChild(wrap);

      this.scene = svgEl("g", {}, this.svg);
      this.els = new Array(this.tl.objects.length);
      this.tl.objects
        .map((spec, i) => ({ spec, i }))
        .sort((a, b) => (a.spec.z - b.spec.z) || (a.i - b.i))
        .forEach(({ spec, i }) => {
          const el = drawObject(spec);
          el.style.display = "none";
          if (spec.tip !== undefined) {
            el.classList.add("tpv-hit");
            el.dataset.tip = spec.tip;
            el.dataset.tex = spec.tex;
          }
          this.scene.appendChild(el);
          this.els[i] = el;
        });
      this.els.forEach(applyFit);

      this.band = document.createElement("div");
      this.band.className = "tpv-band";
      const legendWrap = document.createElement("div");
      legendWrap.className = "tpv-band-legend";
      legendWrap.appendChild(this.dimLegend());
      this.band.appendChild(legendWrap);
      this.algo = document.createElement("div");
      this.algo.className = "tpv-algo-inner";
      this.algo.innerHTML = `<div class="tpv-algo-head">
          <span class="tpv-algo-title"></span>
          <span class="tpv-algo-hop"></span>
        </div>`;
      this.algoSvg = svgEl("svg", { viewBox: "0 0 1000 190", class: "tpv-algo-svg" }, this.algo);
      this.band.appendChild(this.algo);
      this.appendChild(this.band);
      this._algoToken = 0;

      this.stepper = document.createElement("div");
      this.stepper.className = "tpv-stepper";
      this.appendChild(this.stepper);

      const prog = document.createElement("div");
      prog.className = "tpv-program";
      prog.innerHTML = `<div class="tpv-program-head">program</div>`;
      this.programList = document.createElement("div");
      this.programList.className = "tpv-program-list";
      prog.appendChild(this.programList);
      this.appendChild(prog);

      head.querySelector(".tpv-play").addEventListener("click", () => this.playing ? this.stop() : this.play());
      head.querySelector(".tpv-prev").addEventListener("click", () => { this.stop(); this.show(this.cur - 1); });
      head.querySelector(".tpv-next").addEventListener("click", () => { this.stop(); this.show(this.cur + 1); });
      this.posEl = head.querySelector(".tpv-pos");
      this.playBtn = head.querySelector(".tpv-play");
      this.modeBtns = head.querySelectorAll(".tpv-modes button");
      const speedInput = head.querySelector(".tpv-speed input");
      const speedVal = head.querySelector(".tpv-speed-val");
      speedInput.addEventListener("input", () => {
        this.speed = parseFloat(speedInput.value);
        speedVal.textContent = `${this.speed.toFixed(1)}×`;
      });

      this.tabIndex = 0;
      this.addEventListener("keydown", (ev) => {
        if (ev.key === "ArrowRight") { ev.preventDefault(); this.stop(); this.show(this.cur + 1); }
        else if (ev.key === "ArrowLeft") { ev.preventDefault(); this.stop(); this.show(this.cur - 1); }
        else if (ev.code === "Space") { ev.preventDefault(); this.playing ? this.stop() : this.play(); }
      });
      this.initTooltip(wrap);
    }

    /* A small annotated tensor: what the amber decks are made of. */
    dimLegend() {
      const svg = svgEl("svg", {
        viewBox: "-130 -32 460 122",
        class: "tpv-dim-legend",
        preserveAspectRatio: "xMidYMid meet",
      });
      const g = svgEl("g", {}, svg);
      const W = 58, H = 36, DX = 8, DY = 7, N = 3;
      for (let i = N - 1; i >= 0; i--) {
        svgEl("rect", {
          x: i * DX, y: -i * DY, width: W, height: H,
          fill: "var(--cv-act)", "fill-opacity": i === 0 ? 0.85 : 0.4,
          stroke: "var(--cv-actS)", "stroke-opacity": i === 0 ? 1 : 0.5,
          "stroke-width": 1.4,
        }, g);
      }
      const arrow = (x1, y1, x2, y2) => {
        svgEl("line", { x1, y1, x2, y2, stroke: "var(--cv-muted)", "stroke-width": 1.4 }, g);
        const ang = Math.atan2(y2 - y1, x2 - x1);
        const tip = (a) => `${x2 - 7 * Math.cos(ang - a)},${y2 - 7 * Math.sin(ang - a)}`;
        svgEl("path", { d: `M${x2},${y2} L${tip(0.42)} L${tip(-0.42)} Z`, fill: "var(--cv-muted)" }, g);
      };
      const label = (x, y, sym, rest, anchor = "start") => {
        const t = svgEl("text", {
          x, y, "text-anchor": anchor, "dominant-baseline": "central",
          "font-size": 14, fill: "var(--cv-muted)",
        }, g);
        const b = svgEl("tspan", { "font-style": "italic", fill: "var(--cv-text)" }, t);
        b.textContent = sym;
        const r = svgEl("tspan", {}, t);
        r.textContent = ` ${rest}`;
      };
      arrow(0, H + 12, W, H + 12);                       // D: width
      label(W + 12, H + 12, "D", "model dimension");
      arrow(-10, H, -10, 0);                             // T: height
      label(-18, H / 2, "T", "sequence", "end");
      arrow(W + 4, H - 4, W + 4 + (N - 1) * DX + 8, H - 4 - (N - 1) * DY - 8); // B: depth
      label(W + (N - 1) * DX + 20, H - (N - 1) * DY - 16, "B", "batch");
      return svg;
    }

    /* ------------------------------------------------------ mode + lists */
    setMode(mode, force) {
      if (mode === this.mode && !force) return;
      this.stop();
      this.mode = mode;
      this.modeBtns.forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
      // forward mode: forward steps only, without activation saving
      this.visible = this.tl.steps
        .map((st, i) => ({ st, i }))
        .filter(({ st }) => mode === "train" || (!st.bwd && !st.save))
        .map(({ i }) => i);
      this.buildStepper();
      this.buildProgram();
      this.show(-1);
    }

    buildStepper() {
      this.stepper.textContent = "";
      this.segs = this.visible.map((real, vi) => {
        const st = this.tl.steps[real];
        const seg = document.createElement("button");
        seg.className = "tpv-seg" + (st.bwd ? " bwd" : "") + (st.save ? " save" : "");
        seg.title = `${st.bwd ? "◀ " : ""}L${st.layer} · ${PHASE_TITLE[st.phase] || st.phase} · ${st.kind}`;
        seg.addEventListener("click", () => { this.stop(); this.show(vi); });
        this.stepper.appendChild(seg);
        return seg;
      });
    }

    buildProgram() {
      this.programList.textContent = "";
      this.progEntries = [];
      this.startEntry = document.createElement("div");
      this.startEntry.className = "tpv-prog-entry tpv-prog-start";
      this.startEntry.innerHTML = `<div class="tpv-prog-line">initial state</div>`;
      this.startEntry.addEventListener("click", () => { this.stop(); this.show(-1); });
      this.programList.appendChild(this.startEntry);
      let sectionKey = "";
      this.visible.forEach((real, vi) => {
        const st = this.tl.steps[real];
        const key = `${st.bwd ? "b" : "f"}|${st.layer}|${st.phase}`;
        if (key !== sectionKey) {
          sectionKey = key;
          const head = document.createElement("div");
          head.className = `tpv-prog-sec ${st.bwd ? "bwd" : "fwd"}`;
          head.textContent = st.phase === "pipeline"
            ? `${st.bwd ? "◀ backward · " : ""}pipeline`
            : `${st.bwd ? "◀ backward · " : ""}layer ${st.layer} · ${(PHASE_TITLE[st.phase] || st.phase).toLowerCase()}`;
          this.programList.appendChild(head);
        }
        const e = document.createElement("div");
        e.className = "tpv-prog-entry" + (st.bwd ? " bwd" : "");
        const line = document.createElement("div");
        line.className = "tpv-prog-line";
        if (st.save) line.innerHTML = `<span class="tpv-save">save</span>`;
        const eq = document.createElement("span");
        eq.className = "meq";
        eq.innerHTML = st.eqh;
        line.appendChild(eq);
        e.appendChild(line);

        const sub = document.createElement("div");
        sub.className = "tpv-prog-sub";
        if (st.noteh) {
          const note = document.createElement("div");
          note.className = "tpv-prog-note meq";
          note.innerHTML = st.noteh;
          sub.appendChild(note);
        }
        if (st.fwdStep !== undefined) {
          const link = document.createElement("button");
          link.className = "tpv-fwd-link";
          link.textContent = `↳ jump to that forward step`;
          link.addEventListener("click", (ev) => {
            ev.stopPropagation();
            this.stop();
            const fv = this.visible.indexOf(st.fwdStep);
            if (fv >= 0) this.show(fv);
          });
          sub.appendChild(link);
        }
        if (sub.children.length) e.appendChild(sub);
        e.addEventListener("click", () => { this.stop(); this.show(vi); });
        this.programList.appendChild(e);
        this.progEntries.push(e);
      });
      this._noScrollUntil = 0;
      this.programList.addEventListener("scroll", () => {
        this._noScrollUntil = performance.now() + 2500;
      }, { passive: true });
    }

    /* ------------------------------------------------------ state apply */
    applyState(obj, st, visibleFlag) {
      const el = this.els[obj];
      const hidden = !visibleFlag || (this.mode === "fwd" && this.saveObjs.has(obj));
      if (hidden) { el.style.display = "none"; return; }
      el.style.display = "";
      const [x, y, w, o] = st;
      el.setAttribute("transform", `translate(${SX(x).toFixed(1)},${SY(y).toFixed(1)})${w !== 100 ? ` scale(${(w / 100).toFixed(4)})` : ""}`);
      el.setAttribute("opacity", (o / 100).toFixed(3));
    }

    applyBoundary(idx) {
      const s = this.states[idx];
      for (let k = 0; k < this.els.length; k++) {
        this.applyState(k, s.st[k] || [0, 0, 100, 0], !!s.alive[k] && !!s.st[k]);
      }
    }

    show(vi) {
      vi = Math.max(-1, Math.min(vi, this.visible.length - 1));
      this.cur = vi;
      const real = vi < 0 ? -1 : this.visible[vi];
      this.applyBoundary(real + 1);
      this.updateChrome();
    }

    updateChrome() {
      if (!this.segs || !this.progEntries) return;
      const vi = this.cur;
      this.segs.forEach((seg, k) => {
        seg.classList.toggle("done", k < vi);
        seg.classList.toggle("current", k === vi);
      });
      this.progEntries.forEach((e, k) => {
        e.classList.toggle("current", k === vi);
        e.classList.toggle("done", k < vi);
      });
      if (this.startEntry) this.startEntry.classList.toggle("current", vi === -1);
      const cur = vi === -1 ? this.startEntry : this.progEntries[vi];
      if (cur && performance.now() > (this._noScrollUntil || 0)) {
        const list = this.programList;
        const lr = list.getBoundingClientRect();
        const er = cur.getBoundingClientRect();
        const target = list.scrollTop + (er.top - lr.top) - (list.clientHeight - er.height) / 2;
        list.scrollTo({ top: Math.max(0, target), behavior: "smooth" });
      }
      this.posEl.textContent = `${vi + 1} / ${this.visible.length}`;
      this.playBtn.textContent = this.playing ? "❚❚" : "▶";
      const st = vi >= 0 ? this.tl.steps[this.visible[vi]] : null;
      this.updateAlgo(st && RING_TITLES[st.kind] ? st.kind : null);
    }

    /* ------------------------------------------------------- play mode */
    async play() {
      if (this.playing) return;
      this.playing = true;
      const token = ++this._token;
      this.updateChrome();
      while (this.playing && this._token === token && this.cur < this.visible.length - 1) {
        const vi = this.cur + 1;
        const real = this.visible[vi];
        const prevReal = vi === 0 ? -1 : this.visible[vi - 1];
        if (real - prevReal > 1) {
          // silently absorb skipped (hidden) steps between the two
          this.applyBoundary(real);
        }
        this.cur = vi;          // highlight the op we are ABOUT to animate
        this.updateChrome();
        await this.playStep(real, token);
        if (this._token !== token) return;
        this.applyBoundary(real + 1); // exact end state, highlight unchanged
        await new Promise((r) => setTimeout(r, 240 * (2 / this.speed)));
      }
      if (this._token === token) { this.playing = false; this.updateChrome(); }
    }

    stop() {
      this.playing = false;
      this._token++;
      if (this.playBtn) this.updateChrome();
    }

    async playStep(real, token) {
      const step = this.tl.steps[real];
      const from = this.states[real];
      const live = new Map();
      for (const beat of step.beats) {
        if (this._token !== token) return;
        const spawnAt = new Map((beat.sp || []).map((r) => [r[0], r.slice(1)]));
        const moves = [];
        for (const row of beat.ch) {
          const [obj, x, y, w, o] = row;
          const path = row[5]; // interior (x, y) samples for arced flights
          if (this.mode === "fwd" && this.saveObjs.has(obj)) continue;
          const el = this.els[obj];
          const prev = live.get(obj) || from.st[obj];
          const spawning = beat.in.includes(obj) || (!from.alive[obj] && !live.has(obj));
          let start;
          if (spawning) {
            const sp = spawnAt.get(obj);
            start = sp ? [sp[0], sp[1], sp[2], 0] : [x, y, w, 0];
          } else {
            start = prev || [x, y, w, 0];
          }
          el.style.display = "";
          moves.push({ obj, el, start, end: [x, y, w, o], path });
          live.set(obj, [x, y, w, o]);
        }
        const outs = beat.out.map((obj) => this.els[obj]);
        await this.tween(moves, outs, beat.d * (2 / this.speed), token);
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
            if (m.path) {
              const pts = [[m.start[0], m.start[1]]];
              for (let k = 0; k < m.path.length; k += 2) pts.push([m.path[k], m.path[k + 1]]);
              pts.push([m.end[0], m.end[1]]);
              const f = p * (pts.length - 1);
              const seg = Math.min(Math.floor(f), pts.length - 2);
              const t = f - seg;
              v[0] = pts[seg][0] + (pts[seg + 1][0] - pts[seg][0]) * t;
              v[1] = pts[seg][1] + (pts[seg + 1][1] - pts[seg][1]) * t;
            }
            this.applyState(m.obj, v, true);
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
        const tbl = document.createElement("table");
        for (const [k, shape, mem] of [
          ["global", data.global.join(" × "), data.memGlobal],
          ["this device", data.local.join(" × "), data.memLocal],
        ]) {
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

  /* ---------------------------- ring-collective algorithm inset ---------- */
  const RING_TITLES = {
    AllGather: "how it runs: bidirectional ring AllGather — each shard splits in half and circulates both ways; every device keeps a copy as it passes",
    ReduceScatter: "how it runs: bidirectional ring ReduceScatter — one running sum per chunk circles the ring (half clockwise, half counter); every node it passes adds its contribution",
    AllReduce: "how it runs: ReduceScatter, then AllGather — reduce toward owners, then circulate the results",
    AllToAll: "how it runs: ring AllToAll — every chunk hops the shortest way around to its destination device",
    P2PSend: "how it runs: a single point-to-point send between neighboring stages",
  };

  TPVizFigure.prototype.updateAlgo = function (kind) {
    if (kind === this._algoKind) return;
    this._algoKind = kind;
    this._algoToken++;
    this.band.classList.toggle("has-algo", !!kind);
    if (!kind) return;
    this.algo.querySelector(".tpv-algo-title").textContent = RING_TITLES[kind];
    this.runAlgo(kind, ++this._algoToken);
  };

  TPVizFigure.prototype.runAlgo = async function (kind, token) {
    const svg = this.algoSvg;
    const hopEl = this.algo.querySelector(".tpv-algo-hop");
    const N = Math.min(this.tl.objects.length ? 4 : 4, 4);
    const cx = (i) => 170 + i * 220;
    const CY = 95, NW = 150, NH = 120;
    const devVar = (i) => `var(--cv-dev${i})`;
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    while (this._algoToken === token && this.band.classList.contains("has-algo")) {
      svg.textContent = "";
      // ring links (neighbors + wrap arc), arrowed both directions
      for (let i = 0; i < N - 1; i++) {
        svgEl("line", {
          x1: cx(i) + NW / 2, y1: CY, x2: cx(i + 1) - NW / 2, y2: CY,
          stroke: "var(--cv-muted)", "stroke-width": 1.4, "stroke-opacity": 0.6,
        }, svg);
      }
      svgEl("path", {
        d: `M ${cx(0)} ${CY - NH / 2} C ${cx(0)} 8, ${cx(N - 1)} 8, ${cx(N - 1)} ${CY - NH / 2}`,
        fill: "none", stroke: "var(--cv-muted)", "stroke-width": 1.4,
        "stroke-opacity": 0.6, "stroke-dasharray": "5 5",
      }, svg);
      // nodes with N slot columns
      const slotPt = (node, col, row) =>
        [cx(node) - NW / 2 + 18 + col * 30, CY - 22 + row * 34];
      for (let i = 0; i < N; i++) {
        svgEl("rect", {
          x: cx(i) - NW / 2, y: CY - NH / 2, width: NW, height: NH, rx: 10,
          fill: "var(--cv-boxFill)", stroke: devVar(i), "stroke-width": 1.6,
        }, svg);
        const t = svgEl("text", {
          x: cx(i), y: CY - NH / 2 + 14, "text-anchor": "middle",
          "font-size": 12, "font-weight": 600, fill: devVar(i),
        }, svg);
        t.textContent = `Dev ${i}`;
      }
      const chunk = (hue, x, y, o = 1, half = 0) =>
        svgEl("rect", {
          x: x - 10, y: y - 8 + (half ? 9 : 0), width: 20, height: half ? 7 : 16,
          rx: 2.5, fill: hue, "fill-opacity": o, stroke: "var(--cv-text)",
          "stroke-opacity": 0.25, "stroke-width": 0.8,
        }, svg);
      const move = (el, x1, y1, x2, y2, dur) => new Promise((res) => {
        el.setAttribute("transform", `translate(${x1},${y1})`); // no origin flash
        const t0 = performance.now();
        const f = (now) => {
          if (this._algoToken !== token) return res();
          const u = Math.min(1, (now - t0) / dur);
          const p = u * u * (3 - 2 * u);
          const dy = -26 * Math.sin(Math.PI * p); // small arc over the link
          el.setAttribute("transform",
            `translate(${(x1 + (x2 - x1) * p).toFixed(1)},${(y1 + (y2 - y1) * p + dy).toFixed(1)})`);
          if (u < 1) requestAnimationFrame(f); else res();
        };
        requestAnimationFrame(f);
      });
      const flyer = (hue, o, half) => {
        const g = svgEl("g", {}, svg);
        chunk(hue, 0, 0, o, half);
        return g;
      };
      const hopMove = async (jobs, dur) => {
        await Promise.all(jobs.map(async (j) => {
          const el = flyer(j.hue, j.o ?? 1, j.half ?? 0);
          const [x1, y1] = j.from, [x2, y2] = j.to;
          await move(el, x1, y1, x2, y2, dur);
          el.remove();
          if (j.land) j.land();
        }));
      };
      const setHop = (k, total, label) => {
        hopEl.textContent = `${label || "hop"} ${k} / ${total}`;
      };

      if (kind === "AllGather" || kind === "AllReduce") {
        // (AllReduce shows its AG phase second; RS phase first below)
      }

      const runAG = async (label) => {
        // store-and-forward, both directions; slot grid marks copies held
        const held = [];
        for (let i = 0; i < N; i++) {
          for (let h = 0; h < 2; h++) {
            const [x, y] = slotPt(i, i, h);
            held.push(chunk(devVar(i), x, y, 1, 1));
          }
        }
        for (let k = 1; k <= N - 1; k++) {
          if (this._algoToken !== token) return;
          setHop(k, N - 1, label);
          const jobs = [];
          for (let i = 0; i < N; i++) {
            const cwOrigin = (i - k + 1 + N) % N;   // half moving clockwise
            const ccwOrigin = (i + k - 1) % N;      // half moving counterclockwise
            const cwTo = (i + 1) % N, ccwTo = (i - 1 + N) % N;
            jobs.push({
              hue: devVar(cwOrigin), half: 1,
              from: slotPt(i, cwOrigin, 0), to: slotPt(cwTo, cwOrigin, 0),
              land: () => chunk(devVar(cwOrigin), ...slotPt(cwTo, cwOrigin, 0), 1, 1),
            });
            jobs.push({
              hue: devVar(ccwOrigin), half: 1,
              from: slotPt(i, ccwOrigin, 1), to: slotPt(ccwTo, ccwOrigin, 1),
              land: () => chunk(devVar(ccwOrigin), ...slotPt(ccwTo, ccwOrigin, 1), 1, 1),
            });
          }
          await hopMove(jobs, 800);
          await sleep(320);
        }
      };

      const runRS = async (label) => {
        // Every node holds a contribution to every chunk (faded, colored by
        // the chunk's OWNER). One accumulator per chunk circles the ring —
        // half clockwise, half counterclockwise — and at every hop the local
        // contribution visibly slides INTO it, growing the running sum.
        const contrib = {};
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < N; j++) {
            contrib[`${i},${j}`] = chunk(devVar(j), ...slotPt(i, j, 1), 0.45);
          }
        }
        const carryPt = (node, j) => {
          const [x] = slotPt(node, j, 0);
          return [x, CY - 40];
        };
        const accSize = (k) => 16 + k * 7; // the running sum grows per absorb
        const mkAcc = (j, node, k, o) => {
          const [x, y] = carryPt(node, j);
          const g = svgEl("g", { transform: `translate(${x},${y})` }, svg);
          svgEl("rect", {
            x: -accSize(k) / 2, y: -9, width: accSize(k), height: 18, rx: 3,
            fill: devVar(j), "fill-opacity": o,
            stroke: "var(--cv-text)", "stroke-opacity": 0.35, "stroke-width": 1,
          }, g);
          return g;
        };
        const absorb = async (a, node) => {
          const c = contrib[`${node},${a.j}`];
          if (!c) return;
          const [fx, fy] = slotPt(node, a.j, 1);
          const [tx, ty] = carryPt(node, a.j);
          c.remove();
          delete contrib[`${node},${a.j}`];
          const fl = flyer(devVar(a.j), 0.8, 0);
          await move(fl, fx, fy, tx, ty, 340);
          fl.remove();
          a.k += 1;
          a.o = Math.min(1, 0.45 + 0.55 * (a.k / N));
          if (a.el) a.el.remove();
          a.el = mkAcc(a.j, node, a.k, a.o);
        };
        const acc = [];
        for (let j = 0; j < N; j++) {
          const dir = j % 2 === 0 ? +1 : -1; // half go each way around the ring
          const at = (j + dir + N) % N;      // farthest-from-owner start
          acc.push({ j, dir, at, k: 0, o: 0.45, el: null });
        }
        setHop(0, N - 1, label);
        await Promise.all(acc.map((a) => absorb(a, a.at)));
        for (let k = 1; k <= N - 1; k++) {
          if (this._algoToken !== token) return;
          setHop(k, N - 1, label);
          await Promise.all(acc.map(async (a) => {
            const nxt = (a.at + a.dir + N) % N;
            const [x1, y1] = carryPt(a.at, a.j);
            const [x2, y2] = carryPt(nxt, a.j);
            a.el.remove();
            const fl = svgEl("g", {}, svg);
            const inner = svgEl("rect", {
              x: -accSize(a.k) / 2, y: -9, width: accSize(a.k), height: 18, rx: 3,
              fill: devVar(a.j), "fill-opacity": a.o,
              stroke: "var(--cv-text)", "stroke-opacity": 0.35, "stroke-width": 1,
            }, fl);
            await move(fl, x1, y1, x2, y2, 800);
            fl.remove();
            a.at = nxt;
            a.el = mkAcc(a.j, nxt, a.k, a.o);
          }));
          if (this._algoToken !== token) return;
          await Promise.all(acc.map((a) => absorb(a, a.at)));
          await sleep(260);
        }
        // arrived: the fully reduced chunk settles into its owner's slot
        await Promise.all(acc.map(async (a) => {
          const [x1, y1] = carryPt(a.j, a.j);
          const [x2, y2] = slotPt(a.j, a.j, 1);
          a.el.remove();
          const fl = svgEl("g", {}, svg);
          svgEl("rect", {
            x: -accSize(a.k) / 2, y: -9, width: accSize(a.k), height: 18, rx: 3,
            fill: devVar(a.j), "fill-opacity": 1,
            stroke: "var(--cv-text)", "stroke-opacity": 0.35, "stroke-width": 1,
          }, fl);
          await move(fl, x1, y1, x2, y2, 380);
          fl.remove();
          chunk(devVar(a.j), x2, y2, 1);
        }));
      };

      const runA2A = async () => {
        const placed = {};
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < N; j++) {
            placed[`${i},${j}`] = chunk(devVar(j), ...slotPt(i, j, 0), i === j ? 1 : 0.75);
          }
        }
        const maxHops = Math.floor(N / 2);
        for (let k = 1; k <= maxHops; k++) {
          if (this._algoToken !== token) return;
          setHop(k, maxHops);
          const jobs = [];
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              const dist = ((j - i) % N + N) % N;
              const dir = dist <= N / 2 ? +1 : -1;
              const hops = Math.min(dist, N - dist);
              if (hops < k) continue;
              const cur = (i + dir * (k - 1) + N) % N;
              const nxt = (cur + dir + N) % N;
              jobs.push({
                hue: devVar(j), o: 0.85,
                from: slotPt(cur, j, 0), to: slotPt(nxt, j, 0),
                land: () => { if (nxt === j) chunk(devVar(j), ...slotPt(j, j === i ? i : i, 1), 1); },
              });
              const el = placed[`${i},${j}`];
              if (el && k === 1) el.setAttribute("fill-opacity", 0.15);
            }
          }
          await hopMove(jobs, 800);
          await sleep(320);
        }
      };

      if (kind === "AllGather") await runAG();
      else if (kind === "ReduceScatter") await runRS();
      else if (kind === "AllReduce") { await runRS("reduce-scatter · hop"); await sleep(500); await runAG("all-gather · hop"); }
      else if (kind === "AllToAll") await runA2A();
      if (this._algoToken !== token) return;
      await sleep(1600);
    }
  };

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
