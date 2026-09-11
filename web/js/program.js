"use strict";
/* The program panel: the pass as a listing, grouped by (direction, layer,
   phase), live-highlighted, click-to-seek. */

const PHASE_TITLE = { attn: "Attention", mlp: "MLP", moe: "MoE MLP", pipeline: "Pipeline", "": "" };

class Program {
  constructor(listEl, onSeek) {
    this.listEl = listEl;
    this.onSeek = onSeek;
    this.entries = [];
    this.current = -1;
    this._noScrollUntil = 0;
    listEl.addEventListener("scroll", () => { this._noScrollUntil = performance.now() + 3000; }, { passive: true });
  }

  build(tl) {
    this.listEl.textContent = "";
    this.entries = [];
    this.current = -1;
    let section = null, sectionKey = "";
    tl.steps.forEach((st, i) => {
      const key = `${st.bwd ? "b" : "f"}|${st.layer}|${st.phase}`;
      if (key !== sectionKey) {
        sectionKey = key;
        section = document.createElement("div");
        section.className = "prog-section open";
        const head = document.createElement("div");
        head.className = `prog-section-head ${st.bwd ? "bwd" : "fwd"}`;
        const title = st.phase === "pipeline"
          ? `${st.bwd ? "◀ Backward · " : ""}Pipeline`
          : `${st.bwd ? "◀ Backward · " : ""}Layer ${st.layer} · ${PHASE_TITLE[st.phase] || st.phase}`;
        head.innerHTML = `<span class="chev">▶</span><span>${title}</span>`;
        head.addEventListener("click", () => section.classList.toggle("open"));
        section.appendChild(head);
        const wrap = document.createElement("div");
        wrap.className = "prog-entries";
        section.appendChild(wrap);
        this.listEl.appendChild(section);
      }
      const entry = document.createElement("div");
      entry.className = "prog-entry" + (st.bwd ? " bwd" : "") + (st.kind !== "MatMul" && st.kind !== "Gelu" && st.kind !== "SplitQKV" && st.kind !== "MergeQKV" && st.kind !== "AttentionCore" && st.kind !== "AttentionBwd" && st.kind !== "SaveActivation" && st.kind !== "Route" ? " comm" : "");
      const eqline = document.createElement("div");
      eqline.className = "eqline";
      if (st.kind === "SaveActivation") {
        const chip = document.createElement("span");
        chip.className = "savechip";
        chip.textContent = "save";
        eqline.appendChild(chip);
      }
      if (st.eq) eqline.appendChild(eqSvg(st.eq, 13));
      else eqline.textContent = st.kind + (st.mb !== undefined ? ` · mb${st.mb}` : "");
      entry.appendChild(eqline);
      if (st.note) {
        const note = document.createElement("div");
        note.className = "note";
        note.appendChild(eqSvg(st.note, 10));
        entry.appendChild(note);
      }
      if (st.cap) {
        const cap = document.createElement("div");
        cap.className = "cap";
        cap.textContent = st.cap;
        entry.appendChild(cap);
      }
      entry.addEventListener("click", () => this.onSeek(st.t0 + 1));
      section.querySelector(".prog-entries").appendChild(entry);
      this.entries.push({ el: entry, st, section });
    });
  }

  update(t) {
    let cur = -1;
    for (let i = 0; i < this.entries.length; i++) {
      if (this.entries[i].st.t0 <= t) cur = i; else break;
    }
    if (cur === this.current) return;
    this.current = cur;
    this.entries.forEach((e, i) => {
      e.el.classList.toggle("current", i === cur);
      e.el.classList.toggle("done", i < cur);
    });
    if (cur >= 0) {
      const e = this.entries[cur];
      if (!e.section.classList.contains("open")) e.section.classList.add("open");
      if (performance.now() > this._noScrollUntil) {
        e.el.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }
  }
}
