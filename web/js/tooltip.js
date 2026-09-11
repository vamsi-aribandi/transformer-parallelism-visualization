"use strict";
/* Hover metadata for tensors: notation, shapes with concrete dims, sharding,
   memory. Attached by delegation on the canvas. */

const KIND_COLOR = { activation: "var(--act)", weight: "var(--wt)", kv: "var(--kv)", grad: "var(--grad)" };

function initTooltip(stageEl, canvasEl) {
  const tip = document.getElementById("tooltip");

  function show(target, ev) {
    const data = DOC.tooltips[+target.dataset.tip];
    if (!data) return;
    tip.textContent = "";
    const head = document.createElement("div");
    head.className = "tt-eq";
    if (target.dataset.tex) head.appendChild(eqSvg(target.dataset.tex, 15));
    const chip = document.createElement("span");
    chip.className = "kindchip";
    chip.style.color = KIND_COLOR[data.kind] || "var(--text)";
    chip.textContent = data.kind;
    head.appendChild(chip);
    tip.appendChild(head);

    const tbl = document.createElement("table");
    const rows = [
      ["global", `${data.global.join(" × ")}`, data.memGlobal],
      ["this device", `${data.local.join(" × ")}`, data.memLocal],
    ];
    for (const [k, shape, mem] of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${k}</td><td class="num">${shape}</td><td class="num" style="padding-left:12px;color:var(--muted)">${mem}</td>`;
      tbl.appendChild(tr);
    }
    tip.appendChild(tbl);

    const sd = document.createElement("div");
    sd.className = "shard-desc";
    sd.textContent = data.shardDesc;
    tip.appendChild(sd);

    if (data.partial) {
      const pw = document.createElement("div");
      pw.className = "partial-warn";
      pw.textContent = `unreduced partial sum over ${data.partial.join(", ")} — not yet a real value`;
      tip.appendChild(pw);
    }
    tip.hidden = false;
    move(ev);
  }

  function move(ev) {
    const r = stageEl.getBoundingClientRect();
    let x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 14;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    if (x + tw > r.width - 8) x = ev.clientX - r.left - tw - 14;
    if (y + th > r.height - 8) y = ev.clientY - r.top - th - 14;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }

  canvasEl.addEventListener("pointerover", (ev) => {
    const t = ev.target.closest(".hit");
    if (t) show(t, ev);
  });
  canvasEl.addEventListener("pointermove", (ev) => {
    if (!tip.hidden) move(ev);
  });
  canvasEl.addEventListener("pointerout", (ev) => {
    if (!ev.relatedTarget || !ev.relatedTarget.closest || !ev.relatedTarget.closest(".hit")) tip.hidden = true;
  });
}
