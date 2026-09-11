r"""Convert OUR notation's LaTeX (we generate every string in notation.py, so
the grammar is closed) into selectable HTML, KaTeX-style: math italic by
default, upright for \text/\mathrm, real <sub>/<sup> elements.

Handled tokens: \text{..} \mathrm{..} \mathit{..} _{..} ^{..} ^\top \{ \}
\to \cdot \partial \Rightarrow \; \, \quad \ and --- em-dash.
"""

from __future__ import annotations

import html
import re

_CMD = re.compile(r"\\(text|mathrm|mathit)\{")
_SIMPLE = {
    r"\to": " → ",
    r"\cdot": "·",
    r"\partial": "∂",
    r"\Rightarrow": " ⇒ ",
    r"\odot": "⊙",
    r"\top": "⊤",
    r"\{": "{",
    r"\}": "}",
    r"\;": " ",
    r"\,": "\u2009",
    r"\quad": "\u2003",
    r"\ ": " ",
    "---": "—",
    "--": "–",
}


def _read_group(s: str, i: int) -> tuple[str, int]:
    """s[i] == '{'; return (content, index after closing brace)."""
    depth, j = 1, i + 1
    while j < len(s) and depth:
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
        j += 1
    return s[i + 1 : j - 1], j


def tex_to_html(tex: str) -> str:
    out: list[str] = []
    i, n = 0, len(tex)
    while i < n:
        ch = tex[i]
        if ch == "\\":
            m = _CMD.match(tex, i)
            if m:
                body, i = _read_group(tex, m.end() - 1)
                cls = "mu" if m.group(1) in ("text", "mathrm") else "mi"
                out.append(f'<span class="{cls}">{tex_to_html(body)}</span>')
                continue
            for tok, rep in _SIMPLE.items():
                if tok.startswith("\\") and tex.startswith(tok, i):
                    nxt = i + len(tok)
                    # avoid matching a prefix of a longer command name
                    if tok[1].isalpha() and nxt < n and tex[nxt].isalpha():
                        continue
                    out.append(html.escape(rep) if rep in "<>&" else rep)
                    i = nxt
                    break
            else:
                i += 1  # unknown command char: drop the backslash
            continue
        if ch in "_^":
            tag = "sub" if ch == "_" else "sup"
            if i + 1 < n and tex[i + 1] == "{":
                body, i = _read_group(tex, i + 1)
            elif tex.startswith(r"\top", i + 1):
                body, i = r"\top", i + 5
            else:
                body, i = tex[i + 1], i + 2
            out.append(f"<{tag}>{tex_to_html(body)}</{tag}>")
            continue
        if tex.startswith("---", i):
            out.append("—")
            i += 3
            continue
        if ch == "{":  # bare group: invisible, recurse into its content
            body, i = _read_group(tex, i)
            out.append(tex_to_html(body))
            continue
        if ch == "}":
            i += 1
            continue
        out.append(html.escape(ch))
        i += 1
    return "".join(out)
