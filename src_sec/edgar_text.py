# ============================================================
# START DATE   : 2026-11-02
# CODE RUN DATE: 2026-11-02
# TIME TAKEN   : ~5 min
# DESCRIPTION  : Shared helper: strip EDGAR HTML and return paragraphs that contain "tariff" (case-insensitive).
# INPUT FILES  : raw filing HTML bytes (in memory)
# OUTPUT FILES : none
# ============================================================
import re
from html.parser import HTMLParser
MIN_LEN = 60
RE_KW = re.compile(r"tariff", re.I)              # ★ CORE LOGIC: sole keyword filter (case-insensitive)
BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "section"}
VOID = {"br", "meta", "img", "hr", "input", "link"}

class T(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.parts = []; self.skip = 0; self.stack = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in VOID:
            if tag in BLOCK: self.parts.append("\n")
            return
        if tag in ("script", "style", "ix:header") or "display:none" in (a.get("style") or "").replace(" ", ""):
            self.skip += 1; self.stack.append(tag)
        else: self.stack.append(None)
        if tag in BLOCK: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in VOID: return
        if tag in BLOCK: self.parts.append("\n")
        if self.stack and self.stack.pop() is not None: self.skip -= 1
    def handle_data(self, d):
        if not self.skip: self.parts.append(d)

def tariff_paragraphs(raw_bytes):
    html = raw_bytes.decode("utf8", "ignore")
    p = T(); p.feed(html)
    out = []
    for para in "".join(p.parts).replace("\xa0", " ").split("\n"):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) >= MIN_LEN and RE_KW.search(para): out.append(para)
    return out

