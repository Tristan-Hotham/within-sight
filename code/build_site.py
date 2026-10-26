"""Step 4: build ../index.html from page_template.html.

Every number in the text is a placeholder like {{network.20.pairs_visible|int}}
filled from ../data/results.json, and the compact site data (../work/site.json)
is embedded as `const D = ...`.
Formats: int (thousands commas), pct0, pct1, f1, f2, raw, x2 (ratio with x).
Run: python3 build_site.py
"""
import os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
R = json.load(open(os.path.join(ROOT, "data", "results.json")))
site = open(os.path.join(ROOT, "work", "site.json")).read()
# a few Atlas names carry a stray en dash from an encoding slip in the source; show it as a hyphen on the page
site = site.replace("\u2013", "-").replace("\u2014", "-")


def get(path):
    v = R
    for k in path.split("."):
        v = v[int(k)] if isinstance(v, list) else v[k]
    return v


def fmt(v, f):
    if f == "int": return f"{int(round(v)):,}"
    if f == "pct0": return f"{v * 100:.0f}%"
    if f == "pct1": return f"{v * 100:.1f}%"
    if f == "f1": return f"{v:.1f}"
    if f == "f2": return f"{v:.2f}"
    if f == "words":
        w = "zero one two three four five six seven eight nine ten eleven twelve".split()
        return w[int(v)] if int(v) < len(w) else f"{int(v):,}"
    return str(v)


def sub(m):
    path, _, f = m.group(1).partition("|")
    return fmt(get(path.strip()), f.strip() or "raw")


t = open(os.path.join(HERE, "page_template.html")).read().replace("<!--PROSE-->", open(os.path.join(HERE, "prose.html")).read())
t = re.sub(r"\{\{(.+?)\}\}", sub, t)
t = t.replace("/*DATA*/", "const D = " + site + ";")
open(os.path.join(ROOT, "index.html"), "w").write(t)
print("index.html", len(t.encode()) // 1024, "KB")
md = re.sub(r"\{\{(.+?)\}\}", sub, open(os.path.join(HERE, "README_template.md")).read())
open(os.path.join(ROOT, "README.md"), "w").write(md)
print("README.md written")
