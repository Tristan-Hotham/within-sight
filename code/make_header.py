"""Render ../header.png (1200x630 social card) with Playwright Chromium.

Fonts: pass a folder of TTFs as FONT_DIR (Fraunces, Source Serif 4, IBM Plex
Mono from github.com/google/fonts); otherwise the system serif is used.
Run: python3 make_header.py
"""
import os, json, math
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
S = json.load(open(os.path.join(ROOT, "work", "site.json")))
R = S["R"]; FD = os.environ.get("FONT_DIR", "")
F = S["f"]; E = S["e"]["20"]


def proj(lat, lon):
    lat0 = 54.5
    return lon * math.cos(math.radians(lat0)), -lat


xs, ys = zip(*[proj(r[0], r[1]) for r in F])
x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
H, W = 600, 600 * (x1 - x0) / (y1 - y0)
P = [((x - x0) / (y1 - y0) * H, (y - y0) / (y1 - y0) * H) for x, y in zip(xs, ys)]
seg = " ".join(f"M{P[a][0]:.1f} {P[a][1]:.1f}L{P[b][0]:.1f} {P[b][1]:.1f}" for a, b in zip(E[::2], E[1::2]))
dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.1"/>' for x, y in P)
ff = ""
if FD:
    ff = "".join(f"@font-face{{font-family:'{n}';src:url('file://{FD}/{f}');font-weight:{w};}}" for n, f, w in [
        ("Fraunces", "Fraunces[SOFT,WONK,opsz,wght].ttf", "100 900"), ("Source Serif 4", "SourceSerif4[opsz,wght].ttf", "200 900"),
        ("IBM Plex Mono", "IBMPlexMono-Regular.ttf", "400"), ("IBM Plex Mono", "IBMPlexMono-Medium.ttf", "500")])
n20 = R["network"]["20"]
html = f"""<!doctype html><html><head><meta charset="utf-8"><style>{ff}
body{{margin:0;width:1200px;height:630px;background:#15140F;color:#EDE8DC;overflow:hidden;position:relative;font-family:'Source Serif 4',Georgia,serif}}
.k{{position:absolute;left:70px;top:96px;font:500 15px/1 'IBM Plex Mono',monospace;letter-spacing:.14em;text-transform:uppercase;color:#A59E8E;display:flex;gap:12px;align-items:center}}
.k b{{background:#F26A30;color:#15140F;padding:7px 9px;font-weight:500}}
h1{{position:absolute;left:66px;top:140px;margin:0;font:900 132px/.86 Fraunces,Georgia,serif;letter-spacing:-.02em;font-variation-settings:"SOFT" 50}}
h1 em{{font-style:normal;color:#F26A30;display:block}}
p{{position:absolute;left:70px;top:405px;width:520px;margin:0;font-size:27px;line-height:1.36;color:#E1DACB}}
.f{{position:absolute;left:70px;bottom:44px;font:400 14px/1 'IBM Plex Mono',monospace;letter-spacing:.12em;text-transform:uppercase;color:#857D6B}}
svg{{position:absolute;right:40px;top:15px}}
</style></head><body>
<div class="k"><b>Data study</b><span>Archaeology × landscape</span></div>
<h1>Within<em>sight</em></h1>
<p>{n20['pairs_visible']:,} pairs of hillforts are in line of sight within 20 km. Was that by design?</p>
<div class="f">Dr Tristan Hotham · Independent researcher</div>
<svg width="{W:.0f}" height="{H}" viewBox="0 0 {W:.0f} {H}"><path d="{seg}" stroke="#F26A30" stroke-width="1.1" stroke-opacity=".75" fill="none"/><g fill="#EDE8DC" fill-opacity=".4">{dots}</g></svg>
</body></html>"""
tmp = os.path.join(ROOT, "work", "header.html"); open(tmp, "w").write(html)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1200, "height": 630})
    pg.goto("file://" + os.path.abspath(tmp)); pg.wait_for_timeout(500)
    pg.screenshot(path=os.path.join(ROOT, "header.png")); b.close()
print("header.png written")
