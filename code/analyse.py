"""Step 3: network, regional, regression and null-model summaries.

Reads ../work/*, writes ../data/*.csv, ../data/results.json and ../work/site.json
(the compact data embedded in index.html).
Run: python3 analyse.py
"""
import os, json, re
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr, mannwhitneyu
import statsmodels.api as sm
from visibility import WORK
from dem import haversine

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); os.makedirs(DATA, exist_ok=True)
KMS = (10, 20, 30)
R = {}

f = pd.read_csv(os.path.join(WORK, "forts_base.csv"))
topo = pd.read_csv(os.path.join(WORK, "forts_topo.csv"))[["atlas_id", "local_pct", "rel_height_m"]]
f = f.merge(topo, on="atlas_id", how="left")
P = np.load(os.path.join(WORK, "pairs_obs.npz"))
pairs, dist, v2, v5, vs = P["pairs"].astype(np.int64), P["dist"].astype(np.float64), P["v2"], P["v5"], P["vs"]
n = len(f)
lat, lon = f.lat.values, f.lon.values


def comps(edges):
    g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    nc, lab = connected_components(g, directed=False)
    return g, lab


def r1(x, d=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def path_from_pred(pred, i, j):
    p = [j]
    while p[-1] != i:
        p.append(int(pred[p[-1]]))
        if p[-1] < 0:
            return None
    return p[::-1]


def route_info(path):
    km = sum(haversine(lat[a], lon[a], lat[b], lon[b]) for a, b in zip(path[:-1], path[1:])) / 1000
    return dict(hops=len(path) - 1, route_km=r1(km), straight_km=r1(haversine(lat[path[0]], lon[path[0]], lat[path[-1]], lon[path[-1]]) / 1000),
                start=f.name[path[0]], end=f.name[path[-1]], start_short=re.sub(r"\s*\(.*\)\s*$", "", f.name[path[0]]), end_short=re.sub(r"\s*\(.*\)\s*$", "", f.name[path[-1]]), path=[int(x) for x in path], path_names=[f.name[x] for x in path])


R["n_forts"] = n
R["elev_zero_or_below"] = int((f.elev_m <= 0).sum())
net = {}
for km in KMS:
    m = dist <= km * 1000
    e = pairs[m & v2]
    deg = np.bincount(e.ravel(), minlength=n)
    rng_ = np.bincount(pairs[m].ravel(), minlength=n)
    f[f"forts_within_{km}km"] = rng_
    f[f"visible_{km}km"] = deg
    g, lab = comps(e)
    sizes = np.bincount(lab)
    f[f"component_{km}km"] = lab
    f[f"component_size_{km}km"] = sizes[lab]
    multi = sizes[sizes > 1]
    order = np.argsort(-sizes)
    big = order[0]
    nodes = np.flatnonzero(lab == big)
    sub = g[nodes][:, nodes]
    D, pred = shortest_path(sub, directed=False, unweighted=True, return_predecessors=True)
    D[np.isinf(D)] = -1
    i, j = np.unravel_index(np.argmax(D), D.shape)
    diam = route_info([nodes[x] for x in path_from_pred(pred[i], i, j)])
    # farthest straight-line pair linked by any chain, across all components
    best = (0, None)
    for c in order[:40]:
        nd = np.flatnonzero(lab == c)
        if len(nd) < 2: break
        dd = haversine(lat[nd][:, None], lon[nd][:, None], lat[nd][None], lon[nd][None])
        a, b = np.unravel_index(np.argmax(dd), dd.shape)
        if dd[a, b] > best[0]:
            best = (dd[a, b], (c, nd[a], nd[b]))
    c, a, b = best[1]
    nd = np.flatnonzero(lab == c)
    sub2 = g[nd][:, nd]
    _, pr = shortest_path(sub2, directed=False, unweighted=True, return_predecessors=True, indices=[int(np.flatnonzero(nd == a)[0])])
    farthest = route_info([nd[x] for x in path_from_pred(pr[0], int(np.flatnonzero(nd == a)[0]), int(np.flatnonzero(nd == b)[0]))])
    top = []
    for c in order[:10]:
        nd = np.flatnonzero(lab == c)
        cc = f.iloc[nd]
        dd = haversine(lat[nd][:, None], lon[nd][:, None], lat[nd][None], lon[nd][None])
        top.append(dict(component=int(c), size=int(len(nd)), span_km=r1(dd.max() / 1000), countries=cc.country.value_counts().to_dict(),
                        counties=cc.county.value_counts().head(5).to_dict(), edges=int(((lab[e[:, 0]] == c)).sum())))
    net[km] = dict(
        pairs_in_range=int(m.sum()), pairs_visible=int(len(e)), share_visible=r1(len(e) / m.sum(), 4),
        pairs_visible_5m=int((m & v5).sum()), share_visible_5m=r1((m & v5).sum() / m.sum(), 4),
        pairs_visible_snapped=int((m & vs).sum()), share_visible_snapped=r1((m & vs).sum() / m.sum(), 4),
        forts_with_any_in_range=int((rng_ > 0).sum()),
        forts_seeing_any=int((deg > 0).sum()), share_seeing_any=r1((deg > 0).mean(), 4),
        share_seeing_any_of_those_with_neighbours=r1((deg > 0).sum() / (rng_ > 0).sum(), 4),
        isolated=int((deg == 0).sum()), mean_visible=r1(deg.mean(), 2), median_visible=r1(np.median(deg), 1), max_visible=int(deg.max()),
        max_visible_fort=f.name[int(deg.argmax())],
        mean_in_range=r1(rng_.mean(), 2),
        n_components_multi=int(len(multi)), largest_component=int(sizes.max()), largest_component_share=r1(sizes.max() / n, 4),
        forts_in_components_10plus=int(sizes[lab][sizes[lab] >= 10].size), forts_in_components_100plus=int((sizes[lab] >= 100).sum()),
        component_size_bins={b: int(((sizes[lab] >= lo) & (sizes[lab] <= hi)).sum()) for b, lo, hi in
                             [("1", 1, 1), ("2", 2, 2), ("3 to 5", 3, 5), ("6 to 10", 6, 10), ("11 to 50", 11, 50), ("51 to 200", 51, 200), ("201+", 201, 10 ** 9)]},
        top_components=top, diameter=diam, farthest_relay=farthest,
        mean_visible_dist_km=r1(dist[m & v2].mean() / 1000, 2), median_visible_dist_km=r1(np.median(dist[m & v2]) / 1000, 2),
    )
    # distance profile: share visible by 2 km band
    if km == 30:
        bands = np.arange(0, 31, 2)
        R["distance_profile"] = [dict(lo=int(a), hi=int(b), in_range=int(((dist > a * 1000) & (dist <= b * 1000)).sum()),
                                      share=r1(v2[(dist > a * 1000) & (dist <= b * 1000)].mean(), 4)) for a, b in zip(bands[:-1], bands[1:])]
R["network"] = {str(k): v for k, v in net.items()}
f["visible_20km_5m"] = np.bincount(pairs[(dist <= 20000) & v5].ravel(), minlength=n)
f["visible_20km_snapped"] = np.bincount(pairs[(dist <= 20000) & vs].ravel(), minlength=n)
f["share_visible_20km"] = np.where(f.forts_within_20km > 0, f.visible_20km / f.forts_within_20km.clip(lower=1), np.nan)

# histogram of visible at 20 km
h = np.bincount(np.minimum(f.visible_20km.values, 20), minlength=21)
R["hist_visible_20km"] = [int(x) for x in h]
R["hist_visible"] = {str(km): [int(x) for x in np.bincount(np.minimum(f[f"visible_{km}km"].values, 20), minlength=21)] for km in KMS}

# top seers
R["top_seers_20km"] = [dict(name=r.name, county=r.county, visible=int(r.visible_20km), in_range=int(r.forts_within_20km))
                       for r in f.sort_values(["visible_20km", "forts_within_20km"], ascending=[False, True]).head(15).itertuples()]

# regional
reg = []
for c, g in f.groupby("country"):
    m = (dist <= 20000)
    inC = f.country.values
    pm = m & (inC[pairs[:, 0]] == c) & (inC[pairs[:, 1]] == c)
    reg.append(dict(country=c, n=int(len(g)), mean_visible=r1(g.visible_20km.mean(), 2), share_seeing_any=r1((g.visible_20km > 0).mean(), 4),
                    mean_in_range=r1(g.forts_within_20km.mean(), 1), share_pairs_visible=r1(v2[pm].mean(), 4) if pm.sum() else None,
                    median_elev=r1(g.elev_m.median(), 0)))
R["regional"] = sorted(reg, key=lambda d: -d["n"])
cty = []
for c, g in f.groupby("county"):
    if len(g) >= 25:
        cty.append(dict(county=c, country=g.country.iloc[0], n=int(len(g)), mean_visible=r1(g.visible_20km.mean(), 2),
                        share_seeing_any=r1((g.visible_20km > 0).mean(), 3), mean_share_visible=r1(g.share_visible_20km.mean(), 3)))
cty = sorted(cty, key=lambda d: -d["mean_visible"])
R["counties_top"] = cty[:8]; R["counties_bottom"] = cty[-8:]; R["n_counties_25plus"] = len(cty)

# attributes: who sees more?
x = f.copy()
x["multivallate"] = (x.ramparts >= 2).astype(float); x.loc[x.ramparts.isna(), "multivallate"] = np.nan
x["log2_area"] = np.log2(x.area_ha)
x["elev100"] = x.elev_m / 100
attr = {}
rho, p = spearmanr(x.area_ha, x.visible_20km, nan_policy="omit"); attr["spearman_area_visible"] = dict(rho=r1(rho, 3), p=float(p))
rho, p = spearmanr(x.area_ha, x.share_visible_20km, nan_policy="omit"); attr["spearman_area_share"] = dict(rho=r1(rho, 3), p=float(p))
rho, p = spearmanr(x.elev_m, x.visible_20km); attr["spearman_elev_visible"] = dict(rho=r1(rho, 3), p=float(p))
rho, p = spearmanr(x.local_pct, x.visible_20km); attr["spearman_localpct_visible"] = dict(rho=r1(rho, 3), p=float(p))
rho, p = spearmanr(x.forts_within_20km, x.visible_20km); attr["spearman_density_visible"] = dict(rho=r1(rho, 3), p=float(p))
grp = {}
for name, mask in [("univallate", x.ramparts == 1), ("multivallate", x.ramparts >= 2), ("coastal", x.coastal == 1), ("inland", x.coastal == 0)]:
    g = x[mask]
    grp[name] = dict(n=int(len(g)), mean_visible=r1(g.visible_20km.mean(), 2), median_visible=r1(g.visible_20km.median(), 1),
                     share_seeing_any=r1((g.visible_20km > 0).mean(), 3), mean_share_visible=r1(g.share_visible_20km.mean(), 3),
                     mean_in_range=r1(g.forts_within_20km.mean(), 1), median_elev=r1(g.elev_m.median(), 0))
q = pd.qcut(x.area_ha, 4)
grp["area_quartiles"] = [dict(range=f"{iv.left:.2f} to {iv.right:.2f} ha", n=int(len(g)), mean_visible=r1(g.visible_20km.mean(), 2),
                              share_seeing_any=r1((g.visible_20km > 0).mean(), 3), mean_share_visible=r1(g.share_visible_20km.mean(), 3))
                         for iv, g in x.groupby(q, observed=True)]
attr["groups"] = grp
mu = mannwhitneyu(x[x.ramparts >= 2].visible_20km, x[x.ramparts == 1].visible_20km); attr["mw_multi_vs_uni_p"] = float(mu.pvalue)
mu = mannwhitneyu(x[x.coastal == 1].visible_20km, x[x.coastal == 0].visible_20km); attr["mw_coastal_vs_inland_p"] = float(mu.pvalue)

# binomial GLM: visible out of forts within 20 km
d = x[(x.forts_within_20km > 0)].dropna(subset=["log2_area", "multivallate"]).copy()
X = pd.get_dummies(d[["log2_area", "multivallate", "coastal", "elev100", "local_pct", "country"]], columns=["country"], drop_first=True, dtype=float)
X = sm.add_constant(X)
y = np.c_[d.visible_20km, d.forts_within_20km - d.visible_20km]
glm = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC1")
ci = glm.conf_int()
def orow(k, label):
    return dict(term=k, label=label, odds_ratio=r1(np.exp(glm.params[k]), 3), lo=r1(np.exp(ci.loc[k, 0]), 3), hi=r1(np.exp(ci.loc[k, 1]), 3), p=float(glm.pvalues[k]))
attr["glm_share_visible_20km"] = dict(n=int(len(d)), note="Binomial GLM of visible forts out of forts within 20 km; robust (HC1) SEs; nation dummies included; pairs are not independent so intervals are optimistic.",
    rows=[orow("log2_area", "Twice the enclosed area"), orow("multivallate", "Multivallate (2+ ramparts)"), orow("coastal", "Coastal"),
          orow("elev100", "Every 100 m higher"), orow("local_pct", "Local height percentile (0 to 1)")])
# simple OLS on count with density control
X2 = sm.add_constant(pd.get_dummies(d[["log2_area", "multivallate", "coastal", "elev100", "local_pct", "forts_within_20km", "country"]], columns=["country"], drop_first=True, dtype=float))
ols = sm.OLS(d.visible_20km, X2).fit(cov_type="HC1")
attr["ols_visible_20km"] = dict(n=int(len(d)), r2=r1(ols.rsquared, 3), coef={k: dict(b=r1(ols.params[k], 3), lo=r1(ols.conf_int().loc[k, 0], 3), hi=r1(ols.conf_int().loc[k, 1], 3), p=float(ols.pvalues[k]))
                                                                    for k in ["log2_area", "multivallate", "coastal", "elev100", "local_pct", "forts_within_20km"]})
R["attributes"] = attr

# null model
nr = pd.read_csv(os.path.join(WORK, "null_reps.csv"))
nr["visible"] = nr.visible.astype(int)
nm = pd.read_csv(os.path.join(WORK, "null_match.csv"))
nulls = {}
for km in KMS:
    g = nr[nr.max_km == km]
    o = net[km]
    obs_vis, obs_share = o["pairs_visible"], o["pairs_visible"] / o["pairs_in_range"]
    nulls[str(km)] = dict(
        replicates=int(len(g)), observed_visible=obs_vis, null_visible_mean=r1(g.visible.mean(), 1), null_visible_sd=r1(g.visible.std(), 1),
        null_visible_min=int(g.visible.min()), null_visible_max=int(g.visible.max()), ratio=r1(obs_vis / g.visible.mean(), 3),
        z=r1((obs_vis - g.visible.mean()) / g.visible.std(), 1), n_null_ge_obs=int((g.visible >= obs_vis).sum()),
        empirical_p=r1(((g.visible >= obs_vis).sum() + 1) / (len(g) + 1), 4),
        observed_share=r1(obs_share, 4), null_share_mean=r1(g.share.mean(), 4), null_share_sd=r1(g.share.std(), 4),
        null_in_range_mean=r1(g.in_range.mean(), 0),
        observed_any=o["share_seeing_any"], null_any_mean=r1(g.any_visible.mean(), 4),
        observed_mean_deg=o["mean_visible"], null_mean_deg=r1(g.mean_deg.mean(), 2),
        observed_largest=o["largest_component"], null_largest_mean=r1(g.largest_comp.mean(), 0), null_largest_min=int(g.largest_comp.min()), null_largest_max=int(g.largest_comp.max()),
        cross_share_mean=r1(g.cross_share.mean(), 4), cross_share_sd=r1(g.cross_share.std(), 4),
        cross_share_min=r1(g.cross_share.min(), 4), cross_share_max=r1(g.cross_share.max(), 4),
        # site effect: how much more often a real fort sees a matched point than a matched point does
        site_factor=r1((g.cross_share / g.share).mean(), 3),
        # if each real site multiplies the chance of a sight line by the site factor, two real sites should give null x factor^2
        expected_if_sites_only=r1((g.cross_share ** 2 / g.share).mean(), 4),
        expected_if_sites_only_min=r1((g.cross_share ** 2 / g.share).min(), 4), expected_if_sites_only_max=r1((g.cross_share ** 2 / g.share).max(), 4),
        pair_factor=r1((obs_share / (g.cross_share ** 2 / g.share)).mean(), 3),
        pair_factor_min=r1((obs_share / (g.cross_share ** 2 / g.share)).min(), 3), pair_factor_max=r1((obs_share / (g.cross_share ** 2 / g.share)).max(), 3),
        excess_pct=r1(obs_vis / g.visible.mean() - 1, 3), pair_excess_pct=r1((obs_share / (g.cross_share ** 2 / g.share)).mean() - 1, 3), share_ratio=r1(obs_share / g.share.mean(), 3),
        cross_ratio=r1(g.cross_share.mean() / g.share.mean(), 3), obs_over_cross=r1(obs_share / g.cross_share.mean(), 3),
        null_values=[int(v) for v in g.visible], null_shares=[r1(v, 4) for v in g.share], null_largest=[int(v) for v in g.largest_comp])
R["null"] = nulls
R["null_matching"] = dict(
    candidates_per_fort=500, annulus_km=[1, 5], local_radius_km=2, pct_tolerance=0.05, elev_tolerance="max(50 m, 25% of fort elevation)",
    share_forts_matched=r1(nm.matched.mean(), 4), median_matched_candidates=int(nm.n_matched_candidates.median()),
    fort_mean_elev=r1(nm.fort_elev.mean(), 1), null_mean_elev=r1(nm.null_elev_mean.mean(), 1),
    fort_mean_pct=r1(nm.fort_pct.mean(), 3), null_mean_pct=r1(nm.null_pct_mean.mean(), 3),
    fort_median_rel_height=r1(f.rel_height_m.median(), 1))
pd.DataFrame(nr).to_csv(os.path.join(DATA, "null_model.csv"), index=False)

# sanity pairs
def pid(name):
    return int(f.index[f.name.str.startswith(name)][0])
checks = [("British Camp, Herefordshire", "Midsummer Hill, Herefordshire"), ("British Camp, Herefordshire", "Kemerton Camp, Bredon Hill"),
          ("Maiden Castle, Winterborne St Martin", "Eggardon, Dorset"), ("Hambledon Hill, Dorset", "Hod Hill, Dorset"),
          ("Pilsdon Pen, Dorset", "Lambert's Castle, Dorset"), ("Eildon Hill North", "Traprain Law"), ("Maiden Castle, Winterborne St Martin", "Badbury Rings")]
pk = {(int(a), int(b)): i for i, (a, b) in enumerate(pairs)}
sc = []
for a, b in checks:
    i, j = sorted((pid(a), pid(b)))
    k = pk.get((i, j))
    sc.append(dict(a=f.name[i], b=f.name[j], km=r1(haversine(lat[i], lon[i], lat[j], lon[j]) / 1000, 2),
                   visible_2m=None if k is None else bool(v2[k]), visible_5m=None if k is None else bool(v5[k]), visible_snapped=None if k is None else bool(vs[k])))
R["sanity_pairs"] = sc
R["examples"] = {k: dict(name=f.name[pid(v)], visible_20km=int(f.visible_20km[pid(v)]), forts_within_20km=int(f.forts_within_20km[pid(v)]))
                 for k, v in [("rubers", "Rubers Law"), ("yeavering", "Yeavering Bell Camp"), ("maiden", "Maiden Castle, Winterborne"), ("british_camp", "British Camp, Here")]}
R["runtime_s"] = dict(observed=float(open(os.path.join(WORK, "runtime_obs.txt")).read()), null=float(open(os.path.join(WORK, "runtime_null.txt")).read()))
R["params"] = dict(eye_height_m=2, eye_height_sensitivity_m=5, k_refraction=0.13, earth_radius_m=6371000, step_m=45, interpolation="bilinear",
                   dem="AWS Terrain Tiles terrarium, zoom 10 (about 90 m pixels)", distances_km=list(KMS), snap="highest DEM point within a circle of the fort's recorded area (radius 45 to 300 m)")

# ---- outputs
vis20 = {i: [] for i in range(n)}
for a, b in pairs[(dist <= 20000) & v2]:
    vis20[a].append(f.atlas_id[b]); vis20[b].append(f.atlas_id[a])
f["visible_ids_20km"] = [";".join(str(v) for v in sorted(vis20[i])) for i in range(n)]
cols = ["atlas_id", "name", "country", "county", "lat", "lon", "elev_m", "local_pct", "rel_height_m", "area_ha", "ramparts", "coastal",
        "forts_within_10km", "forts_within_20km", "forts_within_30km", "visible_10km", "visible_20km", "visible_30km", "visible_20km_5m", "visible_20km_snapped",
        "share_visible_20km", "component_20km", "component_size_20km", "visible_ids_20km"]
out = f[cols].copy()
for c in ["lat", "lon"]: out[c] = out[c].round(6)
for c in ["elev_m", "rel_height_m"]: out[c] = out[c].round(1)
out["local_pct"] = out.local_pct.round(3); out["share_visible_20km"] = out.share_visible_20km.round(3)
out.to_csv(os.path.join(DATA, "forts_visibility.csv"), index=False)
for km in KMS:
    m = (dist <= km * 1000) & v2
    e = pairs[m]
    pd.DataFrame(dict(atlas_id_a=f.atlas_id.values[e[:, 0]], atlas_id_b=f.atlas_id.values[e[:, 1]], name_a=f.name.values[e[:, 0]], name_b=f.name.values[e[:, 1]],
                      dist_km=(dist[m] / 1000).round(2), visible_5m=v5[m])).to_csv(os.path.join(DATA, f"edges_{km}km.csv"), index=False)
json.dump(R, open(os.path.join(DATA, "results.json"), "w"), indent=1, ensure_ascii=False)

# compact site data
counties = sorted(f.county.unique())
site = dict(
    counties=counties,
    f=[[round(r.lat, 4), round(r.lon, 4), r.name, counties.index(r.county), None if np.isnan(r.area_ha) else r.area_ha,
        None if np.isnan(r.ramparts) else int(r.ramparts), int(r.visible_10km), int(r.visible_20km), int(r.visible_30km), int(r.atlas_id), round(float(r.elev_m)), int(r.coastal)]
       for r in f.itertuples()],
    e={str(km): pairs[(dist <= km * 1000) & v2].ravel().tolist() for km in KMS},
    comp={str(km): f[f"component_{km}km"].tolist() for km in KMS},
    R=R)
json.dump(site, open(os.path.join(WORK, "site.json"), "w"), separators=(",", ":"), ensure_ascii=False)
print(json.dumps({k: R["network"][k]["pairs_visible"] for k in R["network"]}), "done")
