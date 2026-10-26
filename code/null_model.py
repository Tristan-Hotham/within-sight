"""Step 2: null model. Shift every fort to a random point 1 to 5 km away on
similar ground, then recompute visibility among the shifted set.

Similar ground = local elevation percentile within a 2 km radius within 0.05
of the fort's own value AND elevation within max(50 m, 25%) of the fort's.
Writes ../work/null_points.npz, ../work/null_reps.npz and ../work/forts_topo.csv.
Run: python3 null_model.py [n_replicates]
"""
import os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from dem import DEM
from visibility import pairs_within, visibility, WORK

N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_CAND = 500
R_LOCAL = 2000.0
GRID = 150.0
PCT_TOL = 0.05
SEED = 20261026

# offsets of a 150 m grid inside a 2 km circle
_g = np.arange(-R_LOCAL, R_LOCAL + 1, GRID)
_ox, _oy = np.meshgrid(_g, _g)
_in = (_ox ** 2 + _oy ** 2 <= R_LOCAL ** 2) & ~((_ox == 0) & (_oy == 0))
OX, OY = _ox[_in], _oy[_in]


def pix_size(lat):
    return 40075016.686 * np.cos(np.radians(lat)) / (1024 * 256)


def local_topo(dem, lat, lon, chunk=20000):
    """Local elevation percentile and relative height in a 2 km radius."""
    pct = np.empty(len(lat)); rel = np.empty(len(lat)); z = np.empty(len(lat))
    for s in range(0, len(lat), chunk):
        la, lo = lat[s:s + chunk], lon[s:s + chunk]
        gx, gy = dem.to_pix(la, lo); ps = pix_size(la)
        z0 = dem.bilinear(gx, gy)
        zz = dem.nearest(gx[:, None] + OX[None, :] / ps[:, None], gy[:, None] + OY[None, :] / ps[:, None])
        pct[s:s + chunk] = ((zz < z0[:, None]).sum(1) + 0.5 * (zz == z0[:, None]).sum(1)) / zz.shape[1]
        rel[s:s + chunk] = z0 - zz.mean(1)
        z[s:s + chunk] = z0
    return z, pct, rel


def _cross(dem, flat, flon, nlat, nlon, pairs):
    """Real fort i looking at the null point of fort j, and null i at real j."""
    from dem import los, haversine
    fz = dem.elev(flat, flon).astype(np.float64); nz = dem.elev(nlat, nlon).astype(np.float64)
    fgx, fgy = dem.to_pix(flat, flon); ngx, ngy = dem.to_pix(nlat, nlon)
    res = []
    for a, b, (axs, ays, azs, bxs, bys, bzs, la1, lo1, la2, lo2) in [
            (pairs[:, 0], pairs[:, 1], (fgx, fgy, fz, ngx, ngy, nz, flat, flon, nlat, nlon)),
            (pairs[:, 0], pairs[:, 1], (ngx, ngy, nz, fgx, fgy, fz, nlat, nlon, flat, flon))]:
        d = haversine(la1[a], lo1[a], la2[b], lo2[b])
        m = d <= 30000
        v = los(dem.bilinear, axs[a][m], ays[a][m], azs[a][m], bxs[b][m], bys[b][m], bzs[b][m], d[m], 2.0, 2.0)
        res.append((d[m], v))
    d = np.concatenate([r[0] for r in res]); v = np.concatenate([r[1] for r in res])
    return {km: dict(cross_in_range=int((d <= km * 1000).sum()) / 2, cross_visible=int(((d <= km * 1000) & v).sum()) / 2,
                     cross_share=float(((d <= km * 1000) & v).sum() / (d <= km * 1000).sum())) for km in (10, 20, 30)}


def _rep(args):
    r, lat, lon, flat, flon, opairs = args
    dem = DEM()
    z = dem.elev(lat, lon).astype(np.float64)
    pairs, dist = pairs_within(lat, lon)
    v = visibility(dem, lat, lon, z, pairs, dist, 2.0)
    cross = _cross(dem, flat, flon, lat, lon, opairs)
    out = {}
    nF = len(lat)
    for km in (10, 20, 30):
        m = dist <= km * 1000
        deg = np.bincount(pairs[m & v].ravel(), minlength=nF)
        out[km] = dict(in_range=int(m.sum()), visible=int((m & v).sum()), share=float((m & v).sum() / m.sum()),
                       any_visible=float((deg > 0).mean()), mean_deg=float(deg.mean()),
                       largest_comp=int(largest_component(pairs[m & v], nF)), **cross[km])
    import json
    json.dump(out, open(os.path.join(WORK, f"rep_{r:02d}.json"), "w"))
    print("replicate", r + 1, "done", flush=True)
    return r, out


def largest_component(edges, n):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return np.bincount(lab).max()


if __name__ == "__main__":
    t0 = time.time()
    f = pd.read_csv(os.path.join(WORK, "forts_base.csv"))
    dem = DEM()
    lat, lon = f.lat.values, f.lon.values
    z, pct, rel = local_topo(dem, lat, lon)
    f["local_pct"] = pct; f["rel_height_m"] = rel
    f.to_csv(os.path.join(WORK, "forts_topo.csv"), index=False)
    print("fort topo", time.time() - t0)

    rng = np.random.default_rng(SEED)
    n = len(f)
    # candidate points: uniform by area in a 1 to 5 km annulus
    rr = np.sqrt(rng.uniform(1000 ** 2, 5000 ** 2, (n, N_CAND)))
    th = rng.uniform(0, 2 * np.pi, (n, N_CAND))
    dlat = rr * np.cos(th) / 111195.0
    dlon = rr * np.sin(th) / (111195.0 * np.cos(np.radians(lat)))[:, None]
    clat = lat[:, None] + dlat; clon = lon[:, None] + dlon
    cz, cp, _ = local_topo(dem, clat.ravel(), clon.ravel())
    cz = cz.reshape(n, N_CAND); cp = cp.reshape(n, N_CAND)
    print("candidate topo", time.time() - t0)
    ztol = np.maximum(50.0, 0.25 * np.abs(z))
    ok = (cz > 0) & (np.abs(cp - pct[:, None]) <= PCT_TOL) & (np.abs(cz - z[:, None]) <= ztol[:, None])
    n_ok = ok.sum(1)
    # fallback score when too few acceptable candidates: closest in percentile and elevation, land only
    score = np.abs(cp - pct[:, None]) / PCT_TOL + np.abs(cz - z[:, None]) / ztol[:, None] + np.where(cz > 0, 0, 1e6)
    pick_lat = np.empty((N_REP, n)); pick_lon = np.empty((N_REP, n)); exact = np.zeros((N_REP, n), bool)
    for i in range(n):
        idx = np.flatnonzero(ok[i])
        if len(idx) >= N_REP:
            ch = rng.choice(idx, N_REP, replace=False); exact[:, i] = True
        elif len(idx) > 0:
            ch = rng.choice(idx, N_REP, replace=True); exact[:, i] = True
        else:
            best = np.argsort(score[i])[:N_REP]
            ch = rng.choice(best, N_REP, replace=True)
        pick_lat[:, i] = clat[i, ch]; pick_lon[:, i] = clon[i, ch]
    np.savez_compressed(os.path.join(WORK, "null_points.npz"), lat=pick_lat, lon=pick_lon, exact=exact, n_ok=n_ok,
                        cand_pct_abs_diff=np.abs(cp - pct[:, None]).min(1))
    print("forts with >=1 matched candidate", (n_ok > 0).mean(), "with >= N_REP", (n_ok >= N_REP).mean())
    zn = dem.elev(pick_lat.ravel(), pick_lon.ravel()).reshape(N_REP, n)
    _, pn, _ = local_topo(dem, pick_lat.ravel(), pick_lon.ravel()); pn = pn.reshape(N_REP, n)
    pd.DataFrame(dict(atlas_id=f.atlas_id, fort_elev=z, null_elev_mean=zn.mean(0), fort_pct=pct, null_pct_mean=pn.mean(0),
                      n_matched_candidates=n_ok, matched=exact.all(0))).to_csv(os.path.join(WORK, "null_match.csv"), index=False)
    opairs = np.load(os.path.join(WORK, "pairs_obs.npz"))["pairs"].astype(np.int64)
    with Pool(2, maxtasksperchild=1) as pool:
        res = dict(pool.imap_unordered(_rep, [(r, pick_lat[r], pick_lon[r], lat, lon, opairs) for r in range(N_REP)]))
    rows = []
    for r in range(N_REP):
        for km in (10, 20, 30):
            rows.append(dict(replicate=r + 1, max_km=km, **res[r][km]))
    pd.DataFrame(rows).to_csv(os.path.join(WORK, "null_reps.csv"), index=False)
    open(os.path.join(WORK, "runtime_null.txt"), "w").write(f"{time.time() - t0:.1f}\n")
    print("done", time.time() - t0)
