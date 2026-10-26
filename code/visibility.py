"""Step 1: line of sight between every pair of Atlas hillforts within 30 km.

Writes ../work/pairs_obs.npz (all pairs within 30 km with distance and visibility
at 2 m and 5 m eye heights) and ../work/forts_base.csv.
Run: python3 visibility.py
"""
import os, time
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from dem import DEM, los, haversine, R_EARTH

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("FORTS_CSV", "/home/claude/hist/famous-few/data/hillforts_attention.csv")
WORK = os.path.join(HERE, "..", "work"); os.makedirs(WORK, exist_ok=True)
MAXD = 30000.0


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)] * R_EARTH


def pairs_within(lat, lon, maxd=MAXD):
    t = cKDTree(xyz(lat, lon))
    p = t.query_pairs(maxd * 1.001, output_type="ndarray")
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    d = haversine(lat[p[:, 0]], lon[p[:, 0]], lat[p[:, 1]], lon[p[:, 1]])
    keep = d <= maxd
    return p[keep], d[keep]


def visibility(dem, lat, lon, z, pairs, dist, h):
    gx, gy = dem.to_pix(lat, lon)
    a, b = pairs[:, 0], pairs[:, 1]
    return los(dem.bilinear, gx[a], gy[a], z[a], gx[b], gy[b], z[b], dist, h, h)


def snap_to_high(dem, lat, lon, area_ha, rmin=45.0, rmax=300.0, step=15.0):
    """Highest DEM point within r of each fort, r = radius of a circle of the
    fort's recorded area, clipped to rmin..rmax (rmin when area is missing)."""
    r = np.sqrt(np.nan_to_num(area_ha, nan=0.0) * 10000 / np.pi)
    r = np.clip(r, rmin, rmax)
    g = np.arange(-rmax, rmax + 1, step)
    ox, oy = [a.ravel() for a in np.meshgrid(g, g)]
    gx, gy = dem.to_pix(lat, lon)
    ps = 40075016.686 * np.cos(np.radians(lat)) / (1024 * 256)
    X = gx[:, None] + ox[None] / ps[:, None]; Y = gy[:, None] + oy[None] / ps[:, None]
    zz = dem.bilinear(X, Y)
    zz[(ox[None] ** 2 + oy[None] ** 2) > r[:, None] ** 2] = -1e9
    k = zz.argmax(1); i = np.arange(len(lat))
    sl, so = dem.from_pix(X[i, k], Y[i, k])
    return sl, so, zz[i, k].astype(np.float64)


if __name__ == "__main__":
    t0 = time.time()
    f = pd.read_csv(SRC)
    cols = ["atlas_id", "name", "country", "county", "lat", "lon", "area_ha", "ramparts", "scheduled", "n_excav", "dated", "roman", "medieval", "coastal"]
    f = f[cols].copy()
    dem = DEM()
    f["elev_m"] = dem.elev(f.lat.values, f.lon.values)
    lat, lon, z = f.lat.values, f.lon.values, f.elev_m.values.astype(np.float64)
    pairs, dist = pairs_within(lat, lon)
    print(len(pairs), "pairs within 30 km", time.time() - t0)
    v2 = visibility(dem, lat, lon, z, pairs, dist, 2.0); print("2 m done", time.time() - t0)
    v5 = visibility(dem, lat, lon, z, pairs, dist, 5.0); print("5 m done", time.time() - t0)
    # sensitivity: move each fort to the highest ground inside a circle the size of its enclosure
    slat, slon, sz = snap_to_high(dem, lat, lon, f.area_ha.values)
    f["snap_lat"], f["snap_lon"], f["snap_elev_m"] = slat, slon, sz
    sd = haversine(slat[pairs[:, 0]], slon[pairs[:, 0]], slat[pairs[:, 1]], slon[pairs[:, 1]])
    vs = visibility(dem, slat, slon, sz, pairs, sd, 2.0); print("snap done", time.time() - t0)
    np.savez_compressed(os.path.join(WORK, "pairs_obs.npz"), pairs=pairs.astype(np.int32), dist=dist.astype(np.float32), v2=v2, v5=v5, vs=vs)
    f.to_csv(os.path.join(WORK, "forts_base.csv"), index=False)
    for km in (10, 20, 30):
        m = dist <= km * 1000
        print(km, "km: pairs in range", m.sum(), "visible 2 m", v2[m].sum(), "5 m", v5[m].sum(), "snapped", vs[m].sum())
    open(os.path.join(WORK, "runtime_obs.txt"), "w").write(f"{time.time() - t0:.1f}\n")
    print("total", time.time() - t0)
