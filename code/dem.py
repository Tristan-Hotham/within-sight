"""Elevation lookup and line-of-sight tools for the Within Sight analysis.

DEM: AWS Terrain Tiles (Mapzen/Tilezen terrarium), web-mercator zoom 10,
256x256 Int16 tiles concatenated in the order of index["tiles"]. Missing
tiles are sea and read as 0 m.
"""
import json, math, os
import numpy as np

DEM_BIN = os.environ.get("DEM_BIN", "/mnt/user-data/uploads/Downloads/hist_dem_z10.bin")
DEM_IDX = os.environ.get("DEM_IDX", "/mnt/user-data/uploads/Downloads/hist_dem_z10_index.json")
Z = 10
N = 2 ** Z
R_EARTH = 6371000.0
K_REFRACTION = 0.13
STEP_M = 45.0


class DEM:
    def __init__(self, bin_path=DEM_BIN, idx_path=DEM_IDX):
        idx = json.load(open(idx_path))
        keys = idx["tiles"]
        self.flat = np.fromfile(bin_path, dtype="<i2").astype(np.float32)
        xs = np.array([int(k.split("_")[0]) for k in keys]); ys = np.array([int(k.split("_")[1]) for k in keys])
        self.x0, self.y0 = xs.min(), ys.min()
        self.lut = -np.ones((ys.max() - self.y0 + 1, xs.max() - self.x0 + 1), dtype=np.int64)
        self.lut[ys - self.y0, xs - self.x0] = np.arange(len(keys))
        # append one zero tile used for sea / out of range
        self.flat = np.concatenate([self.flat, np.zeros(65536, np.float32)])
        self.sea = len(keys)

    @staticmethod
    def to_pix(lat, lon):
        """Global pixel coordinates (float) at zoom 10, 256 px per tile."""
        lat = np.radians(np.asarray(lat, dtype=np.float64)); lon = np.asarray(lon, dtype=np.float64)
        fx = (lon + 180.0) / 360.0 * N
        fy = (1.0 - np.log(np.tan(lat) + 1.0 / np.cos(lat)) / math.pi) / 2.0 * N
        return fx * 256.0, fy * 256.0

    @staticmethod
    def from_pix(gx, gy):
        fx = np.asarray(gx) / 256.0; fy = np.asarray(gy) / 256.0
        lon = fx / N * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * fy / N))))
        return lat, lon

    def _get(self, ix, iy):
        tx = ix >> 8; ty = iy >> 8
        r = ty - self.y0; c = tx - self.x0
        ok = (r >= 0) & (r < self.lut.shape[0]) & (c >= 0) & (c < self.lut.shape[1])
        t = np.full(ix.shape, self.sea, dtype=np.int64)
        t[ok] = self.lut[r[ok], c[ok]]
        t[t < 0] = self.sea
        return self.flat[t * 65536 + (iy & 255) * 256 + (ix & 255)]

    def nearest(self, gx, gy):
        return self._get(np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64))

    def bilinear(self, gx, gy):
        # pixel centres sit at integer + 0.5
        x = np.asarray(gx) - 0.5; y = np.asarray(gy) - 0.5
        x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
        fx = (x - x0).astype(np.float32); fy = (y - y0).astype(np.float32)
        a = self._get(x0, y0); b = self._get(x0 + 1, y0); c = self._get(x0, y0 + 1); d = self._get(x0 + 1, y0 + 1)
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

    def elev(self, lat, lon):
        gx, gy = self.to_pix(lat, lon)
        return self.bilinear(gx, gy)


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1; dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def los(terrain, gxA, gyA, zA, gxB, gyB, zB, dist, hA=2.0, hB=2.0, k=K_REFRACTION, step=STEP_M,
        chunk_samples=4_000_000):
    """Line of sight for many pairs at once.

    terrain(gx, gy) -> ground height. Observer eye at zA + hA, target at zB + hB.
    A terrain sample at distance x from A blocks the line if
      z(x) + x (d - x) / (2 R') >= eye height on the straight line at x,
    with R' = R / (1 - k) (curvature with standard refraction).
    Samples are taken every `step` metres, excluding the two end points.
    Returns a boolean array: True = visible.
    """
    Reff = R_EARTH / (1.0 - k)
    n = np.maximum(2, np.ceil(dist / step).astype(np.int64))  # segments
    m = n - 1  # interior samples per pair
    out = np.ones(len(dist), dtype=bool)
    starts = np.concatenate([[0], np.cumsum(m)])
    i = 0
    while i < len(dist):
        j = int(np.searchsorted(starts, starts[i] + chunk_samples, side="right")) - 1
        j = max(j, i + 1); j = min(j, len(dist))
        mm = m[i:j]; tot = int(mm.sum())
        pid = np.repeat(np.arange(j - i), mm)
        offs = np.concatenate([[0], np.cumsum(mm)[:-1]])
        kk = np.arange(tot) - np.repeat(offs, mm) + 1
        t = kk / np.repeat(n[i:j], mm)
        ax, ay, bx, by = gxA[i:j][pid], gyA[i:j][pid], gxB[i:j][pid], gyB[i:j][pid]
        z = terrain(ax + (bx - ax) * t, ay + (by - ay) * t)
        d = dist[i:j][pid]; x = t * d
        eyeA = (zA[i:j] + hA)[pid]; eyeB = (zB[i:j] + hB)[pid]
        line = eyeA + (eyeB - eyeA) * t
        clear = line - (z + x * (d - x) / (2 * Reff))
        mins = np.minimum.reduceat(clear, offs)
        out[i:j] = mins > 0
        i = j
    return out
