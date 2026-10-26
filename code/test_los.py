"""Synthetic checks of the line-of-sight function (run: python3 test_los.py)."""
import numpy as np
from dem import los, R_EARTH, K_REFRACTION

# coordinates in "pixels" of 1 m for the synthetic tests: gx = metres along a line
def run(terrain, d, zA, zB, hA=2.0, hB=2.0):
    one = lambda v: np.array([v], dtype=float)
    return bool(los(terrain, one(0), one(0), one(zA), one(d), one(0), one(zB), one(d), hA, hB)[0])

flat = lambda gx, gy: np.zeros_like(gx)
def ridge(h, at, width=200):
    return lambda gx, gy: np.where(np.abs(gx - at) < width / 2, h, 0.0)

Reff = R_EARTH / (1 - K_REFRACTION)
# horizon distance for two 2 m eyes on flat ground: 2 * sqrt(2 * Reff * 2)
hz = 2 * np.sqrt(2 * Reff * 2.0)
res = {
    "flat 3 km, 2 m eyes visible": run(flat, 3000, 0, 0) is True,
    f"flat beyond horizon ({hz/1000:.1f} km) hidden": run(flat, hz * 1.1, 0, 0) is False,
    "flat just inside horizon visible": run(flat, hz * 0.9, 0, 0) is True,
    "50 m ridge between two 0 m points blocks": run(ridge(50, 5000), 10000, 0, 0) is False,
    "50 m ridge between two 100 m hills does not block": run(ridge(50, 5000), 10000, 100, 100) is True,
    # earth bulge at the middle of a 10 km line is 5000^2 / (2 R') = 1.7 m
    "100 m ridge between two 100 m hills (eyes 102 m) clears by the bulge margin": run(ridge(100, 5000), 10000, 100, 100) is True,
    "101 m ridge blocks once curvature is added": run(ridge(101, 5000), 10000, 100, 100) is False,
    "1 m bump does not block 2 m eyes over 2 km": run(ridge(1, 1000), 2000, 0, 0) is True,
    "1 m bump blocks over 10 km (curvature)": run(ridge(1, 5000), 10000, 0, 0) is False,
    "ridge near observer blocks": run(ridge(30, 300), 20000, 20, 200) is False,
}
for k, v in res.items():
    print(("PASS " if v else "FAIL ") + k)
assert all(res.values())
