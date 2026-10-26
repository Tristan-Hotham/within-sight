# Within Sight

Could Britain and Ireland's 4,147 hillforts see each other? Line of sight between every pair of hillforts within 30 km, the networks and relay chains they form, and a test of whether forts see each other more than matched ground nearby does.

**Interactive version:** https://tristan-hotham.github.io/within-sight/

This repository holds the interactive page, the per-fort and per-pair results and the analysis code for the data study *Within Sight: Could Britain's Hillforts See Each Other?* by Dr Tristan Hotham (Independent researcher, ORCID 0000-0003-4931-9349). Companion study: [The Famous Few](https://tristan-hotham.github.io/famous-few/).

## Headline results

- Within 20 km, {{network.20.pairs_visible|int}} of {{network.20.pairs_in_range|int}} pairs of hillforts ({{network.20.share_visible|pct1}}) are in line of sight; {{network.10.pairs_visible|int}} pairs within 10 km and {{network.30.pairs_visible|int}} within 30 km.
- {{network.20.share_seeing_any|pct0}} of hillforts can see at least one other within 20 km. The largest linked network has {{network.20.largest_component|int}} forts.
- Real forts have {{null.20.excess_pct|pct0}} more intervisible pairs within 20 km than twenty sets of forts shifted 1 to 5 km onto matched ground ({{null.20.observed_visible|int}} against a mean of {{null.20.null_visible_mean|int}}).
- A mixed test (real fort to shifted neighbour) suggests the excess is explained by forts sitting in spots with wider views in general: the expected share from that effect alone is {{null.20.expected_if_sites_only|pct1}}, against {{null.20.observed_share|pct1}} observed.

## Files

- `index.html`: the interactive page (Leaflet map, charts).
- `header.png`: social card.
- `data/forts_visibility.csv`: one row per Atlas hillfort with elevation, local height rank, number of forts within and in sight at 10, 20 and 30 km, sensitivity counts (5 m eyes; moved to highest ground), network membership at 20 km and the Atlas ids of the forts visible within 20 km.
- `data/edges_10km.csv`, `data/edges_20km.csv`, `data/edges_30km.csv`: every intervisible pair (2 m eyes), with distance and whether the pair is also visible with 5 m eyes.
- `data/null_model.csv`: results of each of the twenty shifted maps, and the mixed test.
- `data/results.json`: every number quoted on the page.
- `code/`: Python scripts. Run in order: `visibility.py`, `null_model.py`, `analyse.py`, `build_site.py`, `make_header.py`. `dem.py` holds the elevation lookup and line-of-sight function; `test_los.py` checks it on synthetic ground. The elevation tiles are not included (see below).

## Method in brief

Line of sight from 2 m above the ground at each fort's recorded point to 2 m above the ground at the other, sampling the ground every 45 m with bilinear interpolation, with earth curvature and standard refraction (k = 0.13). Sensitivity runs use 5 m eyes, and each fort moved to the highest ground within a circle of its recorded area (45 to 300 m radius). The null model moves each fort to a random point 1 to 5 km away that is on land, within 0.05 of the fort's height rank within 2 km, and within 50 m or 25% of its elevation.

Caveats: a line of sight is not the same as being able to see a fort; the 90 m elevation model smooths summits and has no vegetation; Atlas points are not summits or ramparts; and the forts were not all in use at the same time.

## Sources and licence

- Site data: Lock, G. and Ralston, I. 2017. *Atlas of Hillforts of Britain and Ireland*. https://hillforts.arch.ox.ac.uk (CC BY-SA 4.0), via the Atlas source data released with Middleton, M. 2024, *The Hillforts Primer* (https://github.com/MikeDairsie/hillforts-primer).
- Elevation: AWS Terrain Tiles by Mapzen/Tilezen (terrarium encoding, zoom 10), built from sources including SRTM, GMTED2010, ETOPO1 and national datasets. Full attribution: https://github.com/tilezen/joerd/blob/master/docs/attribution.md
- Base map tiles on the page: Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS user community. Map library: Leaflet.

Because the Atlas data are CC BY-SA 4.0, the derived data and the page text are released under CC BY-SA 4.0. Code is MIT (see `LICENSE`).

I wrote the text, with drafting and editing assistance from Claude (Anthropic), which also helped prepare the analysis code. I reviewed and checked all text and results.
