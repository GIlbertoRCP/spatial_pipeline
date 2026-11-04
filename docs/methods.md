# How it works

## Counting photographers, not photos

One person who uploads 500 shots from their backyard would otherwise create a hot spot all by
themselves. Every count is the number of **distinct photographers**: per hexagon, per (hexagon,
subject), per month and per hour.

## Hexagons on an equal-area projection

Coordinates are projected from WGS84 to CONUS Albers Equal Area (EPSG:5070) before binning, so
every hexagon covers the same ground area and distances are in meters. Hexagons are pointy-topped,
with a configurable center-to-center spacing (`--resolution`, default 1,000 m).

## Which subjects cluster: Moran's *I*

For each subject, Moran's *I* measures whether hexagons with a high share of that subject sit next
to other high-share hexagons. Shares, rather than raw counts, are used by default
(`--normalize share`), so a subject doesn't rank high just because it appears where many people take
photos. Moran's *I* comes with an analytical z-score, and p-values are corrected across subjects
with Benjamini-Hochberg FDR.

Hexagons without any neighbor are left out of the statistics (they still appear on the map). With
binary weights they would inflate *I*: they add to the number of hexagons but not to the neighbor
pairs.

## Where they cluster: Getis-Ord *Gi\**

Moran's *I* says *whether* a subject clusters. Local Getis-Ord *Gi\** says *where*: for every hexagon
it compares the sum over the hexagon and its neighbors with what spatial randomness would give,
expressed as a z-score. With thousands of hexagons there are thousands of tests, so p-values are
FDR-corrected **separately for each subject**. A hexagon is a hot spot when *z* > 0 and the
adjusted p-value is below `--alpha` (default 0.05).

Because the neighborhood is part of the test, a hexagon next to a cluster can be significant even
if it has no photos of that subject itself. The map only draws hexagons that contain the subject.

Comparisons are among photographed hexagons only, so "hot" means hot relative to other places
people photograph, not relative to the empty land in between.

## When: month and hour profiles with lift

For every subject, `timing.compute_timing` counts distinct photographers per month and per hour.
Each profile is divided by the profile of all photos to get the **lift**. People are outdoors
more in summer and at midday, so lift is what separates a real seasonal pattern (elk in the
September rut) from that general bias.

A time of exactly 00:00:00 is treated as "date only" (iNaturalist observations without a time, or
cameras without a clock). Those photos count toward months but not hours.

## Verification

- Gi\* is checked against a direct, one-hexagon-at-a-time implementation of the textbook formula
  (Ord & Getis, 1995).
- FDR adjustment is checked against `scipy.stats.false_discovery_control`.
- The hexagon geometry is checked: neighbors are exactly one resolution apart, and drawn outlines
  match the bins.
- The data sources are tested against simulated APIs that reproduce the real paging limits,
  plus a live check of the iNaturalist fetcher on Denver-area data.
