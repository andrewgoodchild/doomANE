# splatDoom moved

The splat and textured-plane reconstruction (2026-11-08/09) is now its own private repo,
[doomsplat](https://github.com/andrewgoodchild/doomsplat), with its scripts, results and notes. Its data stays on
`/Volumes/Scratch/ane/poc-data/splat`.

What it found that matters here, the camera calibration (fitted against MAP01's WAD):
- depth buffer, walls and ceilings: forward distance = 7.1386 x value + 14.297 map units (not 7.6 x value);
- vertical focal length 195 at 320x200 (390 at 640x400), not 132; every floor placed by row geometry was at 68 % of
  its distance. `doom/seenmap/mapper.py` uses both.
