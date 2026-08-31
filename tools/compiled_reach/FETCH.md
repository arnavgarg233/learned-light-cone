# Optional compiled-reach utility

This utility computes a released weather emulator's compiled direct-step causal reach from its
ONNX graph. It is not part of the default replay and requires a checkpoint that this repository
does not redistribute.

```bash
python tools/compiled_reach/cone_reach.py \
  --checkpoint /path/to/pangu_weather_24.onnx \
  --step-hours 24 \
  --record results/tables/deployed/compiled_reach_pangu.json
```

Expected output on `pangu_weather_24.onnx` at the Pacific source site:

```
  compiled causal reach of a released checkpoint, zero forward passes
  --------------------------------------------------------------------
  checkpoint              pangu_weather_24.onnx
  graph nodes             7,717
  source site             pacific (0 lat, 200 lon)

  compiled reach          18,903.137530 km  (170.0000 deg)
  antipodal distance      20,015.086796 km
  compiled support        88.43% of the globe by area  (839,808 of 1,038,240 cells)
  certified-zero area     11.57% of the globe by area
  physical cone           100.00% of the globe by area  (25,920 km at 300 m/s over 24 h)

  model forward passes    0
  wall clock              2.49 s  (parse 0.05 s, compile 2.41 s)

  retained record         matches all 5 compared fields
                          results/tables/deployed/compiled_reach_pangu.json
```

The output is compared field by field with the retained record, and a mismatch exits nonzero.
The graph compilation itself takes about 2 s on CPU, uses NumPy, and performs no model forward
passes.

## What the number means

`reach.py` is an abstract interpreter over the released ONNX operator sequence. Each tensor
carries a concrete value where one is derivable from graph constants, initializers and shapes,
and a broadcast-compressed boolean dependency array otherwise. `MatMul` and `Conv` treat every
weight entry as structurally nonzero, so the output depends on the whole contraction and the
whole receptive field. Elementwise operators disjoin their operands, `Softmax` and `ReduceMean`
disjoin along their axis, and index operators are applied exactly.

The compiled set is therefore an over-approximation of the true dependency set, and the radius
it returns is an upper bound: the largest distance the operator sequence can carry any
influence at all, whatever the trained values are. It reads the graph, not the trained values,
which is why it costs seconds instead of forward passes.

The 24-hour physical radius at 300 m/s is 25,920 km, beyond the antipodal distance, so that cone
covers the sphere. The nonempty complement of the compiled upper bound is therefore a
certified-zero region inside the physical cone.

## Sites

`SITE=pacific` (default), `north_atlantic`, `southern_ocean`, `siberia`. The certified-zero area
runs from 11.57 percent to 43.45 percent of the globe across the four, and all four are in the
retained record.

## Vendored modules

`reach.py` and `onnx_skim.py` are vendored unmodified from the research tree that produced the
retained record, and their digests are pinned by `tests/test_compiled_reach_path.py`:

| File | Lines | SHA-256 |
|---|---:|---|
| `reach.py` | 737 | `977cab35940c9f9fe4ac77d315e5ce88795f3c21f539b8449ac4f4a5ad4f5c5e` |
| `onnx_skim.py` | 346 | `b25d0138f618ec63a9b606ef53170e02d8358876ad615eb21a5caad8df6bcb0f` |

`cone_reach.py` is the driver: it resolves the site, runs the interpreter, prints the
comparison, checks the retained record and writes its own record to `.cache/`.

## Fetch the checkpoint

The checkpoint is not redistributed here. Pangu-Weather's released ONNX graphs are
CC BY-NC-SA 4.0 and the release forbids commercial use, so fetch the file from the official
release yourself:

- Release: `https://github.com/198808xc/Pangu-Weather` at commit
  `72bdd99096721e1a1f8912c37a9a3aff9ff0a4f2`
- `pangu_weather_24.onnx`, 1181711187 bytes,
  SHA-256 `613a5c140a1399abcaffb4dbce32af373a1f5f56c515704f5be61925bb9fdcfd`
- Download link: the official Google Drive folder linked by that release's README
- Licence: `https://creativecommons.org/licenses/by-nc-sa/4.0/`

Google Drive serves files above 100 MB behind a confirmation page, so the download is a
browser step rather than a scripted one. Verify the byte count and SHA-256 above before running
the utility.

The release also publishes 1 h, 3 h and 6 h graphs. The retained certificate uses the direct
24 h graph. To compile another lead without comparing it with the retained record:

```bash
python tools/compiled_reach/cone_reach.py --checkpoint <file> --record ""
```
