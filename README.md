# zero_levels

`zero_levels` contains utilities to estimate and remove monopole and dipole zero levels in HEALPix maps, plus simple helpers to store iterative fit results.

## Layout

- `src/zero_levels/monodip.py`: monopole, dipole, and TT-plot fitting tools
- `src/zero_levels/IO.py`: SQLite-based storage helpers for iterative results
- `tests/`: basic regression scripts

## Installation

```bash
pip install -e .
```

For test dependencies:

```bash
pip install -e .[test]
```

## Usage

```python
import numpy as np
from zero_levels import TTplots

nside = 256
maps = np.vstack([map_1, map_2, map_3])

tt = TTplots(nside=nside, nside_cluster=16)
mono_dipole = tt.calculate_mono_dipole(maps)
corrected_maps = tt.dep_remove_mono_dipole(maps, mono_dipole)
```

If one map has a fixed monopole or dipole, pass `fixed_pars`, for example `fixed_pars={1: "mono"}`.

## Tests

```bash
pytest
```

## License

This project is licensed under the GNU General Public License v3.0. See `LICENSE`.
