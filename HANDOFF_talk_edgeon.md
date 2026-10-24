# Handoff: LGR4 edge-on talk figures/movies

Status as of 2026-10-24. The work was done on StellarAI (`stellarai-vis1`). The full run is
to be done on Stellar.

## Goal

Make talk figures and movies for the two LGR4 models. Each frame is an edge-on (x-z) view
from z = -500 pc to the top of the box (6.14 kpc).

| model key | directory (under `/projects/EOSTRIKE/TIGRESSPP-CR/`) | snapshots |
|---|---|---|
| `mhd` | `midtall_LGR4_mhd_duale-8pc-b1-mhdbc_diode-fgas_0.90-r200-t2.5-icpx` | 101–600 (500) |
| `crmhd` | `6kpc_LGR4_crmhd_duale-8pc-b1-mhdbc_diode-crbc_lngrad_out-fgas_0.90-r200-t2.5-icpx` | 101–700 (600) |

Each snapshot produces three figures, all with the same fixed frame size (1470×2815 px):

- `slc_NNNN.png`: y=0 slices of n_H, T, v_z. These read the existing `allslc.y/` caches, so
  they are fast.
- `prj_NNNN.png`: Σ_gas, outflow mass flux, inflow mass flux (all gas).
- `flx_NNNN.png`: outflow mass flux for the wc, int and hot phases.

Output goes to `<outdir>/LGR4_{mhd,crmhd}/`. The movies are
`<outdir>/LGR4_{model}_{slc,prj,flx}.mp4`. The default `outdir` is
`/scratch/gpfs/EOST/changgoo/talk_figures/`, which is StellarAI scratch, so pass `--outdir`
on Stellar.

## Definitions

- **Mass flux** is the y-average of ρv_z (`sum * dx / Ly`), in M☉ kpc⁻² yr⁻¹. It is not a
  column integral. Σ is the only field that is a true column (M☉ pc⁻²).
- **Outflow** keeps cells with v_z·z > 0, and **inflow** keeps cells with v_z·z < 0, so
  `f_out + f_in = f`. The figures plot |f_out| and |f_in| on LogNorm(1e-5, 1e-1). Zero
  flux is drawn in the colormap's bottom color (black). The sharp line at z = 0 is real,
  because "outward" flips sign there.
- **Phases** do not overlap:
  - `wc`: T ≤ 2e4 K
  - `int`: 2e4 < T ≤ 5e5 K
  - `hot`: T > 5e5 K
  - `whole`: all gas

  The cut values are stored in the file attribute `Tcuts`.

## Code changes (this commit)

- `python/plot_talk_edgeon.py` (new) is the driver.
  `get_prj_split()` loads `prj.y`. If the cache is in the old format (no `mflux_out` or no
  `int` phase), it rebuilds it with `force_override=True`, which **overwrites
  `prj.y/prj.y.NNNNN.nc` in place**. This is intended: there is no separate folder.
- `python/slc_prj.py::get_prj` now:
  - adds `_out`/`_in` versions of every flux field (`mflux`, `teflux`, `keflux`, `mZflux`,
    `creflux*`). Σ and EM are not split.
  - uses the four non-overlapping phases above.
  - **Behavior change:** `hot` used to mean T > 2e4 K and now means T > 5e5 K.
    `plot_slices.plot_projections` (which plots `wc`/`hot`) shows the new meaning for any
    cache that is rebuilt. Old caches keep the old meaning until they are rebuilt.
- `python/load_sim_tigresspp.py`: bug fix. When `cool_ftn.runtime.csv` was missing (it is
  missing in both LGR4 directories), `__init__` hit a `return`. That skipped:
  - the cooling table, so μ fell back to a constant 1.27 and T was about 2× too high in
    ionized gas;
  - pop_synth, `nghost`, shear, extgrav, `update_derived_fields()` and `cpp_to_cc`.

  The loader now falls back to `<coolftn_file>` (`tigress_coolftn.txt`, columns
  `logT_mu mu cool heat`), and `__init__` always runs to the end. With the fix, the new
  temperatures match the cached `allslc.y` temperatures to about 1%.

Checks on snapshot 600, both models:

- `out + in = net` holds to about 1e-16.
- `wc + int + hot = whole` holds to about 1e-16.
- `whole` Σ and mflux match the old cache to within 2e-7.

## Current state of the data

- `prj.y/prj.y.00600.nc` is already in the new format for both models: 82 MB for MHD and
  157 MB for CRMHD. The old file was 17 MB or 35 MB.
- Every other `prj.y` file is in the old format and will be rebuilt on first read. That
  takes about 2 min per snapshot (from HDF5). The full set comes to roughly 135 GB, and
  `/projects/EOSTRIKE` has about 21 TB free.
- Inspection PNGs for snapshot 600 are in `/scratch/gpfs/EOST/changgoo/talk_figures/`
  (StellarAI scratch).

## Running on Stellar

1. `git pull`, activate the `pyathena` env, and check that
   `/projects/EOSTRIKE/TIGRESSPP-CR/` is visible **on Stellar compute nodes**. It is *not*
   mounted on StellarAI compute nodes, which is why the run was moved. If it isn't
   visible, pass `--basedir-root` pointing at a copy.
2. Check Stellar's partitions and QOS limits with `qos` and the RC docs before choosing a
   walltime.
3. Run a test job on one snapshot first. Memory per process has **not been measured**,
   because the StellarAI test job failed before reaching Python work:
   ```sh
   srun -n 1 /usr/bin/time -v python plot_talk_edgeon.py --model crmhd --num 599 \
       --outdir <stellar_scratch>/talk_figures
   ```
   A rough guess is 2.6 GB of HDF5 per CRMHD snapshot plus dask intermediates, so plan
   for 10–20 GB per rank until it is measured.
4. Full run (from `python/`):
   ```sh
   srun -n <N> python plot_talk_edgeon.py --movie --outdir <stellar_scratch>/talk_figures
   ```
   Snapshots are split round-robin across ranks, and rank 0 makes the movies after a
   barrier. Use `--model mhd|crmhd` to run one model per job.

   On StellarAI, mpi4py links Intel MPI 2021.18 and needed `srun --mpi=pmi2` with
   `I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so`. Check what Stellar's env needs.

## Open items / caveats

- `ruff` isn't installed in the `pyathena` env, so these files have not been run through
  `ruff format`/`check`. Some lines are over 88 characters, and the pre-commit hook will
  reformat them.
- `python/plot_slices.py` has unrelated uncommitted local changes. They were not
  included in this commit.
- The old `prj.y` caches were made with slightly different μ(T) sampling (about 1% in T).
  Rebuilt files can therefore differ from the old ones by a few percent in the hot phase,
  from cells sitting right at the temperature cuts.
