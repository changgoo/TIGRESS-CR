# Talk figure rule (moon theme)

Use this when recreating a paper figure for a talk, or making a new one. Hand it to an agent as is.

---

**Recreate this figure as a talk figure for a reveal.js deck on the moon theme (1280×720 slides).** Rebuild it
from the original data and scripts. Never trace or recolor the paper PNG. Follow every rule below.

## Style
- Start from `shared/talk-moon.mplstyle`: background `#002b36`, text `#eee8d5`, axes and ticks `#93a1a1`, no top or
  right spines, frameless legends, STIX sans math.
- **Text must be readable from the back of the room.** Every label, tick, legend entry and annotation must render at
  **≥ 18 px on the slide**. Check it as
  `font_pt × dpi/72 × (displayed width in px ÷ image width in px) ≥ 18`.
  The defaults (25 pt base, 20 pt legend, 150 dpi) pass for a ~10–12 inch figure shown 600–760 px wide. Make
  the figure smaller rather than the fonts.
- Use these colors consistently across the deck:
  - phases: cool or warm/cold `#268bd2`, intermediate `#6c71c4`, hot `#dc322f`;
  - runs: without CRs (MHD) `#93a1a1`, with CRs (CRMHD) `#cb4b16`;
  - frameworks: TIGRESS-classic `#2aa198`, TIGRESS-NCR `#b58900`, TIGRESS++ `#d33682`;
  - TIGRESS 2020 models: R2 `#4f9fe0`, LGR2 `#a9cff2`, R4 `#f08c2e`, LGR4 `#f7c48f`, R8 `#b07fd6`, LGR8 `#d9c2ee`,
    R16 `#e0544f`. R models are solid or circles; LGR models are dashed or squares.
- For colormaps on the dark background, use `inferno` or `magma`. Mask values below the plotted range so they show
  the background.

## Plain-English labels
- **Titles say what is plotted, in words**: "cool gas: mass flux", not "wc F_M".
- **Axis labels are words plus symbol plus units**, for example "outflow velocity v_out [km/s]" or
  "star formation rate Σ_SFR [M⊙ kpc⁻² yr⁻¹]". The symbol alone is fine only when the slide text defines it.
- Never show internal names in the figure: wc → warm/cold, crmhd → with CRs, mhd → without CRs, sfr40 → Σ_SFR,
  `mass_loading` → mass loading η_M.
- Phase labels give their temperature range once, for example "cool (T < 2×10⁴ K)".
- Annotate one or two key numbers or slopes on the plot at most. The slide bullets carry the rest.
- Put context the plot needs, such as height (|z| = 1 kpc), model name or time window, as a small muted tag in a
  corner.

## Layout
- At most 2×2 panels. Aim for an aspect that fits a 16:9 slide, either beside the bullets or across the full width.
- **Fit the axis range to the data.** Don't leave empty space just to fit a legend; put the legend in a genuinely
  empty region or outside the axes on the right.
- Nothing may overlap: legends with curves, tick labels where panels meet (give each row its own ticks), or tags
  with axis labels.
- When figures will sit side by side or be compared, make them the same size with the same axes. Keep the same
  axis range across heights, models and runs.

## Data honesty
- Every plotted quantity comes from the data. Write the normalization and selection (phase cuts, height, time
  window) in the script's docstring.
- Check derived numbers against the published ones (paper tables, README) and record the agreement. If something
  can't be reproduced, leave it out and say so. Don't fake it, and don't silently substitute something else.
- Quote numbers on the slide only after they are recorded in `science/<module>/spec.md`.

## Files
- Script: `science/<module>/scripts/<name>_moon.py`. Its docstring gives the data paths (environment variable
  plus default), what is computed, the run command and the outputs. Arguments cover variants such as height or
  model.
- Output: `science/<module>/figures/<name>[_variant]_moon.png` (PNG, 150 dpi, tight bbox).
- Register the figure in the inventory in `science/<module>/spec.md`. In the deck, use it **without** `.light`.
- Look at the PNG before using it, then check the slide still fits 720 px with all builds shown.
