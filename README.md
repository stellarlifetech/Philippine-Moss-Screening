# Philippine moss computational screening

Analysis code and data for:

> Hernandez JPT, Lubos LC, Buot IE Jr. Computational Screening of Philippine Moss Morphology as Candidate Templates for Biomimetic Material Design (manuscript under review).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23445619.svg)](https://doi.org/10.5281/zenodo.23445619)

The workflow fuses image-derived morphology descriptors with literature-derived trait scores to prioritize six Philippine mosses (*Sphagnum junghuhnianum*, *Fissidens nobilis*, *Pyrrhobryum spiniforme*, *Entodon plicatus*, *Dendrocyathophorum decolyi*, *Hyophila involuta*). It then tests how robust the ranking is.

**Scope.** This is an exploratory computational framework. The quantitative dataset is one non-standardized photomicrograph per species (n = 1, no replication), so the rankings illustrate how the workflow behaves; they are not species-level conclusions and not evidence of radiation-shielding performance. Transmission values are hypothetical mathematical illustrations. Table 12 is a **synthetic design simulation**: simulated values, not data.

## Repository structure

```
data/                       species-keyed input tables (manuscript Tables 3 and 5)
src/moss_screening.py       full analysis: regenerates Tables 6-12 and S2-S5
src/make_figures.py         Figures 4-6
src/verify_outputs.py       reproduction test: reruns everything and compares with results/
results/                    published outputs (exact files reported in the manuscript)
figures/                    Figures 4-6
tests/                      regression tests (pytest)
reference/image_pipeline.py REFERENCE ONLY - not used for Tables 3 or 6-12 (see below)
reference/artefact_mask.py  reticle/bubble masking for the reprocessing check (Table S6)
source_images/              display-resolution exports of the six photomicrographs
results/sensitivity_masked/ reprocessing check outputs and QC overlays (Table S6)
SHA256SUMS.txt              checksums of every input, output and figure
environment-lock.txt        exact package versions used to produce results/ and figures/
```

## Reproduce the results

```bash
git clone https://github.com/stellarlifetech/Philippine-Moss-Screening.git
cd Philippine-Moss-Screening
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r environment-lock.txt     # exact versions; or requirements.txt for compatible ones
pytest -q                                             # about 10 s
python src/moss_screening.py                          # full run, about 8-10 min
python src/make_figures.py results figures
```

Every stochastic step uses NumPy's default generator with fixed seeds (`20261018`; `20261019` for Table S5). With the package versions in `environment-lock.txt`, the outputs are identical to `results/`. To check this automatically, run `python src/verify_outputs.py` (about 10 minutes): it reruns the analysis in a temporary folder, compares every table with `results/`, and checks the input checksums in `SHA256SUMS.txt`. For a quick check, run `python src/moss_screening.py --out quick --iterations 500 --reps 50`.

The script indexes every row by species name. Built-in consistency tests stop the run if any fused value cannot be recomputed from the inputs of the same species.

## Output files and manuscript tables

| Output file | Manuscript |
|---|---|
| `table06_fused_descriptors.csv` | Table 6 |
| `table07_heuristic.csv` | Table 7 |
| `table08_areal_density.csv` | Table 8 |
| `table09_pareto.csv` | Table 9 |
| `table10a_fusion_sweep.csv`, `table10b_scaling_loso.csv` | Table 10 |
| `table11_mc_*.csv`, `table11_summary_P_rank1.csv`, `table11_global_scores.csv` | Table 11 |
| `table12_synthetic_replication_SIMULATED.csv` | Table 12 (simulated) |
| `tableS2_elasticity.csv` | Table S2 |
| `table10d_unweighted_baselines.csv` | Table S3 |
| `table10c_leave_one_descriptor_out.csv` | Table S4 |
| `table11b_mc_ordinal_pm2.csv` | Table S5 |
| `sensitivity_masked/descriptors_masked_vs_unmasked.csv`, `rank_comparison.csv`, `descriptor_agreement.csv` | Table S6 (reprocessing check) |
| `run_metadata.json` | seed, iterations, attenuation coefficients, convergence check |

## Processing new or replicated images

Organise the raw images as `images/<Species_name>/<voucher_id>/<field_id>.tif`. Then run:

```bash
python reference/image_pipeline.py images/ image_results/            # descriptors per field
python reference/image_pipeline.py images/ image_results/ --sweep    # + processing-parameter sensitivity
```

The pipeline averages fields within each specimen and treats the specimen as the biological unit. It reports species means, SD and range, and the between- and within-specimen variance components. **This reference implementation was not used for any published value.** It adds quality-control rules that were not applied to Table 3. The values in Table 3 come from the authors' original image-analysis script; that script and the original camera files are not in this repository (available from the corresponding author on request), so Table 3 cannot be regenerated exactly. Display-resolution copies of the six images are in `source_images/`.

## Reprocessing check with artefact masking (Table S6)

```bash
python reference/artefact_mask.py source_images results/sensitivity_masked
```

This reprocesses the six display-resolution images with the reference workflow, with and
without automatic masking. Reticle masking is reliable (both scale lines found in all six
images). Bubble masking is conservative and incomplete, so check the QC overlays. Edge
density and continuity agree with Table 3, but packing density and regularity do not.
This is reported in the manuscript as a limitation of Table 3.

## Citation and licence

Please cite the archived code as: Hernandez JPT, Lubos LC, Buot IE Jr. 2026. Philippine moss computational screening: analysis code (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.23445619. Citation metadata is in `CITATION.cff`. The code is released under the MIT licence. Raw photomicrographs are available from the corresponding author (stellarlifetech@gmail.com) on request.
