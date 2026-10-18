# Philippine moss computational screening

Analysis code and data for:

> Hernandez JPT, Lubos LC, Buot IE Jr. Computational Screening of Philippine Moss Morphology as Candidate Templates for Biomimetic Shielding-Oriented Materials. *Biodiversitas* (submitted).

The workflow fuses image-derived morphology descriptors with literature-derived trait scores to prioritize six Philippine mosses (*Sphagnum junghuhnianum*, *Fissidens nobilis*, *Pyrrhobryum spiniforme*, *Entodon plicatus*, *Dendrocyathophorum decolyi*, *Hyophila involuta*). It then tests how robust the ranking is.

**Scope.** The quantitative dataset is one photomicrograph per species (n = 1, no replication). The rankings are preliminary computational priorities. They are not measurements of radiation attenuation. Table 12 is a **synthetic design simulation**: simulated values, not data.

## Repository structure

```
data/                     species-keyed input tables (manuscript Tables 3 and 5)
src/moss_screening.py     full analysis: regenerates Tables 6-12 and S2
src/image_pipeline.py     reference image-processing implementation (Table 2 workflow + QC rules)
src/make_figures.py       Figures 4-6
results/                  outputs of the run reported in the manuscript
figures/                  Figures 4-6
tests/                    regression tests (pytest)
```

## Reproduce the results

```bash
git clone https://github.com/<your-username>/philippine-moss-screening.git
cd philippine-moss-screening
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                             # about 10 s
python src/moss_screening.py                          # full run, about 8-10 min
python src/make_figures.py results figures
```

Every stochastic step uses NumPy's default generator with the fixed seed `20261018`. If you use the same package versions, you get identical tables. For a quick check, run `python src/moss_screening.py --out quick --iterations 500 --reps 50`.

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
| `run_metadata.json` | seed, iterations, attenuation coefficients, convergence check |

## Processing new or replicated images

Organise the raw images as `images/<Species_name>/<voucher_id>/<field_id>.tif`. Then run:

```bash
python src/image_pipeline.py images/ image_results/            # descriptors per field
python src/image_pipeline.py images/ image_results/ --sweep    # + processing-parameter sensitivity
```

The pipeline averages fields within each specimen and treats the specimen as the biological unit. It reports species means, SD and range, and the between- and within-specimen variance components. The values in Table 3 come from the original analysis, not from this reference implementation.

## Citation and licence

Citation metadata is in `CITATION.cff`. The code is released under the MIT licence. Raw photomicrographs are available from the corresponding author (stellarlifetech@gmail.com) on request.
