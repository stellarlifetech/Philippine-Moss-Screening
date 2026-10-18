"""Regression tests: run with `pytest -q` from the repository root."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import moss_screening as ms  # noqa: E402

ms.load_inputs(ROOT / "data")

# Manuscript Table 6 (fused descriptors), keyed by species
TABLE6 = {
    "Sphagnum junghuhnianum":     [0.6828, 0.9167, 0.8531, 0.3604, 1.0000],
    "Fissidens nobilis":          [0.6470, 0.5000, 0.3057, 0.4934, 0.3333],
    "Pyrrhobryum spiniforme":     [0.3015, 0.4336, 0.7000, 0.4167, 0.3333],
    "Entodon plicatus":           [0.5714, 0.2809, 0.4763, 0.3529, 0.1667],
    "Dendrocyathophorum decolyi": [0.0000, 0.3968, 0.3000, 1.0000, 0.5000],
    "Hyophila involuta":          [0.7040, 0.2757, 0.0387, 0.4503, 0.0000],
}


def test_fused_descriptors_match_table6():
    F = ms.fuse(ms.IMAGE, ms.LIT)
    for sp, vals in TABLE6.items():
        assert np.allclose(F.loc[sp, list("PGUSM")].values, vals, atol=6e-5), sp


def test_species_mapping_guard():
    """The species with the highest raw edge density and MC score must hold S = 1."""
    F = ms.fuse(ms.IMAGE, ms.LIT)
    ms.run_consistency_tests(F)
    assert F["S"].idxmax() == "Dendrocyathophorum decolyi"


def test_row_order_does_not_matter():
    F1 = ms.fuse(ms.IMAGE, ms.LIT)
    F2 = ms.fuse(ms.IMAGE.iloc[::-1], ms.LIT.sample(frac=1, random_state=1))
    assert np.allclose(F1.loc[ms.SPECIES].values, F2.loc[ms.SPECIES].values)


def test_heuristic_ranking_table7():
    r = ms.ranks(ms.heuristic(ms.fuse(ms.IMAGE, ms.LIT)))
    assert r.sort_values().index.tolist() == [
        "Sphagnum junghuhnianum", "Fissidens nobilis", "Pyrrhobryum spiniforme",
        "Dendrocyathophorum decolyi", "Entodon plicatus", "Hyophila involuta"]


def test_weights_sum_to_one():
    assert sum(ms.WEIGHTS.values()) == pytest.approx(1.0)


def test_quick_full_run(tmp_path):
    ms.N_ITER, ms.N_REPS = 50, 10
    ms.main(tmp_path)
    assert (tmp_path / "table07_heuristic.csv").exists()
    assert (tmp_path / "table12_synthetic_replication_SIMULATED.csv").exists()
