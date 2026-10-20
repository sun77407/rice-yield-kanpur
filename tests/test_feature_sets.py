"""ADR-028 guard: no model feature set may contain a target-derived feature."""

from src.models.feature_sets import FEATURE_SETS, FORBIDDEN, LINEAR


def test_no_leaky_features() -> None:
    for name, cols in FEATURE_SETS.items():
        assert not FORBIDDEN & set(cols), name
        assert len(cols) == len(set(cols)), name


def test_linear_drops_gdd_only() -> None:
    assert "gdd_kharif" not in LINEAR
    assert "kharif_tmax_mean" in LINEAR
    assert "kharif_rh_mean" in LINEAR
