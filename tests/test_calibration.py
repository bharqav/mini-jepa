"""Calibration unit tests: synthetic logits with known properties (no weights)."""

from jev_qwen.calibration import (
    conf_hit_pairs,
    ece_of,
    fit_temperature,
    nll_of,
    reliability_svg,
    risk_coverage,
    risk_coverage_svg,
    temperature_report,
)


def test_temperature_fixes_overconfidence():
    # 80% accurate at ~95% confidence -> overconfident, T should grow
    logits = [[3.0, 0.0]] * 50
    labels = [0] * 40 + [1] * 10
    t = fit_temperature(logits, labels)
    assert t > 1.0, f"expected T>1 for overconfident logits, got {t}"
    assert nll_of(logits, labels, t) < nll_of(logits, labels, 1.0)


def test_temperature_fixes_underconfidence():
    logits = [[0.4, 0.0]] * 50
    labels = [0] * 50
    t = fit_temperature(logits, labels)
    assert t < 1.0, f"expected T<1 for underconfident logits, got {t}"


def test_perfect_model_has_zero_ece():
    pairs = conf_hit_pairs([[10.0, 0.0]] * 30, [0] * 30)
    e = ece_of(pairs)
    assert e["ece"] < 0.05


def test_risk_coverage_thresholds():
    pairs = [(0.9, True)] * 8 + [(0.1, False)] * 2
    rc = risk_coverage(pairs)
    assert rc["acc_at_0.5"] == 1.0
    assert rc["acc_at_1.0"] == 0.8
    assert len(rc["curve"]) == 10


def test_report_splits_fit_measure():
    logits = [[2.0, 0.0]] * 20 + [[0.0, 2.0]] * 20
    labels = [0] * 20 + [1] * 20
    rep = temperature_report(logits[:20], labels[:20], logits[20:], labels[20:])
    assert rep["fit_n"] == 20 and rep["measure_n"] == 20
    assert rep["post_nll"] <= rep["pre_nll"] + 1e-9


def test_svgs_are_valid_xml():
    e = ece_of(conf_hit_pairs([[2.0, 0.0]] * 10, [0] * 10))
    s1 = reliability_svg(e["bins"])
    s2 = risk_coverage_svg(risk_coverage(conf_hit_pairs([[2.0, 0.0]] * 10, [0] * 10))["curve"])
    for s in (s1, s2):
        assert s.startswith("<svg") and s.strip().endswith("</svg>")
        assert "<polyline" in s or "<rect" in s
