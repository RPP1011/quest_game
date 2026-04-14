from dataclasses import dataclass

from app.calibration.scorer import aggregate, critic_score, mae, pearson, rmse


@dataclass
class _Issue:
    severity: str


def test_critic_score_no_issues():
    assert critic_score([]) == 1.0


def test_critic_score_errors_weighted_more():
    assert critic_score([_Issue("error")]) == 0.75
    assert critic_score([_Issue("warning")]) == 0.9


def test_critic_score_clamped_at_zero():
    issues = [_Issue("error")] * 10
    assert critic_score(issues) == 0.0


def test_mae_rmse_pearson_basics():
    pairs = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.8)]
    assert abs(mae(pairs) - (0.1 + 0.0 + 0.1) / 3) < 1e-9
    assert rmse(pairs) > 0
    assert pearson(pairs) > 0.9  # strongly correlated


def test_pearson_zero_on_constant():
    pairs = [(0.5, 0.1), (0.5, 0.9)]
    assert pearson(pairs) == 0.0


def test_aggregate_wraps_all():
    a = aggregate([(0.2, 0.3), (0.5, 0.5), (0.9, 0.8)])
    assert 0.0 <= a.mae < 0.5
    assert -1.0 <= a.pearson <= 1.0
