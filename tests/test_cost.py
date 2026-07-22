"""Pre-flight cost estimate: token counting, pricing, soft-ceiling logic."""

from __future__ import annotations

from reddit_research.analyze.cost import CostEstimate, estimate_cost, price_for
from reddit_research.defaults import COST_SOFT_CEILING_USD

from .cache_fixtures import comment


def test_price_for_known_and_unknown() -> None:
    (pin, pout), known = price_for("gpt-5.4-mini")
    assert (pin, pout) == (0.75, 4.50)
    assert known is True
    _, known2 = price_for("gpt-9.9-imaginary")
    assert known2 is False


def test_empty_corpus_costs_nothing() -> None:
    est = estimate_cost([], topic="Notion", model_id="gpt-5.4-mini")
    assert est.projected_usd == 0.0
    assert est.output_tokens == 0


def test_estimate_scales_with_corpus_and_is_positive() -> None:
    small = estimate_cost(
        [comment(f"c{i}", body="notion is slow " * 10) for i in range(5)],
        topic="Notion",
        model_id="gpt-5.4-mini",
    )
    big = estimate_cost(
        [comment(f"c{i}", body="notion is slow " * 10) for i in range(200)],
        topic="Notion",
        model_id="gpt-5.4-mini",
    )
    assert small.projected_usd > 0
    assert big.projected_usd > small.projected_usd
    assert big.input_tokens > small.input_tokens


def test_pricier_model_costs_more() -> None:
    comments = [comment(f"c{i}", body="notion " * 20) for i in range(50)]
    mini = estimate_cost(comments, topic="Notion", model_id="gpt-5.4-mini")
    sol = estimate_cost(comments, topic="Notion", model_id="gpt-5.6-sol")
    assert sol.projected_usd > mini.projected_usd


def test_soft_ceiling_default_is_two_dollars() -> None:
    assert COST_SOFT_CEILING_USD == 2.00


def test_estimate_is_a_frozen_dataclass() -> None:
    est = estimate_cost([comment("c1")], topic="Notion", model_id="gpt-5.4-mini")
    assert isinstance(est, CostEstimate)


def test_preflight_runs_before_map_and_can_abort() -> None:
    import pytest

    from reddit_research.analyze.pipeline import run_analysis

    from .cache_fixtures import make_cache, post

    class ExplodingMapModel:
        def batch(self, inputs):  # noqa: ANN001
            raise AssertionError("map (paid call) must not run after preflight aborts")

    cache = make_cache([post("p1", [comment("c1", body="notion is slow", score=10)])])
    seen: list = []

    def preflight(filtered):  # noqa: ANN001
        seen.append(filtered)
        raise RuntimeError("declined")

    with pytest.raises(RuntimeError, match="declined"):
        run_analysis(
            cache,
            map_model=ExplodingMapModel(),
            min_score=5,
            model_id="gpt-5.4-mini",
            preflight=preflight,
        )
    assert len(seen) == 1 and len(seen[0]) == 1  # preflight saw the filtered corpus
