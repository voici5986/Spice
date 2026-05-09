from __future__ import annotations

from typing import Any

from spice.decision.compare_rich import render_compare_rich


def render_decision_card(
    compare_payload: dict[str, Any],
    *,
    use_bars: bool = True,
    width: int | None = None,
) -> str:
    return render_compare_rich(compare_payload, use_bars=use_bars, width=width)
