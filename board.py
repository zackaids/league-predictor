"""A bracket board for Streamlit, on `st.components.v2`.

Streamlit's native widgets can only stack boxes in columns, so the Play-In,
Swiss and knockout views share one renderer instead: plain HTML/CSS/JS in
`assets/board.{js,css}` that draws columns of match cards with odds bars, league
markers and connector lines, follows Streamlit's light/dark theme, and reports
clicks back as three events.

    board.render(data, key="playin", on_pick=..., on_action=..., on_team=...)

`data` is the dict documented at the top of `assets/board.js`. Each callback gets
the event payload ({match, team} / {id} / {name}); they run as widget callbacks,
before the rerun renders, so they may change session state freely.
"""
from __future__ import annotations

from typing import Callable

import streamlit as st

import paths

ASSETS = paths.ROOT / "assets"

# Registered once at import. Re-registering with the same name on every rerun
# would warn and make instances share state.
_board = st.components.v2.component(
    "worlds_board",
    css=(ASSETS / "board.css").read_text(encoding="utf-8"),
    js=(ASSETS / "board.js").read_text(encoding="utf-8"),
)


def render(data: dict, key: str, *, on_pick: Callable[[dict], None] | None = None,
           on_action: Callable[[dict], None] | None = None,
           on_team: Callable[[dict], None] | None = None) -> None:
    handlers = {"pick": on_pick, "action": on_action, "team": on_team}
    callbacks = {}
    for name, fn in handlers.items():
        if fn is not None:
            # Default-arg binding: each closure must keep its own `name` and `fn`.
            callbacks[f"on_{name}_change"] = (
                lambda name=name, fn=fn: fn(dict(getattr(st.session_state[key], name))))
    _board(key=key, data=data, **callbacks)
