"""The dark theme in .streamlit/config.toml: scope, validity and contrast.

Nothing else in the suite can see any of this. AppTest never renders a theme,
and streamlit only *logs* a warning for a config key it does not recognise and
then ignores it -- so a typo'd key, a key in a section that reaches light mode,
or a palette tweak that drops a contrast below its floor would all ship green.
"""

import colorsys
import re
import tomllib
from pathlib import Path

import pytest
import streamlit as st

from streamlit_app import NEGATIVE_COLOR, POSITIVE_COLOR, TINT_ALPHA

CONFIG_PATH = Path(__file__).parent.parent / ".streamlit" / "config.toml"
CONFIG = tomllib.loads(CONFIG_PATH.read_text())
THEME = CONFIG["theme"]
# textColor, primaryColor and the two surfaces are required: every floor below
# is computed from them, so a missing one fails collection by name.
DARK = {k: v for k, v in THEME["dark"].items() if k != "sidebar"}
_SIDEBAR_SECTION = THEME["dark"].get("sidebar", {})
# The sidebar is the dark palette with its own section spread over it, except
# that it swaps the two surfaces unless that section sets them: its panel
# takes secondaryBackgroundColor, and its inputs backgroundColor.
SIDEBAR = {
    **DARK,
    **_SIDEBAR_SECTION,
    "backgroundColor": _SIDEBAR_SECTION.get(
        "backgroundColor", DARK["secondaryBackgroundColor"]
    ),
}
# Each palette with the surface its text and controls sit on.
PALETTES = (DARK, SIDEBAR)
# Alpha streamlit applies on top of textColor, not settable from the config:
# st.caption renders at opacity 0.6, glide draws pinned grid columns -- the
# predicted Sentiment among them -- at globalAlpha 0.6, and the grid's header
# text is textColor at 0.6.
MUTED_ALPHA = 0.6
WHITE = "#ffffff"
# The built-in dark blueColor, for when the config leaves it unset.
BUILTIN_DARK_BLUE = "#0068c9"


def _option_keys(node: dict, prefix: str = "") -> list[str]:
    """Dotted option names, recursing only where streamlit sees a section.

    That is every top-level table, and sidebar/light/dark under theme; any
    other table is a single option's value (server.trustedUserHeaders).
    """
    keys = []
    for name, value in node.items():
        key = f"{prefix}.{name}" if prefix else name
        section = isinstance(value, dict) and (
            not prefix
            or (key.startswith("theme.") and name in {"sidebar", "light", "dark"})
        )
        keys += _option_keys(value, key) if section else [key]
    return keys


def _theme_values() -> list[tuple[str, object]]:
    def leaves(node: dict, prefix: str) -> list[tuple[str, object]]:
        out = []
        for name, value in node.items():
            key = f"{prefix}.{name}"
            out += leaves(value, key) if isinstance(value, dict) else [(key, value)]
        return out

    return leaves(THEME, "theme")


def _rgb(hex_color: str) -> tuple[int, ...]:
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def _over(fg: str, bg: str, alpha: float) -> str:
    """`fg` composited over `bg` at `alpha`, as a hex string."""
    mixed = (
        round(alpha * f + (1 - alpha) * b)
        for f, b in zip(_rgb(fg), _rgb(bg), strict=True)
    )
    return "#" + "".join(f"{c:02x}" for c in mixed)


def _luminance(hex_color: str) -> float:
    def channel(c: int) -> float:
        v = c / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in _rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    """WCAG 2 contrast ratio."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _hls(hex_color: str) -> tuple[float, float, float]:
    r, g, b = (c / 255 for c in _rgb(hex_color))
    return colorsys.rgb_to_hls(r, g, b)


def _hue_distance(a: str, b: str) -> float:
    d = abs(_hls(a)[0] - _hls(b)[0]) * 360 % 360
    return min(d, 360 - d)


class TestScope:
    def test_only_dark_mode_is_customized(self):
        # A key in [theme] or [theme.light] reaches light mode, which is meant
        # to stay streamlit's built-in theme; a lone [theme] block also locks
        # the app to one mode and drops System/Light/Dark from the main menu.
        # Other sections ([server], ...) are not the theme's business.
        assert set(THEME) == {"dark"}
        tables = {k for k, v in THEME["dark"].items() if isinstance(v, dict)}
        assert tables <= {"sidebar"}

    def test_sets_colors_only(self):
        # Fonts, sizes and radii would change on every light/dark toggle, and
        # TEXT_COL_WIDTH / LONG_TEXT_CHARS were measured at the default font.
        names = [key.rsplit(".", 1)[-1] for key, _ in _theme_values()]
        assert names
        assert [n for n in names if not n.endswith(("Color", "Colors"))] == []

    def test_colors_are_six_digit_hex(self):
        # Streamlit also takes other forms ("7a58d8", "#fff", "#rrggbbaa"),
        # which the arithmetic here would misread or crash on.
        for key, value in _theme_values():
            colors = value if isinstance(value, list) else [value]
            for color in colors:
                assert re.fullmatch(r"#[0-9a-fA-F]{6}", str(color)), key

    # Every section, not only the theme: a typo is silent anywhere in the file.
    @pytest.mark.parametrize("key", _option_keys(CONFIG))
    def test_every_key_is_a_streamlit_option(self, key):
        # Streamlit ignores an unknown key with a log warning; get_option is
        # the public lookup, and it raises for a key that is not defined.
        st.get_option(key)


class TestContrast:
    def test_body_text_is_aaa_in_main_and_sidebar(self):
        for palette in PALETTES:
            assert _contrast(palette["textColor"], palette["backgroundColor"]) >= 7

    def test_muted_text_is_aa_in_main_and_sidebar(self):
        for palette in PALETTES:
            surface = palette["backgroundColor"]
            muted = _over(palette["textColor"], surface, MUTED_ALPHA)
            assert _contrast(muted, surface) >= 4.5

    def test_pinned_prediction_is_aa_on_its_tint(self):
        # The pinned Sentiment cell: muted text over the app's own tint, so a
        # change to POSITIVE_COLOR, NEGATIVE_COLOR or TINT_ALPHA is checked
        # against this theme too.
        for sentiment in (POSITIVE_COLOR, NEGATIVE_COLOR):
            cell = _over(sentiment, DARK["backgroundColor"], TINT_ALPHA)
            text = _over(DARK["textColor"], cell, MUTED_ALPHA)
            assert _contrast(text, cell) >= 4.5

    def test_grid_header_text_is_aa(self):
        # Unset, the header is streamlit's bgMix: the two surfaces half and half.
        header = DARK.get(
            "dataframeHeaderBackgroundColor",
            _over(DARK["secondaryBackgroundColor"], DARK["backgroundColor"], 0.5),
        )
        text = _over(DARK["textColor"], header, MUTED_ALPHA)
        assert _contrast(text, header) >= 4.5

    def test_primary_button_label_is_aa(self):
        # Streamlit always draws the primary button's label in white.
        for palette in PALETTES:
            assert _contrast(WHITE, palette["primaryColor"]) >= 4.5

    def test_primary_is_visible_in_main_and_sidebar(self):
        # It also draws focus rings, the selected option and progress bars.
        for palette in PALETTES:
            assert _contrast(palette["primaryColor"], palette["backgroundColor"]) >= 3

    def test_sentiment_bars_read_on_the_background(self):
        for sentiment in (POSITIVE_COLOR, NEGATIVE_COLOR):
            assert _contrast(sentiment, DARK["backgroundColor"]) >= 3

    def test_confidence_bars_read_on_the_grid(self):
        # ProgressColumn(color="blue") draws in blueColor on the grid's cells.
        blue = DARK.get("blueColor", BUILTIN_DARK_BLUE)
        assert _contrast(blue, DARK["backgroundColor"]) >= 3


def test_primary_is_not_a_sentiment_hue():
    # The built-in dark primary was #ff4b4b -- NEGATIVE_COLOR itself -- so the
    # one call to action on the page read as "negative". A near-gray has no
    # hue to confuse (colorsys reports 0, which is red's).
    for palette in PALETTES:
        if _hls(palette["primaryColor"])[2] < 0.1:
            continue
        for sentiment in (POSITIVE_COLOR, NEGATIVE_COLOR):
            assert _hue_distance(palette["primaryColor"], sentiment) >= 60
