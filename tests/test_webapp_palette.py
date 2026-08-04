from __future__ import annotations

from pathlib import Path

from webapp import palette


def test_web_theme_reuses_manuscript_figure_palette() -> None:
    expected_colors = {
        "#262A33",
        "#FFF1E5",
        "#F7F7F7",
        "#D8CDBF",
        "#8A7D70",
        "#8EC7E8",
        "#3FA7D6",
        "#1F78B4",
        "#08306B",
        "#008585",
        "#74A892",
        "#E5C185",
        "#C7522A",
        "#0F6B78",
        "#85A993",
        "#F7C435",
        "#C75A1B",
    }
    module_colors = {
        value
        for name, value in vars(palette).items()
        if name.isupper() and isinstance(value, str)
    }
    assert module_colors == expected_colors

    css = Path("webapp/assets/styles.css").read_text(encoding="utf-8").upper()
    for color in expected_colors:
        assert color in css


def test_streamlit_theme_uses_manuscript_neutrals_and_primary_color() -> None:
    config = Path(".streamlit/config.toml").read_text(encoding="utf-8").upper()
    assert 'PRIMARYCOLOR = "#0F6B78"' in config
    assert 'BACKGROUNDCOLOR = "#FFFFFF"' in config
    assert 'SECONDARYBACKGROUNDCOLOR = "#F7F7F7"' in config
    assert 'TEXTCOLOR = "#262A33"' in config
