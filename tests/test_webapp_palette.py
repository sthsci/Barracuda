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


def test_dash_theme_uses_flat_print_neutrals_and_accessible_focus() -> None:
    css = Path("webapp/assets/styles.css").read_text(encoding="utf-8").upper()
    assert "--BARRACUDA-PRIMARY: #1D515B" in css
    assert "--BARRACUDA-PAPER: #F2F5F2" in css
    assert "--BARRACUDA-SHEET: #FCFDFB" in css
    assert "--BARRACUDA-MIST: #DCE5E0" in css
    assert "--BARRACUDA-INK: #17272C" in css
    assert "GRADIENT(" not in css
    assert ":FOCUS-VISIBLE" in css
    assert "PREFERS-REDUCED-MOTION" in css


def test_dense_plot_layouts_have_explicit_containment_rules() -> None:
    css = Path("webapp/assets/styles.css").read_text(encoding="utf-8")

    assert ".barracuda-donor-posterior-stack" in css
    assert "contain: layout paint" in css
    assert "grid-template-rows: 280px auto auto" in css
