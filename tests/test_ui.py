from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_controls():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "src/mass_spring_twin/ui/app.py"
    ).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert app.session_state["playing"]
    app.button[1].click().run(timeout=30)
    assert not app.session_state["playing"]
    app.button[2].click().run(timeout=30)
    assert app.session_state["clock"] == 0
    app.sidebar.number_input[2].set_value(15.0).run(timeout=30)
    assert not app.exception
    assert any("overdamped" in value.value for value in app.markdown)
