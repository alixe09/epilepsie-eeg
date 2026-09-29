"""La démo s'exécute sans erreur, y compris avec les seuils extrêmes du réglage d'usine."""
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
pytest.importorskip("streamlit")


@pytest.mark.parametrize("enregistrement,methode", [("chb01_03", "cnn_int8"), ("chb16_17", "classique")])
def test_reglage_modifiable(enregistrement, methode):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.query_params["enregistrement"] = enregistrement
    at.query_params["methode"] = methode
    at.run()
    assert not at.exception
    at.toggle[0].set_value(True).run()  # le réglage d'usine doit être une valeur du curseur
    assert not at.exception
    at.select_slider[0].set_value(0.9).run()
    assert not at.exception
    assert len(at.metric) == 3
