# tests/conftest.py
import os
import sys
import pytest

# Asegura que podamos importar desde src/
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import app as app_module  

@pytest.fixture
def client(monkeypatch, tmp_path):
    """
    Crea un test client de Flask y redirige el log de predicciones
    a una carpeta temporal para no tocar archivos reales.
    """
    # Redirige rutas de log a tmp
    log_dir = tmp_path / "data"
    log_file = log_dir / "predictions_log.jsonl"
    monkeypatch.setattr(app_module, "LOG_DIR", str(log_dir))
    monkeypatch.setattr(app_module, "LOG_FILE", str(log_file))

    # Devuelve el cliente de pruebas
    app = app_module.app
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client
