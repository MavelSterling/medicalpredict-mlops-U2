# tests/test_api.py
import json
import app as app_module

def test_report_inicial_vacio(client):
    """Al inicio, /api/report debe estar vacío/por defecto."""
    resp = client.get("/api/report")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["counts_by_category"] == {}
    assert data["last_5_predictions"] == []
    assert data["last_prediction_date"] is None

def test_predict_y_stats_actualizadas(client, monkeypatch):
    """POST /predict debe registrar una predicción y /api/report reflejarla."""
    # Simula el modelo para que devuelva un resultado determinístico
    def fake_predict_medical_diagnosis(symptoms: dict):
        return {
            "diagnosis": "ENFERMEDAD_LEVE",
            "confidence": 0.9,
            "severity_score": 2,
            "most_likely_condition": "Gripe",
            "input_symptoms": symptoms,
            "recommendations": ["Reposo", "Hidratación"]
        }

    # Reemplaza la función importada en app.py
    monkeypatch.setattr(app_module, "predict_medical_diagnosis", fake_predict_medical_diagnosis)

    # 1) Antes de predecir, el reporte está vacío (sanity check)
    resp0 = client.get("/api/report")
    assert resp0.status_code == 200
    data0 = resp0.get_json()
    assert data0["counts_by_category"] == {}

    # 2) Hacemos una predicción
    payload = {
        "fiebre": 8,
        "dolor_cabeza": 6,
        "nausea": 4
    }
    resp1 = client.post("/predict", json=payload)
    assert resp1.status_code == 200
    pred = resp1.get_json()
    assert pred["diagnosis"] == "ENFERMEDAD_LEVE"
    assert pred["most_likely_condition"] == "Gripe"
    assert "confidence" in pred

    # 3) Ahora el reporte debe reflejar la predicción
    resp2 = client.get("/api/report")
    assert resp2.status_code == 200
    data2 = resp2.get_json()

    # Debe haber contado la categoría
    assert "ENFERMEDAD_LEVE" in data2["counts_by_category"]
    assert data2["counts_by_category"]["ENFERMEDAD_LEVE"] >= 1

    # Debe existir última fecha y la última predicción arriba
    assert data2["last_prediction_date"] is not None
    assert len(data2["last_5_predictions"]) >= 1
    assert data2["last_5_predictions"][0]["diagnosis"] == "ENFERMEDAD_LEVE"
