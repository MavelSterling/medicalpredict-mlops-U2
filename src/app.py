"""
Aplicación Flask para el servicio de diagnóstico médico
Desarrollado para el taller de Pipeline de MLOps + Docker
"""

from flask import Flask, render_template, request, jsonify
import json
import logging
import os
from datetime import datetime, timezone
from collections import Counter
import pytz
from model import predict_medical_diagnosis

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)

# Configuración de la aplicación
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Rutas y utilidades para registro de predicciones
LOG_DIR = os.path.join(os.path.dirname(__file__), 'data')
LOG_FILE = os.path.join(LOG_DIR, 'predictions_log.jsonl')
COL_TZ = pytz.timezone('America/Bogota')


def _ensure_log_dir() -> None:
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def _append_prediction_log(entry: dict) -> None:
    """Agrega una predicción al log como JSONL (1 línea por predicción)."""
    try:
        _ensure_log_dir()
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"No se pudo escribir en el log de predicciones: {str(e)}")


def _read_prediction_log() -> list:
    """Lee el archivo JSONL y devuelve la lista de entradas (puede ser vacía)."""
    if not os.path.exists(LOG_FILE):
        return []
    entries = []
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Línea inválida en log de predicciones; se ignora")
    except Exception as e:
        logger.error(f"No se pudo leer el log de predicciones: {str(e)}")
    return entries


def _compute_prediction_stats(entries: list) -> dict:
    """Calcula estadísticas requeridas a partir de la lista de entradas."""
    if not entries:
        return {
            'counts_by_category': {},
            'last_5_predictions': [],
            'last_prediction_date': None
        }

    # Parseo seguro de timestamps (ISO-8601) y orden
    def parse_ts(e):
        ts = e.get('timestamp')
        try:
            # fromisoformat admite sufijo +00:00; si viene con Z, normalizamos
            if isinstance(ts, str) and ts.endswith('Z'):
                ts = ts[:-1] + '+00:00'
            return datetime.fromisoformat(ts)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    entries_sorted = sorted(entries, key=parse_ts)

    # Conteos por categoría (diagnosis)
    diagnoses = [e.get('diagnosis', 'DESCONOCIDO') for e in entries_sorted]
    counts = dict(Counter(diagnoses))

    # Últimas 5 predicciones (más recientes)
    last_5 = entries_sorted[-5:][::-1]
    last_5_compact = []
    for e in last_5:
        raw_ts = e.get('timestamp')
        local_ts_str = None
        try:
            ts_str = raw_ts
            if isinstance(ts_str, str) and ts_str.endswith('Z'):
                ts_str = ts_str[:-1] + '+00:00'
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt_col = dt.astimezone(COL_TZ)
            local_ts_str = dt_col.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            local_ts_str = raw_ts

        last_5_compact.append({
            'timestamp': raw_ts,
            'timestamp_local': local_ts_str,
            'diagnosis': e.get('diagnosis'),
            'most_likely_condition': e.get('most_likely_condition'),
            'confidence': e.get('confidence'),
        })

    # Fecha de la última predicción
    last_date = last_5_compact[0]['timestamp'] if last_5_compact else None
    last_date_local = last_5_compact[0]['timestamp_local'] if last_5_compact else None

    return {
        'counts_by_category': counts,
        'last_5_predictions': last_5_compact,
        'last_prediction_date': last_date,
        'last_prediction_date_local': last_date_local
    }

@app.route('/')
def index():
    """Página principal con interfaz web para médicos"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predicciones de diagnóstico médico
    
    Acepta datos en formato JSON con síntomas del paciente
    """
    try:
        # Obtener datos del request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Validar que se proporcionaron síntomas
        if not data:
            return jsonify({
                'error': 'No se proporcionaron datos de síntomas',
                'diagnosis': 'ERROR',
                'confidence': 0.0
            }), 400
        
        # Convertir valores a float/int
        symptoms = {}
        for key, value in data.items():
            try:
                # Convertir a float si es posible
                symptoms[key] = float(value)
            except (ValueError, TypeError):
                # Si no se puede convertir, usar 0
                symptoms[key] = 0.0
        
        # Realizar predicción
        result = predict_medical_diagnosis(symptoms)

        # Log de la predicción
        logger.info(f"Predicción realizada: {result.get('diagnosis', 'ERROR')}")

        # Registrar predicción en almacenamiento local (JSONL)
        try:
            entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'diagnosis': result.get('diagnosis'),
                'confidence': result.get('confidence'),
                'severity_score': result.get('severity_score'),
                'most_likely_condition': result.get('most_likely_condition'),
                'input_symptoms': result.get('input_symptoms', symptoms),
            }
            _append_prediction_log(entry)
        except Exception as e:
            logger.error(f"No se pudo registrar la predicción: {str(e)}")

        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error en el endpoint /predict: {str(e)}")
        return jsonify({
            'error': f'Error interno del servidor: {str(e)}',
            'diagnosis': 'ERROR',
            'confidence': 0.0
        }), 500

@app.route('/health')
def health_check():
    """Endpoint de health check para monitoreo"""
    return jsonify({
        'status': 'healthy',
        'service': 'medical-diagnosis-service',
        'version': '1.0.0'
    })

@app.route('/symptoms')
def get_available_symptoms():
    """Endpoint para obtener la lista de síntomas disponibles"""
    available_symptoms = [
        'fiebre', 'dolor_cabeza', 'nausea', 'fatiga', 'dolor_pecho',
        'dificultad_respirar', 'dolor_abdominal', 'mareos', 'perdida_peso',
        'tos', 'congestion_nasal', 'dolor_garganta', 'dolor_muscular',
        'dolor_articular', 'erupcion_cutanea', 'sangrado', 'cambios_vision',
        'confusion', 'convulsiones', 'dolor_espalda'
    ]
    
    return jsonify({
        'available_symptoms': available_symptoms,
        'description': 'Lista de síntomas disponibles para diagnóstico',
        'intensity_scale': '0-10 (0 = ausente, 10 = muy severo)'
    })

@app.route('/api/docs')
def api_documentation():
    """Documentación de la API"""
    docs = {
        'title': 'API de Diagnóstico Médico',
        'version': '1.0.0',
        'description': 'API para diagnóstico médico basado en síntomas del paciente',
        'endpoints': {
            'POST /predict': {
                'description': 'Realiza predicción de diagnóstico médico',
                'parameters': {
                    'symptoms': 'Diccionario con síntomas e intensidad (0-10)',
                    'example': {
                        'fiebre': 8,
                        'dolor_cabeza': 6,
                        'nausea': 4
                    }
                },
                'response': {
                    'diagnosis': 'NO_ENFERMO | ENFERMEDAD_LEVE | ENFERMEDAD_AGUDA | ENFERMEDAD_CRONICA | ENFERMEDAD_TERMINAL',
                    'confidence': 'Confianza del diagnóstico (0-1)',
                    'most_likely_condition': 'Condición más probable',
                    'recommendations': 'Lista de recomendaciones'
                }
            },
            'GET /health': {
                'description': 'Health check del servicio'
            },
            'GET /symptoms': {
                'description': 'Lista de síntomas disponibles'
            }
        }
    }
    
    return jsonify(docs)


@app.route('/api/report')
def api_report():
    """Endpoint JSON con estadísticas de predicciones."""
    entries = _read_prediction_log()
    stats = _compute_prediction_stats(entries)
    return jsonify(stats)


@app.route('/report')
def report_view():
    """Vista HTML con reporte para médicos."""
    entries = _read_prediction_log()
    stats = _compute_prediction_stats(entries)
    return render_template('report.html', stats=stats)

@app.errorhandler(404)
def not_found(error):
    """Manejo de errores 404"""
    return jsonify({
        'error': 'Endpoint no encontrado',
        'message': 'El endpoint solicitado no existe'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Manejo de errores 500"""
    return jsonify({
        'error': 'Error interno del servidor',
        'message': 'Ocurrió un error inesperado'
    }), 500

if __name__ == '__main__':
    # Configuración para desarrollo
    app.run(host='0.0.0.0', port=5000, debug=True)
