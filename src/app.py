"""
Aplicación Flask para el servicio de diagnóstico médico
Desarrollado para el taller de Pipeline de MLOps + Docker
"""

from flask import Flask, render_template, request, jsonify
import json
import logging
from model import predict_medical_diagnosis

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)

# Configuración de la aplicación
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

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
                    'diagnosis': 'NO_ENFERMO | ENFERMEDAD_LEVE | ENFERMEDAD_AGUDA | ENFERMEDAD_CRONICA',
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
