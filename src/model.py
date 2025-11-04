"""
Módulo de diagnóstico médico para el sistema de MLOps
Desarrollado para el taller de Pipeline de MLOps + Docker
"""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDiagnosisModel:
    """
    Modelo de diagnóstico médico que simula la predicción de enfermedades
    basado en síntomas del paciente.

    Flujo:
    1. Se calcula un puntaje global de síntomas (overall_score).
    2. Se evalúa qué patrón de enfermedad es más probable (pattern_scores).
    3. Con ambos se estima severidad (determine_severity).
    4. Según severidad:
        - NO_ENFERMO / MOLESTIAS_LEVES:
            -> No mostramos diagnóstico tipo "infección respiratoria".
            -> Recomendaciones suaves (descanso / hidratación).
        - ENFERMEDAD_LEVE / ENFERMEDAD_AGUDA / ENFERMEDAD_CRONICA:
            -> Sí mostramos la condición más probable.
            -> Recomendaciones más estrictas.
    """

    def __init__(self):
        self.symptom_weights = self._initialize_symptom_weights()
        self.disease_patterns = self._initialize_disease_patterns()

    def _initialize_symptom_weights(self) -> Dict[str, float]:
        """Pesos de importancia de cada síntoma."""
        return {
            'fiebre': 0.8,
            'dolor_cabeza': 0.6,
            'nausea': 0.5,
            'fatiga': 0.4,
            'dolor_pecho': 0.9,
            'dificultad_respirar': 0.95,
            'dolor_abdominal': 0.7,
            'mareos': 0.5,
            'perdida_peso': 0.6,
            'tos': 0.6,
            'congestion_nasal': 0.3,
            'dolor_garganta': 0.4,
            'dolor_muscular': 0.4,
            'dolor_articular': 0.5,
            'erupcion_cutanea': 0.6,
            'sangrado': 0.8,
            'cambios_vision': 0.7,
            'confusion': 0.9,
            'convulsiones': 0.95,
            'dolor_espalda': 0.5
        }

    def _initialize_disease_patterns(self) -> Dict[str, List[str]]:
        """Asociación de enfermedades ↔ lista de síntomas típicos."""
        return {
            'infeccion_respiratoria': ['fiebre', 'tos', 'congestion_nasal', 'dolor_garganta'],
            'gastroenteritis': ['nausea', 'dolor_abdominal', 'fatiga'],
            'migrana': ['dolor_cabeza', 'nausea', 'mareos'],
            'ansiedad': ['dolor_pecho', 'dificultad_respirar', 'mareos', 'fatiga'],
            'diabetes': ['perdida_peso', 'fatiga', 'cambios_vision'],
            'hipertension': ['dolor_cabeza', 'mareos', 'dolor_pecho'],
            'artritis': ['dolor_articular', 'dolor_muscular', 'fatiga'],
            'enfermedad_cardiaca': ['dolor_pecho', 'dificultad_respirar', 'fatiga'],
            'enfermedad_renal': ['fatiga', 'nausea', 'dolor_espalda'],
            'enfermedad_hepatica': ['fatiga', 'nausea', 'dolor_abdominal', 'erupcion_cutanea'],
            'enfermedad_autoimmune': ['fatiga', 'dolor_articular', 'erupcion_cutanea', 'fiebre'],
            'cancer': ['perdida_peso', 'fatiga', 'dolor_abdominal', 'sangrado'],
            'enfermedad_neurologica': ['confusion', 'convulsiones', 'cambios_vision', 'mareos']
        }

    def calculate_symptom_score(self, symptoms: Dict[str, Union[float, int]]) -> float:
        """
        Calcula un score global de síntomas ponderado por importancia.
        Intensidad esperada: 0-10 por síntoma.
        Devuelve número entre 0 y 1.
        """
        total_score = 0.0
        total_weight = 0.0

        for symptom, intensity in symptoms.items():
            if symptom in self.symptom_weights:
                weight = self.symptom_weights[symptom]

                # ignorar síntomas ausentes (0)
                if intensity is None:
                    continue
                if intensity <= 0:
                    continue

                # normalizamos intensidad 0-10 -> 0-1
                normalized_intensity = min(max(float(intensity) / 10.0, 0.0), 1.0)

                total_score += normalized_intensity * weight
                total_weight += weight

        if total_weight == 0:
            # nadie reportó nada >0
            return 0.0

        return total_score / total_weight

    def detect_disease_patterns(self, symptoms: Dict[str, Union[float, int]]) -> Dict[str, float]:
        """
        Score por enfermedad según qué tanto coinciden los síntomas reportados
        con el patrón típico de esa enfermedad.
        """
        pattern_scores = {}

        for disease, pattern_symptoms in self.disease_patterns.items():
            score = 0.0

            for symptom in pattern_symptoms:
                if symptom in symptoms and symptoms[symptom] > 0:
                    intensity = min(max(symptoms[symptom] / 10.0, 0.0), 1.0)
                    weight = self.symptom_weights.get(symptom, 0.5)
                    score += intensity * weight

            # normalizamos por el largo del patrón esperado
            if len(pattern_symptoms) > 0:
                pattern_scores[disease] = score / len(pattern_symptoms)
            else:
                pattern_scores[disease] = 0.0

        return pattern_scores

    def determine_severity(
        self,
        overall_score: float,
        pattern_scores: Dict[str, float]
    ) -> Tuple[str, float]:
        """
        Determina severidad clínica de forma determinista y robusta.

        adjusted_score = promedio entre:
          - gravedad global de síntomas (overall_score),
          - el patrón de enfermedad más fuerte (max_pattern_score).

        Clasificación:
        NO_ENFERMO         adjusted_score < 0.15
        MOLESTIAS_LEVES    0.15 - 0.30
        ENFERMEDAD_LEVE    0.30 - 0.60
        ENFERMEDAD_AGUDA   0.60 - 0.80
        ENFERMEDAD_CRONICA >= 0.80
        """
        max_pattern_score = max(pattern_scores.values()) if pattern_scores else 0.0
        adjusted_score = (overall_score + max_pattern_score) / 2.0

        if adjusted_score < 0.15:
            severity_selected = 'NO_ENFERMO'
        elif adjusted_score < 0.30:
            severity_selected = 'MOLESTIAS_LEVES'
        elif adjusted_score < 0.60:
            severity_selected = 'ENFERMEDAD_LEVE'
        elif adjusted_score < 0.80:
            severity_selected = 'ENFERMEDAD_AGUDA'
        else:
            severity_selected = 'ENFERMEDAD_CRONICA'

        return severity_selected, adjusted_score

    def predict_diagnosis(
        self,
        symptoms: Dict[str, Union[float, int]]
    ) -> Dict[str, Union[str, float, Dict, bool]]:
        """
        Pipeline completo:
        - valida input
        - calcula puntaje de síntomas
        - detecta patrones
        - determina severidad
        - decide si mostrar diagnóstico específico
        - genera recomendaciones
        """
        try:
            # Validación mínima
            if not symptoms or len(symptoms) < 3:
                raise ValueError("Se requieren al menos 3 síntomas para el diagnóstico")

            # Score global de síntomas 
            overall_score = self.calculate_symptom_score(symptoms)

            # Coincidencia con patrones de enfermedad
            pattern_scores = self.detect_disease_patterns(symptoms)

            # Severidad clínica final
            severity, adjusted_score = self.determine_severity(overall_score, pattern_scores)

            # Enfermedad más probable (antes de filtrar)
            most_likely_disease, most_likely_score = ("ninguna", 0.0)
            if pattern_scores:
                most_likely_disease, most_likely_score = max(
                    pattern_scores.items(), key=lambda x: x[1]
                )

            # Reglas de coherencia con el front:
            # - NO_ENFERMO / MOLESTIAS_LEVES:
            #   * no mostramos condición específica
            # - ENFERMEDAD_LEVE / ENFERMEDAD_AGUDA / ENFERMEDAD_CRONICA:
            #   * sí mostramos condición probable
            if severity in ["NO_ENFERMO", "MOLESTIAS_LEVES"]:
                show_condition = False
                most_likely_disease = "ninguna"
                most_likely_score = 0.0
            else:
                show_condition = True

            recommendations = self._generate_recommendations(severity)

            result = {
                'diagnosis': severity,
                'confidence': round(overall_score, 3),          # qué tan fuertes son los síntomas reportados (>0)
                'severity_score': round(adjusted_score, 3),     # score combinado usado para clasificar
                'most_likely_condition': most_likely_disease,   # ej. 'enfermedad_cardiaca'
                'condition_confidence': round(most_likely_score, 3),
                'show_condition': show_condition,             
                'symptom_score': round(overall_score, 3),
                'pattern_scores': {k: round(v, 3) for k, v in pattern_scores.items()},
                'recommendations': recommendations,
                'input_symptoms': symptoms
            }

            logger.info(
                f"Diagnóstico generado: {severity} | score={adjusted_score:.3f} | "
                f"condición={most_likely_disease if show_condition else 'N/A'}"
            )
            return result

        except Exception as e:
            logger.error(f"Error en el diagnóstico: {str(e)}")
            return {
                'error': str(e),
                'diagnosis': 'ERROR',
                'confidence': 0.0,
                'show_condition': False,
                'recommendations': []
            }

    def _generate_recommendations(self, severity: str) -> List[str]:
        """
        Recomendaciones basadas en severidad clínica.
        NO_ENFERMO / MOLESTIAS_LEVES -> autocuidado básico.
        ENFERMEDAD_LEVE -> control en casa + considerar consulta.
        ENFERMEDAD_AGUDA -> atención médica pronto.
        ENFERMEDAD_CRONICA -> urgente.
        """
        if severity == 'NO_ENFERMO':
            return [
                "No hay señales de gravedad actuales",
                "Descansar adecuadamente",
                "Mantener buena hidratación (agua, líquidos claros)",
                "Observar si aparecen nuevos síntomas o si alguno empeora"
            ]

        if severity == 'MOLESTIAS_LEVES':
            return [
                "Molestias leves: reposo y buena hidratación",
                "Dormir bien y evitar sobreesfuerzos",
                "Usar analgésicos comunes si es necesario y no hay contraindicaciones",
                "Consultar con un profesional si los síntomas aumentan o duran más de 48h"
            ]

        if severity == 'ENFERMEDAD_LEVE':
            return [
                "Síntomas compatibles con un cuadro leve",
                "Mantener reposo y buena hidratación",
                "Monitorear temperatura y respiración",
                "Consultar a un profesional si persisten más de 48-72h o empeoran"
            ]

        if severity == 'ENFERMEDAD_AGUDA':
            return [
                "CUADRO DE CUIDADO MÉDICO RECOMENDADO",
                "Buscar valoración médica en las próximas 24 horas",
                "Monitorear signos vitales (fiebre alta, dificultad respiratoria)",
                "Evitar automedicación sin indicación profesional",
                "Acudir a urgencias si hay empeoramiento rápido"
            ]

        if severity == 'ENFERMEDAD_CRONICA':
            return [
                "ATENCIÓN MÉDICA URGENTE NECESARIA",
                "Buscar atención especializada inmediatamente",
                "Posible necesidad de intervención hospitalaria",
                "Seguimiento médico continuo recomendado"
            ]

        # fallback
        return [
            "Monitorear evolución de síntomas",
            "Buscar orientación médica ante cualquier duda"
        ]


# Instancia global reusable
diagnosis_model = MedicalDiagnosisModel()

def predict_medical_diagnosis(
    symptoms: Dict[str, Union[float, int]]
) -> Dict[str, Union[str, float, Dict]]:
    """
    Wrapper público.
    Ejemplo de entrada:
        {'fiebre': 8, 'tos': 6, 'mareos': 2, 'fatiga': 4}
    """
    return diagnosis_model.predict_diagnosis(symptoms)


# Ejemplo manual de prueba rápida
if __name__ == "__main__":
    example_symptoms = {
        'dolor_pecho': 10,
        'dificultad_respirar': 10,
        'tos': 10,
        'fatiga': 0,
        'fiebre': 0,
        'mareos': 0
    }

    result = predict_medical_diagnosis(example_symptoms)

    print("=== DIAGNÓSTICO MÉDICO ===")
    print(f"Diagnóstico / Severidad: {result['diagnosis']}")
    print(f"Score severidad: {result['severity_score']}")
    print(f"Confianza síntomas globales: {result['confidence']}")
    if result['show_condition']:
        print(f"Condición más probable: {result['most_likely_condition']}")
        print(f"Confianza condición: {result['condition_confidence']}")
    else:
        print("Condición más probable: (no aplica, sin enfermedad significativa)")
    print("\nRecomendaciones:")
    for rec in result['recommendations']:
        print(f"- {rec}")
