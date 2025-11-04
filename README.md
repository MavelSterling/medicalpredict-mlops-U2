# medicalpredict-mlops-U2

Predicción simplificada de enfermedades en pacientes y preparación del repositorio para prácticas de MLOps (ramas, PRs, versionado y CI/CD con GitHub Actions).

---

## 👥 Integrantes del Proyecto

* **Felipe Guerra**
* **Mavelyn Sterling**

---

## 🌡️ Problema

Se requiere un servicio que simule el diagnóstico de una enfermedad a partir de datos básicos del paciente (edad, síntomas, condiciones). En esta fase no se entrena un modelo real; el enfoque está en la organización del repositorio y el control de cambios con Git.

---

## 🎯 Propósito del repositorio

1. **Estandarizar** la estructura del proyecto (código, documentación y empaquetado en Docker).
2. **Gestionar ramas y PRs** para integrar cambios controlados.

---

## 🗂️ Estructura del proyecto

.
├── README.md                    # Este archivo
├── requirements.txt               # Dependencias de Python
├── .gitignore                       	 # Archivos a excluir de Git
├── src/                             	 # Código fuente del servicio
│   ├── app.py                        # Aplicación Flask principal
│   ├── model.py                    # Función de diagnóstico médico
│   ├── app.py                        # Aplicación Flask principal
│   └── templates/                  # Plantillas HTML
│       └── index.html              # Interfaz web
└── Dockerfile                        # Dockerfile

---

## 🚀 Inicio Rápido

### Prerrequisitos

- Docker instalado
- Python 3.8+ (para ejecución sin docker)

### Ejecución sin Docker

1. **Crear entorno virtual:**

```bash
python -m venv .venv
```

2. **Activar entorno virtual:**

```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

4. **Ejecutar aplicación:**

```bash
python src/app.py
```

### Construcción y Ejecución con Docker

1. **Construir la imagen Docker:**

```bash
docker build -t medical-diagnosis-service .
```

2. **Ejecutar el contenedor:**

```bash
docker run -p 5000:5000 medical-diagnosis-service
```

3. **Acceder al servicio:**
   - Interfaz web: http://localhost:5000
   - Reporte HTML: http://localhost:5000/report
   - Reporte JSON: http://localhost:5000/api/report

---

## 🏥 Servicio de Diagnóstico

El servicio permite a los médicos ingresar síntomas del paciente y obtener un diagnóstico en tiempo real con los siguientes estados (5 categorías):

- **NO ENFERMO**: Paciente sin indicios de enfermedad o con molestias muy leves
- **ENFERMEDAD LEVE**: Síntomas leves que requieren observación
- **ENFERMEDAD AGUDA**: Condición que requiere atención inmediata
- **ENFERMEDAD CRÓNICA**: Condición de larga duración que requiere tratamiento continuo
- **ENFERMEDAD TERMINAL**: Emergencia médica inmediata

---

## 📈 Reporte de Predicciones

Se registra cada predicción realizada en un log local y se expone un reporte con estadísticas básicas:

- Número total de predicciones realizadas por categoría.
- Últimas 5 predicciones realizadas.
- Fecha de la última predicción.

Endpoints:

- `GET /report`: Vista HTML con el reporte para médicos.
- `GET /api/report`: Respuesta JSON con las estadísticas.

Notas:
- El log se guarda en `src/data/predictions_log.jsonl` (1 predicción por línea en formato JSON).
- El almacenamiento es local al contenedor/proceso. Para persistencia entre despliegues, monte un volumen o integre una base de datos.

## 🧪 Casos de Uso

A continuación, se muestran algunos ejemplos de casos de uso:

- Nota: para evaluar correctamente, se deben ingresar mínimo 3 síntomas por paciente.
- `fatiga=2`, `dolor_muscular=1`, `mareos=1` → Diagnóstico esperado: **NO ENFERMO**
- `fiebre=3`, `dolor_cabeza=3`, `dificultad_respirar=5` → Diagnóstico esperado: **ENFERMEDAD LEVE**
- `fiebre=10`, `dolor_pecho=8`, `dificultad_respirar=7` → Diagnóstico esperado: **ENFERMEDAD LEVE**
- `dolor_pecho=7`, `dificultad_respirar=9`, `fatiga=8` → Diagnóstico esperado: **ENFERMEDAD AGUDA**
- `dolor_pecho=10`, `Tos=10`, `dificultad_respirar=10` → Diagnóstico esperado: **ENFERMEDAD CRÓNICA**

---

## 🔧 Tecnologías Utilizadas

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Containerización**: Docker
- **ML**: Scikit-learn, Pandas, NumPy

---

*Proyecto desarrollado por **Felipe Guerra** y **Mavelyn Sterling** para el taller de MLOps - **Maestría en Inteligencia Artificial Aplicada***
