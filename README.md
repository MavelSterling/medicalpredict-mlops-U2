
# medicalpredict-mlops-U2

Predicción simplificada de enfermedades en pacientes y preparación del repositorio para prácticas de MLOps (ramas, PRs, versionado y CI/CD con GitHub Actions).

---

## 👥 Integrantes del Proyecto

* **Felipe Guerra**
* **Mavelyn Sterling**

---

## 🌡️ Problema

Se requiere un servicio que **simule** el diagnóstico de una enfermedad a partir de datos básicos del paciente (edad, síntomas, condiciones). En esta unidad no entrenamos un modelo real; nos enfocamos en  organizar el repositorio y controlar cambios con Git.

---

## 🎯 Propósito del repositorio

1. **Estandarizar** la estructura del proyecto (código, documentación y empaquetado en Docker).
2. **Gestionar ramas y PRs** para integrar cambios controlados.

---

## 📋 Estructura del Proyecto

```


├── README.md                           # Este archivo
├── requirements.txt                    # Dependencias de Python
├── .gitignore                         # Archivos a excluir de Git
├── .venv/                             # Entorno virtual de Python
├── docs/                              # Documentación del pipeline
│   ├── pipeline_design.md            # Diseño del pipeline de MLOps
├── src/                              # Código fuente del servicio
│   ├── app.py                        # Aplicación Flask principal
│   ├── model.py                      # Función de diagnóstico médico
│   ├── requirements.txt              # Dependencias
│   └── templates/                    # Plantillas HTML
│       └── index.html               # Interfaz web
├── Dockerfile                       # Dockerfile 

```

---

## 🚀 Ejecución local

**Con Python:**

```bash
python -m venv .venv
source .venv/bin/activate         # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Predicción simulada (CLI semana 4)
python -m app.service predict --age 25 --symptoms mild_respiratory

# Ver estadísticas
python -m app.service stats
```

**Con Docker:**

```bash
docker build -t medicalpredict-mlops-u2:latest .
docker run --rm medicalpredict-mlops-u2:latest predict --age 25 --symptoms mild_respiratory
docker run --rm -v ${PWD}/data:/app/data medicalpredict-mlops-u2:latest stats
```

---

*Proyecto desarrollado por **Felipe Guerra** y **Mavelyn Sterling** para el taller de MLOps - **Maestría en Inteligencia Artificial Aplicada***
