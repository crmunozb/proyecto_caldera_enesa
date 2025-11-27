# Sistema SCADA en Streamlit para Visualización en Tiempo Real  
**Proyecto Guiado — Facultad de Ingeniería, Universidad de Concepción**

Este repositorio contiene el desarrollo correspondiente a la **Fase 2** del proyecto orientado a la implementación de un **sistema SCADA (Supervisory Control and Data Acquisition)** utilizando **Python** y **Streamlit**, con el objetivo de visualizar en tiempo real variables operacionales asociadas al funcionamiento de la caldera ENESA.

El sistema se alimenta de un **simulador de datos en flujo continuo**, el cual envía una nueva medición cada 10 segundos, emulando el comportamiento de sensores industriales.

---

## 🧩 Objetivos del Proyecto

- Implementar un **simulador de datos** basado en el dataset sintético generado mediante CTGAN en la Fase 1.  
- Desarrollar un **dashboard SCADA en Streamlit** capaz de actualizarse automáticamente.  
- Visualizar las principales variables operacionales:
  - Temperatura  
  - Oxígeno  
  - Humedad  
  - Material particulado  
  - Flujos volumétricos (húmedo y seco)  
  - Presión  
- Construir una arquitectura modular y extensible para integrar datos reales y análisis avanzados en fases posteriores.

---

## 🛠️ Arquitectura del Sistema

```
+----------------------+        +----------------------+        +--------------------------+
|   Datos Sintéticos   |        |     Simulador        |        |           SCADA           |
|     (CTGAN Fase 1)   | -----> | (stream_data.csv)    | -----> |    Dashboard Streamlit    |
+----------------------+        +----------------------+        +--------------------------+
```

- El **simulador** avanza fila por fila del dataset sintético.  
- Cada 10 segundos escribe una nueva medición en `stream_data.csv`.  
- El **SCADA** lee continuamente este archivo y actualiza los gráficos en tiempo real.

---

## 📦 Requisitos

- Python 3.9+
- Librerías necesarias:
  - `streamlit`
  - `pandas`
  - `plotly`
  - `streamlit-autorefresh`

Instalación recomendada:

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución

### 1. Iniciar el simulador
```bash
python simulador.py
```

Esto generará el archivo `stream_data.csv` e irá actualizándolo automáticamente.

### 2. Ejecutar el sistema SCADA
```bash
streamlit run scada.py
```

El dashboard se abrirá en el navegador predeterminado.  
Los gráficos se actualizan cada 10 segundos.

---

## 📈 Variables Visualizadas

El SCADA muestra en tiempo real las siguientes variables del proceso:

- Temperatura de gases de salida (°C)
- Concentración de oxígeno (%)
- Humedad (%)
- Material particulado (mg/m³)
- Flujo húmedo (m³/min)
- Flujo seco (Nm³/min)
- Presión (atm)

Todas estas variables se representan mediante gráficos interactivos desarrollados con **Plotly**.

---

## 🗂️ Estructura del Repositorio

```
├── simulador.py          # Simulador de datos en flujo continuo
├── scada.py              # Dashboard SCADA en Streamlit
├── stream_data.csv       # Archivo actualizado periódicamente por el simulador
├── data/                 # Dataset sintético utilizado en la Fase 1
├── img/                  # Gráficos o capturas exportadas
└── README.md             # Documentación del proyecto
```

---

## 📊 Capturas del Sistema

Las capturas completas del sistema SCADA, incluyendo visualizaciones de todas las variables operacionales (temperatura, oxígeno, humedad, flujos, material particulado y presión), se encuentran disponibles en el siguiente documento:

📄 **[SCADA - Caldera ENESA.pdf](reports/SCADA%20-%20Caldera%20ENESA.pdf)**

Este archivo incluye las imágenes exportadas directamente desde el dashboard desarrollado en Streamlit.


## 🔮 Próximos Pasos

- Incorporar límites operativos dinámicos con alertas visuales.  
- Añadir indicadores industriales (gauges, semáforos, barras de estado).  
- Estudiar integración con sensores reales de la caldera ENESA.  
- Desarrollar un módulo para calcular la eficiencia energética del sistema.  
- Implementar algoritmos de detección temprana de anomalías.  
- Incorporar modelos predictivos para anticipar el comportamiento de variables críticas.

---

## 📄 Licencia

Proyecto elaborado con fines académicos para la asignatura *Proyecto Computacional Guiado II*.

---

## 👤 Autor

**Cristóbal Muñoz Barrios**  
Proyecto Computacional Guiado II — Ingeniería Civil Informática UdeC  
Universidad de Concepción  
Año: 2025
