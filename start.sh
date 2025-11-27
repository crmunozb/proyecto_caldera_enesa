#!/bin/bash

echo "===== Iniciando Sistema SCADA ENESA ====="

# Activar el entorno virtual
if [ -d "venv" ]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate
else
    echo "ERROR: No se encontró el entorno virtual 'venv'"
    exit 1
fi

# Ejecutar simulador en segundo plano
echo "Ejecutando simulador..."
python3 simulador/simulador.py &
SIM_PID=$!
echo "Simulador corriendo con PID $SIM_PID"

# Esperar unos segundos para asegurar que stream_data.csv se genere
sleep 2

# Ejecutar SCADA
echo "Ejecutando SCADA..."
streamlit run scada/scada.py 2>/dev/null


# Si cierras Streamlit, detenemos el simulador
echo "Cerrando simulador..."
kill $SIM_PID

echo "===== Sistema SCADA finalizado ====="
