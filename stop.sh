#!/bin/bash

echo "===== Deteniendo Sistema SCADA ENESA ====="

# -----------------------------
# DETENER SIMULADOR
# -----------------------------
SIM_PIDS=$(pgrep -f "simulador/simulador.py")

if [ -n "$SIM_PIDS" ]; then
    echo "Deteniendo simulador (PID: $SIM_PIDS)..."
    kill $SIM_PIDS
    sleep 1

    # Verificar si sigue vivo
    if pgrep -f "simulador/simulador.py" > /dev/null; then
        echo "Simulador aún activo. Forzando cierre..."
        kill -9 $SIM_PIDS
    fi

    echo "[OK] Simulador detenido."
else
    echo "[INFO] No se encontró el simulador en ejecución."
fi


# -----------------------------
# DETENER STREAMLIT (SCADA)
# -----------------------------
SCADA_PIDS=$(pgrep -f "streamlit run scada.py")

if [ -n "$SCADA_PIDS" ]; then
    echo "Deteniendo SCADA Streamlit (PID: $SCADA_PIDS)..."
    kill $SCADA_PIDS
    sleep 1

    if pgrep -f "streamlit run scada.py" > /dev/null; then
        echo "SCADA aún activo. Forzando cierre..."
        kill -9 $SCADA_PIDS
    fi

    echo "[OK] SCADA detenido."
else
    echo "[INFO] No se encontró ninguna instancia de Streamlit ejecutándose."
fi


echo "===== Sistema SCADA detenido correctamente ====="
