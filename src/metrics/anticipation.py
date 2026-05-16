"""Anticipation-time metric for early micro-stop detection — numpy only."""

import numpy as np


def calcular_anticipaciones(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    max_anticipacion: int = 20,
) -> np.ndarray:
    """Per-event lookback distances of the earliest matching positive prediction.

    For every event index e where y_real[e] == 1, this scans the window
    [e - max_anticipacion, e] for positive predictions in y_pred and, if any
    exist, records (e - earliest_alert) as the anticipation in steps.

    Args:
        y_real: 1-D array of binary ground-truth labels.
        y_pred: 1-D array of binary predicted labels (same length as y_real).
        max_anticipacion: maximum look-back window in steps.

    Returns:
        np.ndarray of int anticipation distances, length equal to the number
        of events with at least one prediction inside the window. Empty if none.
    """
    eventos_reales = np.where(y_real == 1)[0]
    predicciones_positivas = np.where(y_pred == 1)[0]

    anticipaciones = []
    for evento in eventos_reales:
        candidatos = predicciones_positivas[
            (predicciones_positivas >= evento - max_anticipacion)
            & (predicciones_positivas <= evento)
        ]
        if len(candidatos) > 0:
            primera_alerta = candidatos[0]
            anticipaciones.append(int(evento - primera_alerta))

    return np.array(anticipaciones, dtype=int)


def calcular_anticipacion(
    y_det: np.ndarray,
    pred_det: np.ndarray,
    max_look_back: int = 20,
) -> float:
    """Mean of the SMALLEST positive-prediction lookback per event (closest alert).

    Distinct from calcular_anticipaciones: for each event index e where
    y_det[e] == 1, this scans look_back = 1, 2, ..., max_look_back and stops at
    the first index where pred_det[e - look_back] == 1, then averages those
    smallest-lookback values across events. Returns 0.0 if no event was
    matched (or there were no events).
    """
    micro_pos = np.where(y_det == 1)[0]
    if len(micro_pos) == 0:
        return 0.0

    anticipaciones = []
    for mp in micro_pos:
        for look_back in range(1, min(mp + 1, max_look_back + 1)):
            if pred_det[mp - look_back] == 1:
                anticipaciones.append(look_back)
                break

    return float(np.mean(anticipaciones)) if anticipaciones else 0.0
