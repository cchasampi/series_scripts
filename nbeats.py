"""Modelo final N-BEATSx para radiación a 15 minutos (s=96).

Variables meteorológicas históricas: temperatura, humedad, precipitación y
viento promedio a 2 m. Variables futuras conocidas: ciclos diario y anual.
"""

from pathlib import Path
import argparse
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NBEATS, NBEATSx
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


FREQ = "15min"
S = 96
H = S
INPUT_SIZE = 7 * S
TEST_SIZE = S
MAX_STEPS = 250
SEED = 7

HIST_EXOG = ["temperatura", "humedad", "precipitacion", "viento_2m"]
FUTR_EXOG = ["sin_dia", "cos_dia", "sin_anio", "cos_anio"]
ORIGINALES = {
    "radiacion": "y",
    "veloc_de_viento_prom_2m": "viento_2m",
}


def agregar_calendario(df):
    slot = df["ds"].dt.hour * 4 + df["ds"].dt.minute // 15
    dia_anio = df["ds"].dt.dayofyear
    df["sin_dia"] = np.sin(2 * np.pi * slot / S)
    df["cos_dia"] = np.cos(2 * np.pi * slot / S)
    df["sin_anio"] = np.sin(2 * np.pi * dia_anio / 365.25)
    df["cos_anio"] = np.cos(2 * np.pi * dia_anio / 365.25)
    return df


def ultimo_tramo_completo(df, columnas):
    valido = df[columnas].notna().all(axis=1).to_numpy()
    inicios = np.flatnonzero(valido & np.r_[True, ~valido[:-1]])
    finales = np.flatnonzero(valido & np.r_[~valido[1:], True])
    if not len(inicios):
        raise ValueError("No existe un tramo continuo con todas las variables.")
    return df.iloc[inicios[-1]:finales[-1] + 1].copy()


def preparar_datos(ruta_csv, max_dias):
    columnas = ["fecha", "radiacion", "temperatura", "humedad", "precipitacion",
                "veloc_de_viento_prom_2m"]
    df = pd.read_csv(ruta_csv, sep=";", usecols=columnas, low_memory=False)
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    for c in columnas[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")
    df = df.drop_duplicates("fecha", keep="last").set_index("fecha")
    df.loc[df["radiacion"] < 0, "radiacion"] = np.nan
    df.loc[df["precipitacion"] < 0, "precipitacion"] = np.nan
    df.loc[df["veloc_de_viento_prom_2m"] < 0, "veloc_de_viento_prom_2m"] = np.nan

    df = df.resample(FREQ).mean().rename(columns=ORIGINALES)
    for c in ["y", "temperatura", "humedad", "viento_2m"]:
        df[c] = df[c].interpolate("time", limit=8, limit_area="inside")
    # En un tramo válido, la ausencia de registro de lluvia se considera lluvia nula.
    df["precipitacion"] = df["precipitacion"].fillna(0)
    df = ultimo_tramo_completo(df, ["y"] + HIST_EXOG)
    if max_dias > 0:
        df = df.tail(max_dias * S)
    minimo = INPUT_SIZE + TEST_SIZE + H
    if len(df) < minimo:
        raise ValueError(f"Hay {len(df)} registros continuos y se necesitan {minimo}.")

    df = df.reset_index().rename(columns={"fecha": "ds"})
    df["unique_id"] = "radiacion_catamarca"
    return agregar_calendario(df)


def metricas(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    diurno = np.abs(y) > 20
    mape = np.mean(np.abs((y[diurno] - pred[diurno]) / y[diurno])) * 100 if diurno.any() else np.nan
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAPE_diurno_pct": float(mape),
        "R2": float(r2_score(y, pred)),
        "Sesgo": float(np.mean(pred - y)),
    }


def crear_modelos(max_steps):
    comun = dict(h=H, input_size=INPUT_SIZE, max_steps=max_steps,
                 loss=MAE(), scaler_type="standard", random_seed=SEED)
    exog = dict(
        stack_types=["identity", "trend", "seasonality", "exogenous"],
        n_blocks=[1, 1, 1, 1],
        mlp_units=4 * [[256, 256]],
    )
    return [
        NBEATS(**comun, alias="NBEATS_base"),
        NBEATSx(**comun, **exog, futr_exog_list=FUTR_EXOG,
                alias="NBEATSx_calendario"),
        NBEATSx(**comun, hist_exog_list=HIST_EXOG, futr_exog_list=FUTR_EXOG,
                alias="NBEATSx_meteo_calendario", **exog),
    ]


def identificar_salidas(pred):
    esperadas = ["NBEATS_base", "NBEATSx_calendario", "NBEATSx_meteo_calendario"]
    faltantes = [c for c in esperadas if c not in pred.columns]
    if faltantes:
        raise RuntimeError(f"Faltan salidas {faltantes}. Columnas: {list(pred.columns)}")
    return tuple(esperadas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="unificada.csv")
    ap.add_argument("--salida", default="resultados_nbeatsx_final_s96")
    ap.add_argument("--max-dias", type=int, default=180)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    np.random.seed(SEED)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    datos = preparar_datos(args.csv, args.max_dias)
    train, test = datos.iloc[:-TEST_SIZE].copy(), datos.iloc[-TEST_SIZE:].copy()
    print(f"Registros: {len(datos):,}; train: {len(train):,}; test: {len(test):,}")

    nf = NeuralForecast(models=crear_modelos(args.max_steps), freq=FREQ)
    nf.fit(df=train)
    futr_test = test[["unique_id", "ds"] + FUTR_EXOG]
    pred_modelos = nf.predict(futr_df=futr_test).reset_index(drop=True)
    c_base, c_cal, c_final = identificar_salidas(pred_modelos)
    nombres = {
        c_base: "N-BEATS",
        c_cal: "N-BEATSx calendario",
        c_final: "N-BEATSx meteorología histórica + calendario",
    }
    pred = pred_modelos.merge(test[["unique_id", "ds", "y"]], on=["unique_id", "ds"])

    tabla = pd.DataFrame([
        {"Modelo": etiqueta, **metricas(pred["y"], pred[col])}
        for col, etiqueta in nombres.items()
    ]).sort_values("RMSE")
    tabla.to_csv(salida / "metricas_testing.csv", index=False)
    pred.rename(columns=nombres).to_csv(salida / "pronosticos_testing.csv", index=False)
    print("\n", tabla.to_string(index=False))

    plt.figure(figsize=(15, 6))
    plt.plot(pred["ds"], pred["y"], color="black", lw=2, label="Observado")
    for col, etiqueta in nombres.items():
        plt.plot(pred["ds"], pred[col], "--", label=etiqueta)
    plt.title("Comparación N-BEATS/N-BEATSx para radiación con s=96")
    plt.xlabel("Fecha"); plt.ylabel("Radiación (W/m²)")
    plt.grid(alpha=0.2); plt.legend(); plt.tight_layout()
    plt.savefig(salida / "comparacion_testing.png", dpi=180, bbox_inches="tight")
    plt.close()

    residuos = (pred["y"] - pred[c_final]).dropna()
    max_lag = min(S - 1, len(residuos) - 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(pred.loc[residuos.index, "ds"], residuos)
    axes[0, 0].axhline(0, color="black"); axes[0, 0].set_title("Residuos")
    axes[0, 1].hist(residuos, bins=25, edgecolor="white"); axes[0, 1].set_title("Distribución")
    plot_acf(residuos, lags=max_lag, zero=False, ax=axes[1, 0])
    axes[1, 0].set_title(f"FAC hasta el rezago {max_lag}")
    axes[1, 1].scatter(pred.loc[residuos.index, c_final], residuos, s=12)
    axes[1, 1].axhline(0, color="black"); axes[1, 1].set_title("Residuos vs pronóstico")
    plt.tight_layout(); plt.savefig(salida / "diagnostico_residuos.png", dpi=180)
    plt.close()
    lags_lb = [x for x in [1, 24, 48, 96] if x < len(residuos)]
    acorr_ljungbox(residuos, lags=lags_lb, return_df=True).to_csv(salida / "ljung_box.csv")

    # Modelo final completo. Solo necesita el calendario futuro; las variables
    # meteorológicas son históricas y ya están contenidas en 'datos'.
    modelo_final = NBEATSx(
        h=H, input_size=INPUT_SIZE, max_steps=args.max_steps, loss=MAE(),
        scaler_type="standard", hist_exog_list=HIST_EXOG,
        futr_exog_list=FUTR_EXOG, random_seed=SEED, alias="NBEATSx_final",
        stack_types=["identity", "trend", "seasonality", "exogenous"],
        n_blocks=[1, 1, 1, 1], mlp_units=4 * [[256, 256]],
    )
    nf_final = NeuralForecast(models=[modelo_final], freq=FREQ)
    nf_final.fit(df=datos)
    fechas = pd.date_range(datos["ds"].max() + pd.Timedelta(minutes=15), periods=H, freq=FREQ)
    futuro = agregar_calendario(pd.DataFrame({"unique_id": "radiacion_catamarca", "ds": fechas}))
    pron = nf_final.predict(futr_df=futuro[["unique_id", "ds"] + FUTR_EXOG]).reset_index(drop=True)
    pron = pron.merge(futuro[["unique_id", "ds"] + FUTR_EXOG], on=["unique_id", "ds"])
    c_pron = "NBEATSx_final"
    pron[c_pron] = pron[c_pron].clip(lower=0)
    pron.to_csv(salida / "pronostico_futuro_24h.csv", index=False)

    plt.figure(figsize=(15, 6))
    hist = datos.tail(2 * S)
    plt.plot(hist["ds"], hist["y"], color="0.45", label="Observado")
    plt.plot(pron["ds"], pron[c_pron], "--", lw=2, label="Pronóstico final")
    plt.axvline(datos["ds"].max(), color="black", ls=":")
    plt.title("Pronóstico final de radiación a 24 horas")
    plt.xlabel("Fecha"); plt.ylabel("Radiación (W/m²)")
    plt.grid(alpha=0.2); plt.legend(); plt.tight_layout()
    plt.savefig(salida / "pronostico_futuro.png", dpi=180, bbox_inches="tight")
    plt.close()

    resumen = {
        "objetivo": "radiacion", "frecuencia": FREQ, "s": S,
        "input_size": INPUT_SIZE, "horizonte": H,
        "historicas": HIST_EXOG, "futuras_conocidas": FUTR_EXOG,
        "registros": len(datos),
        "criterio_variables": "Granger/relación física, disponibilidad y baja redundancia",
    }
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados guardados en: {salida.resolve()}")


if __name__ == "__main__":
    main()
