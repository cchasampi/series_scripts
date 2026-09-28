"""NeuralProphet semanal mejorado para radiación solar.

Compara distintas ventanas autorregresivas e incorpora temperatura, humedad,
precipitación y viento a 2 m como regresores. Selecciona por RMSE de testing.
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
from neuralprophet import NeuralProphet, set_log_level
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


FREQ = "W-SUN"
TEST_SIZE = 52
HORIZON = 26
EPOCHS = 250
SEED = 7
REGRESORES = ["temperatura", "humedad", "precipitacion", "viento_2m"]


def preparar_datos(ruta):
    cols = ["fecha", "radiacion", "temperatura", "humedad",
            "precipitacion_acum_diaria", "veloc_de_viento_prom_2m"]
    df = pd.read_csv(ruta, sep=";", usecols=cols, low_memory=False)
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")
    df = df.drop_duplicates("fecha", keep="last").set_index("fecha")
    df.loc[df["radiacion"] < 0, "radiacion"] = np.nan
    df.loc[df["veloc_de_viento_prom_2m"] < 0, "veloc_de_viento_prom_2m"] = np.nan

    # Precipitación: primero un acumulado por día y luego suma semanal.
    lluvia = df["precipitacion_acum_diaria"].resample("D").max()
    lluvia = lluvia.resample(FREQ).sum(min_count=1).rename("precipitacion")
    semanal = pd.concat([
        df["radiacion"].resample(FREQ).mean().rename("y"),
        df["temperatura"].resample(FREQ).mean().rename("temperatura"),
        df["humedad"].resample(FREQ).mean().rename("humedad"),
        lluvia,
        df["veloc_de_viento_prom_2m"].resample(FREQ).mean().rename("viento_2m"),
    ], axis=1)
    semanal["y"] = semanal["y"].interpolate(limit=2, limit_area="inside")
    for c in ["temperatura", "humedad", "viento_2m"]:
        semanal[c] = semanal[c].interpolate(limit=2, limit_area="inside")
    semanal["precipitacion"] = semanal["precipitacion"].fillna(0).clip(lower=0)
    semanal = semanal.dropna()
    return semanal.reset_index().rename(columns={"fecha": "ds"})


def metricas(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    eps = 1e-8
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "MSE": float(mean_squared_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAPE_pct": float(np.mean(np.abs((y-p)/np.maximum(np.abs(y), eps))) * 100),
        "sMAPE_pct": float(np.mean(2*np.abs(y-p)/np.maximum(np.abs(y)+np.abs(p), eps))*100),
        "R2": float(r2_score(y, p)),
        "Sesgo": float(np.mean(p-y)),
    }


def crear_modelo(n_lags, epochs, con_meteo):
    m = NeuralProphet(
        growth="linear",
        n_changepoints=15,
        yearly_seasonality=10,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        n_lags=n_lags,
        n_forecasts=1,
        ar_layers=[32, 16] if n_lags else [],
        epochs=epochs,
        batch_size=32,
        learning_rate=None,
        loss_func="Huber",
        normalize="standardize",
        collect_metrics=True,
    )
    if con_meteo:
        for reg in REGRESORES:
            m.add_future_regressor(reg, normalize=True)
    return m


def evaluar(modelo, train, test, n_lags, con_meteo):
    columnas = ["ds", "y"] + (REGRESORES if con_meteo else [])
    entrada = pd.concat([train.tail(n_lags), test], ignore_index=True)[columnas]
    pred = modelo.predict(entrada)
    pred = pred[pred["ds"].isin(test["ds"])][["ds", "y", "yhat1"]]
    return pred.dropna(subset=["y", "yhat1"])


def escenario_estacional(datos, periodos):
    """Regresores futuros iguales a las mismas semanas del año anterior."""
    fechas = pd.date_range(datos["ds"].max() + pd.offsets.Week(weekday=6),
                           periods=periodos, freq=FREQ)
    patron = datos[REGRESORES].tail(52).reset_index(drop=True)
    escenario = patron.iloc[:periodos].copy()
    escenario.insert(0, "ds", fechas)
    return escenario


def pronostico_recursivo(modelo, historia, escenario, n_lags, con_meteo):
    columnas = ["ds", "y"] + (REGRESORES if con_meteo else [])
    hist = historia[columnas].copy().reset_index(drop=True)
    filas = []
    for _, esc in escenario.iterrows():
        reg_df = pd.DataFrame([esc[["ds"] + REGRESORES]]) if con_meteo else None
        futuro = modelo.make_future_dataframe(
            hist, regressors_df=reg_df, periods=1, n_historic_predictions=False
        )
        pred = modelo.predict(futuro)
        candidatos = pred.dropna(subset=["yhat1"])
        candidatos = candidatos[candidatos["ds"] > hist["ds"].max()]
        if candidatos.empty:
            raise RuntimeError(f"No se generó pronóstico para {esc['ds']}.")
        valor = max(0.0, float(candidatos.iloc[-1]["yhat1"]))
        nueva = {"ds": pd.to_datetime(esc["ds"]), "y": valor}
        if con_meteo:
            nueva.update({r: float(esc[r]) for r in REGRESORES})
        hist = pd.concat([hist, pd.DataFrame([nueva])], ignore_index=True)
        filas.append({"ds": nueva["ds"], "yhat1": valor, **({r:nueva[r] for r in REGRESORES} if con_meteo else {})})
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="unificada.csv")
    ap.add_argument("--salida", default="resultados_neuralprophet_mejorado")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    np.random.seed(SEED)
    set_log_level("ERROR")
    salida = Path(args.salida); salida.mkdir(parents=True, exist_ok=True)

    datos = preparar_datos(args.csv)
    train, test = datos.iloc[:-TEST_SIZE].copy(), datos.iloc[-TEST_SIZE:].copy()
    configuraciones = [
        {"nombre":"NP_AR52", "lags":52, "meteo":False},
        {"nombre":"NP_Meteo_AR26", "lags":26, "meteo":True},
        {"nombre":"NP_Meteo_AR52", "lags":52, "meteo":True},
        {"nombre":"NP_Meteo_AR104", "lags":104, "meteo":True},
    ]

    filas, preds = [], {}
    for cfg in configuraciones:
        print(f"Entrenando {cfg['nombre']}...")
        m = crear_modelo(cfg["lags"], args.epochs, cfg["meteo"])
        columnas_fit = ["ds", "y"] + (REGRESORES if cfg["meteo"] else [])
        m.fit(train[columnas_fit].copy(), freq=FREQ, progress=None)
        p = evaluar(m, train, test, cfg["lags"], cfg["meteo"])
        filas.append({"Modelo":cfg["nombre"], **metricas(p["y"],p["yhat1"])})
        preds[cfg["nombre"]] = p.rename(columns={"yhat1":cfg["nombre"]})[["ds",cfg["nombre"]]]

    resultados = pd.DataFrame(filas).sort_values("RMSE").reset_index(drop=True)
    comparacion = test[["ds","y"]].copy()
    for nombre,p in preds.items(): comparacion = comparacion.merge(p,on="ds",how="left")
    comparacion["Ingenuo_estacional_52"] = datos["y"].shift(52).iloc[-TEST_SIZE:].to_numpy()
    resultados = pd.concat([resultados,pd.DataFrame([{
        "Modelo":"Ingenuo_estacional_52", **metricas(comparacion["y"],comparacion["Ingenuo_estacional_52"])
    }])],ignore_index=True).sort_values("RMSE")
    resultados.to_csv(salida/"metricas_modelos.csv",index=False)
    comparacion.to_csv(salida/"pronosticos_testing.csv",index=False)
    print("\n",resultados.to_string(index=False))

    mejor = resultados[resultados["Modelo"].str.startswith("NP_")].iloc[0]["Modelo"]
    cfg = next(x for x in configuraciones if x["nombre"]==mejor)
    plt.figure(figsize=(14,6))
    plt.plot(train.tail(104)["ds"],train.tail(104)["y"],color="0.6",label="Historial")
    plt.plot(comparacion["ds"],comparacion["y"],color="black",label="Observado")
    plt.plot(comparacion["ds"],comparacion[mejor],"--",lw=2,label=mejor)
    plt.plot(comparacion["ds"],comparacion["Ingenuo_estacional_52"],":",label="Ingenuo estacional")
    plt.axvline(test["ds"].iloc[0],color="0.3",ls=":")
    plt.title("NeuralProphet mejorado: radiación semanal")
    plt.xlabel("Fecha");plt.ylabel("Radiación media semanal (W/m²)")
    plt.grid(alpha=.2);plt.legend();plt.tight_layout()
    plt.savefig(salida/"comparacion_testing.png",dpi=180,bbox_inches="tight");plt.close()

    residuos=(comparacion["y"]-comparacion[mejor]).dropna()
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(comparacion.loc[residuos.index,"ds"],residuos);ax[0].axhline(0,color="black")
    plot_acf(residuos,lags=min(26,len(residuos)//2-1),zero=False,ax=ax[1])
    plt.tight_layout();plt.savefig(salida/"diagnostico_residuos.png",dpi=180);plt.close()
    lb_lags=[x for x in [1,13,26] if x<len(residuos)]
    acorr_ljungbox(residuos,lags=lb_lags,return_df=True).to_csv(salida/"ljung_box.csv")

    print(f"\nReentrenando modelo final: {mejor}")
    final=crear_modelo(cfg["lags"],args.epochs,cfg["meteo"])
    columnas_final = ["ds", "y"] + (REGRESORES if cfg["meteo"] else [])
    final.fit(datos[columnas_final].copy(),freq=FREQ,progress=None)
    escenario=escenario_estacional(datos,HORIZON)
    futuro=pronostico_recursivo(final,datos,escenario,cfg["lags"],cfg["meteo"])
    futuro.to_csv(salida/"pronostico_futuro_26_semanas.csv",index=False)
    plt.figure(figsize=(14,6));hist=datos.tail(104)
    plt.plot(hist["ds"],hist["y"],color="0.5",label="Observado")
    plt.plot(futuro["ds"],futuro["yhat1"],"--",lw=2,label="Pronóstico")
    plt.axvline(datos["ds"].max(),color="black",ls=":")
    plt.title(f"Pronóstico de radiación a 26 semanas: {mejor}")
    plt.xlabel("Fecha");plt.ylabel("Radiación media semanal (W/m²)")
    plt.grid(alpha=.2);plt.legend();plt.tight_layout()
    plt.savefig(salida/"pronostico_futuro.png",dpi=180,bbox_inches="tight");plt.close()

    resumen={"modelo_final":mejor,"lags":cfg["lags"],"usa_meteorologia":cfg["meteo"],
             "regresores":REGRESORES if cfg["meteo"] else [],"horizonte":HORIZON,
             "nota":"Pronóstico condicionado al escenario estacional de regresores."}
    (salida/"resumen.json").write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Resultados guardados en: {salida.resolve()}")


if __name__=="__main__": main()
