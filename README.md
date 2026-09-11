# Predicción de Riesgo Materno con Datos de Dispositivos IoT

Trabajo práctico final — *Introducción al Aprendizaje Automático*, Licenciatura en Ciencia de Datos (UNSAM), 1er cuatrimestre de 2026. Realizado en grupo.

## Contexto y problema

Una organización sin fines de lucro provee dispositivos IoT wearables a hospitales públicos de zonas rurales de Bangladesh. Estos dispositivos miden continuamente los signos vitales de mujeres embarazadas. El objetivo es que el sistema detecte automáticamente a las pacientes de alto riesgo y alerte al hospital para priorizar su atención, optimizando recursos escasos.

**Pregunta central:** ¿puede un sistema IoT con modelos de Machine Learning optimizar la detección temprana de riesgo materno en hospitales públicos de bajos recursos?

> El escenario de la ONG es un marco narrativo construido por el grupo para darle sentido de aplicación real al ejercicio; el dataset y los resultados son reales.

## Dataset

[Maternal Health Risk](https://archive.ics.uci.edu/dataset/863/maternal+health+risk) — UCI Machine Learning Repository (ID 863), recolectado entre 2018 y 2020 en 6 hospitales de Dhaka y Khulna (Bangladesh), combinando sensores IoT (Arduino) y registros clínicos. El nivel de riesgo fue etiquetado por médicos según umbrales clínicos estándar.

- 1014 pacientes, sin valores faltantes
- 6 features numéricas: `Age`, `SystolicBP`, `DiastolicBP`, `BS` (glucosa), `BodyTemp`, `HeartRate`
- Target original de 3 clases: low / mid / high risk (40% / 33% / 27%)

## Preprocesamiento

- **Binarización del target:** se fusionaron `low` y `mid` risk en una sola clase, ya que el análisis exploratorio mostró que ambas son prácticamente indistinguibles en las variables más predictivas (`SystolicBP` y `BS`). Resultado: 73.2% (low/mid) vs. 26.8% (high risk).
- **Desbalance de clases:** manejado con `class_weight='balanced'` en ambos modelos, sin over/undersampling.
- **Split:** 80/20 estratificado (811 train / 203 test).
- **Escalado:** no aplicado — ambos modelos son basados en árboles, invariantes a la escala.

## Métricas

| Métrica | Por qué se usó |
|---|---|
| **Recall (high risk)** — métrica principal | En este contexto, un falso negativo (no alertar a una paciente de alto riesgo) es el peor error posible |
| F1-macro | Controla que el modelo no dispare demasiadas falsas alarmas que saturen al hospital |
| AUC-ROC | Permite analizar el trade-off detección/falsas alarmas y elegir el umbral óptimo |

*Accuracy no se usó por ser poco informativa frente al desbalance de clases.*

## Modelos

**Benchmark (reglas clínicas OMS)** — sin ML: `SystolicBP ≥ 140` o `BS ≥ 7.8` → high risk.

**Árbol de Decisión** — `max_depth` optimizado por validación cruzada (k=5).

**Random Forest** — `GridSearchCV` (k=5, scoring='recall') sobre `n_estimators`, `max_depth`, `min_samples_leaf`.

### Resultados

| Modelo | Recall (high risk) | F1-macro | AUC-ROC |
|---|---|---|---|
| Benchmark (test) | 0.907 | 0.776 | — |
| Árbol de Decisión (CV) | 0.899 ± 0.034 | 0.912 ± 0.013 | 0.938 ± 0.013 |
| Random Forest (CV) | 0.908 ± 0.032 | 0.890 ± 0.030 | 0.963 ± 0.013 |
| **Random Forest (test, umbral óptimo 0.634)** | **0.907** | **0.902** | **0.964** |

**Modelo elegido:** Random Forest, por su mejor Recall en validación cruzada, mayor AUC-ROC y resultados en test consistentes con CV (buena generalización). El umbral de decisión (0.634) se ajustó maximizando `TPR - FPR` sobre la curva ROC.

**Importancia de features:** `BS` (0.41) y `SystolicBP` (0.22) concentran más del 60% de la importancia total; `Age` y `HeartRate` aportan poco.

## Conclusiones

- El benchmark de reglas simples ya detecta el 90.7% de los casos de alto riesgo — un punto de partida sorprendentemente efectivo.
- El aporte real del Machine Learning no fue mejorar el Recall (ya estaba cerca del techo), sino **reducir drásticamente las falsas alarmas**: F1-macro pasó de 0.776 a 0.902.
- `BS` y `SystolicBP` son los predictores dominantes, lo que valida retroactivamente las reglas clínicas usadas en el benchmark.

## Trabajo futuro

- Explorar modelos de ensamble más avanzados (XGBoost, AdaBoost)
- Incorporar variables clínicas adicionales (semana de gestación, embarazos previos)
- Validar el modelo en otros contextos geográficos
- Volver al target de 3 clases para un análisis de riesgo más granular

## Stack

Python 3 (Google Colab) · pandas · numpy · matplotlib · seaborn · scikit-learn (`DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV`, `cross_val_score`, `cross_val_predict`, `roc_curve`) · `ucimlrepo`

## Cómo correrlo

```bash
pip install -r requirements.txt
jupyter notebook Modelos_predictivos_de_riesgo_materno.ipynb
```
