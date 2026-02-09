# Fraud Detection System v1.0.0

## 1. Modelo Utilizado y Justificación
Se ha implementado un modelo basado en LightGBM (Light Gradient Boosting Machine) utilizando un enfoque de Gradient Boosting Decision Trees (GBDT).

### Justificación Técnica
- Eficiencia en Inferencia: Diseñado para cumplir con latencias de 10–30 ms, optimizando la profundidad de los árboles (max_depth: 6) para un recorrido rápido en CPU.

- Manejo de Datos Tabulares: LightGBM gestiona nativamente variables categóricas de alta cardinalidad (como BIN Bank) y datos faltantes sin necesidad de imputación costosa.

- Arquitectura SOLID: El sistema está desacoplado mediante interfaces, permitiendo el intercambio por otros modelos (XGBoost/CatBoost) sin alterar el pipeline de datos.

---

### Ingeniería de Características (Feature Engineering)
El modelo se alimenta exclusivamente de variables pre-autorización para evitar el target leakage:

#### A) Parsing de timestamp (sin fuga)
- Se elige la columna de tiempo:
  - primaria: `Creation date` (si parsea bien),
  - fallback: `Fecha`.
- Se derivan features de calendario:
  - `tx_hour`, `tx_dow`, `tx_day`, `tx_month`, `tx_is_weekend`.

**Por qué importa:** el fraude muestra patrones temporales (horarios, fines de semana, campañas, estacionalidad).

#### B) Transformaciones monetarias (robustas)
- `valor_log1p` a partir de `Valor`
- `processing_value_log1p` a partir de `Processing value`

**Por qué importa:** montos con cola larga → `log1p` estabiliza y mejora la separabilidad.

#### C) Señales de inconsistencia (mismatch)
- `country_mismatch`: `Country` vs `Country BIN ISO`
- `currency_mismatch`: `Transaction currency` vs `Processing currency`

**Por qué importa:** inconsistencias país/moneda vs BIN suelen correlacionar con fraude o anomalías.

#### D) Velocity features (sin fuga)
Se agregan conteos en ventana 24h, **excluyendo el evento actual**:

- `cnt_account_24h`: apariciones de `Account Id` en las últimas 24h
- `cnt_card_24h`: apariciones de `Visible number` en las últimas 24h

**Por qué importa:** capturan ráfagas (bots, pruebas de tarjeta, ataques por lotes).  
**Detalle crítico:** “excluir el evento actual” evita leakage.

#### E) Categóricas: encoding robusto + unknown
- Se mapean categóricas a enteros durante `fit()`.
- En `transform()`, valores no vistos → `-1` (unknown), evitando romper producción por nuevas categorías.

#### F) Eliminación de columnas peligrosas
Se excluyen columnas que son **resultado** del sistema o “label leakage”, por ejemplo:
- `Status`, `Response code`, `Bank Response code`, `Message response error`, `Authorization code`

Y también columnas altamente únicas / propensas a overfit:
- `Order Id`, `Reference`, `Description`

#### Split temporal (para evaluar generalización)
El dataset se separa por mes:
- **test = último mes**
- **valid = penúltimo mes**
- **train = resto**

Esto fuerza una evaluación más realista (generalización a “futuro cercano”).

---

## 2. Evaluación del Modelo
La evaluación se realizó sobre un Split Temporal Triple, utilizando el mes de agosto de 2020 como conjunto de prueba independiente.

### Métricas globales (split `test`)
- **ROC-AUC:** 0.9154
- **Average Precision (AP):** 0.9246
- **LogLoss:** 0.3696
- **Brier Score:** 0.1191

**Lectura técnica:**
- **ROC-AUC ~0.91**: muy buena capacidad de ranking (separa fraude/no fraude).
- **AP ~0.92**: alto y consistente con la forma de la PR curve (buena precisión para recalls amplios).
- **LogLoss/Brier**: sugieren probabilidades razonables para operar con umbrales (aunque siempre conviene calibrar en v2).

### Artefactos generados
- Curvas:
  - `src/artifacts/eval/roc_curve.png`
  - `src/artifacts/eval/pr_curve.png`
- Reporte detallado:
  - `src/artifacts/eval/eval_results.json`
- Importancia de variables:
  - `src/artifacts/eval/feature_importance.csv`
  - `src/artifacts/eval/feature_importance.png`

---

### Triage por umbrales (estrategia operativa y de mejora de dataset)

El objetivo no es “clasificar perfecto” desde el día 1, sino **operar seguro y recolectar mejores datos**.

Valores actuales:
- **T_low:** 0.2036 (auto-approve si score < T_low)
- **T_high:** 0.7117 (auto-decline si score ≥ T_high)
- **review:** 0.20 < score < 0.71

Rates resultantes:

- **approve_rate ≈ 20.0%**
- **review_rate ≈ 46.8%**
- **decline_rate ≈ 33.2%**

Calidad por zona:
- **auto-decline:** decline_precision ≈ 0.951 y decline_recall ≈ 0.793
  - Significa: Cuando rechazas en automático, le atinas muy seguido (95%), y capturas ~79% de los positivos en esa zona.
- **auto-approve:** approve_fraud_rate ≈ 0.077
  - Significa: Hay 7.7% de “fraude” en lo que apruebas automático

**Intención del triage:**
- Disminuir riesgo (rechazar solo cuando el modelo está muy seguro).
- Llevar el “costo” al review, donde se obtiene feedback humano / outcomes.
- **Aumentar dataset** con ejemplos informativos (zona gris) y **reducir sesgos** progresivamente.

---

## 4) Consideraciones y/o limitaciones

### Limitaciones Identificadas
**Sesgo Geográfico Dominante:** El modelo presenta una dependencia extrema de la variable Country_BIN_ISO. Ha identificado que países como US y BO tienen tasas de rechazo superiores al 95%, lo que genera un riesgo de "bloqueo geográfico" injusto para clientes legítimos internacionales.

**Proxy Labeling:** Al no contar con etiquetas de fraude real (contracargos confirmados), el modelo está entrenado para imitar al sistema anterior. Esto significa que heredamos los sesgos y errores del pasado.

**Volumen de Datos:** el modelo tiende a memorizar IDs específicos, por eso hemos eliminado ese feature inicialmente.

### Estrategia de Triage

La recomendación de umbrales ($T_{low}=0.20, T_{high}=0.71$ según el objetivo de negocio) tiene un propósito estratégico:

- Aislamiento de Sesgo: Al enviar los casos "grises" a revisión manual, el equipo de expertos podrá generar etiquetas reales que no dependan de reglas automáticas.

- Enriquecimiento del Dataset: Estos nuevos datos servirán para re-entrenar la version 1.1, eliminando gradualmente variables de identidad y priorizando el comportamiento puro.

- Optimización Operativa: El objetivo a corto plazo es ajustar la calibración para reducir el 46.8% de revisión manual a un nivel manejable (<15%) sin incrementar el riesgo financiero.
---