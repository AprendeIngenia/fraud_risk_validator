
def lgbm_fast_inference_params():
    """
    modelo personalizable
    """

    return {
        "objective": "binary",     # tarea: 0 - 1
        "boosting_type": "gbdt",   # gradient boosting decision tree
        "learning_rate": 0.05,     # pasos de aprendizaje

        # complejidad del arbol
        "max_depth": 6,            # 2^max_depth
        "num_leaves": 32,          # tamaño maximo del arbol (nodos)
        "min_data_in_leaf": 300,   # hojas grandes => menos splits
        "min_gain_to_split": 0.0,  # ganancia minima para hacer un corte

        # entrenamiento
        "feature_fraction": 0.85,  # columnas usadas por epoca
        "bagging_fraction": 0.85,  # filas usadas por epoca
        "bagging_freq": 1,         # bagging por epoca

        # Regularización
        "lambda_l1": 0.0,          # no aplica penalizacion pr pesos grandes de forma lineal
        "lambda_l2": 1.0,          # aplica penalizacion para evitar pesos excesivos

        # eficiencia
        "max_bin": 255,            # equilibrar velocidad y precisión

        # Repro y logging
        "verbosity": -1,           # desavtiva warinings
        "n_jobs": -1,              # usa todos los nucleos
        "seed": 42,                # semilla - reproducibilidad
    }
