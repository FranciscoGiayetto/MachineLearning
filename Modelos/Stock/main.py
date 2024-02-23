import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar el conjunto de datos en formato CSV
data = pd.read_csv('datos.csv')

# Ingeniería de características
data['registro_fecha'] = pd.to_datetime(data['registro_fecha'])
data['mes'] = data['registro_fecha'].dt.month
data['dia_semana'] = data['registro_fecha'].dt.dayofweek

# Codificación one-hot para variables categóricas
data = pd.get_dummies(data, columns=['nombre', 'descripcion', 'categoria_nombre'])

# Separar las características (X) y la variable objetivo (y)
X = data.drop(['codigo', 'stock_agotado', 'registro_fecha'], axis=1)
y = data['stock_agotado']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar los hiperparámetros del modelo
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'min_data_in_leaf': 5,  # Modificación importante
}

# Crear conjuntos de datos de LightGBM
train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)

# Entrenar el modelo
num_round = 100
model = lgb.train(params, train_data, num_round, valid_sets=[train_data, test_data])

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Convertir probabilidades a etiquetas (0 o 1) para la clasificación binaria
y_pred_labels = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred_labels)
conf_matrix = confusion_matrix(y_test, y_pred_labels)
classification_rep = classification_report(y_test, y_pred_labels)

print('Precisión del modelo:', accuracy)
print('Matriz de Confusión:')
print(conf_matrix)
print('Informe de Clasificación:')
print(classification_rep)
