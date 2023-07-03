from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Datos de entrada y salida para la compuerta AND
entrada = [[0, 0], [0, 1], [1, 0], [1, 1]]
salida = [0, 0, 0, 1]

# Crear el modelo de la red neuronal
modelo = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', solver='adam', max_iter=5000)

# Entrenar el modelo
modelo.fit(entrada, salida)

# Realizar predicciones
entrada_prueba = [[0, 0], [0, 1], [1, 0], [1, 1]]
predicciones = modelo.predict(entrada_prueba)

# Mostrar resultado
for i in range(len(entrada_prueba)):
    print(f"Entrada: {entrada_prueba[i]}, Predicción: {predicciones[i]}")