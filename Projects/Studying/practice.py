import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Análise de preços usando Regressão Linear

data = {
    "Tamanho (m²)": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    "Preço (mil)": [200, 250, 280, 310, 350, 400, 420, 450, 480, 500]
}

df = pd.DataFrame(data)

# Separando variáveis (X = Tamanho da casa, y = Preço da Casa)

X = df[["Tamanho (m²)"]]
y = df["Preço (mil)"]

# Dividindo os dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazendo previsões
y_pred = modelo.predict(X_test)

# Avaliando o modelo
erro = mean_absolute_error(y_test, y_pred)

# Resultados
print(f"Coeficiente (inclinação): {modelo.coef_[0]:.2f}")
print(f"Interceptação (onde cruza o eixo Y): {modelo.intercept_:.2f}")
print(f"Erro médio absoluto: {erro:.2f} mil")

# Gráfico
plt.scatter(X, y, label="Dados reais", color="blue")
plt.plot(X, modelo.predict(X), label="Regressão Linear", color="red")
plt.xlabel("Tamanho (m²)")
plt.ylabel("Preço (mil)")
plt.legend()
plt.show()

# Fazer testes usando o modelo