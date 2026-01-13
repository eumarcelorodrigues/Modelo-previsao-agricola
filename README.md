# Modelo-previsao-agricola

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

"""
MÓDULO DE PREVISÃO DE PRODUTIVIDADE AGRÍCOLA (CROP YIELD)
Este script carrega dados agrícolas, processa variáveis categóricas,
treina um modelo de Random Forest e avalia a importância de cada fator.
"""

def carregar_e_limpar_dados(caminho_arquivo):
    """
    Carrega o dataset e realiza a codificação de variáveis categóricas.
    
    Args:
        caminho_arquivo (str): Caminho para o ficheiro CSV.
    Returns:
        df (DataFrame): Dados processados.
        encoders (dict): Dicionário com os transformadores para uso futuro.
    """
    df = pd.read_csv(caminho_arquivo)
    
    # Identificar colunas que não são numéricas
    colunas_categoricas = ['Soil_Type', 'Region', 'Season', 'Crop_Type', 'Irrigation_Type']
    label_encoders = {}

    # Converter texto em números (ex: 'Clay' -> 0, 'Sandy' -> 1)
    for col in colunas_categoricas:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    return df, label_encoders

# 1. Preparação dos Dados
df_processado, encoders = carregar_e_limpar_dados('crop-yield.csv')

# Definir Features (X) e Alvo (y)
X = df_processado.drop('Crop_Yield_ton_per_hectare', axis=1)
y = df_processado['Crop_Yield_ton_per_hectare']

# Divisão entre treino (80%) e teste (20%) para validação do modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Construção do Modelo
# Usamos o RandomForestRegressor pela sua robustez com dados não-lineares
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 3. Avaliação de Performance
previsoes = modelo.predict(X_test)
mae = mean_absolute_error(y_test, previsoes)
r2 = r2_score(y_test, previsoes)

print(f"--- Performance do Modelo ---")
print(f"Erro Médio Absoluto (MAE): {mae:.2f} ton/ha")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

# 4. Análise de Importância de Variáveis
def plotar_importancia(modelo, colunas):
    """
    Gera um gráfico de barras com os fatores que mais influenciam a colheita.
    """
    importancias = modelo.feature_importances_
    indices = importancias.argsort()

    plt.figure(figsize=(10, 6))
    plt.title('Importância dos Fatores na Produtividade')
    plt.barh(range(len(indices)), importancias[indices], color='forestgreen')
    plt.yticks(range(len(indices)), [colunas[i] for i in indices])
    plt.xlabel('Impacto Relativo (0 a 1)')
    plt.tight_layout()
    plt.show()

# Executar a visualização
plotar_importancia(modelo, X.columns)
