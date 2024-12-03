import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

# Função para carregar e limpar o dataset
def preprocess_data(input_file='./datasets/housing.csv', output_file='./datasets/housing_clean.csv'):
    # Carregar os dados
    df = pd.read_csv(input_file)
    
    # Exibir as primeiras linhas do dataset
    print("Dataset Original:")
    print(df.head())

    # Tratar valores ausentes
    # Utilizando KNNImputer para preencher valores ausentes nas colunas numéricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    knn_imputer = KNNImputer(n_neighbors=5)  # Número de vizinhos para o imputador

    # Aplicar imputação nos dados numéricos
    df[numeric_columns] = knn_imputer.fit_transform(df[numeric_columns])

    # Manter a coluna 'ocean_proximity' como está, sem alteração
    # A coluna 'ocean_proximity' não será afetada pelo imputador

    # Multiplicar o 'median_income' por 10.000 para ajustar a escala e arredondar para inteiros
    df['median_income'] = (df['median_income'] * 10000).round().astype(float)

    # Exibir as primeiras linhas do dataset após o tratamento
    print("Dataset Após Tratamento de Valores Ausentes e Ajuste da Escala de 'median_income':")
    print(df.head())

    # Opcional: Remover outliers utilizando o método IQR (Interquartile Range)
    # Definindo um critério de outlier para as variáveis numéricas
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    
    # Limite superior e inferior para remoção de outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrando os outliers
    df_filtered = df[~((df[numeric_columns] < lower_bound) | (df[numeric_columns] > upper_bound)).any(axis=1)]
    
    # Exibir informações após remoção de outliers
    print(f"Após Remoção de Outliers, o dataset contém {df_filtered.shape[0]} linhas.")

    # Salvar o dataset limpo em um novo arquivo
    df_filtered.to_csv(output_file, index=False)
    print(f"Dataset limpo salvo em {output_file}")

# Executando a função de pré-processamento
if __name__ == "__main__":
    preprocess_data()
