import pytest
import pandas as pd

df_housing_sem_nulo = pd.read_csv('sem_nulos.csv')
df_sem_outlier_e_nulos = pd.read_csv('housing_no_nulls_nor_outliers.csv')

def test_sem_nulos():
    nulls = df_housing_sem_nulo.isna().sum().all()
    nulls_2 = df_sem_outlier_e_nulos.isna().sum().all()
    assert nulls == 0 and nulls_2 == 0


def rodar_testes():
    pytest.main(["-v", "-s"])

rodar_testes()
