import pytest
import pandas as pd

df_housing_sem_nulo = pd.read_csv('sem_nulos.csv')

def test_sem_nulos():
    nulls = df_housing_sem_nulo.loc[(df_housing_sem_nulo['total_bedrooms'].isna() == True)]
    assert len(nulls) == 0


def rodar_testes():
    pytest.main(["-v", "-s"])

rodar_testes()
