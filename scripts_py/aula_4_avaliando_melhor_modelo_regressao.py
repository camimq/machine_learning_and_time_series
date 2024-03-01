# %% [markdown]
# # Aula 5 | Avaliando o melhor modelo de regressão
# 
# ## Prevendo valores de imóveis
# 
# Na aula de hoje, vamos explorar um _dataset_ que contém algumas características sobre imóveis, tais como área, andar, suites, vista entre outros atributos.
# 
# Nosso desafio de hoje será tentar encontrar uma forma de criar um algoritmo preditivo que utilize essas características para predizer o valor do imóvel.
# 
# ### Atributos
# 
# - Ordem: coluna ID
# - Valor: valor do imóvel
# - Area: tamanho da área do imóvel
# - IA: idade do imóvel
# - Andar: quantidade de andares
# - Suites: quantidade de suítes
# - Vista: se o imóvel possui uma boa vista ou não
# - DistBM: distância do imóvel do mar
# - SemRuido: se o imóvel é localizado em uma região calma ou não
# - AV100m: distância próxima à área verde.
# 
# Vamos começar a trabalhar com os dados!

# %% [markdown]
# ## Código

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# %%
# importa base de imóveis
imoveis = pd.read_csv('https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/bases/Valorizacao_Ambiental.csv', sep = ';')
imoveis.head()

# %%
imoveis.info()

# %%
imoveis.shape

# %%
# verifica se a base tem dados nulos
imoveis.isnull().sum()

# %%
# estatística descritiva básica com arredondamento dos dados
imoveis.describe().round(2)

# %% [markdown]
# ### Identificando a variável _target_.

# %%
plt.hist(imoveis['Valor'], bins = 5)

plt.ylabel('Frequência')
plt.xlabel('Valor')
plt.title('Histograma da variável valor')