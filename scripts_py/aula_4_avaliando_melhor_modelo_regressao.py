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

# %%
imoveis['raiz_valor'] = np.sqrt(imoveis['Valor'])

# %%
imoveis.head()

# %%
plt.hist(imoveis['raiz_valor'], bins = 5)

plt.ylabel('Frequência')
plt.xlabel('Valor')
plt.title('Histograma da variável valor após raiz quadrada aplicada')

# %% [markdown]
# ### Explorando outras variáveis

# %% [markdown]
# #### Explorando variáveis quantitativas

# %%
plt.figure(figsize = (24, 20))

plt.subplot(4, 2, 1)
fig = imoveis.boxplot(column = 'Valor')
fig.set_title(' ')
fig.set_ylabel('Valor em R$')

plt.subplot(4, 2, 2)
fig = imoveis.boxplot(column = 'Area')
fig.set_title(' ')
fig.set_ylabel('Área em m²')

plt.subplot(4, 2, 3)
fig = imoveis.boxplot(column = 'IA')
fig.set_title(' ')
fig.set_ylabel('Idade do Imóvel')

plt.subplot(4, 2, 4)
fig = imoveis.boxplot(column = 'Andar')
fig.set_title(' ')
fig.set_ylabel('Andar')

plt.subplot(4, 2, 5)
fig = imoveis.boxplot(column = 'DistBM')
fig.set_title(' ')
fig.set_ylabel('Distância do Mar')

plt.subplot(4, 2, 6)
fig = imoveis.boxplot(column = 'Suites')
fig.set_title(' ')
fig.set_ylabel('Quantidade de Suites')

# %%
# matriz de correlação
correlation_matrix = imoveis.corr().round(2)

fig, ax = plt.subplots(figsize = (8,8))
sb.heatmap(data = correlation_matrix, annot = True, linewidths= .5, ax = ax)

# %%