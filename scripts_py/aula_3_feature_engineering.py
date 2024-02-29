# %% [markdown]
# # Feature Engineering

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# %%
dados = pd.read_csv('https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/bases/data.csv', sep = ',')

# %%
dados.head()

# %%
# cria matriz de correção
correlation_matrix = dados.corr(numeric_only = True).round(2)

fig, ax = plt.subplots(figsize = (8,8))
sns.heatmap(data = correlation_matrix, annot = True, linewidth = .5, ax = ax)

# %%
# separa os dados
x = dados[['sqft_living', 'bathrooms']].values
y = dados['price'].values

# %%
# cria gráfico de dispersão
sns.scatterplot(data = dados, x = 'sqft_living', y = 'price')

# %%
# cria gráfico de dispersão
sns.scatterplot(data = dados, x = 'bathrooms', y = 'price')

# %%
# cria os dois gráficos de dispersão juntos
fig, ax = plt.subplots(figsize = (12, 4))

ax.scatter(x[:,0], y)
ax.scatter(x[:,1], y)

# %%
# cria histograma dos dados
sns.histplot(data = dados, x = 'sqft_living', kde = True)

# %%
sns.histplot(data = dados, x = 'bathrooms', kde = True)

# %%
hist_variaveis = pd.DataFrame(dados, columns = ['sqft_living', 'bathrooms'])
hist_variaveis.sqft_living.hist()
hist_variaveis.bathrooms.hist()

# %%



