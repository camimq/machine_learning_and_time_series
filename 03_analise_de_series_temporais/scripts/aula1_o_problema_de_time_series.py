# %% [markdown]
# # O problema de time series

# %% [markdown]
# - Série Temporal = Funçao que depende do tempo
# 
# - Sazonalidade 
# - Tendência
# - Resíduo

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# %%
# Tendência - direção
# Sazonalidade (recorrência das oscilações) - entender como funcionam as oscilações da série temporal
# Resíduo - o que sobra do sinal

df = pd.read_csv('https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/03_analise_de_series_temporais/bases/Electric_Production.csv')
df.head()


# %%
df.info()

# %%
# transforma a coluna Index em uma coluna de data no formato MM/DD/AAAA
df.index = pd.to_datetime(df.DATE, format = '%m-%d-%Y')

# %%
# DROPA coluna DATE
# axis = 1 - indica  que é pra deletar a coluna DATA
df.drop('DATE', inplace = True, axis = 1)

# %%
df.head()

# %%
df.info()

# %%
# a data virou índice
# localiza pelo índice, que agora é uma data
df.loc['1985-05-01']

# %%
# plota o padrão da série temporal
plt.plot(df.index, df.Value)

# %%
# Decomposição de série temporal
resultados = seasonal_decompose(df)

# %%
# cria

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (15,10))

resultados.observed.plot(ax=ax1)
resultados.trend.plot(ax=ax2)
resultados.seasonal.plot(ax=ax3)
resultados.resid.plot(ax=ax4)

plt.tight_layout()
# %%

# %% [markdown]
# ### Conceitos Estatísticos | Série Tempora Estacionária ou Não Estacionária

# %% [markdown]
# - **Série Temporal Estacionária**: movimentação constante. É uma série temporal em que a média, variância e co-variância dos dados, não desloca no tempo.
# - **Série Temporal Não Estacionária**:  não é uma movimentação constante, é variável. É uma série temporal em que a média, variância e co-variância dos dados varia com o tempo.

# %%
# Estacionária ou não estacionária
# Teste ADF - Augmented Dickey-Fuller Test (determina se existe ou não uma estacionariedade de uma série temporal)

# H0 - Hipótese Nula (não é estacionária)
# H1 - Hipótese Alternativa (rejeição da hipótese nula; estacionária)

# pvalue = 0.05 (5%), então rejeitamos H0 com um nível de confiança de 95%.

# %%
from statsmodels.tsa.stattools import adfuller

# %%
sns.set_style('darkgrid')

# %%
X = df.Value.values

# %%
result = adfuller(X)

print('Teste ADF')
print(f'Teste Estatístico: {result[0]}')
print(f'P-Value: {result[1]}')
print('Valores críticos:')

for key, value in result[4].items():
    print(f'\t{key} : {value}')

# %%
media_movel = df.rolling(12).mean()

f, ax = plt.subplots()
df.plot(ax = ax, legend=False)
media_movel.plot(ax=ax, legend=False, color = 'r')
plt.tight_layout()

# %%
df_log = np.log(df)
media_movel_log = df_log.rolling(12).mean()

f, ax = plt.subplots()
df_log.plot(ax=ax, legend=False)
media_movel_log.plot(ax=ax, legend=False, color='r')
plt.tight_layout()

# %%
df_s = (df_log - media_movel_log).dropna()
media_movel_s = df_s.rolling(12).mean()

std = df_s.rolling(12).std()

f, ax = plt.subplots()
df_s.plot(ax=ax, legend=False)
media_movel_s.plot(ax=ax, legend=False, color='r')
std.plot(ax=ax, legend=False, color='g')
plt.tight_layout()

# %%
X_s = df_s.Value.values
result_s = adfuller(X_s)

print('Teste ADF')
print(f'Teste Estatístico: {result_s[0]}')
print(f'P-Value: {result_s[1]}')
print('Valores críticos:')

for key, value in result_s[4].items():
    print(f'\t{key} : {value}')

# %%
df_diff = df_s.diff(1)
media_movel_diff = df_diff.rolling(12).mean()

std_diff = df_diff.rolling(12).std

# %%
f, ax = plt.subplots()
df_diff.plot(ax=ax, legend=False)
media_movel_diff.plot(ax=ax, legend=False, color='r')
std_diff.plot(ax=ax, legend=False, color='g')
plt.tight_layout()

# %%

X_diff = df_diff.Value.dropna().values
result_diff = adfuller(X_diff)

print('Teste ADF')
print(f'Teste Estatístico: {result_diff[0]}')
print(f'P-Value: {result_diff[1]}')
print('Valores críticos:')

for key, value in result_diff[4].items():
    print(f'\t{key} : {value}')