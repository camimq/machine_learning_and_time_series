# %% [markdown]
# # Análise Exploratória de dados

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

# %%
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
# f, ax = plt.subplots()
# df_diff.plot(ax=ax, legend=False)
# media_movel_diff.plot(ax=ax, legend=False, color='r')
# std_diff.plot(ax=ax, legend=False, color='g')
# plt.tight_layout()

# %%

X_diff = df_diff.Value.dropna().values
result_diff = adfuller(X_diff)

print('Teste ADF')
print(f'Teste Estatístico: {result_diff[0]}')
print(f'P-Value: {result_diff[1]}')
print('Valores críticos:')

for key, value in result_diff[4].items():
    print(f'\t{key} : {value}')

# %%
# ARIMA - (AR): autoregressivo, I: integrado, MA: moving average
# A(x, y, z) - ACF, PACF

# %%
lag_acf = acf(df_diff.dropna(), nlags=25)
lag_pacf = pacf(df_diff.dropna(), nlags=25)

# %%
# 5% da autocorrelação (ACF)
# 1.96 / sqrt(N-d) -> N é o número de pontos do DF e D, é o número de vezes que nós diferenciamos o DF

# %%
plt.plot(lag_acf)

plt.axhline(y= -1.96/(np.sqrt((len(df_diff) - 1))), linestyle = '--', color='gray', linewidth = 0.7)
plt.axhline(y= 0, linestyle = '--', color='gray', linewidth = 0.7)
plt.axhline(y= 1.96/(np.sqrt((len(df_diff) - 1))), linestyle = '--', color='gray', linewidth = 0.7)

plt.title('ACF - Autocorrelação')
plt.show()

plt.plot(lag_pacf)

plt.axhline(y= -1.96/(np.sqrt((len(df_diff) - 1))), linestyle = '--', color='gray', linewidth = 0.7)
plt.axhline(y= 0, linestyle = '--', color='gray', linewidth = 0.7)
plt.axhline(y= 1.96/(np.sqrt((len(df_diff) - 1))), linestyle = '--', color='gray', linewidth = 0.7)

plt.title('PACF - Autocorrelação Parcial')
plt.show()

# %%
# A(x, y, z)

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %%
plot_acf(df.Value)
plot_pacf(df.Value)
plt.show()

# %%

df_forecasting = pd.read_csv(r'G:\My Drive\6. Estudos\1. FIAP\Fase 2 - Machine Learning & Time Series\03. Análise de Séries Temporais\train.csv', index_col='id', parse_dates=['date'])
df_forecasting.head()

# %%
df_forecasting

# %%
# mostra quantas lojas únicas temos no dataset
df_forecasting['store_nbr'].nunique()

# %%

# escolha a loja 1 para trabalhar
df_loja1 = df_forecasting.loc[df_forecasting['store_nbr'] == 1, ['date', 'family', 'sales']]
df_loja1 = df_loja1.rename(columns={'date' : 'ds', 'sales' : 'y', 'family' : 'unique_id'})
df_loja1

# %%
# separa a base, pegando 2013 como treino
# prepara os tres primeiros meses de 2014 para validação
treino = df_loja1.loc[df_loja1['ds'] < '2014-01-01'] # cria base / variável de treino
valid = df_loja1.loc[(df_loja1['ds'] >= '2014-01-01') & (df_loja1['ds'] < '2014-04-01')] # cria base / variável de validação
h = valid['ds'].nunique()  # cria base / variável que irão trabalhar com o tempo em que será trabalhada a previsão

# %%
h

# %%
def wmape(y_true, y_pred):
    return np.abs(y_true-y_pred).sum() / np.abs(y_true).sum()

# %%
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, SeasonalWindowAverage, AutoARIMA

# %%
model = StatsForecast(models=[Naive()], freq='D', n_jobs=-1)
model.fit(treino)

forecast_df = model.predict(h=h, level=[90])
forecast_df = forecast_df.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

wmape1 = wmape(forecast_df['y'].values, forecast_df['Naive'].values)
print(f'WMAPE: {wmape1: .2%}')

model.plot(treino, forecast_df, level=[90], unique_ids = ['MEATS', 'PERSONAL CARE'], engine='matplotlib', max_insample_length=90)

# %%
forecast_df

# %%
model_s = StatsForecast(models=[SeasonalNaive(season_length=7)], freq='D', n_jobs=-1)
model_s.fit(treino)

forecast_dfs = model_s.predict(h=h, level=[90])
forecast_dfs = forecast_dfs.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

wmape2 = wmape(forecast_dfs['y'].values, forecast_dfs['SeasonalNaive'].values)
print(f'WMAPE: {wmape2: .2%}')

model_s.plot(treino, forecast_dfs, level=[90], unique_ids = ['MEATS', 'PERSONAL CARE'], engine='matplotlib', max_insample_length=90)

# %%
# model_sm = StatsForecast(models=[SeasonalWindowAverage(season_length=7, window_size=2)], freq='D', n_jobs=-1)
# model_sm.fit(treino)

# forecast_dfsm = model_sm.predict(h=h, level=[90])
# forecast_dfsm = forecast_dfsm.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

# wmape3 = wmape(forecast_dfsm['y'].values, forecast_dfsm['SeasWA'].values)
# print(f'WMAPE: {wmape3: .2%}')

# model_sm.plot(treino, forecast_dfsm, level=[90], unique_ids = ['MEATS', 'PERSONAL CARE'], engine='matplotlib', max_insample_length=90)

# %%
# ARIMA - AR: olha para as vendas do passado e acha uma correlação futura
# I - quantidade que a série foi diferenciada / MA: Média Móvel
model_a = StatsForecast(models=[AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
model_a.fit(treino)

forecast_a = model_a.predict(h=h, level=[90])
forecast_a = forecast_a.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

wmape4 = wmape(forecast_a['y'].values, forecast_a['AutoARIMA'].values)
print(f'WMAPE: {wmape4: .2%}')

model_a.plot(treino, forecast_a, level=[90], unique_ids = ['MEATS', 'PERSONAL CARE'], engine='matplotlib', max_insample_length=90)

# %%
model_a = StatsForecast(models=[Naive(),AutoARIMA(season_length=7), SeasonalNaive(season_length=7)], freq='D', n_jobs=-1)
model_a.fit(treino)

forecast_a = model_a.predict(h=h, level=[90])
forecast_a = forecast_a.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

wmape5 = wmape(forecast_a['y'].values, forecast_a['Naive'].values)
wmape6 = wmape(forecast_a['y'].values, forecast_a['AutoARIMA'].values)
wmape7 = wmape(forecast_a['y'].values, forecast_a['SeasonalNaive'].values)
print(f'WMAPE: {wmape5: .2%}')
print(f'WMAPE: {wmape6: .2%}')
print(f'WMAPE: {wmape7: .2%}')

model_a.plot(treino, forecast_a, level=[90], unique_ids = ['MEATS', 'PERSONAL CARE'], engine='matplotlib', max_insample_length=90)


