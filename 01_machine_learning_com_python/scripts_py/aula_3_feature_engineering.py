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
# processo de transformação de escalas
# importando bibliotecas para transformação
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# %%
# StandarScalar - padroniza 
# padronização dos dados

scaler = StandardScaler()

x_std = scaler.fit_transform(x)

# %%
x

# %%
x_std

# %%
x_std = pd.DataFrame(x_std, columns = ['sqt_living', 'bathrooms'])
x_std.sqt_living.hist();
x_std.bathrooms.hist();

# %%

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
dados.head()

# %%
# divide os dados em x e y
x = dados[['sqft_living', 'bathrooms']].values
y = dados['price'].values

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 7)

# %%
len(x_train)

# %%
len(x_test)

# %%
x_train

# %%
# usando o MinMaxScaller
scaler = MinMaxScaler()

scaler.fit(x_train)

# %%
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# %%
x_train_scaled

# %%
model = LinearRegression()

model.fit(x_train_scaled, y_train)

# %%
y_pred = model.predict(x_test_scaled)

# %%
MAE = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('MAE: ', MAE)
print('R2: ', r2)

# %%
model_normal = LinearRegression()

model_normal.fit(x_train, y_train)

# %%
y_pred_normal = model_normal.predict(x_test)

# %%
MAE = mean_absolute_error(y_test, y_pred_normal)
r2 = r2_score(y_test, y_pred_normal)

print('MAE: ', MAE)
print('R2: ', r2)