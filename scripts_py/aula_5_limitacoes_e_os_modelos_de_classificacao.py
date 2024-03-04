# %% [markdown]
# # Aula 5 | Limitações e os Modelos de Classificação

# %% [markdown]
# ## Prevendo atrasos de vôos

# %% [markdown]
# Nosso _case_ a ser analisado na aula de hoje é um _dataset_ que contém algumas características sobre informações de vôos.
# 
# Nosso foco, basicmente, consiste em prever se um determinado vôo sofrerá atrasos, à partir da informação da partida programada.
# 
# - **Flight**: número do vôo
# - **Time**: horário de partida do vôo
# - **Lenght**: duração do vôo
# - **Airline**: companhia aérea
# - **AirportFrom**: origem
# - **AirportTo**: destino
# - **DayOfWeek**: dia da semana
# - **Class**: classe de atraso
# 
# 

# %% [markdown]
# https://raw.githubusercontent.com/FIAP/Pos_Tech_DTAT/61a634996879d5c1384af6f567e29659ad68b727/Aula%2006/Base%20de%20Dados/airlines.csv

# %%
# Importando bibliotecas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Importando a base de dados
df = pd.read_csv('https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/bases/airlines.csv', sep=',')

# %%
# Verifica as primeiras linhas para entender os dados
df.head(3)

# %%
# Analisa linhas e colunas
df.shape

# %% [markdown]
# ## Analisando a consistência dos dados

# %% [markdown]
# Vamos validar se o _dataset_ contém dados nulos e valores duplicados.

# %%
# Analisa valores nulos
df.isnull().sum()

# %%
# Limpa dados nulos
df = df.dropna()

# %% [markdown]
# Conforme analisado, não temos nenhum valor nulo na base de dados.

# %%
# Verifica dados duplicados
duplicated_cols = []
for col in df.columns:
    if df[col].duplicated().any():
        duplicated_cols.append(col)
print(duplicated_cols)

# %% [markdown]
# Conforme análise sobre os valores duplicados, podemos considerar o comportamento normal pois podemos ter vários casos com o mesmo valor na base.

# %% [markdown]
# ## Análise exploratória dos dados

# %% [markdown]
# Vamos construir uma análise inicial para conhecer os dados. Primeiramente, vamos aplicar a análise estatística descritiva dos dados e fazer algumas inferências sobre os dados.

# %%
df.describe()

# %% [markdown]
# Inferência:
# 
# - Média de duração de vôos e de 133
# - Desvio padrão: 70
# 
# Se o desvio padrão é baixo em relação à média, isso significa que a maioria dos valores estão próximos da média e que os dados estão mais concentrados em torno da média.
# 
# Vamos dar uma olhada nos gráficos:
# 
# Vamos analisar o tempo do vôo utilizando um gráfico **_violin_** em conjunto com o **boxplot**:

# %%
# Podemos também mesclar os dois tipos de gráficos para entender nossos valores discrepantes
fig, ax = plt.subplots(figsize = (8,6))
# Configurando o violin plot
sns.violinplot(x='Length', data = df, ax = ax, color = 'lightgray')
# Por baixo vamos criar um boxplot
sns.boxplot(x = 'Length', data = df, ax = ax, whis = 1.5, color='darkblue')
ax.set_title('Visualização Box Plor e Violin Plot')

plt.show()

# %%
sns.violinplot(x='Class', y='Length', data = df)
plt.show()

# %% [markdown]
# Perceba que a distribuição dos dados entre as classes de atraso sim e não, são bem parecidas. Vamos olhar a média de duração dos vôos.

# %%
atraso_voo = df.groupby('Class')
atraso_voo.describe().T

# %% [markdown]
# Analisando as estatísticas acima, percebe que a variável `Time` é mais discrepante que `Length`.

# %%
sns.violinplot(x = 'Class', y = 'Time', data = df)
plt.show()

# %% [markdown]
# Analisando as companhias aéreas x atrasos dos voos

# %%
sns.countplot(x = 'Airline', hue = 'Class', data = df)

# %% [markdown]
# Inferência dos dados:podemos observar que todas as companhias aéreas possuem a classe de atraso porém são menores do que os vôos sem atraso. Apenas a companhia aérea **WN** possui um grande número de atrasos, ultrapassando o total de voos realizados _on time_.
# 
# Vamos analisar os dias da semana que possuem maior concentração de atrasos.

# %%
diaSemana = list(range(1,8))
sns.countplot(x = 'DayOfWeek', data = df, order = diaSemana)

# %% [markdown]
# A maior concentração de atrasos ocorre na quarta-feira.
# 
# Vamos analisar se a base está equilibrada com o número de atrasos e não atrasos.

# %%
sns.countplot(x = 'Class', data = df)

# %% [markdown]
# Observe que a qui podemos ter um problema ao construi um modelo de classificação. A base de dados não está equilibrada e a falta de equilíbrio na base de dados pode deixar o algoritmo enviesado.
# 
# Esse tipo de problema é chamada de **desbalanceamento de classes**.
# 
# Quando uma classe é muito mais frequente que as outras no conjunto de dados, o modelo tende a dar mais importância a essa classe, o que pode levar a uma classificação incorreta das classes minoritárias.
# 
# Vamos criar um classificar utilizando os dados desbalanceados e vamos analisar o que pode acontecer no modelo de _machine learning_.

# %% [markdown]
# ## Pré-processamento da base

# %% [markdown]
# Para utilizar a companhia aérea dentro do modelo, vamos realizar a transformação da _label enconding_ nos dados.

# %%
df.head(3)

# %%
from sklearn.preprocessing import LabelEncoder

# %%
df['AirportFrom'] = LabelEncoder().fit_transform(df['AirportFrom'])
df['AirportTo'] = LabelEncoder().fit_transform(df['AirportTo'])

# %%
df['Airline'] = LabelEncoder().fit_transform(df['Airline'])

# %%
df.head(3)

# %% [markdown]
# ## Separando a base de dados

# %%
from sklearn.model_selection import train_test_split

# %%