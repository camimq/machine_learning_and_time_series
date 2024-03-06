# %% [markdown]
# # Clusterização
# 
# O agrupamento  é uma técnica para dividir os dados  em diferentes grupos, na qual os registros em cada grupo são semelhantes uns aos outros. Os grupos podem ser usados diretamente, analisando mais a fundo ou passados como uma característica ou resultado para um modelo de regressão ou classificação.
# ## Grupo de Consumidores
# 
# Vamos aprender a realizar um modelo de clusterização utilizando um case de segmentação de clientes de um shopping. Como podemos criar grupos de consumidores dado algumas caracteríticas de perfis?
# 
# ## Sobre a base de dados:
# 
# Esse conjunto de dados ilustra alguns dados dos consumidores de um shopping. A base possui algumas features como: gênero, idade, renda anual e pontuação de gastos.
# 

# %% [markdown]
# ### Bibliotecas utilizadas

# %%
import pandas as pd

# Plot dos gráficos
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

# Algoritmos de Agrupamento
from sklearn.cluster import KMeans, DBSCAN

# Avaliacao de desemepnho
from sklearn.metrics import adjusted_rand_score, silhouette_score

# %% [markdown]
# # Algumas principais técnicas de clusterização
# 
# 

# %%
dados = pd.read_csv("https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/02_machine_learning_avancado/bases/mall.csv", sep=',')

# %%
dados.shape

# %%
dados.head()

# %% [markdown]
# ## Limpeza dos dados

# %%
dados.isnull().sum()

# %% [markdown]
# ## Análise exploratória dos dados
# 
# - Conhecer os dados, identificar padrões, encontrar anomalias, etc.

# %%
dados.describe()

# %%
dados['Annual Income (k$)'].median()

# %% [markdown]
# Analisando a distribuição das variáveis:

# %%
dados.hist(figsize=(12,12))

# %% [markdown]
# Analisando a correlação entre as variáveis:

# %%
plt.figure(figsize=(6,4))
sns.heatmap(dados[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr(method = 'pearson'), annot=True, fmt=".1f");

# %% [markdown]
# Analisando a proporção entre gêneros:

# %%
dados['Gender'].value_counts()

# %% [markdown]
# Boa proporção entre os generos que temos disponíveis em nossos dados.
# 
# Vamos fazer um gráfico completo com todos os dados para checarmos possíveis agrupamentos que podem ser realizados.

# %%
sns.pairplot(dados, hue="Gender")
plt.show()

# %% [markdown]
# Aparentemente o Annual Income e o Spending Score permitem alguns agrupamentos dos nossos dados. 
# 
# Podemos trabalhar com eles.

# %% [markdown]
# ## Feature Scaling
# 
# Verificar a necessidade de utilizar a padronização ou normalização dos dados

# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Feature Engineer

# %%
scaler = StandardScaler() 
# scaler = MinMaxScaler() 
scaler.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])

# %%
dados_Escalonados = scaler.transform(dados[['Annual Income (k$)','Spending Score (1-100)']])

# %%
dados_Escalonados

# %% [markdown]
# ### Criando os agrupamentos
# Vamos criar agrupamentos com diferentes metodologias:
# 
# ### 1 - K-Means

# %% [markdown]
# **Sobre o modelo:**
# O K-Means parte da ideia de quebrar o espaço multidimensional de dados em partições a partir do centróide dos dados. Após inicializar os centróides de forma aleatória sobre os dados, o K-Means **calcula a distância dos dados para os centros mais próximos**. Esse cálculo da distância é realizado várias vezes até que os dados sejam agrupados da melhor forma possível de acordo com a distância mais próxima de um centróide (ponto centro de dado na qual será formado o grupo).
# 
# **Hiperparametros:**
# Definição do K. Para definir esse valor de K, é necessário utilizar o **método Elbow** para encontrar o melhor hiperparâmetros de K. O método Elbow consiste no cálculo da soma dos erros quadráticos.
# 
# **Vantagens:**
# Implementação simplificado e possui uma certa facilidade em lidar com qualquer medida de similaridade entre os dados.
# 
# **Desvantagem:**
# Difícil definir o melhor K. Sensível a outliers. Não consegue distinguir grupos em dados não-globulares.
# 
# Para mais informação: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# %% [markdown]
# Executando o algoritmo sem feature scaling

# %%
# Definindo o modelo de clusterizacao. K-MEANS com 6 clusters
kmeans = KMeans(n_clusters=6,random_state=0) # Definindo os hiperparametros do algoritmo (definir o número de grupo = cluster)

# Implementando o K-Means nos dados:
kmeans.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])

# Salvando os centroides de cada cluster
centroides = kmeans.cluster_centers_

# Salvando os labels dos clusters para cada exemplo
kmeans_labels = kmeans.predict(dados[['Annual Income (k$)','Spending Score (1-100)']])

# %% [markdown]
# Executando com feature scaling

# %%
# Definindo o modelo de clusterizacao. K-MEANS com 6 clusters
kmeans_escalonados = KMeans(n_clusters=6,random_state=0)

# Implementando o K-Means nos dados:
kmeans.fit(dados_Escalonados)

# Salvando os centroides de cada cluster
centroides_escalonados = kmeans.cluster_centers_

# Salvando os labels dos clusters para cada exemplo
kmeans_labels_escalonado = kmeans.predict(dados_Escalonados)

# %%
dados_Escalonados = pd.DataFrame(dados_Escalonados, columns = ['Annual Income (k$)','Spending Score (1-100)'])

# %%
dados_Escalonados.head()

# %%
dados_Escalonados['Grupos'] = kmeans_labels_escalonado
dados_Escalonados.head()

# %%
dados['Grupos'] = kmeans_labels
dados.head()

# %% [markdown]
# Vamos analisar a nossa previsao e os centroides:

# %%
pd.Series(kmeans_labels).value_counts()

# %%
centroides #espaço tridimensional (salário e score de gasto)

# %% [markdown]
# ### Clusters com feature scaling

# %%
# plotando os dados identificando com os seus clusters
plt.scatter(dados_Escalonados[['Annual Income (k$)']],dados_Escalonados[['Spending Score (1-100)']], c=kmeans_labels_escalonado, alpha=0.5, cmap='rainbow')
plt.xlabel('Salario Anual')
plt.ylabel('Pontuação de gastos')

# plotando os centroides
plt.scatter(centroides_escalonados[:, 0], centroides_escalonados[:, 1], c='black', marker='X', s=200, alpha=0.5)
plt.rcParams['figure.figsize'] = (10, 5)
plt.show()

# %% [markdown]
# ### Clusters sem feature scaling

# %%
# plotando os dados identificando com os seus clusters
plt.scatter(dados[['Annual Income (k$)']],dados[['Spending Score (1-100)']], c=kmeans_labels, alpha=0.5, cmap='rainbow')
plt.xlabel('Salario Anual')
plt.ylabel('Pontuação de gastos')
# plotando os centroides
plt.scatter(centroides[:, 0], centroides[:, 1], c='black', marker='X', s=200, alpha=0.5)
plt.rcParams['figure.figsize'] = (10, 5)
plt.show()

# %% [markdown]
# Escolhendo a quantidade de grupos usando o método do "cotovelo":

# %%
# Lista com a quantidade de clusters que iremos testar
k = list(range(1, 10))
print(k)

# %%
# Armazena o SSE (soma dos erros quadraticos) para cada quantidade de k
sse = []

# Roda o K-means para cada k fornecido
for i in k:
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])
    sse.append(kmeans.inertia_) # cálculo do erro do k-mens (mudar o centroide dos dados)

plt.rcParams['figure.figsize'] = (10, 5)
# Plota o gráfico com a soma dos erros quadraticos
plt.plot(k, sse, '-o')
plt.xlabel(r'Número de clusters')
plt.ylabel('Inércia')
plt.show()

# %%
dados.groupby('Grupos')['Age'].mean()

# %%
dados.groupby('Grupos')['Annual Income (k$)'].mean()

# %% [markdown]
# Podemos notar que após 3 ou 5 clusters a soma do erro quadratico tem uma redução na forma com a qual a função está decrescendo. Assim podemos adotar 5 clusters. Checando os resultados para 5 clusters:

# %%
# Definindo o modelo de clusterizacao. K-MEANS com 5 clusters
kmeans = KMeans(n_clusters=5,random_state=0)

# Implementando o K-Means nos dados:
kmeans.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])

# Salvando os centroides de cada cluster
centroides = kmeans.cluster_centers_

# Salvando os labels dos clusters para cada exemplo
kmeans_labels = kmeans.predict(dados[['Annual Income (k$)','Spending Score (1-100)']])

# %%
# plotando os dados identificando com os seus clusters
plt.scatter(dados[['Annual Income (k$)']],dados[['Spending Score (1-100)']], c=kmeans_labels, alpha=0.5, cmap='rainbow')
plt.xlabel('Salario Anual')
plt.ylabel('Pontuação de gastos')
# plotando os centroides
plt.scatter(centroides[:, 0], centroides[:, 1], c='black', marker='X', s=200, alpha=0.5)
plt.rcParams['figure.figsize'] = (10, 5)
plt.show()

# %%
dados_grupo_1 = dados[dados['Grupos'] == 1]
dados_grupo_1

# %%
dados_grupo_2 = dados[dados['Grupos'] == 2]

# %%
dados_grupo_3 = dados[dados['Grupos'] == 3]

# %%
dados_grupo_4 = dados[dados['Grupos'] == 4]

# %%
dados_grupo_1['Annual Income (k$)'].mean() #grupo 1 azul

# %%
dados_grupo_2['Annual Income (k$)'].mean() #grupo 2 roxo

# %%
dados_grupo_3['Annual Income (k$)'].mean() #grupo 3 laranja

# %%
dados_grupo_3['Age'].mean()

# %%
dados_grupo_2['Age'].mean() # grupo 2 roxo

# %%
dados_grupo_4['Annual Income (k$)'].mean() # grupo 4 vermelho

# %%
dados_grupo_3['Spending Score (1-100)'].mean() # grupo 4

# %%
plt.figure(figsize=(6,4))
sns.heatmap(dados_grupo_1.groupby('Grupos').corr(numeric_only = True, method = 'pearson'), annot=True, fmt=".1f");

# %% [markdown]
# ### 2 - DBSCAN
# 
# **Sobre o modelo:**
# O DBSCAN é um algoritmo que agrupa os dados com base em **densidade (alta concentração de dados)**. Muito bom para tirar ruídos. O agrupamentos dos dados é calculado com base nos core (quantidade de pontos mínmos que seja igual ou maior a definição do MinPts), border (ponto de fronteira dos dados) e noise (ruído).
# 
# **Hiperparametro:**
# Eps (raio ao redor de um dado). MinPts (mínimo de pontos dentro do raio para que seja agrupado).
# 
# **Vantagem:**
# Capacidade de trabalhar com outliers. Trabalha com base de dados grande.
# 
# **Desvantagem:**
# Dificuldade para lidar com cluster dentro de cluster. Dificuldade para lidar com dados de alta dimensionalidade. Dificuldade em encontrar o raio de vizinhança ao tentar agrupar dados com distância média muito distinta (clusters mais densos que outros).
# 
# Para mais informação: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html 

# %%
#Criando o modelo:
dbscan = DBSCAN(eps=10, min_samples=8)
#Ajustando aos dados
dbscan.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])

dbscan_labels = dbscan.labels_
dbscan_labels


# %% [markdown]
# Labels com -1 foram classificados como outliers

# %%
# Plotando o grafico:
plt.scatter(dados[['Annual Income (k$)']],dados[['Spending Score (1-100)']], c=dbscan_labels, alpha=0.5, cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# %%
# Plotando o grafico sem os outliers:
# mascara para outlier
mascara = dbscan_labels>=0

# plotando o gráfico
plt.scatter(dados[['Annual Income (k$)']][mascara],dados[['Spending Score (1-100)']][mascara], c=dbscan_labels[mascara], alpha=0.5, cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


# %% [markdown]
# Checando a quantidade de valores que foram classificados como Outliers:

# %%
list(mascara).count(False)

# %% [markdown]
# ## Como validar uma clusterização?
# 
# Temos dois tipos:
# - Interna: Quanto bom foi o meu agrupamento?
# - Externa: Como parecido estão os meus dois algoritmos comparados?

# %% [markdown]
# ### Avaliando o Desempenho dos Algoritmos
# 
# ### Tipo Externo:
# 
# (a) Usando o **Adjusted Rand Index**
# 
# Compara o desempenho quando forem fornecidos datasets com labels geradas de forma aleatória. Quando essas labels estão muito diferente, o valor se aproxima de 0, o que sugere um resultado negativo, ou seja, clusters não próximos.

# %% [markdown]
# Comparação entre K-Means e DBSCAN:

# %%
adjusted_rand_score(kmeans_labels,dbscan_labels)

# %% [markdown]
# #### Tipo interno:
# 
# (b) Avaliando a métrica de **Silhouette**
# 
# Mede o formato do cluster obtido: avalia a distância entre os centros dos clusters, nesse caso, queremos maximizar as distâncias)
# 
# Valores próximos a -1, significa clusters ruins, próximo a 1, clusters bem separados.

# %% [markdown]
# KMEANS:

# %%
silhouette_score(dados[['Annual Income (k$)','Spending Score (1-100)']],kmeans_labels)

# %% [markdown]
# DBSCAN:

# %%
silhouette_score(dados[['Annual Income (k$)','Spending Score (1-100)']],dbscan_labels)


