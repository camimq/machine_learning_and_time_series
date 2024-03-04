# %% [markdown]
# # Fraude de cartão de crédito

# %% [markdown]
# ## Fonte de Dados:

# %% [markdown]
# - Os pagamentos digitais estão evoluindo, mas os criminosos cibernéticos também.
# - De acordo com o Data Breach Index, mais de 5 milhões de registros são roubados diariamente, uma estatística preocupante que mostra - a fraude ainda é muito comum tanto para pagamentos do tipo Cartão-Presente quanto Cartão-Não Presente.
# - No mundo digital de hoje, onde trilhões de transações com cartões acontecem por dia, a detecção de fraudes é um desafio.

# %% [markdown]
# ## Explicação do das variáveis:

# %% [markdown]
# - `distancefromhome`: a distância de casa onde a transação aconteceu.
# - `distancefromlast_transaction`: a distância da última transação aconteceu.
# - `ratiotomedianpurchaseprice`: Razão da transação do preço de compra para o preço de compra mediano.
# - `repeat_retailer`: É a transação que aconteceu do mesmo varejista.
# - `used_chip`: É a transação através de chip (cartão de crédito).
# - `usedpinnumber`: A transação aconteceu usando o número PIN.
# - `online_order`: A transação é um pedido online.
# - `fraude`: A transação é fraudulenta.

# %%
import pandas as pd # Para trabalhar com dados tabulares
from sklearn.model_selection import train_test_split #separação dos dados
from sklearn.neighbors import KNeighborsClassifier #modelo de machine learning classificação
from sklearn.metrics import accuracy_score #avaliação do modelo
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score #métrica de avaliação
from sklearn.preprocessing import StandardScaler, MinMaxScaler #Feature Engineer
import matplotlib.pyplot as plt #gráficos
import seaborn as sns #gráficos
import numpy as np #transformação dos dados
import warnings #remoção de avisos

# %%
dados = pd.read_csv('https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/02_machine_learning_avancado/bases/card_transdata.csv', sep=',') 

# %%
dados.head(3) # analisando os primeiros dados

# %%
dados.shape 

# %% [markdown]
# ## Tratando inconsistências na base

# %%
dados.isnull().sum() 

# %%
dados = dados.dropna()

# %% [markdown]
# ## Análise exploratória de dados

# %%
dados.describe()

# %%
#Número de transações fraudulentas
dados[dados["fraud"] == 1].fraud.count() #filtro com contagem dos dados

# %%
Total = len(dados)
Total

# %%
Total = len(dados)
TotalNaoFraudes = dados[dados["fraud"] == 0].fraud.count()
TotalFraudes = dados[dados["fraud"] == 1].fraud.count()

Percentual_Fraudes = TotalFraudes / Total 

print("Total de dados: ", Total)
print("Total de não fraudes: ", TotalNaoFraudes)
print("Total de fraudes: ", TotalFraudes)
print("Percentual de fraudes na base: ", (round(Percentual_Fraudes, 2)*100), "%")

# %%
categororias = ["Non-Fraud", "Fraud"]
plt.pie(dados["fraud"].value_counts(), labels = categororias, autopct = "%.0f%%", explode= (0, 0.1), colors = ("g", "r"))
plt.show()

# %%
dados_fraudes = dados[dados["fraud"] == 1]

# %%
dados_fraudes.head(5)

# %%
dados_fraudes.describe().T

# %%
plt.figure(figsize = (10,10)) #Configurando o tamanho da visualização

plt.subplot(2,2,1)
sns.countplot(x = "repeat_retailer", palette = "Paired", data = dados_fraudes) #Aconteceu no mesmo varejista?

plt.subplot(2,2,2)
sns.countplot(x = "used_chip", palette = "Paired", data = dados_fraudes) #Uso de cartão de crédito?

plt.subplot(2,2,3)
sns.countplot(x = "used_pin_number", palette = "Paired", data = dados_fraudes) #Utilizou o mesmo número de PIN?

plt.subplot(2,2,4)
sns.countplot(x = "online_order", palette = "Paired", data = dados_fraudes) #Foi em uma compra online?

# %%
Colunas_Numericas = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price"]
for column in Colunas_Numericas:
    plt.figure()
    plot = dados_fraudes[column]
    sns.histplot(plot, bins=10, kde=True)
    plt.show()

# %%
for column in [0, 1, 2]:
    dados_fraudes.iloc[:, column] = np.log10(dados_fraudes.iloc[:, column]) #transformação logarítma

# %%
Colunas_Numericas_Normal = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price"]
for column in Colunas_Numericas_Normal:
    plt.figure()
    plot = dados_fraudes[column]
    sns.histplot(plot, bins=10, kde=True)
    plt.show()

# %%
round(dados_fraudes.distance_from_last_transaction.mean(),2)

# %%
round(dados_fraudes.distance_from_last_transaction.std(),2)

# %%
correlation_matrix = dados.corr().round(2)

fig, ax = plt.subplots(figsize=(8,8))    
sns.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)

# %% [markdown]
# ## Criação do modelo de Machine Learning

# %% [markdown]
# ### Separação da base de treino e teste

# %%
x = dados[['distance_from_home','ratio_to_median_purchase_price', 'online_order']]
y = dados['fraud'] #target

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=7) #20% para teste e 80% de treino

# %% [markdown]
# ### Feature Egineering

# %% [markdown]
# Comparação da escala normal das variáveis vs. escalonamento das variáveis
# 
# - Comparação do real x dados transformado (normalização e padronização)
# - padronização: zera a média e deixa o desvio padrão unitário.( obteremos desvios-padrão menores por meio do processo de normalização minmaxscaler).
# normalização: coloca a variável na escala entre 0 até 1.
# - Análise da plotagem real e verificar se está muito diferente da plotagem com standerscaler e minmaxsclaer.
# - Escolha o tipo de transformação de escala que melhor se adequa a suas variáveis. Se o desenho do gráfico mudar, você está descaracterizando o dado.
# - Transformar e normlaizar a escala das variáveis.

# %%
#scaler = StandardScaler() 
scaler = MinMaxScaler() 
scaler.fit(x_train)

# %%
# Na hora de transformar, devemos transformar ambos os conjuntos
x_train_escalonado = scaler.transform(x_train)#treino
x_test_escalonado = scaler.transform(x_test)#teste

# %%
x_train

# %%
x_train_escalonado

# %% [markdown]
# ## Configurando o modelo

# %%
error = []

# Calculating error for K values between 1 and 10
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_escalonado, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

# %%
plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# %%
# Hiperparametro do nosos modelo é o número de vizinhos considerado (n_neighbors)
modelo_classificador = KNeighborsClassifier(n_neighbors=5)

# %%
# Está fazendo o treinamento do meu modelo de ML
modelo_classificador.fit(x_train_escalonado, y_train)

# %%
y_predito = modelo_classificador.predict(x_test_escalonado) #defininfo as predições

# %%
print('Training set score: {:.4f}'.format(modelo_classificador.score(x_train, y_train)))
print('Test set score: {:.4f}'.format(modelo_classificador.score(x_test, y_test)))

# %%
# Acurácia do modelo
print(accuracy_score(y_test, y_predito))


