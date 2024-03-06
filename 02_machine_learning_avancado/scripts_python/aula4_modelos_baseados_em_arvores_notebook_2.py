# %% [markdown]
# ## Modelos baseados em árvores
# 
# Nesse notebook, você irá aprender como aplicar modelos baseados em árvores no case de análise de transações fraudulentas de cartões de crédito.

# %%
import pandas as pd                      
import matplotlib.pyplot as plt          
import seaborn as sns                    
import numpy as np                        

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree  
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

# %%
# subindo a base de dados
dados = pd.read_csv("https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/02_machine_learning_avancado/bases/card_transdata.csv", sep=",")

# %%
dados.head()

# %%
dados.isnull().sum() 

# %%
# limpando dados nulos
dados = dados.dropna()

# %%
dados.isnull().sum() 

# %%
# analisando correlações
correlation_matrix = dados.corr().round(2)

fig, ax = plt.subplots(figsize=(8,8))    
sns.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)

# %%
# Separando os dados
x = dados.drop(columns=['fraud'])
y = dados['fraud'] # O que eu quero prever. (Target)

# %%
# Separando em bases de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=7) 

# %% [markdown]
# ## Criando modelos de árvores:
# 
# 
# ### Decion Tree
# Vamos criar um modelo simples, utilizando todas as variáveis do modelo.

# %%
dt = DecisionTreeClassifier(random_state=7, criterion='entropy', max_depth = 2)

# %%
dt.fit(x_train, y_train)

# %%
y_predito = dt.predict(x_test) 

# %%

tree.plot_tree(dt)

# %%
class_names = ['Fraude', 'Não Fraude']
label_names = ['distance_from_home', 'distance_from_last_transaction',	'ratio_to_median_purchase_price',	'repeat_retailer',	'used_chip',	'used_pin_number',	'online_order']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,15), dpi=300)

tree.plot_tree(dt,
               feature_names = label_names, 
               class_names=class_names,
               filled = True)

fig.savefig('imagename.png')

# %%
# Metricas de precisão, revocação, f1-score e acurácia.
print(accuracy_score(y_test, y_predito)) #relatório de validação das métrica de desempenho.

# %% [markdown]
# ### Random Forest

# %%
rf = RandomForestClassifier(n_estimators=5, max_depth = 2,  random_state=7) 

rf.fit(x_train, y_train) 



# %%
estimator = rf.estimators_

# %%
y_predito_random_forest = rf.predict(x_test) 

# %%
# Metricas de precisão, revocação, f1-score e acurácia.
print(accuracy_score(y_test, y_predito_random_forest)) # relatório de validação das métrica de desempenho.

# %%
class_names = ['Fraude', 'Não Fraude']
label_names = ['distance_from_home', 'distance_from_last_transaction',	'ratio_to_median_purchase_price',	'repeat_retailer',	'used_chip',	'used_pin_number',	'online_order']


# %%
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[0],
               feature_names = label_names, 
               class_names=class_names,
               filled = True);
fig.savefig('rf_individualtree.png')

# %% [markdown]
# Plotando todas as árvores geradas:

# %%
fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for index in range(0, 5):
    tree.plot_tree(rf.estimators_[index],
                   feature_names = label_names, 
                   class_names=class_names,
                   filled = True,
                   ax = axes[index]);

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_5trees.png')

# %%
print (rf.score(x_train, y_train)) 
print(rf.score(x_test, y_test))

# %% [markdown]
# O **score** nos dá uma visão da precisão média da floresta aleatória nos dados fornecidos


