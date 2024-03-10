# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# lê o arquivo CSV já passando o parse_dates como 0, para transformá-lo em datetime
# transforma a coluna DATE em index
df = pd.read_csv('https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/03_analise_de_series_temporais/bases/Electric_Production.csv', parse_dates=[0], index_col="DATE")
df.head()

# %%
df.info()


