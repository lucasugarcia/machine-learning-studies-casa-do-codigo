import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from collections import Counter

df = pd.read_csv('buscas.csv')

x_df = df[['home', 'busca', 'logado']]
y_df = df['comprou']

x_dummies_df = pd.get_dummies(x_df)
y_dummies_df = y_df

x = x_dummies_df.values
y = y_dummies_df.values

# Algoritmo de classicação das buscas
porcentagem_treino = 0.9
tamanho_treino = int(porcentagem_treino * len(y))
tamanho_teste = len(y) - tamanho_treino

treino_dados = x[:tamanho_treino]
treino_marcacoes = y[:tamanho_treino]

teste_dados = x[-tamanho_teste:]
teste_marcacoes = y[-tamanho_teste:]

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

acertos = resultado == teste_marcacoes

total_acertos = sum(acertos)
total_elementos = len(teste_dados)

taxa_acerto = 100.0 * (total_acertos / total_elementos)

print("Taxa de acerto do algoritmo: %f" % taxa_acerto)
print(total_elementos)

# Algoritmo para verificar a taxa de acerto base
acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_base = 100.0 * (acerto_base / len(teste_marcacoes))

print("Taxa de acerto base: %f" % taxa_de_acerto_base)