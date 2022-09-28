import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter

# teste inicial: home, busca, logado => comprou
# home, busca
# home, logado
# busca, logado
# busca: 75% (8 testes)

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
fit_and_predict(modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

modelo = AdaBoostClassifier()
fit_and_predict(modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

# Algoritmo para verificar a taxa de acerto base
acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_base = 100.0 * (acerto_base / len(teste_marcacoes))

print("Taxa de acerto base: %f" % taxa_de_acerto_base)

def fit_and_predict(modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    
    modelo.fit(treino_dados, treino_marcacoes)
    
    resultado = modelo.predict(teste_dados)
    
    acertos = resultado == teste_marcacoes
    
    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)
    print(total_de_elementos)
    
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    
    print("Taxa de acerto do algoritmo: %f" % taxa_de_acerto)