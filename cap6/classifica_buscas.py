import pandas as pd
from collections import Counter

# teste inicial: home, busca, logado => comprou
# home, busca
# home, logado
# busca, logado
# busca: 75% (8 testes)

df = pd.read_csv('buscas2.csv')

x_df = df[['home', 'busca', 'logado']]
y_df = df['comprou']

x_dummies_df = pd.get_dummies(x_df)
y_dummies_df = y_df

x = x_dummies_df.values
y = y_dummies_df.values

# Algoritmo de classicação das buscas
porcentagem_treino = 0.8
porcentagem_teste = 0.1

tamanho_treino = int(porcentagem_treino * len(y))
tamanho_teste = int(porcentagem_teste * len(y))
tamanho_validacao = len(y) - tamanho_treino - tamanho_teste

treino_dados = x[:tamanho_treino]
treino_marcacoes = y[:tamanho_treino]

fim_treino = tamanho_treino + tamanho_teste

teste_dados = x[tamanho_treino:fim_treino]
teste_marcacoes = y[tamanho_treino:fim_treino]

validacao_dados = x[fim_treino:]
validacao_marcacoes = y[fim_treino:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    
    modelo.fit(treino_dados, treino_marcacoes)
    
    resultado = modelo.predict(teste_dados)
    
    acertos = resultado == teste_marcacoes
    
    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)
    
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    
    mensagem = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto)

    print(mensagem)

    return taxa_de_acerto

from sklearn.naive_bayes import MultinomialNB
modelo_multinomial = MultinomialNB()
resultado_multinomial = fit_and_predict("MultinomialNB", modelo_multinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modelo_adaboost = AdaBoostClassifier()
resultado_adaboost = fit_and_predict("AdaBoostClassifier", modelo_adaboost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if(resultado_multinomial > resultado_adaboost):
    vencedor = modelo_multinomial
else:
    vencedor = modelo_adaboost

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes
    
    total_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = 100.0 * total_acertos / total_de_elementos

    mensagem = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
    print(mensagem)

teste_real(vencedor, validacao_dados, validacao_marcacoes)

# Algoritmo para verificar a taxa de acerto base
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)
