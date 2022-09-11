from sklearn.naive_bayes import MultinomialNB

# [é gordinho?, tem perna curta?, late?]
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]

cachorro1 = [1, 1, 1]
cachorro2 = [0, 1, 1]
cachorro3 = [0, 1, 1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

# 1 representa porco
# -1 representa cachorro
marcacoes = [1, 1, 1, -1, -1, -1]

animal_misterioso1 = [1, 1, 1]
animal_misterioso2 = [1, 0, 0]
animal_misterioso3 = [0, 0, 1]

teste = [animal_misterioso1, animal_misterioso2, animal_misterioso3]

marcacoes_teste = [-1, 1, -1]

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

resultado = modelo.predict(teste)

diferencas = resultado - marcacoes_teste

acertos = [d for d in diferencas if d == 0]

total_acertos = len(acertos)

total_elementos = len(teste)

taxa_de_acerto = 100 * (total_acertos / total_elementos)

print(resultado)
print(diferencas)
print(taxa_de_acerto)