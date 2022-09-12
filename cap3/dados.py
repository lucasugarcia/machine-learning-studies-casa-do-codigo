import csv

def carregar_buscas():
    # Dados
    x = []

    # Marcações
    y = []

    arquivo = open('buscas.csv', 'r')

    leitor = csv.reader(arquivo)

    next(leitor)

    for home, busca, logado, comprou in leitor:
        dado = [int(home), busca, int(logado)]

        x.append(dado)
        y.append(int(comprou))

    return x, y