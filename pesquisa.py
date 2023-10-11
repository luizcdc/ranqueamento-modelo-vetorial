"""
Sistema de Recuperação de Informação - Modelo Booleano Extendido

Autores:

    * Alberto Lucas
    * Luiz Cavalcanti

Permite a realização de consultas em uma base de documentos e ranqueamento dos documentos de acordo com a consulta,
utilizando o modelo vetorial. É possível especificar pesos para os termos da consulta, opcionalmente.

Exemplo de uso:

    python pesquisa.py -q "machine learning" -w # Consulta ponderada, pesos especificados pelo usuário
    python pesquisa.py -q "machine learning" # Consulta não ponderada
    python pesquisa.py # Query e ponderação inseridas pelo usuário

Usando a biblioteca pandas, carregamos os documentos da base de documentos 'doc.csv', contendo título, link e abstract
de cada documento, e os armazenamos no DataFrame `documentos`.

Caso os pesos de cada termo da consulta não sejam especificados manualmente, os termos são ponderados através de TF-IDF
com base nas frequências de termos na base de documentos, resultando em um vetor representativo da consulta.

A função rank_documentos realiza a pesquisa e classifica os documentos com base na consulta do usuário.

Instanciamos um vetorizador TF-IDF usando TfidfVectorizer da biblioteca scikit-learn, utilizando como vocabulário base
para o espaço vetorial os termos presentes na consulta do usuário. Isso nos permite converter os textos dos abstracts
dos documentos em representações numéricas vetoriais que serão usadas para calcular a similaridade entre os documentos.

Calculamos a similaridade entre a consulta vetorizada e cada um os documentos vetorizados usando cosine_similarity,
que, pela natureza da função cosseno aponta maior similaridade quanto menor o ângulo entre os vetores. Como os vetores
têm seus valores sempre positivos, o ângulo varia entre 0 e 90 graus.

A função retorna então uma lista de tuplas (índice do documento, título do documento, link do documento, similaridade)
ordenada de forma decrescente pela similaridade.

Por fim, apresentamos o ranqueamento dos documentos na saída, incluindo o título e a similaridade para cada documento.
"""
import pandas as pd
import numpy as np
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vetorial import rank_documentos

import booleano_extendido

# Carregar a base de documentos a partir do CSV
documento_csv = "doc.csv"
documentos = pd.read_csv(documento_csv)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-m",
        "--modelo",
        help="Modelo de busca a ser utilizado",
        choices=["booleano", "vetorial"],
    )
    argparser.add_argument(
        "-q", "--query", help="Query da busca", type=str, required=False
    )
    argparser.add_argument(
        "-w",
        "--weighted",
        help="Ativa a especificação de pesos para as palavras da busca",
        action="store_true",
        required=False,
    )

    args = argparser.parse_args()

    if args.query:
        query = args.query
    else:
        if args.modelo == "vetorial":
            args.weighted = (
                input("Deseja ponderar os termos da consulta? (s/n): ").strip() == "s"
            )

    if args.modelo == "booleano":
        print("Modelo booleano selecionado.")
        print("São permitidos buscas de:")
        # Create a dictionary mapping options to functions
        print(" 1 - Term Weighting: Ponderação de Termos")
        print(" 2 - Proximity Searching: Busca por Proximidade")
        print(" 3 - Fuzzy Matching: Correspondência Difusa")
        print(" 4 - Conceptual Searching: Busca Conceitual")
        print(" 5 - Field-Based Searching: Busca Baseada em Campo")
        print(" 6 - Exclusion Searches: Pesquisas de Exclusão")
        # Get user input
        choice = int(input("Digite a opção que deseja: "))
        match choice:
            case 1:
                print("Exemplo: ", "0.1 / 0.2 / machine / learning / and")
                weight1 = float(input("Digite o peso para a primeira palavra da consulta: "))
                weight2 = float(input("Digite o peso para a segunda palavra da consulta: "))
                query1 = input("Digite a primeira palavra da consulta: ")
                query2 = input("Digite a segunda palavra da consulta: ")
                and_or = input("Digite o tipo de operação: or/and/not: ")
                booleano_extendido.term_weighting_search(weight1, weight2, query1, query2, and_or)
            case 2:
                print("Exemplo: ", "Academic / attention / 8")
                query1 = input("Digite a primeira palavra da consulta: ")
                query2 = input("Digite a segunda palavra da consulta: ")
                distance = input("Digite a distância entre os termos: ")
                booleano_extendido.proximity_search_operation(query1, query2, distance)
            case 3:
                print("Exemplo: ", "artiifcial")
                query = input("Digite sua frase de consulta: ")
                booleano_extendido.fuzzy_search_operation(query)
            case 4:
                print("Exemplo: ", "pedagogy")
                query = input("Digite sua frase de consulta: ")
                booleano_extendido.conceptual_search_operation(query)
            case 5:
                print("Exemplo: ", "Cryptographic / Title")
                #print("Exemplo: ", "Liability")
                query = input("Digite sua frase de consulta: ")
                field = input("Digite o campo que deseja pesquisar: Title/Link/Abstract: ")
                booleano_extendido.field_based_search_operation(query, field)
            case 6:
                print("Exemplo: ", "/ artificial intelligence -deep learning /")
                query = input("Digite sua frase de consulta: ")
                booleano_extendido.exclusion_search_operation(query)
        return
    elif args.modelo == "vetorial":
        query = input("Digite sua frase de consulta: ")

        weights = vocab_indexes = None
        if args.weighted:
            stop_words = set(TfidfVectorizer(stop_words="english").get_stop_words())
            split_query = list(dict.fromkeys(query.split(" ")))
            weights = {}
            print("Digite pesos no intervalo [0, 1) para cada termo da consulta.\n")
            for word in split_query:
                if word not in stop_words:
                    weight = input(f"Digite o peso para o termo {word}: 0.")
                    weights[word] = float(f"0.{weight}")
                else:
                    weights[word] = 0.0
            print()

            vocab_indexes = {word: i for i, word in enumerate(split_query)}

            # weights precisam ser convertidos em  <1xK sparse matrix of type '<class 'numpy.float64'>'
            # in Compressed Sparse Row format>, onde K é o número de termos distintos da consulta
            weights = np.array([weights[word] for word in split_query]).reshape(1, -1)

        resultados = rank_documentos(query, vocab_indexes, weights)
    else:
        raise NotImplementedError("Modelo não implementado")

    for i, title, link, score in resultados:
        print(f"Rank {i} ({score:.2f}): Título: {title[0:150]}")


if __name__ == "__main__":
    main()
