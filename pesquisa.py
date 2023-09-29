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

# Carregar a base de documentos a partir do CSV
documento_csv = "doc.csv"
documentos = pd.read_csv(documento_csv)


def rank_documentos(
    query: str, vocab_indexes: dict = None, weights: np.array = None
) -> list[tuple]:
    """
    Realiza o ranqueamento dos documentos de acordo com a query de consulta e os pesos, se fornecidos.

    :param query: Consulta a ser realizada
    :param vocab_indexes: Dicionário de índices do vocabulário (necessário pois os pesos foram construídos manualmente)
    :param weights: Pesos para os termos da consulta
    :return: Lista de tuplas (índice do documento, título do documento, link do documento, similaridade)
    """
    assert (vocab_indexes is not None and weights is not None) or (
        vocab_indexes is None and weights is None
    )
    tfidf_vectorizer = TfidfVectorizer(
        vocabulary=list(dict.fromkeys(query.lower().split(" ")))
        if vocab_indexes is None
        else vocab_indexes,
        stop_words="english",
        # sublinear_tf=True
    )

    # combina dois passos: calcular a frequência de cada termo da consulta na base de pesquisa e calcular o TF-IDF
    # de cada documento
    tfidf_matrix = tfidf_vectorizer.fit_transform(documentos["Abstract"])

    # Vetorizar a consulta, caso os pesos não tenham sido especificados manualmente
    query_vector = (
        weights if weights is not None else tfidf_vectorizer.transform([query.lower()])
    )

    # Calcular a similaridade entre a consulta e os documentos
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Ordena os documentos de acordo com a similaridade
    documento_scores = list(enumerate(cosine_similarities))
    documento_scores = sorted(documento_scores, key=lambda x: x[1], reverse=True)

    resultados = []
    for rank, (index, score) in enumerate(documento_scores, start=1):
        documento = documentos.iloc[index]
        resultados.append((rank, documento["Title"], documento["Link"], score))

    return resultados


def main():
    argparser = argparse.ArgumentParser()

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
        query = input("Digite sua frase de consulta: ")
        args.weighted = (
            input("Deseja ponderar os termos da consulta? (s/n): ").strip() == "s"
        )

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

    for i, title, link, score in resultados:
        print(f"Rank {i} ({score:.2f}): Título: {title[0:150]}")


if __name__ == "__main__":
    main()
