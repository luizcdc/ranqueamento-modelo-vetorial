"""
Sistema de Recuperação de Informação - Modelo Vetorial com Relevance Feedback (Rocchio Algorithm)

Autores:

    * Alberto Lucas
    * Luiz Cavalcanti

Permite a realização de consultas em uma base de documentos e ranqueamento dos documentos de acordo
com a consulta, utilizando o modelo vetorial. A relevância dos documentos pode ser indicada pelo
usuário, que pode então realizar uma nova consulta com base nos documentos relevantes.

Exemplo de uso:

    python pesquisa.py -q "machine learning" # Consulta não ponderada
    python pesquisa.py # Query e ponderação inseridas pelo usuário
"""
import random
import pandas as pd
import numpy as np
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar a base de documentos a partir do CSV
documento_csv = "doc.csv"
documentos = pd.read_csv(documento_csv)
rocchio_offset = None # aqui temos o vetor resultante do algoritmo de Rocchio, que é
# persistido entre as consultas

ALPHA = 0.75
BETA = 0.25

def calc_rocchio_offset(relevant: list, irrelevant: list):
    """
    Calcula o vetor resultante do algoritmo de Rocchio, que é utilizado para a ponderação da consulta.

    :param relevant: Lista de documentos relevantes
    :param irrelevant: Lista de documentos irrelevantes
    :return: Vetor resultante do algoritmo de Rocchio
    """
    global rocchio_offset
    
    relevant_part = ALPHA * np.sum(relevant, axis=0)
    irrelevant_part = BETA * np.sum(irrelevant, axis=0)
    rocchio_offset = relevant_part - irrelevant_part

def rank_documentos(query: str) -> list[tuple]:
    """
    Realiza o ranqueamento dos documentos de acordo com a query de consulta.

    :param query: Consulta a ser realizada
    :return: Lista de tuplas (índice do documento, título do documento, link do documento, similaridade)
    """
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
    )

    # combina dois passos: calcular a frequência de cada termo da consulta na base de pesquisa e calcular o TF-IDF
    # de cada documento
    tfidf_matrix = tfidf_vectorizer.fit_transform(documentos["Abstract"])

    # Vetorizar a consulta, caso os pesos não tenham sido especificados manualmente
    query_vector = tfidf_vectorizer.transform([query.lower()])

    if rocchio_offset is not None:
        query_vector = query_vector + rocchio_offset

    # Calcular a similaridade entre a consulta e os documentos
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Ordena os documentos de acordo com a similaridade
    documento_scores = list(enumerate(cosine_similarities))
    documento_scores = sorted(documento_scores, key=lambda x: x[1], reverse=True)

    resultados = []
    for rank, (index, score) in enumerate(documento_scores, start=1):
        documento = documentos.iloc[index]
        resultados.append((rank, index, documento["Title"], documento["Link"], score))

    return resultados, tfidf_matrix


def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("-q", "--query", help="Query da busca", type=str, required=False)

    args = argparser.parse_args()

    if args.query:
        query = args.query
    else:
        query = input("Digite sua frase de consulta: ")

    resultados, tfidf_matrix = rank_documentos(query)
    
    resultados = resultados[:5]
    for rank, index, title, link, score in resultados:
        print(f"({score:.2f}): Doc. {index}. Título: {title[0:150]}")
    
    if "s" in input("Deseja indicar a relevância dos documentos? S/n").lower():
        relevantes = []
        irrelevantes = []
        for rank, index, title, link, score in resultados:
            if "s" in input(f"O documento {index} é relevante? S/n").lower():
                relevantes.append(tfidf_matrix[index])
            else:
                irrelevantes.append(tfidf_matrix[index])
        calc_rocchio_offset(relevantes, irrelevantes)
        resultados, tfidf_matrix = rank_documentos(query)
        for rank, index, title, link, score in resultados[:5]:
            print(f"{rank}º: ({score:.2f}): Doc. {index}. Título: {title[0:150]}")

if __name__ == "__main__":
    main()
