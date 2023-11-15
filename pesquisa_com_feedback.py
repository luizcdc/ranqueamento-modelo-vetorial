"""
Sistema de Recuperação de Informação - Modelo Vetorial com Relevance Feedback

Autores:

    * Alberto Lucas
    * Luiz Cavalcanti

Permite a realização de consultas em uma base de documentos e ranqueamento dos documentos de acordo
com a consulta, utilizando o modelo vetorial.

Exemplo de uso:

    python pesquisa.py -q "machine learning" # Consulta não ponderada
    python pesquisa.py # Query e ponderação inseridas pelo usuário

Utilizamos a biblioteca pandas para carregar títulos, links e resumos de 'doc.csv' em um DataFrame 
documentos. 

Termos são ponderados via TF-IDF, gerando vetores de consulta. A função rank_documentos classifica 
os documentos com base nesta consulta.

Utilizamos TfidfVectorizer da scikit-learn com os termos da consulta como vocabulário, convertendo
os resumos em vetores numéricos. A similaridade entre a consulta e os documentos é calculada usando
cosine_similarity.

A função rank_documentos retorna uma lista de tuplas com índice, título, link e similaridade dos 
documentos, ordenados decrescentemente. Os resultados são exibidos com título e similaridade de 
cada documento.
"""
import pandas as pd
import numpy as np
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar a base de documentos a partir do CSV
documento_csv = "doc.csv"
documentos = pd.read_csv(documento_csv)


def rank_documentos(query: str) -> list[tuple]:
    """
    Realiza o ranqueamento dos documentos de acordo com a query de consulta.

    :param query: Consulta a ser realizada
    :return: Lista de tuplas (índice do documento, título do documento, link do documento, similaridade)
    """
    tfidf_vectorizer = TfidfVectorizer(
        vocabulary=list(dict.fromkeys(query.lower().split(" "))),
        stop_words="english",
    )

    # combina dois passos: calcular a frequência de cada termo da consulta na base de pesquisa e calcular o TF-IDF
    # de cada documento
    tfidf_matrix = tfidf_vectorizer.fit_transform(documentos["Abstract"])

    # Vetorizar a consulta, caso os pesos não tenham sido especificados manualmente
    query_vector = tfidf_vectorizer.transform([query.lower()])

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

    argparser.add_argument("-q", "--query", help="Query da busca", type=str, required=False)

    args = argparser.parse_args()

    if args.query:
        query = args.query
    else:
        query = input("Digite sua frase de consulta: ")

    resultados = rank_documentos(query)

    for i, title, link, score in resultados[:5]:
        print(f"Rank {i} ({score:.2f}): Título: {title[:150]}")


if __name__ == "__main__":
    main()
