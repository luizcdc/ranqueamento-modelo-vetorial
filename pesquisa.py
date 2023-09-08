'''
Carregamos os documentos de um arquivo CSV chamado 'doc.csv' usando a biblioteca pandas e os armazenamos em um DataFrame chamado documentos.

Criamos um vetorizador TF-IDF usando TfidfVectorizer da biblioteca scikit-learn. Isso nos permite converter os textos dos resumos dos documentos em representações numéricas que podem ser usadas para calcular a similaridade entre os documentos.

A função rank_documentos é definida para realizar a pesquisa e classificar os documentos com base na consulta do usuário.

Na função rank_documentos, a consulta do usuário é vetorizada usando o vetorizador TF-IDF.

Calculamos a similaridade de cosseno entre a consulta vetorizada e todos os documentos vetorizados usando cosine_similarity.

Os resultados da similaridade são armazenados em uma lista de tuplas (índice do documento, similaridade) e classificados em ordem decrescente de similaridade.

E por fim, apresentamos o ranqueamento dos documentos na saída, incluindo o título, o link e a similaridade para cada documento. O ranqueamento começa em 1 para o documento mais relevante.
'''


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar a base de documentos a partir do CSV
documento_csv = 'doc.csv'
documentos = pd.read_csv(documento_csv)

# Criar um vetorizador TF-IDF para representar os documentos
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documentos['Abstract'])

# Função para realizar a pesquisa e classificar os documentos
def rank_documentos(query):
    # Vetorizar a consulta
    query_vector = tfidf_vectorizer.transform([query])

    # Calcular a similaridade de cosseno entre a consulta e os documentos
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Criar uma lista de tuplas (índice do documento, similaridade) e classificá-la
    documento_scores = list(enumerate(cosine_similarities))
    documento_scores = sorted(documento_scores, key=lambda x: x[1], reverse=True)

    # Apresentar o ranqueamento dos documentos
    for i, (index, score) in enumerate(documento_scores, start=1):
        documento = documentos.iloc[index]
        print(f"Rank {i}: Título: {documento['Title']}, Link: {documento['Link']}, Similaridade: {score:.2f}")

# Receber a consulta do usuário
query = input("Digite sua consulta: ")

# Realizar o ranqueamento dos documentos
rank_documentos(query)
