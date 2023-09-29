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
