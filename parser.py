import re


class SyntaxError(ValueError):
    pass


def parse(query: str) -> list:
    """Faz o parsing da gramática da consulta e retorna uma pilha em notação polonesa."""
    stack_tokens = []
    size = len(query)
    currpos = 0
    while currpos < size:
        tipo, token, currpos = next_token(query, currpos)
        if tipo == "":
            break
        if tipo == "SPACE":
            continue
        stack_tokens.append(token)
    return stack_tokens


def next_token(query: str, i: int) -> tuple[str, str | float, int]:
    """
    Possible tokens:
    " ", "AND", "OR", "<", ">", "'", a word/phrase (between ''), or a float (between <>)
    """

    if i >= len(query):
        return "", "", i

    if query[i] == " ":
        return "SPACE", " ", i + 1

    if query[i] == "'":
        w = re.search(r"'(([^']|\\')*)'", query[i:]).group(1)
        w.replace(r"\'", "'")
        return "TOKEN", f"'{w}'", i + len(w) + 2

    if query[i] == "<":
        num = re.search(r"<([\d.,]*)>", query[i:]).group(1)
        l = len(num) + 2
        num = float(num.replace(",", "."))
        return "FLOAT", num, i + l

    if query[i : i + 3] == "AND":
        return "AND", "AND", i + 3

    if query[i : i + 2] == "OR":
        return "OR", "OR", i + 2

    raise SyntaxError("Erro de parsing da consulta")
