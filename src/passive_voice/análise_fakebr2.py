import spacy

nlp = spacy.load("pt_core_news_sm")

def detectar_voz_passiva(texto):
    doc = nlp(texto)
    for token in doc:
        if token.dep_ in ("aux:pass", "nsubj:pass"):
            return "true"
    return "fake"

def detectar_evidencialidade(texto):
    doc = nlp(texto)
    verbos_evidenciais = {
        "dizer", "relatar", "comentar", "alegar", "acreditar", "informar",
        "afirmar", "revelar", "indicar", "reportar", "sugerir",
        "declarar", "anunciar", "noticiar", "especular", "constatar",
        "garantir", "confirmar", "negar", "alertar", "prever", "apontar",
        "mencionar", "registrar", "explicar",
        "enfatizar", "ressaltar", "chamar atenção", "assinalar", "levantar", "questionar",
        "interpretar", "avaliar", "julgar", "argumentar", "observar", "sustentar",
        "ressaltou", "ponderar", "considerar", "estimular", "reforçar", "expor",
        "acrescentar", "justificar", "contestar", "detalhar", "ilustrar", "citar",
        "divulgar", "concluir", "pronunciar", "testemunhar", "comprovar",
        "opinar", "sublinhar", "explicitar", "narrar", "incluir", "insistir"
    }

    for token in doc:
        if token.lemma_ in verbos_evidenciais and token.pos_ == "VERB":
            sujeito = [child for child in token.children if child.dep_ in ("nsubj", "nsubj:pass")]
            if not sujeito:
                return "true"
    return "fake"

conectores_argumentativos = {
    "portanto", "porém", "além disso", "no entanto", "contudo", "todavia",
    "assim", "dessa forma", "por conseguinte", "ou seja", "em outras palavras",
    "logo", "então", "por isso", "isto é", "desse modo"
}

def detectar_conectores_argumentativos(texto):
    doc = nlp(texto)
    for token in doc:
        if token.text.lower() in conectores_argumentativos:
            return "true"
    return "fake"



