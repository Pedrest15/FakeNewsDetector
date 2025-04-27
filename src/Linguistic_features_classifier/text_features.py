import spacy, pyphen
from string import punctuation

nlp = spacy.load("pt_core_news_sm")
dic = pyphen.Pyphen(lang="pt_BR")
palavras_familiares = nlp.Defaults.stop_words

def contar_silabas(palavra):
    s = dic.inserted(palavra)
    return 1 if '-' not in s else len(s.split('-'))

def eh_palavra_dificil(token):
    if not token.is_alpha:
        return False
    if token.pos_ == "PROPN":
        return False
    if '-' in token.text:
        return False
    if token.text.lower() in palavras_familiares:
        return False
    return contar_silabas(token.text) >= 3

def contar_palavras_doc(doc):
    return sum(1 for t in doc if not t.is_space and not t.is_punct)

def contar_palavras_unicas_doc(doc):
    words_alpha_ling = [t.text.lower() for t in doc if t.is_alpha and not t.is_space and not t.is_punct]
    unique_words_ling = set(words_alpha_ling)
    return len(unique_words_ling)

def contar_verbos(tokens_ling):
    return sum(1 for t in tokens_ling if t.pos_ == "VERB")

def contar_substantivos(tokens_ling):
    return sum(1 for t in tokens_ling if t.pos_ in ("NOUN", "PROPN"))

def contar_adjetivos(tokens_ling):
    return sum(1 for t in tokens_ling if t.pos_ == "ADJ")

def contar_adverbios(tokens_ling):
    return sum(1 for t in tokens_ling if t.pos_ == "ADV")

def contar_pronome(tokens_ling):
    return sum(1 for t in tokens_ling if t.pos_ == "PRON")

def contar_stopwords(tokens_ling):
    return sum(1 for t in tokens_ling if t.is_stop)

def numero_de_sentencas_doc(doc):
    return len(list(doc.sents))

def palavras_por_sentenca_doc(doc):
    num_sent = numero_de_sentencas_doc(doc)
    total_tokens = contar_palavras_doc(doc)
    return total_tokens / num_sent if num_sent > 0 else 0

def contar_palavras_complexas(tokens_ling):
    return sum(1 for t in tokens_ling if eh_palavra_dificil(t))

def pausalidade_doc(num_punct, num_sentencas):
    """
    Razão entre sinais de pontuação e sentenças.
    """
    return num_punct / num_sentencas

def informalidade_doc(doc, total_palavras):
    return sum(1 for t in doc if t.is_oov) / total_palavras if total_palavras > 0 else 0

def proporcao_entidades_especificas(doc, total_palavras):
    tipos_especificos = {"DATE", "TIME", "GPE", "LOC", "CARDINAL", "ORDINAL"}
    ents_especificas = [ent for ent in doc.ents if ent.label_ in tipos_especificos]
    return len(ents_especificas) / total_palavras if total_palavras > 0 else 0

def contar_exclamacoes(doc):
    return sum(1 for t in doc if t.text == '!')

    # ------ Métricas ------
def extrair_metricas_otimizado(doc):
    """
    Recebe um objeto spaCy Doc e extrai as métricas de legibilidade/complexidade.
    É uma versão equivalente a 'extrair_metricas_completas',
    mas sem chamar nlp(texto)' dentro.
    """

    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    tokens_with_punct = [t for t in doc]

    total_tokens_with_punct = len(tokens_with_punct)
    total_tokens = len(tokens)
    total_palavras = contar_palavras_doc(doc)

    num_sentences = numero_de_sentencas_doc(doc)
    punctuation = [t for t in doc if t.is_punct]
    total_punctuation = len(punctuation)

    # substitui "text, tokens_ling" pelas versões doc-based.
    n_type_token_ratio = (contar_palavras_unicas_doc(doc) / total_palavras) if total_palavras > 0 else 0
    size_of_sentences_in_words = (total_palavras / num_sentences) if num_sentences > 0 else 0
    n_verb_to_token_ratio = (contar_verbos(tokens) / total_tokens_with_punct) if total_tokens_with_punct > 0 else 0
    n_noun_to_token_ratio = (contar_substantivos(tokens) / total_tokens_with_punct) if total_tokens_with_punct > 0 else 0
    n_adjective_to_token_ratio = (contar_adjetivos(tokens) / total_tokens_with_punct) if total_tokens_with_punct > 0 else 0
    n_adverb_to_token_ratio = (contar_adverbios(tokens) / total_tokens_with_punct) if total_tokens_with_punct > 0 else 0
    n_pronoun_to_token_ratio = (contar_pronome(tokens) / total_tokens_with_punct) if total_tokens_with_punct > 0 else 0
    n_stopword_to_token_ratio = (contar_stopwords(tokens) / total_tokens_with_punct) if total_tokens_with_punct > 0 else 0
    pausality = pausalidade_doc(len(punctuation), num_sentences)
    tokens_per_sentence = palavras_por_sentenca_doc(doc)

    # complexidade
    total_silabas = sum(contar_silabas(t.text) for t in tokens)
    media_silabas_palavra = (total_silabas / total_palavras) if total_palavras > 0 else 0
    num_palavras_complexas_ = contar_palavras_complexas(tokens)
    percentual_palavras_complexas_ = (num_palavras_complexas_ / total_palavras) * 100 if total_palavras > 0 else 0

    gunning_fog_ = 0
    if total_palavras > 0 and num_sentences > 0:
        gunning_fog_ = 0.4 * (tokens_per_sentence + (percentual_palavras_complexas_ / 100))

    vocab_palavras = set(t.text.lower() for t in tokens if t.is_alpha)
    vocab_size = len(vocab_palavras)
    brunet_ = 0
    if total_palavras > 0:
        brunet_ = total_palavras ** (vocab_size ** (-0.165))


    # psicolinguística
    informalidade = informalidade_doc(doc, total_palavras)
    especificidade = proporcao_entidades_especificas(doc, total_palavras)
    numero_de_exclamacoes = contar_exclamacoes(doc)

    # ------ dicionário ------
    return {
        "type_token_ratio": n_type_token_ratio,
        "number_of_sentences": num_sentences,
        "size_of_sentences_in_words": size_of_sentences_in_words,
        "verb_to_token_ratio": n_verb_to_token_ratio,
        "noun_to_token_ratio": n_noun_to_token_ratio,
        "adjective_to_token_ratio": n_adjective_to_token_ratio,
        "adverb_to_token_ratio": n_adverb_to_token_ratio,
        "pronoun_to_token_ratio": n_pronoun_to_token_ratio,
        "stopword_to_token_ratio": n_stopword_to_token_ratio,
        "pausality": pausality,
        "num_palavras": total_palavras,
        "num_tokens": len(tokens_with_punct),
        "num_sentencas": num_sentences,
        "media_tokens_sentenca": tokens_per_sentence,
        "total_silabas": total_silabas,
        "media_silabas_palavra": media_silabas_palavra,
        "num_palavras": total_palavras,
        "num_palavras_complexas": num_palavras_complexas_,
        "percentual_palavras_complexas": percentual_palavras_complexas_,
        "gunning_fog": gunning_fog_,
        "brunet": brunet_,
        "informalidade": informalidade,
        "especificidade": especificidade,
        "numero_de_exclamacoes": numero_de_exclamacoes
    }