import streamlit as st
from src.linguistic_features_classifier.extract_metrics import main
from src.passive_voice.análise_fakebr2 import detectar_voz_passiva, \
    detectar_evidencialidade, detectar_conectores_argumentativos
from src.psycolinguistic_classifier.psyco_stats import psycometric_classification
from src.ratio_classifier.ratio_classifier import ratio_classifier
from src.vocabulary_classifier.vocabulary_classifier import vocab_classifier

st.title("Detector de Fake News")

# Caixa de texto
news_text = st.text_area("Digite a notícia que deseja verificar:")

# Selectbox para escolha do modelo
model_options = ["Atributos Linguísticos", "Vocabulário", "Proporções Linguísticas", 
                 "Psycolinguistic Classifier", "Voz Passiva", "Evidencialidade",
                 "Conectores Argumentativos"]
selected_model = st.selectbox("Selecione o modelo de detecção:", model_options)

# Botão de avaliação
if st.button("Avaliar Fake News"):
    if not news_text.strip():
        st.warning("Por favor, insira uma notícia para avaliar.")
    else:
        if selected_model == "Atributos Linguísticos":
            result = main(text=news_text)
        elif selected_model == "Vocabulário":
            result = vocab_classifier.news_classifier(sentence=news_text)
        elif selected_model == "Proporções Linguísticas":
            result = ratio_classifier.news_classifier(text=news_text)
        elif selected_model == "Psycolinguistic Classifier":
            result = psycometric_classification(text=news_text)
        elif selected_model == "Voz Passiva":
            result = detectar_voz_passiva(texto=news_text)
        elif selected_model == "Evidencialidade":
            result = detectar_evidencialidade(texto=news_text)
        elif selected_model == "Conectores Argumentativos":
            result = detectar_conectores_argumentativos(texto=news_text)

        # Exibe o resultado
        with st.expander("Resultado da Avaliação", expanded=True):
            st.markdown(f"**Modelo selecionado:** {selected_model}")
            st.write("**Resultado da notícia:**")
            
            if result=="true":
                st.success("Parece ser uma notícia verdadeira!")
            else:
                st.error("""Cuidado, pode se tratar de uma notícia falsa (Fake News).
                         Cheque em fontes confiáveis.""")

