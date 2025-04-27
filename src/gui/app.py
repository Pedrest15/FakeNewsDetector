import streamlit as st
from psycolinguistic_classifier.psyco_stats import psycometric_classification
from ratio_classifier.ratio_classifier import ratio_classifier
from vocabulary_classifier.vocabulary_classifier import vocab_classifier

st.title("Detector de Fake News")

# Caixa de texto
news_text = st.text_area("Digite a notícia que deseja verificar:")

# Selectbox para escolha do modelo
model_options = ["Ratio Classifier", "Psycolinguistic Classifier", "Vocabulary Classifier"]
selected_model = st.selectbox("Selecione o modelo de detecção:", model_options)
if selected_model == "Ratio Classifier":
    st.info("Modelo baseado em analisar a frequência de adjetivos, advérbios e valores numéricos em uma notícia.")

# Botão de avaliação
if st.button("Avaliar Fake News"):
    if not news_text.strip():
        st.warning("Por favor, insira uma notícia para avaliar.")
    else:
        if selected_model == "Ratio Classifier":
            result = ratio_classifier.news_classifier(text=news_text)
        elif selected_model == "Psycolinguistic Classifier":
            result = psycometric_classification(text=news_text)
        elif selected_model == "Vocabulary Classifier":
            result = vocab_classifier.news_classifier(sentence=news_text)

        # Exibe o resultado
        with st.expander("Resultado da Avaliação", expanded=True):
            st.markdown(f"**Modelo selecionado:** {selected_model}")
            st.write("**Resultado da notícia:**")
            
            if result=="true":
                st.success("Parece ser uma notícia verdadeira!")
            else:
                st.error("""Cuidado, pode se tratar de uma notícia falsa (Fake News).
                         Cheque em fontes confiáveis.""")

