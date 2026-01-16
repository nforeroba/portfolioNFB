# Dependencias
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import scipy
import gensim
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Modo amplio de Streamlit por defecto
st.set_page_config(
    page_title="Smartphone Survey Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para personalizar la apariencia
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        font-size: 16px;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# NLTK stopwords
@st.cache_resource
def download_stopwords():
    nltk.download("stopwords", quiet=True)
    return stopwords.words("english")

stop_words = download_stopwords()

# Cargar y procesar datos encuesta
@st.cache_data
def load_and_process_data():
    csv_url = 'https://raw.githubusercontent.com/nforeroba/portfolioNFB/main/smartphone_encuesta_CGPT.csv'
    df = pd.read_csv(csv_url)
    df_restr = df.set_axis(['resp_1', 'resp_2', 'resp_3', 'resp_4'], axis='columns').melt(
        value_vars=['resp_1', 'resp_2', 'resp_3', 'resp_4'],
        var_name='resp'
    )
    
    # Tokenization
    tokens = []
    for survey_response in df_restr['value']:
        l = gensim.utils.simple_preprocess(str(survey_response))
        tokens.append(l)
    
    # Remover stopwords
    tokens_stopwords_removed = [[word for word in i if word == 'under' or word not in stop_words] for i in tokens]
    
    # Formación de bigramas
    bigram = gensim.models.Phrases(tokens_stopwords_removed, min_count=3, threshold=6, delimiter="-")
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    bigram_list = [bigram_mod[i] for i in tokens_stopwords_removed]
    df_bigram = df_restr.copy()
    df_bigram['tokens'] = bigram_list
    
    return df_bigram

df_bigram = load_and_process_data()

# Modelo LDA (Latent Dirichlet Allocation)
@st.cache_resource
def create_lda_model(num_topics, tokenized_text):
    dictionary = corpora.Dictionary(tokenized_text)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_text]
    lda_model = LdaModel(
        corpus, 
        num_topics=num_topics, 
        id2word=dictionary, 
        passes=10, 
        alpha='auto', 
        eta='auto', 
        random_state=42
    )
    return lda_model, corpus, dictionary

# Header
st.title("📱 Smartphone Features Survey Analysis")
st.markdown("### What do consumers want from their smartphones?")
st.markdown("---")

# Panel lateral
with st.sidebar:
    st.markdown("**¡Hola! Hello! Hallo! こんにちは！👋**")
    st.info("I used NLP (Gensim, NLTK) to analyze raw survey data related to smartphone feature preferences. I also used LDA (Latent Dirichlet Allocation) for topic modeling to identify key themes and visualize them with pyLDAvis and WordClouds.")
    
    st.markdown("---")
    num_topics_input = st.number_input(
        "Number of Topics", 
        min_value=2, 
        max_value=20, 
        value=4,
        help="Adjust the number of topics for LDA modeling"
    )
    
    num_words_input = st.slider(
        "Words in Word Clouds", 
        min_value=10, 
        max_value=40, 
        value=20,
        help="Number of words to display in word clouds"
    )
    
    st.markdown("---")
    st.markdown("**Created by:** Nicolás Forero Baena")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/nforeroba/portfolioNFB)")
    st.markdown("[![Portfolio](https://img.shields.io/badge/Portfolio-4285F4?style=flat&logo=google-chrome&logoColor=white)](https://nicolasfbportfolio.netlify.app/)")

tokenized_text = df_bigram['tokens']
lda_model, corpus, dictionary = create_lda_model(num_topics_input, tokenized_text)

# Pestañas de visualización
tab1, tab2, tab3 = st.tabs(["📊 Topic Modeling Visualization", "☁️ Word Clouds by Topic", "📈 Topic Summary"])

with tab1:
    st.subheader("Interactive Topic Model (LDA) Visualization")
    st.markdown("Explore the relationships between topics and their most relevant terms.")
    
    # Explicación del parámetro lambda
    with st.expander("ℹ️ Understanding the Lambda Parameter (λ)", expanded=False):
        st.markdown("""
        The **lambda (λ)** slider in the visualization controls how terms are ranked for each topic:
        
        **λ = 1 (Default):**
        - Shows terms ranked by their **probability within the topic**
        - Highlights words that appear frequently in a specific topic
        - Good for understanding the overall theme of a topic
        - May show common words that appear across multiple topics
        
        **λ = 0:**
        - Shows terms ranked by their **exclusivity to the topic**
        - Highlights words that are unique or distinctive to that topic
        - Useful for identifying what makes each topic different from others
        - May show less frequent but more discriminative terms
        
        **Intermediate values (0 < λ < 1):**
        - Provide a balance between frequency and exclusivity
        - λ ≈ 0.6 is often recommended for optimal interpretation
        
        💡 **Tip:** Try adjusting the lambda slider in the visualization to explore both common and distinctive terms for each topic!
        """)
    
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        pyldavis_html = pyLDAvis.prepared_data_to_html(vis)
        components.html(pyldavis_html, height=800, scrolling=True)

with tab2:
    st.subheader("Word Clouds for Each Topic")
    st.markdown("Visual representation of the most important words in each topic.")
    
    # Grilla de nubes de palabras
    num_cols = 2 if num_topics_input >= 4 else 1
    
    for row in range((num_topics_input + num_cols - 1) // num_cols):
        cols = st.columns(num_cols)
        for col_idx, col in enumerate(cols):
            topic_idx = row * num_cols + col_idx
            if topic_idx < num_topics_input:
                with col:
                    words_in_topic = lda_model.show_topic(topic_idx, topn=num_words_input)
                    topic_word_freq = {word: freq for word, freq in words_in_topic}
                    
                    # Crear la nube de palabras
                    wordcloud = WordCloud(
                        width=800, 
                        height=600, 
                        background_color='white',
                        colormap='viridis',
                        relative_scaling=0.5,
                        min_font_size=10
                    ).generate_from_frequencies(topic_word_freq)
                    
                    # Visualizar la nube de palabras
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'Topic {topic_idx + 1}', fontsize=16, fontweight='bold', pad=20)
                    st.pyplot(fig)
                    plt.close()

with tab3:
    st.subheader("Topic Summary")
    st.markdown("Top words and their weights for each topic.")
    
    # Visualización de resumen de temas
    for topic_idx in range(num_topics_input):
        with st.expander(f"📌 Topic {topic_idx + 1}", expanded=(topic_idx == 0)):
            words_in_topic = lda_model.show_topic(topic_idx, topn=15)
            
            # DataFrame de palabras y pesos
            topic_df = pd.DataFrame(words_in_topic, columns=['Word', 'Weight'])
            topic_df['Weight'] = topic_df['Weight'].round(4)
            topic_df.index = range(1, len(topic_df) + 1)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    topic_df,
                    use_container_width=True,
                    height=400
                )
            
            with col2:
                # Gráfico de barras de pesos de palabras
                st.bar_chart(
                    topic_df.set_index('Word')['Weight'],
                    height=400
                )
