# Import necessary libraries
import streamlit as st
import pandas as pd
import gensim
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud

# Download NLTK stopwords
nltk.download("stopwords")

# Load the survey data
csv_url = 'https://raw.githubusercontent.com/nforeroba/portfolioNFB/main/smartphone_encuesta_CGPT.csv'
df = pd.read_csv(csv_url)

# Reshape survey data
df_restr = df.set_axis(['resp_1', 'resp_2', 'resp_3', 'resp_4'], axis='columns').melt(
    value_vars=['resp_1', 'resp_2', 'resp_3', 'resp_4'],
    var_name='resp'
)

# Tokenization
tokens = []
for survey_response in df_restr['value']:
    l = gensim.utils.simple_preprocess(str(survey_response))
    tokens.append(l)

# Remove stopwords from survey tokens
stop_words = stopwords.words("english")
tokens_stopwords_removed = [[word for word in i if word == 'under' or word not in stop_words] for i in tokens]

# Bigram formation
bigram = gensim.models.Phrases(tokens_stopwords_removed, min_count=3, threshold=6, delimiter="-")
bigram_mod = gensim.models.phrases.Phraser(bigram)
bigram_list = [bigram_mod[i] for i in tokens_stopwords_removed]

# Add monogram and bigram tokens to the reshaped df
df_bigram = df_restr.copy()
df_bigram['tokens'] = bigram_list

# LDA (Latent Dirichlet Allocation) for TOPIC MODELLING
# Corpus of indexed tokens
tokenized_text = df_bigram['tokens']
dictionary = corpora.Dictionary(tokenized_text)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_text]

num_topics = 4
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, alpha='auto', eta='auto')

# Streamlit app
st.title("Smartphone Features Survey Analysis: What do consumers want?")

# Sidebar with user input

st.sidebar.title("Introduction")
st.sidebar.write("Hola! Hello! Hallo! こんにちは！")
st.sidebar.write("I used NLP (Gensim, NLTK) to analyze raw survey data related to smartphone feature preferences.")
st.sidebar.write("Use the sliders and inputs to interact with the analysis. Best regards, Nicolás.")

num_topics_input = st.sidebar.number_input("Number of Topics", min_value=1, max_value=20, value=num_topics)
num_words_input = st.sidebar.number_input("Number of Words in Word Clouds", min_value=1, max_value=40, value=20)

# Re-run LDA model with new number of topics if changed
if num_topics_input != num_topics:
    num_topics = num_topics_input
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=100, alpha='auto', eta='auto')

# Visualize topics using pyLDAvis
st.subheader("Topic Modeling (LDA) Visualization")
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyldavis_html = pyLDAvis.prepared_data_to_html(vis)
st.components.v1.html(pyldavis_html, height=1000, width=1250)

# User input for selecting a specific topic
selected_topic = st.slider("Select a Topic", min_value=1, max_value=num_topics, value=1)

# Visualize the selected topic's word cloud
st.subheader(f"Word Cloud for Selected Topic ({selected_topic})")
selected_topic -= 1  # Adjust to zero-based index
words_in_topic = lda_model.show_topic(selected_topic, topn=num_words_input)
topic_word_freq = {word: freq for word, freq in words_in_topic}
wordcloud = WordCloud(width=1000, height=1000, background_color='white').generate_from_frequencies(topic_word_freq)
st.image(wordcloud.to_array(), caption=f'Topic {selected_topic + 1}', width=800)
