import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from scipy.spatial import distance as ssd
import numpy as np
import streamlit as st

data_path = "bridge2ai-Voice/bridge2ai-voice-corpus-1//b2ai-voice-corpus-1-dictionary.csv"
rcdict = pd.read_csv(data_path)


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@st.cache_data
def load_model(model_path='models/sentence-transformers/all-MiniLM-L6-v2'):

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    return tokenizer, model

tokenizer, model = load_model()

def embed_sentences(text_list):

    # Tokenize sentences
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


@st.cache_data
def embed_corpus(c):
    return embed_sentences(c)



corpus = rcdict['Field Label'].values.tolist()
field_ids = rcdict['Variable / Field Name'].values.tolist()

corpus_as_vector = embed_corpus(corpus)

search_string = st.text_input('Search string', 'age')

search_embedding = embed_sentences([search_string,])

# Compute cosine similarity scores for the search string to all other sentences
sims = []
for embedding in corpus_as_vector:
    sims.append(1 - ssd.cosine(search_embedding[0], embedding))

# Sort sentences by similarity score in descending order (the most similar ones are first)
sorted_index = np.argsort(sims)[::-1]

sentences_sorted = np.array(corpus)[sorted_index]
field_ids_sorted = np.array(field_ids)[sorted_index]
sims = np.array(sims)[sorted_index]

col1, col2 = st.columns(2)

with col1:

    cutoff = st.number_input("Cutoff", 0.0, 1.0, 0.3)

    plt.plot(sims)
    plt.title("Cosine similarity")
    st.pyplot(plt)


with col2:

    sentences_to_show = sentences_sorted[sims > cutoff].tolist()
    field_ids_to_show = field_ids_sorted[sims > cutoff].tolist()

    final_df = pd.DataFrame({'field_ids': field_ids_to_show, 'field_desc': sentences_to_show})

    st.table(final_df)