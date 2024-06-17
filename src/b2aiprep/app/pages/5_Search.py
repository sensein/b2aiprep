import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from scipy.spatial import distance as ssd
import numpy as np
import streamlit as st
import typing as t

def mean_pooling(model_output, attention_mask):
    """Mean pool the model output over the tokens factoring in attention."""
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@st.cache_data
def load_model(model_path="sentence-transformers/all-MiniLM-L6-v2"):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    return tokenizer, model


tokenizer, model = load_model()


def embed_sentences(text_list):
    # Compute token embeddings
    # Batch this for memory efficiency
    batch_size = 10
    sentence_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        # Tokenize sentences
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**encoded_input)
        batch_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        sentence_embeddings.append(batch_embeddings)

    sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
    return sentence_embeddings


@st.cache_data
def embed_corpus(c):
    return embed_sentences(c)

@st.cache_data
def download_and_load_data_dictionary() -> pd.DataFrame:
    """Load the data dictionary from the public release GitHub repo.
    
    https://github.com/eipm/bridge2ai-redcap/

    Returns:
        pd.DataFrame: The data dictionary as a DataFrame.
    """
    data_dictionary_url = (
        "https://raw.githubusercontent.com/eipm/bridge2ai-redcap/main/data/bridge2ai_voice_project_data_dictionary.csv"
    )
    return pd.read_csv(data_dictionary_url)

rcdict = download_and_load_data_dictionary()


def extract_descriptions(df: pd.DataFrame) -> t.Tuple[t.List[str], t.List[str]]:
    """Extract the descriptions from the data dictionary."""

    # There are a number of fields in the data dictionary which do not have a label
    # For example, page_1, page2, page3... of the eConsent
    # We will only consider fields with a label, as the ones missing label
    # are not useful for our search.
    idx = df["Field Label"].notnull()
    corpus = df.loc[idx, "Field Label"].values.tolist()
    field_ids = df.loc[idx, "Variable / Field Name"].values.tolist()
    return corpus, field_ids

corpus, field_ids = extract_descriptions(rcdict)

# TODO: This is a very memory hungry operation, spikes with ~40 GB of memory.
corpus_as_vector = embed_corpus(corpus)

st.markdown(
    """
    # Search the data dictionary

    The following text box allows you to semantically search the data dictionary.
    You can use it to find the name for fields collected in the study.
    """
)
search_string = st.text_input("Search string", "age")

search_embedding = embed_sentences(
    [
        search_string,
    ]
)

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
    final_df = pd.DataFrame(
        {"field_ids": field_ids_to_show, "field_desc": sentences_to_show}
    )
    st.table(final_df)
