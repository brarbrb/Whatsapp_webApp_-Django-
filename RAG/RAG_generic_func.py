# --------------------------- Import libraries --------------------------------------------------
import os

from typing import List, Dict

import re
import pandas as pd
from pandas import DataFrame
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings("ignore")

import cohere
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from tqdm import tqdm

# ----------------------------------------------------------------------------------------------


# --------------------------- Load environment variables (APIs keys) ---------------------------
from dotenv import load_dotenv

load_dotenv(override=True)

# Replace with your own PineCone API KEY
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")

# Replace with your own Cohere API KEY
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
# ------------------------------------------------------------------------------------------------


# --------------------------- Prepare Data functions --------------------------------------------
## **Preprocessing & Embedding the data**
def load_and_embedd_dataset(
    dataset: DataFrame,
    model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2"),
) -> DataFrame:
    """
    Return a dataset with  column of embedd text field using a sentence-transformer model
    Args:
        dataset_name: The name of the dataset to load
        model: The model to use for embedding
    Returns:
        tuple: A Dataset containing the new column of the the embeddings
    """

    print("Loading and embedding the dataset")

    # build the text we embed
    dataset["doc_text"] = (
        dataset["sender_user_id"]
        + ": "
        + dataset["text"].fillna("")
        + "\n"
        + dataset["receiver_user_id"]
        + ": "
        + dataset["answer"].fillna("")
    )

    # compute embeddings as numpy array [n_rows, dim]
    embeddings = model.encode(
        dataset["doc_text"].tolist(),
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    dataset["embedding"] = list(embeddings)

    print("Done!")
    return dataset, embeddings


## **Build the context from the conversation**
def build_context(
    df: pd.DataFrame,
    conv_id: str,
    k: int = 10,
    text_col: str = "text",
) -> str:
    """
    Return a string with the last k turns from a given conversation.

    Each line looks like:
        <sender_user_id>: <message>

    Args:
        df: DataFrame with at least ['conv_id', 'sender_user_id', text_col, 'sent_at'].
        conv_id: The conversation id to extract from.
        k: Number of turns to include (from the end of the conversation).
        text_col: Column name that holds the actual message text.

    Returns:
        A single multi line string that can be dropped straight into the prompt.
    """
    conv_df = (
        df[df["conv_id"] == conv_id]
        .sort_values("sent_at")  # or 'conv_turn' if you prefer
        .tail(k)
    )

    lines = [
        f"{row['sender_user_id']}: {row[text_col]}" for _, row in conv_df.iterrows()
    ]

    context = "\n".join(lines)
    return context


## **Build the user style from their messages**
def build_user_style(
    df: pd.DataFrame,
    user_id: str,
    k: int = 10,
    text_col: str = "text",
    random_sample: bool = True,
    seed: int | None = 42,
) -> str:
    """
    Return a string that represents the typical style of a given user,
    built from k of their messages.

    Each line looks like:
        <message>

    Args:
        df: DataFrame with at least ['sender_user_id', text_col].
        user_id: The user whose style we want to capture.
        k: Number of messages to use.
        text_col: Column with the text of the message.
        random_sample: If True sample k messages randomly, else take the last k.
        seed: Random seed for reproducibility when random_sample is True.

    Returns:
        A single multi line string with example messages in the user's style.
    """
    user_df = df[df["sender_user_id"] == user_id].copy()

    if len(user_df) == 0:
        return ""

    user_df = user_df.sort_values("sent_at")

    if random_sample and len(user_df) > k:
        rng = np.random.default_rng(seed)
        idx = rng.choice(user_df.index.to_list(), size=k, replace=False)
        user_df = user_df.loc[idx].sort_values("sent_at")
    else:
        user_df = user_df.tail(k)

    lines = [str(msg) for msg in user_df[text_col].tolist()]
    user_style = "\n".join(lines)
    return lines, user_style


# ------------------------------------------------------------------------------------------------


# --------------------------- Pinecone Vector DB functions ---------------------------------------
# **Create Pinecone index if it does not exist**
def create_pinecone_index(
    index_name: str,
    dimension: int,
    metric: str = "cosine",
):
    """
    Create a pinecone index if it does not exist
    Args:
        index_name: The name of the index
        dimension: The dimension of the index
        metric: The metric to use for the index
    Returns:
        Pinecone: A pinecone object which can later be used for upserting vectors and connecting to VectorDBs
    """

    print("Creating a Pinecone index...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    print("Done!")
    return pc


## **Upsert vectors to Pinecone index**
def upsert_vectors(
    index: Pinecone, dataset: DataFrame, embeddings: np.ndarray, batch_size: int = 128
):
    """
    Upsert vectors to a pinecone index
    Args:
        index: The pinecone index object
        embeddings: The embeddings to upsert
        dataset: The dataset containing the metadata
        batch_size: The batch size to use for upserting
    Returns:
        An updated pinecone index
    """
    print("Upserting the embeddings to the Pinecone index...")

    # Get all column names except 'embedding' for metadata
    metadata_fields = [col for col in dataset.columns if col != "embedding"]

    # Generate unique IDs for each row
    ids = [str(i) for i in range(shape[0])]

    # Build metadata dict for each row
    meta = []
    for _, row in dataset.iterrows():
        entry = {col: row[col] for col in metadata_fields}
        meta.append(entry)  # Extract full metadata

    # Create list of (id, vector, metadata) tuples for upserting
    to_upsert = list(zip(ids, embeddings, meta))

    # Upsert in batches
    for i in tqdm(range(0, len(to_upsert), batch_size)):
        i_end = min(i + batch_size, len(to_upsert))
        index.upsert(vectors=to_upsert[i:i_end])

    print("Upserting complete!")
    return index


# ------------------------------------------------------------------------------------------------


## **Augment the prompt with retrieved context**
def augment_prompt(
    query: str,
    user_style: str,
    context: str,
    model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2"),
    index=None,
) -> str:

    results = [float(val) for val in list(model.encode(query))]

    # get top 10 results from knowledge base
    query_results = index.query(
        vector=results, top_k=5, include_values=True, include_metadata=True
    )["matches"]
    text_matches = [match["metadata"]["answer"] for match in query_results]

    # get the text from the results
    answers = "\n\n".join(text_matches)

    # feed into an augmented prompt
    improved_prompt = f"""
    
      You write WhatsApp replies *exactly as the user would*.

      Your job: given a new incoming message, write the reply the user is most likely to send.

      You are given:
      1) **query** – the new incoming message you must answer.
      2) **similar_past_answers** – real replies the user wrote in the past to similar messages.  
         Use them for tone, vibe, typical phrasing, emojis, and attitude.
      3) **user_style_examples** – random messages the user wrote in other chats.  
         Use them to mimic writing style, vocabulary, length, energy level, and emoji habits.
      4) **recent_context** – the recent messages in this same chat (both sides).  
         Your reply must fit naturally after this context.

      ### Rules:
      - Write the reply **as the user**, in first person.
      - Match the **language**, **tone**, and **emotion** of the conversation.
      - Keep it natural for WhatsApp: short to medium length, can include emojis.
      - If the query contains multiple questions – answer all.
      - If necessary info is missing – ask a short clarifying question.
      - **Never** mention examples, past messages, embeddings, or that you're an AI.
      - **Only output the final WhatsApp-style reply. No explanations.**

      ---

      ### query:
      {query}

      ### similar_past_answers for similar queries:
      {answers}

      ### user_style_examples:
      {user_style}

      ### recent_context:
      {context}
      """

    return improved_prompt, answers


def rag_pipleine(kb_knoloedge_path, query):

    whatsapp_chats = pd.read_csv(kb_knoloedge_path)

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    model_emb = SentenceTransformer(EMBEDDING_MODEL)
    kb_df_all, embeddings = load_and_embedd_dataset(whatsapp_chats, model_emb)
    kb_df_to_barbara = kb_df_all[
        kb_df_all["receiver_user_id"] == "u_barbara"
    ].sort_values(by="conv_turn")
    INDEX_NAME = "chats-index"

    # Create the vector database
    # We are passing the index_name and the size of our embeddings
    pc = create_pinecone_index(INDEX_NAME, shape[1])

    # Upsert the embeddings to the Pinecone index
    index = pc.Index(INDEX_NAME)
    index_upserted = upsert_vectors(index, kb_df_to_barbara, embeddings)

    # build context and user style strings
    context = build_context(kb_df_all, conv_id="chat:u_barbara_u_maayan", k=10)
    barbara_messages, user_style = build_user_style(
        kb_df_all, user_id="u_barbara", k=10
    )

    query = (
        "i need help with my students, did you taught them already the embeddings ppt?"
    )

    augmented_prompt, source_knowledge = augment_prompt(
        query, user_style, context, model=model_emb, index=index
    )
    co = cohere.Client(api_key=COHERE_API_KEY)
    response = co.chat(
        model="command-a-03-2025",
        message=augmented_prompt,
    )
    response.text

    return response.text


def main():
    ## **Loading the Knowledge Base data**
    kb_path = "./RAG_data/KB_data.csv"
    query = (
        "i need help with my students, did you taught them already the embeddings ppt?"
    )
    generated_message = rag_pipleine(kb_path, query)


if __name__ == "__main__":
    main()
