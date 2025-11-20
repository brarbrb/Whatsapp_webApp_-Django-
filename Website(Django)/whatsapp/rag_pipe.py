# --------------------------- Import libraries --------------------------------------------------
import os
import cohere
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
# to get (APIs keys) from environment variables
from dotenv import load_dotenv

load_dotenv(override=True)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def upsert_vectors(
    index, dataset: DataFrame, embeddings: np.ndarray, batch_size: int = 128
):
    """
    Upsert vectors to a pinecone index
    Args:
        index: The pinecone index object
        embeddings: The embeddings to upsert (same length/order as dataset)
        dataset: The dataset containing the metadata
        batch_size: The batch size to use for upserting
    Returns:
        The pinecone index (for convenience)
    """
    print("Upserting the embeddings to the Pinecone index...")

    # Get all column names except 'embedding' for metadata
    metadata_fields = [col for col in dataset.columns if col != "embedding"]

    # Generate unique IDs for each row
    ids = [str(i) for i in range(len(dataset))]

    # Build metadata dict for each row
    meta = []
    for _, row in dataset.iterrows():
        entry = {col: row[col] for col in metadata_fields}
        meta.append(entry)

    # Create list of (id, vector, metadata) tuples for upserting
    to_upsert = list(zip(ids, embeddings, meta))

    # Upsert in batches
    for i in tqdm(range(0, len(to_upsert), batch_size)):
        i_end = min(i + batch_size, len(to_upsert))
        index.upsert(vectors=to_upsert[i:i_end])

    print("Upserting complete!")
    return index

def augment_prompt(
    query: str,
    user_style: str,
    context: str,
    model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2"),
    index=None,
    instructions = "",
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
      1) **query** - the new incoming message you must answer.
      2) **similar_past_answers** - real replies the user wrote in the past to similar messages.  
         Use them for tone, vibe, typical phrasing, emojis, and attitude.
      3) **user_style_examples** - random messages the user wrote in other chats.  
         Use them to mimic writing style, vocabulary, length, energy level, and emoji habits.
      4) **recent_context** - the recent messages in this same chat (both sides).  
         Your reply must fit naturally after this context.
      5) **instructions** (optional) - specific instructions to follow when generating the reply.
         Your reply must follow these instructions closely.

      ### Rules:
      - Write the reply **as the user**, in first person.
      - Match the **language**, **tone**, and **emotion** of the conversation.
      - Keep it natural for WhatsApp: short to medium length, can include emojis.
      - If the query contains multiple questions - answer all.
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

      ### instrucrions:
      {instructions}
      """

    return improved_prompt, answers

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

def load_and_embedd_dataset(
    dataset: DataFrame,
    model: SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL),
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

class WhatsAppRAG:
    """
    Generic class that initializes the whole RAG stack once,
    and exposes a single `generate_reply(query, ...)` method.
    """

    def __init__(
        self,
        kb_path: str,
        receiver_user_id: str = "u_barbara",
        context_conv_id: str = "chat:u_barbara_u_maayan",
        index_name: str = "chats-index",
        embedding_model_name: str = EMBEDDING_MODEL,
        cohere_model_name: str = "command-a-03-2025",
        k_style_messages: int = 10,
        k_context_messages: int = 10,
    ):
        """
        Args:
            kb_path: path to the CSV with chats (your KB_data.csv)
            receiver_user_id: which user we generate replies *for*
            context_conv_id: which conversation to use for recent context
            index_name: Pinecone index name
            embedding_model_name: sentence-transformer model name
            cohere_model_name: Cohere chat model name
            k_style_messages: how many messages to use for style
            k_context_messages: how many turns to use for recent context
        """
        self.kb_path = kb_path
        self.receiver_user_id = receiver_user_id
        self.context_conv_id = context_conv_id
        self.index_name = index_name
        self.cohere_model_name = cohere_model_name
        self.k_style_messages = k_style_messages
        self.k_context_messages = k_context_messages

        # -------- Embedding model --------
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(embedding_model_name)

        # -------- Load & embed KB --------
        print(f"Loading KB from: {kb_path}")
        kb_df = pd.read_csv(kb_path)
        self.kb_df_all, self.embeddings_all = load_and_embedd_dataset(
            kb_df, model=self.embed_model
        )

        # Keep only messages *to* this user for the index
        self.kb_df_for_user = (
            self.kb_df_all[self.kb_df_all["receiver_user_id"] == receiver_user_id]
            .sort_values(by="conv_turn")
            .reset_index(drop=True)
        )

        # Align embeddings with kb_df_for_user (same order & length)
        # We already stored embeddings in the 'embedding' column as numpy arrays.
        self.embeddings_for_user = np.stack(self.kb_df_for_user["embedding"].to_numpy())

        # -------- Pinecone index & upsert --------
        dim = self.embeddings_for_user.shape[1]
        self.pc = create_pinecone_index(self.index_name, dimension=dim)

        print("Connecting to Pinecone index...")
        self.index = self.pc.Index(self.index_name)

        # Upsert only the entries for this user
        upsert_vectors(self.index, self.kb_df_for_user, self.embeddings_for_user)

        # -------- User style --------
        _, self.user_style_examples = build_user_style(
            self.kb_df_all,
            user_id=self.receiver_user_id,
            k=self.k_style_messages,
            text_col="text",
            random_sample=True,
            seed=42,
        )

        # -------- Cohere client --------
        print("Initializing Cohere client...")
        self.cohere_client = cohere.Client(api_key=COHERE_API_KEY)

        print("RAG initialization complete.")

    # ------------------------------------------------------------------
    def _build_context(self, conv_id: str | None = None, k: int | None = None) -> str:
        """
        Internal helper: build recent conversation context.
        """
        conv_id = conv_id or self.context_conv_id
        k = k or self.k_context_messages
        return build_context(
            self.kb_df_all,
            conv_id=conv_id,
            k=k,
            text_col="text",
        )

    def _build_augmented_prompt(self, query: str, context: str, instructions: str) -> str:
        """
        Internal helper: use your augment_prompt to build the full LLM prompt.
        """
        improved_prompt, _ = augment_prompt(
            query=query,
            user_style=self.user_style_examples,
            context=context,
            model=self.embed_model,
            index=self.index,
            instructions=instructions,
        )
        return improved_prompt

    # ------------------------------------------------------------------
    def generate_reply(
        self,
        query: str,
        conv_id: str | None = None,
        k_context: int | None = None,
        instructions: str = "",
    ) -> str:
        """
        Public method: take a new incoming message and return a WhatsApp-style reply.
        """
        # Build context from the KB
        context = self._build_context(conv_id=conv_id, k=k_context)
        # Build augmented prompt
        augmented_prompt = self._build_augmented_prompt(query=query, context=context, instructions=instructions)
        # Call Cohere
        response = self.cohere_client.chat(
            model=self.cohere_model_name,
            message=augmented_prompt,
        )
        # Return just the text (the WhatsApp-style answer)
        return response.text


def main():
    kb_path = "./RAG_data/KB_data.csv"

    rag = WhatsAppRAG(
        kb_path=kb_path,
        receiver_user_id="u_barbara",
        context_conv_id="chat:u_barbara_u_maayan",
    )

    query = "i need help with my students, did you taught them already the embeddings ppt?"
    reply = rag.generate_reply(query)
    print("Generated reply:")
    print(reply)


if __name__ == "__main__":
    main()