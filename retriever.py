"""_summary_

This file defines a Retriever class for handling different retrieval modes, including 'bm25', 'embed', and 'hybrid'.
It utilizes various retrievers from the Langchain library to process documents and return relevant results.

Functionality:
- The Retriever class initializes with a specified mode, embedding model name, and vector database.
- It provides methods to build a schema corpus from a DataFrame and retrieve documents based on the selected mode.

Usage:
1. Import this file and create an instance of the Retriever class:
   ```python
   retriever_instance = Retriever(mode="your_mode", embed_model_name="your_model_name", db=your_db)
   ```
2. Call the retrieve_schema method with a query and DataFrame:
   ```python
   results = retriever_instance.retrieve_schema("your_query", your_dataframe)
   ```
"""

from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings


class Retriever:
    # Initializes the Retriever class with mode, embedding model name, database, and top_k value
    def __init__(self, mode, embed_model_name, db, top_k = 5):
        """
        Initialize the Retriever class.

        Args:
            mode: Retrieval mode, optional values are 'bm25', 'embed', and 'hybrid'.
            embed_model_name: The name of the embedding model.
            db: Vector database instance, optional values are 'FAISS' or 'Chroma'.
            top_k: The number of results to return, default is 5.
        """
        self.mode = mode
        self.embed_model_name = embed_model_name
        self.db = db
        self.top_k = top_k
        # Sets the embedder based on the retrieval mode
        if self.mode == 'bm25':
            self.embedder = None
        else:
            self.embedder = HuggingFaceEmbeddings(model_name=self.embed_model_name)

    # Builds a schema corpus from the provided DataFrame
    def build_schema_corpus(self, df):
        docs = []
        for col_name in df.columns:
            result_text = f'{{"column_name": "{col_name}", "dtype": "{df[col_name].dtype}"}}'
            docs.append(Document(page_content=col_name, metadata={'result_text': result_text}))
        return docs
    
    # Creates an embed retriever from the documents
    def get_embed_retriever(self, docs):
        db = self.db.from_documents(docs, self.embedder)
        return db.as_retriever(search_kwargs={'k': self.top_k})
    
    # Creates a BM25 retriever from the documents
    def get_bm25_retriever(self, docs):
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = self.top_k
        return bm25_retriever
    
    # Creates a hybrid retriever combining embed and BM25 retrievers
    def get_hybrid_retriever(self, docs):
        embed_retriever = self.get_embed_retriever(docs)
        bm25_retriever = self.get_bm25_retriever(docs)
        return EnsembleRetriever(retrievers=[embed_retriever, bm25_retriever], weights=[0.5, 0.5])
    
    # Selects Retriever based on the specified mode
    def get_retriever(self, df):
        docs = self.build_schema_corpus(df)
        if self.mode == 'embed':
            return self.get_embed_retriever(docs)
        if self.mode == 'bm25':
            return self.get_bm25_retriever(docs)
        if self.mode == 'hybrid':
            return self.get_hybrid_retriever(docs)

    # Retrieves schema based on the query and DataFrame, with optional evaluation
    def retrieve_schema(self, query, df, evaluate=False):
        results = self.get_retriever(df).invoke(query)
        if evaluate:
            return [doc.page_content for doc in results]
        observations = [doc.metadata['result_text'] for doc in results if 'result_text' in doc.metadata]
        return observations
        # The evaluate parameter determines the output format:
        # If True, only the retrieved column names are returned for calculating retrieval metrics.
        # If False, all retrieved information is returned for system processing.



