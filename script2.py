import pandas as pd
import kagglehub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from utils.config import params

def initialize_embeddings():
    """Initialize and return the embeddings model"""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def load_data():
    path = kagglehub.dataset_download("nechbamohammed/research-papers-dataset")
    file_name = "dblp-v10.csv"
    df = pd.read_csv(path + file_name)
    return df[(df['year'] >= 2017) & (df['n_citation'] >= 1)]

def create_documents(filtered_df):
    documents = []
    for _, row in filtered_df.iterrows():
        if pd.isna(row['abstract']) or str(row['abstract']).strip() == '':
            continue
            
        document = Document(
            page_content=str(row['abstract']),
            metadata={
                'id': str(row['id']),
                'title': str(row['title']),
                'authors': str(row['authors']),
                'year': int(row['year']),
                'venue': str(row['venue']),
                'n_citation': int(row['n_citation']),
                'references': str(row['references'])
            }
        )
        documents.append(document)
    return documents

def store_embeddings(documents, embeddings):
    return PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=params.collection_name,
        connection=params.psql_conn_string,
        pre_delete_collection=False,
        use_jsonb=True
    )

def main():
    embeddings = initialize_embeddings()
    filtered_df = load_data()
    documents = create_documents(filtered_df)
    vector_store = store_embeddings(documents, embeddings)
    return vector_store

if __name__ == "__main__":
    main()
