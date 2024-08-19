import time
from pymilvus import DataType, CollectionSchema, FieldSchema, Collection, Partition, connections
import os
from pymilvus import utility
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
import logging
from langchain.document_loaders import SitemapLoader, WebBaseLoader
import nest_asyncio
from config import openai_api_credentials, azure_openai_config, connection_string, collection_name, index_params,fields, index_field, url, metadata, text_field, vector_field, primary_field

for key, value in openai_api_credentials.items():
    os.environ[key] = value

def create_collection(collection_name, fields, index_params, connection_string, index_field):
    schema = CollectionSchema(fields=fields, description='Page embeddings collection')
    client = connections.connect(user=connection_string["milvus_user"], password=connection_string["milvus_password"], host=connection_string["milvus_host"], port=connection_string["milvus_port"])
    logging.info("Connection Established...")
    collection = Collection(name=collection_name, schema=schema)
    logging.info(f"Existing collections :{utility.list_collections()}")
    collection.create_index(field_name=index_field, index_params=index_params)
    collection.load()

def create_docs(url, metadata):
    loader = WebBaseLoader(url)
    docs = loader.load()
    loader.requests_per_second = 2
    loader.requests_kwargs = {"verify": False}
    print(f"Loading docs : {docs}")
    for document in docs:
        document.metadata.clear()
        for key, value in metadata.items():
            print( key, value)
            document.metadata[key] = value
    return docs

def load_data(docs, connection_string, collection_name, text_field,vector_field, primary_field ):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    logging.info("Generating embeddings...")
    embeddings = OpenAIEmbeddings(deployment=azure_openai_config["AZURE_OPEN_AI_DEPLOYMENT_NAME_ADA"], chunk_size=16)
    logging.info("Loading document vectors to database..")
    milvus_store = Milvus(embedding_function=embeddings,
                          connection_args={"host": connection_string["milvus_host"], "port": connection_string["milvus_port"] , "user": connection_string["milvus_user"] ,
                                           "password": connection_string["milvus_password"]},
                          collection_name=collection_name, text_field=text_field, vector_field=vector_field,
                          primary_field=primary_field)
    vector_ids = milvus_store.add_documents(documents=texts)
    return vector_ids


def main():
    # Variables Functions
    create_collection(collection_name, fields, index_params, connection_string, index_field)
    docs = create_docs(url, metadata)
    vector_ids = load_data(docs, connection_string, collection_name, text_field,vector_field, primary_field)
    print(vector_ids)


main()










