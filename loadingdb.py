import pymilvus
import time
from pymilvus import DataType, CollectionSchema, FieldSchema, Collection, Partition, connections
import os
from pymilvus import utility
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader
# from pymilvus import connections
# from app.Utilities.configuration import get_configuration_property, get_secret_property
import logging
import requests
from bs4 import BeautifulSoup
# from langchain.vectorstores import Milvus

# MILVUS-HOST = "localhost"
# MILVUS-PORT = "19530"
# MILVUS-USER = "applicationuser"
# MILVUS-PASSWORD = "User@123"
OPENAI_API_KEY = "2e4be0b1dd7749dcbcc64d9737c215b8"
OPENAI_API_BASE = "https://cog-nvi-ainav-openai-poc.openai.azure.com/"
OPENAI_API_TYPE= "azure"
OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPEN_AI_DEPLOYMENT_NAME_GPT35 = "poc_deployment_gpt35"
AZURE_OPEN_AI_LLM_MODEL_GPT35 = "gpt-35-turbo"
AZURE_OPEN_AI_DEPLOYMENT_NAME_ADA = "text-embedding-ada-002"
AZURE_OPEN_AI_LLM_MODEL_ADA = "text-embedding-ada-002"
AZURE_OPEN_AI_GET_EMBEDDING = "/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-03-15-preview"
# Milvus connection settings

milvus_host = 'localhost'
milvus_port = 19530
milvus_user = "applicationuser"
milvus_password = "User@123"


os.environ["OPENAI_API_TYPE"] = OPENAI_API_TYPE
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION

# Milvus collection settings
# collection_name = 'Structured_data_collection'
collection_name = 'Unstructured_data_collection'
_unstructured_fields = [
    FieldSchema(name='pid', dtype=DataType.INT64, description='PIds', is_primary=True, auto_id=True),
    FieldSchema(name='ID', dtype=DataType.INT64, description='Ids'),
    FieldSchema(name='FolderName_URL', dtype=DataType.VARCHAR, description='Folder Name', max_length=200),
    FieldSchema(name='FileName_Title', dtype=DataType.VARCHAR, description='Filename', max_length=200),
    FieldSchema(name='Extracted_Text', dtype=DataType.VARCHAR, description='Page Content', max_length=35000),
    FieldSchema(name='Embeddings', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=1536),
    FieldSchema(name='TYPE', dtype=DataType.VARCHAR, description='Type', max_length=200),
]
_unstructured_schema = CollectionSchema(fields=_unstructured_fields, description='Page embeddings collection')

# Index parameters
index_params = {
    'index_type': 'IVF_FLAT',
    'metric_type': 'L2',
    'params': {'nlist': 1024}
}

# Milvus connection
client = connections.connect(user=milvus_user, password=milvus_password, host='localhost', port='19530')
print("Connection established", client)
collection = Collection(name=collection_name, schema=_unstructured_schema)
print(utility.list_collections())
collection.create_index(field_name="Embeddings", index_params=index_params)
collection.load()
# Create Milvus collection
# if collection_name in utility.list_collections():
#     client.drop_collection(collection_name)

# collection = Collection(name=collection_name, schema=_unstructured_schema)
# collection.create_index(field_name='Embeddings', index_params=index_params)

#
#
from langchain.document_loaders import SitemapLoader, WebBaseLoader
import nest_asyncio

nest_asyncio.apply()

output_file = open("scraped_content.txt", "w", encoding="utf-8")


loader= WebBaseLoader("https://python.langchain.com/docs/modules/agents/")

docs = loader.load()
loader.requests_per_second = 2
loader.requests_kwargs = {"verify": False}

# print(docs)
#
# file_path = "scraped_content.txt"
#
# # data = {
# #             'FolderName_URL': 'Sample Folder Name',
# #             'FileName_Title': 'Sample Filename',
# #             'Extracted_Text': 'Sample Page Content',
# #             'Embeddings': [0.1, 0.2, 0.3, ...],  # Replace with actual embedding
# #             'TYPE': 'Sample Type'
# #         }
id = 1
pid = str(int(time.time()))
print(pid)
for document in docs:
    document.metadata.clear()  # Clear existing metadata
    document.metadata['pid'] = int(pid)
    document.metadata['ID'] = int(id)
    document.metadata['FolderName_URL'] = "foldername"
    document.metadata['FileName_Title'] = "LangchainAgents"
    # document.metadata['Extracted_Text'] = document.page_content
    document.metadata['TYPE'] = "Sample Type"
    id += 1
    # print(document.metadata)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
#
logging.info("Generating embeddings...")
embeddings = OpenAIEmbeddings(deployment=AZURE_OPEN_AI_DEPLOYMENT_NAME_ADA, chunk_size=16)
# print(f"embeddings:{embeddings}")
#
# # Set up a vector store used to save the vector embeddings. Here we use Milvus as the vector store.
logging.info("Loading document vectors to database..")
# print(dir(Milvus))
# print(dir(embeddings))
milvus_store = Milvus( embedding_function=embeddings,
                      connection_args={"host": milvus_host, "port": milvus_port, "user": milvus_user,
                                       "password": milvus_password},
                      collection_name=collection_name, text_field='Extracted_Text', vector_field='Embeddings', primary_field='pid')
print(milvus_store)
#
vector_ids = milvus_store.add_documents(documents=texts)
print(vector_ids)
# vector_store = Milvus.from_documents(
#     docs,
#     embedding=embeddings,
#     connection_args={"host": milvus_host, "port": milvus_port, "user": milvus_user, "password": milvus_password},
# )
# print(collection.num_entities )
# # os.remove(file_path)
#
