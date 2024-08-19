import pymilvus
from pymilvus import DataType, CollectionSchema, FieldSchema, Collection, Partition
import requests
from bs4 import BeautifulSoup

# Milvus connection settings
milvus_host = 'localhost'
milvus_port = 19530

# Milvus collection settings
collection_name = 'Structured_data_collection'
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
client = pymilvus.Connection(host=milvus_host, port=milvus_port)

# Create Milvus collection
if collection_name in client.list_collections():
    client.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=_unstructured_schema)
collection.create_index(field_name='Embeddings', index_params=index_params)

# Web scraping and data insertion
sitemap_url = 'https://langchain.readthedocs.io/sitemap.xml'

def scrape_and_insert_data(sitemap_url, collection):
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.text, 'xml')

    for url in soup.find_all('loc'):
        webpage_url = url.get_text()

        # Replace this with your code to scrape webpage data and create embeddings
        # For example, you can use libraries like requests, BeautifulSoup, and a model to generate embeddings

        # Sample code for inserting data (replace with your actual data)
        data = {
            'FolderName_URL': 'Sample Folder Name',
            'FileName_Title': 'Sample Filename',
            'Extracted_Text': 'Sample Page Content',
            'Embeddings': [0.1, 0.2, 0.3, ...],  # Replace with actual embedding
            'TYPE': 'Sample Type'
        }

        collection.insert([data])

scrape_and_insert_data(sitemap_url, collection)

# Disconnect from Milvus
client.disconnect()