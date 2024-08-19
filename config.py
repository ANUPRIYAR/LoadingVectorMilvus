import time
from pymilvus import DataType, CollectionSchema, FieldSchema, Collection, Partition, connections

# Common
openai_api_credentials = {}
openai_api_credentials["OPENAI_API_KEY"] = ".................."
openai_api_credentials["OPENAI_API_BASE"] = "https://.................openai.azure.com/"
openai_api_credentials["OPENAI_API_TYPE"] = "azure"
openai_api_credentials["OPENAI_API_VERSION"]  = "2023-03-15-preview"

azure_openai_config = {}
azure_openai_config["AZURE_OPEN_AI_DEPLOYMENT_NAME_ADA"] = "text-embedding-ada-002"
azure_openai_config["AZURE_OPEN_AI_LLM_MODEL_ADA"] = "text-embedding-ada-002"
AZURE_OPEN_AI_GET_EMBEDDING = "/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-03-15-preview"


connection_string = {}
connection_string["milvus_host"] = 'localhost'
connection_string["milvus_port"] = 19530
connection_string["milvus_user"] = "applicationuser"
connection_string["milvus_password"] = "User@123"


# Unstructured
# collection_name = 'Unstructured_data_collection'
# _unstructured_fields = [
#     FieldSchema(name='pid', dtype=DataType.INT64, description='PIds', is_primary=True, auto_id=True),
#     FieldSchema(name='ID', dtype=DataType.INT64, description='Ids'),
#     FieldSchema(name='FolderName_URL', dtype=DataType.VARCHAR, description='Folder Name', max_length=200),
#     FieldSchema(name='FileName_Title', dtype=DataType.VARCHAR, description='Filename', max_length=200),
#     FieldSchema(name='Extracted_Text', dtype=DataType.VARCHAR, description='Page Content', max_length=35000),
#     FieldSchema(name='Embeddings', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=1536),
#     FieldSchema(name='TYPE', dtype=DataType.VARCHAR, description='Type', max_length=200),
# ]
# index_params = {
#     'index_type': 'IVF_FLAT',
#     'metric_type': 'L2',
#     'params': {'nlist': 1024}
# }
# fields = _unstructured_fields
# index_field = "Embeddings"
# url = 'https://python.langchain.com/docs/modules/agents/'
# id = 1
# metadata = {'pid': int(str(int(time.time()))), 'ID': int(id + 1) ,'FolderName_URL': 'FolderName_URL', 'FileName_Title':'FileName_Title', 'TYPE':'TYPE' }
# text_field = 'Extracted_Text'
# vector_field = 'Embeddings'
# primary_field = 'pid'


# Structured
_structured_fields = [FieldSchema(name='pid', dtype=DataType.INT64, description='PIds', is_primary=True, auto_id=True),
        FieldSchema(name='id', dtype=DataType.VARCHAR, description='Ids', max_length=120),
        FieldSchema(name='assetversionid', dtype=DataType.VARCHAR, description='AssetVersionId', max_length=120),
        FieldSchema(name='name', dtype=DataType.VARCHAR, description='Name', max_length=200),
        FieldSchema(name='type', dtype=DataType.VARCHAR, description='Type', max_length=200),
        FieldSchema(name='opp_text', dtype=DataType.VARCHAR, description='Content', max_length=65535),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=1536),
        FieldSchema(name='bc_ids', dtype=DataType.VARCHAR, description='BCIDs', max_length=5000),
        FieldSchema(name='market', dtype=DataType.VARCHAR, description='Market', max_length=35000),
        FieldSchema(name='industry', dtype=DataType.VARCHAR, description='Industry', max_length=35000),
        FieldSchema(name='function', dtype=DataType.VARCHAR, description='Function', max_length=35000),
        FieldSchema(name='service', dtype=DataType.VARCHAR, description='service', max_length=35000),
        FieldSchema(name='market_unit', dtype=DataType.VARCHAR, description='market_unit', max_length=5000),
        FieldSchema(name='security_tag', dtype=DataType.VARCHAR, description='security_tag', max_length=5000)
    ]


collection_name = 'Structured_data_collection'

index_params = {
    'index_type': 'IVF_FLAT',
    'metric_type': 'L2',
    'params': {'nlist': 1024}
}
fields = _structured_fields
index_field = 'embedding'
# url = 'https://python.langchain.com/docs/modules/agents/'
url = 'https://www.oracle.com/in/database/what-is-database/'
id = 1
# metadata = {'pid': int(str(int(time.time()))), 'id': 'id' ,'assetversionid': 'assetversionid',
#             'name': 'name',
#             'FileName_Title':'FileName_Title', 'type':'type', 'bc_ids': 'bc_ids', 'market': 'market', 'industry': 'industry',
#             'function' : 'function', 'service': 'service', 'market_unit': 'market_unit', 'security_tag': 'security_tag',
#           }
metadata = {'pid': int(str(int(time.time()))), 'id': 'id' ,'assetversionid': 'assetversionid',
            'name': 'Oracle',
            'FileName_Title':'Oracle_database', 'type':'type', 'bc_ids': 'bc_ids', 'market':
                'market',
            'industry': 'industry',
            'function' : 'function', 'service': 'service', 'market_unit': 'market_unit', 'security_tag': 'security_tag',
          }
text_field = 'opp_text'
vector_field = 'embedding'
primary_field = 'pid'