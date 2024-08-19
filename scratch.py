from pymilvus import DataType, CollectionSchema, FieldSchema, Collection, Partition, connections

_unstructured_fields = [
    FieldSchema(name='pid', dtype=DataType.INT64, description='PIds', is_primary=True, auto_id=True),
    FieldSchema(name='ID', dtype=DataType.INT64, description='Ids'),
    FieldSchema(name='FolderName_URL', dtype=DataType.VARCHAR, description='Folder Name', max_length=200),
    FieldSchema(name='FileName_Title', dtype=DataType.VARCHAR, description='Filename', max_length=200),
    FieldSchema(name='Extracted_Text', dtype=DataType.VARCHAR, description='Page Content', max_length=35000),
    FieldSchema(name='Embeddings', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=1536),
    FieldSchema(name='TYPE', dtype=DataType.VARCHAR, description='Type', max_length=200),
]

for field in _unstructured_fields:
    print(dir(field))
    field_d = field.to_dict()
    print(field_d.items())
    print(field_d.type)
    if field.dtype == 5:

    # if field.type ==
    # break

