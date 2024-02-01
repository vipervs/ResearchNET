import requests
import json
import os

api_key = 'dataset-4UX7eUmzbbPIlUBVmDG7Hbwy'
BASE_URL = 'http://localhost/v1'

def create_knowledge(api_key, knowledge_name):
    url = f'{BASE_URL}/datasets'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {'name': knowledge_name}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def upload_document(api_key, dataset_id, file_path):
    url = f'{BASE_URL}/datasets/{dataset_id}/document/create_by_file'
    headers = {'Authorization': f'Bearer {api_key}'}
    
    # This matches the automatic mode without detailed rules as per your function's intent
    data = {
        "indexing_technique": "high_quality",
        "process_rule": {"mode": "automatic"}
    }
    
    # Prepare the 'data' part of the request as a JSON string
    data_string = json.dumps(data)
    
    # The 'files' parameter is used for multipart/form-data
    files = {
        'data': ('', data_string, 'application/json'),
        'file': (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf')  # Ensure the MIME type matches your file type
    }
    
    response = requests.post(url, headers=headers, files=files)
    try:
        return response.json()
    except json.JSONDecodeError:
        print("Failed to decode JSON. Here's the response text:")
        print(response.text)
        raise

def check_embedding_status(api_key, dataset_id, batch):
    url = f'{BASE_URL}/datasets/{dataset_id}/documents/{batch}/indexing-status'
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(url, headers=headers)
    return response.json()

def query_vector_db(api_key, dataset_id, query, page=1, limit=20):
    url = f'{BASE_URL}/datasets/{dataset_id}/documents'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    params = {'keyword': query, 'page': page, 'limit': limit}
    response = requests.get(url, headers=headers, params=params)
    return response.json()


knowledge_name = 'Test'
dataset_id = '05998e7f-deb2-436d-91a8-ce821b5e4aca'
script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, '2008.07278v1.pdf')
batch = 'your_batch_id'

# Create a new Knowledge
#knowledge = create_knowledge(api_key, knowledge_name)
#print(knowledge)

# Upload a document
document = upload_document(api_key, dataset_id, file_path)
print(document)

# Check embedding status
#status = check_embedding_status(api_key, dataset_id, batch)
#print(status)

# Query the vector database
#query_result = query_vector_db(api_key, dataset_id, "sample query")
#print(query_result)
