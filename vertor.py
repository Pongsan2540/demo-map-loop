import requests
import argparse
from elasticsearch import Elasticsearch

def create_vector(url_vector, name_vector):
    try: 
        es = Elasticsearch(url_vector).options(basic_auth=("elastic", "changeme"))

        # Check if the index already exists
        if es.indices.exists(index=name_vector):
            print(f"Index '{name_vector}' already exists.")
            return {"status": "exists", "index": name_vector}

        mapping = {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": 384
                    },
                    "text": {
                        "type": "text"
                    },
                    "timeStamp": {
                        "type": "text"
                    },
                    "bbox": {
                        "type": "keyword"
                    },
                    "id_database": {
                        "type": "text"
                    }
                }
            }
        }

        response = es.options(ignore_status=400).indices.create(index=name_vector, body=mapping)

        print(response)  # Print the dictionary response
        return response  # Return the dictionary response

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


def delete_vertor(url_vertor, name_vertor):
    try:
        es = Elasticsearch(url_vertor, basic_auth=("elastic", "changeme"))
        index_name = name_vertor 

        response = es.indices.delete(index=index_name, ignore=[400, 404])

        if response.get("acknowledged", False): 
            print("Index deleted successfully")
            result = "Index deleted successfully"
        else:
            print(f"Failed to delete index: {response}")
            result = f"Failed to delete index: {response}"

    except Exception as e:
        print(f"Error: {e}")
        result = f"Error: {e}"

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api")
    parser.add_argument('--name_vertor', default='my-vertor', help="my-vertor")
    parser.add_argument('--url_vertor', default='http://localhost:9200', help="http://localhost:9200")
    parser.add_argument('--type_method', default='create', help="create or delete")

    args = parser.parse_args()

    name_vertor = args.name_vertor 
    url_vertor = args.url_vertor 
    type_method = args.type_method 

    delete_vertor(url_vertor, name_vertor)
    create_vector(url_vertor, name_vertor)

    #if type_method == "create" :
    #     create_vector(url_vertor, name_vertor)
    #elif type_method == "delete" :
    #    delete_vertor(url_vertor, name_vertor)
    #else :
    #    print("Choosing the wrong method")

