import requests
import json

class BPMNSearch:
    def __init__(self, default_url="http://localhost:9200"):
        self.default_url = default_url

    def _search(self, url, index_or_type, query, size, is_mu_search):
        headers = {'Content-Type': 'application/json'}
        if False: #'text' in query:
            inner_query = {
                "bool": {
                    "should": [
                        {"match": {"name": query['text']}},
                        {"match": {"description": query['text']}}
                    ]
                }
            }
        else:
            inner_query = {"match_all": {}}
    
        data = {
            "size": size,
            "query": {
                "script_score": {
                    "query": inner_query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query['embedding'],
                        }
                    }
                }
            }
        }
        if is_mu_search:
            response = requests.post(f"{url}/{index_or_type}/search", headers=headers, data=json.dumps(data))
        else:
            response = requests.get(f"{url}/{index_or_type}/_search", headers=headers, data=json.dumps(data))

        response.raise_for_status()  # Raises a HTTPError if the response status is 4xx, 5xx
        return response.json()

    def knn_search(self, index, query, size=10, url=None):
        if url is None:
            url = self.default_url

        json_response = self._search(url, index, query, size, is_mu_search=False)
        hits = json_response['hits']['hits']
        hits = sorted(hits, key=lambda x: x['_score'], reverse=True)
        return [(hit['_source'], hit['_score']) for hit in hits[:size]]

    def mu_knn_search(self, resource_type, query, size=10, url=None):
        if url is None:
            url = self.default_url
        return self._search(url, resource_type, query, size, is_mu_search=True)