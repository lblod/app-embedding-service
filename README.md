# app-embedding-service
 
Backend for creating embeddings for BPMN files and text, leveraging advanced machine learning techniques to enhance search capabilities. Built on the mu.semte.ch microservices stack, this service combines lexical and semantic search to provide more accurate and meaningful search results.

This repository is a [mu-project](https://github.com/mu-semtech/mu-project), it includes the minimal set of services required to run the openproceshuis.

## What does it do?

This project demonstrates how the search functionality of mu-search can be improved using Machine Learning (ML). In this project, the lexical search of mu-search is combined with semantic search to enable natural language queries (e.g., 'give me a BPMN about handling disputes'). The main focus of this project is on BPMN files, but it generalizes to any kind of data that can be represented as a graph (i.e., text, HTML, etc.). For instance, text is a set of sentences, paragraphs, or chapters (nodes) arranged in a specific order (edges).


The service implemented in this project functions by creating embeddings of the provided data (BPMN files, text, etc.). Embeddings are a type of representation that convert data into numerical vectors, capturing the inherent relationships and features within the data. Each dimension in an embedding vector corresponds to a particular feature or attribute, and the values indicate the presence and intensity of these features.  

To illustrate this, imagine a hypothetical embedding model that creates 3-dimensional embeddings with features 'Living being', 'Feline', and 'Furniture'. Here’s how different entities might be represented:  

* A 'Cat' might be represented as [0.8, 0.9, -0.2], indicating it is highly associated with both 'Living being' and 'Feline', but not with 'Furniture'.  
* 'Filip van België' might be represented as [0.8, -0.2, -0.6], suggesting it is associated with 'Living being', but not with 'Feline' or 'Furniture'.  
* A 'Table' might be represented as [-0.4, -0.6, 0.7], indicating it is not associated with 'Living being' or 'Feline', but is strongly associated with 'Furniture'.  

By using embeddings, we can search for information based on semantic similarity (meaning) rather than strict lexical matching. For instance, when searching for 'Furniture', we want results like 'tables', 'stools', and other similar items, even if they don't explicitly use the word 'Furniture'. This semantic understanding enables more intuitive and relevant search results.

The service is based on Graph Convolutional Networks (GCN). It converts the data (BPMN file or text string) to a set of nodes and edges, stores it in a graph representation, and creates embeddings of the nodes (i.e., name, documentation, etc.). The graph objects are passed to a ML model that learns to condense all the different node embeddings into a single embedding of the BPMN file or provided text. This embedding can be used in Elasticsearch to perform a dense vector search.

### Specifically for BPMN files:
1. The service extracts the ID, name, and documentation from all flow objects (lanes, events, tasks, activities, etc.) and connecting objects.  
2. Converts this data to a BPMNGraph (using library.BPMNGraph), which in turn creates a NetworkX graph from the detected nodes and edges.  
3. Using BPMNGraphEmbedderKeras (from library.BPMNGraphEmbedder), the textual attributes (name, documentation) are embedded using a pre-trained Sentence-Transformer to create node features.  
4. The node features and connections (adjacency matrix) are used as inputs for a custom-made GCN model that outputs a single embedding. This embedding encodes both the textual and structural information of the BPMN file.  
5. The embeddings of the nodes and graphs are stored in the triple store.  
6. Mu-search indexes the nodes and graphs (search-nodes and search-graphs).
7. The service provides a /search endpoint that takes in a query, embeds the query, and performs a dense vector search using mu-search.


## Getting started

1. make sure all [requirements](#Requirements-and-assumptions) are met
2. clone this repository

```bash
git clone https://github.com/lblod/app-embedding-service
```

3. run the project

```bash
cd /path/to/mu-project
```

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

You can shut down using `docker-compose stop` and remove everything using `docker-compose rm`.

## Provided endpoints

This project is build using the [lblod/mu-python-ml-fastapi-template](https://github.com/lblod/mu-python-ml-fastapi-template) and provides an interactive API documentation and exploration web user interface at `http://endpoint:2000/docs#/`. Here, all endpoints are documented and available for testing.

Most important endpoints:

* **POST /tasks/bpmn:** This endpoint accepts a `.bpmn` file, creates an embedding task, and returns a `task_uuid`. This `task_uuid` can be used to track the status and result of the embedding process.

```
curl -X 'POST' \
  'http://localhost:2000/tasks/bpmn' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@Sport-Subsidieaanvraag-GrootstGemeneDeler_v0.3.bpmn'
```

* **POST /tasks/text:** This endpoint accepts a string of text, creates an embedding task, and returns a `task_uuid`. This `task_uuid` can be used to track the status and result of the embedding process.

```

curl -X 'POST' \
  'http://localhost:2000/tasks/text' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Dit is een voorbeeld van een zin die ik wil embedden via deze fantastische service!"
}'


```
* **GET /tasks/{task_uuid}:** This endpoint retrieves the status and attributes of a task given its `task_uuid`. This allows users to check whether the embedding process is complete and to retrieve the resulting embeddings.

```
curl -X 'GET' \
  'http://localhost:2000/tasks/b6480a3a-298e-11ef-a785-0242c0a80002' \
  -H 'accept: application/json'
```
* **POST /search:** This endpoint encodes a given text into an embedding and performs a dense vector search in Elasticsearch to find semantically similar documents. 

```
curl -X 'POST' \
  'http://localhost:2000/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '"geef mij een bpmn over rijbewijzen"'
```


## Overview of services

- [mu-identifier](https://github.com/mu-semtech/mu-identifier)
- [mu-dispatcher](https://github.com/mu-semtech/mu-dispatcher)
- [mu-cl-resources](https://github.com/mu-semtech/mu-cl-resources)