# app-embedding-service
 
Backend for creating embeddings for BPMN files and text, leveraging advanced machine learning techniques to enhance search capabilities. Built on the mu.semte.ch microservices stack, this service combines lexical and semantic search to provide more accurate and meaningful search results.

## What does it do?

This project demonstrates how the search functionality of mu-search can be improved using Machine Learning (ML). In this project, the lexical search of mu-search is combined with semantic search to enable natural language queries (e.g., 'give me a BPMN about handling disputes'). The main focus of this project is on BPMN files, but **it generalizes to any kind of data** that can be represented as a graph (i.e., text, HTML, etc.). For instance, text is a set of sentences, paragraphs, or chapters (nodes) arranged in a specific order (edges).


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

This project is build using the [lblod/mu-python-ml-fastapi-template](https://github.com/lblod/mu-python-ml-fastapi-template) and provides an interactive API documentation and exploration web user interface at `http://endpoint/docs#/`. Here, all endpoints are documented and available for testing.

### Most important endpoints:

* **POST /tasks/bpmn:** This endpoint accepts a `.bpmn` file, creates an embedding task, and returns a `task_uuid`. This `task_uuid` can be used to track the status and result of the embedding process.

```
curl -X 'POST' \
  'http://endpoint/tasks/bpmn' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@Sport-Subsidieaanvraag-GrootstGemeneDeler_v0.3.bpmn'
```

* **POST /tasks/text:** This endpoint accepts a string of text, creates an embedding task, and returns a `task_uuid`. This `task_uuid` can be used to track the status and result of the embedding process.

```

curl -X 'POST' \
  'http://endpoint/tasks/text' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Dit is een voorbeeld van een zin die ik wil embedden via deze fantastische service!"
}'


```
* **GET /tasks/{task_uuid}:** This endpoint retrieves the status and attributes of a task given its `task_uuid`. This allows users to check whether the embedding process is complete and to retrieve the resulting embeddings.

```
curl -X 'GET' \
  'http://endpoint/tasks/b6480a3a-298e-11ef-a785-0242c0a80002' \
  -H 'accept: application/json'
```
* **POST /search:** This endpoint encodes a given text into an embedding and performs a dense vector search in Elasticsearch to find semantically similar documents. 

```
curl -X 'POST' \
  'http://endpoint:2000/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '"geef mij een bpmn over rijbewijzen"'
```


## Deep dive into the architecture

### Handling of request
The service is designed to handle large amounts of request at ones, it creates tasks and completes the tasks one-by-one (or more depending on the number of workers) in the background.

On the startup of the service, a `library.BPMNWorker.BPMNWorkerManager` is created, which in turn initializes a number of `library.BPMNWorker.BPMNWorker` instances with a reference to a singleton `library.BPMNQueue` from which the workers fetch new tasks. Each `library.BPMNWorker.BPMNWorker` initializes a `library.BPMNGraphEmbedder` instance that it uses to process incoming tasks. While a worker is active, it periodically calls `Work` and checks if there are any pending tasks in the queue. If there are no pending tasks, it goes back into sleep mode and waits again for a fixed duration depending on the sleep_time parameter. When a pending task is detected, the worker calls process_task(`library.BPMNTask`) and creates a `library.BPMNGraph` object from the data stored in the `library.BPMNTask`, adding the embeddings to the `BPMNGraph` using the `library.BPMNGraphEmbedder` (use `BPMNGraphEmbedder` for text or `BPMNGraphEmbedderKeras` for BPMNGraphs) it was initialized with. After completing the embedding process, the BPMNGraph is converted to a JSON object using BPMNGraph.to_dict(), and a post request is made to the specified endpoint (default: `localhost/tasks/results`) in the task, for storage onto the triple store.

### Handling of search
For search, the service itself initializes a `library.BPMNGraphEmbedder` instance to create embeddings of the search query and uses a `library.search` instance with a URL of the mu-search endpoint (`http://search`) or Elasticsearch endpoint (`http://localhost:9200`). It performs a dense vector search using either the raw endpoint or the POST on the mu-search endpoint (`http://search/search`) that accepts a raw Elasticsearch query.

### Working with Text vs BPMN Files

The model is designed to work primarily with BPMN diagrams; however, it can easily be extended to work with any kind of text documents as well. This is because any kind of text can be converted to a graph by splitting it into nodes based on sentences, paragraphs, HTML tags, etc. (see `library.chunking`, `library.BPMNGraphEmbedder.TextEmbedder`). Using `library.BPMNGraph` and the `process_txt` and `process_text` functions, a TXT file or raw text can be processed into a BPMNGraph object that, in turn, can be processed identically to a BPMN file using the `library.BPMNGraphEmbedder.BPMNGraphEmbedder` or `library.BPMNGraphEmbedder.BPMNGraphEmbedderKeras`.**`library.BPMNGraph` automatically detects the data type based on the extension and defaults to `text`. No additional steps are required to get a baseline working. Simple create tasks using POST \tasks\text with the text data.**



A branch with the minimal necessary changes on how to use it with text (agendapunten) is shown in [branch/lokaal-beslist](https://github.com/lblod/app-embedding-service/tree/lokaal-beslist).


However, it is important to note that:

* The default embedding model `paraphrase-multilingual-MiniLM-L12-v2` is not ideal for text (this can be set using the environment variable `SENTENCE_EMBEDDING_MODEL`). Better results can be achieved using `textgain/tags-allnli-GroNLP-bert-base-dutch-cased`.
* The custom Keras models used in `library.BPMNGraphEmbedder.BPMNGraphEmbedderKeras` are designed to work with `paraphrase-multilingual-MiniLM-L12-v2` embeddings of size 384 and are explicitly trained on matching BPMN files with search queries. They have not been tested on different data. Further fine-tuning on text-query pairs should be done if you want to use the custom models for text. Otherwise, stick to the average embeddings (stored under `BPMNGraph.get_graph()["embedding_average"]` when using `library.BPMNGraphEmbedder.BPMNGraphEmbedderKeras`) or the embeddings of `library.BPMNGraphEmbedder.BPMNGraphEmbedder`.
* **When using the model for text search together with `textgain/tags-allnli-GroNLP-bert-base-dutch-cased` or any other sentence-transformer model from [Hugging Face](https://huggingface.co), you will need to change the `dims` parameter in mu-search accordingly** (e.g., 768 for BERT-based models).
* Fine-tuning is always advised for the best results.

# finetuning of search and similarity models

## Training a model for search
See the Jupyter notebook (`model-training/search_spektral.ipynb`) which describes the process of training a custom GCN network using Spektral with the artificial data generated by (`model-training/extraction_openai.ipynb`).

For the search, we use an artificial dataset created using OpenAI GPT-4, partially manually annotated using [Prodigy](https://prodi.gy). For this manual evaluation in Prodigy, a custom recipe `bpmn-search` is created to evaluate BPMN diagrams using the textual descriptions generated in `model-training/extraction_openai.ipynb`. The custom recipe can be found under `model-training/prodigy`.

For the training of the Spektral model, the model expects a `spektral.data.Dataset` object, which requires converting the BPMNGraph objects to `spektral.data.Graph` objects. This is done using the `BPMNGraphProcessor` within the `BPMNGraphDataset`, an implementation of `spektral.data.Dataset` that takes in a list of samples structured as follows:

```json
{
    "query": "str",
    "query_embedding": "'paraphrase-multilingual-MiniLM-L12-v2' embedding",
    "bpmn": "BPMNGraph",
    "label": "label"
}
```

BPMNGraph objects can be generated by the `library.BPMNGraph` class by passing the path to the file during initialization. Query embeddings can be generated by either the `library.BPMNGraphEmbedder` models or directly from the `sentence-transformer` library. Labels are 0 or 1, depending on whether the query should match the bpmn file or not.

Once the Spektral dataset has been created, a model architecture can be designed using a combination of Spektral and Keras layers. GCNConv is used for propagating information to neighboring nodes, with more layers allowing for further "hops.". Models are available within the `source\models` folder and on [svercoutere/bpmn-search-0.1.0](https://huggingface.co/svercoutere/bpmn-search-0.1.0).

Here is an example of how you can define and create the models:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Dot
from spektral.layers import GCNConv, GlobalAttentionPool
from tensorflow.keras.metrics import Precision, Recall, AUC

def create_models(n_features, embedding_dims):
    # Define the inputs for node features and adjacency matrix
    X_in = Input(shape=(None, n_features), name='Node_features')
    A_in = Input((None, ), sparse=True, name='Adjacency_matrix')
    Z_in = Input((embedding_dims, ), name='Query_embedding')

    # Define the graph convolutional network layers
    graph_conv = GCNConv(embedding_dims, activation='relu', name='graph_conv')([X_in, A_in])
    dropout = Dropout(0.2, name='dropout')(graph_conv)
    graph_conv_2 = GCNConv(embedding_dims, activation='relu', name='graph_conv_2')([dropout, A_in])
    dropout_2 = Dropout(0.2, name='dropout_2')(graph_conv_2)
    graph_conv_3 = GCNConv(embedding_dims, activation='relu', name='graph_conv_3')([dropout_2, A_in])

    # Pooling to create a graph embedding
    graph_embedding = GlobalAttentionPool(embedding_dims, name='global_pool')(graph_conv_3)

    # Perform dot product between the graph embedding and the query embedding
    dot_product = Dot(axes=1, normalize=True, name='dot_product')([graph_embedding, Z_in])

    # Create the embedding model
    embedding_model = Model(inputs=[X_in, A_in], outputs=graph_embedding, name='embedding_model')

    # Create the prediction model
    prediction_model = Model(inputs=[X_in, A_in, Z_in], outputs=dot_product, name='prediction_model')

    # Compile the models
    prediction_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall(), AUC()])
    embedding_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall(), AUC()])

    return prediction_model, embedding_model

# Example usage:
# Create the models with specified number of node features and embedding dimensions
model, embedding_model = create_models(n_features=dataset.n_node_features, embedding_dims=dataset.embedding_dims)

# Print the model summary
model.summary()
```
Once the Spektral dataset has been created, a data loader can be implemented to efficiently feed the data into the model during training. This involves creating a Spektral `Loader` that handles batching and shuffling of the dataset. The model can then be trained and tested using typical Keras methods such as `fit` and `evaluate`. The `fit` method is used to train the model on the training dataset, while the `evaluate` method assesses the model's performance on the testing dataset. By leveraging these standard Keras functionalities, the process of training and testing the GCN model with Spektral becomes straightforward and consistent with other deep learning workflows, enabling effective model development and evaluation.

## Training a Model for Similarity

See the Jupyter notebook (`model-training/compare_spektral.ipynb`) which describes the process of training a custom GCN network using Spektral with the artificial data generated by (`model-training/extraction_camunda.ipynb`) using the data of [camunda/bpmn-for-research](https://github.com/camunda/bpmn-for-research).

For the similarity/compare model, we use an artificial dataset created using the Camunda dataset and perform a pairwise comparison between random BPMN files from four possible categories (restaurant, credit-scoring, recourse, dispatching). The model gives a score between 0 and 1 based on the number of edits needed to transform one BPMN file into another (`minimum edit distance`) in terms of node and edge operations (replace, insert, delete).

For the training of the Spektral model, the model expects a `spektral.data.Dataset` object, which requires converting the BPMNGraph objects to `spektral.data.Graph` objects. This is done using the `BPMNGraphProcessor` within the `BPMNGraphDataset`, an implementation of `spektral.data.Dataset` that takes in a list of samples structured as follows:

```json
{
    "query": "BPMNGraph",
    "target": "BPMNGraph",
    "score": "float [0.0-1.0]",
    "query_class": "str",
    "target_class": "str"
}
```

BPMNGraph objects can be generated by the `library.BPMNGraph` class by passing the path to the file during initialization. Minimum edit distances can be calculated using using the `Networkx` python package as can be see in (`model-training/extraction_camunda.ipynb`).


The remaining steps are largely identical to the previous section on the search model. An example of how to design one for the compare model can be seen in `model-training/compare_spektral.ipynb`. Once the Spektral dataset has been created, a data loader can be implemented to efficiently feed the data into the model during training. This involves creating a Spektral `Loader` that handles batching and shuffling of the dataset. The model can then be trained and tested using typical Keras methods such as `fit` and `evaluate`. The `fit` method is used to train the model on the training dataset, while the `evaluate` method assesses the model's performance on the testing dataset. By leveraging these standard Keras functionalities, the process of training and testing the GCN model with Spektral becomes straightforward and consistent with other deep learning workflows, enabling effective model development and evaluation.

# Future steps: what should be done next?
  * **Start gathering information:** Gather BPMN files, user queries, and other relevant data as training can only be done with enough data.
  * Requirements: It's important to have a large enough dataset of BPMN files with enough variety to train the embeddings and validate the model. For example, if all the data is about the same process (handling of disputes), the model will not be able to generalize to other processes. It's hard to quantify how much data is needed, but a few hundred BPMN files should be a good start.
    * Train our own text embeddings for the nodes and edges using a model such as RoBERTa trained on BPMN files ([svercoutere/robbert-2023-dutch-base-abb](https://huggingface.co/svercoutere/robbert-2023-dutch-base-abb)).
    * Validate the current Spektral model using the new data.
    * Change the architecture of the GNN or GCN model if needed (possibly add more layers, change the activation functions, etc.).
    * Merge the two models: There is a lot of overlap between the two models, so it might be possible to merge them into a single model and use the graph embeddings for both tasks.

# Demo

A small demo of the search functionality is available under `source\demos`. The demo uses the Python `streamlit` package to create a simple interface where queries can be made using the mu-search dense vector search with the embedding created from the service.

To run the demo, you can follow these steps:

1. **Install Streamlit**:
   Ensure you have Streamlit installed in your environment. You can install it using pip:
   ```bash
   pip install streamlit
   ```

2. **Run the Streamlit Application**:
   Navigate to the `source\demos` directory and run the Streamlit app:
   ```bash
   streamlit run demo_semantic_search.py
   ```

3. **Using the Interface**:
   Open the provided local URL (typically `http://localhost:8501`) in a web browser. The interface allows you to input queries, which will be processed using the mu-search dense vector search with embeddings generated by the service. 
