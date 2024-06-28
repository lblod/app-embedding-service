# Standard library imports
import json
import os

# Third party imports
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, UploadFile, HTTPException, File, Request, Body
from fastapi.responses import JSONResponse, HTMLResponse
from werkzeug.utils import secure_filename
from pydantic import BaseModel
from typing import List, Optional

# Local application imports (from image)
from helpers import generate_uuid, query, update
from escape_helpers import sparql_escape_string, sparql_escape_uri



from library.BPMNSearch import BPMNSearch
from library.BPMNTask import BPMNTaskStatus
from library.BPMNWorker import BPMNWorkerManager
from library.BPMNGraphEmbedder import BPMNGraphEmbedderKeras


default_graph = os.getenv('DEFAULT_GRAPH', "http://mu.semte.ch/graphs/public")
queue_graph = os.getenv('QUEUE_GRAPH', "http://mu.semte.ch/graphs/tasks")
sentence_model = os.getenv('SENTENSE_EMBEDDING_MODEL', "paraphrase-multilingual-MiniLM-L12-v2")
embedding_model = os.getenv('EMBEDDING_MODEL', "/app/models/bpmn_search_0.1.0/bpmn_search_embedding_model.h5")

upload_folder = '/app/uploads/bpmn/'

accepted_file_extensions = ['.bpmn', '.txt']

# Initialize the graphEmbedder
print(f"Initializing the graphEmbedder with sentence model: {sentence_model} and graph model: {embedding_model}")
graphEmbedder = BPMNGraphEmbedderKeras(sentence_model=sentence_model, graph_model = embedding_model)

# Initialize the graphSearch
graphSearch = BPMNSearch("http://search")

# Initialize the worker manager
worker_manager = BPMNWorkerManager(1, 
                                   sleep_time=10, 
                                   queue_endpoint="http://localhost/tasks", 
                                   graph_endpoint="http://localhost/tasks/results",
                                   sentence_model=sentence_model,
                                   graph_model=embedding_model)

class Embedding(BaseModel):
    embedding: List[float]

class GraphDict(BaseModel):
    graph: dict
    nodes: List[dict]

class TextTask(BaseModel):
    text: str

async def batch_update(queries, request : Request):
    """
    Execute a batch of SPARQL INSERT queries.

    This function receives a list of SPARQL INSERT queries and a request object.
    It iterates over the list of queries and executes each one.

    Args:
    queries: A list of SPARQL INSERT queries to be executed.
    request: A Request object that contains client request information.

    Returns:
    None. The function executes the queries but does not return anything.
    """
    for query in queries:
        update(query, request)

def generate_insert_query(graph_dict, sparql_graph=default_graph):
    """
    Generate SPARQL INSERT queries from a dictionary.

    Parameters:
    graph_dict (dict): The dictionary to generate queries from.
    sparql_graph (str): The URI of the SPARQL graph to insert data into.

    Returns:
    list: A list of SPARQL INSERT queries.
    """

    # Initialize the list of queries
    queries = []

    # Generate the query for the graph
    uid = graph_dict['graph']['uuid']
    encoded_uri = sparql_escape_uri(f"http://deepsearch.com/{uid}")
    query = "PREFIX ds: <http://deepsearch.com/search#>"
    query += "PREFIX mu: <http://mu.semte.ch/vocabularies/core/>"
    query += f"INSERT DATA {{ GRAPH <{sparql_graph}> {{ {encoded_uri} a ds:Graph ; ds:filePath {sparql_escape_string(graph_dict['graph']['file_path'])}; ds:data {sparql_escape_string(graph_dict['graph']['data'])} ; ds:embedding {sparql_escape_string(json.dumps(graph_dict['graph']['embedding']))} ; mu:uuid {sparql_escape_string(uid)} . }} }}"
    queries.append(query)

    # Generate the queries for the nodes
    for node in graph_dict['nodes']:
        if node.get('embedding') is not None:
            node_uid = generate_uuid()
            node_encoded_uri = sparql_escape_uri(f"http://deepsearch.com/{node_uid}")
            query = "PREFIX ds: <http://deepsearch.com/search#>"
            query += "PREFIX mu: <http://mu.semte.ch/vocabularies/core/>"
            query += f"INSERT DATA {{ GRAPH <{sparql_graph}> {{ {encoded_uri} ds:hasNode {node_encoded_uri} . {node_encoded_uri} a ds:Node ; ds:name {sparql_escape_string(node.get('name'))} ; ds:embedding {sparql_escape_string(json.dumps(node.get('embedding')))} ; mu:uuid {sparql_escape_string(node_uid)} . }} }}"
            queries.append(query)

    return queries



@app.on_event("startup")
async def startup_event():
    worker_manager.start_workers()

@app.on_event("shutdown")
async def shutdown_event():
    worker_manager.stop_workers()

@app.post("/restart_workers")
async def restart_workers():
    worker_manager.stop_workers()
    worker_manager.start_workers()
    return {"message": "Workers restarted successfully"}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <html>
        <head>
            <title>FastAPI Home</title>
        </head>
        <body>
            <h1>Welcome to our embedding service for semantic search over text and bpmn graphs</h1>
            <p>For API documentation, visit <a href="/docs">/docs</a></p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
#Handling tasks
@app.post("/tasks/bpmn", tags=["tasks"])
async def create_task(file: UploadFile, request: Request):
    """
    Create a new task and add it to the task queue.

    This function receives a file, saves it, creates a new task with the given attributes, 
    and adds it to the task queue in the application graph.

    Args:
        file (UploadFile): The file to be processed.

    Returns:
        JSONResponse: A JSON response containing the 'task' field which is the 
        URI of the newly created task.
    """
    # Check if the file is a BPMN file

    # Check if the file has an accepted extension
    if not any(file.filename.endswith(ext) for ext in accepted_file_extensions):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a file with one of the following extensions: " + ', '.join(accepted_file_extensions))

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Generate a UUID for the task
    uuid = generate_uuid()

    # Create the task URI
    task_uri = f"http://deepsearch.com/tasks/{uuid}"

    # Create the SPARQL INSERT query
    query = f"""
        PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
        PREFIX task: <http://deepsearch.com/task#>

        INSERT DATA {{
            GRAPH <{queue_graph}> {{
                <{task_uri}> a task:Task ;
                    mu:uuid "{uuid}" ;
                    task:status "{BPMNTaskStatus.PENDING.value}" ;
                    task:type "bpmn-embedding" ;
                    task:data "{file_path}" ;
                    task:graph_id "{uuid}" ;
                    task:retry_count "0" ;
                    task:log "" .
            }}
        }}
    """

    # Execute the SPARQL INSERT query
    try:
        update(query, request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(content={"task": uuid})


# Handling tasks from raw text (testing)
@app.post("/tasks/text", tags=["tasks"])
async def create_task_from_text(task: TextTask, request: Request):
    """
    Create a new task from raw text and add it to the task queue.
    """

    # Generate a UUID for the task
    uuid = generate_uuid()

    # Create the task URI
    task_uri = f"http://deepsearch.com/tasks/{uuid}"

    # Create the SPARQL INSERT query
    query = f"""
        PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
        PREFIX task: <http://deepsearch.com/task#>

        INSERT DATA {{
            GRAPH <{queue_graph}> {{
                <{task_uri}> a task:Task ;
                    mu:uuid "{uuid}" ;
                    task:status "{BPMNTaskStatus.PENDING.value}" ;
                    task:type "text-embedding" ;
                    task:data {sparql_escape_string(task.text)} ;
                    task:graph_id "{uuid}" ;
                    task:retry_count "0" ;
                    task:log "" .
            }}
        }}
    """

    # Execute the SPARQL INSERT query
    try:
        update(query, request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(content={"task": uuid})

@app.get("/tasks/{task_uuid}", tags=["tasks-monitoring"])
async def get_task(task_uuid: str, request: Request):
    """
    Get a task.

    This function receives a task UUID, queries the task queue in the application graph 
    for the task with the given UUID, and returns all its attributes.

    Args:
        task_uuid (str): The UUID of the task.

    Returns:
        dict: The task attributes.
    """
    # Create the SPARQL SELECT query
    query_string = f"""
            PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
            PREFIX task: <http://deepsearch.com/task#>

            SELECT ?status ?type ?data ?graph_id ?retry_count ?log WHERE {{
                GRAPH <{queue_graph}> {{
                    ?task a task:Task ;
                        mu:uuid "{task_uuid}" ;
                        task:status ?status ;
                        task:type ?type ;
                        task:data ?data ;
                        task:graph_id ?graph_id ;
                        task:retry_count ?retry_count ;
                        task:log ?log .
                }}
            }}
        """

    # Execute the SPARQL SELECT query
    try:
        result = query(query_string, request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check if a task with the given UUID exists
    if not result:
        raise HTTPException(status_code=404, detail="Task not found.")

    # logging
    print("Task found. Returning task attributes.")
    print(result)

    # Return the task attributes
    task = result["results"]["bindings"][0]
    return {
        "status": str(task["status"]["value"]),
        "type": str(task["type"]["value"]),
        "data": str(task["data"]["value"]),
        "graph_id": str(task["graph_id"]["value"]),
        "retry_count": int(task["retry_count"]["value"]),
        "log": str(task["log"]["value"]),
    }

@app.get("/tasks", tags=["tasks-monitoring"])
async def get_tasks(request: Request, status: BPMNTaskStatus, limit: int = 1):
    """
    Get all tasks with a specified status.

    This function receives a status, queries the task queue in the application graph 
    for tasks with the given status, and returns all their attributes.

    Args:
        status (str): The status of the tasks to retrieve.

    Returns:
        list: A list of tasks with their attributes.
    """
    # Create the SPARQL SELECT query
    query_string = f"""
        PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
        PREFIX task: <http://deepsearch.com/task#>

        SELECT ?uuid ?type ?data ?graph_id ?retry_count ?log WHERE {{
            GRAPH <{queue_graph}> {{
                ?task a task:Task ;
                    mu:uuid ?uuid ;
                    task:status "{status.value}" ;
                    task:type ?type ;
                    task:data ?data ;
                    task:graph_id ?graph_id ;
                    task:retry_count ?retry_count ;
                    task:log ?log .
            }}
        }}
        LIMIT {limit}
    """

    # Execute the SPARQL SELECT query
    try:
        result = query(query_string, request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check if any tasks with the given status exist
    if not result:
        raise HTTPException(status_code=404, detail="No tasks found with the specified status.")

    # Return the task attributes
    tasks = result["results"]["bindings"]
    return [
        {
            "uuid": str(task["uuid"]["value"]),
            "status": status,
            "type": str(task["type"]["value"]),
            "data": str(task["data"]["value"]),
            "graph_id": str(task["graph_id"]["value"]),
            "retry_count": int(task["retry_count"]["value"]),
            "log": str(task["log"]["value"]),
        }
        for task in tasks
    ]

# Handling task results

@app.post("/tasks/results", tags=["tasks"])
async def store_task_results(request: Request, graph_dict: GraphDict, sparql_graph: Optional[str] = default_graph):
    """
    Endpoint for generating SPARQL INSERT queries from a dictionary.

    Parameters:
    graph_dict (dict): The dictionary to generate queries from.
    sparql_graph (str): The URI of the SPARQL graph to insert data into.

    Returns:
    list: A list of SPARQL INSERT queries.
    """
    try:
        queries = generate_insert_query(graph_dict.dict(), sparql_graph)
        # Assuming batch_update is an async function
        await batch_update(queries, request)

        return {"status": "success", "message": "Batch update completed successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Handling search
@app.post("/encode", response_model=Embedding, tags=["embedding"])
async def encode(text: str = Body(...)):
    """
    Encode a given text into an embedding.

    This function receives a JSON payload with a 'text' field, 
    encodes the text into an embedding using the graphEmbedder's 
    embedding model, and returns the embedding as a JSON response.

    Returns:
    A JSON response containing the 'embedding' field which is a list 
    of floats representing the embedding of the input text.
    """
    encoded_text = graphEmbedder.get_embedding_model().encode(text)
    return {"embedding": encoded_text.tolist()}

@app.post("/search", tags=["search"])
async def search(text: str = Body(...)):
    """
    Encode a given text into an embedding and perform a k-NN search in Elasticsearch.

    This function receives a JSON payload with a 'text' field, 
    encodes the text into an embedding using the graphEmbedder's 
    embedding model, and sends a POST request to the Elasticsearch 
    endpoint to perform a k-NN search.

    Returns:
    A JSON response containing the 'hits' field which is a list 
    of hits returned by the k-NN search.
    """
    # Encode the text into an embedding
    encoded_text = graphEmbedder.get_embedding_model().encode(text)
    encoded_text = encoded_text.tolist()

    # Logging
    print(f"Text: {text}")
    print(f"Encoded Text: {encoded_text}")

    # Perform a k-NN search in Elasticsearch
    hits = graphSearch.mu_knn_search("search-graphs", {"text":text,"embedding": encoded_text}, size=10)

    return {"response": hits}