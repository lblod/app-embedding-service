import requests
import time
import json
import threading
import concurrent.futures
from library.BPMNTask import BPMNTask, BPMNTaskStatus
from library.BPMNGraph import BPMNGraph
from library.BPMNGraphEmbedder import BPMNGraphEmbedder
from library.BPMNQueue import BPMNQueue

class BPMNWorkerManager:
    def __init__(self, worker_count, queue_endpoint, graph_endpoint, sleep_time = 5, sentence_model='paraphrase-multilingual-MiniLM-L12-v2', graph_model='/app/models/bpmn_search_0.1.0/bpmn_search_embedding_model.h5', **kwargs):
        self.worker_count = worker_count
        self.queue_endpoint = queue_endpoint
        self.graph_endpoint = graph_endpoint
        self.sleep_time = sleep_time

        self.sentence_model = sentence_model
        self.graph_model = graph_model

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.workers = [BPMNWorker(queue_endpoint, graph_endpoint, sleep_time, sentence_model, graph_model, **kwargs) for _ in range(worker_count)]

    def start_workers(self):
        for worker in self.workers:
            worker.reset()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_count)
        self.futures = [self.executor.submit(worker.work) for worker in self.workers]

    def stop_workers(self):
        for worker in self.workers:
            worker.stop()
        self.executor.shutdown(wait=True)

class BPMNWorker:
    def __init__(self, queue_endpoint, graph_endpoint, sleep_time, sentence_model = 'paraphrase-multilingual-MiniLM-L12-v2', graph_model = '/app/models/bbpmn_search_0.1.0/bpmn_search_embedding_model.h5', **kwargs):
        self.queue_endpoint = queue_endpoint
        self.graph_endpoint = graph_endpoint
        self.queue = BPMNQueue.get_instance(queue_endpoint)
        self.graphEmbedder = BPMNGraphEmbedder(sentence_model)
        self.stop_event = threading.Event()  # Replace stop_flag with an Event
        self.sleep_time = sleep_time  # Add sleep time

        for key, value in kwargs.items():
            setattr(self, key, value)

    def post_results(self, task, bpmn_graph):
        post_data = bpmn_graph.to_dict()
        num_nodes = len(post_data.get("nodes", []))
        # perform post request to the results_endpoint
        response = requests.post(self.graph_endpoint, json=post_data)
        response_data = response.json()
        if response.status_code != 200:
            error_message = "Failed to post results. "
            error_message += f"Failed for following post request: {post_data}. "
            error_message += f"Error server response: {response_data}. "

            raise Exception(error_message)
    
        else:
            #print(f"Results posted successfully. Inserted {num_nodes} nodes.")
            task.update_log(f"Inserted {num_nodes} nodes.")
    
    def process_task(self, task):
        task.update_status(BPMNTaskStatus.RUNNING)
    
        try:
            # Process the BPMN graph
            #print(f"1.Processing graph in worker: {task.graph_id}")
            bpmn_graph = BPMNGraph(graph_uuid = task.graph_id, data = task.data)
            #print(f"2.Graph representation successfully loaded. ")
            self.graphEmbedder.process_graph(bpmn_graph,text_attributes=["text","name","documentation"], logging=False)
            #print(f"3.Graph embedding successfully computed. ")
            # Post the results back to the graph
            self.post_results(task, bpmn_graph)
            #print(f"4.Results posted successfully. ")
            # If the graph processing is successful, set task status to completed
            task.update_status(BPMNTaskStatus.COMPLETED)
            #print(f"5.Task completed successfully. ")
    
        except Exception as e:
            error_message = f"Failed to process BPMN graph: {e}"
            task.update_log(error_message)
    
            # If the graph processing fails, set task status to failed
            task.update_status(BPMNTaskStatus.FAILED)
    
    def stop(self):
        self.stop_event.set()  # Set the stop event

    def reset(self):
        self.stop_event.clear()  # Clear the stop event
        
    def work(self):
        while not self.stop_event.is_set():  # Check the stop event
            task = self.queue.get_pending_task()
            if task:
                print(f"Processing task: {task.uuid}")
                self.process_task(task)
            else:
                print(f"No pending tasks. Sleeping for {self.sleep_time} seconds...")
                self.stop_event.wait(self.sleep_time)  # Wait for stop event or sleep time

