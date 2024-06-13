from enum import Enum
from typing import Optional
from library.BPMNQueue import BPMNQueue
import os

class BPMNTaskStatus(Enum):
    ASSIGNED = "assigned"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class BPMNTask:
    def __init__(self, uuid: str, queue: BPMNQueue, status: BPMNTaskStatus = BPMNTaskStatus.PENDING, type: str = "bpmn-embedding", retry_count: int = 0, log: str = "", **kwargs):
        self.uuid = uuid
        self.queue = queue
        self.status = status
        self.type = type
        self.retry_count = retry_count
        self.log = log

        for key, value in kwargs.items():
            setattr(self, key, value)


    # The following methods generate SPARQL queries to update the status, retry count, and log of a task in the task queue.
    def generate_update_status_query(self):
        return f"""
            PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
            PREFIX task: <http://deepsearch.com/task#>

            DELETE {{
                GRAPH <{self.queue.queueGraph}> {{
                    <http://deepsearch.com/tasks/{self.uuid}> task:status ?status .
                }}
            }}
            INSERT {{
                GRAPH <{self.queue.queueGraph}> {{
                    <http://deepsearch.com/tasks/{self.uuid}> task:status "{self.status.value}" .
                }}
            }}
            WHERE {{
                GRAPH <{self.queue.queueGraph}> {{
                    <http://deepsearch.com/tasks/{self.uuid}> task:status ?status .
                }}
            }}
        """

    def generate_increment_retry_count_query(self):
        return f"""
            PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
            PREFIX task: <http://deepsearch.com/task#>

            DELETE {{
                GRAPH <{self.queue.queueGraph}> {{
                    <http://deepsearch.com/tasks/{self.uuid}> task:retry_count ?retry_count .
                }}
            }}
            INSERT {{
                GRAPH <{self.queue.queueGraph}> {{
                    <http://deepsearch.com/tasks/{self.uuid}> task:retry_count "{self.retry_count}" .
                }}
            }}
            WHERE {{
                GRAPH <{self.queue.queueGraph}> {{
                    <http://deepsearch.com/tasks/{self.uuid}> task:retry_count ?retry_count .
                }}
            }}
        """

    def generate_update_log_query(self):
        return f"""
            PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
            PREFIX task: <http://deepsearch.com/task#>

            DELETE {{
                GRAPH <{self.queue.queueGraph}> {{
                    <http://deepsearch.com/tasks/{self.uuid}> task:log ?log .
                }}
            }}
            INSERT {{
                GRAPH <{self.queue.queueGraph}> {{
                    <http://deepsearch.com/tasks/{self.uuid}> task:log "{self.log}" .
                }}
            }}
            WHERE {{
                GRAPH <{self.queue.queueGraph}> {{
                    <http://deepsearch.com/tasks/{self.uuid}> task:log ?log .
                }}
            }}
        """


  # The following methods update the status, retry count, and log of a task in the task queue.
    def update_status(self, new_status: BPMNTaskStatus):
        self.status = new_status
        query = self.generate_update_status_query()
        try:
            self.queue.update(query)
        except Exception as e:
            print(f"Failed to update status: {e}")

    def increment_retry_count(self):
        self.retry_count += 1
        query = self.generate_increment_retry_count_query()
        try:
            self.queue.update(query)
        except Exception as e:
            print(f"Failed to increment retry count: {e}")

    def update_log(self, new_log: str):
        self.log = new_log
        query = self.generate_update_log_query()
        try:
            self.queue.update(query)
        except Exception as e:
            print(f"Failed to update log: {e}")
    
    
    @staticmethod
    def from_dict(json_dict, queue: BPMNQueue):

        # Extract known parameters
        known_params = {
            'queue': queue,
            'uuid': json_dict['uuid'],
            'status': BPMNTaskStatus(json_dict['status']),
            'type': json_dict.get('type', "search-embedding"),  # Use get() to provide default value
            'retry_count': json_dict.get('retry_count', 0),  # Use get() to provide default value
            'log': json_dict.get('log', "")  # Use get() to provide default value
        }
    
        # Extract any additional parameters
        additional_params = {k: v for k, v in json_dict.items() if k not in known_params}
    
        # Create the BPMNTask instance with both known and additional parameters
        return BPMNTask(**known_params, **additional_params)