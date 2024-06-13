import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
import os
import uuid

from library.chunking import TextChunker


class BPMNGraph:
    """
    A class used to process BPMN files and store the information in a NetworkX graph.

    ...

    Attributes
    ----------
    data : str
        a string representing the path to a BPMN file, or the text of a BPMN file
    namespaces : dict
        a dictionary mapping the bpmn prefix to its URI
    graph : NetworkX.DiGraph
        a directed graph representing the BPMN file

    Methods
    -------
    process_elements(process, namespaces, nodes, logging=False)
        Processes elements with a 'name' or 'id' attribute within the BPMN file.
    process_lanes(process, namespaces, nodes, edges, logging=False)
        Processes lane elements within the BPMN file.
    process_edges(process, namespaces, edges, tag, logging=False)
        Processes edges of a given tag within the BPMN file.
    process_bpmn(logging=False)
        Processes the BPMN file and stores the information in the graph.
    draw_graph(fig_size=10)
        Draws the graph.
    get_all_paths(logging=False)
        get all paths in the graph. Returns a list of nodes of each path.
    """
        
    def __init__(self, data, graph_uuid=None, logging=False):
        """
        Constructs a BPMNGraph object from a BPMN file or a text string.
    
        Parameters
        ----------
        data : str
            a string representing the path to the BPMN file or the text to process.
        graph_uuid : str, optional
            a string representing the unique identifier for the graph. If None, a new UUID is generated.
        logging : bool, optional
            if True, enables logging during the processing of the BPMN file. Defaults to False.
    
        The function initializes the graph_uuid (generating a new UUID if necessary), file_path, namespaces, and network attributes, and then calls the process_bpmn or process_text method to process the BPMN file or text string.
        """
    
        if graph_uuid is None:
            graph_uuid = str(uuid.uuid4())
    
        self.graph_uuid = graph_uuid
        self.namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        self.network = nx.DiGraph()
        self.file_path = None
        self.data = data

        if data is not None:
            if os.path.isfile(data):
                self.file_path = data
                _, file_extension = os.path.splitext(self.file_path)
                if file_extension == '.bpmn':
                    self.process_bpmn(logging=logging)
                elif file_extension == '.txt':
                    self.process_txt(logging=logging)
                else:
                    raise ValueError(f"Unsupported file extension: {file_extension}")
            else:
                self.process_text(data, logging=logging)
        else:
            raise ValueError("Data must be provided.")

    def get_nx_graph(self):
        return self.network
    
    def get_graph(self):
        return self.network.graph
    
    def get_nodes(self):
        return self.network.nodes(data=True)
    
    def get_edges(self):
        return self.network.edges(data=True)
   
    def process_elements(self, process, namespaces, nodes, logging=False):
        """
        This function processes elements within a BPMN file and creates nodes for each element.

        Parameters:
        process (xml.etree.ElementTree.Element): The root element of the BPMN XML file.
        namespaces (dict): A dictionary mapping namespace prefixes to full namespace URIs.
        nodes (dict): A dictionary to store the created nodes. Keys are node IDs, values are dictionaries containing node attributes.
        logging (bool, optional): If True, prints information about each processed node. Defaults to False.

        The function iterates over all elements in the 'process' element that have a 'name' or 'id' attribute.
        For each element, it extracts the 'id', 'name', and 'type' attributes, as well as any documentation text.
        It also finds all incoming and outgoing connections for the element.
        If the element has no incoming or outgoing connections and no name or documentation, it is skipped.
        Otherwise, a node is created with the extracted attributes and connections, and added to the 'nodes' dictionary.

        If 'logging' is True, the function also prints information about each processed node.
        """
        for element in process.findall('.//*', namespaces):
            node_id = element.attrib.get('id', '')
            node_name = element.attrib.get('name', '')
            node_type = element.tag.replace('{http://www.omg.org/spec/BPMN/20100524/MODEL}', '')
            
            documentation = element.find('.//bpmn:documentation', namespaces)
            documentation_text = ''
            if documentation is not None and documentation.text is not None:
                documentation_text = documentation.text.replace('\n', '').replace('\t', '').strip()

            incoming_connections = [incoming.text for incoming in element.findall('.//bpmn:incoming', namespaces)]
            outgoing_connections = [outgoing.text for outgoing in element.findall('.//bpmn:outgoing', namespaces)]

            # Skip elements without incoming or outgoing connections
            if not incoming_connections and not outgoing_connections and node_name == '' and documentation_text == '':
                continue

            # Add the node to the graph
            nodes[node_id] = {
                'type': node_type,
                'name': node_name,
                'documentation': documentation_text,
                'incoming': incoming_connections,
                'outgoing': outgoing_connections
            }

            if logging:
                # Print the element
                print("---------------------------------------------")
                print('id:', node_id)
                print('type:', node_type)
                print('name:', node_name)
                print('documentation:', documentation_text)
                print('incoming:', incoming_connections)
                print('outgoing:', outgoing_connections)

    def process_lanes(self, process, namespaces, nodes, edges, logging = False):
        """Process lane elements within the BMPN file."""
        for lane in process.findall('.//bpmn:lane', namespaces):
            lane_id = lane.attrib['id']
            lane_type = lane.tag.replace('{http://www.omg.org/spec/BPMN/20100524/MODEL}', '')
            lane_name = lane.attrib.get('name', '')

            documentation = lane.find('.//bpmn:documentation', namespaces)
            documentation_text = ''
            if documentation is not None and documentation.text is not None:
                documentation_text = documentation.text.replace('\n', '').replace('\t', '').strip()

            incoming_connections = [process.attrib.get('id', '')]
            flowNodeRefs = [flowNodeRef.text for flowNodeRef in lane.findall('.//bpmn:flowNodeRef', namespaces)]

            # Add the node to the graph
            nodes[lane_id] = {
                'type': lane_type,
                'name': lane_name,
                'documentation': documentation_text,
                'incoming': incoming_connections,
                'outgoing': flowNodeRefs
            }

            for flowNodeRef in flowNodeRefs:
                edges[lane_id] = (lane_id, flowNodeRef)


            if logging:
                # Print the lane
                print("---------------------------------------------")
                print('id:', lane_id)
                print('type:', lane_type)
                print('name:', lane_name)
                print('documentation:', documentation_text)
                print('flowNodeRefs:', flowNodeRefs)
                print('incoming:', incoming_connections)

    def process_process(self, process, namespaces, nodes, edges, logging = False):

        process_id = process.attrib.get('id', '')
        process_name = process.attrib.get('name', '')
        process_type = process.tag.replace('{http://www.omg.org/spec/BPMN/20100524/MODEL}', '')

        documentation = process.find('.//bpmn:documentation', namespaces)
        documentation_text = ''
        if documentation is not None and documentation.text is not None:
            documentation_text = documentation.text.replace('\n', '').replace('\t', '').strip()

        # get all the ids of the lanes in the process
        outgoing_connections = [lane.attrib.get('id', '') for lane in process.findall('.//bpmn:lane', namespaces)]

        # Add the node to the graph
        nodes[process_id] = {
            'type': process_type,
            'name': process_name,
            'documentation': documentation_text,
            'incoming': [],
            'outgoing': outgoing_connections
        }

        for edge in outgoing_connections:
            edges[process_id] = (process_id, edge)

        if logging:
            # Print the process
            print("---------------------------------------------")
            print('id:', process_id)
            print('type:', process_type)
            print('name:', process_name)
            print('documentation:', documentation_text)
            print('outgoing:', outgoing_connections)

    def process_edges(self, process, namespaces, edges, tag, logging = False):
        """Process edges of a given tag within the BPMN file."""
        for edge in process.findall(f'.//bpmn:{tag}', namespaces):
            edge_id = edge.attrib['id']
            edge_type = edge.tag.replace('{http://www.omg.org/spec/BPMN/20100524/MODEL}', '')
            edge_name = edge.attrib.get('name', '')

            documentation = edge.find('.//bpmn:documentation', namespaces)
            documentation_text = ''
            if documentation is not None and documentation.text is not None:
                documentation_text = documentation.text.replace('\n', '').replace('\t', '').strip()

            source = edge.attrib.get('sourceRef', '')
            target = edge.attrib.get('targetRef', '')

            # Add the edge to the graph
            edges[edge_id] = (source, target)

            if logging:
                # Print the edge
                print("---------------------------------------------")
                print('id:', edge_id)
                print('type:', edge_type)
                print('name:', edge_name)
                print('documentation:', documentation_text)
                print('source:', source)
                print('target:', target)

    def add_super_node(self, processes, nodes):
        """Add a super node to the graph that connects to all processes. With name of the file as the name of the super node."""
        
        super_node_id = 'super_node'
        nodes[super_node_id] = {
            'type': 'superNode',
            'name': self.file_path,
            'documentation': '',
            'incoming': [],
            'outgoing': [process.attrib.get('id', '') for process in processes]
        }
        return super_node_id

    def add_edges_from_super_node(self, processes, super_node_id, edges):
        for process in processes:
            edge_id = f"{super_node_id}_{process.attrib.get('id', '')}"
            edges[edge_id] = (super_node_id, process.attrib.get('id', ''))

    def process_bpmn(self, logging=False):
        """
        Process a BPMN file and build a graph from it.

        Args:
            logging (bool, optional): If True, print log messages. Defaults to False.
        """    
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        processes = root.findall('.//bpmn:process', self.namespaces)

        tmp_nodes = {}
        tmp_edges = {}

        for process in processes:
            self.process_process(process, self.namespaces, tmp_nodes, tmp_edges, logging)
            self.process_elements(process, self.namespaces, tmp_nodes, logging)
            self.process_lanes(process, self.namespaces, tmp_nodes, tmp_edges, logging)
            self.process_edges(process, self.namespaces, tmp_edges, 'sequenceFlow', logging)

        # Add super node to the graph that connects to all processes
        super_node_id = self.add_super_node(processes, tmp_nodes)
        #super_node_id =  self.add_super_node(tmp_nodes, tmp_nodes)

        # Add edges from super node to all processes
        self.add_edges_from_super_node(processes, super_node_id, tmp_edges)
        #self.add_edges_from_super_node(tmp_nodes, super_node_id, tmp_edges)

        # Add the nodes
        for node_id, node_data in tmp_nodes.items():
            self.network.add_node(node_id, **node_data)

        # Add edges to the graph
        for edge_id, edge_data in tmp_edges.items():
            self.network.add_edge(*edge_data, id=edge_id)

        # Add self loops
        for node_id in self.network.nodes():
            self.network.add_edge(node_id, node_id)
    
    def process_txt(self, logging=False):
        """
        Read a text from a file.
    
        Args:
            logging (bool, optional): If True, print log messages. Defaults to False.
        """
        # Read the text from the file
        with open(self.file_path, 'r') as file:
            text = file.read()
    
        # Process the text
        self.process_text(text, logging=logging)
      
    def process_text(self, text, logging=False):
        """
        Process a text and build a graph from it.
    
        Args:
            text (str): The text to process.
            logging (bool, optional): If True, print log messages. Defaults to False.
        """
        # Split the text into chunks (sentences)
        chunks = TextChunker.chunk_by_max_words(text, 128)
    
        # Create a node for each chunk and connect each node to the next one
        for i, chunk in enumerate(chunks):
            self.network.add_node(i, text=chunk)
            if i > 0:
                self.network.add_edge(i - 1, i)
    
        # Add a supernode
        supernode_id = len(chunks)
        self.network.add_node(supernode_id, name=self.graph_uuid)
    
        # Connect the supernode to all other nodes and add self loops
        [self.network.add_edge(supernode_id, node_id) for node_id in range(supernode_id)]
        [self.network.add_edge(node_id, node_id) for node_id in self.network.nodes()]
    
    def draw_graph(self, fig_size=10):
        """Draws the BMPN graph."""
        # Create a larger figure
        plt.figure(figsize=(fig_size, fig_size))  # You can adjust the size as needed

        graph = self.network

        # Draw the graph
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, node_color='lightblue', node_size=fig_size)

        # Draw labels
        labels = {node: data.get('name', '') for node, data in graph.nodes(data=True)}

        texts = []
        for node, (x, y) in pos.items():
            texts.append(plt.text(x, y, labels[node], fontsize=fig_size, ha='center', va='center'))

        adjust_text(texts)

        plt.show()

    def get_all_paths(self, logging=False):
        """Get all paths in the graph. Returns a list of nodes of each path."""

        graph = self.network
        start_nodes = [node_id for node_id, data in graph.nodes(data=True) if data["type"] == "startEvent"]
        end_nodes = [node_id for node_id, data in graph.nodes(data=True) if data["type"] == "endEvent"]

        all_paths = []

        for start_node in start_nodes:
            for end_node in end_nodes:
                path_sequence = []
                for path in nx.all_simple_paths(graph, start_node, end_node):
                    for node in path:
                        path_sequence.append(node)

                all_paths.append(path_sequence)              

        return all_paths
    
    def to_dict(self):
        graph_info = {}
    
        if 'embedding' in self.get_graph() and isinstance(self.get_graph().get("embedding",[]), np.ndarray):
            graph_info = {
                'graph': {
                    'uuid': self.graph_uuid,
                    'file_path': self.file_path,
                    'data': self.data,
                    'embedding': np.array(self.get_graph().get("embedding",[])).tolist()
                },
                'nodes': []
            }
    
            for _, node in self.get_nodes():
                if 'embedding' in node and isinstance(node.get("embedding",[]), np.ndarray):
                    node_info = {
                        'name': node.get('name', ''),
                        'text': node.get('text', ''),
                        'embedding': np.array(node.get("embedding",[])).tolist()
                    }
                    graph_info['nodes'].append(node_info)
    
        return graph_info