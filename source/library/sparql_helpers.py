from SPARQLWrapper import SPARQLWrapper, JSON
import os

MU_APPLICATION_GRAPH = os.environ.get('MU_APPLICATION_GRAPH')

sparqlQuery = SPARQLWrapper(os.environ.get('MU_SPARQL_ENDPOINT'), returnFormat=JSON)
sparqlUpdate = SPARQLWrapper(os.environ.get('MU_SPARQL_UPDATEPOINT'), returnFormat=JSON)
sparqlUpdate.method = 'POST'


if os.environ.get('MU_SPARQL_TIMEOUT'):
    timeout = int(os.environ.get('MU_SPARQL_TIMEOUT'))
    sparqlQuery.setTimeout(timeout)
    sparqlUpdate.setTimeout(timeout)

def query(the_query):
    """Execute the given SPARQL query (select/ask/construct) on the triplestore and returns the results in the given return Format (JSON by default)."""
    sparqlQuery.setQuery(the_query)
    return sparqlQuery.query().convert()

def update(the_query):
    """Execute the given update SPARQL query on the triplestore. If the given query is not an update query, nothing happens."""
    sparqlUpdate.setQuery(the_query)
    if sparqlUpdate.isSparqlUpdateRequest():
        sparqlUpdate.query()