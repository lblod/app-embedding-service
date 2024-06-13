  # Run `docker-compose restart dispatcher` after updating
  # this file.

defmodule Dispatcher do
  use Matcher
  define_accept_types [
    html: ["text/html", "application/xhtml+html"],
    json: ["application/json", "application/vnd.api+json"],
    upload: ["multipart/form-data"],
    sparql_json: ["application/sparql-results+json"],
    any: [ "*/*" ],
  ]

  define_layers [ :api_services, :api, :frontend, :not_found ]



  match "/bpmn-nodes",  %{ accept: [:json], layer: :api } do
    Proxy.forward conn, [], "http://resource/bpmn-nodes/"
  end

  match "/bpmn-graphs", %{ accept: [:json], layer: :api } do
    Proxy.forward conn, [], "http://resource/bpmn-graphs/"
  end



  ###############
  # SPARQL
  ###############

  match "/sparql", %{ layer: :sparql, accept: %{ sparql: true } } do
    forward conn, [], "http://database:8890/sparql"
  end


  ###############################################################
  # SEARCH
  ###############################################################

  match "/search/*path", %{  accept: %{ json: true }, layer: :api} do
    Proxy.forward conn, path, "http://search/"
  end

  ###############################################################
  # service
  ###############################################################


  match "/*path", %{accept: %{ json: true }, layer: :api } do
    Proxy.forward conn, path, "http://poc-embedder/"
  end

  ###############################################################
  # errors
  ###############################################################

  match "/*_path", %{ accept: [:any], layer: :not_found} do
    send_resp(conn, 404, "{\"error\": {\"code\": 404}}")
  end
end
