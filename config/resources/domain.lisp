(in-package :mu-cl-resources)

;;;;
;; NOTE
;; docker-compose stop; docker-compose rm; docker-compose up
;; after altering this file.

;; Describe your resources here

;; The general structure could be described like this:
;;
;; (define-resource <name-used-in-this-file> ()
;;   :class <class-of-resource-in-triplestore>
;;   :properties `((<json-property-name-one> <type-one> ,<triplestore-relation-one>)
;;                 (<json-property-name-two> <type-two> ,<triplestore-relation-two>>))
;;   :has-many `((<name-of-an-object> :via ,<triplestore-relation-to-objects>
;;                                    :as "<json-relation-property>")
;;               (<name-of-an-object> :via ,<triplestore-relation-from-objects>
;;                                    :inverse t ; follow relation in other direction
;;                                    :as "<json-relation-property>"))
;;   :has-one `((<name-of-an-object :via ,<triplestore-relation-to-object>
;;                                  :as "<json-relation-property>")
;;              (<name-of-an-object :via ,<triplestore-relation-from-object>
;;                                  :as "<json-relation-property>"))
;;   :resource-base (s-url "<string-to-which-uuid-will-be-appended-for-uri-of-new-items-in-triplestore>")
;;   :on-path "<url-path-on-which-this-resource-is-available>")


;; An example setup with a catalog, dataset:





  (define-resource search-graph ()
    :class (s-prefix "search:Graph")
    :properties `((:file :string ,(s-prefix "search:filePath"))
                  (:data :string ,(s-prefix "search:data"))
                  (:embedding :string ,(s-prefix "search:embedding")))
    :has-many `((search-node :via ,(s-prefix "search:hasNode")
                      :as "nodes"))
    :resource-base (s-url "http://my-application.com/search-graphs/")
    :on-path "search-graphs")

  (define-resource search-node ()
  :class (s-prefix "search:Node")
  :properties `((:name :string ,(s-prefix "search:name"))
                (:embedding :string ,(s-prefix "search:embedding")))
  :has-one `((search-graph :via ,(s-prefix "search:hasNode")
                    :inverse t
                    :as "graph"))
  :resource-base (s-url "http://my-application.com/search-nodes/")
  :on-path "search-nodes")
