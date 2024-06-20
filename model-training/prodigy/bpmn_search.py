import prodigy
from prodigy.components.loaders import JSONL

@prodigy.recipe('bpmn-search',
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    labels=("Comma-separated label string", "option", "l", str),
)
def bpmn_recipe(dataset, source, labels):
    stream = JSONL(source)  # Load the data from the source file
    labels = labels.split(',')  # Split the labels string into a list

    return {
        'dataset': dataset,  # The dataset to save annotations to
        'stream': stream,  # The stream of examples
        'view_id': 'html',  # The annotation interface to use
        'config': {
            'labels': labels,  # The labels for the tasks
            'html_template': """
            <div id="wrapper" class="prodigy-title-wrapper c01152">
                <div id="title" class="prodigy-title c01150 c01147 c01151"">Is the proposed relevancy between the query and document correct?</div>
            </div>
            <div id="summary" class="{{meta.predicted_label.related}}">
                <p><strong>Search query:</strong> {{query}}</p>
                <p><strong>Document:</strong> {{meta.file}}</p>
                <p><strong>Motivation for assigned label:</strong> {{meta.predicted_label.motivation}}</p>
            </div>
            <div id="document">
                <p><strong>Process description:</strong> {{content}}</p>
            </div>""",
            'global_css': """
                body {
                    font-family: Arial, sans-serif;
                    font-size: 20px;
                }
                #title {
                    font-size: 1.5em;
                    font-weight: bold;
                    margin-bottom: 1em;
                }
                #document, #summary {
                    color: #2C3E50;
                    padding: 1em;
                    margin-bottom: 1em;
                    margin-top: 1em;
                }
                #document {
                    background-color: #ECF0F1;
                    white-space: pre-wrap;
                }
                ._Content-root-0-1-166 {
                    white-space: normal;
                }
                .true {
                    background-color: #2ecc3136;
                }
                .false {
                    background-color: #ff57224f;
                }
            """,  # The CSS rules
        }
    }