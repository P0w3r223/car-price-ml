"""The published mini site: aggregates exported from the model, then rendered to HTML.

``export`` writes ``docs/data/*.json`` and ``build`` renders the page from them. The version
below is the contract between those two halves, kept here so the renderer can check it
without importing the exporter's model stack.
"""

# Bump when the shape of any exported aggregate changes. The page build refuses a file
# stamped with anything else rather than rendering whatever fields it recognises — the same
# reason `model.load_model` refuses an artifact fit on a different feature set.
AGGREGATE_SCHEMA = 1
