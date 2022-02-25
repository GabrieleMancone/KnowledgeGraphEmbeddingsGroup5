All’interno della cartella Knowledge Graph Embedding si ha:

  •  ModelSelectionModelName.py is the file to train the specific ModelName on GoT.
  •  ModelModelNameHarryPotter.py s the file to train the specific ModelName on HarryPotter.
  •  PredictingLinks.py is the file that contains the function for predicting links.
  •  EvalModelName.py is the file to evaluate the specific ModelName on GoT.
  •  EvalModelNameHarryPotter.py is the file to evaluate the specific ModelName on Harry Potter.
  •  Convert.py is the file to convert csv file in nt file to display the graph.
  •  GraphLinking.py is the file to connect to DBMS of Neo4j to display the graph

  •  GoT.csv are triples used for GoT.
  •  Triple.csv are triples used for Harry Potter.
  •  tesh_sheet.csv are triple used to build the graph.
  •  my_graph.nt is the converted file of tesh_sheet.csv to display graph on Neo4j.

  •  saved_model is the folder that contains best model e prediction links for every configuration analyzed.
