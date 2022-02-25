import rdflib

g = rdflib.Graph(store='Neo4j')
theconfig = {'uri': "neo4j://localhost:7687",
             'database': 'got',
             'auth': {'user': "neo4j", 'pwd': "got"}}

g.open(theconfig, create = False)
g.load("my_graph.nt", format="nt")