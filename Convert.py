import csv
from rdflib import Graph, URIRef

input_file = csv.DictReader(open("test_sheet.csv"))

output_graph = Graph()

for row in input_file:

	row = dict(row)

	output_graph.add((URIRef(row['Subject URI']), URIRef(row['Predicate URI']), URIRef(row['Object URI'])) )

output_graph.serialize(destination='my_graph.nt', format='nt')