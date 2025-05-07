from neo4j import GraphDatabase

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
# URI = "neo4j+s://ae03c8f0.databases.neo4j.io"
# AUTH = ("neo4j", "Fa01ciGZHymObLA2cOv-UDQ96BCSr3Uq6Tlqur1Ye8E")

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "Kavan#7377")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    print("Connected to Neo4j database.")

    