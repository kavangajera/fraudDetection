from pymongo import MongoClient
from neo4j import GraphDatabase
import uuid

def migrate_transactions_to_mongo(neo4j_uri, neo4j_user, neo4j_password, mongo_uri="mongodb+srv://devarshi:Deva123@cluster0.8e2qpsv.mongodb.net/Bank2"):
    # Connect to MongoDB
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client['bank_fraud_db']
    transactions_collection = db['transactions']

    # Connect to Neo4j
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # Cypher query to extract all transaction details
    cypher_query = """
    MATCH (a:Account)-[t:TRANSACTION]->(b:Account)
    RETURN
        t.id AS id,
        t.createdDate AS createdDate,
        t.amt AS amt,
        t.description AS description,
        t.currency AS currency,
        t.type AS type,
        a.account_id AS from_account,
        b.account_id AS to_account
    """

    with neo4j_driver.session() as session:
        results = session.run(cypher_query)
        for record in results:
            transaction = {
                "createdDate": record["createdDate"],
                "amt": record["amt"],
                "description": record["description"],
                "currency": record["currency"],
                "type": record["type"],
                "from_account": record["from_account"],
                "to_account": record["to_account"],
                "txn_id": str(uuid.uuid4())
            }
            transactions_collection.insert_one(transaction)

    # Close connections
    neo4j_driver.close()
    mongo_client.close()

if __name__ == "__main__":
    # Replace with your actual Neo4j credentials
    NEO4J_URI = "neo4j+s://ae03c8f0.databases.neo4j.io"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "Fa01ciGZHymObLA2cOv-UDQ96BCSr3Uq6Tlqur1Ye8E"  # Replace with your actual password

    migrate_transactions_to_mongo(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)