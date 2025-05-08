from flask import Blueprint, jsonify, request
import json
import uuid
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId

# Import the BankFraudGraphGenerator class
from network_creation import BankFraudGraphGenerator, FraudDetectionQueries

# Create the blueprint
bank_bp = Blueprint('bank', __name__)

# MongoDB connection
client = MongoClient('mongodb+srv://devarshi:Deva123@cluster0.8e2qpsv.mongodb.net/Bank2')
db = client['bank_fraud_db']
transactions_collection = db['transactions']
fraud_graphs_collection = db['fraud_graphs']

# Neo4j connection parameters
NEO4J_URI = "neo4j+s://ae03c8f0.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Fa01ciGZHymObLA2cOv-UDQ96BCSr3Uq6Tlqur1Ye8E"  # Replace with your actual password


@bank_bp.route('/create-graph', methods=['POST'])
def create_graph():
    """
    Creates a bank fraud graph with specified parameters
    Returns the number of accounts, transactions, and fraud accounts in JSON format
    Stores the entire graph data in MongoDB
    """
    try:
        # Get parameters from request or use defaults
        data = request.get_json() or {}
        num_accounts = data.get('num_accounts', 100)
        num_transactions = data.get('num_transactions', 500)
        num_fraud_accounts = data.get('num_fraud_accounts', 5)
        
        # Create graph generator
        generator = BankFraudGraphGenerator(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        
        # Generate graph data
        graph_data = generator.generate_graph(
            num_accounts=num_accounts,
            num_transactions=num_transactions,
            num_fraud_accounts=num_fraud_accounts
        )
        
        # Store transactions in MongoDB with txn_id
        for transaction in graph_data['transactions']:
            # Add a unique transaction ID
            transaction['txn_id'] = str(uuid.uuid4())
            # Store each transaction separately
            transactions_collection.insert_one(transaction)
        
        # Store the entire graph data for reference
        graph_id = fraud_graphs_collection.insert_one({
            'created_at': datetime.now(),
            'parameters': {
                'num_accounts': num_accounts,
                'num_transactions': num_transactions,
                'num_fraud_accounts': num_fraud_accounts
            },
            'stats': graph_data['stats'],
            'accounts': graph_data['accounts']
        }).inserted_id
        
        # Return the parameters and stats
        return jsonify({
            'num_accounts': num_accounts,
            'num_transactions': num_transactions,
            'num_fraud_accounts': num_fraud_accounts,
            'graph_id': str(graph_id),
            'stats': graph_data['stats']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bank_bp.route('/get-all-transactions', methods=['GET'])
def get_all_transactions():
    """
    Returns all transactions stored in MongoDB
    """
    try:
        # Retrieve all transactions
        cursor = transactions_collection.find({})
        
        # Format transactions for response
        transactions_list = []
        for transaction in cursor:
            # Convert ObjectId to string
            if '_id' in transaction:
                transaction['_id'] = str(transaction['_id'])
            
            # Include the transaction in the response
            transactions_list.append({
                'txn_id': transaction['txn_id'],
                'from': transaction['from'],
                'to': transaction['to'],
                'amt': transaction['amt'],
                'type': transaction['type'],
                'currency': transaction['currency'],
                'description': transaction['description'],
                'createdDate': transaction['createdDate']
            })
        
        return jsonify({'transactions': transactions_list})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bank_bp.route('/get-transaction/<txn_id>', methods=['GET'])
def get_transaction(txn_id):
    """
    Returns a specific transaction by txn_id
    """
    try:
        # Find the transaction by txn_id
        transaction = transactions_collection.find_one({'txn_id': txn_id})
        
        if not transaction:
            return jsonify({'error': 'Transaction not found'}), 404
        
        # Convert ObjectId to string
        if '_id' in transaction:
            transaction['_id'] = str(transaction['_id'])
        
        return jsonify(transaction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bank_bp.route('/get-graph/<graph_id>', methods=['GET'])
def get_graph(graph_id):
    """
    Returns a specific fraud graph by graph_id
    """
    try:
        # Find the graph by graph_id
        graph = fraud_graphs_collection.find_one({'_id': ObjectId(graph_id)})
        
        if not graph:
            return jsonify({'error': 'Graph not found'}), 404
        
        # Convert ObjectId to string
        if '_id' in graph:
            graph['_id'] = str(graph['_id'])
        
        return jsonify(graph)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    



import csv
import io
from flask import Response
from neo4j import GraphDatabase

# Neo4j driver initialization (if not already initialized elsewhere)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

import os

# Ensure this directory exists
DATA_DIR = os.path.join(os.getcwd(), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

@bank_bp.route('/filtered-accounts', methods=['POST'])
def filtered_accounts():
    """
    Receives a list of account objects and saves a CSV of matched accounts from Neo4j,
    including the is_suspicious flag from the input, to data/acc_list_with_params.csv
    """
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({"error": "Invalid data format. Expected a list of objects."}), 400

        # Map accountNumber to is_suspicious
        account_map = {
            item["accountNumber"]: item.get("is_suspicious", False)
            for item in data if "accountNumber" in item
        }
        account_numbers = list(account_map.keys())

        if not account_numbers:
            return jsonify({"error": "No valid accountNumber fields found."}), 400

        query = """
        MATCH (a:Account)
        WHERE a.accountNumber IN $account_numbers
        RETURN a
        """

        records = []
        with driver.session() as session:
            result = session.run(query, account_numbers=account_numbers)
            for record in result:
                node = record["a"]
                node_data = dict(node)
                acc_number = node_data.get("accountNumber")
                node_data["is_suspicious"] = account_map.get(acc_number, False)
                records.append(node_data)

        if not records:
            return jsonify({"message": "No matching accounts found in Neo4j."}), 404

        # Write to CSV
        fieldnames = sorted(set().union(*(row.keys() for row in records)))
        csv_path = os.path.join(DATA_DIR, 'acc_list_with_params.csv')

        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        return jsonify({
            "message": "CSV saved successfully.",
            "path": f"data/acc_list_with_params.csv",
            "record_count": len(records)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
