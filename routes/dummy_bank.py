from flask import Blueprint, request, jsonify
import os
from dotenv import load_dotenv
import sys
import datetime


from model.graph import Neo4jFraudDataGenerator
from model.transaction import Transaction

# Load environment variables
load_dotenv()

# Create blueprint
bank_bp = Blueprint('bank', __name__)

@bank_bp.route('/create-graph', methods=['POST'])
def create_graph():
    """
    Create a graph of bank accounts and transactions with fraud patterns
    
    Expected JSON input:
    {
        "account_count": 10000,
        "min_transactions": 60000,
        "fraud_patterns": {
            "cycle": 100,
            "star": 150,
            "chain": 80,
            "layered": 70
        },
        "clear_db": true
    }
    """
    try:
        data = request.get_json()
        
        # Extract parameters from request
        account_count = data.get('account_count', 10000)
        min_transactions = data.get('min_transactions', account_count * 6)
        fraud_patterns_config = data.get('fraud_patterns', {
            'cycle': account_count // 40,
            'star': account_count // 40,
            'chain': account_count // 40,
            'layered': account_count // 40
        })
        clear_db = data.get('clear_db', False)
        
        # Validate inputs
        if not isinstance(account_count, int) or account_count <= 0:
            return jsonify({"error": "account_count must be a positive integer"}), 400
            
        if not isinstance(min_transactions, int) or min_transactions <= 0:
            return jsonify({"error": "min_transactions must be a positive integer"}), 400
            
        total_fraud = sum(fraud_patterns_config.values())
        if total_fraud > account_count * 0.5:
            return jsonify({
                "error": f"Total fraud accounts ({total_fraud}) exceeds 50% of total accounts ({account_count})",
                "hint": "Reduce fraud pattern counts or increase account_count"
            }), 400
        
        # Get Neo4j connection details
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not neo4j_uri or not neo4j_username or not neo4j_password:
            return jsonify({"error": "Neo4j connection details not found in environment variables"}), 500
        
        # Create generator
        generator = Neo4jFraudDataGenerator(neo4j_uri, neo4j_username, neo4j_password)
        
        # Clear database if requested
        if clear_db:
            generator.clear_database()
        
        # Step 1: Create network structure with fraud patterns
        G, fraud_accounts, community_map = generator.create_network_structure(
            account_count, fraud_patterns_config
        )
        
        # Step 2: Calculate network metrics
        account_metrics = generator.analyze_network(G, fraud_accounts, community_map)
        
        # Step 3: Generate transactions based on network structure
        transactions = generator.generate_transactions_from_graph(G, account_metrics, min_transactions)
        
        # Step 4: Save accounts to Neo4j
        generator.save_accounts_to_neo4j(G, account_metrics)
        
        # Step 5: Save transactions to Neo4j
        generator.save_transactions_to_neo4j(transactions)
        
        # Step 6: Save transactions to MongoDB
        # Use the Transaction model to insert transactions
        Transaction.insert_many(transactions)
        
        # Generate response
        stats = {
            "total_accounts": account_count,
            "suspicious_accounts": len(fraud_accounts),
            "transactions": len(transactions),
            "fraud_patterns": {
                pattern: count for pattern, count in fraud_patterns_config.items() if count > 0
            }
        }
        
        # Close the connection
        generator.close()
        
        return jsonify({
            "message": "Graph data generated successfully",
            "stats": stats
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bank_bp.route('/clear-transactions', methods=['DELETE'])
def clear_transactions():
    """Clear all transactions from MongoDB"""
    try:
        count = Transaction.clear_all()
        return jsonify({
            "message": "All transactions cleared successfully",
            "previous_count": count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bank_bp.route('/get-all-transactions', methods=['GET'])
def get_all_transactions():
    """
    Get all transactions from MongoDB
    
    Optional query parameters:
    - limit: Max number of transactions to return (default: 1000)
    - skip: Number of transactions to skip (for pagination)
    - from_account: Filter by source account
    - to_account: Filter by destination account
    - min_amount: Filter by minimum amount
    - max_amount: Filter by maximum amount
    - transaction_type: Filter by transaction type
    """
    try:
        # Get query parameters
        # limit = request.args.get('limit', 1000, type=int)
        skip = request.args.get('skip', 0, type=int)
        from_account = request.args.get('from_account')
        to_account = request.args.get('to_account')
        min_amount = request.args.get('min_amount', type=float)
        max_amount = request.args.get('max_amount', type=float)
        transaction_type = request.args.get('transaction_type')
        
        # Build query using the Transaction model
        query = Transaction.build_query(
            from_account, to_account, min_amount, max_amount, transaction_type
        )
        
        # Get transactions with pagination
        transactions, total_count = Transaction.get_all(skip, query)
        
        return jsonify({
            "transactions": transactions,
            "metadata": {
                "total": total_count,
                # "limit": limit,
                "skip": skip,
                "count": len(transactions)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500