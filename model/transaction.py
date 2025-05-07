from pymongo import MongoClient
import os
from dotenv import load_dotenv
import datetime
from bson.objectid import ObjectId

# Load environment variables
load_dotenv()

# Setup MongoDB connection
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client.bank_transactions
transactions_collection = db.transactions

class Transaction:
    """Model for bank transactions"""
    
    @staticmethod
    def insert_many(transactions):
        """
        Insert multiple transactions into MongoDB
        
        Args:
            transactions (list): List of transaction dictionaries
        
        Returns:
            list: List of inserted IDs
        """
        if not transactions:
            return []
            
        # Convert datetime objects to strings if needed
        for txn in transactions:
            if isinstance(txn.get('createdDate'), datetime.datetime):
                txn['createdDate'] = txn['createdDate'].strftime("%Y-%m-%dT%H:%M:%S")
        
        result = transactions_collection.insert_many(transactions)
        return result.inserted_ids
    
    @staticmethod
    def get_all(skip=0, query=None):
        """
        Get transactions from MongoDB with pagination
        
        Args:
            limit (int): Maximum number of transactions to return
            skip (int): Number of transactions to skip
            query (dict): MongoDB query for filtering
            
        Returns:
            tuple: (transactions, total_count)
        """
        if query is None:
            query = {}
            
        cursor = transactions_collection.find(
            query, 
            {'_id': 0}  # Exclude MongoDB _id field
        ).skip(skip)
        
        transactions = list(cursor)
        total_count = transactions_collection.count_documents(query)
        
        return transactions, total_count
    
    @staticmethod
    def build_query(from_account=None, to_account=None, min_amount=None, 
                   max_amount=None, transaction_type=None):
        """
        Build MongoDB query from parameters
        
        Args:
            from_account (str): Source account filter
            to_account (str): Destination account filter
            min_amount (float): Minimum transaction amount
            max_amount (float): Maximum transaction amount
            transaction_type (str): Transaction type filter
            
        Returns:
            dict: MongoDB query dictionary
        """
        query = {}
        
        if from_account:
            query['from'] = from_account
            
        if to_account:
            query['to'] = to_account
            
        if transaction_type:
            query['type'] = transaction_type
            
        # Handle amount range
        if min_amount is not None or max_amount is not None:
            amount_query = {}
            if min_amount is not None:
                amount_query['$gte'] = float(min_amount)
            if max_amount is not None:
                amount_query['$lte'] = float(max_amount)
            if amount_query:
                query['amt'] = amount_query
                
        return query
    
    @staticmethod
    def clear_all():
        """Delete all transactions from the collection"""
        transactions_collection.delete_many({})
        return transactions_collection.count_documents({})