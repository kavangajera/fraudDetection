import os
import random
import datetime
import networkx as nx
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
from collections import defaultdict

class Neo4jFraudDataGenerator:
    def __init__(self, uri, username, password):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
        
    def clear_database(self):
        """Clear all nodes and relationships from the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared successfully")
    
    def create_network_structure(self, account_count=10000, fraud_patterns_config=None):
        """
        Create a NetworkX graph with controlled structures for later analysis
        
        Parameters:
        - account_count: Total number of accounts to generate
        - fraud_patterns_config: Dictionary with keys as pattern types and values as account counts
          Example: {'cycle': 100, 'star': 150, 'chain': 80, 'layered': 70}
        
        Returns a tuple of (graph, fraud_accounts, community_map)
        """
        print("Creating network structure...")
        
        # Initialize directed graph
        G = nx.DiGraph()
        
        # If no fraud pattern config provided, use default distribution
        if fraud_patterns_config is None:
            fraud_account_count = 300
            fraud_patterns_config = {
                'cycle': fraud_account_count // 4,
                'star': fraud_account_count // 4,
                'chain': fraud_account_count // 4,
                'layered': fraud_account_count // 4
            }
        
        # Calculate total fraud accounts
        total_fraud_accounts = sum(fraud_patterns_config.values())
        
        # Add all accounts as nodes
        for i in range(account_count):
            G.add_node(f"ACC{i:05d}", 
                       type="SAVINGS" if i % 2 == 0 else "CURRENT",
                       balance=round(random.uniform(1000, 100000), 2),
                       user=f"User_{i}",
                       freq=0,
                       regularIntervalTransaction=0,
                       suspicious=False)
        
        # Create normal communities (80% of accounts in communities)
        normal_community_count = 20
        accounts_per_community = int(0.8 * (account_count - total_fraud_accounts) / normal_community_count)
        normal_accounts = [f"ACC{i:05d}" for i in range(account_count - total_fraud_accounts)]
        random.shuffle(normal_accounts)
        
        community_map = {}
        for community_id in range(normal_community_count):
            start_idx = community_id * accounts_per_community
            end_idx = start_idx + accounts_per_community
            community_accounts = normal_accounts[start_idx:end_idx]
            
            # Assign community ID to each account
            for account in community_accounts:
                community_map[account] = community_id
            
            # Create dense connections within community
            for _ in range(int(1.5 * len(community_accounts))):
                from_acc = random.choice(community_accounts)
                to_acc = random.choice(community_accounts)
                if from_acc != to_acc and not G.has_edge(from_acc, to_acc):
                    G.add_edge(from_acc, to_acc)
        
        # Add some inter-community connections
        for _ in range(int(0.2 * account_count)):
            comm1, comm2 = random.sample(range(normal_community_count), 2)
            from_accounts = [acc for acc, comm in community_map.items() if comm == comm1]
            to_accounts = [acc for acc, comm in community_map.items() if comm == comm2]
            
            if from_accounts and to_accounts:
                from_acc = random.choice(from_accounts)
                to_acc = random.choice(to_accounts)
                G.add_edge(from_acc, to_acc)
        
        # Create fraud patterns
        fraud_accounts = []
        
        # Map of pattern types to their creation functions
        pattern_functions = {
            'cycle': self._create_cycle_pattern,
            'star': self._create_star_pattern,
            'chain': self._create_chain_pattern,
            'layered': self._create_layered_pattern
        }
        
        # Current index for fraud accounts
        start_idx = account_count - total_fraud_accounts
        
        # Create each pattern with the specified number of accounts
        for pattern_type, count in fraud_patterns_config.items():
            if pattern_type in pattern_functions and count > 0:
                end_idx = start_idx + count
                pattern_accounts = [f"ACC{i:05d}" for i in range(start_idx, end_idx)]
                
                # Generate the pattern and collect fraud accounts
                pattern_fraud_accounts = pattern_functions[pattern_type](G, pattern_accounts, community_map)
                fraud_accounts.extend(pattern_fraud_accounts)
                
                start_idx = end_idx
                
                print(f"Created {pattern_type} pattern with {len(pattern_fraud_accounts)} accounts")
            
        # Mark accounts as suspicious
        for account in fraud_accounts:
            G.nodes[account]['suspicious'] = True
            
        print(f"Network structure created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        print(f"Created {len(fraud_accounts)} suspicious accounts")
        
        return G, fraud_accounts, community_map
    
    def _create_cycle_pattern(self, G, accounts, community_map):
        """Create a fraud pattern: cycles where money goes in loops"""
        if len(accounts) < 5:
            return accounts
            
        fraud_groups = np.array_split(accounts, max(1, len(accounts) // 6))
        fraud_accounts = []
        
        for group in fraud_groups:
            group = group.tolist()
            fraud_accounts.extend(group)
            
            # Assign to a new fraud community
            community_id = max(community_map.values(), default=-1) + 1
            for account in group:
                community_map[account] = community_id
            
            # Create a cycle
            for i in range(len(group)):
                G.add_edge(group[i], group[(i+1) % len(group)])
                
            # Add some connections to normal accounts for camouflage
            for _ in range(2):
                normal_accounts = [acc for acc in G.nodes() if acc not in fraud_accounts]
                if normal_accounts:
                    normal_acc = random.choice(normal_accounts)
                    cycle_acc = random.choice(group)
                    # Connection in both directions
                    G.add_edge(normal_acc, cycle_acc)
                    G.add_edge(cycle_acc, normal_acc)
                
        return fraud_accounts
    
    def _create_star_pattern(self, G, accounts, community_map):
        """Create a fraud pattern: central hub with multiple spokes"""
        if len(accounts) < 4:
            return accounts
            
        fraud_groups = np.array_split(accounts, max(1, len(accounts) // 8))
        fraud_accounts = []
        
        for group in fraud_groups:
            group = group.tolist()
            fraud_accounts.extend(group)
            
            # Assign to a new fraud community
            community_id = max(community_map.values(), default=-1) + 1
            for account in group:
                community_map[account] = community_id
            
            # Select a central hub
            hub = group[0]
            spokes = group[1:]
            
            # Hub receives from all spokes
            for spoke in spokes:
                G.add_edge(spoke, hub)
                
            # Hub sends to some spokes (money laundering pattern)
            for spoke in random.sample(spokes, k=min(3, len(spokes))):
                G.add_edge(hub, spoke)
                
            # Add connections to normal accounts
            normal_accounts = [acc for acc in G.nodes() if acc not in fraud_accounts]
            if normal_accounts:
                for _ in range(min(3, len(normal_accounts))):
                    normal_acc = random.choice(normal_accounts)
                    G.add_edge(normal_acc, hub)
                
        return fraud_accounts
    
    def _create_chain_pattern(self, G, accounts, community_map):
        """Create a fraud pattern: linear chains with high betweenness"""
        if len(accounts) < 5:
            return accounts
            
        fraud_groups = np.array_split(accounts, max(1, len(accounts) // 5))
        fraud_accounts = []
        
        for group in fraud_groups:
            group = group.tolist()
            fraud_accounts.extend(group)
            
            # Assign to a new fraud community
            community_id = max(community_map.values(), default=-1) + 1
            for account in group:
                community_map[account] = community_id
            
            # Create a chain
            for i in range(len(group) - 1):
                G.add_edge(group[i], group[i+1])
                
            # Add entry points (normal accounts -> first account in chain)
            normal_accounts = [acc for acc in G.nodes() if acc not in fraud_accounts]
            if normal_accounts:
                for _ in range(2):
                    normal_acc = random.choice(normal_accounts)
                    G.add_edge(normal_acc, group[0])
                    
            # Add exit points (last account -> normal accounts)
            if normal_accounts:
                for _ in range(2):
                    normal_acc = random.choice(normal_accounts)
                    G.add_edge(group[-1], normal_acc)
                
        return fraud_accounts
    
    def _create_layered_pattern(self, G, accounts, community_map):
        """Create a fraud pattern: layered structure with intermediate accounts"""
        if len(accounts) < 6:
            return accounts
            
        fraud_groups = np.array_split(accounts, max(1, len(accounts) // 10))
        fraud_accounts = []
        
        for group in fraud_groups:
            group = group.tolist()
            fraud_accounts.extend(group)
            
            # Assign to a new fraud community
            community_id = max(community_map.values(), default=-1) + 1
            for account in group:
                community_map[account] = community_id
            
            # Divide into layers
            n_layers = min(3, len(group) // 2)
            layers = np.array_split(group, n_layers)
            
            # Connect each layer to the next
            for layer_idx in range(len(layers) - 1):
                for src in layers[layer_idx]:
                    for dst in layers[layer_idx + 1]:
                        G.add_edge(src, dst)
            
            # Add entry points
            normal_accounts = [acc for acc in G.nodes() if acc not in fraud_accounts]
            if normal_accounts and len(layers) > 0:
                for _ in range(3):
                    normal_acc = random.choice(normal_accounts)
                    first_layer_acc = random.choice(layers[0])
                    G.add_edge(normal_acc, first_layer_acc)
            
            # Add exit points
            if normal_accounts and len(layers) > 1:
                for _ in range(3):
                    normal_acc = random.choice(normal_accounts)
                    last_layer_acc = random.choice(layers[-1])
                    G.add_edge(last_layer_acc, normal_acc)
                
        return fraud_accounts
    
    def analyze_network(self, G, fraud_accounts, community_map):
        """Calculate network metrics for all accounts"""
        print("Analyzing network metrics...")
        
        # Calculate degree metrics
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        # Calculate PageRank
        pagerank = nx.pagerank(G, alpha=0.85)
        
        # Calculate betweenness (on a sample for large graphs)
        if len(G) > 5000:
            # Use approximate betweenness for large graphs
            betweenness = nx.betweenness_centrality(G, k=100)
        else:
            betweenness = nx.betweenness_centrality(G)
        
        # Calculate triangle counts (for undirected version of the graph)
        UG = G.to_undirected()
        triangles = nx.triangles(UG)
        
        # Identify cycles (limited to cycles up to length 4 for performance)
        cycle_counts = defaultdict(int)
        for node in G.nodes():
            for cycle in nx.simple_cycles(G.subgraph(list(G.neighbors(node)) + [node])):
                if 3 <= len(cycle) <= 4:  # Count only 3-cycles and 4-cycles
                    for account in cycle:
                        cycle_counts[account] += 1
        
        # Calculate community sizes
        community_sizes = defaultdict(int)
        for account, community in community_map.items():
            community_sizes[community] += 1
        
        # Calculate intermediate accounts (accounts that serve as bridges)
        intermediate_accounts = {}
        for account in G.nodes():
            # An account is intermediate if it has both incoming and outgoing edges
            intermediate_accounts[account] = 1 if in_degree.get(account, 0) > 0 and out_degree.get(account, 0) > 0 else 0
        
        # Calculate fraud scores (simplified version)
        fraud_scores = {}
        for account in G.nodes():
            # Base score starts at 0.1
            score = 0.1
            
            # Increase score for high betweenness
            if betweenness.get(account, 0) > np.percentile(list(betweenness.values()), 80):
                score += 0.2
                
            # Increase score for being part of many cycles
            if cycle_counts.get(account, 0) >= 2:
                score += 0.15
                
            # Increase score for intermediate accounts
            if intermediate_accounts.get(account, 0) == 1:
                score += 0.1
                
            # Increase score for unusual degree patterns
            total_degree = in_degree.get(account, 0) + out_degree.get(account, 0)
            if total_degree > 0:
                ratio = in_degree.get(account, 0) / total_degree
                if ratio > 0.8 or ratio < 0.2:  # Highly imbalanced in/out ratio
                    score += 0.15
            
            # Adjust score for known fraud accounts
            if account in fraud_accounts:
                score = min(0.7 + score * 0.3, 0.99)  # Ensure high but varied scores
            else:
                score = min(score, 0.6)  # Cap normal account scores
                
            fraud_scores[account] = round(score, 2)
        
        # Compile all metrics
        account_metrics = {}
        for account in G.nodes():
            community = community_map.get(account, -1)
            account_metrics[account] = {
                "accountNumber": account,
                "pagerank": round(pagerank.get(account, 0), 6),
                "in_degree": in_degree.get(account, 0),
                "out_degree": out_degree.get(account, 0),
                "betweenness": round(betweenness.get(account, 0), 6),
                "community_id": community,
                "community_size": community_sizes.get(community, 1),
                "triangle_count": triangles.get(account, 0),
                "cycle_count": cycle_counts.get(account, 0),
                "intermediate_accounts": intermediate_accounts.get(account, 0),
                "is_fraud": 1 if account in fraud_accounts else 0,
                "fraud_score": fraud_scores.get(account, 0)
            }
            
        print("Network analysis completed")
        return account_metrics
        
    def generate_transactions_from_graph(self, G, account_metrics, min_transactions=60000):
        """Generate transactions based on the graph structure"""
        print(f"Generating at least {min_transactions} transactions from graph structure...")
        
        transactions = []
        
        # First, generate transactions for all edges in the graph
        for from_acc, to_acc in G.edges():
            # Determine if this is a suspicious transaction
            is_suspicious = (account_metrics[from_acc]["is_fraud"] == 1 or 
                            account_metrics[to_acc]["is_fraud"] == 1)
            
            # Generate 1-3 transactions for each edge
            num_txns = random.randint(1, 3)
            for _ in range(num_txns):
                # Generate transaction amount (higher for suspicious transactions)
                if is_suspicious:
                    amount = round(random.uniform(500, 10000), 2)
                else:
                    amount = round(random.uniform(100, 3000), 2)
                
                # Generate transaction date (suspicious transactions more recent)
                if is_suspicious:
                    days_ago = random.randint(0, 90)  # Last 3 months
                else:
                    days_ago = random.randint(0, 364)  # Last year
                    
                txn_date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
                
                # Create transaction
                transaction = {
                    "from": from_acc,
                    "to": to_acc,
                    "amt": amount,
                    "type": random.choice(["TRANSFER", "PAYMENT"]),
                    "currency": "INR",
                    "description": f"Txn_{len(transactions)}",
                    "createdDate": txn_date.strftime("%Y-%m-%dT%H:%M:%S")
                }
                transactions.append(transaction)
        
        # If we need more transactions, add random ones
        while len(transactions) < min_transactions:
            # Get random accounts
            all_accounts = list(G.nodes())
            from_acc = random.choice(all_accounts)
            to_acc = random.choice(all_accounts)
            while from_acc == to_acc:
                to_acc = random.choice(all_accounts)
            
            # Generate transaction
            amount = round(random.uniform(100, 3000), 2)
            days_ago = random.randint(0, 364)
            txn_date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
            
            transaction = {
                "from": from_acc,
                "to": to_acc,
                "amt": amount,
                "type": random.choice(["TRANSFER", "PAYMENT"]),
                "currency": "INR",
                "description": f"Txn_{len(transactions)}",
                "createdDate": txn_date.strftime("%Y-%m-%dT%H:%M:%S")
            }
            transactions.append(transaction)
        
        print(f"Generated {len(transactions)} transactions")
        return transactions
    
    def save_accounts_to_neo4j(self, G, account_metrics):
        """Save account nodes to Neo4j"""
        print("Saving accounts to Neo4j...")
        
        # Create constraint for account number uniqueness
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT account_number_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.accountNumber IS UNIQUE")
                print("Constraint created successfully")
            except Exception as e:
                print(f"Constraint may already exist: {e}")
        
        # Process accounts in batches
        accounts = []
        account_count = len(G.nodes())
        batch_size = 1000
        
        for i, account in enumerate(G.nodes()):
            node_data = G.nodes[account]
            metrics = account_metrics[account]
            
            # Combine node data with metrics
            account_data = {
                "accountNumber": account,
                "type": node_data["type"],
                "balance": node_data["balance"],
                "user": node_data["user"],
                "freq": node_data["freq"],
                "regularIntervalTransaction": node_data["regularIntervalTransaction"],
                "suspicious": node_data["suspicious"],
                "pagerank": metrics["pagerank"],
                "in_degree": metrics["in_degree"],
                "out_degree": metrics["out_degree"],
                "betweenness": metrics["betweenness"],
                "community_id": metrics["community_id"],
                "community_size": metrics["community_size"],
                "triangle_count": metrics["triangle_count"],
                "cycle_count": metrics["cycle_count"],
                "intermediate_accounts": metrics["intermediate_accounts"],
                "is_fraud": metrics["is_fraud"],
                "fraud_score": metrics["fraud_score"]
            }
            accounts.append(account_data)
            
            # Save in batches
            if (i + 1) % batch_size == 0 or i == account_count - 1:
                with self.driver.session() as session:
                    session.run("""
                    UNWIND $accounts AS account
                    CREATE (a:Account)
                    SET a = account
                    """, accounts=accounts)
                print(f"Created {len(accounts)} accounts (total: {i+1})")
                accounts = []
    
    def save_transactions_to_neo4j(self, transactions):
        """Save transaction relationships to Neo4j"""
        print("Saving transactions to Neo4j...")
        
        # Process transactions in batches
        batch_size = 1000
        for batch in range(0, len(transactions), batch_size):
            current_batch = transactions[batch:batch + batch_size]
            
            with self.driver.session() as session:
                session.run("""
                UNWIND $transactions AS txn
                MATCH (from:Account {accountNumber: txn.from})
                MATCH (to:Account {accountNumber: txn.to})
                CREATE (from)-[r:TRANSACTION {
                    amt: txn.amt,
                    type: txn.type,
                    currency: txn.currency,
                    description: txn.description,
                    createdDate: datetime(txn.createdDate)
                }]->(to)
                """, transactions=current_batch)
                
            print(f"Created {len(current_batch)} transactions (total: {min(batch + batch_size, len(transactions))})")
        
    def generate_data(self, account_count=10000, fraud_patterns_config=None, min_transactions=60000):
        """Generate accounts and transactions with fraud patterns"""
        # Step 1: Create network structure with fraud patterns
        G, fraud_accounts, community_map = self.create_network_structure(
            account_count, fraud_patterns_config
        )
        
        # Step 2: Calculate network metrics
        account_metrics = self.analyze_network(G, fraud_accounts, community_map)
        
        # Step 3: Generate transactions based on network structure
        transactions = self.generate_transactions_from_graph(G, account_metrics, min_transactions)
        
        # Step 4: Save accounts to Neo4j
        self.save_accounts_to_neo4j(G, account_metrics)
        
        # Step 5: Save transactions to Neo4j
        self.save_transactions_to_neo4j(transactions)
        
        print("Data generation complete!")
        print(f"Generated {account_count} accounts ({len(fraud_accounts)} suspicious)")
        print(f"Generated {len(transactions)} transactions")
        
        # Return summary statistics for verification
        return {
            "total_accounts": account_count,
            "suspicious_accounts": len(fraud_accounts),
            "transactions": len(transactions),
            "fraud_patterns": {
                pattern: sum(1 for acc in accs if acc in fraud_accounts)
                for pattern, accs in fraud_patterns_config.items()
            } if fraud_patterns_config else {}
        }

def get_user_input_for_patterns():
    """Get input from user for fraud pattern configuration"""
    print("\nFraud Pattern Configuration:")
    print("-" * 30)
    print("Enter the number of accounts for each fraud pattern type:")
    
    patterns = {
        'cycle': "Cycle Pattern (money flows in loops)",
        'star': "Star Pattern (central hub with spokes)",
        'chain': "Chain Pattern (linear transaction chains)",
        'layered': "Layered Pattern (multi-layer transaction structure)"
    }
    
    fraud_patterns_config = {}
    
    for pattern_key, pattern_desc in patterns.items():
        while True:
            try:
                count = input(f"{pattern_desc} count: ")
                if not count.strip():  # Allow empty input to skip
                    count = 0
                else:
                    count = int(count)
                    
                if count < 0:
                    print("Please enter a non-negative number")
                    continue
                    
                fraud_patterns_config[pattern_key] = count
                break
            except ValueError:
                print("Please enter a valid number")
    
    # Print summary
    total_fraud = sum(fraud_patterns_config.values())
    print(f"\nTotal fraud accounts: {total_fraud}")
    
    return fraud_patterns_config

def main():
    """Main function to run the data generator"""
    try:
        # First try to import NetworkX
        import networkx as nx
    except ImportError:
        print("NetworkX is required for this script. Please install it using:")
        print("pip install networkx")
        return
    
    load_dotenv()  # Load environment variables from .env file
    
    # Get connection details from environment variables
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri or not username or not password:
        print("Please provide NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables")
        print("You can set these in a .env file or as environment variables")
        return
    
    # Create Neo4j data generator
    generator = Neo4jFraudDataGenerator(uri, username, password)
    
    try:
        # Ask for confirmation before clearing database
        clear_db = input("Clear existing database? (y/n): ").lower() == 'y'
        if clear_db:
            generator.clear_database()
        
        # Get account count
        while True:
            try:
                account_count = input("Enter number of accounts to generate (default: 10000): ")
                if not account_count.strip():
                    account_count = 10000
                else:
                    account_count = int(account_count)
                
                if account_count <= 0:
                    print("Please enter a positive number")
                    continue
                break
            except ValueError:
                print("Please enter a valid number")
        
        # Get transaction count
        while True:
            try:
                min_transactions = input(f"Enter minimum number of transactions to generate (default: {account_count * 6}): ")
                if not min_transactions.strip():
                    min_transactions = account_count * 6
                else:
                    min_transactions = int(min_transactions)
                
                if min_transactions <= 0:
                    print("Please enter a positive number")
                    continue
                break
            except ValueError:
                print("Please enter a valid number")
        
        # Get fraud pattern configuration
        fraud_patterns_config = get_user_input_for_patterns()
        
        # Confirm total fraud accounts
        total_fraud = sum(fraud_patterns_config.values())
        if total_fraud > account_count * 0.5:
            print(f"Warning: {total_fraud} fraud accounts is more than 50% of total accounts ({account_count})")
            confirm = input("Continue anyway? (y/n): ").lower()
            if confirm != 'y':
                return
        
        # Generate data
        stats = generator.generate_data(account_count, fraud_patterns_config, min_transactions)
        
        # Print summary statistics
        print("\nGeneration Summary:")
        print("-" * 30)
        print(f"Total accounts: {stats['total_accounts']}")
        print(f"Suspicious accounts: {stats['suspicious_accounts']}")
        print(f"Transactions: {stats['transactions']}")
        print("\nFraud Pattern Distribution:")
        for pattern, count in stats['fraud_patterns'].items():
            if count > 0:
                print(f"  - {pattern.capitalize()}: {count} accounts")
        
    finally:
        # Close the driver connection
        generator.close()

if __name__ == "__main__":
    main()