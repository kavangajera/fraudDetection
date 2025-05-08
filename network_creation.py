import random
import datetime
from datetime import timedelta
import json
from neo4j import GraphDatabase
import networkx as nx

class BankFraudGraphGenerator:
    """
    A class for generating bank fraud graph data with advanced graph metrics
    using NetworkX and storing in Neo4j
    """
    
    def __init__(self, uri, user, password):
        """Initialize with Neo4j connection parameters"""
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(uri=uri, auth=(user, password))
        self.graph = nx.DiGraph()  # Main directed graph
        self.undirected_graph = None  # Will be created when needed
        
    def __del__(self):
        """Close Neo4j connection when object is destroyed"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
    
    def _clear_database(self):
        """Clear existing data in Neo4j database"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        except Exception as e:
            print(f"Error clearing database: {e}")
    
    def _generate_accounts(self, num_accounts):
        """Generate bank account nodes"""
        accounts = []
        for i in range(1, num_accounts + 1):
            account_number = f"ACC{i:06d}"
            accounts.append(account_number)
            
            # Define node attributes
            node_type = random.choice(["SAVINGS", "CURRENT", "BUSINESS"])
            balance = round(random.uniform(1000, 100000), 2)
            user_name = f"User_{i}"
            transaction_frequency = random.randint(1, 50)
            regular_interval = random.choice([True, False])
            
            # Add node to NetworkX graph
            self.graph.add_node(
                account_number, 
                type=node_type, 
                balance=balance, 
                user=user_name, 
                freq=transaction_frequency,
                regularIntervalTransaction=regular_interval,
                suspicious=False
            )
            
            # Add node to Neo4j
            try:
                with self.driver.session() as session:
                    session.run(
                        """
                        CREATE (a:Account {
                            accountNumber: $account_number,
                            type: $type,
                            balance: $balance,
                            user: $user,
                            freq: $freq,
                            regularIntervalTransaction: $regular,
                            suspicious: $suspicious
                        })
                        """,
                        account_number=account_number,
                        type=node_type,
                        balance=balance,
                        user=user_name,
                        freq=transaction_frequency,
                        regular=regular_interval,
                        suspicious=False
                    )
            except Exception as e:
                print(f"Error creating node {account_number}: {e}")
        
        return accounts
    
    def _mark_fraud_accounts(self, accounts, num_fraud_accounts):
        """Select and mark fraud accounts"""
        fraud_accounts = random.sample(accounts, num_fraud_accounts)
        for acc in fraud_accounts:
            # Update node property for fraud accounts
            self.graph.nodes[acc]['suspicious'] = True
            
            # Update in Neo4j
            try:
                with self.driver.session() as session:
                    session.run(
                        "MATCH (a:Account {accountNumber: $account}) SET a.suspicious = true",
                        account=acc
                    )
            except Exception as e:
                print(f"Error updating fraud account {acc}: {e}")
        
        return fraud_accounts
    
    def _generate_transactions(self, accounts, fraud_accounts, num_transactions):
        """Generate transactions (edges) between accounts"""
        transactions = []
        start_date = datetime.datetime.now() - datetime.timedelta(days=30)
        
        for _ in range(num_transactions):
            # Select source account with bias towards fraud accounts
            if random.random() < 0.3:  # 30% chance to select a fraud account as source
                from_acc = random.choice(fraud_accounts) if fraud_accounts else random.choice(accounts)
            else:
                from_acc = random.choice(accounts)
            
            # Select target account with bias
            if from_acc in fraud_accounts and random.random() < 0.7:
                potential_recipients = random.sample(accounts, min(5, len(accounts)))
                to_acc = random.choice(potential_recipients)
            else:
                to_acc = random.choice(accounts)
                # Avoid self-loops
                while to_acc == from_acc:
                    to_acc = random.choice(accounts)
            
            # Generate transaction amount
            if from_acc in fraud_accounts:
                amount = round(random.uniform(1000, 50000), 2)
            else:
                amount = round(random.uniform(10, 5000), 2)
            
            # Generate transaction date
            days_offset = random.randint(0, 29)
            txn_date = start_date + datetime.timedelta(days=days_offset, hours=random.randint(0, 23), minutes=random.randint(0, 59))
            txn_date_str = txn_date.strftime("%Y-%m-%dT%H:%M:%S")
            
            # Create transaction record
            transaction = {
                "from": from_acc,
                "to": to_acc,
                "amt": amount,
                "type": random.choice(["TRANSFER", "PAYMENT"]),
                "currency": "INR",
                "description": f"Txn_{len(transactions)}",
                "createdDate": txn_date_str
            }
            transactions.append(transaction)
            
            # Add edge to NetworkX graph
            self.graph.add_edge(
                from_acc, to_acc, 
                amt=amount,
                type=transaction["type"],
                currency="INR",
                description=transaction["description"],
                createdDate=txn_date_str
            )
            
            # Add transaction to Neo4j
            try:
                with self.driver.session() as session:
                    session.run(
                        """
                        MATCH (from:Account {accountNumber: $from_acc})
                        MATCH (to:Account {accountNumber: $to_acc})
                        CREATE (from)-[:TRANSACTION {
                            amt: $amount,
                            type: $type,
                            currency: $currency,
                            description: $description,
                            createdDate: $created_date
                        }]->(to)
                        """,
                        from_acc=from_acc,
                        to_acc=to_acc,
                        amount=amount,
                        type=transaction["type"],
                        currency="INR",
                        description=transaction["description"],
                        created_date=txn_date_str
                    )
            except Exception as e:
                print(f"Error creating transaction {from_acc} -> {to_acc}: {e}")
        
        return transactions
    
    def _get_undirected_graph(self):
        """Get or create undirected version of the graph for certain metrics"""
        if self.undirected_graph is None:
            self.undirected_graph = self.graph.to_undirected()
        return self.undirected_graph
    
    def _detect_communities(self):
        """Detect communities using available methods"""
        G_undirected = self._get_undirected_graph()
        communities = {}
        community_sizes = {}
        
        try:
            # Try Louvain method first (better but requires additional package)
            from community import best_partition
            partition = best_partition(G_undirected)
            for node, community_id in partition.items():
                communities[node] = community_id
                if community_id not in community_sizes:
                    community_sizes[community_id] = 0
                community_sizes[community_id] += 1
            community_count = len(set(partition.values()))
        except:
            print("python-louvain not available, using spectral clustering or synthetic communities")
            
            # Try spectral clustering if graph is connected
            try:
                if nx.is_connected(G_undirected):
                    # Use spectral clustering to create communities
                    # Determine number of communities based on graph size
                    num_communities = max(3, min(10, len(G_undirected.nodes()) // 15))
                    
                    # Convert to adjacency matrix for spectral clustering
                    adj_matrix = nx.to_numpy_array(G_undirected)
                    
                    # Use NetworkX's spectral clustering
                    clusters = nx.spectral_clustering(adj_matrix, n_clusters=num_communities)
                    
                    # Assign nodes to communities
                    for i, node in enumerate(G_undirected.nodes()):
                        community_id = int(clusters[i])
                        communities[node] = community_id
                        if community_id not in community_sizes:
                            community_sizes[community_id] = 0
                        community_sizes[community_id] += 1
                    
                    community_count = num_communities
                else:
                    # Fall back to connected components for disconnected graphs
                    components = list(nx.connected_components(G_undirected))
                    for i, component in enumerate(components):
                        for node in component:
                            communities[node] = i
                        community_sizes[i] = len(component)
                    community_count = len(components)
            except:
                # Create synthetic communities if spectral clustering fails
                print("Spectral clustering failed, creating synthetic communities")
                nodes = list(G_undirected.nodes())
                num_communities = max(3, min(10, len(nodes) // 15))
                
                # Create communities with preferential attachment
                # Fraud accounts and their neighbors tend to be in same community
                fraud_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('suspicious', False)]
                
                # Start with fraud-centered communities
                for i, fraud_node in enumerate(fraud_nodes):
                    comm_id = i % num_communities
                    communities[fraud_node] = comm_id
                    if comm_id not in community_sizes:
                        community_sizes[comm_id] = 0
                    community_sizes[comm_id] += 1
                    
                    # Add neighbors with high probability
                    for neighbor in nx.all_neighbors(G_undirected, fraud_node):
                        if neighbor not in communities and random.random() < 0.7:
                            communities[neighbor] = comm_id
                            community_sizes[comm_id] += 1
                
                # Assign remaining nodes
                remaining = [n for n in nodes if n not in communities]
                for node in remaining:
                    # Choose community with some randomness
                    comm_id = random.randint(0, num_communities - 1)
                    communities[node] = comm_id
                    if comm_id not in community_sizes:
                        community_sizes[comm_id] = 0
                    community_sizes[comm_id] += 1
                
                community_count = num_communities
        
        return communities, community_sizes, community_count
    
    def _calculate_metrics(self, accounts, fraud_accounts):
        """Calculate all graph metrics using NetworkX"""
        print("Calculating graph metrics...")
        
        # Basic centrality metrics
        pagerank_scores = nx.pagerank(self.graph)
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        betweenness_scores = nx.betweenness_centrality(self.graph)
        
        # Community detection
        communities, community_sizes, community_count = self._detect_communities()
        
        # Triangle count using undirected graph
        G_undirected = self._get_undirected_graph()
        triangle_counts = nx.triangles(G_undirected)
        
        # Cycle detection - simplified approach for performance
        cycle_counts = {}
        for account in accounts:
            try:
                # Find cycles of length 2-3 that include this node
                cycles = 0
                neighbors = set(self.graph.successors(account)).union(set(self.graph.predecessors(account)))
                neighbors.add(account)
                subgraph = self.graph.subgraph(list(neighbors))
                
                for length in range(2, 4):
                    for cycle in nx.simple_cycles(subgraph):
                        if account in cycle and len(cycle) <= length:
                            cycles += 1
                cycle_counts[account] = cycles
            except Exception as e:
                # If it fails, just set to 0
                cycle_counts[account] = 0
        
        # Intermediate Account Detection (nodes that connect others)
        intermediate_accounts = {}
        for account in accounts:
            paths = 0
            for source in self.graph.predecessors(account):
                for target in self.graph.successors(account):
                    if source != target and source != account and target != account:
                        paths += 1
            intermediate_accounts[account] = paths
        
        # Fraud Score Calculation
        fraud_scores = {}
        for account in accounts:
            # Initialize score
            score = 0
            
            # Add points for high centrality
            score += pagerank_scores.get(account, 0) * 1000
            score += betweenness_scores.get(account, 0) * 100
            
            # Add points for unusual transaction patterns
            score += intermediate_accounts.get(account, 0) * 0.5
            score += cycle_counts.get(account, 0) * 2
            
            # Add points for degree imbalance
            in_deg = in_degrees.get(account, 0)
            out_deg = out_degrees.get(account, 0)
            if in_deg > 0 and out_deg > 0:
                score += abs(in_deg - out_deg) * 0.5
            
            # Normalize score
            fraud_scores[account] = min(round(score, 2), 100)
        
        # Ensure known fraud accounts have high scores
        for account in fraud_accounts:
            fraud_scores[account] = max(fraud_scores[account], 75)
        
        # Update Neo4j with scores
        try:
            with self.driver.session() as session:
                for account, score in fraud_scores.items():
                    session.run(
                        """
                        MATCH (a:Account {accountNumber: $account})
                        SET a.fraudScore = $score,
                            a.pagerank = $pagerank,
                            a.betweenness = $betweenness,
                            a.inDegree = $in_degree,
                            a.outDegree = $out_degree,
                            a.communityId = $community_id,
                            a.communitySize = $community_size,
                            a.triangleCount = $triangle_count,
                            a.cycleCount = $cycle_count,
                            a.intermediateAccount = $intermediate_account
                        """,
                        account=account,
                        score=score,
                        pagerank=pagerank_scores.get(account, 0),
                        betweenness=betweenness_scores.get(account, 0),
                        in_degree=in_degrees.get(account, 0),
                        out_degree=out_degrees.get(account, 0),
                        community_id=communities.get(account, 0),
                        community_size=community_sizes.get(communities.get(account, 0), 0),
                        triangle_count=triangle_counts.get(account, 0),
                        cycle_count=cycle_counts.get(account, 0),
                        intermediate_account=intermediate_accounts.get(account, 0)
                    )
        except Exception as e:
            print(f"Error updating metrics in Neo4j: {e}")
        
        return {
            "pagerank": pagerank_scores,
            "in_degrees": in_degrees,
            "out_degrees": out_degrees,
            "betweenness": betweenness_scores,
            "communities": communities,
            "community_sizes": community_sizes,
            "community_count": community_count,
            "triangle_counts": triangle_counts,
            "cycle_counts": cycle_counts,
            "intermediate_accounts": intermediate_accounts,
            "fraud_scores": fraud_scores
        }
    
    def _compile_account_data(self, accounts, fraud_accounts, metrics):
        """Compile all account data with metrics for output"""
        account_data_list = []
        for account in accounts:
            node_data = self.graph.nodes[account]
            community_id = metrics["communities"].get(account, 0)
            
            account_data = {
                "accountNumber": account,
                "type": node_data["type"],
                "balance": node_data["balance"],
                "user": node_data["user"],
                "freq": node_data["freq"],
                "regularIntervalTransaction": node_data["regularIntervalTransaction"],
                "suspicious": node_data["suspicious"],
                "pagerank": round(metrics["pagerank"].get(account, 0), 6),
                "in_degree": metrics["in_degrees"].get(account, 0),
                "out_degree": metrics["out_degrees"].get(account, 0),
                "betweenness": round(metrics["betweenness"].get(account, 0), 6),
                "community_id": community_id,
                "community_size": metrics["community_sizes"].get(community_id, 1),
                "triangle_count": metrics["triangle_counts"].get(account, 0),
                "cycle_count": metrics["cycle_counts"].get(account, 0),
                "intermediate_account": metrics["intermediate_accounts"].get(account, 0),
                "is_fraud": account in fraud_accounts,
                "fraud_score": metrics["fraud_scores"].get(account, 0)
            }
            
            account_data_list.append(account_data)
        
        return account_data_list
    
    def generate_graph(self, num_accounts, num_transactions, num_fraud_accounts, output_file=None):
        """
        Main method to generate the fraud graph and compute all metrics
        
        Parameters:
        - num_accounts: Total number of accounts (nodes)
        - num_transactions: Total number of transactions (edges)
        - num_fraud_accounts: Number of accounts that are fraudulent
        - output_file: Optional file path to save the JSON output
        
        Returns:
        - Dictionary with accounts and transactions data
        """
        # Clear database
        self._clear_database()
        
        # Generate accounts
        accounts = self._generate_accounts(num_accounts)
        
        # Mark fraud accounts
        fraud_accounts = self._mark_fraud_accounts(accounts, num_fraud_accounts)
        
        # Generate transactions
        transactions = self._generate_transactions(accounts, fraud_accounts, num_transactions)
        
        # Calculate all metrics
        metrics = self._calculate_metrics(accounts, fraud_accounts)
        
        # Compile account data with metrics
        account_data_list = self._compile_account_data(accounts, fraud_accounts, metrics)
        
        # Prepare output data
        output_data = {
            "accounts": account_data_list,
            "transactions": transactions,
            "stats": {
                "total_accounts": len(accounts),
                "fraud_accounts": len(fraud_accounts),
                "total_transactions": len(transactions),
                "community_count": metrics["community_count"]
            }
        }
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
        
        return output_data


class FraudDetectionQueries:
    """Class for generating Cypher queries for fraud detection"""
    
    @staticmethod
    def get_query_dict():
        """Returns a dictionary of useful Cypher queries for fraud detection"""
        queries = {
            "Find Accounts with High Fraud Scores": """
    MATCH (a:Account)
    WHERE a.fraudScore > 70
    RETURN a.accountNumber, a.fraudScore, a.suspicious
    ORDER BY a.fraudScore DESC
            """,
            
            "Find Suspicious Transaction Patterns": """
    MATCH p=(a:Account)-[:TRANSACTION]->(b:Account)-[:TRANSACTION]->(c:Account)
    WHERE a.accountNumber = c.accountNumber AND a <> b
    RETURN a.accountNumber AS account, COUNT(p) AS cycleCount
    ORDER BY cycleCount DESC
    LIMIT 10
            """,
            
            "Find Accounts with Unusual Transaction Volumes": """
    MATCH (a:Account)
    WITH a, size((a)<-[:TRANSACTION]-()) AS inDegree, 
         size((a)-[:TRANSACTION]->()) AS outDegree
    WHERE abs(inDegree - outDegree) > 5
    RETURN a.accountNumber, inDegree, outDegree, abs(inDegree - outDegree) AS degree_difference
    ORDER BY degree_difference DESC
            """,
            
            "Find High Value Transactions": """
    MATCH (a:Account)-[t:TRANSACTION]->(b:Account)
    WHERE t.amt > 10000
    RETURN a.accountNumber AS from, b.accountNumber AS to, t.amt AS amount, t.createdDate
    ORDER BY amount DESC
    LIMIT 20
            """,
            
            "Find Intermediate Accounts (Money Funnels)": """
    MATCH (source:Account)-[:TRANSACTION]->(middle:Account)-[:TRANSACTION]->(destination:Account)
    WHERE source <> destination AND source <> middle AND middle <> destination
    WITH middle, COUNT(DISTINCT source) AS sourcesCount, COUNT(DISTINCT destination) AS destinationsCount
    WHERE sourcesCount > 3 AND destinationsCount > 3
    RETURN middle.accountNumber, middle.fraudScore, sourcesCount, destinationsCount,
           middle.communityId, middle.communitySize, middle.triangleCount, middle.cycleCount
    ORDER BY sourcesCount + destinationsCount DESC
            """,
            
            "Community Analysis": """
    MATCH (a:Account)
    RETURN a.communityId, COUNT(a) as communitySize, 
           avg(a.fraudScore) as avgFraudScore,
           count(CASE WHEN a.suspicious THEN 1 END) as fraudAccounts
    ORDER BY avgFraudScore DESC
            """
        }
        
        return queries


# Example usage
if __name__ == "__main__":
    uri = "neo4j+s://ae03c8f0.databases.neo4j.io"
    user = "neo4j"
    password = "Fa01ciGZHymObLA2cOv-UDQ96BCSr3Uq6Tlqur1Ye8E"  # Replace with your actual password
    
    # Generate data
    generator = BankFraudGraphGenerator(uri, user, password)
    data = generator.generate_graph(
        num_accounts=10000,
        num_transactions=60000,
        num_fraud_accounts=377,
        output_file="bank_fraud_data.json"
    )
    
    # Print useful Cypher queries
    print("\nUseful Cypher queries for fraud detection:")
    for name, query in FraudDetectionQueries.get_query_dict().items():
        print(f"\n--- {name} ---")
        print(query)