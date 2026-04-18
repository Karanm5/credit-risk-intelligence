"""
Graph-based feature engineering for credit risk assessment.
Models customer-merchant relationships as a network to detect hidden risk patterns.

This is the UNIQUE differentiator of this platform:
- Traditional credit scoring looks at individual behaviour
- Graph features capture network effects and risk propagation
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import logging
from concurrent.futures import ProcessPoolExecutor
import warnings

logger = logging.getLogger(__name__)


@dataclass
class GraphConfig:
    """Configuration for graph feature computation."""
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 100
    community_resolution: float = 1.0
    min_edge_weight: float = 0.0
    risk_propagation_depth: int = 2
    risk_decay_factor: float = 0.5


class TransactionGraphBuilder:
    """
    Builds a bipartite graph from transaction data.
    
    Nodes: Customers and Merchants
    Edges: Transactions (weighted by amount/frequency)
    """
    
    def __init__(self, config: Optional[GraphConfig] = None):
        """
        Initialise graph builder.
        
        Args:
            config: Optional GraphConfig for parameters
        """
        self.config = config or GraphConfig()
        self.graph: Optional[nx.Graph] = None
        self.customer_nodes: Set[str] = set()
        self.merchant_nodes: Set[str] = set()
    
    def build_graph(
        self, 
        transactions: pd.DataFrame,
        customer_col: str = "customer_id",
        merchant_col: str = "merchant_id",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp"
    ) -> nx.Graph:
        """
        Build transaction graph from DataFrame.
        
        Args:
            transactions: Transaction DataFrame
            customer_col: Customer ID column
            merchant_col: Merchant ID column
            amount_col: Transaction amount column
            timestamp_col: Transaction timestamp column
            
        Returns:
            NetworkX graph
        """
        # Aggregate transactions by customer-merchant pair
        edge_data = (
            transactions
            .groupby([customer_col, merchant_col])
            .agg({
                amount_col: ["sum", "mean", "count"],
                timestamp_col: ["min", "max"]
            })
            .reset_index()
        )
        
        # Flatten column names
        edge_data.columns = [
            customer_col, merchant_col, 
            "total_amount", "avg_amount", "txn_count",
            "first_txn", "last_txn"
        ]
        
        # Create graph
        self.graph = nx.Graph()
        
        # Add nodes with type attribute
        customers = transactions[customer_col].unique()
        merchants = transactions[merchant_col].unique()
        
        self.customer_nodes = set(f"C_{c}" for c in customers)
        self.merchant_nodes = set(f"M_{m}" for m in merchants)
        
        for c in customers:
            self.graph.add_node(f"C_{c}", node_type="customer", original_id=c)
        
        for m in merchants:
            self.graph.add_node(f"M_{m}", node_type="merchant", original_id=m)
        
        # Add edges
        for _, row in edge_data.iterrows():
            customer_node = f"C_{row[customer_col]}"
            merchant_node = f"M_{row[merchant_col]}"
            
            # Edge weight combines frequency and amount
            weight = np.log1p(row["txn_count"]) * np.log1p(row["total_amount"])
            
            if weight >= self.config.min_edge_weight:
                self.graph.add_edge(
                    customer_node,
                    merchant_node,
                    weight=weight,
                    total_amount=row["total_amount"],
                    avg_amount=row["avg_amount"],
                    txn_count=row["txn_count"],
                    first_txn=row["first_txn"],
                    last_txn=row["last_txn"]
                )
        
        logger.info(
            f"Built graph with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )
        
        return self.graph
    
    def get_subgraph_for_customer(
        self, 
        customer_id: str, 
        depth: int = 2
    ) -> nx.Graph:
        """
        Extract ego-network subgraph for a customer.
        
        Args:
            customer_id: Customer identifier
            depth: Number of hops from customer
            
        Returns:
            Subgraph containing customer's network neighborhood
        """
        customer_node = f"C_{customer_id}"
        
        if customer_node not in self.graph:
            return nx.Graph()
        
        return nx.ego_graph(self.graph, customer_node, radius=depth)


class GraphFeatureEngineer:
    """
    Computes graph-based features for credit risk assessment.
    
    Features include:
    - Centrality measures (PageRank, degree, betweenness)
    - Community detection and community-level risk
    - Risk propagation from known risky entities
    - Structural features (clustering, path lengths)
    """
    
    def __init__(self, config: Optional[GraphConfig] = None):
        """
        Initialise graph feature engineer.
        
        Args:
            config: Optional GraphConfig
        """
        self.config = config or GraphConfig()
        self.graph_builder = TransactionGraphBuilder(config)
    
    def compute_all_features(
        self, 
        transactions: pd.DataFrame,
        merchant_risk_scores: Optional[Dict[str, float]] = None,
        customer_col: str = "customer_id",
        merchant_col: str = "merchant_id",
        amount_col: str = "amount"
    ) -> pd.DataFrame:
        """
        Compute all graph features for customers.
        
        Args:
            transactions: Transaction DataFrame
            merchant_risk_scores: Optional dict of merchant_id -> risk_score
            customer_col: Customer ID column
            merchant_col: Merchant ID column
            amount_col: Amount column
            
        Returns:
            DataFrame with graph features per customer
        """
        # Build graph
        graph = self.graph_builder.build_graph(
            transactions, customer_col, merchant_col, amount_col
        )
        
        # Get unique customers
        customers = transactions[customer_col].unique()
        
        # Compute centrality features
        centrality_features = self._compute_centrality_features(graph, customers)
        
        # Compute community features
        community_features = self._compute_community_features(graph, customers)
        
        # Compute structural features
        structural_features = self._compute_structural_features(graph, customers)
        
        # Compute risk propagation features
        if merchant_risk_scores:
            risk_features = self._compute_risk_propagation(
                graph, customers, merchant_risk_scores
            )
        else:
            risk_features = pd.DataFrame({
                customer_col: customers,
                "merchant_risk_exposure": 0.0,
                "high_risk_merchant_ratio": 0.0
            })
        
        # Merge all features
        result = centrality_features.merge(
            community_features, on=customer_col
        ).merge(
            structural_features, on=customer_col
        ).merge(
            risk_features, on=customer_col
        )
        
        return result
    
    def _compute_centrality_features(
        self, 
        graph: nx.Graph, 
        customers: np.ndarray
    ) -> pd.DataFrame:
        """Compute centrality-based features."""
        
        # PageRank - identifies influential nodes
        pagerank = nx.pagerank(
            graph, 
            alpha=self.config.pagerank_alpha,
            max_iter=self.config.pagerank_max_iter,
            weight="weight"
        )
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(graph)
        
        # Weighted degree (sum of edge weights)
        weighted_degree = dict(graph.degree(weight="weight"))
        
        # Betweenness centrality (expensive - sample for large graphs)
        if graph.number_of_nodes() > 5000:
            # Sample for large graphs
            betweenness = nx.betweenness_centrality(
                graph, k=min(500, graph.number_of_nodes())
            )
        else:
            betweenness = nx.betweenness_centrality(graph)
        
        # Build result DataFrame
        results = []
        for customer in customers:
            node = f"C_{customer}"
            results.append({
                "customer_id": customer,
                "pagerank_score": pagerank.get(node, 0),
                "degree_centrality": degree_centrality.get(node, 0),
                "weighted_degree": weighted_degree.get(node, 0),
                "betweenness_centrality": betweenness.get(node, 0)
            })
        
        return pd.DataFrame(results)
    
    def _compute_community_features(
        self, 
        graph: nx.Graph, 
        customers: np.ndarray
    ) -> pd.DataFrame:
        """Compute community detection features."""
        
        # Louvain community detection
        try:
            communities = nx.community.louvain_communities(
                graph, 
                resolution=self.config.community_resolution,
                seed=42
            )
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return pd.DataFrame({
                "customer_id": customers,
                "community_id": -1,
                "community_size": 0,
                "community_density": 0.0
            })
        
        # Map nodes to communities
        node_to_community = {}
        for idx, community in enumerate(communities):
            for node in community:
                node_to_community[node] = idx
        
        # Compute community-level statistics
        community_sizes = {i: len(c) for i, c in enumerate(communities)}
        
        # Community density (internal edges / possible internal edges)
        community_densities = {}
        for idx, community in enumerate(communities):
            subgraph = graph.subgraph(community)
            n = len(community)
            if n > 1:
                density = nx.density(subgraph)
            else:
                density = 0.0
            community_densities[idx] = density
        
        # Build result DataFrame
        results = []
        for customer in customers:
            node = f"C_{customer}"
            comm_id = node_to_community.get(node, -1)
            
            results.append({
                "customer_id": customer,
                "community_id": comm_id,
                "community_size": community_sizes.get(comm_id, 0),
                "community_density": community_densities.get(comm_id, 0.0)
            })
        
        return pd.DataFrame(results)
    
    def _compute_structural_features(
        self, 
        graph: nx.Graph, 
        customers: np.ndarray
    ) -> pd.DataFrame:
        """Compute structural graph features."""
        
        # Clustering coefficient
        clustering = nx.clustering(graph)
        
        # Number of merchants (direct connections)
        merchant_counts = {}
        for customer in customers:
            node = f"C_{customer}"
            if node in graph:
                neighbors = list(graph.neighbors(node))
                merchant_counts[customer] = len([
                    n for n in neighbors if n.startswith("M_")
                ])
            else:
                merchant_counts[customer] = 0
        
        # Average neighbor degree
        avg_neighbor_degree = nx.average_neighbor_degree(graph, weight="weight")
        
        # Build result DataFrame
        results = []
        for customer in customers:
            node = f"C_{customer}"
            results.append({
                "customer_id": customer,
                "clustering_coefficient": clustering.get(node, 0),
                "num_merchants": merchant_counts.get(customer, 0),
                "avg_neighbor_degree": avg_neighbor_degree.get(node, 0)
            })
        
        return pd.DataFrame(results)
    
    def _compute_risk_propagation(
        self, 
        graph: nx.Graph,
        customers: np.ndarray,
        merchant_risk_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Compute risk propagation features.
        
        Risk propagates from high-risk merchants to connected customers,
        with decay based on distance and edge weight.
        """
        
        # Add risk scores to merchant nodes
        for merchant_id, risk_score in merchant_risk_scores.items():
            node = f"M_{merchant_id}"
            if node in graph:
                graph.nodes[node]["risk_score"] = risk_score
        
        results = []
        
        for customer in customers:
            node = f"C_{customer}"
            
            if node not in graph:
                results.append({
                    "customer_id": customer,
                    "merchant_risk_exposure": 0.0,
                    "high_risk_merchant_ratio": 0.0,
                    "max_merchant_risk": 0.0,
                    "weighted_avg_merchant_risk": 0.0
                })
                continue
            
            # Get connected merchants
            neighbors = list(graph.neighbors(node))
            merchant_neighbors = [n for n in neighbors if n.startswith("M_")]
            
            if not merchant_neighbors:
                results.append({
                    "customer_id": customer,
                    "merchant_risk_exposure": 0.0,
                    "high_risk_merchant_ratio": 0.0,
                    "max_merchant_risk": 0.0,
                    "weighted_avg_merchant_risk": 0.0
                })
                continue
            
            # Collect risk scores
            merchant_risks = []
            edge_weights = []
            
            for merchant_node in merchant_neighbors:
                risk = graph.nodes[merchant_node].get("risk_score", 0.0)
                weight = graph.edges[node, merchant_node].get("weight", 1.0)
                merchant_risks.append(risk)
                edge_weights.append(weight)
            
            merchant_risks = np.array(merchant_risks)
            edge_weights = np.array(edge_weights)
            
            # Weighted average risk exposure
            if edge_weights.sum() > 0:
                weighted_avg_risk = np.average(merchant_risks, weights=edge_weights)
            else:
                weighted_avg_risk = merchant_risks.mean() if len(merchant_risks) > 0 else 0.0
            
            # High risk merchant ratio (risk > 0.5)
            high_risk_ratio = (merchant_risks > 0.5).mean()
            
            results.append({
                "customer_id": customer,
                "merchant_risk_exposure": merchant_risks.sum(),
                "high_risk_merchant_ratio": high_risk_ratio,
                "max_merchant_risk": merchant_risks.max() if len(merchant_risks) > 0 else 0.0,
                "weighted_avg_merchant_risk": weighted_avg_risk
            })
        
        return pd.DataFrame(results)
    
    def compute_second_order_risk(
        self, 
        graph: nx.Graph,
        customers: np.ndarray,
        customer_risk_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Compute second-order risk: risk from other customers 
        who shop at the same merchants.
        
        This captures "guilt by association" patterns.
        """
        
        results = []
        
        for customer in customers:
            node = f"C_{customer}"
            
            if node not in graph:
                results.append({
                    "customer_id": customer,
                    "peer_risk_exposure": 0.0,
                    "high_risk_peer_ratio": 0.0
                })
                continue
            
            # Get 2-hop customer neighbors (customers who share merchants)
            peer_risks = []
            
            for merchant in graph.neighbors(node):
                if not merchant.startswith("M_"):
                    continue
                    
                for peer in graph.neighbors(merchant):
                    if peer.startswith("C_") and peer != node:
                        peer_id = peer[2:]  # Remove "C_" prefix
                        peer_risk = customer_risk_scores.get(peer_id, 0.0)
                        peer_risks.append(peer_risk)
            
            if peer_risks:
                results.append({
                    "customer_id": customer,
                    "peer_risk_exposure": np.mean(peer_risks),
                    "high_risk_peer_ratio": (np.array(peer_risks) > 0.5).mean(),
                    "num_risky_peers": sum(1 for r in peer_risks if r > 0.5)
                })
            else:
                results.append({
                    "customer_id": customer,
                    "peer_risk_exposure": 0.0,
                    "high_risk_peer_ratio": 0.0,
                    "num_risky_peers": 0
                })
        
        return pd.DataFrame(results)
