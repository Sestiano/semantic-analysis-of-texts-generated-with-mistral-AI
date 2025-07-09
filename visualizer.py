"""
Visualization utilities for network analysis.
Extracted from Hakan.ipynb for better modularity.
"""

import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

warnings.filterwarnings("ignore", category=FutureWarning)


class NetworkVisualizer:
    """Handles visualization of network analysis results."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = {'complex': '#1f77b4', 'vague': '#ff7f0e'}
    
    def create_overview_plots(self, df: pd.DataFrame, 
                            save_path: Optional[str] = None) -> str:
        """
        Create overview plots for network analysis.
        
        Args:
            df: DataFrame with network metrics
            save_path: Path to save the figure
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Network Analysis Overview', fontsize=16, fontweight='bold')
        
        # Temperature vs Density
        self._plot_temperature_metric(df, 'density', 'Density', axes[0, 0])
        
        # Temperature vs Clustering
        self._plot_temperature_metric(df, 'clustering', 'Clustering Coefficient', axes[0, 1])
        
        # Nodes vs Edges scatter
        self._plot_nodes_edges(df, axes[1, 0])
        
        # Temperature vs Average Degree
        self._plot_temperature_metric(df, 'avg_deg', 'Average Degree', axes[1, 1])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = 'visualizations/network_overview.png'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_temperature_metric(self, df: pd.DataFrame, metric: str, 
                               ylabel: str, ax: plt.Axes) -> None:
        """Plot temperature vs metric for different prompt types."""
        if 'prompt_type' in df.columns:
            for prompt_type in df['prompt_type'].unique():
                sub = df[df['prompt_type'] == prompt_type]
                if not sub.empty:
                    sub = sub.sort_values('temperature')
                    ax.plot(sub['temperature'], sub[metric], 
                           marker='o', linewidth=2, markersize=6, 
                           color=self.colors.get(prompt_type, 'gray'), 
                           label=prompt_type.capitalize())
        else:
            ax.plot(df['temperature'], df[metric], 'o-', linewidth=2, markersize=6)
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Temperature vs {ylabel}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_nodes_edges(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot nodes vs edges scatter plot."""
        if 'prompt_type' in df.columns:
            for prompt_type in df['prompt_type'].unique():
                sub = df[df['prompt_type'] == prompt_type]
                if not sub.empty:
                    ax.scatter(sub['nodes'], sub['edges'], 
                             s=60, alpha=0.7, 
                             color=self.colors.get(prompt_type, 'gray'), 
                             label=prompt_type.capitalize())
        else:
            ax.scatter(df['nodes'], df['edges'], s=60, alpha=0.7)
        
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Number of Edges')
        ax.set_title('Nodes vs Edges')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_clustering_plots(self, df: pd.DataFrame, metrics: List[str],
                              save_path: Optional[str] = None) -> Optional[str]:
        """
        Create clustering analysis plots.
        
        Args:
            df: DataFrame with network metrics
            metrics: List of metrics to use for clustering
            save_path: Path to save the figure
            
        Returns:
            Path to saved figure or None if failed
        """
        # Prepare data
        X = df[metrics].dropna()
        if len(X) < 4:
            print("Insufficient data for clustering visualization")
            return None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Range of clusters to test
        K_range = range(2, min(8, len(X)//2))
        
        if len(K_range) == 0:
            print("Not enough data points for clustering")
            return None
        
        # Calculate metrics
        inertias = []
        silhouette_scores = []
        calinski_harabasz_scores = []
        
        for k in K_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, labels))
                calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, labels))
            except Exception as e:
                print(f"Clustering failed for k={k}: {e}")
                continue
        
        if not silhouette_scores:
            print("Clustering failed for all k values")
            return None
        
        # Find optimal k
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        # Final clustering
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df_analysis = df.loc[X.index].copy()
        df_analysis['cluster'] = kmeans_final.fit_predict(X_scaled)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clustering Analysis', fontsize=16, fontweight='bold')
        
        # Elbow curve
        axes[0, 0].plot(K_range, inertias, 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette scores
        axes[0, 1].plot(K_range, silhouette_scores, 'ro-')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Analysis')
        axes[0, 1].axvline(optimal_k, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz scores
        axes[1, 0].plot(K_range, calinski_harabasz_scores, 'go-')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('Calinski-Harabasz Index')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cluster composition heatmap
        if 'prompt_type' in df_analysis.columns:
            cluster_composition = df_analysis.groupby(['cluster', 'prompt_type']).size().unstack(fill_value=0)
            sns.heatmap(cluster_composition, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title('Cluster Composition')
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = 'figures/clustering_analysis.png'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_pca_plots(self, df: pd.DataFrame, metrics: List[str],
                        save_path: Optional[str] = None) -> Optional[str]:
        """
        Create PCA analysis plots.
        
        Args:
            df: DataFrame with network metrics
            metrics: List of metrics to use for PCA
            save_path: Path to save the figure
            
        Returns:
            Path to saved figure or None if failed
        """
        # Prepare data
        X = df[metrics].dropna()
        if len(X) < 3:
            print("Insufficient data for PCA visualization")
            return None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        try:
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            
            # Explained variance
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            # Loadings
            n_components = min(3, len(metrics))
            loadings = pca.components_[:n_components].T * np.sqrt(pca.explained_variance_[:n_components])
            loadings_df = pd.DataFrame(loadings, index=metrics, 
                                      columns=[f'PC{i+1}' for i in range(n_components)])
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('PCA Analysis', fontsize=16, fontweight='bold')
            
            # Scree plot
            axes[0, 0].plot(range(1, len(explained_var)+1), explained_var, 'bo-')
            axes[0, 0].set_xlabel('Principal Component')
            axes[0, 0].set_ylabel('Explained Variance')
            axes[0, 0].set_title('Scree Plot')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Cumulative variance
            axes[0, 1].plot(range(1, len(cumulative_var)+1), cumulative_var, 'ro-')
            axes[0, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].set_xlabel('Number of Components')
            axes[0, 1].set_ylabel('Cumulative Variance')
            axes[0, 1].set_title('Cumulative Variance')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Biplot PC1 vs PC2
            if X_pca.shape[1] >= 2:
                if 'prompt_type' in df.columns:
                    colors = [self.colors.get(pt, 'gray') for pt in df.loc[X.index, 'prompt_type']]
                else:
                    colors = 'blue'
                
                axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7)
                axes[1, 0].set_xlabel(f'PC1 ({explained_var[0]:.2%})')
                axes[1, 0].set_ylabel(f'PC2 ({explained_var[1]:.2%})')
                axes[1, 0].set_title('Biplot PC1 vs PC2')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Loadings heatmap
            sns.heatmap(loadings_df.T, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, ax=axes[1, 1])
            axes[1, 1].set_title('Component Loadings')
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = 'figures/pca_analysis.png'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            print(f"PCA visualization failed: {e}")
            return None
    
    def visualize_network_neighborhood(self, G: nx.Graph, word: str, 
                                     radius: int = 1, layout: str = 'spring',
                                     title_suffix: str = '', k: float = 0.5,
                                     save_path: Optional[str] = None) -> bool:
        """
        Visualize the neighborhood around a word in a network.
        
        Args:
            G: NetworkX graph
            word: Central word
            radius: Maximum radius from central node
            layout: Layout algorithm
            title_suffix: Suffix for title
            k: Spring layout parameter
            save_path: Path to save the figure
            
        Returns:
            True if visualization was successful
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if word not in G:
            ax.text(0.5, 0.5, f'Word "{word}" not found in network.', 
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'Word not found - {title_suffix}')
            ax.axis('off')
            return False
        
        # Extract ego subgraph
        sub = nx.ego_graph(G, word, radius=radius)
        
        if len(sub) == 0:
            ax.text(0.5, 0.5, f'No connections found for "{word}".', 
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'No connections - {title_suffix}')
            ax.axis('off')
            return False
        
        # Layout selection
        layout_funcs = {
            'spring': lambda g: nx.spring_layout(g, seed=42, k=k),
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'spectral': nx.spectral_layout
        }
        pos = layout_funcs.get(layout, nx.spring_layout)(sub)
        
        # Add slight jitter to avoid overlaps
        pos = {n: (x + np.random.normal(0, 0.02), y + np.random.normal(0, 0.02))
               for n, (x, y) in pos.items()}
        
        # Draw edges
        nx.draw_networkx_edges(sub, pos, ax=ax,
                               connectionstyle='arc3,rad=0.1', 
                               alpha=0.6, edge_color='gray', width=1.5)
        
        # Draw nodes
        node_colors = ['#FF6B6B' if n == word else '#4ECDC4' for n in sub]
        node_sizes = [200 if n == word else 80 for n in sub]
        
        nx.draw_networkx_nodes(sub, pos, ax=ax,
                               node_color=node_colors, 
                               node_size=node_sizes,
                               alpha=0.8)
        
        # Add labels
        centrality = nx.degree_centrality(sub)
        min_c, max_c = min(centrality.values()), max(centrality.values())
        
        bbox = dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8)
        for n, (x, y) in pos.items():
            font_size = 14 if n == word else 8 + 4 * (centrality[n] - min_c) / (max_c - min_c or 1)
            ax.text(x, y + 0.03, n, fontsize=font_size, ha='center', va='center', 
                    bbox=bbox, weight='bold' if n == word else 'normal')
        
        # Title
        title = f'Network Neighborhood: "{word}" (radius={radius}) - {title_suffix}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return True
    
    def load_network_from_file(self, file_path: str) -> Optional[nx.Graph]:
        """
        Load a network from an edge list file.
        
        Args:
            file_path: Path to the edge list file
            
        Returns:
            NetworkX graph or None if failed
        """
        try:
            G = nx.Graph()
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        G.add_edge(parts[0], parts[1])
            return G
        except Exception as e:
            print(f"Error loading network from {file_path}: {e}")
            return None
    
    def create_comparative_plot(self, df: pd.DataFrame, metric: str,
                               save_path: Optional[str] = None) -> str:
        """
        Create a comparative plot for a specific metric.
        
        Args:
            df: DataFrame with network metrics
            metric: Metric to plot
            save_path: Path to save the figure
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'prompt_type' in df.columns:
            for prompt_type in df['prompt_type'].unique():
                sub = df[df['prompt_type'] == prompt_type]
                if not sub.empty:
                    sub = sub.sort_values('temperature')
                    ax.plot(sub['temperature'], sub[metric], 
                           marker='o', linewidth=2, markersize=8, 
                           color=self.colors.get(prompt_type, 'gray'), 
                           label=prompt_type.capitalize())
        else:
            ax.plot(df['temperature'], df[metric], 'o-', linewidth=2, markersize=8)
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Temperature vs {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = f'visualizations/{metric}_comparison.png'
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
