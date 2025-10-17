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
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

warnings.filterwarnings("ignore", category=FutureWarning)


class NetworkVisualizer:
    """Handles visualization of network analysis results."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        plt.style.use(style)
        self.figsize = figsize
        self.colors = {'complex': '#1f77b4', 'vague': '#ff7f0e'}
    
    def _save_fig(self, save_path: str, default_path: str) -> str:
        """Save figure and return path."""
        path = save_path or default_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    
    def create_overview_plots(self, df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Create overview plots for network analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Network Analysis Overview', fontsize=16, fontweight='bold')
        
        self._plot_temperature_metric(df, 'density', 'Density', axes[0, 0])
        self._plot_temperature_metric(df, 'clustering', 'Clustering Coefficient', axes[0, 1])
        self._plot_nodes_edges(df, axes[1, 0])
        self._plot_temperature_metric(df, 'avg_deg', 'Average Degree', axes[1, 1])
        
        plt.tight_layout()
        return self._save_fig(save_path, 'visualizations/network_overview.png')
    
    def _plot_temperature_metric(self, df: pd.DataFrame, metric: str, ylabel: str, ax: plt.Axes) -> None:
        """Plot temperature vs metric for different prompt types."""
        if 'prompt_type' in df.columns:
            for pt in df['prompt_type'].unique():
                if not (sub := df[df['prompt_type'] == pt].sort_values('temperature')).empty:
                    ax.plot(sub['temperature'], sub[metric], marker='o', linewidth=2, 
                           markersize=6, color=self.colors.get(pt, 'gray'), label=pt.capitalize())
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
            for pt in df['prompt_type'].unique():
                if not (sub := df[df['prompt_type'] == pt]).empty:
                    ax.scatter(sub['nodes'], sub['edges'], s=60, alpha=0.7,
                             color=self.colors.get(pt, 'gray'), label=pt.capitalize())
        else:
            ax.scatter(df['nodes'], df['edges'], s=60, alpha=0.7)
        
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Number of Edges')
        ax.set_title('Nodes vs Edges')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_clustering_plots(self, df: pd.DataFrame, metrics: List[str],
                              save_path: Optional[str] = None) -> Optional[str]:
        """Create clustering analysis plots."""
        X = df[metrics].dropna()
        if len(X) < 4:
            print("Insufficient data for clustering visualization")
            return None
        
        X_scaled = StandardScaler().fit_transform(X)
        K_range = range(2, min(8, len(X)//2))
        
        if not K_range:
            print("Not enough data points for clustering")
            return None
        
        # Calculate metrics
        inertias, sil_scores, ch_scores = [], [], []
        for k in K_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                inertias.append(kmeans.inertia_)
                sil_scores.append(silhouette_score(X_scaled, labels))
                ch_scores.append(calinski_harabasz_score(X_scaled, labels))
            except Exception as e:
                print(f"Clustering failed for k={k}: {e}")
        
        if not sil_scores:
            print("Clustering failed for all k values")
            return None
        
        optimal_k = K_range[np.argmax(sil_scores)]
        df_analysis = df.loc[X.index].copy()
        df_analysis['cluster'] = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(X_scaled)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clustering Analysis', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(K_range, inertias, 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(K_range, sil_scores, 'ro-')
        axes[0, 1].axvline(optimal_k, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Analysis')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(K_range, ch_scores, 'go-')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('Calinski-Harabasz Index')
        axes[1, 0].grid(True, alpha=0.3)
        
        if 'prompt_type' in df_analysis.columns:
            cluster_comp = df_analysis.groupby(['cluster', 'prompt_type']).size().unstack(fill_value=0)
            sns.heatmap(cluster_comp, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title('Cluster Composition')
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        return self._save_fig(save_path, 'figures/clustering_analysis.png')
    
    def create_pca_plots(self, df: pd.DataFrame, metrics: List[str],
                        save_path: Optional[str] = None) -> Optional[str]:
        """Create PCA analysis plots."""
        X = df[metrics].dropna()
        if len(X) < 3:
            print("Insufficient data for PCA visualization")
            return None
        
        try:
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            n_comp = min(3, len(metrics))
            loadings = pca.components_[:n_comp].T * np.sqrt(pca.explained_variance_[:n_comp])
            loadings_df = pd.DataFrame(loadings, index=metrics, columns=[f'PC{i+1}' for i in range(n_comp)])
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('PCA Analysis', fontsize=16, fontweight='bold')
            
            axes[0, 0].plot(range(1, len(explained_var)+1), explained_var, 'bo-')
            axes[0, 0].set_xlabel('Principal Component')
            axes[0, 0].set_ylabel('Explained Variance')
            axes[0, 0].set_title('Scree Plot')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(range(1, len(cumulative_var)+1), cumulative_var, 'ro-')
            axes[0, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].set_xlabel('Number of Components')
            axes[0, 1].set_ylabel('Cumulative Variance')
            axes[0, 1].set_title('Cumulative Variance')
            axes[0, 1].grid(True, alpha=0.3)
            
            if X_pca.shape[1] >= 2:
                colors = ([self.colors.get(pt, 'gray') for pt in df.loc[X.index, 'prompt_type']] 
                         if 'prompt_type' in df.columns else 'blue')
                axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7)
                axes[1, 0].set_xlabel(f'PC1 ({explained_var[0]:.2%})')
                axes[1, 0].set_ylabel(f'PC2 ({explained_var[1]:.2%})')
                axes[1, 0].set_title('Biplot PC1 vs PC2')
                axes[1, 0].grid(True, alpha=0.3)
            
            sns.heatmap(loadings_df.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[1, 1])
            axes[1, 1].set_title('Component Loadings')
            
            plt.tight_layout()
            return self._save_fig(save_path, 'figures/pca_analysis.png')
            
        except Exception as e:
            print(f"PCA visualization failed: {e}")
            return None
    
    def visualize_network_neighborhood(self, G: nx.Graph, word: str, radius: int = 1, 
                                     layout: str = 'spring', title_suffix: str = '', 
                                     k: float = 0.5, save_path: Optional[str] = None) -> bool:
        """Visualize the neighborhood around a word in a network."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if word not in G:
            ax.text(0.5, 0.5, f'Word "{word}" not found in network.', ha='center', va='center', fontsize=12)
            ax.set_title(f'Word not found - {title_suffix}')
            ax.axis('off')
            return False
        
        sub = nx.ego_graph(G, word, radius=radius)
        if len(sub) == 0:
            ax.text(0.5, 0.5, f'No connections found for "{word}".', ha='center', va='center', fontsize=12)
            ax.set_title(f'No connections - {title_suffix}')
            ax.axis('off')
            return False
        
        layout_funcs = {
            'spring': lambda g: nx.spring_layout(g, seed=42, k=k),
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'spectral': nx.spectral_layout
        }
        pos = layout_funcs.get(layout, nx.spring_layout)(sub)
        pos = {n: (x + np.random.normal(0, 0.02), y + np.random.normal(0, 0.02)) for n, (x, y) in pos.items()}
        
        nx.draw_networkx_edges(sub, pos, ax=ax, connectionstyle='arc3,rad=0.1', alpha=0.6, 
                              edge_color='gray', width=1.5)
        
        node_colors = ['#FF6B6B' if n == word else '#4ECDC4' for n in sub]
        node_sizes = [200 if n == word else 80 for n in sub]
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        centrality = nx.degree_centrality(sub)
        min_c, max_c = min(centrality.values()), max(centrality.values())
        bbox = dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8)
        
        for n, (x, y) in pos.items():
            font_size = 14 if n == word else 8 + 4 * (centrality[n] - min_c) / (max_c - min_c or 1)
            ax.text(x, y + 0.03, n, fontsize=font_size, ha='center', va='center', 
                   bbox=bbox, weight='bold' if n == word else 'normal')
        
        ax.set_title(f'Network Neighborhood: "{word}" (radius={radius}) - {title_suffix}', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return True
    
    def load_network_from_file(self, file_path: str) -> Optional[nx.Graph]:
        """Load a network from an edge list file."""
        try:
            G = nx.Graph()
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(parts := line.strip().split('\t')) >= 2:
                        G.add_edge(parts[0], parts[1])
            return G
        except Exception as e:
            print(f"Error loading network from {file_path}: {e}")
            return None
    
    def create_comparative_plot(self, df: pd.DataFrame, metric: str,
                               save_path: Optional[str] = None) -> str:
        """Create a comparative plot for a specific metric."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'prompt_type' in df.columns:
            for pt in df['prompt_type'].unique():
                if not (sub := df[df['prompt_type'] == pt].sort_values('temperature')).empty:
                    ax.plot(sub['temperature'], sub[metric], marker='o', linewidth=2, markersize=8,
                           color=self.colors.get(pt, 'gray'), label=pt.capitalize())
        else:
            ax.plot(df['temperature'], df[metric], 'o-', linewidth=2, markersize=8)
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Temperature vs {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self._save_fig(save_path, f'visualizations/{metric}_comparison.png')