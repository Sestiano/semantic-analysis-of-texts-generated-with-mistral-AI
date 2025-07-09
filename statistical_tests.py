import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

class StatisticalAnalyzer:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def clean_data(self, data):
        """Clean data by removing infinite values and NaN"""
        if isinstance(data, pd.Series):
            # Replace infinite values with NaN
            data = data.replace([np.inf, -np.inf], np.nan)
            # Drop NaN values
            return data.dropna()
        elif isinstance(data, pd.DataFrame):
            # Replace infinite values with NaN
            data = data.replace([np.inf, -np.inf], np.nan)
            # Drop rows with any NaN
            return data.dropna()
        return data
    
    def run_normality_tests(self, df):
        """Run normality tests on network metrics"""
        print("Running normality tests...")
        results = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if col not in ['temperature']]
        
        for metric in metric_cols:
            if metric in df.columns:
                data = self.clean_data(df[metric])
                if len(data) > 3:
                    try:
                        # Shapiro-Wilk test
                        shapiro_stat, shapiro_p = stats.shapiro(data)
                        
                        # Jarque-Bera test
                        jb_stat, jb_p = stats.jarque_bera(data)
                        
                        results.append({
                            'analysis_type': 'normality_test',
                            'metric': metric,
                            'test': 'shapiro_wilk',
                            'statistic': shapiro_stat,
                            'p_value': shapiro_p,
                            'significant': shapiro_p < self.alpha
                        })
                        
                        results.append({
                            'analysis_type': 'normality_test',
                            'metric': metric,
                            'test': 'jarque_bera',
                            'statistic': jb_stat,
                            'p_value': jb_p,
                            'significant': jb_p < self.alpha
                        })
                        
                    except Exception as e:
                        print(f"Error in normality test for {metric}: {e}")
                        
        return results
    
    def run_pairwise_comparisons(self, df):
        """Run pairwise comparisons between groups"""
        print("Running pairwise comparisons...")
        results = []
        
        if len(df['prompt_type'].unique()) >= 2:
            prompt_types = df['prompt_type'].unique()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            metric_cols = [col for col in numeric_cols if col not in ['temperature']]
            
            for metric in metric_cols:
                if metric in df.columns:
                    group1 = self.clean_data(df[df['prompt_type'] == prompt_types[0]][metric])
                    group2 = self.clean_data(df[df['prompt_type'] == prompt_types[1]][metric])
                    
                    if len(group1) > 0 and len(group2) > 0:
                        try:
                            # Mann-Whitney U test
                            mw_stat, mw_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                                                (len(group2) - 1) * group2.var()) / 
                                               (len(group1) + len(group2) - 2))
                            cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
                            
                            results.append({
                                'analysis_type': 'group_comparison',
                                'metric': metric,
                                'test': 'mann_whitney_u',
                                'statistic': mw_stat,
                                'p_value': mw_p,
                                'effect_size': cohens_d,
                                'group1': prompt_types[0],
                                'group2': prompt_types[1],
                                'group1_mean': group1.mean(),
                                'group2_mean': group2.mean()
                            })
                            
                        except Exception as e:
                            print(f"Error in pairwise comparison for {metric}: {e}")
                            
        return results
    
    def run_anova_analysis(self, df):
        """Run ANOVA analysis"""
        print("Running ANOVA analysis...")
        results = []
        
        # Group by temperature ranges for ANOVA
        df_clean = df.copy()
        df_clean['temp_group'] = pd.cut(df_clean['temperature'], bins=3, labels=['Low', 'Medium', 'High'])
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if col not in ['temperature']]
        
        for metric in metric_cols:
            if metric in df_clean.columns:
                try:
                    groups = []
                    for name, group in df_clean.groupby('temp_group'):
                        cleaned_group = self.clean_data(group[metric])
                        if len(cleaned_group) > 0:
                            groups.append(cleaned_group)
                    
                    if len(groups) >= 2:
                        f_stat, f_p = stats.f_oneway(*groups)
                        
                        results.append({
                            'analysis_type': 'anova',
                            'metric': metric,
                            'f_statistic': f_stat,
                            'p_value': f_p,
                            'significant': f_p < self.alpha,
                            'groups': 'temperature_ranges'
                        })
                        
                except Exception as e:
                    print(f"Error in ANOVA for {metric}: {e}")
                    
        return results
    
    def calculate_effect_sizes(self, df):
        """Calculate effect sizes"""
        print("Running effect size calculations...")
        results = []
        
        if len(df['prompt_type'].unique()) >= 2:
            prompt_types = df['prompt_type'].unique()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            metric_cols = [col for col in numeric_cols if col not in ['temperature']]
            
            for metric in metric_cols:
                if metric in df.columns:
                    group1 = self.clean_data(df[df['prompt_type'] == prompt_types[0]][metric])
                    group2 = self.clean_data(df[df['prompt_type'] == prompt_types[1]][metric])
                    
                    if len(group1) > 0 and len(group2) > 0:
                        try:
                            # Cohen's d
                            pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + (len(group2) - 1) * group2.var()) / (len(group1) + len(group2) - 2))
                            cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
                            
                            results.append({
                                'analysis_type': 'effect_size',
                                'metric': metric,
                                'cohens_d': cohens_d,
                                'group1': prompt_types[0],
                                'group2': prompt_types[1],
                                'group1_mean': group1.mean(),
                                'group2_mean': group2.mean()
                            })
                            
                        except Exception as e:
                            print(f"Error calculating effect size for {metric}: {e}")
                            
        return results
    
    def run_pca_analysis(self, df):
        """Run PCA analysis"""
        print("Running PCA analysis...")
        results = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if col not in ['temperature']]
        
        if len(metric_cols) > 1:
            try:
                # Prepare data for PCA
                pca_data = df[metric_cols].copy()
                pca_data = pca_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(pca_data) > 0:
                    # Standardize data
                    scaler = StandardScaler()
                    pca_data_scaled = scaler.fit_transform(pca_data)
                    
                    # Perform PCA
                    pca = PCA()
                    pca_result = pca.fit_transform(pca_data_scaled)
                    
                    # Calculate explained variance
                    explained_var = pca.explained_variance_ratio_
                    cumulative_var = np.cumsum(explained_var)
                    
                    for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
                        results.append({
                            'analysis_type': 'pca',
                            'component': f'PC{i+1}',
                            'explained_variance': var,
                            'cumulative_variance': cum_var,
                            'eigenvalue': pca.explained_variance_[i]
                        })
                        
            except Exception as e:
                print(f"Error in PCA analysis: {e}")
                
        return results
    
    def run_clustering_analysis(self, df):
        """Run clustering analysis"""
        print("Running clustering analysis...")
        results = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if col not in ['temperature']]
        
        if len(metric_cols) > 1 and len(df) > 3:
            try:
                cluster_data = df[metric_cols].copy()
                cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(cluster_data) > 0:
                    # Standardize data
                    scaler = StandardScaler()
                    cluster_data_scaled = scaler.fit_transform(cluster_data)
                    
                    # Try different numbers of clusters
                    for k in range(2, min(6, len(cluster_data))):
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        cluster_labels = kmeans.fit_predict(cluster_data_scaled)
                        
                        if len(set(cluster_labels)) > 1:
                            score = silhouette_score(cluster_data_scaled, cluster_labels)
                            
                            results.append({
                                'analysis_type': 'clustering',
                                'n_clusters': k,
                                'silhouette_score': score,
                                'inertia': kmeans.inertia_
                            })
                            
            except Exception as e:
                print(f"Error in clustering analysis: {e}")
                
        return results
    
    def run_regression_analysis(self, df):
        """Run correlation analysis with temperature"""
        print("Running regression analysis...")
        results = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if col not in ['temperature']]
        
        for metric in metric_cols:
            if metric in df.columns:
                try:
                    # Clean both temperature and metric data
                    temp_data = self.clean_data(df['temperature'])
                    metric_data = self.clean_data(df[metric])
                    
                    # Get common indices
                    common_idx = temp_data.index.intersection(metric_data.index)
                    if len(common_idx) > 3:
                        temp_clean = temp_data.loc[common_idx]
                        metric_clean = metric_data.loc[common_idx]
                        
                        # Pearson correlation
                        pearson_r, pearson_p = stats.pearsonr(temp_clean, metric_clean)
                        
                        # Spearman correlation
                        spearman_r, spearman_p = stats.spearmanr(temp_clean, metric_clean)
                        
                        results.append({
                            'analysis_type': 'correlation',
                            'metric': metric,
                            'correlation_type': 'pearson',
                            'correlation': pearson_r,
                            'p_value': pearson_p,
                            'predictor': 'temperature'
                        })
                        
                        results.append({
                            'analysis_type': 'correlation',
                            'metric': metric,
                            'correlation_type': 'spearman',
                            'correlation': spearman_r,
                            'p_value': spearman_p,
                            'predictor': 'temperature'
                        })
                        
                except Exception as e:
                    print(f"Error in correlation analysis for {metric}: {e}")
                    
        return results
    
    def run_advanced_analysis(self, df):
        """Run comprehensive advanced analysis"""
        print("Starting comprehensive statistical analysis...")
        all_results = []
        
        try:
            # Clean the dataframe first
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df_clean) == 0:
                print("No valid data after cleaning")
                return pd.DataFrame()
            
            # Run all analysis methods
            print("Running normality tests...")
            all_results.extend(self.run_normality_tests(df_clean))
            
            print("Running pairwise comparisons...")
            all_results.extend(self.run_pairwise_comparisons(df_clean))
            
            print("Running ANOVA analysis...")
            all_results.extend(self.run_anova_analysis(df_clean))
            
            print("Calculating effect sizes...")
            all_results.extend(self.calculate_effect_sizes(df_clean))
            
            print("Running PCA analysis...")
            all_results.extend(self.run_pca_analysis(df_clean))
            
            print("Running clustering analysis...")
            all_results.extend(self.run_clustering_analysis(df_clean))
            
            print("Running regression analysis...")
            all_results.extend(self.run_regression_analysis(df_clean))
            
            if all_results:
                print(f"Advanced analysis completed: {len(all_results)} results generated")
                return pd.DataFrame(all_results)
            else:
                print("No advanced analysis results generated")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error in advanced analysis: {e}")
            return pd.DataFrame()