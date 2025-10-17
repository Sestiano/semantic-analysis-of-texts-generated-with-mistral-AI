import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class StatisticalAnalyzer:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def clean_data(self, data):
        """Clean data by removing infinite values and NaN"""
        data = data.replace([np.inf, -np.inf], np.nan)
        return data.dropna()
    
    def _get_metric_cols(self, df):
        """Get numeric columns excluding temperature"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_cols if col != 'temperature']
    
    def run_normality_tests(self, df):
        """Run normality tests on network metrics"""
        print("Running normality tests...")
        results = []
        
        for metric in self._get_metric_cols(df):
            data = self.clean_data(df[metric])
            if len(data) > 3:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(data)
                    jb_stat, jb_p = stats.jarque_bera(data)
                    
                    results.extend([
                        {
                            'analysis_type': 'normality_test',
                            'metric': metric,
                            'test': 'shapiro_wilk',
                            'statistic': shapiro_stat,
                            'p_value': shapiro_p,
                            'significant': shapiro_p < self.alpha
                        },
                        {
                            'analysis_type': 'normality_test',
                            'metric': metric,
                            'test': 'jarque_bera',
                            'statistic': jb_stat,
                            'p_value': jb_p,
                            'significant': jb_p < self.alpha
                        }
                    ])
                except Exception as e:
                    print(f"Error in normality test for {metric}: {e}")
                        
        return results
    
    def _calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                            (len(group2) - 1) * group2.var()) / 
                           (len(group1) + len(group2) - 2))
        return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
    
    def run_pairwise_comparisons(self, df):
        """Run pairwise comparisons between groups"""
        print("Running pairwise comparisons...")
        results = []
        
        prompt_types = df['prompt_type'].unique()
        if len(prompt_types) < 2:
            return results
            
        for metric in self._get_metric_cols(df):
            group1 = self.clean_data(df[df['prompt_type'] == prompt_types[0]][metric])
            group2 = self.clean_data(df[df['prompt_type'] == prompt_types[1]][metric])
            
            if len(group1) > 0 and len(group2) > 0:
                try:
                    mw_stat, mw_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                    cohens_d = self._calculate_cohens_d(group1, group2)
                    
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
        
        df_clean = df.copy()
        df_clean['temp_group'] = pd.cut(df_clean['temperature'], bins=3, labels=['Low', 'Medium', 'High'])
        
        for metric in self._get_metric_cols(df_clean):
            try:
                groups = [self.clean_data(group[metric]) 
                         for _, group in df_clean.groupby('temp_group') 
                         if len(self.clean_data(group[metric])) > 0]
                
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
        
        prompt_types = df['prompt_type'].unique()
        if len(prompt_types) < 2:
            return results
            
        for metric in self._get_metric_cols(df):
            group1 = self.clean_data(df[df['prompt_type'] == prompt_types[0]][metric])
            group2 = self.clean_data(df[df['prompt_type'] == prompt_types[1]][metric])
            
            if len(group1) > 0 and len(group2) > 0:
                try:
                    results.append({
                        'analysis_type': 'effect_size',
                        'metric': metric,
                        'cohens_d': self._calculate_cohens_d(group1, group2),
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
        
        metric_cols = self._get_metric_cols(df)
        if len(metric_cols) <= 1:
            return results
            
        try:
            pca_data = self.clean_data(df[metric_cols])
            if len(pca_data) == 0:
                return results
                
            scaler = StandardScaler()
            pca = PCA()
            pca.fit(scaler.fit_transform(pca_data))
            
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
        
        metric_cols = self._get_metric_cols(df)
        if len(metric_cols) <= 1 or len(df) <= 3:
            return results
            
        try:
            cluster_data = self.clean_data(df[metric_cols])
            if len(cluster_data) == 0:
                return results
                
            scaler = StandardScaler()
            cluster_data_scaled = scaler.fit_transform(cluster_data)
            
            for k in range(2, min(6, len(cluster_data))):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(cluster_data_scaled)
                
                if len(set(labels)) > 1:
                    results.append({
                        'analysis_type': 'clustering',
                        'n_clusters': k,
                        'silhouette_score': silhouette_score(cluster_data_scaled, labels),
                        'inertia': kmeans.inertia_
                    })
        except Exception as e:
            print(f"Error in clustering analysis: {e}")
                
        return results
    
    def run_regression_analysis(self, df):
        """Run correlation analysis with temperature"""
        print("Running regression analysis...")
        results = []
        
        for metric in self._get_metric_cols(df):
            try:
                temp_data = self.clean_data(df['temperature'])
                metric_data = self.clean_data(df[metric])
                common_idx = temp_data.index.intersection(metric_data.index)
                
                if len(common_idx) > 3:
                    temp_clean = temp_data.loc[common_idx]
                    metric_clean = metric_data.loc[common_idx]
                    
                    pearson_r, pearson_p = stats.pearsonr(temp_clean, metric_clean)
                    spearman_r, spearman_p = stats.spearmanr(temp_clean, metric_clean)
                    
                    results.extend([
                        {
                            'analysis_type': 'correlation',
                            'metric': metric,
                            'correlation_type': 'pearson',
                            'correlation': pearson_r,
                            'p_value': pearson_p,
                            'predictor': 'temperature'
                        },
                        {
                            'analysis_type': 'correlation',
                            'metric': metric,
                            'correlation_type': 'spearman',
                            'correlation': spearman_r,
                            'p_value': spearman_p,
                            'predictor': 'temperature'
                        }
                    ])
            except Exception as e:
                print(f"Error in correlation analysis for {metric}: {e}")
                    
        return results
    
    def run_advanced_analysis(self, df):
        """Run comprehensive advanced analysis"""
        print("Starting comprehensive statistical analysis...")
        
        try:
            df_clean = self.clean_data(df)
            if len(df_clean) == 0:
                print("No valid data after cleaning")
                return pd.DataFrame()
            
            all_results = []
            for method in [self.run_normality_tests, self.run_pairwise_comparisons,
                          self.run_anova_analysis, self.calculate_effect_sizes,
                          self.run_pca_analysis, self.run_clustering_analysis,
                          self.run_regression_analysis]:
                all_results.extend(method(df_clean))
            
            if all_results:
                print(f"Advanced analysis completed: {len(all_results)} results generated")
                return pd.DataFrame(all_results)
            
            print("No advanced analysis results generated")
            return pd.DataFrame()
                
        except Exception as e:
            print(f"Error in advanced analysis: {e}")
            return pd.DataFrame()