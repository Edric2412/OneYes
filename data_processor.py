import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import numpy as np
import hashlib
import plotly.express as px
import logging

class DataProcessor:
    def __init__(self):
        pass

    def load_data(self, uploaded_file):
        try:
            file_extension = uploaded_file.name.split('.')[-1]
            if file_extension == 'csv':
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0) # Reset stream position to the beginning
                    df = pd.read_csv(uploaded_file, encoding='latin1')
            
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def clean_data(self, df):
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - df.shape[0]

        # For simplicity, fill all NaN values with the mean of their respective columns
        # This is a basic approach; more sophisticated imputation might be needed for specific datasets
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                df.loc[:, column] = df[column].fillna(df[column].mean())
            elif df[column].dtype == 'object':
                df.loc[:, column] = df[column].fillna(df[column].mode()[0])

        missing_values_before = df.isnull().sum().to_dict()

        cleaning_summary = {
            "before": {"rows": initial_rows, "missing_values": missing_values_before},
            "after": {"rows": df.shape[0]},
            "duplicates_removed": duplicates_removed,
            "missing_values_filled": True # Assuming all missing values are handled by fillna
        }
        return df, cleaning_summary

    def preprocess_data(self, df):
        # Exclude non-numeric columns and the 'id' column if it exists
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'id' in numeric_cols:
            numeric_cols.remove('id')
        
        if not numeric_cols:
            st.error("No numeric columns found for clustering.")
            return None, None

        X = df[numeric_cols]

        # Drop features with zero variance
        # This prevents issues with StandardScaler and PCA
        variances = X.var()
        to_drop = variances[variances == 0].index.tolist()
        if to_drop:
            st.warning(f"Dropping zero-variance features: {', '.join(to_drop)}")
            X = X.drop(columns=to_drop)
            numeric_cols = X.columns.tolist()

        if not numeric_cols:
            st.error("No numeric columns remaining after dropping zero-variance features.")
            return None, None

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Optional PCA reduction to 95% explained variance
        # This part would typically be controlled by a user setting in the UI
        if X_scaled.shape[1] > 1: # Only apply PCA if more than one feature
            pca = PCA(n_components=0.95) # Retain 95% of variance
            X_scaled = pca.fit_transform(X_scaled)
            st.info(f"PCA applied: Reduced dimensions to {X_scaled.shape[1]} while retaining 95% variance.")

        # Return a serializable representation of the preprocessor details
        preprocessor_details = {
            "scaler": "StandardScaler",
            "pca_applied": True if X_scaled.shape[1] < X.shape[1] else False,
            "n_components_after_pca": X_scaled.shape[1]
        }
        return X_scaled, preprocessor_details

    def apply_clustering(self, X, algorithm_name, n_clusters='auto'):
        model = None
        labels = None
        
        try:
            if algorithm_name == 'KMeans':
                if n_clusters == 'auto':
                    # Use Silhouette Score to find optimal k
                    best_k = 3 # Default
                    if X.shape[0] > 1:
                        silhouette_scores = []
                        k_range = range(2, min(11, X.shape[0])) # Try k from 2 to 10 or n_samples-1
                        for k in k_range:
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            kmeans.fit(X)
                            if len(set(kmeans.labels_)) > 1: # Ensure more than one cluster for silhouette score
                                score = silhouette_score(X, kmeans.labels_)
                                silhouette_scores.append((score, k))
                        if silhouette_scores:
                            best_k = max(silhouette_scores)[1]
                        else:
                            best_k = 2 # Default to 2 if no valid silhouette scores
                    n_clusters = best_k
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            elif algorithm_name == 'AgglomerativeClustering':
                if n_clusters == 'auto':
                    # Use Silhouette Score to find optimal k
                    best_k = 3 # Default
                    if X.shape[0] > 1:
                        silhouette_scores = []
                        k_range = range(2, min(11, X.shape[0])) # Try k from 2 to 10 or n_samples-1
                        for k in k_range:
                            agg_clustering = AgglomerativeClustering(n_clusters=k)
                            labels = agg_clustering.fit_predict(X)
                            if len(set(labels)) > 1: # Ensure more than one cluster for silhouette score
                                score = silhouette_score(X, labels)
                                silhouette_scores.append((score, k))
                        if silhouette_scores:
                            best_k = max(silhouette_scores)[1]
                        else:
                            best_k = 2 # Default to 2 if no valid silhouette scores
                    n_clusters = best_k
                model = AgglomerativeClustering(n_clusters=n_clusters)
            elif algorithm_name == 'DBSCAN':
                # DBSCAN does not take n_clusters directly. Requires eps and min_samples.
                # These parameters are highly dependent on the dataset.
                # For simplicity, using default values or requiring manual input.
                # In a real app, you'd want to guide the user or use heuristics.
                model = DBSCAN(eps=0.5, min_samples=5) # Example values
            elif algorithm_name == 'GaussianMixture':
                if n_clusters == 'auto':
                    # Use BIC to find the best number of components
                    lowest_bic = np.inf
                    best_k = 1 # Default to 1 component if no better option found
                    n_components_range = range(1, min(11, X.shape[0])) # Try components from 1 to 10 or n_samples-1
                    if X.shape[0] == 0: # Handle empty data case
                        n_clusters = best_k
                    else:
                        for n_components in n_components_range:
                            gmm = GaussianMixture(n_components=n_components, random_state=42)
                            gmm.fit(X)
                            bic = gmm.bic(X)
                            if bic < lowest_bic:
                                lowest_bic = bic
                                best_k = n_components
                        n_clusters = best_k
                model = GaussianMixture(n_components=n_clusters, random_state=42)
            elif algorithm_name == 'HierarchicalClustering':
                if n_clusters == 'auto':
                    n_clusters = 3
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                st.warning(f"Unknown algorithm: {algorithm_name}")
                return None, None

            if model:
                if algorithm_name == 'GaussianMixture':
                    labels = model.fit_predict(X)
                else:
                    labels = model.fit_predict(X)

        except Exception as e:
            st.error(f"Error applying {algorithm_name} clustering: {e}")
            return None, None
            
        return labels, model

    def evaluate_clustering(self, X, labels):
        metrics = {}
        if len(set(labels)) > 1: # Need at least 2 clusters for silhouette score
            metrics["silhouette_score"] = silhouette_score(X, labels)
        else:
            metrics["silhouette_score"] = None # Or a suitable placeholder

        metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
        metrics["davies_bouldin_score"] = davies_bouldin_score(X, labels)
        metrics["n_clusters"] = len(set(labels)) - (1 if -1 in labels else 0) # Exclude noise points for DBSCAN
        return metrics

    def visualize_clusters(self, X, labels, algorithm_name):
        df_viz = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df_viz['Cluster'] = labels.astype(str) # Convert to string for categorical coloring

        # Define a consistent color palette for clusters
        # Using Plotly's 'Plotly' qualitative color sequence for consistency
        color_palette = px.colors.qualitative.Plotly

        if X.shape[1] >= 2: # For 2D or 3D scatter plot
            if X.shape[1] >= 3:
                fig = px.scatter_3d(df_viz, x='feature_0', y='feature_1', z='feature_2', color='Cluster', title=f'{algorithm_name} Clustering (3D)', color_discrete_sequence=color_palette)
            else:
                fig = px.scatter(df_viz, x='feature_0', y='feature_1', color='Cluster', title=f'{algorithm_name} Clustering (2D)', color_discrete_sequence=color_palette)
        else:
            # If data is 1D, we cannot create a 2D or 3D scatter plot. Return None.
            # A different visualization approach (e.g., histogram) would be needed for 1D data.
            return None
        
        return fig

    def get_cache_key(self, file_content_bytes, algorithm, n_clusters_config, preprocessor_details_str):
        # Generate a unique cache key based on file content hash, algorithm, n_clusters_config, and preprocessor details
        file_hash = hashlib.md5(file_content_bytes).hexdigest()
        # Include preprocessor_details_str in the hash to ensure cache invalidation if preprocessing changes
        combined_hash_input = f"{file_hash}-{algorithm}-{n_clusters_config}-{preprocessor_details_str}"
        return hashlib.md5(combined_hash_input.encode('utf-8')).hexdigest()

    def save_to_cache(self, key, data):
        st.session_state[key] = data

    def load_from_cache(self, key):
        return st.session_state.get(key)

    def process_dataset(self, uploaded_file):
        df = self.load_data(uploaded_file)
        if df is not None:
            df_cleaned, cleaning_summary = self.clean_data(df)
            if df_cleaned is not None:
                return df_cleaned, cleaning_summary
        return None, None

    def log_dataset_processing(self, username, filename, original_shape, cleaning_summary, algorithms_run):
        log_message = f"User: {username}, File: {filename}, Original Shape: {original_shape}, " \
                      f"Cleaning Summary: {cleaning_summary}, Algorithms Run: {algorithms_run}"
        from models import log_dataset_processing
        log_dataset_processing(username, filename, original_shape, cleaning_summary, algorithms_run)

    def profile_clusters(self, df, labels, numeric_cols):
        df_clustered = df.copy()
        df_clustered['Cluster'] = labels

        # Calculate mean values of each feature per cluster
        cluster_profiles = df_clustered.groupby('Cluster')[numeric_cols].mean()
        return cluster_profiles

    def create_radar_chart(self, metrics_df, selected_algorithms):
        if metrics_df is None or metrics_df.empty or len(metrics_df) < 1:
            return None

        # Metrics to include in the radar chart. Ensure they are numeric.
        # We might need to normalize them if their scales are vastly different.
        # For now, let's assume Silhouette (higher better), Calinski-Harabasz (higher better), Davies-Bouldin (lower better)
        # We need to make Davies-Bouldin comparable (e.g., invert or normalize so higher is better)
        # Or, clearly label axes. For simplicity, we'll plot as is and note interpretation.

        # Ensure required columns exist and are numeric
        required_cols = ['Algorithm', 'Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
        for col in required_cols[1:]: # Skip 'Algorithm'
            if col not in metrics_df.columns or not pd.api.types.is_numeric_dtype(metrics_df[col]):
                st.warning(f"Radar chart: Column '{col}' is missing or not numeric. Skipping radar chart.")
                return None
        
        # Handle NaN values - fill with a value that indicates poor performance or drop algorithm
        # For simplicity, let's fill with 0 for scores where higher is better, and a high number for DB (lower is better)
        # This is a simplistic approach; more sophisticated normalization/handling might be needed.
        metrics_df_radar = metrics_df.copy()
        if 'Silhouette Score' in metrics_df_radar.columns:
            metrics_df_radar['Silhouette Score'] = metrics_df_radar['Silhouette Score'].fillna(0)
        if 'Calinski-Harabasz Score' in metrics_df_radar.columns:
            metrics_df_radar['Calinski-Harabasz Score'] = metrics_df_radar['Calinski-Harabasz Score'].fillna(0)
        if 'Davies-Bouldin Score' in metrics_df_radar.columns:
            # Invert Davies-Bouldin: lower is better, so 1 / (1 + DB) makes higher better
            # Handle division by zero or inf if DB is very large
            metrics_df_radar['Davies-Bouldin Score_Inv'] = metrics_df_radar['Davies-Bouldin Score'].apply(lambda x: 1 / (1 + x) if pd.notna(x) else 0)
            metrics_df_radar['Davies-Bouldin Score_Inv'] = metrics_df_radar['Davies-Bouldin Score_Inv'].fillna(0) # Fill NaN after inversion

        # Store original values for tooltips
        original_metrics = {
            'Silhouette Score': metrics_df_radar['Silhouette Score'].tolist(),
            'Calinski-Harabasz Score': metrics_df_radar['Calinski-Harabasz Score'].tolist(),
            'Davies-Bouldin Score': metrics_df_radar['Davies-Bouldin Score'].tolist() # Keep original DB for tooltip
        }

        # Apply min-max scaling for normalization (0-1 range)
        # Silhouette Score: higher is better, range [-1, 1]
        # Calinski-Harabasz Score: higher is better, range [0, inf)
        # Davies-Bouldin Score (Inverted): higher is better, range (0, 1]

        normalized_cols = {}
        for col in ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score_Inv']:
            if col in metrics_df_radar.columns:
                min_val = metrics_df_radar[col].min()
                max_val = metrics_df_radar[col].max()
                if max_val == min_val:
                    normalized_cols[f'{col}_normalized'] = [0.5] * len(metrics_df_radar) # Assign a neutral value if all are same
                else:
                    normalized_cols[f'{col}_normalized'] = (metrics_df_radar[col] - min_val) / (max_val - min_val)
                normalized_cols[f'{col}_normalized'] = normalized_cols[f'{col}_normalized'].fillna(0) # Fill NaN after normalization

        for k, v in normalized_cols.items():
            metrics_df_radar[k] = v

        # Calculate a combined performance score (simple average of normalized scores)
        metrics_df_radar['Overall Performance_normalized'] = metrics_df_radar[['Silhouette Score_normalized', 'Calinski-Harabasz Score_normalized', 'Davies-Bouldin Score_Inv_normalized']].mean(axis=1)
        metrics_df_radar['Overall Performance_normalized'] = metrics_df_radar['Overall Performance_normalized'].fillna(0)

        # Identify the top performer
        top_performer_algo = None
        if not metrics_df_radar.empty:
            top_performer_row = metrics_df_radar.loc[metrics_df_radar['Overall Performance_normalized'].idxmax()]
            top_performer_algo = top_performer_row['Algorithm']
            top_performer_score = top_performer_row['Overall Performance_normalized']

        # Prepare data for px.line_polar
        df_plot = metrics_df_radar.melt(id_vars=['Algorithm'],
                                        value_vars=['Silhouette Score_normalized', 'Calinski-Harabasz Score_normalized', 'Davies-Bouldin Score_Inv_normalized', 'Overall Performance_normalized'],
                                        var_name='Metric', value_name='Value')

        # Map normalized metric names to display names for the chart
        metric_display_names = {
            'Silhouette Score_normalized': 'Silhouette (Higher = Better)',
            'Calinski-Harabasz Score_normalized': 'Calinski-Harabasz (Higher = Better)',
            'Davies-Bouldin Score_Inv_normalized': 'DB (Higher = Better)',
            'Overall Performance_normalized': 'Overall Performance'
        }
        df_plot['Metric_Display'] = df_plot['Metric'].map(metric_display_names)

        # Add original values for tooltips
        df_plot['Original_Value'] = df_plot.apply(lambda row:
            original_metrics['Silhouette Score'][metrics_df_radar['Algorithm'].tolist().index(row['Algorithm'])] if row['Metric'] == 'Silhouette Score_normalized' else (
            original_metrics['Calinski-Harabasz Score'][metrics_df_radar['Algorithm'].tolist().index(row['Algorithm'])] if row['Metric'] == 'Calinski-Harabasz Score_normalized' else (
            original_metrics['Davies-Bouldin Score'][metrics_df_radar['Algorithm'].tolist().index(row['Algorithm'])] if row['Metric'] == 'Davies-Bouldin Score_Inv_normalized' else np.nan
            )), axis=1)

        # Streamlit multiselect for algorithm selection
        # all_algorithms = metrics_df_radar['Algorithm'].unique().tolist()
        # selected_algorithms = st.multiselect("Select algorithms to display:", all_algorithms, default=all_algorithms)
        
        # if not selected_algorithms:
        #     st.info("Please select at least one algorithm to display the radar chart.")
        #     return None
        
        df_plot_filtered = df_plot[df_plot['Algorithm'].isin(selected_algorithms)]
        
        # Create a 3D radar chart using go.Scatterpolar
        fig = go.Figure()
        
        # Define a base z-coordinate for each algorithm to create depth
        # This can be adjusted to create more or less separation
        z_offset_multiplier = 0.1 # Adjust this value for more or less depth
        
        for i, algo in enumerate(selected_algorithms):
            algo_data = df_plot_filtered[df_plot_filtered['Algorithm'] == algo]
            r_values = algo_data['Value'].tolist()
            theta_values = algo_data['Metric_Display'].tolist()
            original_values = algo_data['Original_Value'].tolist()
        
            # Close the loop for the polygon
            r_values.append(r_values[0])
            theta_values.append(theta_values[0])
            original_values.append(original_values[0])
        
            # Assign a unique z-coordinate for each algorithm
            # This creates the 'depth' effect
            z_values = [i * z_offset_multiplier] * len(r_values)
        
            # Get color for the algorithm
            # Using Plotly's qualitative colors
            color_index = i % len(px.colors.qualitative.Plotly)
            line_color = px.colors.qualitative.Plotly[color_index]
            fill_color = line_color.replace('rgb', 'rgba').replace(')', ', 0.2)') # Semi-transparent fill with 0.2 opacity
        
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                mode='lines',
                name=algo,
                fill='toself',
                fillcolor=fill_color,
                line=dict(color=line_color, width=2),
                hoverinfo='text',
                hovertext=[f'<b>{algo}</b><br>{t}: {r:.2f}<br>Original: {o:.2f}' for t, r, o in zip(algo_data['Metric_Display'], algo_data['Value'], algo_data['Original_Value'])]
            ))
        
        # Update layout for better aesthetics and readability
        fig.update_layout(
            title={
                'text': "<b>Clustering Algorithm Performance Radar Chart</b><br><sup>All metrics normalized to [0, 1]; higher = better.</sup>",
                'y':0.99, # Adjusted to be within [0, 1] range
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'pad': {'t': 20}
            },
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.25, 0.5, 0.75, 1],
                    tickfont=dict(size=9),
                    gridcolor='lightgray',
                    linecolor='lightgray'
                ),
                angularaxis=dict(
                    tickfont=dict(size=9),
                    rotation=90,
                    direction='clockwise',
                    period=1
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=80, r=80, t=150, b=80), # Increased top margin to accommodate title
            height=600,
            template="plotly_dark"
        )

        # st.plotly_chart(fig, use_container_width=True)

        if top_performer_algo:
            st.subheader(f"🏆 Top Performing Algorithm: {top_performer_algo} (Overall Score: {top_performer_score:.2f})")

        return fig
