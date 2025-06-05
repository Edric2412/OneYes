import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
from authenticator import Authentication # Actual import
from data_processor import DataProcessor # Actual import
import models
import plotly.graph_objects as go
import json
import logging
import datetime
import io # Still useful for uploaded_file.getvalue()

# Initialize logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory of the current file to sys.path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if current_file_dir not in sys.path:
    sys.path.append(current_file_dir)

# Import local modules
try:
    from authenticator import Authentication
    from data_processor import DataProcessor
    from models import update_user, save_analysis_run, get_user_analysis_history, get_analysis_run_details, clear_user_analysis_history, delete_user_analysis_run
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}. Ensure authenticator.py, data_processor.py, and models.py are in the same directory or sys.path is correctly configured.", exc_info=True)
    st.error(f"Critical import error: {e}. Application might not function correctly. Please check the logs.")
    # Fallback to prevent complete crash if mocks are not desired but imports fail
    if 'Authentication' not in globals(): Authentication = None
    if 'DataProcessor' not in globals(): DataProcessor = None
    # Ensure model functions are defined for mock usage if models.py is missing
    if 'update_user' not in globals(): update_user = None
    if 'save_analysis_run' not in globals(): save_analysis_run = None
    if 'get_user_analysis_history' not in globals(): get_user_analysis_history = None
    if 'get_analysis_run_details' not in globals(): get_analysis_run_details = None
    if 'clear_user_analysis_history' not in globals(): clear_user_analysis_history = None
    if 'delete_user_analysis_run' not in globals(): delete_user_analysis_run = None


def render_profile_page():
    try:
        auth = Authentication()
        if not auth.require_authentication():
            return

        st.title("User Profile")
        user_info = auth.get_user_info()
        if not user_info:
            st.error("Could not retrieve user information. Please try logging out and back in.")
            logger.error("render_profile_page: auth.get_user_info() returned None or empty.")
            return

        current_username = user_info.get("username")
        current_full_name = user_info.get("full_name", "")
        current_email = user_info.get("email", "")

        if not current_username:
            st.error("Username not found in user information. Critical error.")
            logger.error("render_profile_page: Username missing from user_info.")
            return

        with st.form("profile_form"):
            st.text_input("Username", value=current_username, disabled=True, key="profile_username_disabled")
            full_name = st.text_input("Full Name", value=current_full_name, key="profile_full_name")
            email = st.text_input("Email", value=current_email, key="profile_email")
            submit_button = st.form_submit_button("Update Profile")

            if submit_button:
                if not full_name.strip() or not email.strip():
                    st.warning("Full Name and Email cannot be empty.")
                else:
                    update_data = {"full_name": full_name, "email": email}
                    try:
                        if update_user and update_user(current_username, update_data):
                            st.session_state.full_name = full_name
                            st.session_state.email = email
                            logger.info(f"User {current_username} profile updated successfully.")
                            st.success("Profile updated successfully!")
                            st.rerun()
                        elif not update_user:
                             st.error("Profile update function not available.")
                        else:
                            st.error("Failed to update profile. Please try again.")
                            logger.warning(f"update_user call failed for user {current_username}.")
                    except Exception as e:
                        st.error(f"An error occurred while updating profile: {e}")
                        logger.error(f"Exception during profile update for {current_username}: {e}", exc_info=True)
    except Exception as e:
        st.error(f"An error occurred on the profile page: {e}")
        logger.error(f"Error in render_profile_page: {e}", exc_info=True)

def render_data_pipeline_page():
    try:
        auth = Authentication()
        if not auth.require_authentication():
            return

        st.title("Data Clustering Pipeline")
        try:
            processor = DataProcessor()
        except Exception as e:
            st.error(f"Failed to initialize Data Processor: {e}")
            logger.error(f"Error initializing DataProcessor: {e}", exc_info=True)
            return

        # Initialize session state keys for file and results management
        session_keys_to_init = {
            'active_file_content': None, 'active_file_name': None,
            'clustering_results': None, 'df_cleaned_for_download': None,
            'metrics_df_for_display': None, 'show_clustering_results_section': False,
            'preprocessor_details_for_display': None, 'original_shape_for_display': None,
            'cleaning_summary_for_display': None, 'algorithms_run_for_display': None,
            'select_algo_download_labels': None # Ensure this is also initialized
        }
        for key, default_value in session_keys_to_init.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"], key="data_pipeline_uploader")

        if uploaded_file is not None:
            new_file_content = uploaded_file.getvalue()
            # Check if it's a new file or content has changed
            if st.session_state.active_file_content != new_file_content or \
               st.session_state.active_file_name != uploaded_file.name:
                logger.info(f"New file uploaded or file content changed: {uploaded_file.name}")
                st.session_state.active_file_content = new_file_content
                st.session_state.active_file_name = uploaded_file.name
                
                # Reset analysis-related states for the new file
                st.session_state.clustering_results = None
                st.session_state.df_cleaned_for_download = None
                st.session_state.metrics_df_for_display = None
                st.session_state.show_clustering_results_section = False
                st.session_state.preprocessor_details_for_display = None
                st.session_state.original_shape_for_display = None
                st.session_state.cleaning_summary_for_display = None
                st.session_state.algorithms_run_for_display = None
                st.session_state.select_algo_download_labels = None # Reset selectbox choice
            uploaded_file.seek(0) # Crucial after getvalue() if file object is used again

        # Main processing logic: proceeds if a file's content is active in session state
        if st.session_state.active_file_content is not None:
            current_file_for_processing = io.BytesIO(st.session_state.active_file_content)
            current_file_for_processing.name = st.session_state.active_file_name
            file_content_bytes_for_cache = st.session_state.active_file_content

            # Data loading and initial processing (assumed to be reasonably fast or cached by DataProcessor)
            # Wrapped in a spinner that makes sense if these ops are not instant
            with st.spinner("Loading and preparing data..."):
                try:
                    df = processor.load_data(current_file_for_processing)
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    logger.error(f"Exception in processor.load_data: {e}", exc_info=True)
                    df = None
            
            if df is not None and not df.empty:
                st.subheader("Data Preview")
                st.dataframe(df.head(5))

                try:
                    df_cleaned, cleaning_summary = processor.clean_data(df.copy())
                except Exception as e:
                    st.error(f"Error during data cleaning: {e}")
                    logger.error(f"Exception in processor.clean_data: {e}", exc_info=True)
                    df_cleaned, cleaning_summary = None, None

                if df_cleaned is not None and not df_cleaned.empty:
                    if cleaning_summary is None: # Ensure cleaning_summary has a default structure
                         cleaning_summary = {
                            "before": {"rows": df.shape[0], "missing_values": {}},
                            "after": {"rows": df_cleaned.shape[0]},
                            "duplicates_removed": "N/A"
                        }
                    st.subheader("Cleaning Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Original Rows", cleaning_summary.get("before", {}).get("rows", "N/A"))
                    with col2: st.metric("Cleaned Rows", cleaning_summary.get("after", {}).get("rows", "N/A"))
                    with col3: st.metric("Duplicates Removed", cleaning_summary.get("duplicates_removed", "N/A"))
                    missing_before = cleaning_summary.get("before", {}).get("missing_values", {})
                    if missing_before:
                        st.dataframe(pd.DataFrame(list(missing_before.items()), columns=['Column', 'Missing Count']))
                    else: st.info("No missing values data in summary or no missing values found.")


                    # Clustering Configuration (always visible if data is cleaned)
                    st.subheader("Clustering Analysis")
                    st.subheader("Clustering Configuration")
                    algorithms_options = {
                        'KMeans': {'supports_auto': True, 'default_k': 3, 'max_k': 10},
                        'AgglomerativeClustering': {'supports_auto': True, 'default_k': 3, 'max_k': 10},
                        'HierarchicalClustering': {'supports_auto': True, 'default_k': 3, 'max_k': 10},
                        'GaussianMixture': {'supports_auto': True, 'default_k': 3, 'max_k': 10}
                    }
                    selected_algorithms_from_ui = st.multiselect(
                        "Select Clustering Algorithms", options=list(algorithms_options.keys()),
                        default=list(algorithms_options.keys())[:1], key="clustering_algo_select"
                    )
                    # Initialize session state for algorithm configurations
                    for algo_name_init in algorithms_options.keys():
                        if f'{algo_name_init}_config_type' not in st.session_state:
                            st.session_state[f'{algo_name_init}_config_type'] = 'Auto'
                        if f'{algo_name_init}_manual_n_clusters' not in st.session_state:
                            st.session_state[f'{algo_name_init}_manual_n_clusters'] = algorithms_options[algo_name_init].get('default_k', 3)
                    
                    if selected_algorithms_from_ui:
                        for algo in selected_algorithms_from_ui:
                            st.write(f"**{algo} Configuration:**")
                            # ... (radio buttons and number inputs for config as before) ...
                            algo_cfg_key, algo_manual_n_key = f"{algo}_config_type", f"{algo}_manual_n_clusters"
                            cfg_type_idx = 0 if st.session_state.get(algo_cfg_key, 'Auto') == 'Auto' else 1
                            cluster_cfg_type = st.radio(f"Clusters for {algo}:", ('Auto', 'Manual'), key=algo_cfg_key, horizontal=True, index=cfg_type_idx)
                            if cluster_cfg_type == 'Manual':
                                st.number_input(f"Num clusters for {algo}", min_value=2,
                                                max_value=algorithms_options[algo].get('max_k', 20),
                                                value=st.session_state.get(algo_manual_n_key, algorithms_options[algo].get('default_k', 3)),
                                                key=algo_manual_n_key)
                            elif not algorithms_options[algo].get('supports_auto', False):
                                st.caption(f"Note: '{algo}' auto k-selection might use a default.")
                    else:
                        st.info("Select one or more clustering algorithms to configure.")


                    if st.button("Run Clustering Analysis", key="run_clustering_button"):
                        if not selected_algorithms_from_ui:
                            st.warning("Please select at least one clustering algorithm.")
                        else:
                            with st.spinner("Running clustering algorithms..."):
                                X_processed, preprocessor_details_run = processor.preprocess_data(df_cleaned.copy())
                                if X_processed is None or X_processed.shape[0] == 0:
                                    st.error("Data preprocessing failed or resulted in no data.")
                                    st.session_state.show_clustering_results_section = False
                                    return 

                                results_run, metrics_data_run = {}, []
                                for algorithm in selected_algorithms_from_ui:
                                    config_type = st.session_state.get(f'{algorithm}_config_type', 'Auto')
                                    algo_n_clusters_config = st.session_state.get(f'{algorithm}_manual_n_clusters', algorithms_options[algorithm].get('default_k', 3)) if config_type == 'Manual' else 'auto'
                                    
                                    preprocessor_details_str = json.dumps(preprocessor_details_run, sort_keys=True) if isinstance(preprocessor_details_run, dict) else str(preprocessor_details_run or "None")
                                    cache_key = processor.get_cache_key(file_content_bytes_for_cache, algorithm, str(algo_n_clusters_config), preprocessor_details_str)
                                    cached_results = processor.load_from_cache(cache_key)

                                    if cached_results:
                                        results_run[algorithm] = cached_results
                                    else:
                                        try:
                                            labels, model = processor.apply_clustering(X_processed, algorithm, algo_n_clusters_config)
                                            if labels is not None and model is not None:
                                                metrics = processor.evaluate_clustering(X_processed, labels)
                                                actual_n_clusters = len(set(lbl for lbl in labels if lbl != -1))
                                                if metrics: metrics["n_clusters_found"] = actual_n_clusters
                                                fig_viz = processor.visualize_clusters(X_processed, labels, algorithm) if X_processed.shape[0] > 0 and labels is not None else None
                                                results_run[algorithm] = {"labels": labels, "metrics": metrics, "visualization": fig_viz, "model": model}
                                                processor.save_to_cache(cache_key, results_run[algorithm])
                                            else: results_run[algorithm] = None 
                                        except Exception as e:
                                            st.error(f"Error applying {algorithm}: {e}")
                                            logger.error(f"Exception in apply_clustering for {algorithm}: {e}", exc_info=True)
                                            results_run[algorithm] = None
                                
                                # Store results and context in session state for display
                                st.session_state.clustering_results = results_run
                                st.session_state.df_cleaned_for_download = df_cleaned.copy() # df_cleaned from this scope
                                st.session_state.preprocessor_details_for_display = preprocessor_details_run
                                st.session_state.original_shape_for_display = df.shape # df from this scope
                                st.session_state.cleaning_summary_for_display = cleaning_summary # cleaning_summary from this scope
                                st.session_state.algorithms_run_for_display = selected_algorithms_from_ui # from UI at time of run

                                metrics_df_run = None
                                if any(res for res in results_run.values() if res is not None):
                                    for algo, res_item in results_run.items():
                                        if res_item and res_item.get("metrics"):
                                            # ... (populate metrics_data_run as before) ...
                                            current_metrics = res_item["metrics"]
                                            metrics_data_run.append({
                                                "Algorithm": algo,
                                                "Silhouette Score": current_metrics.get("silhouette_score"),
                                                "Calinski-Harabasz Score": current_metrics.get("calinski_harabasz_score"),
                                                "Davies-Bouldin Score": current_metrics.get("davies_bouldin_score"),
                                                "Number of Clusters Found": current_metrics.get("n_clusters_found")
                                            })
                                    if metrics_data_run:
                                        metrics_df_run = pd.DataFrame(metrics_data_run)
                                st.session_state.metrics_df_for_display = metrics_df_run.copy() if metrics_df_run is not None else None
                                
                                st.session_state.show_clustering_results_section = True

                                # Save analysis run (uses variables from this run's scope)
                                current_username_log = auth.get_username()
                                if current_username_log and save_analysis_run:
                                    try: 
                                        processor.log_dataset_processing(current_username_log, st.session_state.active_file_name, df.shape, cleaning_summary, [alg for alg, r_alg in results_run.items() if r_alg])
                                        analysis_run_data = {
                                            "file_name": st.session_state.active_file_name, 
                                            "original_shape": df.shape,
                                            "cleaning_summary": cleaning_summary, 
                                            "preprocessor_details": preprocessor_details_run,
                                            "algorithms_run": selected_algorithms_from_ui,
                                            "results": { algo_r: {
                                                    "metrics": r_item.get("metrics"),
                                                    "visualization_json": r_item.get("visualization").to_json() if r_item.get("visualization") else None,
                                                } for algo_r, r_item in results_run.items() if r_item },
                                            "metrics_summary_df_json": metrics_df_run.to_json(orient='split') if metrics_df_run is not None else None 
                                        }
                                        save_analysis_run(current_username_log, analysis_run_data)
                                        st.success("Analysis run saved to history.")
                                    except Exception as e: 
                                        st.warning(f"Failed to log or save analysis run: {e}")
                                        logger.error(f"Error saving analysis run for {current_username_log}: {e}", exc_info=True)
                                # st.rerun() # Consider if needed, usually not. UI will update with new session state.

                else: # df_cleaned is None or empty
                    st.error("Data cleaning failed or resulted in an empty dataset.")
                    st.session_state.show_clustering_results_section = False
            else: # df is None or empty
                st.error("Failed to load data or uploaded data is empty.")
                st.session_state.show_clustering_results_section = False

        # --- RESULTS DISPLAY SECTION ---
        # This section is rendered if show_clustering_results_section is True,
        # drawing all data from session_state.
        if st.session_state.get('show_clustering_results_section', False):
            results_to_display = st.session_state.get('clustering_results')
            metrics_df_to_display = st.session_state.get('metrics_df_for_display')
            df_cleaned_from_state = st.session_state.get('df_cleaned_for_download')
            active_file_name_from_state = st.session_state.get('active_file_name', "N/A")
            # algorithms_run_for_display_from_state = st.session_state.get('algorithms_run_for_display', [])


            if results_to_display and any(res for res in results_to_display.values() if res is not None):
                if metrics_df_to_display is not None and not metrics_df_to_display.empty:
                    st.subheader("Clustering Metrics Comparison")
                    st.dataframe(metrics_df_to_display.round(3))
                    
                    def create_metric_bar_chart(m_df, y_col_df, t_chart):
                        if y_col_df in m_df.columns and not m_df[y_col_df].isnull().all():
                            st.plotly_chart(px.bar(m_df,x="Algorithm",y=y_col_df,color="Algorithm",title=t_chart), use_container_width=True, key=f"chart_disp_{y_col_df.lower().replace(' ','_')}")
                    
                    chart_cols_titles = [
                        ("Silhouette Score", "Silhouette Score Comparison (higher is better)"),
                        ("Calinski-Harabasz Score", "Calinski-Harabasz Score Comparison (higher is better)"),
                        ("Davies-Bouldin Score", "Davies-Bouldin Score Comparison (lower is better)"),
                        ("Number of Clusters Found", "Number of Clusters Found")]
                    for score_col, score_title in chart_cols_titles:
                        create_metric_bar_chart(metrics_df_to_display, score_col, score_title)

                    st.subheader("Algorithm Performance Radar Chart")
                    all_algorithms_radar = metrics_df_to_display['Algorithm'].unique().tolist()
                    # Ensure radar select default is robust
                    default_radar_selection = [algo for algo in st.session_state.get("radar_algo_select_val", all_algorithms_radar) if algo in all_algorithms_radar]
                    if not default_radar_selection and all_algorithms_radar: default_radar_selection = all_algorithms_radar
                    
                    selected_algorithms_for_radar = st.multiselect("Select algorithms for Radar Chart:", all_algorithms_radar, default=default_radar_selection, key="radar_algo_select_val")
                    if selected_algorithms_for_radar:
                        try:
                            radar_fig = processor.create_radar_chart(metrics_df_to_display, selected_algorithms_for_radar)
                            if radar_fig: st.plotly_chart(radar_fig, use_container_width=True, key="radar_chart_comparison_disp")
                            else: st.info("Radar chart could not be generated.")
                        except Exception as e: st.warning(f"Could not generate radar chart: {e}")
                    else: st.info("Select algorithms to display radar chart.")

                st.subheader("Cluster Visualizations")
                viz_count_disp = 0
                for algo_disp, res_item_disp in results_to_display.items():
                    if res_item_disp and res_item_disp.get("visualization"):
                        st.write(f"### {algo_disp}"); st.plotly_chart(res_item_disp["visualization"], use_container_width=True, key=f"viz_disp_{algo_disp}"); viz_count_disp+=1
                if viz_count_disp == 0: st.info("No visualizations available.")
                
                # --- DOWNLOAD REPORTS SECTION (NOW STABLE) ---
                st.subheader("Download Reports")
                if metrics_df_to_display is not None and not metrics_df_to_display.empty:
                    csv_metrics = metrics_df_to_display.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download Metrics Table (CSV)", data=csv_metrics, file_name='clustering_metrics.csv', mime='text/csv', key='download_metrics_csv_disp')

                # Download section for labeled data
                if results_to_display and df_cleaned_from_state is not None:
                    download_algo_options = [
                        algo_dl for algo_dl in results_to_display.keys()
                        if results_to_display[algo_dl] and
                           'labels' in results_to_display[algo_dl] and
                           results_to_display[algo_dl]['labels'] is not None
                    ]

                    if download_algo_options:
                        # Gracefully handle default for selectbox if previous selection is no longer valid
                        current_selection = st.session_state.get('select_algo_download_labels')
                        if current_selection not in download_algo_options:
                            st.session_state.select_algo_download_labels = download_algo_options[0] if download_algo_options else None
                        
                        selected_algo_for_download = st.selectbox(
                            "Select algorithm for downloading labeled data:",
                            options=download_algo_options,
                            key='select_algo_download_labels' # Key makes its state persistent
                        )
                        
                        if selected_algo_for_download and selected_algo_for_download in results_to_display:
                            df_labels_base = df_cleaned_from_state.copy()
                            labels_for_df = results_to_display[selected_algo_for_download]['labels']
                            
                            if len(labels_for_df) == len(df_labels_base):
                                df_labels_base[f'{selected_algo_for_download}_Cluster'] = labels_for_df
                                csv_labeled_data = df_labels_base.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label=f"Download Cleaned Data with {selected_algo_for_download} Labels (CSV)",
                                    data=csv_labeled_data,
                                    file_name=f'cleaned_data_{selected_algo_for_download}_labels.csv',
                                    mime='text/csv',
                                    key=f'download_labeled_data_csv_disp_{selected_algo_for_download}' # Dynamic key ensures button updates
                                )
                            else:
                                st.warning(f"Could not prepare labeled data for {selected_algo_for_download}: Label count ({len(labels_for_df)}) mismatch with data rows ({len(df_labels_base)}). Please re-run analysis.")
                    else:
                        st.info("No algorithm results with labels are available for download from the current analysis.")
                elif not results_to_display:
                    st.info("Run clustering analysis to enable downloads for labeled data.")
                elif df_cleaned_from_state is None:
                    st.warning("Cleaned data for download is missing. Please re-run the clustering analysis.")

                # Full interactive report (HTML)
                if metrics_df_to_display is not None: # Ensure metrics_df is available for HTML report
                    st.markdown("--- \n**Interactive Report (HTML)**")
                    metrics_table_html_content = metrics_df_to_display.to_html(index=False, classes="dataframe")
                    visualizations_html_content_parts = []
                    for algo_html, res_item_html in results_to_display.items():
                        if res_item_html and res_item_html.get("visualization"):
                            try:
                                fig_for_html = res_item_html["visualization"]
                                plot_div = fig_for_html.to_html(full_html=False, include_plotlyjs='cdn')
                                visualizations_html_content_parts.append(f"<div class=\"chart-container\"><h3>{algo_html} Clustering Visualization</h3>{plot_div}</div>\n")
                            except Exception as e:
                                logger.error(f"Error embedding HTML viz for {algo_html}: {e}", exc_info=True)
                                visualizations_html_content_parts.append(f"<p>Could not embed visualization for {algo_html}: {e}</p>\n")
                    visualizations_html_final_content = "".join(visualizations_html_content_parts)
                    current_timestamp_html = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Use active_file_name_from_state for file name in report
                    full_report_html_str = f"""<!DOCTYPE html> 
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clustering Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"; margin: 0; padding: 0; background-color: #f0f2f6; color: #333; }}
        .container {{ max-width: 1100px; margin: 30px auto; background-color: #ffffff; padding: 25px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #1a59a7; }} /* A deep blue for headers */
        h1 {{ text-align: center; border-bottom: 2px solid #1a59a7; padding-bottom: 15px; margin-bottom: 20px; font-size: 2em; }}
        h2 {{ border-bottom: 1px solid #dfe3e8; padding-bottom: 8px; margin-top: 35px; font-size: 1.6em; }}
        h3 {{ font-size: 1.3em; margin-top: 25px; color: #3385d9; }} /* Lighter blue for sub-headers */
        .dataframe {{ margin-bottom: 25px; border-collapse: collapse; width: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.05); overflow-x: auto; }}
        .dataframe th, .dataframe td {{ border: 1px solid #e1e4e8; padding: 12px 15px; text-align: left; font-size: 0.95em; }}
        .dataframe th {{ background-color: #2c7be5; color: white; font-weight: 600; }} /* Primary button blue */
        .dataframe tr:nth-child(even) {{ background-color: #f6f8fa; }}
        .dataframe tr:hover {{ background-color: #f1f8ff; }}
        .chart-container {{ margin-bottom: 35px; border: 1px solid #dfe3e8; padding: 20px; border-radius: 8px; background-color: #fbfcfd; box-shadow: 0 2px 5px rgba(0,0,0,0.03); }}
        p {{ line-height: 1.7; font-size: 1em; }}
        .report-meta p {{ margin-bottom: 5px; font-size: 0.9em; color: #586069; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Clustering Analysis Report</h1>
        <div class="report-meta">
            <p><strong>File:</strong> {active_file_name_from_state}</p>
            <p><strong>Generated on:</strong> {current_timestamp_html}</p>
        </div>
        <h2>Metrics Summary</h2>
        {metrics_table_html_content}
        <h2>Clustering Visualizations</h2>
        {visualizations_html_final_content}
    </div>
</body>
</html>
                    """
                    full_report_html_bytes = full_report_html_str.encode('utf-8')
                    st.download_button(label="Download Interactive Report (HTML)", data=full_report_html_bytes, file_name='clustering_report.html', mime='text/html', key='download_html_report_disp')
            else: # No results were generated or are available to display
                st.info("No clustering results available to display or download.")
        
        # This handles the case where there's an active file but results haven't been generated yet
        # (e.g., after uploading a file but before clicking "Run Clustering Analysis")
        # No specific message needed here as the UI for "Run Clustering Analysis" will be visible.

        elif uploaded_file is None and st.session_state.active_file_content is None: 
            # This is the initial state or if the user somehow clears the file without uploading a new one.
            st.info("Please upload a CSV or Excel file to begin analysis.")
            # Clear all analysis-related session state if no file is active.
            for key_to_clear in ['active_file_content', 'active_file_name', 'clustering_results', 
                                 'df_cleaned_for_download', 'metrics_df_for_display', 
                                 'show_clustering_results_section', 'preprocessor_details_for_display',
                                 'original_shape_for_display', 'cleaning_summary_for_display',
                                 'algorithms_run_for_display', 'select_algo_download_labels']:
                st.session_state.pop(key_to_clear, None)


    except Exception as e:
        st.error(f"An unexpected error occurred on the data pipeline page: {e}")
        logger.error(f"Error in render_data_pipeline_page: {e}", exc_info=True)


def render_analysis_history_page():
    st.title("Analysis History")
    auth = Authentication()
    username = auth.get_username()
    if not username:
        st.warning("Please login to view your analysis history."); return

    if not get_user_analysis_history:
        st.error("Analysis history function not available."); return
        
    history_runs = get_user_analysis_history(username) 
    if not history_runs:
        st.info("No analysis history found."); return

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        run_options = {f"{run['timestamp']} - {run.get('file_name', 'N/A')}": run['_id'] for run in history_runs}
        selected_run_display = st.selectbox("Select an Analysis Run to View:", options=list(run_options.keys()), key="history_run_select")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space
        if st.button("Clear All History", key="clear_all_history_button"): # Add clear all history button
            if st.session_state.get("username"): # Ensure user is logged in
                if clear_user_analysis_history: # Check if function exists
                    clear_user_analysis_history(st.session_state.username)
                    st.success("All analysis history cleared!")
                    st.rerun()
                else:
                    st.error("Clear history function not available.")


    if selected_run_display and get_analysis_run_details:
        selected_run_id = run_options[selected_run_display]
        
        run_details = None # Initialize
        # Button to delete selected analysis
        # Place delete button before displaying details, so it can act and rerun
        if st.button("Delete Selected Analysis", key=f"delete_selected_analysis_button_{selected_run_id}"): # Unique key per run
            if st.session_state.get("username") and selected_run_id:
                if delete_user_analysis_run: # Check if function exists
                    delete_user_analysis_run(st.session_state.username, selected_run_id)
                    st.success(f"Analysis run {selected_run_display} deleted!") # Use display name for success message
                    st.rerun() # Rerun to refresh history list
                else:
                    st.error("Delete analysis run function not available.")
            return # Stop further processing for this run if deleted

        run_details = get_analysis_run_details(selected_run_id) # Get details after potential delete action

        if run_details:
            st.subheader(f"Details for Analysis on {run_details.get('file_name', 'N/A')} at {run_details['timestamp']}")
            st.write("**Original File Shape:**", str(run_details.get('original_shape')))
            st.write("**Cleaning Summary:**"); st.json(run_details.get('cleaning_summary', {}))
            st.write("**Preprocessor Details:**"); st.json(run_details.get('preprocessor_details', {}))
            st.write("**Algorithms Run:**", ', '.join(run_details.get('algorithms_run', [])))

            metrics_df_history = None
            if run_details.get('metrics_summary_df_json'):
                st.subheader("Metrics Summary")
                try:
                    metrics_df_history = pd.read_json(run_details['metrics_summary_df_json'], orient='split')
                    st.dataframe(metrics_df_history.round(3))
                    csv_data = metrics_df_history.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Metrics as CSV",
                        data=csv_data,
                        file_name=f"metrics_summary_{selected_run_id}.csv",
                        mime="text/csv",
                        key=f"download_metrics_csv_hist_{selected_run_id}"
                    )
                except Exception as e: st.warning(f"Could not load metrics summary table: {e}")
            
            st.subheader("Visualizations")
            results_history = run_details.get('results', {})
            viz_count_history = 0
            for algo_hist, res_item_hist in results_history.items():
                if res_item_hist and res_item_hist.get("visualization_json"):
                    st.write(f"#### {algo_hist}")
                    try:
                        fig_json_hist = json.loads(res_item_hist["visualization_json"])
                        fig_history = go.Figure(fig_json_hist)
                        st.plotly_chart(fig_history, use_container_width=True, key=f"history_viz_{algo_hist}_{selected_run_id}")
                        viz_count_history += 1
                    except Exception as e:
                        st.warning(f"Could not render visualization for {algo_hist}: {e}")
                        logger.error(f"Error rendering history viz for {algo_hist}, run {selected_run_id}: {e}", exc_info=True)
            if viz_count_history == 0: st.info("No visualizations found for this analysis run.")

            # Prepare HTML content for download
            # Simplified HTML content generation logic from history
            html_report_parts = [
                f"<!DOCTYPE html><html><head><title>Analysis Report: {run_details.get('file_name', 'N/A')}</title></head><body>",
                f"<h1>Analysis Report: {run_details.get('file_name', 'N/A')}</h1>",
                f"<p>Timestamp: {run_details.get('timestamp', 'N/A')}</p>",
            ]
            if metrics_df_history is not None:
                html_report_parts.append("<h2>Metrics Summary</h2>")
                html_report_parts.append(metrics_df_history.to_html(index=False))
            
            preprocessor_details_hist = run_details.get('preprocessor_details', {})
            if preprocessor_details_hist:
                html_report_parts.append("<h2>Preprocessor Details</h2>")
                html_report_parts.append(f"<pre>{json.dumps(preprocessor_details_hist, indent=2)}</pre>")

            algorithms_run_hist = run_details.get('algorithms_run', [])
            if algorithms_run_hist:
                 html_report_parts.append("<h2>Algorithms Run</h2>")
                 html_report_parts.append(f"<p>{', '.join(algorithms_run_hist)}</p>")
            
            # Placeholder for visualizations in HTML history report (actual embedding is more complex here)
            html_report_parts.append("<h2>Visualizations</h2><p>(Visualizations are viewable on the dashboard. Full interactive HTML report is available from the Data Pipeline page after a fresh run.)</p>")
            html_report_parts.append("</body></html>")
            full_html_content_hist = "\n".join(html_report_parts)

            st.download_button(
                label="Download Report Summary (HTML)",
                data=full_html_content_hist.encode('utf-8'),
                file_name=f"analysis_report_summary_{selected_run_id}.html",
                mime="text/html",
                key=f"download_html_hist_summary_{selected_run_id}"
            )

        elif not get_analysis_run_details: # This check is redundant if outer 'if' is true
            st.error("Function to get analysis run details is not available.")
        else: 
            st.error(f"Could not retrieve details for selected run: {selected_run_display}")


def render_user_dashboard():
    try:
        auth = Authentication() 
        if not auth.require_authentication(): return

        full_name = st.session_state.get("full_name", "User")
        role = st.session_state.get("role", "user")
        st.sidebar.title(f"Welcome, {full_name}")
        st.sidebar.write(f"Role: {role.capitalize()}")
        st.sidebar.divider()

        page_options = ["Data Pipeline", "Analysis History", "Profile"] # Changed order to make Data Pipeline default
        if 'dashboard_page' not in st.session_state or st.session_state.dashboard_page not in page_options:
            st.session_state.dashboard_page = "Data Pipeline" 
        
        current_page_idx = page_options.index(st.session_state.dashboard_page)
        selected_page = st.sidebar.radio("Navigation", page_options, key="user_dashboard_nav_radio", index=current_page_idx)
        
        if selected_page != st.session_state.dashboard_page:
            st.session_state.dashboard_page = selected_page
            st.rerun()

        if st.session_state.dashboard_page == "Profile": render_profile_page()
        elif st.session_state.dashboard_page == "Data Pipeline": render_data_pipeline_page()
        elif st.session_state.dashboard_page == "Analysis History": render_analysis_history_page()
        
        st.sidebar.divider()
        if st.sidebar.button("Logout", key="sidebar_logout_button"): auth.logout()
    except Exception as e:
        st.error(f"An error occurred in the user dashboard: {e}")
        logger.error(f"Error in render_user_dashboard: {e}", exc_info=True)

# --- Mock Classes (for standalone testing) ---
class MockAuthentication:
    def __init__(self):
        for k, v_default in {'authenticated': False, 'username': "testuser", 'full_name': "Test User", 'email': "test@example.com", 'role': "user"}.items():
            if k not in st.session_state: st.session_state[k] = v_default
    def require_authentication(self):
        if not st.session_state.get('authenticated'):
            st.info("Mock: Please login.")
            if st.button("Mock Login", key="mock_login_button"):
                st.session_state.authenticated = True
                st.session_state.update({'username':"mockuser",'full_name':"Mock User",'email':"mock@example.com",'role':"tester"})
                st.rerun()
            return False
        return True
    def get_user_info(self): return {k:st.session_state.get(k) for k in ['username','full_name','email','role'] if k in st.session_state} if st.session_state.get('authenticated') else None
    def get_username(self): return st.session_state.get("username") if st.session_state.get('authenticated') else None
    def logout(self):
        st.session_state.authenticated = False
        st.session_state.dashboard_page = "Data Pipeline" # Reset to default page
        for k in ['username','full_name','email','role', 
                  'active_file_content', 'active_file_name', 'clustering_results', 
                  'df_cleaned_for_download', 'metrics_df_for_display', 
                  'show_clustering_results_section', # Clear analysis states on logout
                  'select_algo_download_labels'
                  ]: 
            st.session_state.pop(k, None)
        st.success("Mock: Logged out."); st.rerun()

class MockDataProcessor:
    def load_data(self, file_like_object): # Expects a file-like object (e.g., BytesIO or UploadedFile)
        # Simple mock: if it has a name and it's typical, create a DataFrame
        if hasattr(file_like_object, 'name') and (file_like_object.name.endswith(".csv") or file_like_object.name.endswith(".xls") or file_like_object.name.endswith(".xlsx")):
            # In a real scenario, you'd use pd.read_csv(file_like_object) or pd.read_excel(file_like_object)
            # For mock, just return a fixed DataFrame
            return pd.DataFrame({'A':[1,2,1,2,1]*4,'B':[3,4,3,4,4]*4,'C':[5,6,5,6,5]*4,'D':[7,8,7,8,8]*4, 'E':[1,1,2,2,1]*4})
        return None
        
    def clean_data(self, df): 
        if df is None or df.empty:
            return pd.DataFrame(), {"before":{"rows":0,"missing_values":{}},"after":{"rows":0},"duplicates_removed":0}
        # Simple mock clean: remove duplicates and fill NaNs if any (though mock data has no NaNs)
        initial_rows = df.shape[0]
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_rows - df_cleaned.shape[0]
        # Mock missing values summary (no actual missing values in mock data)
        missing_before = {col: df[col].isnull().sum() for col in df.columns if df[col].isnull().any()}

        return df_cleaned.copy(), {"before":{"rows":initial_rows,"missing_values":missing_before},"after":{"rows":df_cleaned.shape[0]},"duplicates_removed":duplicates_removed}

    def preprocess_data(self, df): 
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        import numpy # Import numpy for pd.np usage if needed, or just np
        if df is None or df.empty:
            return pd.DataFrame().to_numpy(), {"error": "Empty DataFrame for preprocessing"}

        numeric_cols = df.select_dtypes(include=numpy.number).columns
        if numeric_cols.empty:
             return pd.DataFrame().to_numpy(), {"error": "No numeric columns to process", "imputer":"None","scaler":"None","processed_columns": []}

        X_num = df[numeric_cols].copy()
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X_num)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        return X_scaled, {"imputer":"SimpleImputer(mean)","scaler":"StandardScaler","processed_columns": list(numeric_cols)}
        
    def apply_clustering(self, X, algo, k_cfg): 
        k_val=3
        if isinstance(k_cfg, str) and k_cfg.isdigit(): k_val = int(k_cfg)
        elif isinstance(k_cfg, int): k_val = k_cfg
        
        if X is None or X.shape[0] == 0: return [], {"name":algo,"params":{"n_clusters":k_val}, "error": "Empty data for clustering"}
        # Ensure k_val is at least 1 for modulo, and sensible for clustering
        k_val = max(1, k_val)
        if X.shape[0] < k_val and k_val > 1: # If less data points than clusters, assign all to cluster 0 or handle as error
            # This mock will just assign all to 0 if k_val > num_samples. A real implementation might error or adjust k.
             return [0 for _ in range(X.shape[0])], {"name":algo,"params":{"n_clusters":k_val, "adjusted_k": 1 if X.shape[0]>0 else 0}}


        return [i%k_val for i in range(X.shape[0])], {"name":algo,"params":{"n_clusters":k_val}}
        
    def evaluate_clustering(self, X, lbls): 
        num_unique_labels = len(set(l for l in lbls if l != -1)) # Exclude noise label if present
        if not lbls or X is None or X.shape[0] < 2 or num_unique_labels < 2 : # Need at least 2 clusters for most metrics
            return {"silhouette_score":None,"calinski_harabasz_score":None,"davies_bouldin_score":None,"n_clusters_found":num_unique_labels}
        return {"silhouette_score":0.55,"calinski_harabasz_score":150.0,"davies_bouldin_score":0.8,"n_clusters_found":num_unique_labels}
    
    def visualize_clusters(self, X, lbls, algo):
        title_suffix = " (Mock Data)"
        template_to_use = "plotly" 

        if X is None or X.shape[0] == 0 or lbls is None or len(lbls) == 0:
            fig = go.Figure()
            fig.update_layout(title=f"{algo} Clusters{title_suffix} - No Data/Labels", template=template_to_use)
            return fig

        # Ensure labels are strings for categorical coloring
        str_lbls = [str(l) for l in lbls]

        if X.shape[1] >= 2:
            fig = px.scatter(x=X[:, 0], y=X[:, 1], color=str_lbls, title=f"{algo} Clusters{title_suffix}")
        elif X.shape[1] == 1:
            fig = px.scatter(x=range(len(X)), y=X[:, 0], color=str_lbls, title=f"{algo} Clusters (1D){title_suffix}")
        else: 
            fig = go.Figure()
            fig.update_layout(title=f"{algo} Clusters{title_suffix} - Not enough dimensions for scatter plot", template=template_to_use)
            return fig
            
        fig.update_layout(template=template_to_use) 
        return fig
        
    def get_cache_key(self, file_content_bytes, algorithm_name_str, n_clusters_config_str, preprocessor_details_string):
        import hashlib
        if not isinstance(file_content_bytes, bytes): # Ensure it's bytes
            file_content_bytes = str(file_content_bytes).encode('utf-8')
        content_hash = hashlib.md5(file_content_bytes).hexdigest(); combined = f"content:{content_hash}_algo:{algorithm_name_str}_k_config:{n_clusters_config_str}_preprocess:{preprocessor_details_string}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def load_from_cache(self, key): logger.info(f"Mock: Cache load attempt for {key}"); return None # Mock never finds in cache
    def save_to_cache(self, key, data): logger.info(f"Mock: Cache save for {key}") # Mock save does nothing
    
    def create_radar_chart(self, df_m, selected_algorithms_for_radar_input):
        if df_m is None or df_m.empty or not selected_algorithms_for_radar_input: return None
        df_m_filtered = df_m[df_m['Algorithm'].isin(selected_algorithms_for_radar_input)].copy()

        if not df_m_filtered.empty and "Algorithm" in df_m_filtered.columns:
            metric_cols_for_radar = ["Silhouette Score", "Calinski-Harabasz Score", "Davies-Bouldin Score"]
            available_metric_cols = [col for col in metric_cols_for_radar if col in df_m_filtered.columns and df_m_filtered[col].notna().any()]
            if not available_metric_cols: return None

            df_r = df_m_filtered.set_index("Algorithm")[available_metric_cols].reset_index()
            for col in available_metric_cols: df_r[col] = pd.to_numeric(df_r[col], errors='coerce')
            df_r.dropna(subset=available_metric_cols, how='all', inplace=True) # Drop rows where all radar metrics are NaN
            if df_r.empty: return None

            df_r_melted = df_r.melt(id_vars="Algorithm",var_name="Metric",value_name="Score")
            df_r_melted.dropna(subset=['Score'], inplace=True) # Drop specific metric scores that are NaN

            if not df_r_melted.empty:
                # Normalize scores per metric (0 to 1)
                for metric_name in df_r_melted["Metric"].unique():
                    slc = df_r_melted["Metric"]==metric_name
                    scores_for_metric = df_r_melted.loc[slc, "Score"]
                    if not scores_for_metric.empty:
                        min_v, max_v = scores_for_metric.min(), scores_for_metric.max()
                        if pd.notna(min_v) and pd.notna(max_v):
                            if max_v > min_v : 
                                # Davies-Bouldin is lower is better, invert for radar visualization if desired (not done here, simple normalization)
                                df_r_melted.loc[slc,"Score"] = (scores_for_metric - min_v) / (max_v - min_v)
                            elif max_v == min_v and min_v != 0 : df_r_melted.loc[slc,"Score"] = 0.5 # All same non-zero value
                            else: df_r_melted.loc[slc,"Score"] = 0.0 # All zero or min=max=0
                        else: df_r_melted.loc[slc,"Score"] = 0.0 # If min/max are nan after to_numeric
                    else: df_r_melted.loc[slc,"Score"] = 0.0 # All scores for this metric were NaN
                
                if df_r_melted['Score'].notna().any(): # Check if any valid scores left after normalization
                    fig = px.line_polar(df_r_melted,r="Score",theta="Metric",color="Algorithm",line_close=True,title="Algorithm Radar (Mock, Normalized 0-1)")
                    fig.update_layout(template="plotly") 
                    return fig
        return None

    def log_dataset_processing(self,*args): logger.info(f"Mock: Logging dataset processing: {args}")

# Mock model functions
def mock_update_user(usr,data): logger.info(f"Mock: Updating user {usr} with {data}"); return True
def mock_save_analysis_run(username, analysis_data): logger.info(f"Mock: Saving analysis for {username}, file: {analysis_data.get('file_name')}"); return True

_mock_analysis_history_store = {} # Global store for mock history

def mock_get_user_analysis_history(username):
    logger.info(f"Mock: Getting history for {username}")
    if username not in _mock_analysis_history_store: # Initialize with some data if first time for this user
        base_time = datetime.datetime.now()
        # Create a mock figure for history (ensure MockDataProcessor is available)
        try:
            mock_fig_processor = MockDataProcessor()
            # Need some mock X and labels for visualize_clusters
            mock_X_hist = pd.DataFrame({'x':[1,2,3,4],'y':[3,1,2,5]}).to_numpy()
            mock_labels_hist = [0,1,0,1]
            mock_fig_data = mock_fig_processor.visualize_clusters(mock_X_hist, mock_labels_hist, "KMeans_Hist_Mock")
            mock_fig_json = mock_fig_data.to_json() if mock_fig_data else None
        except Exception as e:
            logger.error(f"Error creating mock figure for history: {e}")
            mock_fig_json = None

        _mock_analysis_history_store[username] = [
            {"_id": "mock_run_1", "timestamp": (base_time - datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), "file_name": "hist_data1.csv",
             "original_shape": [10,2], "cleaning_summary":{"before_rows":10, "after_rows":10}, "preprocessor_details":{"scaler":"Standard"}, 
             "algorithms_run":["KMeans_Hist_Mock"],
             "metrics_summary_df_json": pd.DataFrame({"Algorithm":["KMeans_Hist_Mock"],"Silhouette Score":[0.5]}).to_json(orient='split'),
             "results": {"KMeans_Hist_Mock": {"metrics":{"silhouette_score":0.5}, "visualization_json": mock_fig_json }}
             },
            # Add another mock run if desired
        ]
    return _mock_analysis_history_store.get(username, [])

def mock_get_analysis_run_details(run_id):
    logger.info(f"Mock: Getting details for {run_id}")
    for username_key in _mock_analysis_history_store: # Iterate through all users' histories
        for run in _mock_analysis_history_store[username_key]:
            if run['_id'] == run_id:
                return run
    return None

def mock_clear_user_analysis_history(username):
    logger.info(f"Mock: Clearing all analysis history for {username}")
    if username in _mock_analysis_history_store:
        _mock_analysis_history_store[username] = []
    return True # Indicate success

def mock_delete_user_analysis_run(username, run_id):
    logger.info(f"Mock: Deleting analysis run {run_id} for {username}")
    if username in _mock_analysis_history_store:
        initial_len = len(_mock_analysis_history_store[username])
        _mock_analysis_history_store[username] = [run for run in _mock_analysis_history_store[username] if run['_id'] != run_id]
        return len(_mock_analysis_history_store[username]) < initial_len # True if deleted
    return False


if __name__ == "__main__":
    use_mocks_for_main = False

    # Check if actual modules or their functions are available
    if Authentication is None or DataProcessor is None or \
       update_user is None or save_analysis_run is None or \
       get_user_analysis_history is None or get_analysis_run_details is None or \
       clear_user_analysis_history is None or delete_user_analysis_run is None: # Added new model functions
        use_mocks_for_main = True
        logger.warning("One or more critical components (Authentication, DataProcessor, model functions) are not available. Forcing Mocks for standalone run.")

    if use_mocks_for_main:
        st.sidebar.warning("RUNNING IN MOCK MODE")
        Authentication = MockAuthentication
        DataProcessor = MockDataProcessor
        update_user = mock_update_user
        save_analysis_run = mock_save_analysis_run
        get_user_analysis_history = mock_get_user_analysis_history
        get_analysis_run_details = mock_get_analysis_run_details
        clear_user_analysis_history = mock_clear_user_analysis_history
        delete_user_analysis_run = mock_delete_user_analysis_run
    else: # Ensure all functions are defined even if not using mocks (e.g. if some imports failed selectively)
        if Authentication is None: Authentication = MockAuthentication # Fallback if specific import failed
        if DataProcessor is None: DataProcessor = MockDataProcessor
        if update_user is None: update_user = mock_update_user
        if save_analysis_run is None: save_analysis_run = mock_save_analysis_run
        if get_user_analysis_history is None: get_user_analysis_history = mock_get_user_analysis_history
        if get_analysis_run_details is None: get_analysis_run_details = mock_get_analysis_run_details
        if clear_user_analysis_history is None: clear_user_analysis_history = mock_clear_user_analysis_history
        if delete_user_analysis_run is None: delete_user_analysis_run = mock_delete_user_analysis_run


    render_user_dashboard()
