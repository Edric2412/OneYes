import streamlit as st
import pandas as pd
import sys
import os
import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from authenticator import Authentication
from models import get_all_users, get_all_processing_logs, create_user, update_user, deactivate_user, reactivate_user, delete_user
import bcrypt

def render_user_management_page():
    """
    Render the user management page for admins
    """
    auth = Authentication()
    if not auth.require_authentication():
        return
    
    st.title("User Management")
    
    # Create tabs for different user management functions
    tab1, tab2 = st.tabs(["View/Edit Users", "Create New User"])
    
    with tab1:
        # Get all users
        users = get_all_users()
        
        if users:
            # Convert to DataFrame for display
            users_df = pd.DataFrame(users)
            
            # Format dates for display
            if 'created_at' in users_df.columns:
                users_df['created_at'] = users_df['created_at'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else '')
            if 'last_login' in users_df.columns:
                users_df['last_login'] = users_df['last_login'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else 'Never')
            
            # Display users table
            st.dataframe(users_df[['username', 'full_name', 'email', 'role', 'active', 'created_at', 'last_login']])
            
            # User actions
            st.subheader("User Actions")
            
            # Select user for actions
            selected_username = st.selectbox("Select User", [user['username'] for user in users])
            
            # Get selected user
            selected_user = next((user for user in users if user['username'] == selected_username), None)
            
            if selected_user:
                # Display user details
                st.write(f"**Username:** {selected_user['username']}")
                st.write(f"**Full Name:** {selected_user['full_name']}")
                st.write(f"**Email:** {selected_user['email']}")
                st.write(f"**Role:** {selected_user['role']}")
                st.write(f"**Status:** {'Active' if selected_user.get('active', True) else 'Inactive'}")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Activate/Deactivate button
                    if selected_user.get('active', True):
                        if st.button("Deactivate User"):
                            if deactivate_user(selected_user['username']):
                                st.success(f"User {selected_user['username']} deactivated successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to deactivate user")
                    else:
                        if st.button("Activate User"):
                            if reactivate_user(selected_user['username']):
                                st.success(f"User {selected_user['username']} activated successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to activate user")
                
                with col2:
                    # Confirmation checkbox for deletion
                    confirm_delete = st.checkbox(f"I confirm that I want to permanently delete {selected_user['username']}. This action cannot be undone.", key=f"confirm_delete_{selected_user['username']}")
                    
                    # Delete button
                    if st.button("Delete User", type="primary"):
                        if confirm_delete:
                            if delete_user(selected_user['username']):
                                st.success(f"User {selected_user['username']} deleted successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to delete user")
                        else:
                            st.warning("Please check the confirmation box to delete the user.")
        else:
            st.info("No users found")
    
    with tab2:
        # Create new user form
        st.subheader("Create New User")
        
        with st.form("create_user_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            role = st.selectbox("Role", ["user", "admin"])
            
            submit_button = st.form_submit_button("Create User")
            
            if submit_button:
                if username and password and confirm_password and full_name and email:
                    if password == confirm_password:
                        # Hash password
                        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                        
                        # Create user in database
                        success, message = create_user(username, hashed_password, role, email, full_name)
                        
                        if success:
                            st.success(f"User {username} created successfully!")
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")

def render_upload_logs_page():
    """
    Render the upload logs page for admins
    """
    auth = Authentication()
    if not auth.require_authentication():
        return
    
    st.title("Upload Logs")
    
    # Get all processing logs
    logs = get_all_processing_logs()
    
    if logs:
        # Convert to DataFrame for display
        logs_df = pd.DataFrame(logs)
        
        # Format dates for display
        if 'timestamp' in logs_df.columns:
            logs_df['timestamp'] = logs_df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else '')
        
        # Format dataset size
        if 'dataset_size' in logs_df.columns:
            logs_df['dataset_size'] = logs_df['dataset_size'].apply(lambda x: f"{x[0]} rows × {x[1]} columns" if isinstance(x, tuple) else str(x))
        
        # Format algorithms used
        if 'algorithms_used' in logs_df.columns:
            logs_df['algorithms_used'] = logs_df['algorithms_used'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        
        # Add filters
        st.subheader("Filters")
        
        # Username filter
        usernames = ['All'] + sorted(logs_df['username'].unique().tolist())
        selected_username = st.selectbox("Filter by Username", usernames)
        
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.datetime.now() - datetime.timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.datetime.now())
        
        # Apply filters
        filtered_df = logs_df.copy()
        
        if selected_username != 'All':
            filtered_df = filtered_df[filtered_df['username'] == selected_username]
        
        if 'timestamp' in filtered_df.columns:
            filtered_df['date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
            filtered_df = filtered_df[
                (filtered_df['date'] >= start_date) & 
                (filtered_df['date'] <= end_date)
            ]
            filtered_df = filtered_df.drop(columns=['date'])
        
        # Display filtered logs
        st.subheader("Processing Logs")
        st.dataframe(filtered_df[['username', 'filename', 'dataset_size', 'algorithms_used', 'timestamp']])
        
        # Display statistics
        st.subheader("Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Uploads", len(filtered_df))
        with col2:
            st.metric("Unique Users", filtered_df['username'].nunique())
        with col3:
            st.metric("Most Active User", filtered_df['username'].value_counts().idxmax() if not filtered_df.empty else "N/A")
        
        # Display log details
        st.subheader("Log Details")
        
        # Select log for details
        if not filtered_df.empty:
            selected_log_index = st.selectbox("Select Log", range(len(filtered_df)), format_func=lambda i: f"{filtered_df.iloc[i]['username']} - {filtered_df.iloc[i]['filename']} ({filtered_df.iloc[i]['timestamp']})")
            
            # Get selected log
            selected_log = filtered_df.iloc[selected_log_index]
            
            # Display log details
            st.write(f"**Username:** {selected_log['username']}")
            st.write(f"**Filename:** {selected_log['filename']}")
            st.write(f"**Dataset Size:** {selected_log['dataset_size']}")
            st.write(f"**Algorithms Used:** {selected_log['algorithms_used']}")
            st.write(f"**Timestamp:** {selected_log['timestamp']}")
            
            # Display cleaning summary if available
            if 'cleaning_summary' in selected_log and selected_log['cleaning_summary'] is not None:
                st.subheader("Cleaning Summary")
                
                cleaning_summary = selected_log['cleaning_summary']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Rows", cleaning_summary.get('before', {}).get('rows', 'N/A'))
                with col2:
                    st.metric("Cleaned Rows", cleaning_summary.get('after', {}).get('rows', 'N/A'))
                with col3:
                    st.metric("Duplicates Removed", cleaning_summary.get('duplicates_removed', 'N/A'))
    else:
        st.info("No upload logs found")

def render_admin_dashboard():
    """
    Render the admin dashboard with navigation
    """
    auth = Authentication()
    if not auth.require_authentication():
        return
    
    # Sidebar navigation
    st.sidebar.title(f"Welcome, {st.session_state.full_name}")
    st.sidebar.write(f"Role: {st.session_state.role.capitalize()}")
    
    # Navigation options
    page = st.sidebar.radio("Navigation", ["User Management", "Upload Logs"])
    
    # Render selected page
    if page == "User Management":
        render_user_management_page()
    elif page == "Upload Logs":
        render_upload_logs_page()
    
    # Add logout button to sidebar
    if st.sidebar.button("Logout"):
        auth.logout()
