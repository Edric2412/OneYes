# Streamlit Clustering App - Documentation

## Overview
This application is a Streamlit web application with role-based authentication (Admin/User) that allows users to upload raw datasets, run a data-cleaning and clustering pipeline, and compare clustering algorithm scores.

## Features

### Authentication
- **Login/Signup**: Secure login and signup functionality with password hashing
- **Role-based Access**: Different dashboards for Admin and User roles
- **Session Management**: Persistent user sessions

### User Dashboard
- **Profile Management**: View and edit personal details
- **Data Pipeline**:
  - File upload (CSV/Excel)
  - Data preview and cleaning summary
  - Preprocessing (scaling, encoding, imputation)
  - Multiple clustering algorithms (KMeans, Agglomerative, Hierarchial, Gaussian Mixture)
  - Evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
  - Results visualization and comparison
  - Caching to avoid recomputation
  - **Analysis History**: View, clear, and delete past analysis runs.
  - **Data Export**: Download metric tables as CSV and basic HTML reports.

### Admin Dashboard
- **User Management**: View, create, deactivate/reactivate, and delete user accounts
- **Upload Logs**: View and filter dataset processing logs with timestamps and summaries

## Technical Details
- **Framework**: Streamlit
- **Authentication**: Custom implementation with bcrypt for password hashing
- **Storage**: MongoDB for users, logs, and analysis history
- **Data Processing**: Pandas, scikit-learn
- **Visualization**: Plotly, Matplotlib
- **Caching**: Custom file-based caching system

## Project Structure
```
streamlit_app/
├── app.py                  # Main entry point
├── setup_database.py       # Database initialization script
├── todo.md                 # Project task list
├── data/                   # Data storage directory (for initial setup or temporary files)
├── cache/                  # Cache directory for clustering results
├── src/                    # Source code
│   ├── main.py             # Main application logic
│   ├── auth/               # Authentication module
│   │   └── authenticator.py
│   ├── database/           # Database module
│   │   ├── models.py       # Database interface
│   │   └── mongodb_storage.py # MongoDB storage implementation
├── cache/                  # Cache directory for clustering results
├── src/                    # Source code
│   ├── main.py             # Main application logic
│   ├── auth/               # Authentication module
│   │   └── authenticator.py
│   ├── database/           # Database module
│   │   ├── models.py       # Database interface
│   │   └── file_storage.py # File-based storage implementation
│   ├── pipeline/           # Data processing pipeline
│   │   └── data_processor.py
│   ├── pages/              # Application pages
│   │   ├── user_dashboard.py
│   │   ├── admin_dashboard.py
│   │   └── signup.py
│   └── utils/              # Utility functions
└── venv/                   # Virtual environment
```

## Setup and Running

### Prerequisites
- Python 3.11 or higher
- Streamlit and other dependencies (installed via requirements.txt)

### Installation
1. Clone the repository or extract the provided files
2. Navigate to the project directory
3. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Initialize the database:
   ```
   python setup_database.py
   ```

### Running the Application
1. Start the Streamlit server:
   ```
   streamlit run app.py
   ```
2. Access the application in your web browser at the provided URL (typically http://localhost:8501)

## Usage Guide

### For Users
1. **Login/Signup**:
   - Use the login page to access your account
   - New users can sign up using the signup link

2. **Profile Management**:
   - View and edit your profile information
   - Update your full name and email

3. **Data Pipeline**:
   - Upload a CSV or Excel file
   - View data preview and cleaning summary
   - Run clustering analysis with selected parameters
   - Compare clustering algorithm results
   - View visualizations of clusters

### For Admins
1. **User Management**:
   - View all user accounts
   - Create new users (including admin users)
   - Deactivate/reactivate user accounts
   - Delete user accounts

2. **Upload Logs**:
   - View all dataset processing logs
   - Filter logs by username and date range
   - View detailed information about each processing job

## Customization
- **Adding New Clustering Algorithms**: Extend the `apply_clustering` method in `data_processor.py`
- **Modifying User Fields**: Update the user model in `mongodb_storage.py` and related forms
- **Changing Visualization**: Modify the visualization code in `data_processor.py`

## Security Notes
- Passwords are hashed using bcrypt before storage
- User authentication is session-based
- Admin access is restricted to users with the admin role

## Troubleshooting
- **File Upload Issues**: Ensure the file is in CSV or Excel format
- **Clustering Errors**: Check that the dataset has numeric columns and no missing values
- **Login Problems**: Verify credentials or reset the database by deleting the data directory

## Future Enhancements
- Add more clustering algorithms
- Add export functionality for results
- Enhance visualization options

