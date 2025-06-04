from mongodb_storage import MongoDBStorage

# Initialize MongoDB storage
storage = MongoDBStorage()
storage.connect()
storage.create_admin_user('admin', 'admin123')

# MongoDB-compatible function names to maintain compatibility with existing code
def get_database():
    """
    Return a dummy database object for compatibility
    """
    return {"users": None, "processing_logs": None}

def create_user(username, password_hash, role, email, full_name):
    """
    Create a new user in the file storage
    """
    return storage.create_user(username, password_hash, role, email, full_name)

def get_user(username):
    """
    Retrieve user by username from file storage
    """
    return storage.get_user(username)

def get_all_users():
    """
    Retrieve all users from file storage
    """
    return storage.get_all_users()

def update_user(username, update_data):
    """
    Update user information in file storage
    """
    return storage.update_user(username, update_data)

def deactivate_user(username):
    """
    Deactivate a user account in file storage
    """
    return storage.deactivate_user(username)

def reactivate_user(username):
    """
    Reactivate a user account in file storage
    """
    return storage.reactivate_user(username)

def delete_user(username):
    """
    Delete a user from file storage
    """
    return storage.delete_user(username)

def log_dataset_processing(username, filename, dataset_size, cleaning_summary, algorithms_used):
    """
    Log dataset processing information to file storage
    """
    return storage.log_dataset_processing(username, filename, dataset_size, cleaning_summary, algorithms_used)

def get_all_processing_logs():
    """
    Retrieve all dataset processing logs from file storage
    """
    return storage.get_all_processing_logs()

def get_user_processing_logs(username):
    """
    Retrieve processing logs for a specific user from file storage
    """
    return storage.get_user_processing_logs(username)

def save_analysis_run(username, analysis_data):
    """
    Saves a single clustering analysis run.
    """
    return storage.save_analysis_run(username, analysis_data)

def get_user_analysis_history(username):
    """
    Retrieves all analysis runs for a specific user.
    """
    return storage.get_user_analysis_history(username)

def get_analysis_run_details(analysis_id):
    """
    Retrieves a specific analysis run by its ID.
    """
    return storage.get_analysis_run_details(analysis_id)

def clear_user_analysis_history(username):
    """
    Clears all analysis runs for a specific user.
    """
    return storage.clear_user_analysis_history(username)

def delete_user_analysis_run(username, analysis_id):
    """
    Deletes a specific analysis run for a user.
    """
    return storage.delete_user_analysis_run(username, analysis_id)
