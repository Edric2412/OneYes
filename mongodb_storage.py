import os
import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from bson.objectid import ObjectId
import logging
import bcrypt
import streamlit as st

logger = logging.getLogger(__name__)

class MongoDBStorage:
    """
    MongoDB storage implementation for user management and logging.
    """
    def __init__(self):
        self.client = None
        self.db = None
        self.connect()

    def connect(self):
        """
        Establishes connection to MongoDB.
        """
        try:
            # Get MongoDB URI from Streamlit secrets
            mongo_uri = st.secrets["MONGO_URI"]
            self.client = MongoClient(mongo_uri)
            self.client.admin.command('ping') # Test connection
            self.db = self.client['cluster_app_db'] # Database name
            logger.info("Successfully connected to MongoDB.")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            self.client = None
            self.db = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during MongoDB connection: {e}")
            self.client = None
            self.db = None

    def _get_users_collection(self, role='user'):
        if self.db is not None:
            if role == 'admin':
                return self.db['admins']
            else:
                return self.db['users']
        return None

    def _get_logs_collection(self):
        if self.db is not None:
            return self.db['processing_logs']
        return None

    def _get_analysis_history_collection(self):
        if self.db is not None:
            return self.db['analysis_history']

    def clear_user_analysis_history(self, username):
        """
        Clears all analysis runs for a specific user.
        """
        analysis_history_collection = self._get_analysis_history_collection()
        if analysis_history_collection is None:
            return False
        try:
            result = analysis_history_collection.delete_many({"username": username})
            return result.deleted_count > 0
        except OperationFailure as e:
            logger.error(f"Failed to clear analysis history for {username}: {e}")
            return False

    def delete_user_analysis_run(self, username, analysis_id):
        """
        Deletes a specific analysis run for a user.
        """
        analysis_history_collection = self._get_analysis_history_collection()
        if analysis_history_collection is None:
            return False
        try:
            result = analysis_history_collection.delete_one({"username": username, "_id": ObjectId(analysis_id)})
            return result.deleted_count > 0
        except OperationFailure as e:
            logger.error(f"Failed to delete analysis run {analysis_id} for {username}: {e}")
            return False
        return None

    def create_user(self, username, password_hash, role, email, full_name):
        """
        Creates a new user in MongoDB.
        """
        users_collection = self._get_users_collection(role)
        if users_collection is None:
            return False, "Database not connected."

        if users_collection.find_one({"username": username}):
            return False, "Username already exists."

        user_data = {
            "username": username,
            "password": password_hash,
            "role": role,
            "email": email,
            "full_name": full_name,
            "active": True,
            "created_at": datetime.datetime.now(),
            "last_login": None
        }
        try:
            result = users_collection.insert_one(user_data)
            return True, str(result.inserted_id)
        except OperationFailure as e:
            logger.error(f"Failed to create user {username} in MongoDB: {e}")
            return False, f"Database operation failed: {e}"

    def get_user(self, username, role=None):
        """
        Retrieves a user by username from MongoDB.
        If role is specified, it searches in that specific collection.
        Otherwise, it searches in both 'users' and 'admins' collections.
        """
        if role:
            users_collection = self._get_users_collection(role)
            if users_collection is None:
                return None
            return users_collection.find_one({"username": username})
        else:
            # Search in users collection first
            users_collection = self._get_users_collection('user')
            if users_collection is not None:
                user = users_collection.find_one({"username": username})
                if user:
                    return user

            # If not found, search in admins collection
            admins_collection = self._get_users_collection('admin')
            if admins_collection is not None:
                admin_user = admins_collection.find_one({"username": username})
                if admin_user:
                    return admin_user
            return None

    def get_all_users(self):
        """
        Retrieves all users from MongoDB, excluding password hashes.
        """
        all_users = []
        users_collection = self._get_users_collection('user')
        if users_collection is not None:
            all_users.extend(list(users_collection.find({}, {"password": 0})))

        admins_collection = self._get_users_collection('admin')
        if admins_collection is not None:
            all_users.extend(list(admins_collection.find({}, {"password": 0})))

        return all_users

    def update_user(self, username, update_data, role=None):
        """
        Updates user information in MongoDB.
        """
        if role:
            users_collection = self._get_users_collection(role)
            if users_collection is None:
                return False
            try:
                result = users_collection.update_one(
                    {"username": username},
                    {"$set": update_data}
                )
                return result.modified_count > 0
            except OperationFailure as e:
                logger.error(f"Failed to update user {username} in MongoDB: {e}")
                return False
        else:
            # Try updating in users collection first
            users_collection = self._get_users_collection('user')
            if users_collection is not None:
                result = users_collection.update_one(
                    {"username": username},
                    {"$set": update_data}
                )
                if result.modified_count > 0:
                    return True

            # If not found or not modified, try in admins collection
            admins_collection = self._get_users_collection('admin')
            if admins_collection is not None:
                result = admins_collection.update_one(
                    {"username": username},
                    {"$set": update_data}
                )
                return result.modified_count > 0
            return False

    def deactivate_user(self, username, role=None):
        """
        Deactivates a user account in MongoDB.
        """
        return self.update_user(username, {"active": False}, role)

    def reactivate_user(self, username, role=None):
        """
        Reactivates a user account in MongoDB.
        """
        return self.update_user(username, {"active": True}, role)

    def delete_user(self, username, role=None):
        """
        Deletes a user from MongoDB.
        """
        if role:
            users_collection = self._get_users_collection(role)
            if users_collection is None:
                return False
            try:
                result = users_collection.delete_one({"username": username})
                return result.deleted_count > 0
            except OperationFailure as e:
                logger.error(f"Failed to delete user {username} from MongoDB: {e}")
                return False
        else:
            # Try deleting from users collection first
            users_collection = self._get_users_collection('user')
            if users_collection is not None:
                result = users_collection.delete_one({"username": username})
                if result.deleted_count > 0:
                    return True

            # If not found or not deleted, try in admins collection
            admins_collection = self._get_users_collection('admin')
            if admins_collection is not None:
                result = admins_collection.delete_one({"username": username})
                return result.deleted_count > 0
            return False

    def create_admin_user(self, username, password):
        """
        Creates an admin user if one does not already exist.
        """
        admin_user = self.get_user(username, role='admin')
        if not admin_user:
            hashed_password = self.hash_password(password)
            self.create_user(username, hashed_password, role='admin', email=f"{username}@example.com", full_name=f"Admin {username.capitalize()}")
            logger.info(f"Admin user '{username}' created.")
        else:
            logger.info(f"Admin user '{username}' already exists.")

    def hash_password(self, password):
        """
        Hashes a password using bcrypt.
        """
        # Generate a salt and hash the password
        # The salt is automatically included in the hash
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def log_dataset_processing(self, username, filename, dataset_size, cleaning_summary, algorithms_used):
        logs_collection = self._get_logs_collection()
        if logs_collection is None:
            logger.warning("Attempted to log dataset processing, but database is not connected.")
            return False, "Database not connected."

        log_entry = {
            "username": username,
            "filename": filename,
            "dataset_size": dataset_size,
            "cleaning_summary": cleaning_summary,
            "algorithms_used": algorithms_used,
            "timestamp": datetime.datetime.now()
        }
        try:
            result = logs_collection.insert_one(log_entry)
            return True, str(result.inserted_id)
        except OperationFailure as e:
            logger.error(f"Failed to log dataset processing for user {username} in MongoDB: {e}")
            return False, f"Database operation failed: {e}"

    def get_all_processing_logs(self):
        logs_collection = self._get_logs_collection()
        if logs_collection is None:
            return []
        return list(logs_collection.find().sort("timestamp", -1))

    def get_user_processing_logs(self, username):
        logs_collection = self._get_logs_collection()
        if logs_collection is None:
            return []
        return list(logs_collection.find({"username": username}).sort("timestamp", -1))

    def save_analysis_run(self, username, analysis_data):
        """
        Saves a single clustering analysis run to MongoDB.
        analysis_data should be a dictionary containing all relevant info,
        including parameters, metrics, and Plotly figure JSON.
        """
        history_collection = self._get_analysis_history_collection()
        if history_collection is None:
            logger.warning("Attempted to save analysis run, but database is not connected.")
            return False, "Database not connected."

        analysis_entry = {
            "username": username,
            "timestamp": datetime.datetime.now(),
            **analysis_data # Unpack all provided analysis data
        }
        try:
            result = history_collection.insert_one(analysis_entry)
            return True, str(result.inserted_id)
        except OperationFailure as e:
            logger.error(f"Failed to save analysis run for user {username} in MongoDB: {e}")
            return False, f"Database operation failed: {e}"

    def get_user_analysis_history(self, username):
        """
        Retrieves all analysis runs for a specific user, sorted by timestamp descending.
        """
        history_collection = self._get_analysis_history_collection()
        if history_collection is None:
            return []
        return list(history_collection.find({"username": username}).sort("timestamp", -1))

    def get_analysis_run_details(self, analysis_id):
        """
        Retrieves a specific analysis run by its MongoDB ObjectId.
        """
        history_collection = self._get_analysis_history_collection()
        if history_collection is None:
            return None
        try:
            from bson.objectid import ObjectId # Import here to avoid issues if bson is not available globally
            return history_collection.find_one({"_id": ObjectId(analysis_id)})
        except Exception as e:
            logger.error(f"Failed to retrieve analysis run {analysis_id}: {e}")
            return None