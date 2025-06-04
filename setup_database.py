import os
import sys
import bcrypt

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mongodb_storage import MongoDBStorage

def setup_database():
    """
    Initialize the file-based database and create an admin user
    """
    print("Setting up file-based database...")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize storage
    storage = MongoDBStorage()
    storage.connect()
    
    # Create admin user
    admin_username = "admin"
    password = "admin123"  # Default password, should be changed after first login
    
    # Check if admin user exists
    if not storage.get_user(admin_username):
        # Hash password
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        # Create admin user
        success, message = storage.create_user(
            admin_username, 
            hashed_password, 
            "admin", 
            "admin@example.com", 
            "Administrator"
        )
        
        if success:
            print(f"Created admin user with username '{admin_username}' and password '{password}'")
        else:
            print(f"Failed to create admin user: {message}")
    else:
        print("Admin user already exists")
    
    print("Database setup complete!")

if __name__ == "__main__":
    setup_database()
