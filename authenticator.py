import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import bcrypt
import datetime
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import create_user, get_user, update_user

class Authentication:
    def __init__(self):
        """
        Initialize the authentication system
        """
        # Initialize session state variables if they don't exist
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'role' not in st.session_state:
            st.session_state.role = None
        if 'full_name' not in st.session_state:
            st.session_state.full_name = None
        if 'email' not in st.session_state:
            st.session_state.email = None
    
    def login_form(self):
        """
        Display login form and handle authentication
        """
        st.title("Login")
        # Create login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if not username or not password:
                    st.error("Please enter both username and password")
                    logger.warning("Login attempt with empty username or password.")
                    return

                try:
                    # Get user from database
                    user = get_user(username)
                    
                    if user:
                        if not user.get('active', True):
                            st.error("Account is deactivated. Please contact an administrator.")
                            logger.warning(f"Login attempt for deactivated user: {username}")
                            return

                        # Verify password
                        if bcrypt.checkpw(password.encode(), user['password'].encode()):
                            # Update last login time
                            update_user(username, {"last_login": datetime.datetime.now()})
                            
                            # Set session state
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.role = user['role']
                            st.session_state.full_name = user.get('full_name', '')
                            st.session_state.email = user.get('email', '')
                            
                            # Clear current_page to allow main.py to route to dashboard
                            if 'current_page' in st.session_state:
                                del st.session_state.current_page

                            # Redirect based on role
                            st.success(f"Welcome, {username}!")
                            logger.info(f"User '{username}' logged in successfully.")
                            st.rerun()
                        else:
                            st.error("Invalid password")
                            logger.warning(f"Failed login attempt for user '{username}': Invalid password.")
                    else:
                        st.error("Username not found")
                        logger.warning(f"Failed login attempt: Username '{username}' not found.")
                except Exception as e:
                    st.error(f"An error occurred during login: {e}")
                    logger.error(f"Error during login for user '{username}': {e}", exc_info=True)
        
        # Add button to navigate to signup page
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Don't have an account? Sign up"):
                logger.info("Navigating to signup page.")
                st.session_state.current_page = "signup"
                st.rerun()
        with col2:
            if st.button("Forgot password?"):
                logger.info("Navigating to forgot password page.")
                st.session_state.current_page = "forgot_password"
                st.rerun()
    
    def signup_form(self):
        """
        Display signup form and handle user registration
        """
        # st.write("DEBUG: authenticator.signup_form() called") # Updated debug message
        st.title("Sign Up")
        
        # Create signup form
        with st.form("signup_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            submit_button = st.form_submit_button("Sign Up")
            
            if submit_button:
                if not (username and password and confirm_password and full_name and email):
                    st.error("Please fill in all fields")
                    logger.warning("Signup attempt with incomplete fields.")
                    return

                if password != confirm_password:
                    st.error("Passwords do not match")
                    logger.warning(f"Signup attempt for user '{username}': Passwords do not match.")
                    return
                
                try:
                    # Hash password
                    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                    
                    # Create user in database (regular users only, not admins)
                    success, message = create_user(username, hashed_password, "user", email, full_name)
                    
                    if success:
                        st.success("Account created successfully! Please log in.")
                        logger.info(f"User '{username}' signed up successfully.")
                        st.session_state.current_page = "login" # Redirect to login page
                        st.rerun()
                    else:
                        st.error(message)
                        logger.error(f"Signup failed for user '{username}': {message}")
                except Exception as e:
                    st.error(f"An error occurred during signup: {e}")
                    logger.error(f"Error during signup for user '{username}': {e}", exc_info=True)
        
        # Add link back to login page
        if st.button("Already have an account? Login"):
            logger.info("Navigating to login page from signup.")
            st.session_state.current_page = "login"
            st.rerun()

    def forgot_password_form(self):
        """
        Display forgot password form and handle password reset requests
        """
        st.title("Forgot Password")

        with st.form("forgot_password_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            new_password = st.text_input("New Password", type="password")
            confirm_new_password = st.text_input("Confirm New Password", type="password")
            submit_button = st.form_submit_button("Reset Password")

            if submit_button:
                if not (username and email and new_password and confirm_new_password):
                    st.error("Please fill in all fields.")
                    logger.warning("Forgot password attempt with incomplete fields.")
                    return

                if new_password != confirm_new_password:
                    st.error("New passwords do not match.")
                    logger.warning("Forgot password attempt: New passwords do not match.")
                    return

                try:
                    user = get_user(username)
                    if user and user.get('email') == email:
                        # Simulate sending a verification email and then allowing password change
                        # In a real application, this would involve a token sent to email
                        # and a separate form for entering the new password after verification.
                        hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                        update_user(username, {"password": hashed_password})
                        st.success("Your password has been reset successfully. Please log in with your new password.")
                        logger.info(f"Password for user '{username}' reset successfully.")
                        st.session_state.current_page = "login"
                        st.rerun()
                    else:
                        st.error("Username or email is incorrect.")
                        logger.warning(f"Forgot password attempt with incorrect username '{username}' or email '{email}'.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logger.error(f"Error during forgot password for user '{username}': {e}", exc_info=True)

        if st.button("Back to Login"):
            logger.info("Navigating back to login page from forgot password.")
            st.session_state.current_page = "login"
            st.rerun()

    def logout(self):
        """
        Handle user logout
        """
        try:
            username = st.session_state.get('username', 'unknown')
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.full_name = None
            st.session_state.email = None
            st.session_state.current_page = "login" # Redirect to login page
            logger.info(f"User '{username}' logged out successfully.")
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during logout: {e}")
            logger.error(f"Error during logout for user '{username}': {e}", exc_info=True)
    
    def is_authenticated(self):
        """
        Check if user is authenticated
        """
        return st.session_state.authenticated
    
    def get_username(self):
        """
        Get current username
        """
        return st.session_state.username
    
    def get_role(self):
        """
        Get current user role
        """
        return st.session_state.role
    
    def get_user_info(self):
        """
        Get current user information
        """
        return {
            "username": st.session_state.username,
            "role": st.session_state.role,
            "full_name": st.session_state.full_name,
            "email": st.session_state.email
        }
    
    def require_authentication(self):
        """
        Require authentication to access a page
        """
        if not self.is_authenticated():
            st.warning("Please log in to access this page")
            return False
        return True
    
    def require_admin(self):
        """
        Require admin role to access a page
        """
        self.require_authentication()
        if self.get_role() != "admin":
            st.error("You do not have permission to access this page")
            st.stop()
    
    def create_admin_user(self, username, password, email, full_name):
        """
        Create an admin user (should be used only for initial setup)
        """
        # Hash password
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        # Create admin user in database
        return create_user(username, hashed_password, "admin", email, full_name)

    def admin_signup_form(self):
        """
        Display admin signup form and handle admin user registration
        """
        st.title("Admin Sign Up")
        
        # Create admin signup form
        with st.form("admin_signup_form"):
            username = st.text_input("Admin Username")
            password = st.text_input("Admin Password", type="password")
            confirm_password = st.text_input("Confirm Admin Password", type="password")
            full_name = st.text_input("Admin Full Name")
            email = st.text_input("Admin Email")
            submit_button = st.form_submit_button("Create Admin")
            
            if submit_button:
                if not (username and password and confirm_password and full_name and email):
                    st.error("Please fill in all fields")
                    logger.warning("Admin signup attempt with incomplete fields.")
                    return

                if password != confirm_password:
                    st.error("Passwords do not match")
                    logger.warning(f"Admin signup attempt for user '{username}': Passwords do not match.")
                    return
                
                try:
                    # Hash password
                    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                    
                    # Create admin user in database
                    success, message = create_user(username, hashed_password, "admin", email, full_name)
                    
                    if success:
                        st.success("Admin account created successfully!")
                        logger.info(f"Admin user '{username}' created successfully.")
                    else:
                        st.error(message)
                        logger.error(f"Admin signup failed for user '{username}': {message}")
                except Exception as e:
                    st.error(f"An error occurred during admin signup: {e}")
                    logger.error(f"Error during admin signup for user '{username}': {e}", exc_info=True)

    def reset_password(self):
        """
        Display password reset form and handle password change
        """
        st.title("Reset Password")
        
        with st.form("reset_password_form"):
            username = st.text_input("Username")
            old_password = st.text_input("Old Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_new_password = st.text_input("Confirm New Password", type="password")
            submit_button = st.form_submit_button("Reset Password")
            
            if submit_button:
                if not (username and old_password and new_password and confirm_new_password):
                    st.error("Please fill in all fields")
                    logger.warning("Password reset attempt with incomplete fields.")
                    return

                if new_password != confirm_new_password:
                    st.error("New passwords do not match")
                    logger.warning(f"Password reset attempt for user '{username}': New passwords do not match.")
                    return
                
                try:
                    user = get_user(username)
                    if user and bcrypt.checkpw(old_password.encode(), user['password'].encode()):
                        hashed_new_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                        update_user(username, {"password": hashed_new_password})
                        st.success("Password reset successfully!")
                        logger.info(f"User '{username}' successfully reset password.")
                        st.session_state.current_page = "login"
                        st.rerun()
                    else:
                        st.error("Invalid username or old password")
                        logger.warning(f"Password reset attempt for user '{username}': Invalid username or old password.")
                except Exception as e:
                    st.error(f"An error occurred during password reset: {e}")
                    logger.error(f"Error during password reset for user '{username}': {e}", exc_info=True)

    def update_user_profile(self):
        """
        Display user profile update form and handle updates
        """
        st.title("Update Profile")

        username = st.session_state.username
        user = get_user(username)

        if user:
            with st.form("update_profile_form"):
                new_full_name = st.text_input("Full Name", value=user.get('full_name', ''))
                new_email = st.text_input("Email", value=user.get('email', ''))
                new_password = st.text_input("New Password (leave blank to keep current)", type="password")
                confirm_new_password = st.text_input("Confirm New Password", type="password")
                submit_button = st.form_submit_button("Update Profile")

                if submit_button:
                    updates = {}
                    if new_full_name != user.get('full_name', ''):
                        updates['full_name'] = new_full_name
                    if new_email != user.get('email', ''):
                        updates['email'] = new_email

                    if new_password:
                        if new_password == confirm_new_password:
                            hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                            updates['password'] = hashed_password
                        else:
                            st.error("New passwords do not match")
                            logger.warning(f"Profile update attempt for user '{username}': New passwords do not match.")
                            return
                    
                    if updates:
                        try:
                            update_user(username, updates)
                            st.success("Profile updated successfully!")
                            logger.info(f"User '{username}' profile updated successfully.")
                            # Update session state if full name or email changed
                            if 'full_name' in updates:
                                st.session_state.full_name = updates['full_name']
                            if 'email' in updates:
                                st.session_state.email = updates['email']
                            st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred during profile update: {e}")
                            logger.error(f"Error updating profile for user '{username}': {e}", exc_info=True)
                    else:
                        st.info("No changes to apply.")
                        logger.info(f"User '{username}' submitted profile form with no changes.")
        else:
            st.error("User not found.")
            logger.error(f"Attempted to update profile for non-existent user: {username}")
