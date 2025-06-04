import streamlit as st
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Streamlit page - should be the first Streamlit command
# Placed here before any other st calls, especially session state access.
st.set_page_config(
    page_title="Clustering App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
# Using a dictionary for defaults makes it cleaner
session_defaults = {
    'authenticated': False,
    'username': None,
    'role': None,
    'full_name': None,
    'email': None,
    'current_page': "login"  # Default page, will be refined by routing logic
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Ensure 'authenticated' is False if not set or None
if 'authenticated' not in st.session_state or st.session_state.authenticated is None:
    st.session_state.authenticated = False

# Add the parent directory of the current file to sys.path
# This allows importing modules from the same directory as this script.
# Note: For more complex projects, consider Python packaging or structuring with a root source folder.
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if current_file_dir not in sys.path:
    sys.path.append(current_file_dir)

# Import local modules after potentially modifying sys.path
try:
    from authenticator import Authentication
    from user_dashboard import render_user_dashboard
    from admin_dashboard import render_admin_dashboard
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}. Ensure authenticator.py, user_dashboard.py, and admin_dashboard.py are in the same directory or sys.path is correctly configured.", exc_info=True)
    st.error(f"Critical import error: {e}. Application cannot start. Please check the logs.")
    st.stop() # Stop execution if essential modules are missing

# Initialize authentication
try:
    auth = Authentication()
except Exception as e:
    logger.error(f"Failed to initialize Authentication class: {e}", exc_info=True)
    st.error(f"Error initializing authentication system: {e}. Application cannot start.")
    st.stop()


# Main application logic
def main():
    logger.info(f"Main function started. Current session state: authenticated={st.session_state.get('authenticated')}, page={st.session_state.get('current_page')}")

    # --- Page Routing Logic ---
    query_params = st.query_params
    requested_page_from_query = query_params.get("page")
    if requested_page_from_query:
        requested_page_from_query = requested_page_from_query[0] if isinstance(requested_page_from_query, list) else requested_page_from_query
    else:
        requested_page_from_query = None
    
    logger.info(f"Requested page from query parameters: {requested_page_from_query}")

    current_session_page = st.session_state.get('current_page')
    is_authenticated = st.session_state.get('authenticated', False)
    
    public_pages = ["login", "signup", "forgot_password"]
    default_unauthenticated_page = "login"
    default_authenticated_page = "dashboard"

    target_page = None

    # 1. Determine initial target based on query_param, then session_page, then default.
    if requested_page_from_query:
        target_page = requested_page_from_query
        logger.info(f"Routing: Prioritizing query param page: '{target_page}'")
    elif current_session_page:
        target_page = current_session_page
        logger.info(f"Routing: Using current session page: '{target_page}'")
    # If target_page is still None, it will be set by auth-based defaults below.

    # 2. Adjust target_page based on authentication status
    if not is_authenticated:
        if target_page not in public_pages:
            logger.info(f"Routing: Not authenticated. Page '{target_page}' is not public or not set. Setting to '{default_unauthenticated_page}'.")
            target_page = default_unauthenticated_page
        # Else: target_page is already a valid public page, keep it.
        logger.info(f"Routing: Not authenticated. Final target page: '{target_page}'.")
    else:  # User is authenticated
        if target_page in public_pages or target_page is None: # If trying to access a public auth page or no page specified
            logger.info(f"Routing: Authenticated. Page '{target_page}' is public or not set. Setting to '{default_authenticated_page}'.")
            target_page = default_authenticated_page
        # Else: target_page is a valid authenticated page (e.g., "dashboard", "settings"), keep it.
        logger.info(f"Routing: Authenticated. Final target page: '{target_page}'.")
    
    # 3. Update session state and URL if the target_page has changed or needs to be canonicalized.
    # This ensures consistency and handles reruns properly.
    if st.session_state.get('current_page') != target_page:
        st.session_state.current_page = target_page
        logger.info(f"Routing: Session state 'current_page' updated to: '{target_page}'. Triggering rerun for consistency.")
        # Update query parameter to reflect the current page, this also triggers a rerun.
        # Only set it if it's different or if we want to enforce a page in the URL.
        if requested_page_from_query != target_page:
             st.query_params["page"] = target_page # This will cause a rerun
        else:
             st.rerun() # Ensure rerun even if query param matched but session state changed
        # st.rerun() or st.query_params assignment stops execution of the current script run here.

    page_to_display = st.session_state.current_page
    logger.info(f"Page to display: {page_to_display}")

    # --- Page Rendering ---
    try:
        if page_to_display == "signup":
            logger.info("Displaying signup form.")
            auth.signup_form()
        elif page_to_display == "login":
            logger.info("Displaying login form.")
            auth.login_form()
        elif page_to_display == "forgot_password":
            logger.info("Displaying forgot password form.")
            auth.forgot_password_form()
        elif page_to_display == "dashboard":
            if st.session_state.authenticated: # Explicitly check auth for dashboard
                logger.info(f"User '{st.session_state.username}' authenticated. Rendering dashboard.")
                if st.session_state.role == "admin":
                    logger.info("Rendering admin dashboard.")
                    render_admin_dashboard()
                else:
                    logger.info("Rendering user dashboard.")
                    render_user_dashboard()
            else:
                # This case should ideally be handled by routing redirecting to login
                logger.warning("Attempted to display dashboard while not authenticated. Redirecting to login.")
                st.session_state.current_page = "login"
                st.query_params["page"] = "login" # Triggers rerun
                # auth.login_form() # Avoid rendering directly, let rerun handle it
        else:
            # Fallback for unknown page or if routing somehow failed
            logger.warning(f"Unknown page '{page_to_display}' or invalid state. Defaulting to login page.")
            st.session_state.current_page = default_unauthenticated_page # Go to a known safe page
            st.query_params["page"] = st.session_state.current_page # Triggers rerun
            # auth.login_form()

    except AttributeError as e:
        if "'Authentication' object has no attribute" in str(e):
            logger.error(f"Method not found in Authentication class: {e}. Check authenticator.py.", exc_info=True)
            st.error(f"Authentication system error: A required function is missing. ({e})")
        else:
            logger.error(f"An AttributeError occurred: {e}", exc_info=True)
            st.error(f"An unexpected attribute error occurred: {e}. Please try again later.")
    except Exception as e:
        logger.error(f"An error occurred in main application logic: {e}", exc_info=True)
        st.error("An unexpected error occurred. Please try again later.")

# Entry point
if __name__ == "__main__":
    main()