import streamlit as st
import os
import sys
import importlib

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from authenticator import Authentication

def create_signup_page():
    """
    Create the signup page
    """
    auth = Authentication()
    auth.signup_form()

# Create the signup page
if __name__ == "__main__":
    create_signup_page()
