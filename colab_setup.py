"""
Google Colab Setup Script for Stock Prediction Project
Run this in a Colab cell to set up the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Install packages from requirements.txt
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install additional packages that might be needed
    additional_packages = [
        "nltk",
        "spacy",
        "transformers[torch]"
    ]
    
    for package in additional_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except:
            print(f"Warning: Could not install {package}")
    
    print("Package installation completed!")

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    
    import nltk
    
    # Download required NLTK data
    nltk_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
    
    for data in nltk_data:
        try:
            nltk.download(data, quiet=True)
            print(f"Downloaded {data}")
        except:
            print(f"Warning: Could not download {data}")
    
    print("NLTK data download completed!")

def download_spacy_model():
    """Download spaCy model"""
    print("Downloading spaCy model...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("spaCy model downloaded successfully!")
    except:
        print("Warning: Could not download spaCy model")

def setup_environment():
    """Set up the complete environment"""
    print("Setting up Google Colab environment for Stock Prediction Project...")
    
    # Install requirements
    install_requirements()
    
    # Download NLTK data
    download_nltk_data()
    
    # Download spaCy model
    download_spacy_model()
    
    print("\n" + "="*50)
    print("SETUP COMPLETED!")
    print("="*50)
    print("Your environment is now ready for stock prediction analysis.")
    print("You can now import and use the modules from this project.")
    print("\nExample usage:")
    print("from data_collector import DataCollector")
    print("from sentiment_analyzer import SentimentAnalyzer")
    print("from ml_models import StockPredictor")

if __name__ == "__main__":
    setup_environment() 