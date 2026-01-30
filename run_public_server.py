
import os
import sys
import time
from pyngrok import ngrok
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

def start_public_server():
    """
    Starts the Streamlit application and exposes it to the public internet using ngrok.
    """
    # 1. Start ngrok tunnel
    logger.info("Initializing ngrok tunnel...")
    
    try:
        # Force the new auth token to ensure we bypass the bandwidth limit
        ngrok.set_auth_token("38zAoaD3SJp4WVe7RF3od3e6knp_4Auo8qniYsnwQJB74oXJt")
        
        # Open a HTTP tunnel on the default Streamlit port 8501
        public_url = ngrok.connect(8501).public_url
        logger.success(f"Tunnel established successfully!")
        logger.info(f"========================================================")
        logger.info(f"PUBLIC ACCESS URL: {public_url}")
        logger.info(f"========================================================")
        
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}")
        logger.warning("Make sure you have configured your authtoken using: ngrok config add-authtoken <TOKEN>")
        return

    # 2. Start Streamlit
    logger.info("Starting Streamlit application...")
    
    # We use os.system to run the streamlit command. 
    # This will block this script, but the tunnel is already open in the background thread managed by pyngrok.
    cmd = "streamlit run streamlit_frontend.py --server.port 8501 --server.headless true"
    
    try:
        os.system(cmd)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        ngrok.kill()

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                 Study Buddy - Public Server                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    start_public_server()
