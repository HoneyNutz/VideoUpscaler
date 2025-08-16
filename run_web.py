#!/usr/bin/env python3
"""
Run the Video Upscaler web interface.
"""
import os
import sys
from app import create_app, socketio

def main():
    # Create the Flask application
    app = create_app()
    
    # Get configuration from environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5001))  # Changed default port to 5001
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    print(f"Starting Video Upscaler web interface on http://{host}:{port}")
    print("Press Ctrl+C to stop")
    
    # Run the application with Socket.IO
    socketio.run(app, 
                host=host, 
                port=port, 
                debug=debug, 
                use_reloader=debug,
                allow_unsafe_werkzeug=debug)

if __name__ == '__main__':
    main()
