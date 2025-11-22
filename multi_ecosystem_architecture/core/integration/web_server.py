# core/integration/web_server.py
"""
A simple, multi-threaded web server to expose system status for the dashboard.
"""

import http.server
import socketserver
import json
import threading
import yaml
from pathlib import Path
from typing import Dict, Any
import os
import logging

logger = logging.getLogger("Système.WebServer")

class SystemStatusHandler(http.server.SimpleHTTPRequestHandler):
    """Handles API requests for system status."""

    system_instance = None
    ecosystem_config = None

    def do_GET(self):
        if self.path == '/api/status':
            self._handle_api_status()
        else:
            # The server is now running from the dashboard directory,
            # so the default handler will find the files.
            super().do_GET()

    def _handle_api_status(self):
        """Handle the /api/status endpoint."""
        if self.system_instance is None:
            self.send_error(500, "System instance not available")
            return

        try:
            # Extract data from the live system
            governance_gateway = self.system_instance.get("components", {}).get("governance_gateway")
            if governance_gateway is None:
                self.send_error(500, "Governance gateway not found in system")
                return

            data = {
                "repositories": self.ecosystem_config.get("repositories", []),
                "security_report": governance_gateway.get_security_report(),
                "audit_trail": governance_gateway.get_audit_trail(limit=50)
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*') # For local development
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))

        except Exception as e:
            logger.error(f"API Error: {e}", exc_info=True)
            self.send_error(500, f"Error getting system status: {e}")

class WebServer:
    """A threaded web server for the system dashboard."""

    def __init__(self, port: int = 8080):
        self.port = port
        self.httpd = None
        self.thread = None
        self.handler = SystemStatusHandler
        self.original_cwd = Path.cwd()

    def start(self, system_instance):
        """Start the server in a background thread."""
        logger.info("WebServer starting...")
        self.handler.system_instance = system_instance

        repo_root = self.original_cwd
        logger.info(f"Repo root detected as: {repo_root}")

        config_path = repo_root / 'multi_ecosystem_architecture/ecosystem_config.yml'
        dashboard_dir = repo_root / 'multi_ecosystem_architecture/dashboard'

        logger.info(f"Attempting to load ecosystem config from: {config_path}")
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.handler.ecosystem_config = yaml.safe_load(f)
            logger.info("Ecosystem config loaded successfully.")
        else:
            logger.warning(f"Ecosystem config not found at {config_path}")
            self.handler.ecosystem_config = {}

        logger.info(f"Dashboard directory target: {dashboard_dir}")
        if not dashboard_dir.exists() or not dashboard_dir.is_dir():
            logger.error(f"Dashboard directory not found at {dashboard_dir}. Aborting server start.")
            return

        try:
            logger.info(f"Changing current directory to {dashboard_dir}")
            os.chdir(dashboard_dir)
            logger.info(f"Current directory is now: {Path.cwd()}")

            self.httpd = socketserver.TCPServer(("", self.port), self.handler)

            self.thread = threading.Thread(target=self.httpd.serve_forever)
            self.thread.daemon = True
            logger.info("Starting server thread...")
            self.thread.start()
            logger.info(f"Dashboard server started and listening on http://localhost:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start web server: {e}", exc_info=True)
        finally:
            pass

    def stop(self):
        """Stop the server and return to original directory."""
        logger.info("Stopping web server...")
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            logger.info("Server shut down.")

        if Path.cwd() != self.original_cwd:
            os.chdir(self.original_cwd)
            logger.info(f"Restored original CWD: {self.original_cwd}")
        logger.info("WebServer stopped.")

if __name__ == '__main__':
    print("This script is intended to be imported and run by the main system.")
