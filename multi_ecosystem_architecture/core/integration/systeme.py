# core/integration/systeme.py
"""
Système Integration Layer
Wires all components together into a cohesive system.
"""

import asyncio
import logging
import configparser
from dataclasses import dataclass
from typing import Dict, Optional, Any
from pathlib import Path

import os
from core.layer1_orchestration.async_router import AsyncTaskRouter
from core.layer2_hrl.manager import HRLManager
from core.layer2_hrl.ace_playbook import ACEPlaybook, ACECurator
from core.layer3_governance.hardened_gateway import HardenedGovernanceGateway
from core.layer2_hrl.neural_uvfa import NeuralUVFA
from core.validation.schemas import MessageSigner
from core.red_team_engine.engine import EnhancedRedTeamEngine
from core.integration.web_server import WebServer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("Système")

@dataclass
class SystemConfig:
    audit_log_path: Path
    playbook_db_path: Path
    red_team_playbook_path: Path

class SystemeFactory:
    """Factory for creating fully wired Système instances."""

    @staticmethod
    def load_config(config_path: str = "multi_ecosystem_architecture/config.ini") -> SystemConfig:
        """Load configuration from an INI file, resolving paths to be absolute."""
        parser = configparser.ConfigParser()
        parser.read(config_path)

        # Resolve all paths relative to the config file's location to make them absolute
        base_path = Path(config_path).parent.resolve()

        return SystemConfig(
            audit_log_path=base_path / parser.get("Paths", "audit_log"),
            playbook_db_path=base_path / parser.get("Paths", "playbook_db"),
            red_team_playbook_path=base_path / parser.get("Paths", "red_team_playbook")
        )

    @staticmethod
    def create(config: Optional[SystemConfig] = None) -> Dict[str, Any]:
        """Create and wire all system components based on configuration."""
        if config is None:
            config = SystemeFactory.load_config()

        # Shared security components
        secret_key_hex = os.environ.get("SECRET_KEY")
        if not secret_key_hex:
            # For dashboard verification, we can use a dummy key
            logger.warning("SECRET_key environment variable not set, using a dummy key.")
            secret_key_hex = '0' * 64
        secret_key = bytes.fromhex(secret_key_hex)


        # Layer 2: HRL with Neural UVFA
        playbook = ACEPlaybook(str(config.playbook_db_path))
        curator = ACECurator(playbook)
        uvfa = NeuralUVFA(state_dim=32, goal_dim=16)
        manager = HRLManager(playbook_path=None)

        # Layer 1: Orchestration with validation
        router = AsyncTaskRouter(curator=curator)

        # Layer 3: Hardened Governance
        policies = {
            "disallowed_actions": [
                "delete_production_data",
                "drop_database",
                "execute_arbitrary_code"
            ]
        }
        governance = HardenedGovernanceGateway(
            policies,
            str(config.audit_log_path),
            secret_key=secret_key
        )

        # Message signer for A2A
        signer = MessageSigner(secret_key)

        # Red Team
        system_components = {
            "task_router": router,
            "governance_gateway": governance,
            "hrl_manager": manager,
            "ace_playbook": playbook,
            "uvfa": uvfa,
            "signer": signer,
        }
        red_team = EnhancedRedTeamEngine(str(config.red_team_playbook_path), system_components)

        # Web server for dashboard
        web_server = WebServer()

        logger.info("Système factory initialization complete")

        # The full system object to be passed to the web server
        full_system = {
            "router": router,
            "governance": governance,
            "red_team": red_team,
            "components": system_components,
            "uvfa": uvfa,
            "signer": signer,
            "web_server": web_server,
        }

        # Start the web server with the full system context
        web_server.start(full_system)

        return full_system

async def main():
    """Demo the integrated system."""
    print("=" * 60)
    print("SYSTÈME MULTI-ECOSYSTEM INTEGRATION DEMO")
    print("=" * 60)

    config = SystemeFactory.load_config()
    systeme = SystemeFactory.create(config)

    governance = systeme["governance"]

    # Run some dummy actions to populate the audit log for the dashboard
    logger.info("Running dummy actions to populate audit log...")
    governance.validate_action({"type": "read_data"}, "agent_1")
    governance.validate_action({"type": "write_data"}, "agent_2")
    governance.validate_action({"type": "delete_production_data"}, "agent_3_malicious")
    governance.validate_action({"type": "safe_action"}, "agent_4")
    governance.validate_action({"type": "safe_action"}, "agent_1")
    governance.validate_action({"type": "execute", "code": "eval('danger')"}, "agent_5_malicious")
    logger.info("Dummy actions complete.")

    print("\nDashboard is running in the background.")
    print("Visit http://localhost:8080 to view the live dashboard.")
    print("Press Ctrl+C to stop the system.")

    try:
        # Keep the main thread alive to let the background server run
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        systeme["web_server"].stop()
        print("System stopped.")

if __name__ == "__main__":
    # Ensure a secret key is set for the demo
    if "SECRET_KEY" not in os.environ:
        os.environ["SECRET_KEY"] = os.urandom(32).hex()

    asyncio.run(main())
