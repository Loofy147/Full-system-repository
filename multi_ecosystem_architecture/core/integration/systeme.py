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

from core.layer1_orchestration.async_router import AsyncTaskRouter
from core.layer2_hrl.manager import HRLManager
from core.layer2_hrl.ace_playbook import ACEPlaybook, ACECurator
from core.layer3_governance.enhanced_gateway import EnhancedGovernanceGateway
from core.red_team_engine.engine import EnhancedRedTeamEngine

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
        """Load configuration from an INI file."""
        parser = configparser.ConfigParser()
        parser.read(config_path)

        base_path = Path(config_path).parent

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

        # Layer 1: Orchestration
        router = AsyncTaskRouter()

        # Layer 2: HRL + ACE
        playbook = ACEPlaybook(str(config.playbook_db_path))
        curator = ACECurator(playbook)
        manager = HRLManager(playbook_path=None)

        # Layer 3: Governance
        policies = {
            "disallowed_actions": [
                "delete_production_data",
                "drop_database",
                "execute_arbitrary_code"
            ]
        }
        governance = EnhancedGovernanceGateway(policies, str(config.audit_log_path))

        # Red Team
        system_components = {
            "task_router": router,
            "governance_gateway": governance,
            "hrl_manager": manager,
            "ace_playbook": playbook,
        }
        red_team = EnhancedRedTeamEngine(str(config.red_team_playbook_path), system_components)

        logger.info("Système factory initialization complete")

        return {
            "router": router,
            "governance": governance,
            "red_team": red_team,
            "components": system_components
        }

async def main():
    """Demo the integrated system."""
    print("=" * 60)
    print("SYSTÈME MULTI-ECOSYSTEM INTEGRATION DEMO")
    print("=" * 60)

    config = SystemeFactory.load_config()
    systeme = SystemeFactory.create(config)

    governance = systeme["governance"]

    valid, msg = governance.validate_action({"type": "read_data"}, "agent_1")
    print(f"Read action: {msg}")

    valid, msg = governance.validate_action({"type": "delete_production_data"}, "agent_1")
    print(f"Delete action: {msg}")

    valid, msg = governance.validate_action({"type": "execute", "code": "eval(user_input)"}, "agent_1")
    print(f"Eval action: {msg}")

    red_team = systeme["red_team"]

    if red_team.playbook.get("attack_scenarios"):
        result = await red_team.run_scenario("GOV-001")
        print(f"Attack result: {result.vulnerability_found}")

    print("\n" + red_team.generate_report())

if __name__ == "__main__":
    asyncio.run(main())
