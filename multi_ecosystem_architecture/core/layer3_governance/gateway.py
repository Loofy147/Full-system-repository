# core/layer3_governance/gateway.py

class GovernanceGateway:
    """
    Governance Gateway for policy enforcement, security scanning, and HITL.
    """
    def __init__(self, policies, security_scanner):
        self.policies = policies
        self.security_scanner = security_scanner
        print("GovernanceGateway initialized.")

    def enforce_policies(self, action):
        """
        Enforce predefined policies on an action.

        :param action: A dictionary describing the action to be taken.
        :return: True if the action is compliant, False otherwise.
        """
        action_type = action.get('type')
        if action_type in self.policies.get('disallowed_actions', []):
            print(f"Policy violation: Action '{action_type}' is disallowed.")
            return False

        print(f"Action '{action_type}' is compliant with policies.")
        return True

    def scan_for_security(self, artifact):
        """
        Scan a code or data artifact for security vulnerabilities.

        :param artifact: A dictionary representing the artifact to be scanned.
        :return: True if the artifact is secure, False otherwise.
        """
        scan_result = self.security_scanner.scan(artifact.get('content'))
        if not scan_result['secure']:
            print(f"Security vulnerability found: {scan_result['issue']}")
            return False

        print("Artifact passed security scan.")
        return True

    def human_in_the_loop(self, message):
        """
        Request Human-in-the-Loop (HITL) intervention.

        :param message: The message to display for the HITL review.
        :return: True if approved, False if denied.
        """
        print(f"\n--- HUMAN-IN-THE-LOOP INTERVENTION ---")
        print(message)
        response = input("Approve? (yes/no): ").lower()
        return response == 'yes'

# Dummy Security Scanner for demonstration
class DummySecurityScanner:
    def scan(self, content):
        if "dangerously_eval" in content:
            return {'secure': False, 'issue': 'Use of eval is detected'}
        return {'secure': True, 'issue': None}

if __name__ == '__main__':
    # Example Usage
    policies = {
        'disallowed_actions': ['delete_production_data']
    }
    scanner = DummySecurityScanner()
    gateway = GovernanceGateway(policies, scanner)

    # Policy Enforcement Example
    compliant_action = {'type': 'read_data'}
    non_compliant_action = {'type': 'delete_production_data'}
    gateway.enforce_policies(compliant_action)
    gateway.enforce_policies(non_compliant_action)

    # Security Scan Example
    secure_artifact = {'content': 'print("hello world")'}
    insecure_artifact = {'content': 'dangerously_eval("user_input")'}
    gateway.scan_for_security(secure_artifact)
    gateway.scan_for_security(insecure_artifact)

    # HITL Example
    # This will require manual input in the console.
    approved = gateway.human_in_the_loop("A critical action is pending. Should it proceed?")
    print(f"HITL decision: {'Approved' if approved else 'Denied'}")
