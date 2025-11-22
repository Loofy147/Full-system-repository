// script.js

document.addEventListener('DOMContentLoaded', () => {
    const ecosystemGrid = document.getElementById('ecosystem-grid');
    const logList = document.getElementById('log-list');
    const redTeamToggle = document.getElementById('red-team-toggle');
    const API_ENDPOINT = 'http://localhost:8080/api/status';

    // Keep track of rendered logs to avoid duplicates
    const renderedLogSequences = new Set();

    async function fetchData() {
        try {
            const response = await fetch(API_ENDPOINT);
            if (!response.ok) {
                log(`Error fetching data: ${response.statusText}`, 'error');
                return;
            }
            const data = await response.json();
            console.log('Received data:', data); // Add this line for debugging
            updateDashboard(data);
        } catch (error) {
            log(`Failed to connect to the backend: ${error}`, 'error');
            console.error('Fetch error:', error); // Also log the error to the console
        }
    }

    function updateDashboard(data) {
        // 1. Update Ecosystem Nodes
        updateEcosystemNodes(data.repositories || []);

        // 2. Update Logs
        updateLogs(data.audit_trail || []);
    }

    function updateEcosystemNodes(repositories) {
        ecosystemGrid.innerHTML = ''; // Clear existing nodes
        repositories.forEach(repo => {
            const node = document.createElement('div');
            // Normalize status for CSS class (e.g., "Critical - P0" -> "critical")
            const statusClass = (repo.status || 'unknown').split(' ')[0].toLowerCase();
            node.className = `node status-${statusClass}`;
            node.id = `node-${repo.name}`;

            const nameSpan = document.createElement('span');
            nameSpan.className = 'node-name';
            nameSpan.textContent = repo.name;

            const roleSpan = document.createElement('span');
            roleSpan.className = 'node-role';
            roleSpan.textContent = repo.role || 'No role defined';

            node.appendChild(nameSpan);
            node.appendChild(roleSpan);

            ecosystemGrid.appendChild(node);
        });
    }

    function updateLogs(auditTrail) {
        auditTrail.forEach(entry => {
            if (renderedLogSequences.has(entry.sequence)) {
                return; // Skip already rendered logs
            }

            const message = `[${entry.agent_id}] Action: ${entry.action}, Result: ${entry.result}`;
            const logType = getLogType(entry.result);

            log(message, logType, new Date(entry.timestamp * 1000));
            renderedLogSequences.add(entry.sequence);
        });
    }

    function getLogType(result) {
        const securityResults = ['rate_limited', 'security_violation', 'policy_violation', 'validation_failed'];
        if (securityResults.includes(result)) {
            return 'red-team'; // Use red-team styling for security events
        }
        return 'system';
    }

    function log(message, type = 'system', timestamp = new Date()) {
        const li = document.createElement('li');
        li.textContent = `[${timestamp.toLocaleTimeString()}] ${message}`;

        // Apply class based on log type and Red Team mode
        if (type === 'red-team' && redTeamToggle.checked) {
            li.className = 'log-red-team';
        } else if (type === 'error') {
            li.className = 'log-error';
        } else {
            li.className = 'log-system';
        }

        logList.appendChild(li);

        // Auto-scroll only if the user is already at the bottom
        if (logList.scrollTop + logList.clientHeight >= logList.scrollHeight - 30) {
            logList.scrollTop = logList.scrollHeight;
        }
    }

    // Refresh log visibility when Red Team mode is toggled
    redTeamToggle.addEventListener('change', () => {
        logList.innerHTML = '';
        renderedLogSequences.clear();
        fetchData(); // Refetch to re-render logs with new styling
    });

    // Fetch data every 3 seconds
    setInterval(fetchData, 3000);

    // Initial fetch
    fetchData();
    log('Dashboard initialized. Connecting to backend...');
});
