// script.js

document.addEventListener('DOMContentLoaded', () => {
    const ecosystemGrid = document.getElementById('ecosystem-grid');
    const logList = document.getElementById('log-list');
    const redTeamToggle = document.getElementById('red-team-toggle');

    const repositories = [
        { name: 'outonomos-system', status: 'critical' },
        { name: 'Herarchecal-agent', status: 'critical' },
        { name: 'Agents-box-comunication', status: 'active' },
        { name: 'Intellectual-intelligence.-', status: 'active' },
        { name: 'AutoGPT', status: 'active' },
        { name: 'a-star-search-algorithm', status: 'active' },
        { name: 'aframe-a-painter-fork', status: 'active' },
        { name: 'agent-based-modeling', status: 'active' },
        { name: 'AgentVerse', status: 'active' },
        { name: 'agenta', status: 'active' },
        { name: 'November-', status: 'deprecated' },
    ];

    // Create ecosystem nodes
    repositories.forEach(repo => {
        const node = document.createElement('div');
        node.className = `node status-${repo.status.toLowerCase()}`;
        node.id = `node-${repo.name}`;
        node.textContent = repo.name;
        ecosystemGrid.appendChild(node);
    });

    // Logging function
    function log(message, type = 'system') {
        const li = document.createElement('li');
        li.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        li.className = type === 'red-team' ? 'log-red-team' : 'log-system';
        logList.appendChild(li);
        logList.scrollTop = logList.scrollHeight; // Auto-scroll
    }

    // System simulation
    setInterval(() => {
        if (!redTeamToggle.checked) {
            const randomRepo = repositories[Math.floor(Math.random() * repositories.length)];
            log(`System activity detected in ${randomRepo.name}.`, 'system');
        }
    }, 3000);

    // Red Team simulation
    let redTeamInterval;
    redTeamToggle.addEventListener('change', () => {
        if (redTeamToggle.checked) {
            log('Red Team mode ACTIVATED.', 'red-team');
            redTeamInterval = setInterval(() => {
                const randomRepo = repositories[Math.floor(Math.random() * repositories.length)];
                const targetNode = document.getElementById(`node-${randomRepo.name}`);

                if (targetNode) {
                    log(`Adversarial attack targeting ${randomRepo.name}...`, 'red-team');
                    targetNode.classList.add('red-team-attack');
                    setTimeout(() => {
                        targetNode.classList.remove('red-team-attack');
                    }, 1500);
                }
            }, 2000);
        } else {
            log('Red Team mode DEACTIVATED.', 'system');
            clearInterval(redTeamInterval);
        }
    });

    log('Dashboard initialized.');
});
