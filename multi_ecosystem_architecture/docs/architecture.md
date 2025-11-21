# Multi-Ecosystem Architecture Diagram

This diagram visualizes the 4-layer architecture of the multi-ecosystem system, including the data flow and the role of the Red Team Engine.

```mermaid
graph TD
    subgraph "User / External Systems"
        A[User Request / Task]
    end

    subgraph "Système Multi-Ecosystem Core"
        direction LR

        subgraph "Layer 4: Resilience"
            L4[Resilience Patterns <br> Circuit Breaker, Retry w/ Backoff]
        end

        subgraph "Layer 3: Governance"
            L3[Governance Gateway <br> Policy Enforcement, Security Scanning, HITL]
        end

        subgraph "Layer 2: Hierarchical Reinforcement Learning"
            L2_Manager[HRL Manager]
            L2_Playbook[ACE Playbook]
            L2_Worker[HRL Worker <br> w/ HER]

            L2_Manager -- Reads --> L2_Playbook
            L2_Manager -- Delegates --> L2_Worker
        end

        subgraph "Layer 1: Orchestration"
            L1[Task Router <br> MCP / A2A Communication]
        end

    end

    subgraph "Red Team Engine"
        RT[Adversarial Testing Engine]
    end

    subgraph "Integrated Ecosystems"
        E[Ecosystem Repositories <br> (e.g., AutoGPT, AgentVerse)]
    end

    %% Data Flow
    A --> L1
    L1 --> L2_Manager
    L2_Worker -- Executes Action --> L3
    L3 -- Compliant Action --> L4
    L4 -- Resilient Call --> E

    %% Red Team Attack Flow
    RT -- Adversarial Attacks --> L1
    RT -- Adversarial Attacks --> L2_Manager
    RT -- Adversarial Attacks --> L3
    RT -- Adversarial Attacks --> L4
    RT -- Adversarial Attacks --> E

```
