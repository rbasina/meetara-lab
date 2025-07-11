## Configuration

The Trinity Core is designed to be highly configurable. All domain-to-model mappings,
training parameters, and tier definitions are managed in a single, unified file:

- `config/trinity_config.yaml`

This file is the single source of truth for the entire system. It is read by the
`SmartTrinityConfigManager` in `trinity_core/core_components/config_manager.py`,
which then provides the configuration to all other agents and components.
This centralized approach ensures consistency and simplifies management. 