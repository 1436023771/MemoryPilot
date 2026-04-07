# Skill: general

## Name
general

## Summary
General-purpose execution skill for coding, command execution, web search, and tool orchestration.

## Activation
- Default skill when no specialized skill is selected.

## Prompt Key
agents.langgraph.final_user_prompt

## Allowed Tools
- web_search
- retrieve_pg_knowledge
- run_python_code
- run_docker_command

## Tool Display Names
- web_search: 联网搜索
- retrieve_pg_knowledge: 知识库检索
- run_python_code: Python计算器
- run_docker_command: Docker沙箱

## Optional Assets
- scripts/: optional helper scripts
- knowledge/: optional static knowledge files
