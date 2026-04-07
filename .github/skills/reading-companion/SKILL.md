# Skill: reading-companion

## Name
reading-companion

## Summary
Reading companion skill for book/chapter analysis, character relationship tracing, timeline understanding, and narrative-oriented retrieval.

## Activation
- Preferred when route is knowledge.
- Fallback keyword activation includes: 书, 章节, chapter, 剧情, plot, 角色, character, 人物, 设定, timeline, 时间线, narrative, 叙事.

## Prompt Key
agents.langgraph.reading_companion_prompt

## Allowed Tools
- retrieve_pg_knowledge
- run_python_code
- run_docker_command

## Tool Display Names
- retrieve_pg_knowledge: 书籍检索
- run_python_code: 数据处理
- run_docker_command: 高级计算

## Optional Assets
- scripts/: optional helper scripts
- knowledge/: optional static knowledge files
