from __future__ import annotations

import json
import re
from typing import Callable

from langchain_openai import ChatOpenAI

from app.core.config import get_settings

try:
    from langsmith import tracing_context
except Exception:  # noqa: BLE001
    tracing_context = None

FactTuple = tuple[str, str, str]


def extract_structured_facts_regex(
    user_text: str,
    format_fact: Callable[[str, str], str],
    normalize_text: Callable[[str], str],
) -> list[FactTuple]:
    """Legacy regex extraction kept as fallback path."""
    patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"我叫\s*([^，。,.!?！？\s]+)"), "name"),
        (re.compile(r"我是\s*([^，。,.!?！？\s]+)"), "name"),
        (re.compile(r"我喜欢\s*([^。！？!?]+)"), "like"),
        (re.compile(r"我偏好\s*([^。！？!?]+)"), "like"),
        (re.compile(r"我不喜欢\s*([^。！？!?]+)"), "dislike"),
        (re.compile(r"我的目标是\s*([^。！？!?]+)"), "goal"),
    ]

    extracted: list[FactTuple] = []
    for pattern, key in patterns:
        for match in pattern.finditer(user_text):
            value = normalize_text(match.group(1))
            if not value:
                continue
            value = value.rstrip("。.!！?？").strip()
            if not value:
                continue
            extracted.append((key, value, format_fact(key, value)))

    seen: set[tuple[str, str]] = set()
    deduped: list[FactTuple] = []
    for key, value, text in extracted:
        identity = (key, value)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append((key, value, text))
    return deduped


def extract_candidate_facts_single_turn(
    user_text: str,
    format_fact: Callable[[str, str], str],
    normalize_text: Callable[[str], str],
) -> list[FactTuple]:
    """Legacy single-turn LLM extraction kept as fallback path."""
    try:
        settings = get_settings()
        model = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

        prompt = f"""你是记忆提取助手。分析用户输入，提取【只有用户本人才能提供的个人信息】。

用户输入：
{user_text}

关键指示：
1) 【只】提取用户关于自己的信息。例如：姓名、目标、偏好、厌恶、拥有的技能、个人背景
2) 【严格拒绝】提取任何知识性内容、问题答案、通用建议、解释说明之类的内容
3) 判断标准："这句话是不是用户在描述自己的某个属性或信息？"如果答案是"是"才提取
4) 情景：如果用户提到第三方信息，记为"message"类型
5) 如果一句话既包含用户信息也包含其他内容，只提取用户信息部分
6) 用户对助手执行方式的稳定约束属于 preference，应提取

信息类别：
- name: 用户的姓名
- goal: 用户的长期目标或计划
- like: 用户喜欢的事物或风格
- dislike: 用户厌恶或不喜欢的事物
- skill: 用户具备的技能或能力
- background: 用户的背景信息
- preference: 用户对交互方式或执行方式的偏好/约束
- message: 关于他人或其他信息

返回 JSON 数组，格式示例：
[
  {{"key": "name", "value": "小李", "importance": 10}},
  {{"key": "goal", "value": "学习深度学习", "importance": 9}}
]

没有符合条件的信息时返回 []"""

        if tracing_context is not None:
            with tracing_context(enabled=False):
                response = model.invoke(prompt)
        else:
            response = model.invoke(prompt)
        if hasattr(response, "content"):
            json_str = str(response.content).strip()
        else:
            json_str = str(response).strip()

        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        data = json.loads(json_str)
        if not isinstance(data, list):
            data = []

        key_mapping = {
            "name": "name",
            "user_name": "name",
            "like": "like",
            "preference": "like",
            "dislike": "dislike",
            "goal": "goal",
            "objective": "goal",
            "skill": "skill",
            "background": "background",
        }

        extracted: list[FactTuple] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip().lower()
            value = str(item.get("value", "")).strip()
            if not key or not value:
                continue
            normalized_key = key_mapping.get(key, key)
            extracted.append((normalized_key, value, format_fact(normalized_key, value)))

        seen: set[tuple[str, str]] = set()
        deduped: list[FactTuple] = []
        for key, value, text in extracted:
            identity = (key, value)
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append((key, value, text))
        return deduped
    except (json.JSONDecodeError, ValueError, TypeError, Exception) as e:
        print(f"[Warning] Fallback single-turn extraction failed ({e}), falling back to regex extraction")
        return extract_structured_facts_regex(user_text, format_fact, normalize_text)
