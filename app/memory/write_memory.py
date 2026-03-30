from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.memory.fallback import extract_candidate_facts_single_turn, extract_structured_facts_regex

try:
    from langsmith import tracing_context
except Exception:  # noqa: BLE001
    tracing_context = None


@dataclass(frozen=True)
class MemoryFact:
    """结构化记忆事实，用于冲突更新和去重。"""

    key: str
    value: str
    text: str


_SINGLE_VALUE_KEYS = {"name", "goal"}


def _normalize_text(text: str) -> str:
    """清洗并规范化文本。"""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    if cleaned[-1] not in {"。", ".", "!", "！", "?", "？"}:
        cleaned = f"{cleaned}。"
    return cleaned


def _format_fact(key: str, value: str) -> str:
    """将结构化 key/value 映射为可读记忆句子。"""
    mapping = {
        "name": f"用户姓名：{value}",
        "like": f"用户喜欢：{value}",
        "dislike": f"用户不喜欢：{value}",
        "goal": f"用户目标：{value}",
        "skill": f"用户技能：{value}",
        "background": f"用户背景：{value}",
        "preference": f"用户偏好：{value}",
        "message": f"用户提供的信息：{value}",
    }
    text = mapping.get(key, f"{key}：{value}")
    return _normalize_text(text)



def extract_candidate_facts(user_text: str) -> list[MemoryFact]:
    """对外暴露文本事实提取接口，当前仅作为 fallback 使用。"""
    facts = extract_candidate_facts_single_turn(
        user_text=user_text,
        format_fact=_format_fact,
        normalize_text=_normalize_text,
    )
    return [MemoryFact(key=key, value=value, text=text) for key, value, text in facts]


def _flatten_message_content(content: Any) -> str:
    """将 LangChain message.content 统一转成文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _format_dialogue_for_memory(messages: list[Any], max_turns: int = 6) -> str:
    """将最近多轮会话格式化为可供 LLM 总结的对话文本。"""
    if not messages:
        return ""

    # 每轮通常包含 user+assistant 两条消息。
    max_messages = max(2, max_turns * 2)
    recent_messages = messages[-max_messages:]

    lines: list[str] = []
    for msg in recent_messages:
        msg_type = getattr(msg, "type", "")
        role = "用户" if msg_type == "human" else "助手" if msg_type == "ai" else "系统"
        text = _flatten_message_content(getattr(msg, "content", ""))
        if not text:
            continue
        lines.append(f"{role}: {text}")

    return "\n".join(lines).strip()


def extract_candidate_facts_from_dialogue(messages: list[Any], max_turns: int = 6) -> list[MemoryFact]:
    """
    从最近多轮对话中提取长期记忆：先让 LLM 总结历史，再抽取结构化事实。
    若失败则返回空列表，由上层决定是否回退到单句提取。
    返回 MemoryFact 对象以保留原始 key/value 对。
    """
    dialogue_text = _format_dialogue_for_memory(messages, max_turns=max_turns)
    if not dialogue_text:
        return []

    try:
        settings = get_settings()
        model = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

        prompt = f"""你是长期记忆提取助手。下面是最近多轮对话，你的任务是从用户说出的话中提取关于【用户本人】的重要信息。

对话历史：
{dialogue_text}

提取规则（很重要！）：
1) 【只】从"用户:"部分提取用户本人的信息，不要从"助手:"提取。
2) 提取的必须是关于用户本身的事实：姓名、长期目标、个人偏好、特殊厌恶、拥有的技能、背景信息
2.1) 用户对助手执行方式的长期约束也属于可写入偏好
3) 严格忽略助手提供的知识、检索结果、推荐、建议等【非用户信息】
4) 严格忽略一次性、临时性内容（如"这次请求"、"当前问题"、"现在的时间"等）
5) 严格忽略寒暄、重复确认、感谢等社交语言
6) 如果用户提到了关于第三方的信息，记为"message"类型
7) 遇到矛盾时，优先使用用户最新、最明确的表述

判断标准：信息必须满足"这是用户本人说出的关于自己的事实"，否则不提取。

请仅返回 JSON 数组，每个元素格式：
{{"key": "name|goal|like|dislike|skill|background|preference|message", "value": "...", "importance": 1-10}}

没有可保存信息时返回 []。不要输出任何额外文本。"""

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
            return []

        key_mapping = {
            "name": "name",
            "user_name": "name",
            "like": "like",
            "preference": "preference",
            "dislike": "dislike",
            "goal": "goal",
            "objective": "goal",
            "skill": "skill",
            "background": "background",
        }

        extracted: list[MemoryFact] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip().lower()
            value = str(item.get("value", "")).strip()
            if not key or not value:
                continue
            normalized_key = key_mapping.get(key, key)
            extracted.append(
                MemoryFact(
                    key=normalized_key,
                    value=value,
                    text=_format_fact(normalized_key, value),
                )
            )

        seen: set[tuple[str, str]] = set()
        deduped: list[MemoryFact] = []
        for fact in extracted:
            identity = (fact.key, fact.value)
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(fact)

        return deduped
    except (json.JSONDecodeError, ValueError, TypeError, Exception) as e:
        print(f"[Warning] Dialogue memory extraction failed ({e}), skipping history extraction")
        return []


def _is_conflicting_line(line: str, incoming_keys: set[str]) -> bool:
    """判断旧记忆是否与本轮写入产生冲突（需要被替换）。"""
    normalized = line.strip()
    if not normalized or normalized.startswith("#"):
        return False

    # name 与 goal 视为单值字段：写新值时替换旧值。
    if "name" in incoming_keys:
        if normalized.startswith("用户姓名："):
            return True
        if re.fullmatch(r"我是[^，。,.!?！？\s]+。?", normalized):
            return True
    if "goal" in incoming_keys:
        if normalized.startswith("用户目标："):
            return True
        if normalized.startswith("我的目标是"):
            return True

    return False


def append_memory_facts(memory_file: Path, candidate_facts: list[MemoryFact]) -> list[str]:
    """将候选事实对象存档，产生已写入事实的显示文本。"""
    if not candidate_facts:
        return []

    if not memory_file.exists():
        memory_file.parent.mkdir(parents=True, exist_ok=True)

    struct_facts = candidate_facts
    if not struct_facts:
        return []

    # 读取现有文件
    existing_content = memory_file.read_text(encoding="utf8") if memory_file.exists() else ""
    existing_lines = existing_content.strip().split("\n") if existing_content.strip() else []

    # 从原始行中提取已存在的事实
    # 直接按 key=value 格式解析（使用 | 分隔）
    existing_facts: dict[tuple[str, str], str] = {}  # (key, value) -> full_line
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # 格式: key|value
        if "|" in stripped:
            parts = stripped.split("|", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value:
                    existing_facts[(key, value)] = stripped

    # 提取本轮中要写入的 key 集群及其新值
    incoming_keys: set[str] = {fact.key for fact in struct_facts}
    incoming_key_values: dict[str, str] = {fact.key: fact.value for fact in struct_facts}

    # 按冲突规则改写旧记忆（仅删除值发生变化的行）
    filtered_lines: list[str] = []
    for line in existing_lines:
        should_keep = True
        stripped = line.strip()
        
        if stripped and not stripped.startswith("#"):
            # 新格式: key|value
            if "|" in stripped:
                parts = stripped.split("|", 1)
                if len(parts) == 2:
                    old_key = parts[0].strip()
                    old_value = parts[1].strip()
                    if old_key in _SINGLE_VALUE_KEYS and old_key in incoming_keys:
                        if incoming_key_values.get(old_key) != old_value:
                            should_keep = False

            # 检查这一行是否需要被替换（值发生变化）
            if should_keep and stripped.startswith("用户姓名："):
                old_value = stripped.removeprefix("用户姓名：").split("；", 1)[0].rstrip("。.!！?？").strip()
                if "name" in incoming_keys and incoming_key_values.get("name") != old_value:
                    should_keep = False
            elif should_keep and stripped.startswith("用户目标："):
                old_value = stripped.removeprefix("用户目标：").split("；", 1)[0].rstrip("。.!！?？").strip()
                if "goal" in incoming_keys and incoming_key_values.get("goal") != old_value:
                    should_keep = False
        
        if should_keep:
            filtered_lines.append(line)

    newly_written: list[str] = []
    new_lines: list[str] = []
    for fact in struct_facts:
        # 检查 (key, value) 对是否已存在
        if (fact.key, fact.value) not in existing_facts:
            newly_written.append(fact.text)
            new_lines.append(f"{fact.key}|{fact.value}")

    if not newly_written and filtered_lines == existing_lines:
        # 无任何变化
        return []

    # 拼接最终输出
    new_content = "\n".join(filtered_lines)
    if filtered_lines and new_content:
        new_content += "\n"
    if new_lines:
        new_content += "\n".join(new_lines)

    # 回写文件
    memory_file.write_text(new_content.strip() + "\n", encoding="utf8")

    return newly_written
