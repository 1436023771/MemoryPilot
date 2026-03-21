from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path

from langchain_openai import ChatOpenAI

from app.config import get_settings


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
        "name": f"用户姓名：{value}；我是{value}",
        "like": f"用户喜欢：{value}；我喜欢{value}",
        "dislike": f"用户不喜欢：{value}；我不喜欢{value}",
        "goal": f"用户目标：{value}；我的目标是{value}",
        "skill": f"用户技能：{value}；我会{value}",
        "background": f"用户背景：{value}",
        "preference": f"用户偏好：{value}",
    }
    text = mapping.get(key, f"{key}：{value}")
    return _normalize_text(text)


def _extract_facts_via_llm(user_text: str) -> list[MemoryFact]:
    """
    使用 LLM 从用户输入中提取重要信息及其记忆价值。
    返回 MemoryFact 列表；如果 LLM 解析失败，回退到正则提取。
    """
    try:
        settings = get_settings()
        model = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,  # 低温度确保稳定的JSON输出
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

        prompt = f"""你是一个记忆提取助手。分析用户的输入，提取关于用户的重要信息。

用户输入：
{user_text}

请以JSON格式返回提取的信息。JSON应该包含数组，每个元素代表一个信息片段。
每个元素包含：
- key: 信息类别（如 'name', 'like', 'dislike', 'goal', 'skill', 'background', 'preference' 等）
- value: 具体的信息内容（尽可能简洁）
- importance: 重要程度（1-10，10最重要）

只返回JSON，格式如下示例：
[
  {{"key": "name", "value": "小李", "importance": 10}},
  {{"key": "like", "value": "简洁回答", "importance": 8}}
]

如果没有重要信息，返回空数组 []"""

        response = model.invoke(prompt)
        
        # 尝试从响应中解析JSON
        # LLM 返回 AIMessage 对象，需要提取 content
        if hasattr(response, "content"):
            json_str = response.content.strip()
        else:
            json_str = str(response).strip()
        # 处理可能的markdown代码块
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

        extracted: list[MemoryFact] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            key = item.get("key", "").strip().lower()
            value = item.get("value", "").strip()
            if not key or not value:
                continue
            
            # 归一化key：允许多种名称
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
            normalized_key = key_mapping.get(key, key)
            
            text = _format_fact(normalized_key, value)
            extracted.append(MemoryFact(key=normalized_key, value=value, text=text))

        # 去重并保持顺序
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
        # LLM 解析失败时，输出警告并回退到正则
        print(f"[Warning] LLM extraction failed ({e}), falling back to regex extraction")
        return _extract_structured_facts_regex(user_text)


def _extract_structured_facts_regex(user_text: str) -> list[MemoryFact]:
    """
    原有的正则提取方式，作为 LLM 提取的备选方案。
    """
    patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"我叫\s*([^，。,.!?！？\s]+)"), "name"),
        (re.compile(r"我是\s*([^，。,.!?！？\s]+)"), "name"),
        (re.compile(r"我喜欢\s*([^。！？!?]+)"), "like"),
        (re.compile(r"我偏好\s*([^。！？!?]+)"), "like"),
        (re.compile(r"我不喜欢\s*([^。！？!?]+)"), "dislike"),
        (re.compile(r"我的目标是\s*([^。！？!?]+)"), "goal"),
    ]

    extracted: list[MemoryFact] = []
    for pattern, key in patterns:
        for match in pattern.finditer(user_text):
            value = _normalize_text(match.group(1))
            if not value:
                continue
            value = value.rstrip("。.!！?？").strip()
            if not value:
                continue
            text = _format_fact(key, value)
            extracted.append(MemoryFact(key=key, value=value, text=text))

    # 去重并保持顺序
    seen: set[tuple[str, str]] = set()
    deduped: list[MemoryFact] = []
    for fact in extracted:
        identity = (fact.key, fact.value)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(fact)
    return deduped


def extract_candidate_facts(user_text: str) -> list[str]:
    """对外暴露文本事实提取接口，使用 LLM 驱动的提取。"""
    facts = _extract_facts_via_llm(user_text)
    return [fact.text for fact in facts]


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


def append_memory_facts(memory_file: Path, candidate_facts: list[str]) -> list[str]:
    """将候选事实对象存档，产生已写入的事实。
    
    接受多种输入格式：
    - 已格式化的事实（如 "用户姓名：小李；我是小李。"）
    - 原始用户输入（如 "我喜欢Python"）
    - 两者混合
    """
    if not candidate_facts:
        return []

    if not memory_file.exists():
        memory_file.parent.mkdir(parents=True, exist_ok=True)

    # 将输入规范化为 MemoryFact 列表
    struct_facts: list[MemoryFact] = []
    for fact_str in candidate_facts:
        if not fact_str or not fact_str.strip():
            continue
        
        normalized = _normalize_text(fact_str)
        
        # 尝试解析为已格式化的事实
        if normalized.startswith("用户姓名："):
            value = normalized.removeprefix("用户姓名：").split("；", 1)[0].rstrip("。.!！?？").strip()
            if value:
                struct_facts.append(MemoryFact(key="name", value=value, text=_format_fact("name", value)))
        elif normalized.startswith("用户喜欢："):
            value = normalized.removeprefix("用户喜欢：").split("；", 1)[0].rstrip("。.!！?？").strip()
            if value:
                struct_facts.append(MemoryFact(key="like", value=value, text=_format_fact("like", value)))
        elif normalized.startswith("用户不喜欢："):
            value = normalized.removeprefix("用户不喜欢：").split("；", 1)[0].rstrip("。.!！?？").strip()
            if value:
                struct_facts.append(MemoryFact(key="dislike", value=value, text=_format_fact("dislike", value)))
        elif normalized.startswith("用户目标："):
            value = normalized.removeprefix("用户目标：").split("；", 1)[0].rstrip("。.!！?？").strip()
            if value:
                struct_facts.append(MemoryFact(key="goal", value=value, text=_format_fact("goal", value)))
        else:
            # 尝试用正则提取
            extracted = _extract_structured_facts_regex(normalized)
            struct_facts.extend(extracted)

    if not struct_facts:
        return []

    # 读取现有文件
    existing_content = memory_file.read_text(encoding="utf8") if memory_file.exists() else ""
    existing_lines = existing_content.strip().split("\n") if existing_content.strip() else []

    # 从原始行中提取已存在的事实
    existing_facts: dict[tuple[str, str], str] = {}  # (key, value) -> full_line
    existing_facts_list: list[MemoryFact] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        
        # 尝试解析为 MemoryFact
        if stripped.startswith("用户姓名："):
            value = stripped.removeprefix("用户姓名：").split("；", 1)[0].rstrip("。.!！?？").strip()
            if value:
                existing_facts[("name", value)] = stripped
                existing_facts_list.append(MemoryFact(key="name", value=value, text=stripped))
        elif stripped.startswith("用户喜欢："):
            value = stripped.removeprefix("用户喜欢：").split("；", 1)[0].rstrip("。.!！?？").strip()
            if value:
                existing_facts[("like", value)] = stripped
                existing_facts_list.append(MemoryFact(key="like", value=value, text=stripped))
        elif stripped.startswith("用户不喜欢："):
            value = stripped.removeprefix("用户不喜欢：").split("；", 1)[0].rstrip("。.!！?？").strip()
            if value:
                existing_facts[("dislike", value)] = stripped
                existing_facts_list.append(MemoryFact(key="dislike", value=value, text=stripped))
        elif stripped.startswith("用户目标："):
            value = stripped.removeprefix("用户目标：").split("；", 1)[0].rstrip("。.!！?？").strip()
            if value:
                existing_facts[("goal", value)] = stripped
                existing_facts_list.append(MemoryFact(key="goal", value=value, text=stripped))

    # 提取本轮中要写入的 key 集群及其新值
    incoming_keys: set[str] = {fact.key for fact in struct_facts}
    incoming_key_values: dict[str, str] = {fact.key: fact.value for fact in struct_facts}

    # 按冲突规则改写旧记忆（仅删除值发生变化的行）
    filtered_lines: list[str] = []
    for line in existing_lines:
        should_keep = True
        stripped = line.strip()
        
        if stripped and not stripped.startswith("#"):
            # 检查这一行是否需要被替换（值发生变化）
            if stripped.startswith("用户姓名："):
                old_value = stripped.removeprefix("用户姓名：").split("；", 1)[0].rstrip("。.!！?？").strip()
                if "name" in incoming_keys and incoming_key_values.get("name") != old_value:
                    should_keep = False
            elif stripped.startswith("用户目标："):
                old_value = stripped.removeprefix("用户目标：").split("；", 1)[0].rstrip("。.!！?？").strip()
                if "goal" in incoming_keys and incoming_key_values.get("goal") != old_value:
                    should_keep = False
        
        if should_keep:
            filtered_lines.append(line)

    newly_written: list[str] = []
    for fact in struct_facts:
        # 检查 (key, value) 对是否已存在
        if (fact.key, fact.value) not in existing_facts:
            newly_written.append(fact.text)

    if not newly_written and filtered_lines == existing_lines:
        # 无任何变化
        return []

    # 拼接最终输出
    new_content = "\n".join(filtered_lines)
    if filtered_lines and new_content:
        new_content += "\n"
    if newly_written:
        new_content += "\n".join(newly_written)

    # 回写文件
    memory_file.write_text(new_content.strip() + "\n", encoding="utf8")

    return newly_written
