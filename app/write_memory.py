from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path


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
    }
    return _normalize_text(mapping[key])


def _extract_structured_facts(user_text: str) -> list[MemoryFact]:
    """从用户输入提取结构化事实。"""
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
    """对外暴露文本事实提取接口。"""
    return [fact.text for fact in _extract_structured_facts(user_text)]


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


def _parse_line_to_fact(line: str) -> MemoryFact | None:
    """将现有记忆行解析为结构化事实。"""
    normalized = _normalize_text(line)
    if not normalized:
        return None

    if normalized.startswith("用户姓名："):
        value = normalized.removeprefix("用户姓名：").split("；", 1)[0].rstrip("。.!！?？").strip()
        return MemoryFact(key="name", value=value, text=_format_fact("name", value))
    if normalized.startswith("用户喜欢："):
        value = normalized.removeprefix("用户喜欢：").split("；", 1)[0].rstrip("。.!！?？").strip()
        return MemoryFact(key="like", value=value, text=_format_fact("like", value))
    if normalized.startswith("用户不喜欢："):
        value = normalized.removeprefix("用户不喜欢：").split("；", 1)[0].rstrip("。.!！?？").strip()
        return MemoryFact(key="dislike", value=value, text=_format_fact("dislike", value))
    if normalized.startswith("用户目标："):
        value = normalized.removeprefix("用户目标：").split("；", 1)[0].rstrip("。.!！?？").strip()
        return MemoryFact(key="goal", value=value, text=_format_fact("goal", value))

    legacy_name = re.fullmatch(r"我是([^，。,.!?！？\s]+)。?", normalized)
    if legacy_name:
        value = legacy_name.group(1).strip()
        return MemoryFact(key="name", value=value, text=_format_fact("name", value))

    legacy_goal = re.fullmatch(r"我的目标是(.+?)。?", normalized)
    if legacy_goal:
        value = legacy_goal.group(1).strip()
        return MemoryFact(key="goal", value=value, text=_format_fact("goal", value))

    return None


def append_memory_facts(memory_file: Path, facts: list[str]) -> list[str]:
    """将新事实写入长期记忆，并对冲突字段执行更新。"""
    if not facts:
        return []

    memory_file.parent.mkdir(parents=True, exist_ok=True)

    # 入参 facts 可能是旧格式文本，这里通过结构化提取统一处理。
    structured_facts: list[MemoryFact] = []
    for text in facts:
        normalized = _normalize_text(text)
        if not normalized:
            continue

        if normalized.startswith("用户姓名："):
            value = normalized.removeprefix("用户姓名：").split("；", 1)[0].rstrip("。.!！?？").strip()
            structured_facts.append(MemoryFact(key="name", value=value, text=_format_fact("name", value)))
        elif normalized.startswith("用户喜欢："):
            value = normalized.removeprefix("用户喜欢：").split("；", 1)[0].rstrip("。.!！?？").strip()
            structured_facts.append(MemoryFact(key="like", value=value, text=_format_fact("like", value)))
        elif normalized.startswith("用户不喜欢："):
            value = normalized.removeprefix("用户不喜欢：").split("；", 1)[0].rstrip("。.!！?？").strip()
            structured_facts.append(MemoryFact(key="dislike", value=value, text=_format_fact("dislike", value)))
        elif normalized.startswith("用户目标："):
            value = normalized.removeprefix("用户目标：").split("；", 1)[0].rstrip("。.!！?？").strip()
            structured_facts.append(MemoryFact(key="goal", value=value, text=_format_fact("goal", value)))
        else:
            # 对自然语言事实再跑一次规则提取，兼容旧调用方式。
            structured_facts.extend(_extract_structured_facts(normalized))

    if not structured_facts:
        return []

    # 对本轮新事实去重（同 key+value）。
    seen_incoming: set[tuple[str, str]] = set()
    incoming: list[MemoryFact] = []
    for fact in structured_facts:
        identity = (fact.key, fact.value)
        if identity in seen_incoming:
            continue
        seen_incoming.add(identity)
        incoming.append(fact)

    if memory_file.exists():
        lines = memory_file.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    existing_single_values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parsed = _parse_line_to_fact(stripped)
        if parsed and parsed.key in _SINGLE_VALUE_KEYS:
            existing_single_values[parsed.key] = parsed.value

    # 单值字段如果值未变化，不需要触发更新。
    filtered_incoming: list[MemoryFact] = []
    for fact in incoming:
        if fact.key in _SINGLE_VALUE_KEYS and existing_single_values.get(fact.key) == fact.value:
            continue
        filtered_incoming.append(fact)
    incoming = filtered_incoming

    if not incoming:
        return []

    incoming_keys = {fact.key for fact in incoming}

    # 读取现有行，剔除冲突项，并收集已有事实用于去重。
    cleaned_lines: list[str] = []
    existing_fact_lines: set[str] = set()
    for line in lines:
        if _is_conflicting_line(line, incoming_keys):
            continue
        cleaned_lines.append(line)
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            existing_fact_lines.add(_normalize_text(stripped))

    newly_written: list[str] = []
    for fact in incoming:
        if fact.text in existing_fact_lines:
            continue
        newly_written.append(fact.text)
        existing_fact_lines.add(fact.text)

    if not newly_written:
        # 即使无新增，如果触发了冲突清理，也需要回写文件。
        if cleaned_lines != lines:
            final_text = "\n".join(cleaned_lines).rstrip() + "\n"
            memory_file.write_text(final_text, encoding="utf-8")
        return []

    output_lines = cleaned_lines[:]
    if output_lines and output_lines[-1].strip():
        output_lines.append("")
    output_lines.append("# 自动写入记忆")
    output_lines.extend(newly_written)

    final_text = "\n".join(output_lines).rstrip() + "\n"
    memory_file.write_text(final_text, encoding="utf-8")

    return newly_written
