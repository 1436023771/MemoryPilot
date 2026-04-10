"""Batch directory translation tool for local Japanese light novel files."""

from __future__ import annotations

import os
from pathlib import Path

from app.agents.tool_definition import ToolDefinition
from app.agents.tools.tools_lightnovel_translate import (
    _translate_light_novel_impl,
    clear_translate_log,
)

_MAX_DIR_BATCH_CHARS = 100000
_DEFAULT_FILE_EXTENSIONS = "txt,md,markdown,json"


def _translate_batch_directory_impl(
    source_dir: str,
    target_dir: str,
    style: str = "faithful",
    file_extensions: str = _DEFAULT_FILE_EXTENSIONS,
) -> str:
    """Translate all files in source_dir and write results to target_dir.
    
    Args:
        source_dir: Source directory path containing files to translate.
        target_dir: Target directory path for translated files.
        style: Translation style (literal, faithful, fluent). Defaults to faithful.
        file_extensions: Comma-separated file extensions to process. Defaults to txt,md,markdown,json.
    
    Returns:
        Summary of translation results.
    """
    source_path = Path(source_dir).expanduser().resolve()
    target_path = Path(target_dir).expanduser().resolve()

    if not source_path.exists():
        return f"翻译失败: 源目录不存在 {source_dir}"

    if not source_path.is_dir():
        return f"翻译失败: 源路径不是目录 {source_dir}"

    try:
        target_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        return f"翻译失败: 无法创建目标目录 {target_dir} ({exc})"

    extensions = [
        ext.strip().lstrip(".").lower()
        for ext in (file_extensions or _DEFAULT_FILE_EXTENSIONS).split(",")
        if ext.strip()
    ]

    clear_translate_log()
    results: dict[str, str] = {}
    total_chars = 0
    skipped = 0
    failed = 0

    for file_path in sorted(source_path.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lstrip(".").lower()
        if suffix not in extensions:
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            results[str(file_path.relative_to(source_path))] = f"read_error: {exc}"
            failed += 1
            continue

        if not content.strip():
            skipped += 1
            continue

        char_count = len(content)
        if total_chars + char_count > _MAX_DIR_BATCH_CHARS:
            skipped += 1
            continue

        total_chars += char_count
        translated = _translate_light_novel_impl(text=content, style=style)

        if "翻译失败" in translated:
            results[str(file_path.relative_to(source_path))] = translated
            failed += 1
        else:
            relative_path = file_path.relative_to(source_path)
            out_file = target_path / relative_path

            try:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(translated, encoding="utf-8")
                results[str(relative_path)] = "✓ translated"
            except Exception as exc:  # noqa: BLE001
                results[str(relative_path)] = f"write_error: {exc}"
                failed += 1

    success = len(results) - failed
    summary = (
        f"翻译完成\n"
        f"成功: {success}, 失败: {failed}, 跳过: {skipped}\n"
        f"处理字符数: {total_chars}\n"
        f"目标目录: {target_dir}\n"
    )

    if failed > 0 and len(results) <= 10:
        summary += "详情:\n" + "\n".join(
            f"  {k}: {v}" for k, v in results.items()
        )

    return summary


translate_light_novel_batch = ToolDefinition(
    name="translate_light_novel_batch",
    description="Batch translate Japanese light novel files from source directory to target directory.",
    input_schema={
        "type": "object",
        "properties": {
            "source_dir": {"type": "string", "description": "Source directory containing files to translate."},
            "target_dir": {"type": "string", "description": "Target directory for translated files."},
            "style": {
                "type": "string",
                "enum": ["literal", "faithful", "fluent"],
                "description": "Translation style. Defaults to faithful.",
            },
            "file_extensions": {
                "type": "string",
                "description": "Comma-separated file extensions to process (e.g., txt,md,json). Defaults to txt,md,markdown,json.",
            },
        },
        "required": ["source_dir", "target_dir"],
    },
    handler=_translate_batch_directory_impl,
)


__all__ = ["translate_light_novel_batch"]
