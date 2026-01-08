'''
Description: 封装工具prompt调用, 提供统一的工具描述
Author: zyq
Date: 2026-01-07 17:03:44
LastEditors: zyq
LastEditTime: 2026-01-08 10:05:04
'''

from typing import List, Dict, Any


def _safe_get(info: Dict[str, Any], key: str, default: str = "") -> str:
    val = info.get(key, default)
    return str(val)


def build_tool_lines(tool_desc_list: List[Dict[str, Any]], bullet: str = "-") -> str:
    """将工具元数据转换为简洁的行描述, 便于拼接到提示词"""
    if not tool_desc_list:
        return ""
    lines = []
    for info in tool_desc_list:
        name = _safe_get(info, "name")
        desc = _safe_get(info, "description")
        params = info.get("parameters", "")
        lines.append(f"{bullet}{name}: {desc} |parameters={params}")
    return "\n".join(lines)


def build_enhanced_tool_section(tool_desc_list: List[Dict[str, Any]]) -> str:
    """
    构建包含工具清单与调用格式的提示片段, 适合简单 Agent 使用。
    只负责字符串拼接, 不做状态处理。
    """
    if not tool_desc_list:
        return ""

    parts: List[str] = []
    for info in tool_desc_list:
        parts.append(
            f"工具名称: {info.get('name', '')}\n"
            f"工具描述: {info.get('description', '')}\n"
            f"工具参数定义: {info.get('parameters', '')}\n"
        )
    all_tools_desc = "".join(parts)

    section = "\n\n ## 可用工具列表\n"
    section += "你可以使用以下工具来帮助回答用户问题, 以下是具体的工具描述:\n"
    section += all_tools_desc

    section += "\n## 工具调用格式\n"
    section += "当需要使用工具时，请使用以下格式：\n"
    section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n\n"

    section += "### 参数格式说明\n"
    section += "1. 多个参数：使用 `key=value` 逗号分隔，例如 `[TOOL_CALL:calculator_multiply:a=12,b=8]`\n"
    section += "2. 单个参数：直接 `key=value`，例如 `[TOOL_CALL:search:query=Python编程]`\n"
    section += "3. 简单查询：直接写文本，例如 `[TOOL_CALL:search:Python编程]`\n"

    section += "### 重要提示\n"
    section += "- 参数名必须与工具定义的参数名完全匹配\n"
    section += "- 数字参数直接写数字，不要加引号\n"
    section += "- 工具调用结果会自动插入到对话中，然后基于结果回答\n"
    return section
