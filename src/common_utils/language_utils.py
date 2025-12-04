"""语言工具模块

统一管理 Prompt 语言设置，所有需要获取默认语言的逻辑都应调用此模块的函数。
"""

import os

# 支持的语言列表
SUPPORTED_LANGUAGES = ["en", "zh"]

# 默认语言
DEFAULT_LANGUAGE = "en"


def get_prompt_language() -> str:
    """获取当前的 Prompt 语言设置

    从环境变量 MEMORY_LANGUAGE 获取语言设置，如果未设置或不支持则返回默认值 "en"。

    Returns:
        当前的语言设置，默认为 "en"
    """
    language = os.getenv("MEMORY_LANGUAGE", DEFAULT_LANGUAGE).lower()
    if language not in SUPPORTED_LANGUAGES:
        return DEFAULT_LANGUAGE
    return language


def set_prompt_language(language: str) -> bool:
    """设置 Prompt 语言

    设置环境变量 MEMORY_LANGUAGE，影响后续所有 Prompt 的语言选择。

    Args:
        language: 语言代码，支持 "en" 和 "zh"

    Returns:
        设置是否成功
    """
    language = language.lower()
    if language not in SUPPORTED_LANGUAGES:
        return False
    os.environ["MEMORY_LANGUAGE"] = language
    return True


def is_supported_language(language: str) -> bool:
    """检查语言是否支持

    Args:
        language: 语言代码

    Returns:
        是否支持该语言
    """
    return language.lower() in SUPPORTED_LANGUAGES

