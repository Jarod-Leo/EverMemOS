"""
多语言提示词模块

通过环境变量 MEMORY_LANGUAGE 控制默认语言，支持 'en' 和 'zh'
默认使用英文 ('en')

使用方法：
1. 设置环境变量：export MEMORY_LANGUAGE=zh
2. 现有代码无需修改，直接从 memory_layer.prompts 导入使用（使用默认语言）
3. 动态获取指定语言的 Prompt：使用 get_prompt_by(prompt_name, language)

示例：
    # 方式1：使用默认语言（基于环境变量）
    from memory_layer.prompts import (
        EPISODE_GENERATION_PROMPT,
        CONVERSATION_PROFILE_PART1_EXTRACTION_PROMPT,
        get_foresight_generation_prompt,
    )
    
    # 方式2：动态获取指定语言的 Prompt
    from memory_layer.prompts import get_prompt_by
    prompt = get_prompt_by("EPISODE_GENERATION_PROMPT", language="zh")
"""

from typing import Any, Optional, Callable

from common_utils.language_utils import (
    get_prompt_language,
    set_prompt_language,
    is_supported_language,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
)

# ============================================================================
# Prompt 注册表 - 存储所有可用的 Prompt 名称和对应模块路径
# ============================================================================

# Prompt 名称到模块路径的映射
# 格式: {prompt_name: {language: (module_path, is_function)}}
_PROMPT_REGISTRY = {
    # 对话相关
    "CONV_BOUNDARY_DETECTION_PROMPT": {
        "en": ("memory_layer.prompts.en.conv_prompts", False),
        "zh": ("memory_layer.prompts.zh.conv_prompts", False),
    },
    "CONV_SUMMARY_PROMPT": {
        "en": ("memory_layer.prompts.en.conv_prompts", False),
        "zh": ("memory_layer.prompts.zh.conv_prompts", False),
    },
    # Episode 相关
    "EPISODE_GENERATION_PROMPT": {
        "en": ("memory_layer.prompts.en.episode_mem_prompts", False),
        "zh": ("memory_layer.prompts.zh.episode_mem_prompts", False),
    },
    "GROUP_EPISODE_GENERATION_PROMPT": {
        "en": ("memory_layer.prompts.en.episode_mem_prompts", False),
        "zh": ("memory_layer.prompts.zh.episode_mem_prompts", False),
    },
    "DEFAULT_CUSTOM_INSTRUCTIONS": {
        "en": ("memory_layer.prompts.en.episode_mem_prompts", False),
        "zh": ("memory_layer.prompts.zh.episode_mem_prompts", False),
    },
    # Profile 相关
    "CONVERSATION_PROFILE_EXTRACTION_PROMPT": {
        "en": ("memory_layer.prompts.en.profile_mem_prompts", False),
        "zh": ("memory_layer.prompts.zh.profile_mem_prompts", False),
    },
    "CONVERSATION_PROFILE_PART1_EXTRACTION_PROMPT": {
        "en": ("memory_layer.prompts.en.profile_mem_part1_prompts", False),
        "zh": ("memory_layer.prompts.zh.profile_mem_part1_prompts", False),
    },
    "CONVERSATION_PROFILE_PART2_EXTRACTION_PROMPT": {
        "en": ("memory_layer.prompts.en.profile_mem_part2_prompts", False),
        "zh": ("memory_layer.prompts.zh.profile_mem_part2_prompts", False),
    },
    "CONVERSATION_PROFILE_PART3_EXTRACTION_PROMPT": {
        "en": ("memory_layer.prompts.en.profile_mem_part3_prompts", False),
        "zh": ("memory_layer.prompts.zh.profile_mem_part3_prompts", False),
    },
    "CONVERSATION_PROFILE_EVIDENCE_COMPLETION_PROMPT": {
        "en": ("memory_layer.prompts.en.profile_mem_evidence_completion_prompt", False),
        "zh": ("memory_layer.prompts.zh.profile_mem_evidence_completion_prompt", False),
    },
    # Group Profile 相关
    "CONTENT_ANALYSIS_PROMPT": {
        "en": ("memory_layer.prompts.en.group_profile_prompts", False),
        "zh": ("memory_layer.prompts.zh.group_profile_prompts", False),
    },
    "BEHAVIOR_ANALYSIS_PROMPT": {
        "en": ("memory_layer.prompts.en.group_profile_prompts", False),
        "zh": ("memory_layer.prompts.zh.group_profile_prompts", False),
    },
    # Foresight 相关（函数）
    "get_foresight_generation_prompt": {
        "en": ("memory_layer.prompts.en.foresight_prompts", True),
        "zh": ("memory_layer.prompts.zh.foresight_prompts", True),
    },
    "get_group_foresight_generation_prompt": {
        "en": ("memory_layer.prompts.en.foresight_prompts", True),
        "zh": ("memory_layer.prompts.zh.foresight_prompts", True),
    },
    # Event Log 相关
    "EVENT_LOG_PROMPT": {
        "en": ("memory_layer.prompts.en.event_log_prompts", False),
        "zh": ("memory_layer.prompts.zh.event_log_prompts", False),
    },
}


# ============================================================================
# PromptManager - 动态 Prompt 管理器
# ============================================================================


class PromptManager:
    """Prompt 管理器，支持动态获取不同语言的 Prompt
    
    使用示例：
        manager = PromptManager()
        
        # 获取指定语言的 Prompt
        prompt = manager.get_prompt("EPISODE_GENERATION_PROMPT", language="zh")
        
        # 获取函数类型的 Prompt
        func = manager.get_prompt("get_foresight_generation_prompt", language="en")
        result = func(...)
    """

    def __init__(self):
        """初始化 PromptManager"""
        # 缓存已加载的模块
        self._module_cache: dict[str, Any] = {}

    def _load_module(self, module_path: str) -> Any:
        """动态加载模块
        
        Args:
            module_path: 模块路径，如 "memory_layer.prompts.en.conv_prompts"
            
        Returns:
            加载的模块对象
        """
        if module_path not in self._module_cache:
            import importlib
            self._module_cache[module_path] = importlib.import_module(module_path)
        return self._module_cache[module_path]

    def get_prompt(self, prompt_name: str, language: Optional[str] = None) -> Any:
        """获取指定语言的 Prompt
        
        Args:
            prompt_name: Prompt 名称，如 "EPISODE_GENERATION_PROMPT"
            language: 语言代码，如 "en" 或 "zh"。如果为 None，使用环境变量设置的默认语言
            
        Returns:
            Prompt 字符串或函数
            
        Raises:
            ValueError: 如果 Prompt 名称不存在或语言不支持
        """
        # 使用默认语言
        if language is None:
            language = get_prompt_language()
        
        language = language.lower()
        
        # 检查 Prompt 是否存在
        if prompt_name not in _PROMPT_REGISTRY:
            raise ValueError(f"Unknown prompt: {prompt_name}. Available prompts: {list(_PROMPT_REGISTRY.keys())}")
        
        # 检查语言是否支持
        prompt_info = _PROMPT_REGISTRY[prompt_name]
        if language not in prompt_info:
            raise ValueError(f"Language '{language}' not supported for prompt '{prompt_name}'. Available: {list(prompt_info.keys())}")
        
        # 获取模块路径和类型
        module_path, is_function = prompt_info[language]
        
        # 加载模块并获取 Prompt
        module = self._load_module(module_path)
        return getattr(module, prompt_name)

    def list_prompts(self) -> list[str]:
        """列出所有可用的 Prompt 名称
        
        Returns:
            Prompt 名称列表
        """
        return list(_PROMPT_REGISTRY.keys())

    def get_supported_languages(self, prompt_name: str) -> list[str]:
        """获取指定 Prompt 支持的语言列表
        
        Args:
            prompt_name: Prompt 名称
            
        Returns:
            支持的语言列表
        """
        if prompt_name not in _PROMPT_REGISTRY:
            return []
        return list(_PROMPT_REGISTRY[prompt_name].keys())


# 全局 PromptManager 实例
_prompt_manager = PromptManager()


def get_prompt_by(prompt_name: str, language: Optional[str] = None) -> Any:
    """获取指定语言的 Prompt（便捷函数）
    
    Args:
        prompt_name: Prompt 名称，如 "EPISODE_GENERATION_PROMPT"
        language: 语言代码，如 "en" 或 "zh"。如果为 None，使用环境变量设置的默认语言
        
    Returns:
        Prompt 字符串或函数
        
    Raises:
        ValueError: 如果 Prompt 名称不存在或语言不支持
        
    示例：
        # 获取中文的 Episode 生成 Prompt
        prompt = get_prompt_by("EPISODE_GENERATION_PROMPT", language="zh")
        
        # 获取默认语言的 Prompt
        prompt = get_prompt_by("CONV_BOUNDARY_DETECTION_PROMPT")
        
        # 获取函数类型的 Prompt
        func = get_prompt_by("get_foresight_generation_prompt", language="en")
        result = func(...)
    """
    return _prompt_manager.get_prompt(prompt_name, language)


# ============================================================================
# 向后兼容 - 保持现有的直接导入方式（使用默认语言）
# ============================================================================

# 获取默认语言
_default_language = get_prompt_language()

# 根据默认语言设置动态导入提示词
if _default_language == "zh":
    # ===== 中文提示词 =====
    # 对话相关
    from memory_layer.prompts.zh.conv_prompts import CONV_BOUNDARY_DETECTION_PROMPT, CONV_SUMMARY_PROMPT
    
    # Episode 相关
    from memory_layer.prompts.zh.episode_mem_prompts import (
        EPISODE_GENERATION_PROMPT,
        GROUP_EPISODE_GENERATION_PROMPT,
        DEFAULT_CUSTOM_INSTRUCTIONS,
    )
    
    # Profile 相关
    from memory_layer.prompts.zh.profile_mem_prompts import CONVERSATION_PROFILE_EXTRACTION_PROMPT
    from memory_layer.prompts.zh.profile_mem_part1_prompts import CONVERSATION_PROFILE_PART1_EXTRACTION_PROMPT
    from memory_layer.prompts.zh.profile_mem_part2_prompts import CONVERSATION_PROFILE_PART2_EXTRACTION_PROMPT
    from memory_layer.prompts.zh.profile_mem_part3_prompts import CONVERSATION_PROFILE_PART3_EXTRACTION_PROMPT
    from memory_layer.prompts.zh.profile_mem_evidence_completion_prompt import (
        CONVERSATION_PROFILE_EVIDENCE_COMPLETION_PROMPT,
    )
    
    # Group Profile 相关
    from memory_layer.prompts.zh.group_profile_prompts import (
        CONTENT_ANALYSIS_PROMPT,
        BEHAVIOR_ANALYSIS_PROMPT,
    )
    
    # Foresight 相关
    from memory_layer.prompts.zh.foresight_prompts import (
        get_group_foresight_generation_prompt,
        get_foresight_generation_prompt,
    )
    
    # Event Log 相关
    from memory_layer.prompts.zh.event_log_prompts import EVENT_LOG_PROMPT
    
else:
    # ===== 英文提示词（默认） =====
    # 对话相关
    from memory_layer.prompts.en.conv_prompts import CONV_BOUNDARY_DETECTION_PROMPT, CONV_SUMMARY_PROMPT
    
    # Episode 相关
    from memory_layer.prompts.en.episode_mem_prompts import (
        EPISODE_GENERATION_PROMPT,
        GROUP_EPISODE_GENERATION_PROMPT,
        DEFAULT_CUSTOM_INSTRUCTIONS,
    )
    
    # Profile 相关
    from memory_layer.prompts.en.profile_mem_prompts import CONVERSATION_PROFILE_EXTRACTION_PROMPT
    from memory_layer.prompts.en.profile_mem_part1_prompts import CONVERSATION_PROFILE_PART1_EXTRACTION_PROMPT
    from memory_layer.prompts.en.profile_mem_part2_prompts import CONVERSATION_PROFILE_PART2_EXTRACTION_PROMPT
    from memory_layer.prompts.en.profile_mem_part3_prompts import CONVERSATION_PROFILE_PART3_EXTRACTION_PROMPT
    from memory_layer.prompts.en.profile_mem_evidence_completion_prompt import (
        CONVERSATION_PROFILE_EVIDENCE_COMPLETION_PROMPT,
    )
    
    # Group Profile 相关
    from memory_layer.prompts.en.group_profile_prompts import (
        CONTENT_ANALYSIS_PROMPT,
        BEHAVIOR_ANALYSIS_PROMPT,
    )
    
    # Foresight 相关
    from memory_layer.prompts.en.foresight_prompts import (
        get_group_foresight_generation_prompt,
        get_foresight_generation_prompt,
    )
    
    # Event Log 相关
    from memory_layer.prompts.en.event_log_prompts import EVENT_LOG_PROMPT


# ============================================================================
# 导出函数 - 兼容旧 API
# ============================================================================


def get_current_language() -> str:
    """获取当前语言（兼容旧 API）"""
    return get_prompt_language()


def set_language(language: str) -> None:
    """设置语言（兼容旧 API，需要重启应用才能生效）"""
    if set_prompt_language(language):
        print(f"Language set to: {language}")
    else:
        print(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")


# 导出当前语言信息（兼容旧 API）
CURRENT_LANGUAGE = get_prompt_language()
MEMORY_LANGUAGE = CURRENT_LANGUAGE
