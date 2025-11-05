"""
数据加载器

提供不同数据集的加载功能。
支持自动转换非 Locomo 格式的数据集。
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from evaluation.src.core.data_models import Dataset, Conversation, Message, QAPair
from evaluation.src.converters.registry import get_converter


def load_dataset(dataset_name: str, data_path: str) -> Dataset:
    """
    智能加载数据集（支持自动转换）
    
    Args:
        dataset_name: 数据集名称（如 "locomo", "longmemeval", "personamem"）
        data_path: 数据文件路径或目录路径
        
    Returns:
        Dataset: 标准格式数据集
    """
    data_path_obj = Path(data_path)
    
    # 检查是否需要转换
    converter = get_converter(dataset_name)
    
    if converter:
        # 需要转换的数据集
        if data_path_obj.is_file():
            # 如果给的是文件路径，取其父目录
            data_dir = data_path_obj.parent
        else:
            data_dir = data_path_obj
        
        # 检查是否需要转换
        if converter.needs_conversion(data_dir):
            print(f"📝 Converted file not found, converting {dataset_name}...")
            
            # 构建输入文件路径
            input_files = converter.get_input_files()
            input_paths = {
                key: str(data_dir / filename)
                for key, filename in input_files.items()
            }
            
            # 执行转换
            output_path = str(converter.get_converted_path(data_dir))
            converter.convert(input_paths, output_path)
        
        # 使用 converted 文件
        locomo_file = converter.get_converted_path(data_dir)
    else:
        # 原生 Locomo 格式，直接使用
        if data_path_obj.is_file():
            locomo_file = data_path_obj
        else:
            # 如果是目录，尝试找到 .json 文件
            json_files = list(data_path_obj.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON file found in {data_path_obj}")
            locomo_file = json_files[0]
    
    # 加载 Locomo 格式数据
    return load_locomo_dataset(str(locomo_file))


def load_locomo_dataset(data_path: str) -> Dataset:
    """
    加载 LoCoMo 格式的数据集
    
    Args:
        data_path: Locomo 格式数据文件路径
        
    Returns:
        Dataset: 标准格式数据集
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    conversations = []
    qa_pairs = []
    
    for idx, item in enumerate(raw_data):
        conv_id = str(idx)
        conversation_data = item.get("conversation", {})
        qa_data = item.get("qa", [])
        
        # 转换对话
        conversation = _convert_locomo_conversation(conversation_data, conv_id)
        conversations.append(conversation)
        
        # 转换 QA 对
        for qa_item in qa_data:
            qa_pair = _convert_locomo_qa_pair(qa_item, conv_id)
            qa_pairs.append(qa_pair)
    
    return Dataset(
        dataset_name="locomo",
        conversations=conversations,
        qa_pairs=qa_pairs,
        metadata={"total_conversations": len(conversations)}
    )


def _convert_locomo_conversation(conversation_data: dict, conv_id: str) -> Conversation:
    """转换 LoCoMo 对话"""
    messages = []
    
    # 获取所有 session keys
    session_keys = sorted([
        key for key in conversation_data.keys()
        if key.startswith("session_") and not key.endswith("_date_time")
    ])
    
    for session_key in session_keys:
        session_messages = conversation_data[session_key]
        session_time_key = f"{session_key}_date_time"
        
        if session_time_key in conversation_data:
            # 解析 session 时间戳（可能为 None）
            session_time = _parse_locomo_timestamp(conversation_data[session_time_key])
            
            # 转换每条消息
            for msg_idx, msg in enumerate(session_messages):
                # 如果有时间信息，每条消息间隔 30 秒；否则 timestamp 为 None
                if session_time is not None:
                    msg_timestamp = session_time + timedelta(seconds=msg_idx * 30)
                else:
                    msg_timestamp = None
                
                message = Message(
                    speaker_id=f"{msg['speaker'].lower().replace(' ', '_')}_{conv_id}",
                    speaker_name=msg['speaker'],
                    content=msg['text'],
                    timestamp=msg_timestamp,
                    metadata={
                        "session": session_key,
                        "dia_id": msg.get("dia_id"),
                        "img_url": msg.get("img_url"),
                        "blip_caption": msg.get("blip_caption"),
                    }
                )
                messages.append(message)
    
    return Conversation(
        conversation_id=conv_id,
        messages=messages,
        metadata={
            "speaker_a": conversation_data.get("speaker_a"),
            "speaker_b": conversation_data.get("speaker_b"),
        }
    )


def _convert_locomo_qa_pair(qa_item: dict, conv_id: str) -> QAPair:
    """转换 LoCoMo QA 对"""
    return QAPair(
        question_id=f"{conv_id}_{qa_item.get('question', '')[:20]}",
        question=qa_item.get("question", ""),
        answer=qa_item.get("answer", ""),
        category=qa_item.get("category"),
        evidence=qa_item.get("evidence", []),
        metadata={"conversation_id": conv_id}
    )


def _parse_locomo_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    解析 LoCoMo 的时间格式
    
    输入格式: "6:07 pm on 13 January, 2023"
    特殊值: "Unknown" 或无法解析时返回 None
    输出: datetime 对象或 None
    """
    # 清理字符串
    timestamp_str = timestamp_str.replace("\\s+", " ").strip()
    
    # 处理特殊情况：Unknown 或空字符串
    if timestamp_str.lower() == "unknown" or not timestamp_str:
        # 没有时间信息，返回 None
        return None
    
    try:
        return datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
    except ValueError:
        # 如果解析失败，返回 None 并输出警告
        print(f"⚠️  Warning: Failed to parse timestamp '{timestamp_str}', no timestamp will be set")
        return None

