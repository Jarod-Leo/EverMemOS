"""
Agentic Retrieval 工具函数

提供 LLM 引导的多轮检索所需的工具：
1. Sufficiency Check: 判断检索结果是否充分
2. Multi-Query Generation: 生成多个互补查询
3. Query Refinement: 生成改进的查询（单查询模式）
4. Document Formatting: 格式化文档供 LLM 使用
"""

import json
import asyncio
from typing import List, Tuple, Optional

# 从新框架的 prompts 配置加载
from evaluation.src.utils.prompts import format_prompt


def format_documents_for_llm(
    results: List[Tuple[dict, float]],
    max_docs: int = 10,
    use_episode: bool = True
) -> str:
    """
    格式化检索结果供 LLM 使用
    
    Args:
        results: 检索结果列表 [(doc, score), ...]
        max_docs: 最多包含的文档数
        use_episode: True=使用 Episode Memory，False=使用 Event Log
    
    Returns:
        格式化的文档字符串
    """
    formatted_docs = []
    
    for i, (doc, score) in enumerate(results[:max_docs], start=1):
        subject = doc.get("subject", "N/A")
        
        if use_episode:
            # 使用 Episode Memory 格式（完整叙述）
            episode = doc.get("episode", "N/A")
            
            # 限制 episode 长度（避免 prompt 过长）
            if len(episode) > 500:
                episode = episode[:500] + "..."
            
            doc_text = (
                f"Document {i}:\n"
                f"  Title: {subject}\n"
                f"  Content: {episode}\n"
            )
            formatted_docs.append(doc_text)
        else:
            # 使用 Event Log 格式（原子事实）
            if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
                event_log = doc["event_log"]
                time_str = event_log.get("time", "N/A")
                atomic_facts = event_log.get("atomic_fact", [])
                
                if isinstance(atomic_facts, list) and atomic_facts:
                    # 格式化为：Document N: 标题 + 时间 + 事实列表
                    facts_text = "\n     ".join(atomic_facts[:5])  # 最多显示 5 个 facts
                    if len(atomic_facts) > 5:
                        facts_text += f"\n     ... and {len(atomic_facts) - 5} more facts"
                    
                    doc_text = (
                        f"Document {i}:\n"
                        f"  Title: {subject}\n"
                        f"  Time: {time_str}\n"
                        f"  Facts:\n"
                        f"     {facts_text}\n"
                    )
                    formatted_docs.append(doc_text)
                    continue
            
            # 如果没有 event_log，回退到 episode
            episode = doc.get("episode", "N/A")
            if len(episode) > 500:
                episode = episode[:500] + "..."
            
            doc_text = (
                f"Document {i}:\n"
                f"  Title: {subject}\n"
                f"  Content: {episode}\n"
            )
            formatted_docs.append(doc_text)
    
    return "\n".join(formatted_docs)


def parse_json_response(response: str) -> dict:
    """
    解析 LLM 的 JSON 响应（健壮处理）
    
    Args:
        response: LLM 原始响应字符串
    
    Returns:
        解析后的 JSON 字典
    """
    try:
        # 尝试提取 JSON（LLM 可能在前后添加额外文本）
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
        
        json_str = response[start_idx:end_idx]
        result = json.loads(json_str)
        
        # 验证必需字段
        if "is_sufficient" not in result:
            raise ValueError("Missing 'is_sufficient' field")
        
        # 设置默认值
        result.setdefault("reasoning", "No reasoning provided")
        result.setdefault("missing_information", [])
        
        return result
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠️  Failed to parse LLM response: {e}")
        print(f"  Raw response: {response[:200]}...")
        
        # 保守回退：假设充分（避免不必要的第二轮检索）
        return {
            "is_sufficient": True,
            "reasoning": f"Failed to parse: {str(e)}",
            "missing_information": []
        }


def parse_refined_query(response: str, original_query: str) -> str:
    """
    解析改进后的查询
    
    Args:
        response: LLM 响应
        original_query: 原始查询（用于回退）
    
    Returns:
        改进后的查询字符串
    """
    refined = response.strip()
    
    # 移除常见前缀
    prefixes = ["Refined Query:", "Output:", "Answer:", "Query:"]
    for prefix in prefixes:
        if refined.startswith(prefix):
            refined = refined[len(prefix):].strip()
    
    # 验证长度
    if len(refined) < 5 or len(refined) > 300:
        print(f"  ⚠️  Invalid refined query length ({len(refined)}), using original")
        return original_query
    
    # 避免与原查询完全相同
    if refined.lower() == original_query.lower():
        print(f"  ⚠️  Refined query identical to original, using original")
        return original_query
    
    return refined


async def check_sufficiency(
    query: str,
    results: List[Tuple[dict, float]],
    llm_provider,  # LLMProvider from Memory Layer
    llm_config: dict,
    max_docs: int = 10
) -> Tuple[bool, str, List[str]]:
    """
    检查检索结果是否充分
    
    Args:
        query: 用户查询
        results: 检索结果（Top 10）
        llm_provider: LLM Provider (Memory Layer)
        llm_config: LLM 配置字典
        max_docs: 最多评估的文档数
    
    Returns:
        (is_sufficient, reasoning, missing_information)
    """
    try:
        # 1. 格式化文档
        retrieved_docs = format_documents_for_llm(
            results, 
            max_docs=max_docs,
            use_episode=True
        )
        
        # 2. 使用新框架的 prompt 配置
        prompt = format_prompt(
            "agentic_sufficiency_check",
            query=query,
            retrieved_docs=retrieved_docs
        )
        
        # 3. 调用 LLM
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.0,  # 低温度，判断更稳定
            max_tokens=500,
        )
        
        # 4. 解析 JSON 响应
        result = parse_json_response(result_text)
        
        return (
            result["is_sufficient"],
            result["reasoning"],
            result.get("missing_information", [])
        )
    
    except asyncio.TimeoutError:
        print(f"  ❌ Sufficiency check timeout (30s)")
        # 超时回退：假设充分
        return True, "Timeout: LLM took too long", []
    except Exception as e:
        print(f"  ❌ Sufficiency check failed: {e}")
        import traceback
        traceback.print_exc()
        # 保守回退：假设充分
        return True, f"Error: {str(e)}", []


async def generate_refined_query(
    original_query: str,
    results: List[Tuple[dict, float]],
    missing_info: List[str],
    llm_provider,
    llm_config: dict,
    max_docs: int = 10
) -> str:
    """
    生成改进的查询（单查询模式，用于回退）
    
    Args:
        original_query: 原始查询
        results: Round 1 检索结果（Top 10）
        missing_info: 缺失的信息列表
        llm_provider: LLM Provider
        llm_config: LLM 配置
        max_docs: 最多使用的文档数
    
    Returns:
        改进后的查询字符串
    """
    try:
        # 1. 格式化文档和缺失信息
        retrieved_docs = format_documents_for_llm(
            results, 
            max_docs=max_docs,
            use_episode=True
        )
        missing_info_str = ", ".join(missing_info) if missing_info else "N/A"
        
        # 2. 使用新框架的 prompt 配置
        prompt = format_prompt(
            "agentic_refined_query",
            original_query=original_query,
            retrieved_docs=retrieved_docs,
            missing_info=missing_info_str
        )
        
        # 3. 调用 LLM
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.3,  # 稍高温度，增加创造性
            max_tokens=150,
        )
        
        # 4. 解析和验证
        refined_query = parse_refined_query(result_text, original_query)
        
        return refined_query
    
    except asyncio.TimeoutError:
        print(f"  ❌ Query refinement timeout (30s)")
        return original_query
    except Exception as e:
        print(f"  ❌ Query refinement failed: {e}")
        import traceback
        traceback.print_exc()
        return original_query


def parse_multi_query_response(response: str, original_query: str) -> Tuple[List[str], str]:
    """
    解析多查询生成的 JSON 响应
    
    Args:
        response: LLM 原始响应字符串
        original_query: 原始查询（用于回退）
    
    Returns:
        (queries_list, reasoning)
    """
    try:
        # 提取 JSON
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
        
        json_str = response[start_idx:end_idx]
        result = json.loads(json_str)
        
        # 验证必需字段
        if "queries" not in result or not isinstance(result["queries"], list):
            raise ValueError("Missing or invalid 'queries' field")
        
        queries = result["queries"]
        reasoning = result.get("reasoning", "No reasoning provided")
        
        # 过滤和验证查询
        valid_queries = []
        for q in queries:
            if isinstance(q, str) and 5 <= len(q) <= 300:
                # 避免与原查询完全相同
                if q.lower().strip() != original_query.lower().strip():
                    valid_queries.append(q.strip())
        
        # 至少返回 1 个查询
        if not valid_queries:
            print(f"  ⚠️  No valid queries generated, using original")
            return [original_query], "Fallback: used original query"
        
        # 限制最多 3 个查询
        valid_queries = valid_queries[:3]
        
        print(f"  ✅ Generated {len(valid_queries)} valid queries")
        return valid_queries, reasoning
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠️  Failed to parse multi-query response: {e}")
        print(f"  Raw response: {response[:200]}...")
        
        # 回退：返回原始查询
        return [original_query], f"Parse error: {str(e)}"


async def generate_multi_queries(
    original_query: str,
    results: List[Tuple[dict, float]],
    missing_info: List[str],
    llm_provider,
    llm_config: dict,
    max_docs: int = 5,
    num_queries: int = 3
) -> Tuple[List[str], str]:
    """
    生成多个互补的查询（用于多查询检索）
    
    Args:
        original_query: 原始查询
        results: Round 1 检索结果（Top 5）
        missing_info: 缺失的信息列表
        llm_provider: LLM Provider
        llm_config: LLM 配置
        max_docs: 最多使用的文档数（默认 5）
        num_queries: 期望生成的查询数量（默认 3，实际可能更少）
    
    Returns:
        (queries_list, reasoning)
        queries_list: 生成的查询列表（1-3 个）
        reasoning: LLM 的生成策略说明
    """
    try:
        # 1. 格式化文档和缺失信息
        retrieved_docs = format_documents_for_llm(
            results, 
            max_docs=max_docs,
            use_episode=True
        )
        missing_info_str = ", ".join(missing_info) if missing_info else "N/A"
        
        # 2. 使用新框架的 prompt 配置
        prompt = format_prompt(
            "agentic_multi_query",
            original_query=original_query,
            retrieved_docs=retrieved_docs,
            missing_info=missing_info_str
        )
        
        # 3. 调用 LLM
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.4,  # 稍高温度，增加查询多样性
            max_tokens=300,  # 增加 token 数以支持多个查询
        )
        
        # 4. 解析和验证
        queries, reasoning = parse_multi_query_response(result_text, original_query)
        
        print(f"  [Multi-Query] Generated {len(queries)} queries:")
        for i, q in enumerate(queries, 1):
            print(f"    Query {i}: {q[:80]}{'...' if len(q) > 80 else ''}")
        print(f"  [Multi-Query] Strategy: {reasoning}")
        
        return queries, reasoning
    
    except asyncio.TimeoutError:
        print(f"  ❌ Multi-query generation timeout (30s)")
        # 超时回退：使用原始查询
        return [original_query], "Timeout: used original query"
    except Exception as e:
        print(f"  ❌ Multi-query generation failed: {e}")
        import traceback
        traceback.print_exc()
        # 回退到原始查询
        return [original_query], f"Error: {str(e)}"

