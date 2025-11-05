"""
LLM Judge 评估器

使用 LLM 作为评判器来评估答案的正确性。
"""
import asyncio
import json
from typing import List
from openai import AsyncOpenAI

from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.evaluators.registry import register_evaluator
from evaluation.src.core.data_models import AnswerResult, EvaluationResult
from evaluation.src.utils.prompts import get_prompt, format_prompt


@register_evaluator("llm_judge")
class LLMJudge(BaseEvaluator):
    """LLM 评判器"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # 初始化 OpenAI 客户端
        llm_config = config.get("llm", {})
        self.client = AsyncOpenAI(
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1")
        )
        self.model = llm_config.get("model", "gpt-4o-mini")
        self.num_runs = config.get("num_runs", 3)
    
    async def evaluate(
        self, 
        answer_results: List[AnswerResult]
    ) -> EvaluationResult:
        """
        使用 LLM 评估答案
        
        Args:
            answer_results: 答案结果列表
            
        Returns:
            EvaluationResult: 评估结果
        """
        print(f"\n{'='*60}")
        print(f"Evaluation: LLM Judge (model={self.model}, runs={self.num_runs})")
        print(f"{'='*60}")
        
        detailed_results = []
        total_correct = 0
        
        # 并发评估所有答案
        semaphore = asyncio.Semaphore(10)  # 限制并发数
        
        async def evaluate_single(answer_result: AnswerResult):
            async with semaphore:
                return await self._evaluate_single_answer(answer_result)
        
        tasks = [evaluate_single(ar) for ar in answer_results]
        results = await asyncio.gather(*tasks)
        
        # 统计结果
        for result in results:
            detailed_results.append(result)
            if result.get("is_correct"):
                total_correct += 1
        
        accuracy = total_correct / len(answer_results) if answer_results else 0.0
        
        print(f"\n✅ 评估完成:")
        print(f"   - 总问题数: {len(answer_results)}")
        print(f"   - 正确: {total_correct}")
        print(f"   - 准确率: {accuracy:.2%}")
        
        return EvaluationResult(
            total_questions=len(answer_results),
            correct=total_correct,
            accuracy=accuracy,
            detailed_results=detailed_results,
            metadata={
                "model": self.model,
                "num_runs": self.num_runs
            }
        )
    
    async def _evaluate_single_answer(self, answer_result: AnswerResult) -> dict:
        """评估单个答案"""
        question = answer_result.question
        golden_answer = answer_result.golden_answer
        generated_answer = answer_result.answer
        
        # 多次评估取多数投票
        judgments = []
        for _ in range(self.num_runs):
            is_correct = await self._judge_answer(
                question, golden_answer, generated_answer
            )
            judgments.append(is_correct)
        
        # 多数投票
        is_correct = sum(judgments) > len(judgments) / 2
        
        return {
            "question_id": answer_result.question_id,
            "question": question,
            "golden_answer": golden_answer,
            "generated_answer": generated_answer,
            "is_correct": is_correct,
            "judgments": judgments,
            "category": answer_result.category,
        }
    
    async def _judge_answer(
        self, 
        question: str, 
        golden_answer: str, 
        generated_answer: str
    ) -> bool:
        """
        使用 LLM 判断答案是否正确
        
        Returns:
            True 如果正确，False 如果错误
        """
        # 使用配置化的 prompts
        system_prompt = get_prompt("llm_judge", "system_prompt")
        user_prompt = format_prompt(
            "llm_judge",
            "user_prompt",
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            label = result.get("label", "WRONG")
            
            return label.strip().upper() == "CORRECT"
        
        except Exception as e:
            print(f"  ⚠️ LLM Judge 失败: {e}")
            return False

