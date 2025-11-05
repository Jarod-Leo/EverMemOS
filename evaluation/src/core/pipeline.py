"""
Pipeline 核心

评测流程的编排器，负责协调 Add → Search → Answer → Evaluate 四个阶段。
"""
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from evaluation.src.core.data_models import (
    Dataset, SearchResult, AnswerResult, EvaluationResult
)
from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.utils.logger import setup_logger, get_console
from evaluation.src.utils.saver import ResultSaver
from evaluation.src.utils.checkpoint import CheckpointManager

# 导入答案生成所需的组件
from memory_layer.llm.llm_provider import LLMProvider
# format_prompt 已移除，answer 阶段由各 adapter 自行处理


class Pipeline:
    """
    评测 Pipeline
    
    四阶段流程：
    1. Add: 摄入对话数据并构建索引
    2. Search: 检索相关记忆
    3. Answer: 生成答案
    4. Evaluate: 评估答案质量
    """
    
    def __init__(
        self,
        adapter: BaseAdapter,
        evaluator: BaseEvaluator,
        llm_provider: LLMProvider,
        output_dir: Path,
        run_name: str = "default",
        use_checkpoint: bool = True,
    ):
        """
        初始化 Pipeline
        
        Args:
            adapter: 系统适配器
            evaluator: 评估器
            llm_provider: LLM Provider（用于答案生成）
            output_dir: 输出目录
            run_name: 运行名称（用于区分不同运行）
            use_checkpoint: 是否启用断点续传
        """
        self.adapter = adapter
        self.evaluator = evaluator
        self.llm_provider = llm_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(self.output_dir / "pipeline.log")
        self.saver = ResultSaver(self.output_dir)
        self.console = get_console()
        
        # 断点续传支持
        self.use_checkpoint = use_checkpoint
        self.checkpoint = CheckpointManager(output_dir=output_dir, run_name=run_name) if use_checkpoint else None
        self.completed_stages: set = set()
    
    async def run(
        self,
        dataset: Dataset,
        stages: Optional[List[str]] = None,
        smoke_test: bool = False,
        smoke_messages: int = 10,
        smoke_questions: int = 3,
    ) -> Dict[str, Any]:
        """
        运行完整 Pipeline
        
        Args:
            dataset: 标准格式数据集
            stages: 要执行的阶段列表，None 表示全部
                   可选值: ["add", "search", "answer", "evaluate"]
            smoke_test: 是否为冒烟测试
            smoke_messages: 冒烟测试时的消息数量（默认10）
            smoke_questions: 冒烟测试时的问题数量（默认3）
            
        Returns:
            评测结果字典
        """
        start_time = time.time()
        
        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print("🚀 Evaluation Pipeline", style="bold cyan")
        self.console.print(f"{'='*60}", style="bold cyan")
        self.console.print(f"Dataset: {dataset.dataset_name}")
        self.console.print(f"System: {self.adapter.get_system_info()['name']}")
        self.console.print(f"Stages: {stages or 'all'}")
        if smoke_test:
            self.console.print(f"[yellow]🧪 Smoke Test Mode: {smoke_messages} messages, {smoke_questions} questions[/yellow]")
        self.console.print(f"{'='*60}\n", style="bold cyan")
        
        # 冒烟测试：只处理第一个对话的前 K 条消息和前 K 个问题
        if smoke_test:
            dataset = self._apply_smoke_test(dataset, smoke_messages, smoke_questions)
            self.console.print(f"[yellow]✂️  Smoke test applied:[/yellow]")
            self.console.print(f"[yellow]   - Conversation: {dataset.conversations[0].conversation_id}[/yellow]")
            self.console.print(f"[yellow]   - Messages: {len(dataset.conversations[0].messages)}[/yellow]")
            self.console.print(f"[yellow]   - Questions: {len(dataset.qa_pairs)}[/yellow]\n")
        
        # 过滤掉 Category 5（对抗性问题）
        original_qa_count = len(dataset.qa_pairs)
        dataset.qa_pairs = [qa for qa in dataset.qa_pairs if qa.category != 5]
        filtered_count = original_qa_count - len(dataset.qa_pairs)
        
        if filtered_count > 0:
            self.console.print(f"[dim]🔍 Filtered out {filtered_count} category-5 (adversarial) questions[/dim]")
            self.console.print(f"[dim]   Remaining questions: {len(dataset.qa_pairs)}[/dim]\n")
        
        # 尝试加载 checkpoint
        search_results_data = None
        answer_results_data = None
        
        if self.use_checkpoint and self.checkpoint:
            checkpoint_data = self.checkpoint.load_checkpoint()
            if checkpoint_data:
                self.completed_stages = set(checkpoint_data.get('completed_stages', []))
                # 加载已保存的中间结果
                if 'search_results' in checkpoint_data:
                    search_results_data = checkpoint_data['search_results']
                if 'answer_results' in checkpoint_data:
                    answer_results_data = checkpoint_data['answer_results']
        
        # 默认执行所有阶段
        if stages is None:
            stages = ["add", "search", "answer", "evaluate"]
        
        results = {}
        
        # ===== Stage 1: Add =====
        if "add" in stages and "add" not in self.completed_stages:
            self.logger.info("Starting Stage 1: Add")
            
            # 🔥 传递 checkpoint_manager 以支持细粒度断点续传
            index = await self.adapter.add(
                conversations=dataset.conversations,
                output_dir=self.output_dir,
                checkpoint_manager=self.checkpoint
            )
            
            # 索引元数据（延迟加载，无需持久化）
            results["index"] = index
            self.logger.info("✅ Stage 1 completed")
            
            # 保存 checkpoint
            self.completed_stages.add("add")
            if self.checkpoint:
                self.checkpoint.save_checkpoint(self.completed_stages)
        elif "add" in self.completed_stages:
            self.console.print("\n[yellow]⏭️  Skip Add stage (already completed)[/yellow]")
            # 重新构建索引元数据（延迟加载不需要 pkl 文件）
            index = {
                "type": "lazy_load",
                "memcells_dir": str(self.output_dir / "memcells"),
                "bm25_index_dir": str(self.output_dir / "bm25_index"),
                "emb_index_dir": str(self.output_dir / "vectors"),
                "conversation_ids": [conv.conversation_id for conv in dataset.conversations],
                "use_hybrid_search": True,
                "total_conversations": len(dataset.conversations),
            }
            results["index"] = index
        else:
            # 重新构建索引元数据
            index = {
                "type": "lazy_load",
                "memcells_dir": str(self.output_dir / "memcells"),
                "bm25_index_dir": str(self.output_dir / "bm25_index"),
                "emb_index_dir": str(self.output_dir / "vectors"),
                "conversation_ids": [conv.conversation_id for conv in dataset.conversations],
                "use_hybrid_search": True,
                "total_conversations": len(dataset.conversations),
            }
            results["index"] = index
            self.logger.info("⏭️  Skipped Stage 1, using lazy loading")
        
        # ===== Stage 2: Search =====
        if "search" in stages and "search" not in self.completed_stages:
            self.logger.info("Starting Stage 2: Search")
            
            search_results = await self._run_search(
                dataset.qa_pairs,
                index
            )
            
            self.saver.save_json(
                [self._search_result_to_dict(sr) for sr in search_results],
                "search_results.json"
            )
            results["search_results"] = search_results
            self.logger.info("✅ Stage 2 completed")
            
            # 保存 checkpoint
            self.completed_stages.add("search")
            if self.checkpoint:
                search_results_data = [self._search_result_to_dict(sr) for sr in search_results]
                self.checkpoint.save_checkpoint(
                    self.completed_stages,
                    search_results=search_results_data
                )
        elif "search" in self.completed_stages:
            self.console.print(f"\n[yellow]⏭️  Skip Search stage (already completed)[/yellow]")
            if search_results_data:
                # 从 checkpoint 加载
                search_results = [self._dict_to_search_result(d) for d in search_results_data]
                results["search_results"] = search_results
            elif self.saver.file_exists("search_results.json"):
                # 从文件加载
                search_data = self.saver.load_json("search_results.json")
                search_results = [self._dict_to_search_result(d) for d in search_data]
                results["search_results"] = search_results
        else:
            if self.saver.file_exists("search_results.json"):
                search_data = self.saver.load_json("search_results.json")
                search_results = [self._dict_to_search_result(d) for d in search_data]
                results["search_results"] = search_results
                self.logger.info("⏭️  Skipped Stage 2, loaded existing results")
            else:
                raise FileNotFoundError("Search results not found. Please run 'search' stage first.")
        
        # ===== Stage 3: Answer =====
        if "answer" in stages and "answer" not in self.completed_stages:
            self.logger.info("Starting Stage 3: Answer")
            
            answer_results = await self._run_answer(
                dataset.qa_pairs,
                search_results
            )
            
            self.saver.save_json(
                [self._answer_result_to_dict(ar) for ar in answer_results],
                "answer_results.json"
            )
            results["answer_results"] = answer_results
            self.logger.info("✅ Stage 3 completed")
            
            # 保存 checkpoint
            self.completed_stages.add("answer")
            if self.checkpoint:
                answer_results_dict = [self._answer_result_to_dict(ar) for ar in answer_results]
                self.checkpoint.save_checkpoint(
                    self.completed_stages,
                    search_results=search_results_data if search_results_data else [self._search_result_to_dict(sr) for sr in search_results],
                    answer_results=answer_results_dict
                )
        elif "answer" in self.completed_stages:
            self.console.print(f"\n[yellow]⏭️  Skip Answer stage (already completed)[/yellow]")
            if answer_results_data:
                # 从 checkpoint 加载
                answer_results = [self._dict_to_answer_result(d) for d in answer_results_data]
                results["answer_results"] = answer_results
            elif self.saver.file_exists("answer_results.json"):
                # 从文件加载
                answer_data = self.saver.load_json("answer_results.json")
                answer_results = [self._dict_to_answer_result(d) for d in answer_data]
                results["answer_results"] = answer_results
        else:
            if self.saver.file_exists("answer_results.json"):
                answer_data = self.saver.load_json("answer_results.json")
                answer_results = [self._dict_to_answer_result(d) for d in answer_data]
                results["answer_results"] = answer_results
                self.logger.info("⏭️  Skipped Stage 3, loaded existing results")
            else:
                raise FileNotFoundError("Answer results not found. Please run 'answer' stage first.")
        
        # ===== Stage 4: Evaluate =====
        if "evaluate" in stages and "evaluate" not in self.completed_stages:
            self.logger.info("Starting Stage 4: Evaluate")
            
            eval_result = await self.evaluator.evaluate(answer_results)
            
            self.saver.save_json(
                self._eval_result_to_dict(eval_result),
                "eval_results.json"
            )
            results["eval_result"] = eval_result
            self.logger.info("✅ Stage 4 completed")
            
            # 保存 checkpoint
            self.completed_stages.add("evaluate")
            if self.checkpoint:
                self.checkpoint.save_checkpoint(
                    self.completed_stages,
                    search_results=search_results_data if search_results_data else [self._search_result_to_dict(sr) for sr in search_results],
                    answer_results=answer_results_data if answer_results_data else [self._answer_result_to_dict(ar) for ar in answer_results],
                    eval_results=self._eval_result_to_dict(eval_result)
                )
        elif "evaluate" in self.completed_stages:
            self.console.print("\n[yellow]⏭️  Skip Evaluate stage (already completed)[/yellow]")
        
        # # 所有阶段完成后，删除 checkpoint
        # if self.checkpoint and len(self.completed_stages) == 4:
        #     self.console.print("\n[dim]🧹 Cleaning up checkpoint...[/dim]")
        #     self.checkpoint.delete_checkpoint()
        
        # 生成报告
        elapsed_time = time.time() - start_time
        self._generate_report(results, elapsed_time)
        
        return results
    
    def _apply_smoke_test(
        self, 
        dataset: Dataset, 
        num_messages: int, 
        num_questions: int
    ) -> Dataset:
        """
        应用冒烟测试：只保留第一个对话的前 N 条消息和前 M 个问题
        
        这样可以快速验证完整流程（Add → Search → Answer → Evaluate），
        但只使用少量数据，节省时间。
        
        Args:
            dataset: 原始数据集
            num_messages: 保留的消息数量（用于 Add 阶段），0 表示所有消息
            num_questions: 保留的问题数量（用于 Search/Answer/Evaluate 阶段），0 表示所有问题
            
        Returns:
            裁剪后的数据集
        """
        if not dataset.conversations:
            return dataset
        
        # 只保留第一个对话
        first_conv = dataset.conversations[0]
        conv_id = first_conv.conversation_id
        
        # 截取前 N 条消息（用于 Add）
        # 0 表示保留所有消息
        if num_messages > 0:
            total_messages = len(first_conv.messages)
            first_conv.messages = first_conv.messages[:num_messages]
            msg_desc = f"{len(first_conv.messages)}/{total_messages}"
        else:
            msg_desc = f"{len(first_conv.messages)} (all)"
        
        # 只保留该对话的前 M 个问题（用于 Search/Answer/Evaluate）
        # 0 表示保留所有问题
        conv_qa_pairs = [
            qa for qa in dataset.qa_pairs 
            if qa.metadata.get("conversation_id") == conv_id
        ]
        if num_questions > 0:
            total_questions = len(conv_qa_pairs)
            selected_qa_pairs = conv_qa_pairs[:num_questions]
            qa_desc = f"{len(selected_qa_pairs)}/{total_questions}"
        else:
            selected_qa_pairs = conv_qa_pairs
            qa_desc = f"{len(selected_qa_pairs)} (all)"
        
        self.logger.info(
            f"Smoke test: Conv {conv_id} - "
            f"{msg_desc} messages, "
            f"{qa_desc} questions"
        )
        
        return Dataset(
            dataset_name=dataset.dataset_name + "_smoke",
            conversations=[first_conv],
            qa_pairs=selected_qa_pairs,
            metadata={
                **dataset.metadata, 
                "smoke_test": True, 
                "smoke_messages": num_messages if num_messages > 0 else len(first_conv.messages),
                "smoke_questions": num_questions if num_questions > 0 else len(selected_qa_pairs),
            }
        )
    
    async def _run_search(
        self,
        qa_pairs: List,
        index: Any
    ) -> List[SearchResult]:
        """
        并发执行检索，支持细粒度 checkpoint
        
        按会话分组处理，每处理完一个会话就保存 checkpoint（和 archive 的 stage3 一致）
        """
        print(f"\n{'='*60}")
        print(f"Stage 2/4: Search")
        print(f"{'='*60}")
        
        # 🔥 加载细粒度 checkpoint
        all_search_results_dict = {}
        if self.checkpoint:
            all_search_results_dict = self.checkpoint.load_search_progress()
        
        # 按会话分组 QA 对
        conv_to_qa = {}
        for qa in qa_pairs:
            conv_id = qa.metadata.get("conversation_id", "unknown")
            if conv_id not in conv_to_qa:
                conv_to_qa[conv_id] = []
            conv_to_qa[conv_id].append(qa)
        
        total_convs = len(conv_to_qa)
        processed_convs = set(all_search_results_dict.keys())
        remaining_convs = set(conv_to_qa.keys()) - processed_convs
        
        print(f"Total conversations: {total_convs}")
        print(f"Total questions: {len(qa_pairs)}")
        if processed_convs:
            print(f"Already processed: {len(processed_convs)} conversations (from checkpoint)")
            print(f"Remaining: {len(remaining_convs)} conversations")
        
        semaphore = asyncio.Semaphore(20)
        
        async def search_single(qa):
            async with semaphore:
                conv_id = qa.metadata.get("conversation_id", "0")
                return await self.adapter.search(qa.question, conv_id, index)
        
        # 按会话逐个处理（和 archive 一致）
        for idx, (conv_id, qa_list) in enumerate(sorted(conv_to_qa.items())):
            # 🔥 跳过已处理的会话
            if conv_id in processed_convs:
                print(f"\n⏭️  Skipping Conversation ID: {conv_id} (already processed)")
                continue
            
            print(f"\n--- Processing Conversation ID: {conv_id} ({idx+1}/{total_convs}) ---")
            print(f"    Questions in this conversation: {len(qa_list)}")
            
            # 并发处理这个会话的所有问题
            tasks = [search_single(qa) for qa in qa_list]
            results_for_conv = await asyncio.gather(*tasks)
            
            # 将结果保存为字典格式
            results_for_conv_dict = [
                {
                    "question_id": qa.question_id,
                    "query": qa.question,
                    "conversation_id": conv_id,
                    "results": result.results,
                    "retrieval_metadata": result.retrieval_metadata
                }
                for qa, result in zip(qa_list, results_for_conv)
            ]
            
            all_search_results_dict[conv_id] = results_for_conv_dict
            
            # 🔥 每处理完一个会话就保存检查点（和 archive 一致）
            if self.checkpoint:
                self.checkpoint.save_search_progress(all_search_results_dict)
        
        # 🔥 完成后删除细粒度检查点
        if self.checkpoint:
            self.checkpoint.delete_search_checkpoint()
        
        # 将字典格式转换为 SearchResult 对象列表（保持原有返回格式）
        all_results = []
        for conv_id in sorted(conv_to_qa.keys()):
            if conv_id in all_search_results_dict:
                for result_dict in all_search_results_dict[conv_id]:
                    all_results.append(SearchResult(
                        query=result_dict["query"],
                        conversation_id=result_dict["conversation_id"],
                        results=result_dict["results"],
                        retrieval_metadata=result_dict.get("retrieval_metadata", {})
                    ))
        
        print(f"\n{'='*60}")
        print(f"🎉 All conversations processed!")
        print(f"{'='*60}")
        print(f"✅ Search completed: {len(all_results)} results\n")
        return all_results
    
    async def _run_answer(
        self,
        qa_pairs: List,
        search_results: List[SearchResult]
    ) -> List[AnswerResult]:
        """
        生成答案，支持细粒度 checkpoint
        
        每 SAVE_INTERVAL 个问题保存一次 checkpoint（和 archive 的 stage4 一致）
        """
        print(f"\n{'='*60}")
        print(f"Stage 3/4: Answer")
        print(f"{'='*60}")
        
        SAVE_INTERVAL = 400  # 🔥 和 archive 保持一致：每 400 个任务保存一次
        MAX_CONCURRENT = 50  # 最大并发数
        
        # 🔥 加载细粒度 checkpoint
        all_answer_results = {}
        if self.checkpoint:
            loaded_results = self.checkpoint.load_answer_progress()
            # 转换为 {question_id: AnswerResult} 格式
            for result in loaded_results.values():
                all_answer_results[result["question_id"]] = result
        
        total_qa_count = len(qa_pairs)
        processed_count = len(all_answer_results)
        
        print(f"Total questions: {total_qa_count}")
        if processed_count > 0:
            print(f"Already processed: {processed_count} questions (from checkpoint)")
            print(f"Remaining: {total_qa_count - processed_count} questions")
        
        # 准备待处理的任务
        pending_tasks = []
        for qa, sr in zip(qa_pairs, search_results):
            if qa.question_id not in all_answer_results:
                pending_tasks.append((qa, sr))
        
        if not pending_tasks:
            print(f"✅ All questions already processed!")
            # 转换为 AnswerResult 对象列表（按原始顺序）
            results = []
            for qa in qa_pairs:
                if qa.question_id in all_answer_results:
                    result_dict = all_answer_results[qa.question_id]
                    results.append(AnswerResult(
                        question_id=result_dict["question_id"],
                        question=result_dict["question"],
                        answer=result_dict["answer"],
                        golden_answer=result_dict["golden_answer"],
                        category=result_dict.get("category"),
                        conversation_id=result_dict.get("conversation_id", ""),
                        search_results=result_dict.get("search_results", []),
                    ))
            return results
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        completed = processed_count
        failed = 0
        start_time = time.time()
        
        async def answer_single_with_tracking(qa, search_result):
            nonlocal completed, failed
            
            async with semaphore:
                try:
                    # 构建 context
                    context = self._build_context(search_result)
                    
                    # 🔥 直接调用 adapter 的 answer 方法
                    answer = await self.adapter.answer(
                        query=qa.question,
                        context=context,
                        conversation_id=search_result.conversation_id,
                    )
                    
                    answer = answer.strip()
                
                except Exception as e:
                    print(f"  ⚠️ Answer generation failed for {qa.question_id}: {e}")
                    answer = "Error: Failed to generate answer"
                    failed += 1
                
                result = AnswerResult(
                    question_id=qa.question_id,
                    question=qa.question,
                    answer=answer,
                    golden_answer=qa.answer,
                    category=qa.category,
                    conversation_id=search_result.conversation_id,
                    search_results=search_result.results,
                )
                
                # 保存结果
                all_answer_results[qa.question_id] = {
                    "question_id": result.question_id,
                    "question": result.question,
                    "answer": result.answer,
                    "golden_answer": result.golden_answer,
                    "category": result.category,
                    "conversation_id": result.conversation_id,
                    "search_results": result.search_results,
                }
                
                completed += 1
                
                # 🔥 定期保存 checkpoint（和 archive 一致）
                if self.checkpoint and (completed % SAVE_INTERVAL == 0 or completed == total_qa_count):
                    elapsed = time.time() - start_time
                    speed = completed / elapsed if elapsed > 0 else 0
                    eta = (total_qa_count - completed) / speed if speed > 0 else 0
                    
                    print(f"Progress: {completed}/{total_qa_count} ({completed/total_qa_count*100:.1f}%) | "
                          f"Speed: {speed:.1f} qa/s | Failed: {failed} | ETA: {eta/60:.1f} min")
                    
                    self.checkpoint.save_answer_progress(all_answer_results, completed, total_qa_count)
                
                return result
        
        # 创建所有待处理的任务
        tasks = [
            answer_single_with_tracking(qa, sr)
            for qa, sr in pending_tasks
        ]
        
        # 并发执行
        await asyncio.gather(*tasks)
        
        # 统计信息
        elapsed_time = time.time() - start_time
        success_rate = (completed - failed) / completed * 100 if completed > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"✅ All responses generated!")
        print(f"   - Total questions: {total_qa_count}")
        print(f"   - Successful: {completed - failed}")
        print(f"   - Failed: {failed}")
        print(f"   - Success rate: {success_rate:.1f}%")
        print(f"   - Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")
        print(f"   - Average speed: {total_qa_count/elapsed_time:.1f} qa/s")
        print(f"{'='*60}\n")
        
        # 🔥 完成后删除细粒度检查点（和 archive 一致）
        if self.checkpoint:
            self.checkpoint.delete_answer_checkpoints()
        
        # 转换为 AnswerResult 对象列表（按原始顺序）
        results = []
        for qa in qa_pairs:
            if qa.question_id in all_answer_results:
                result_dict = all_answer_results[qa.question_id]
                results.append(AnswerResult(
                    question_id=result_dict["question_id"],
                    question=result_dict["question"],
                    answer=result_dict["answer"],
                    golden_answer=result_dict["golden_answer"],
                    category=result_dict.get("category"),
                    conversation_id=result_dict.get("conversation_id", ""),
                    search_results=result_dict.get("search_results", []),
                ))
        
        return results
    
    def _build_context(self, search_result: SearchResult) -> str:
        """从检索结果构建上下文"""
        context_parts = []
        
        for idx, result in enumerate(search_result.results[:10], 1):
            content = result.get("content", "")
            context_parts.append(f"{idx}. {content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_report(self, results: Dict[str, Any], elapsed_time: float):
        """生成评测报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("📊 Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 系统信息
        system_info = self.adapter.get_system_info()
        report_lines.append(f"System: {system_info['name']}")
        report_lines.append(f"Time Elapsed: {elapsed_time:.2f}s")
        report_lines.append("")
        
        # 评估结果
        if "eval_result" in results:
            eval_result = results["eval_result"]
            report_lines.append(f"Total Questions: {eval_result.total_questions}")
            report_lines.append(f"Correct: {eval_result.correct}")
            report_lines.append(f"Accuracy: {eval_result.accuracy:.2%}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        report_path = self.output_dir / "report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        # 打印到控制台
        self.console.print("\n" + report_text, style="bold green")
        self.logger.info(f"Report saved to: {report_path}")
    
    # 序列化辅助方法
    def _search_result_to_dict(self, sr: SearchResult) -> dict:
        return {
            "query": sr.query,
            "conversation_id": sr.conversation_id,
            "results": sr.results,
            "retrieval_metadata": sr.retrieval_metadata,
        }
    
    def _dict_to_search_result(self, d: dict) -> SearchResult:
        return SearchResult(**d)
    
    def _answer_result_to_dict(self, ar: AnswerResult) -> dict:
        return {
            "question_id": ar.question_id,
            "question": ar.question,
            "answer": ar.answer,
            "golden_answer": ar.golden_answer,
            "category": ar.category,
            "conversation_id": ar.conversation_id,
            "search_results": ar.search_results,
            "metadata": ar.metadata,
        }
    
    def _dict_to_answer_result(self, d: dict) -> AnswerResult:
        return AnswerResult(**d)
    
    def _eval_result_to_dict(self, er: EvaluationResult) -> dict:
        return {
            "total_questions": er.total_questions,
            "correct": er.correct,
            "accuracy": er.accuracy,
            "detailed_results": er.detailed_results,
            "metadata": er.metadata,
        }

