"""week03 chunking research entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from llama_index.llms.openai_like import OpenAILike

EVAL_TEMPLATE_STR = (
    "我们提供了一个标准答案和一个由模型生成的回答。请你判断模型生成的回答"
    "在语义上是否与标准答案一致、准确且完整。\\n"
    "请只回答 '是' 或 '否'。\\n\\n"
    "标准答案：\\n----------\\n{ground_truth}\\n----------\\n\\n"
    "模型生成的回答：\\n----------\\n{generated_answer}\\n----------\\n"
)

EVAL_TEMPLATE = PromptTemplate(template=EVAL_TEMPLATE_STR)


def configure_llama_index(api_key: str) -> None:
    """初始化 LlamaIndex 全局设置。"""
    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        is_chat_model=True,
    )

    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192,
    )


def auto_redundancy_score(text: str) -> int:
    """利用简单启发式评估上下文冗余度，返回 1-5 分。"""
    cleaned = [seg.strip() for seg in text.split("\n\n") if seg.strip()]
    if not cleaned:
        return 1

    unique_segments = set(cleaned)
    ratio = len(cleaned) / max(len(unique_segments), 1)

    if ratio <= 1.2:
        return 1
    if ratio <= 1.5:
        return 2
    if ratio <= 1.8:
        return 3
    if ratio <= 2.2:
        return 4
    return 5


def context_contains_keywords(context: str, cues: Sequence[str]) -> str:
    """判断检索到的上下文是否覆盖关键词。"""
    if not cues:
        return "否"

    return "是" if all(cue in context for cue in cues) else "否"


def evaluate_splitter(
    splitter,
    documents,
    question: str,
    ground_truth: str,
    splitter_name: str,
    answer_cues: Sequence[str],
    similarity_top_k: int = 5,
) -> Dict[str, object]:
    """执行单个分割器的检索 + 生成评估。"""
    print(f"--- 开始评估: {splitter_name} ---")
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    if isinstance(splitter, SentenceWindowNodeParser):
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
    else:
        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)

    retrieved_nodes = query_engine.retrieve(question)
    retrieved_context = "\n\n".join(node.get_content() for node in retrieved_nodes)

    context_contains_answer = context_contains_keywords(
        retrieved_context, answer_cues
    )

    response = query_engine.query(question)
    generated_answer = str(response)
    eval_response = Settings.llm.predict(
        EVAL_TEMPLATE,
        ground_truth=ground_truth,
        generated_answer=generated_answer,
    )
    answer_is_accurate = "是" if "是" in eval_response else "否"

    redundancy_score = auto_redundancy_score(retrieved_context)

    result = {
        "分割器": splitter_name,
        "上下文包含答案": context_contains_answer,
        "回答准确": answer_is_accurate,
        "上下文冗余度(1-5)": redundancy_score,
        "生成回答": generated_answer.strip().replace("\n", " ")[:100] + "...",
    }

    print(f"检索上下文示例（前400字）：{retrieved_context[:400]}...")
    print(f"--- 完成评估: {splitter_name} ---\n")
    return result


def print_results_table(results: List[Dict[str, object]]) -> None:
    """打印汇总表格并保存到 outputs。"""
    if not results:
        print("没有可供显示的评估结果。")
        return

    df = pd.DataFrame(results)
    print("\n--- 最终评估结果对比 ---")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        # pandas 的 markdown 导出依赖 tabulate；缺失时退回常规表格输出来避免中断执行。
        print("未安装 tabulate，自动降级为普通文本表格展示。")
        print(df.to_string(index=False))

    output_dir = Path(__file__).with_name("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "splitter_eval.csv"
    df.to_csv(output_path, index=False)
    print(f"评估结果已写入 {output_path}")


def main() -> None:
    """作业入口：完成分割器对比实验。"""
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DASHSCOPE_API_KEY 环境变量，请先配置 DashScope API Key。")

    configure_llama_index(api_key)

    data_dir = Path(__file__).with_name("data")
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在，请先准备文档。")

    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    if len(documents) < 3:
        raise ValueError("请至少准备三篇文档以进行对比实验。")

    question = "玄奘是如何在取经途中协调悟空、八戒与沙僧的分工与纪律的？"
    ground_truth = (
        "玄奘依靠慈悲与制度同时约束三名徒弟：他信任悟空担任前锋巡山、负责除妖；"
        "要求八戒背负行李并承担炊事等后勤；让沙僧守护后队与夜间值守，再由白龙马负责运输。"
        "他还制定轮值守夜、记录供奉的规矩，使取经团队始终保持协同。"
    )
    answer_cues = ["玄奘", "悟空", "八戒", "沙僧", "白龙马", "纪律"]

    # chunk_size 表示每个分块的最大长度；chunk_overlap 控制相邻分块之间的重叠字符数，
    # 用来避免上下文被截断后丢失关键信息。
    # SentenceSplitter：基于句号、换行等规则拼接完整句子，保证语义自然边界；
    # TokenTextSplitter：按 token 数量硬切分，适合控制每块精确长度；
    # SentenceWindowNodeParser：在目标句周围附加前后若干句，形成窗口式上下文；
    # MarkdownNodeParser：根据 Markdown 标题层级拆分，保留文档结构。
    sentence_splitter = SentenceSplitter(
        chunk_size=256,  # 缩短块长以提升答案聚焦度
        chunk_overlap=20,
    )
    token_splitter = TokenTextSplitter(
        chunk_size=128,  # 按 token 粒度截断，更精细但数量更多
        chunk_overlap=8,
        separator="\n",
    )
    sentence_window_splitter = SentenceWindowNodeParser.from_defaults(
        window_size=4,  # 更宽窗口以覆盖纪律要求与白龙马等细节
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    markdown_splitter = MarkdownNodeParser()  # 根据 Markdown 层级结构拆分保持语义块

    results = []
    for splitter, name, top_k in [
        (sentence_splitter, "Sentence", 8),
        (token_splitter, "Token", 5),
        (sentence_window_splitter, "Sentence Window", 8),
        (markdown_splitter, "Markdown", 5),
    ]:
        result = evaluate_splitter(
            splitter=splitter,
            documents=documents,
            question=question,
            ground_truth=ground_truth,
            splitter_name=name,
            answer_cues=answer_cues,
            similarity_top_k=top_k,
        )
        results.append(result)

    print_results_table(results)


if __name__ == "__main__":
    main()