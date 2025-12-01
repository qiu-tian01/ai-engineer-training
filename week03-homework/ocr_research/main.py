"""week03 OCR research entrypoint."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Union

import numpy as np
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from llama_index.llms.openai_like import OpenAILike

try:
    from paddleocr import PaddleOCR
except ImportError:
    raise ImportError(
        "PaddleOCR is not installed. Please run 'pip install \"paddlepaddle<=2.6\" \"paddleocr<3.0\"'"
    )


class ImageOCRReader(BaseReader):
    """使用 PP-OCR 从图像中提取文本并返回 Document"""

    def __init__(self, lang="ch", use_gpu=False, **kwargs):
        """
        Args:
            lang (str): OCR 语言 ('ch', 'en', 'fr', etc.)
            use_gpu (bool): 是否使用 GPU 加速
            **kwargs: 其他传递给 PaddleOCR 的参数
        """
        super().__init__()
        self.lang = lang
        # 为了性能，在初始化时加载模型
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **kwargs,
        )

    def load_data(
        self, file: Union[str, Path, List[Union[str, Path]]]
    ) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表。

        Args:
            file: 图像路径字符串或路径列表

        Returns:
            List[Document]: 每个图像对应一个 Document 对象
        """
        if isinstance(file, (str, Path)):
            files = [file]
        else:
            files = file

        documents = []
        for image_path in files:
            image_path_str = str(image_path)
            if not os.path.exists(image_path_str):
                print(f"Warning: Image file not found: {image_path_str}")
                continue

            # 使用 PaddleOCR 提取文本
            result = self._ocr.ocr(image_path_str, cls=True)

            if not result or not result[0]:
                # 如果 OCR 未返回任何结果，则跳过此图像
                print(f"Warning: No text detected in {image_path_str}")
                continue

            text_blocks = []
            confidences = []

            # 遍历所有检测到的文本块
            for i, line in enumerate(result[0]):
                text = line[1][0]
                confidence = line[1][1]

                text_blocks.append(
                    f"[Text Block {i+1}] (conf: {confidence:.2f}): {text}"
                )
                confidences.append(confidence)

            # 拼接所有文本块
            full_text = "\n".join(text_blocks)

            # 计算平均置信度
            avg_confidence = np.mean(confidences) if confidences else 0.0

            # 构造 Document 对象
            doc = Document(
                text=full_text,
                metadata={
                    "image_path": image_path_str,
                    "ocr_model": "PP-OCRv5",
                    "language": self.lang,
                    "num_text_blocks": len(text_blocks),
                    "avg_confidence": float(avg_confidence),
                },
            )
            documents.append(doc)

        return documents


def setup_environment() -> None:
    """配置 LlamaIndex 所需的环境和模型"""
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未检测到 DASHSCOPE_API_KEY 环境变量，请先配置 DashScope API Key。"
        )

    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        is_chat_model=True,
    )
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=3,  # 减小批次大小以避免连接问题
        embed_input_length=8192,
        api_key=api_key,  # 显式传递 API key
    )
    print("LlamaIndex 环境配置完成。")


def main() -> None:
    """作业入口：完成 OCR 图像文本加载器实验。"""
    # --- 1. 准备工作：准备示例图片 ---
    print("--- 步骤 1: 准备示例图片 ---")
    data_dir = Path(__file__).with_name("ocr_images")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 尝试创建示例图片
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = (
            ImageFont.truetype("msyh.ttc", 18)
            if os.path.exists("msyh.ttc")
            else ImageFont.load_default()
        )

        # 扫描文档
        img_doc = Image.new("RGB", (600, 200), color="white")
        d = ImageDraw.Draw(img_doc)
        doc_text = (
            "LlamaIndex is a powerful tool for building and querying knowledge bases. "
            "It supports multiple data sources, including text, PDF, and now even images!"
        )
        d.text((10, 10), doc_text, fill=(0, 0, 0), font=font)
        doc_path = data_dir / "document.png"
        img_doc.save(doc_path)
        print(f"创建扫描文档图片: {doc_path}")

        # 屏幕截图
        img_ui = Image.new("RGB", (400, 150), color=(230, 230, 230))
        d = ImageDraw.Draw(img_ui)
        d.rectangle([10, 10, 100, 40], fill=(0, 120, 215))
        d.text((30, 15), "confirm", fill="white", font=font)
        d.text((120, 15), "username: user_test", fill="black", font=font)
        ui_path = data_dir / "screenshot.png"
        img_ui.save(ui_path)
        print(f"创建屏幕截图: {ui_path}")

        # 自然场景（路牌）
        img_scene = Image.new("RGB", (500, 300), color=(100, 150, 200))
        d = ImageDraw.Draw(img_scene)
        d.rectangle([50, 100, 450, 200], fill="red")
        scene_font = (
            ImageFont.truetype("msyh.ttc", 40)
            if os.path.exists("msyh.ttc")
            else ImageFont.load_default()
        )
        d.text((150, 130), "No Stopping", fill="white", font=scene_font)
        scene_path = data_dir / "sign.png"
        img_scene.save(scene_path)
        print(f"创建自然场景图片: {scene_path}")

        image_files = [doc_path, ui_path, scene_path]

    except (ImportError, OSError) as e:
        print(f"Pillow 或字体文件未找到，跳过图片创建: {e}")
        print("请手动将图片放入 'ocr_images' 目录。")
        # 如果无法创建图片，尝试从目录中查找现有图片
        image_files = [
            p
            for p in data_dir.glob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]
        if not image_files:
            print("未找到图片文件，退出。")
            return

    print(f"共找到 {len(image_files)} 张图片\n")

    # --- 2. 使用 ImageOCRReader 加载图像并生成 Document ---
    print("--- 步骤 2: 使用 ImageOCRReader 加载图像 ---")
    reader = ImageOCRReader(lang="en", use_gpu=False)
    documents = reader.load_data(image_files)

    print(f"成功加载 {len(documents)} 个文档。\n")
    for i, doc in enumerate(documents):
        print(f"--- 文档 {i+1} ---")
        print(f"文本预览: {doc.text[:150]}...")
        print(f"元数据: {doc.metadata}\n")

    # --- 3. 配置 LlamaIndex 环境 ---
    print("--- 步骤 3: 配置 LlamaIndex 环境 ---")
    setup_environment()

    # --- 4. 构建索引并进行查询 ---
    print("\n--- 步骤 4: 构建索引并进行查询 ---")
    # 添加重试机制以处理网络连接问题
    max_retries = 3
    retry_delay = 2  # 秒
    
    for attempt in range(max_retries):
        try:
            print(f"正在构建向量索引（尝试 {attempt + 1}/{max_retries}）...")
            index = VectorStoreIndex.from_documents(documents)
            print("索引构建成功！")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                error_msg = str(e)
                if "Connection" in error_msg or "ConnectionResetError" in error_msg:
                    print(f"连接错误，{retry_delay} 秒后重试... (错误: {error_msg[:100]})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    print(f"构建索引时出错: {error_msg}")
                    raise
            else:
                print(f"构建索引失败，已重试 {max_retries} 次。")
                print("提示：这可能是网络连接问题，请检查：")
                print("1. 网络连接是否稳定")
                print("2. DASHSCOPE_API_KEY 是否有效")
                print("3. 是否触发了 API 速率限制")
                raise
    
    query_engine = index.as_query_engine()

    def query_with_retry(query_engine, question: str, max_retries: int = 3) -> str:
        """带重试机制的查询函数"""
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                response = query_engine.query(question)
                return str(response)
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    if "Connection" in error_msg or "ConnectionResetError" in error_msg:
                        print(f"  连接错误，{retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise
                else:
                    print(f"  查询失败: {error_msg[:100]}")
                    raise

    # 查询1: 关于 LlamaIndex 的问题
    question1 = "What is LlamaIndex?"
    print(f"\n查询 1: {question1}")
    try:
        response1 = query_with_retry(query_engine, question1)
        print(f"回答: {response1}\n")
    except Exception as e:
        print(f"查询失败: {e}\n")

    # 查询2: 关于截图内容的问题
    question2 = "截图里的用户名是什么？"
    print(f"查询 2: {question2}")
    try:
        response2 = query_with_retry(query_engine, question2)
        print(f"回答: {response2}\n")
    except Exception as e:
        print(f"查询失败: {e}\n")

    # 查询3: 关于路牌的问题
    question3 = "红色的牌子上写了什么？"
    print(f"查询 3: {question3}")
    try:
        response3 = query_with_retry(query_engine, question3)
        print(f"回答: {response3}\n")
    except Exception as e:
        print(f"查询失败: {e}\n")

    print("--- OCR 实验完成 ---")


if __name__ == "__main__":
    main()
