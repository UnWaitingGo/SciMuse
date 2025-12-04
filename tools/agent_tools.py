import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf
from smolagents import tool

# 加载 .env 文件
load_dotenv()

# 添加项目根目录到 sys.path，确保能导入 schema
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

# 引入项目内的模块
from schema import (
    RetrieverOutput, 
    EvidenceItem, 
    FigureData, 
    ContentType
)
from tools.vector_db import VectorStoreManager

# ================= Global Setup =================
CONFIG_PATH = BASE_DIR / "config.yaml"

if not CONFIG_PATH.exists():
    print(f"[Warning] Config file not found at {CONFIG_PATH}")
    cfg = None
else:
    cfg = OmegaConf.load(CONFIG_PATH)

# 初始化全局资源
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None and cfg:
        _vector_store = VectorStoreManager(cfg)
    return _vector_store

# ================= Tools Definition =================

@tool
def retriever_tool(query: str) -> str:
    """
    Performs semantic search in the vector database to find relevant text chunks and figures.

    Args:
        query: The search query string. This should be a specific question or topic description.

    Returns:
        RetrieverOutput: An object containing text_evidence and image_evidence lists.
    """
    # 确保资源已初始化
    store = get_vector_store()
    if not store:
        return "System Error: Vector store is not initialized. Please check configuration."

    print(f"[*] [Retriever Tool] Searching for: {query}")
    
    # 1. 执行检索
    # 假设 config 中有 agents.retriever.top_k 配置
    top_k = cfg.agents.retriever.top_k if cfg else 5
    try:
        raw_results = _vector_store.search(query, top_k=top_k)
    except Exception as e:
        # 【建议】捕获潜在的搜索错误
        return f"Search Error: An error occurred while searching: {str(e)}"
    
    text_evidence = []
    image_evidence = []
    
    # 2. 结果分类处理
    for item in raw_results:
        item_type = item.metadata.get("type", "text")
        
        if item_type == ContentType.TEXT.value:
            # === 为了防止 Token 爆炸，截断过长的文本 ===
            content_preview = item.content
            if len(content_preview) > 400:
                content_preview = content_preview[:400] + "...(truncated)"
            
            truncated_item = EvidenceItem(
                id=item.id,
                content=content_preview,
                score=item.score,
                metadata=item.metadata
            )
            text_evidence.append(truncated_item)
            
        elif item_type == ContentType.IMAGE.value:
            fig_data = FigureData(
                figure_id=item.id,
                page_number=item.metadata.get("page_number", 0),
                image_path=item.metadata.get("image_path", ""),
                caption=item.content 
            )
            image_evidence.append(fig_data)
            
    print(f"[*] [Retriever Tool] Found {len(text_evidence)} texts and {len(image_evidence)} images.")
    
    output_lines = []
    
    output_lines.append(f"### Search Results for '{query}'\n")
    
    if text_evidence:
        output_lines.append("#### Text Evidence:")
        for idx, item in enumerate(text_evidence):
            output_lines.append(f"[{idx+1}] (Score: {item.score:.2f}) {item.content}")
    
    if image_evidence:
        output_lines.append("\n#### Image Evidence:")
        for fig in image_evidence:
            output_lines.append(f"- Figure {fig.figure_id} (Page {fig.page_number}): {fig.caption}")

    if not text_evidence and not image_evidence:
        return "No relevant information found."

    # 返回给 Agent 的是一大段这类文本，Agent 可以直接阅读
    return "\n".join(output_lines)