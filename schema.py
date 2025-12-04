from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import uuid

# ==========================================
# 1. 基础枚举与类型定义 (Enums & Basic Types)
# ==========================================

class AgentDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"

class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"

# ==========================================
# 2. 数据层模型 (Data Layer Models)
# 对应图中的: PDF Parser -> Structured Data -> Vector DB
# ==========================================

class TextChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_number: int
    content: str

class FigureData(BaseModel):
    figure_id: str
    page_number: int
    image_path: str  # 或 bytes
    caption: Optional[str] = None

class EvidenceItem(BaseModel):
    """存入 Vector DB 或从 Retriever 检索出的证据单元"""
    id: str
    content: str
    score: Optional[float] = None  # 向量相似度分数
    metadata: Dict[str, Any] = Field(
        description="包含页码、来源(text/figure)等信息"
    )

# ==========================================
# 3. 智能体交互模型 (Agent Interface Schemas)
# ==========================================

# --- Planner Agent ---
class PlannerOutput(BaseModel):
    reasoning: str = Field(..., description="规划任务的思考过程")
    search_queries: List[str] = Field(..., description="生成的检索关键词列表")
    need_visual_understanding: bool = Field(False, description="是否需要分析图片")

# --- VL Agent (Visual Language) ---
class VLInput(BaseModel):
    image_path: str
    query: str = Field(..., description="针对这张图需要回答的问题")

class VLOutput(BaseModel):
    description: str = Field(..., description="图片的详细文字描述")
    insights: str = Field(..., description="从图中提取的关键洞察")

# --- Retriever Agent ---
class RetrieverOutput(BaseModel):
    text_evidence: List[EvidenceItem]
    image_evidence: List[FigureData]

# --- Reasoner Agent ---
class ReasonerInput(BaseModel):
    original_question: str
    retrieved_context: List[EvidenceItem]
    image_descriptions: List[str]

class ReasonerOutput(BaseModel):
    draft_answer: str
    citations: List[str] = Field(..., description="例如 ['Fig.3', 'p.7']")
    reasoning_trace: str

# --- Reviewer Agent (核心控制流) ---
class ReviewerOutput(BaseModel):
    """
    对应图中 Reviewer Agent 的输出
    用于判断 confidence >= 0.8
    """
    confidence_score: float = Field(..., description="0.0 到 1.0 之间的置信度")
    decision: AgentDecision
    critique: Optional[str] = Field(None, description="如果被拒绝，给出具体的修改建议")
    feedback_for_retriever: Optional[str] = Field(None, description="指导 Retriever 如何重新检索")

# ==========================================
# 4. 最终响应 (Final Output)
# ==========================================

class FinalResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    trace_id: str