import chromadb
import time
import os
from typing import List
from omegaconf import DictConfig
from openai import OpenAI

# 引入 VLAgent
from agents.vl_agent import VLAgent
from schema import TextChunk, FigureData, EvidenceItem, ContentType

def generate_embeddings(texts: List[str], config: DictConfig) -> List[List[float]]:
    """
    通用 Embedding 生成函数 (保持不变)
    """
    base_url = config.api.base_url
    api_key = config.api.api_key
    model_id = config.models.embedding.model_id
    
    client = OpenAI(base_url=base_url, api_key=api_key)
    all_embeddings = []
    
    is_modelscope = "modelscope.cn" in base_url
    
    if is_modelscope:
        print(f"[*] 检测到 ModelScope 接口，启用单条限速模式 (共 {len(texts)} 条)...")
        for i, text in enumerate(texts):
            clean_text = text.strip() or "empty_node"
            success = False
            retry_count = 0
            max_retries = 5
            
            while not success and retry_count < max_retries:
                try:
                    response = client.embeddings.create(
                        model=model_id, input=clean_text, encoding_format="float"
                    )
                    if response.data:
                        all_embeddings.append(response.data[0].embedding)
                        success = True
                        time.sleep(0.6) 
                    else:
                        raise ValueError("API响应为空")
                except Exception as e:
                    retry_count += 1
                    wait_time = 2 * retry_count
                    if "429" in str(e):
                        print(f"[!] 触发限流 (429), 冷却 {wait_time} 秒...")
                        time.sleep(wait_time)
                    else:
                        print(f"[!] 第 {i+1} 条失败: {e}, 重试 {retry_count}/{max_retries}")
                        time.sleep(1)
            
            if not success:
                raise Exception(f"生成向量失败，文本: {clean_text[:20]}...")
            if (i + 1) % 10 == 0:
                print(f"    -> 进度: {i + 1}/{len(texts)}")
    else:
        batch_size = config.models.embedding.batch_size
        print(f"[*] 检测到标准接口 (SiliconFlow等)，启用 Batch 高速模式 (Batch Size: {batch_size})...")
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            cleaned_batch = [t if t.strip() else "empty_node" for t in batch_texts]
            try:
                response = client.embeddings.create(
                    model=model_id, input=cleaned_batch, encoding_format="float"
                )
                batch_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [item.embedding for item in batch_data]
                all_embeddings.extend(batch_embeddings)
                print(f"    -> 已处理批次 {i // batch_size + 1} (累计 {len(all_embeddings)}/{len(texts)})")
            except Exception as e:
                print(f"[!] Batch处理失败: {e}")
                raise e

    return all_embeddings

class VectorStoreManager:
    def __init__(self, config: DictConfig):
        self.config = config
        
        # 1. 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=config.vector_db.path)
        
        self.collection = self.client.get_or_create_collection(
            name=config.vector_db.collection_name,
            metadata={"hnsw:space": config.vector_db.similarity_metric}
        )
        
        # 2. 初始化 VL Agent (用于生成图片描述)
        # 注意：这里直接实例化 VLAgent，它会自动读取 config.yaml 和环境变量
        print("[*] 初始化 VL Agent 用于图片理解...")
        self.vl_agent = VLAgent()

    def add_documents(self, text_chunks: List[TextChunk], figure_data: List[FigureData]):
        """
        核心逻辑：文本直接入库，图片经过 VL 理解后入库
        """
        evidence_items: List[EvidenceItem] = []

        # ==========================================
        # 1. 处理纯文本 (Text Chunks)
        # ==========================================
        print(f"[*] 正在处理 {len(text_chunks)} 个文本块...")
        for chunk in text_chunks:
            if len(chunk.content) < 5: continue 
            
            evidence_items.append(EvidenceItem(
                id=chunk.chunk_id,
                content=chunk.content,
                metadata={
                    "page_number": chunk.page_number,
                    "type": ContentType.TEXT.value,  # 关键 Metadata: 标记为 text
                    "source": "pdf_text",
                    "chunk_id": chunk.chunk_id
                }
            ))

        # ==========================================
        # 2. 处理图片 (Figures) -> Mixed Indexing
        # ==========================================
        print(f"[*] 正在处理 {len(figure_data)} 张图片 (调用 VL 模型)...")
        
        for i, fig in enumerate(figure_data):
            # 2.1 检查图片是否存在
            if not os.path.exists(fig.image_path):
                print(f"[!] 图片丢失跳过: {fig.image_path}")
                continue
            
            # 2.2 调用 VL Agent 生成描述
            # 提示词设计：要求模型描述内容并提取关键数据，方便后续文本检索匹配
            print(f"    -> 分析图片 {i+1}/{len(figure_data)}: {os.path.basename(fig.image_path)}")
            
            vl_output = self.vl_agent.analyze_image(
                image_path=fig.image_path,
                query="Describe this scientific image in detail. Include chart type, axis labels, data trends, and any text visible in the image."
            )
            
            # 2.3 构建丰富的内容字符串 (Rich Content)
            # 格式：[Caption] + [VL Description] + [VL Insights]
            # 这样检索 "2023 收入趋势" 既能匹配 Caption 也能匹配视觉描述
            rich_content = (
                f"Image Caption: {fig.caption}\n"
                f"Visual Description: {vl_output.description}\n"
                f"Key Insights: {vl_output.insights}"
            )
            
            # 2.4 构建 EvidenceItem
            evidence_items.append(EvidenceItem(
                id=fig.figure_id,
                content=rich_content, # 这里存的是“描述性文本”
                metadata={
                    "page_number": fig.page_number,
                    "type": ContentType.IMAGE.value, # 关键 Metadata: 标记为 image
                    "image_path": fig.image_path,    # 检索后用于回显图片
                    "source": "pdf_figure",
                    "caption": fig.caption or ""     # 保留原始 caption 在 metadata 备用
                }
            ))

        if not evidence_items:
            print("[*] 无有效数据需入库")
            return

        # ==========================================
        # 3. 统一 Embedding 并入库
        # ==========================================
        try:
            print(f"[*] 开始生成向量 (共 {len(evidence_items)} 条 Item)...")
            texts_to_embed = [item.content for item in evidence_items]
            
            # 批量生成向量
            embeddings = generate_embeddings(texts_to_embed, self.config)
            
            if len(embeddings) != len(evidence_items):
                raise ValueError(f"向量数量不匹配: 需 {len(evidence_items)}, 得 {len(embeddings)}")

            print(f"[*] 正在写入数据库...")
            self.collection.upsert(
                ids=[item.id for item in evidence_items],
                embeddings=embeddings,
                documents=[item.content for item in evidence_items],
                metadatas=[item.metadata for item in evidence_items]
            )
            print(f"[+] 入库成功! 当前库中总数: {self.collection.count()}")
            
        except Exception as e:
            print(f"[!] 入库流程中断: {e}")
            raise e

    def search(self, query: str, top_k: int = 5) -> List[EvidenceItem]:
        """检索测试函数"""
        query_vecs = generate_embeddings([query], self.config)
        
        results = self.collection.query(
            query_embeddings=query_vecs,
            n_results=top_k
        )
        
        items = []
        if results['ids']:
            ids = results['ids'][0]
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            dists = results['distances'][0] if 'distances' in results else [0.0]*len(ids)
            
            for i in range(len(ids)):
                items.append(EvidenceItem(
                    id=ids[i],
                    content=docs[i],
                    score=dists[i],
                    metadata=metas[i]
                ))
        return items