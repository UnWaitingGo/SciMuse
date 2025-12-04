import os
import sys
import time
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv, find_dotenv

# ================= 路径配置 =================
# 1. 获取项目根目录
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name in ['test', 'tests'] else CURRENT_DIR

# 2. 将项目根目录、tools、agents 加入 sys.path，确保模块能相互找到
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "tools"))
sys.path.append(str(ROOT_DIR / "agents"))

# 解决 Windows 控制台乱码
sys.stdout.reconfigure(encoding='utf-8')

from tools.pdf_parser import PDFParser
from tools.vector_db import VectorStoreManager
from schema import ContentType

# ================= 配置开关 =================
MAX_FIGURES_TO_PROCESS = 3  # 设置为 None 则处理全部图片，设置数字(如 3)则只处理前3张(省 Token)
CLEAN_DB_BEFORE_RUN = True  # 是否在运行前清空旧的 Collection

def main():
    # 1. 加载环境变量
    load_dotenv(find_dotenv(), override=True)
    print(f"[*] 项目根目录: {ROOT_DIR}")

    # 2. 加载 Config
    config_path = ROOT_DIR / "config.yaml"
    if not config_path.exists():
        print(f"[!] 找不到配置文件: {config_path}")
        return
    config = OmegaConf.load(config_path)

    # 3. 初始化模块
    print("-" * 50)
    try:
        print("[*] 初始化 PDF Parser (MinerU)...")
        parser = PDFParser(config)
        
        print("[*] 初始化 Vector DB Manager (Chroma + VLAgent)...")
        vector_manager = VectorStoreManager(config)
        
        # 可选：清空旧数据方便测试
        if CLEAN_DB_BEFORE_RUN:
            try:
                # 注意：Chroma 的 delete collection 比较暴力，生产环境慎用
                vector_manager.client.delete_collection(config.vector_db.collection_name)
                print("[*] 旧 Collection 已删除，正在重新创建空库...")
                # 重新初始化以创建新库
                vector_manager = VectorStoreManager(config)
            except Exception:
                pass # 如果库不存在会报错，忽略即可

    except Exception as e:
        print(f"[!] 模块初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 获取 PDF 文件
    pdf_dir = Path(config.paths.data.pdfs)
    if not pdf_dir.is_absolute():
        pdf_dir = ROOT_DIR / config.paths.data.pdfs
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] {pdf_dir} 下无 PDF 文件")
        return

    # 5. 处理流程
    for pdf_path in pdf_files:
        print(f"\n{'='*20} 处理文件: {pdf_path.name} {'='*20}")
        
        # A. 解析 PDF
        try:
            print(f"[*] [解析] 正在提取文本和图片信息...")
            text_chunks, figures = parser.parse_pdf(str(pdf_path))
            print(f"    -> 提取到 {len(text_chunks)} 段文本")
            print(f"    -> 提取到 {len(figures)} 张图片")
            
            # B. 过滤图片数量 (测试用)
            if MAX_FIGURES_TO_PROCESS is not None and len(figures) > MAX_FIGURES_TO_PROCESS:
                print(f"[*] [测试模式] 仅处理前 {MAX_FIGURES_TO_PROCESS} 张图片 (共 {len(figures)} 张)...")
                figures = figures[:MAX_FIGURES_TO_PROCESS]
            
            # C. 入库 (Text + Image w/ VLAgent)
            print(f"[*] [入库] 开始 Mixed Indexing (文本 + 图片视觉理解)...")
            start_time = time.time()
            vector_manager.add_documents(text_chunks, figures)
            print(f"    -> 入库耗时: {time.time() - start_time:.2f} 秒")

        except Exception as e:
            print(f"[!] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 6. 验证检索 (Test Search)
    print(f"\n{'='*20} 检索验证 (Mixed Retrieval) {'='*20}")
    
    # 这里定义针对你 PDF 的测试问题
    test_queries = [
        "What is the system architecture diagram?",  # 泛文本搜索
        "Describe the camera geometry shown in the figure.", # 针对 Figure 1
        "How does the feedback coefficient lambda affect the Mean Squared Error (MSE)?", # 针对 Figure 3/4
    ]

    for q in test_queries:
        print(f"\n[Q] 提问: {q}")
        results = vector_manager.search(q, top_k=3)
        
        for i, res in enumerate(results):
            type_icon = "🖼️ [IMG]" if res.metadata.get("type") == ContentType.IMAGE.value else "📄 [TXT]"
            score = f"{res.score:.4f}"
            
            # 打印摘要
            content_preview = res.content[:100].replace('\n', ' ') + "..."
            print(f"   {i+1}. {type_icon} (Dist: {score}) {content_preview}")
            
            # 如果是图片，打印出它关联的文件路径，证明找对了
            if res.metadata.get("type") == ContentType.IMAGE.value:
                print(f"      -> 文件: {res.metadata.get('image_path')}")
                # 打印一部分 Insight 看看效果
                if "Key Insights:" in res.content:
                    insight_part = res.content.split("Key Insights:")[1][:100]
                    print(f"      -> 洞察: {insight_part}...")

if __name__ == "__main__":
    main()