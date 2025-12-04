import sys
import os
from pathlib import Path

# 将项目根目录添加到 python path 以便导入模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from tools.agent_tools import retriever_tool
from agents.vl_agent import VLAgent

from schema import RetrieverOutput, VLOutput

# 解决 Windows 控制台可能的中文乱码
sys.stdout.reconfigure(encoding='utf-8')

def test_retriever_tool():
    print("\n=== 测试 Retriever Tool ===")
    query = "transformer architecture details" # 可以根据你向量库里的实际内容修改
    
    try:
        # 直接调用工具函数 (Retriever 依然是函数式 Tool)
        result: RetrieverOutput = retriever_tool(query)
        
        print(f"输入查询: {query}")
        print(f"文本证据数量: {len(result.text_evidence)}")
        print(f"图片证据数量: {len(result.image_evidence)}")
        
        if result.text_evidence:
            print(f"Top Text Sample: {result.text_evidence[0].content[:100]}...")
            
        if result.image_evidence:
            print(f"Top Image Path: {result.image_evidence[0].image_path}")
            
        return result
    except Exception as e:
        print(f"Retriever Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_vl_agent(image_path: str):
    print("\n=== 测试 VL Agent (Image Understanding) ===")
    
    if not os.path.exists(image_path):
        print(f"[跳过] 测试图片未找到: {image_path}")
        print("请先运行 PDF 解析测试生成图片。")
        return

    query = "What is presented in this figure? Please explain detailedly."
    
    try:
        # 修改点 2: 实例化 Agent 类
        print("[*] 初始化 VL Agent...")
        vl_agent = VLAgent()
        
        # 修改点 3: 调用实例方法 analyze_image
        result: VLOutput = vl_agent.analyze_image(image_path, query)
        
        print(f"目标图片: {image_path}")
        print("-" * 30)
        print(f"[描述 (Description)]:\n{result.description}")
        print("-" * 30)
        print(f"[洞察 (Insights)]:\n{result.insights}")
        print("-" * 30)
        
    except Exception as e:
        print(f"VL Agent 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 1. 测试检索
    retriever_result = test_retriever_tool()
    
    # 2. 测试图片理解
    # 策略：如果检索结果里有图片，就用那张图片测试。
    # 如果没有，尝试找 data/processed/images 下的第一张图片
    target_image = None
    
    if retriever_result and retriever_result.image_evidence:
        target_image = retriever_result.image_evidence[0].image_path
        print(f"[*] 使用检索结果中的图片: {target_image}")
    else:
        # Fallback: 查找目录
        img_dir = PROJECT_ROOT / "data/processed/images"
        if img_dir.exists():
            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            if images:
                target_image = str(images[0])
                print(f"[*] 使用目录下第一张图片: {target_image}")
    
    if target_image:
        test_vl_agent(target_image)
    else:
        print("\n[!] 未找到用于测试 VL Agent 的图片。")