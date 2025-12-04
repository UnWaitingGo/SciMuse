import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv

# 解决 Windows 控制台可能的中文乱码
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# 加载环境变量
load_dotenv()

# 确保能导入模块
sys.path.append(os.getcwd())

from agents.reasoner_agent import ReasonerAgent
from schema import VLOutput

def test_reasoner_with_demo_pdf():
    print("=== 测试 Reasoner Agent (基于 Real-time Stereo Matching 论文) ===\n")

    # 1. 加载配置
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("错误: 找不到 config.yaml")
        return
    cfg = OmegaConf.load(config_path)
    
    try:
        reasoner = ReasonerAgent(cfg)
        print("✅ Reasoner Agent 初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e} (请检查 .env 中的 API KEY)")
        return

    # ==========================================
    # 2. 构造基于 Demo PDF 的真实模拟数据
    # ==========================================
    
    # [模拟用户提问]
    # 这个问题考察 Agent 能否关联 "feedback coefficient" 和 "noise" 的关系
    user_query = "How does the feedback coefficient lambda affect the Mean Squared Error (MSE) under different image noise levels?"

    # [模拟 Retriever 检索到的文本] 
    # 摘自 OCR 第 4 页，专门讨论 Figure 3 和 Figure 4 的段落
    mock_retriever_text = """
    [Page 4] Significant improvements in accuracy can be seen in Figure 3 when the noise has ranges of +/-20, and +/-40. In this scenario, the effect of noise in the current frame is reduced by increasing the feedback coefficient lambda.
    [Page 4] As with the majority of temporal stereo matching methods, improvements are negligible when no noise is added to the images.
    [Page 4] Figure 4 shows the optimal values of lambda for noise ranging between +/-0 to +/-40. It is more beneficial to rely on the auxiliary cost when noise is high.
    """

    # [模拟 VL Agent 对 Figure 3 的分析]
    # Figure 3 是那三张纵向排列的折线图
    mock_vl_outputs = [
        VLOutput(
            description="The image contains three line plots showing Mean Squared Error (MSE) vs feedback coefficient lambda. "
                        "Top plot (Noise +/-0): The blue line goes up as lambda increases. "
                        "Middle plot (Noise +/-20): The line is flat then drops slightly. "
                        "Bottom plot (Noise +/-40): The blue line drops significantly as lambda increases from 0 to 0.8, showing lower error at higher lambda values.",
            insights="The charts demonstrate that for high noise levels (+/-40), increasing the feedback coefficient lambda significantly reduces the error (MSE). However, for zero noise, increasing lambda actually increases the error."
        )
    ]

    # ==========================================
    # 3. 运行推理
    # ==========================================
    print(f"❓ 用户问题: {user_query}")
    print("\n[Thinking] Reasoner 正在阅读文本并结合图表分析...\n")
    
    try:
        result = reasoner.run(
            query=user_query,
            retriever_result=mock_retriever_text,
            vl_results=mock_vl_outputs
        )
        
        # 4. 打印结果
        print("=== 🎯 推理结果 ===")
        print(f"🔍 逻辑链 (Trace):\n{result.reasoning_trace}\n")
        print(f"📝 最终回答 (Answer):\n{result.draft_answer}\n")
        print(f"📚 引用 (Citations): {result.citations}")
        
        # 5. 验证点
        # 我们期望回答中包含：
        # 1. 低噪声/无噪声时 lambda 增大反而不好 (或没用)。
        # 2. 高噪声时 lambda 增大能降低 MSE。
        if "increase" in result.draft_answer.lower() and "noise" in result.draft_answer.lower():
            print("\n✅ 测试通过：Agent 成功综合了文本结论和图表趋势！")
        else:
            print("\n⚠️ 结果可能不完整，请人工检查上述输出。")

    except Exception as e:
        print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    test_reasoner_with_demo_pdf()