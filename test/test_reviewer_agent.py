import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv

# 解决 Windows 控制台可能的中文乱码
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    
load_dotenv()
sys.path.append(os.getcwd())

from agents.reviewer_agent import ReviewerAgent
from schema import ReasonerOutput

def test_reviewer():
    print("=== 测试 Reviewer Agent ===")
    
    cfg = OmegaConf.load("config.yaml")
    reviewer = ReviewerAgent(cfg)
    
    query = "feedback coefficient lambda 对 MSE 有什么影响？"
    
    # ----------------------------------------
    # 案例 1: 模拟一个优秀的回答 (应该 ACCEPT)
    # ----------------------------------------
    good_reasoner_output = ReasonerOutput(
        draft_answer="根据图 3 显示，在高噪声环境下（+/-40），增大反馈系数 lambda 可以显著降低 MSE。然而，在无噪声环境下，增大 lambda 反而会导致 MSE 略微上升 [Page 4, Fig.3]。",
        citations=["Fig.3", "Page 4"],
        reasoning_trace="Combined text and visual insights..."
    )
    
    print("\n---------- 测试案例 1: 优秀回答 ----------")
    res1 = reviewer.review(query, good_reasoner_output)
    print(f"Decision: {res1.decision}")
    print(f"Score: {res1.confidence_score}")
    print(f"Critique: {res1.critique}")
    
    if res1.decision == "ACCEPT" and res1.confidence_score >= 0.8:
        print("✅ 通过：成功识别好答案")
    else:
        print("❌ 失败：未能识别好答案")

    # ----------------------------------------
    # 案例 2: 模拟一个糟糕的回答 (应该 REJECT)
    # ----------------------------------------
    bad_reasoner_output = ReasonerOutput(
        draft_answer="lambda 是一个参数，它对结果有影响。我们可以通过调整它来改变模型行为。",
        citations=[], # 缺失引用
        reasoning_trace="No info found"
    )

    print("\n---------- 测试案例 2: 糟糕回答 ----------")
    res2 = reviewer.review(query, bad_reasoner_output)
    print(f"Decision: {res2.decision}")
    print(f"Score: {res2.confidence_score}")
    print(f"Critique: {res2.critique}")
    print(f"Feedback: {res2.feedback_for_retriever}")

    if res2.decision == "REJECT":
        print("✅ 通过：成功拦截坏答案")
    else:
        print("❌ 失败：未能拦截坏答案")

if __name__ == "__main__":
    test_reviewer()