import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv

# 解决 Windows 控制台可能的中文乱码
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    
# 加载环境
load_dotenv()
sys.path.append(os.getcwd())

from agents.planner_agent import PlannerAgent
from schema import PlannerOutput

def test_planner():
    print("=== 测试 Planner Agent ===\n")

    # 1. 初始化
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("❌ 找不到 config.yaml")
        return
    
    cfg = OmegaConf.load(config_path)
    planner = PlannerAgent(cfg)
    print("✅ Planner Agent 初始化成功")

    # ---------------------------------------------------------
    # 测试案例 1: 涉及图表分析的问题 (Visual=True)
    # ---------------------------------------------------------
    query_visual = "Figure 3 里展示的 MSE 误差随着 lambda 变化有什么趋势？"
    print(f"\n❓ 测试案例 1: {query_visual}")
    
    res_visual = planner.plan(query_visual)
    
    print(f"   -> 思考: {res_visual.reasoning}")
    print(f"   -> 搜索词: {res_visual.search_queries}")
    print(f"   -> 需要看图: {res_visual.need_visual_understanding}")

    if res_visual.need_visual_understanding is True:
        print("   ✅ 通过：正确识别视觉需求")
    else:
        print("   ❌ 失败：未能识别视觉需求")

    # ---------------------------------------------------------
    # 测试案例 2: 纯理论问题 (Visual=False)
    # ---------------------------------------------------------
    query_text = "这篇论文提出的 Temporal Cost Aggregation 算法公式是什么？"
    print(f"\n❓ 测试案例 2: {query_text}")
    
    res_text = planner.plan(query_text)
    
    print(f"   -> 搜索词: {res_text.search_queries}")
    print(f"   -> 需要看图: {res_text.need_visual_understanding}")
    
    # 检查是否生成了相关的特定关键词
    if any("formula" in q.lower() or "equation" in q.lower() or "aggregation" in q.lower() for q in res_text.search_queries):
        print("   ✅ 通过：生成了相关的技术搜索词")
    else:
        print("   ⚠️ 警告：搜索词可能不够精准")

    # ---------------------------------------------------------
    # 验证 Schema 输出类型
    # ---------------------------------------------------------
    if isinstance(res_text, PlannerOutput):
        print("\n✅ Schema 验证通过：输出类型严格符合 PlannerOutput 定义")
    else:
        print("\n❌ Schema 验证失败")

if __name__ == "__main__":
    test_planner()