from agents.research_agents import (
    PlannerAgent,
    RetrieverAgent,
    ReasonerAgent,
    ReviewerAgent
)

# 研究问题
question = "Transformer模型的核心创新是什么?"

print("🤔 研究问题:", question)
print("\n" + "="*50)

# 1️⃣ 任务分解
print("\n1️⃣ 任务分解中...")
planner = PlannerAgent()
plan = planner.plan(question)
print(f"📋 分解为 {len(plan['sub_tasks'])} 个子任务:")
for i, task in enumerate(plan['sub_tasks'], 1):
    print(f"   {i}. {task}")

# 2️⃣ 证据检索
print("\n2️⃣ 检索证据中...")
retriever = RetrieverAgent()
evidence = retriever.retrieve(plan['sub_tasks'], top_k=5)
total_evidence = sum(e['evidence_count'] for e in evidence)
print(f"🔍 找到 {total_evidence} 条证据")

# 3️⃣ 推理生成
print("\n3️⃣ 生成答案中...")
reasoner = ReasonerAgent()
answer = reasoner.reason(
    question=question,
    evidence=evidence,
    require_citations=True
)
print(f"💡 答案: {answer['answer']}")
print(f"📊 置信度: {answer['confidence']:.2f}")

# 4️⃣ 质量检查
print("\n4️⃣ 质量检查中...")
reviewer = ReviewerAgent()
review = reviewer.review(
    question=question,
    answer=answer['answer'],
    evidence=evidence,
    confidence=answer['confidence']
)
print(f"✅ 最终置信度: {review['final_confidence']:.2f}")
print(f"🔄 需要迭代: {'是' if review['need_iterate'] else '否'}")
if review['issues']:
    print(f"⚠️ 发现问题: {', '.join(review['issues'])}")

print("\n" + "="*50)
print("🎉 分析完成!")