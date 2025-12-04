import os
import sys
import shutil
from pathlib import Path
import time
import gradio as gr
from omegaconf import OmegaConf
from dotenv import load_dotenv

# --- 引入后端模块 ---
# 确保当前目录在 sys.path 中
sys.path.append(str(Path(__file__).parent))

from agents.planner_agent import PlannerAgent
from agents.retriever_agent import RetrieverAgent
from agents.reasoner_agent import ReasonerAgent
from agents.reviewer_agent import ReviewerAgent
from tools.pdf_parser import PDFParser
from tools.vector_db import VectorStoreManager
from schema import AgentDecision, ReasonerOutput

# 加载环境
load_dotenv()

# --- 全局配置初始化 ---
CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

cfg = OmegaConf.load(CONFIG_PATH)

# 初始化全局 Agent 实例 (避免每次请求都重新加载模型，节省开销)
print("[*] Initializing Agents...")
planner = PlannerAgent(cfg)
retriever = RetrieverAgent(cfg)
reasoner = ReasonerAgent(cfg)
reviewer = ReviewerAgent(cfg)
print("[*] Agents Ready.")

# ==========================================
# 核心逻辑封装 (Generator 模式)
# ==========================================

def ingest_pdf(file_obj):
    """
    处理 PDF 上传和入库
    """
    if file_obj is None:
        return "⚠️ 请先上传一个 PDF 文件。"
    
    pdf_path = file_obj.name # Gradio 临时路径
    filename = os.path.basename(pdf_path)
    
    yield f"🚀 开始处理: {filename} ...\n"
    
    try:
        # 1. 解析
        yield f"📄 [Parser] 正在解析 PDF 结构和提取图片 (调用 MinerU)...\n"
        parser = PDFParser(cfg)
        text_chunks, figure_data = parser.parse_pdf(pdf_path)
        yield f"✅ 解析完成: 提取文本 {len(text_chunks)} 段, 图片 {len(figure_data)} 张。\n"
        
        # 2. 入库
        yield f"💾 [VectorDB] 正在进行 VL 图片理解与向量化存储...\n"
        vector_db = VectorStoreManager(cfg)
        vector_db.add_documents(text_chunks, figure_data)
        
        yield f"🎉 **入库成功！**\n文档 `{filename}` 已准备好，请切换到 Chat 标签页进行提问。"
        
    except Exception as e:
        yield f"❌ **处理失败**: {str(e)}"


def chat_pipeline(user_message, history):
    """
    执行 RAG 流程，并流式输出中间步骤日志和最终回复
    """
    if not user_message:
        yield history, "请输入问题。"
        return

    # 初始化日志缓冲区
    logs = "### 🤖 Agent Workflow Logs\n"
    
    # 1. Planner 阶段
    logs += "\n#### 1️⃣ Planner Agent\n*正在分析用户意图...*\n"
    yield  "正在规划检索策略...", logs
    
    try:
        plan = planner.plan(user_message)
        logs += f"**Reasoning**: {plan.reasoning}\n"
        logs += f"**Search Queries**: `{plan.search_queries}`\n"
        logs += f"**Visual Check**: {'✅ Yes' if plan.need_visual_understanding else '❌ No'}\n"
        yield "检索计划已生成...", logs
    except Exception as e:
        logs += f"❌ Planner Error: {str(e)}\n"
        yield f"系统错误: {str(e)}", logs
        return

    # 2. Retriever 阶段
    logs += "\n#### 2️⃣ Retriever Agent\n*正在执行向量检索...*\n"
    aggregated_context = ""
    
    for i, query in enumerate(plan.search_queries):
        logs += f"- 🔍 Searching: *{query}* ...\n"
        yield f"正在检索: {query}...", logs
        
        res = retriever.run(query)
        # 截取一部分结果显示在日志中，避免太长
        preview = res[:200].replace('\n', ' ') + "..."
        logs += f"  - Result: {preview}\n"
        aggregated_context += f"\n--- Search Result for '{query}' ---\n{res}\n"

    # 3. Reasoner & Reviewer Loop
    logs += "\n#### 3️⃣ Reasoner & Reviewer Loop\n*生成答案与自我审查...*\n"
    
    max_retries = 2
    current_attempt = 0
    feedback = ""
    final_answer_obj = None
    
    while current_attempt <= max_retries:
        logs += f"\n**Attempt {current_attempt + 1}**\n"
        yield f"正在生成回答 (第 {current_attempt+1} 次尝试)...", logs
        
        # Reasoner
        effective_query = user_message
        if feedback:
            effective_query += f"\n(Critique from previous turn: {feedback})"
            
        draft_output = reasoner.run(
            query=effective_query,
            retriever_result=aggregated_context,
            vl_results=[] # 图片信息已在 retrieved context 中
        )
        logs += "✍️ **Draft Generated**.\n"
        
        # Reviewer
        review = reviewer.review(user_message, draft_output)
        logs += f"🧐 **Review Decision**: `{review.decision.value}` (Score: {review.confidence_score})\n"
        
        if review.decision == AgentDecision.ACCEPT:
            final_answer_obj = draft_output
            logs += "✅ **Passed!**\n"
            yield final_answer_obj.draft_answer, logs # 最终输出
            break
        else:
            logs += f"⚠️ **Rejected**: {review.critique}\n"
            feedback = review.critique
            
            if review.feedback_for_retriever:
                logs += f"🔄 **Supplemental Search**: {review.feedback_for_retriever}\n"
                supp_evidence = retriever.run(review.feedback_for_retriever)
                aggregated_context += f"\n--- Supplemental ---\n{supp_evidence}\n"
            
            current_attempt += 1
            yield f"回答未通过审查，正在重试 ({current_attempt}/{max_retries})...", logs

    # Final Handling
    if final_answer_obj:
        # 格式化最终引用
        final_text = final_answer_obj.draft_answer
        if final_answer_obj.citations:
            final_text += "\n\n**📚 Citations:**\n" + "\n".join([f"- {c}" for c in final_answer_obj.citations])
        yield final_text, logs
    else:
        logs += "\n❌ Failed to generate satisfactory answer.\n"
        yield f"抱歉，经过多次尝试，我无法生成满足质量要求的回答。\n最后一次草稿：\n{draft_output.draft_answer}", logs


# ==========================================
# Gradio UI 构建
# ==========================================

# 自定义 CSS 优化样式
custom_css = """
#log_panel {
    background-color: #f9f9f9; 
    border: 1px solid #e0e0e0; 
    border-radius: 8px; 
    padding: 10px; 
    font-family: monospace; 
    font-size: 0.9em;
    height: 600px; 
    overflow-y: scroll;
}
"""

with gr.Blocks(title="SciMuse Agentic RAG", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🧪 SciMuse: 多智能体科研文献分析系统")
    gr.Markdown("基于 Planner-Retriever-Reasoner-Reviewer 架构的深度文档理解助手。")

    with gr.Tabs():
        # --- Tab 1: 知识库入库 ---
        with gr.Tab("📂 知识库 (Ingest)"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="上传 PDF 论文", 
                        file_types=[".pdf"],
                        file_count="single"
                    )
                    ingest_btn = gr.Button("🚀 开始解析入库", variant="primary")
                
                with gr.Column(scale=2):
                    ingest_log = gr.Textbox(
                        label="处理日志", 
                        placeholder="等待上传...", 
                        lines=15,
                        interactive=False
                    )
            
            # 绑定事件
            ingest_btn.click(
                fn=ingest_pdf,
                inputs=file_input,
                outputs=ingest_log
            )

        # --- Tab 2: 智能问答 ---
        with gr.Tab("💬 深度研读 (Chat)"):
            with gr.Row():
                # 左侧：聊天窗口
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Research Assistant",
                        type="messages", # Gradio 5.x 推荐的新格式
                        height=600,
                        avatar_images=(None, "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=SciMuse") # 可选头像
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            show_label=False, 
                            placeholder="请输入关于论文的问题 (例如: Figure 3 展示了什么趋势?)...",
                            container=False,
                            scale=4
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)
                
                # 右侧：思维链日志
                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 Agent Thoughts (思维链)")
                    log_output = gr.Markdown(
                        value="Waiting for query...", 
                        elem_id="log_panel"
                    )

            # --- 事件处理函数 ---
            def user_msg(user_message, history):
                # 1. 把用户消息加入历史并清空输入框
                if not user_message: return history, ""
                return history + [{"role": "user", "content": user_message}], ""

            def bot_response(history):
                # 获取最后一条用户消息
                user_message = history[-1]["content"]
                
                # 调用生成器
                pipeline_generator = chat_pipeline(user_message, history)
                
                # 初始响应占位
                history.append({"role": "assistant", "content": "..."})
                
                for response_text, log_text in pipeline_generator:
                    # 更新最后一条 Assistant 的消息
                    history[-1]["content"] = response_text
                    # 同时更新历史和侧边栏日志
                    yield history, log_text

            # 绑定回车和点击事件
            msg_input.submit(
                user_msg, [msg_input, chatbot], [chatbot, msg_input], queue=False
            ).then(
                bot_response, [chatbot], [chatbot, log_output]
            )
            
            submit_btn.click(
                user_msg, [msg_input, chatbot], [chatbot, msg_input], queue=False
            ).then(
                bot_response, [chatbot], [chatbot, log_output]
            )

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,    # 如果需要公网链接，设为 True
        show_error=True
    )