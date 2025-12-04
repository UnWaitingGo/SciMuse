import os
import argparse
import sys
from pathlib import Path
from typing import List

from omegaconf import OmegaConf
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# 引入 Agents
from agents.planner_agent import PlannerAgent
from agents.retriever_agent import RetrieverAgent
from agents.reasoner_agent import ReasonerAgent
from agents.reviewer_agent import ReviewerAgent

# 引入 Tools 和 Schema
from tools.pdf_parser import PDFParser
from tools.vector_db import VectorStoreManager
from schema import AgentDecision, ReasonerOutput

# 加载环境变量
load_dotenv()

# 初始化 Rich 终端输出
console = Console()

def load_config(config_path: str = "config.yaml"):
    if not os.path.exists(config_path):
        console.print(f"[bold red]Config file not found at {config_path}[/bold red]")
        sys.exit(1)
    return OmegaConf.load(config_path)

def ingest_document(cfg, pdf_path: str):
    """
    文档摄入流程：解析 PDF -> 提取文本和图片 -> VL 理解图片 -> 存入向量库
    """
    console.print(Panel(f"Starting Ingestion for: {pdf_path}", title="SciMuse Ingestion", style="bold green"))

    # 1. 解析 PDF
    try:
        parser = PDFParser(cfg)
        text_chunks, figure_data = parser.parse_pdf(pdf_path)
        console.print(f"[green]Parsed PDF successfully:[/green] {len(text_chunks)} text chunks, {len(figure_data)} figures.")
    except Exception as e:
        console.print(f"[bold red]PDF Parsing Failed:[/bold red] {str(e)}")
        return

    # 2. 存入向量数据库 (包含 VL 图片分析过程)
    try:
        vector_db = VectorStoreManager(cfg)
        vector_db.add_documents(text_chunks, figure_data)
        console.print("[bold green]Ingestion Complete! Document is ready for search.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Vector DB Storage Failed:[/bold red] {str(e)}")

def run_chat_pipeline(cfg, user_query: str):
    """
    问答主流程：Planner -> Retriever -> Reasoner -> Reviewer -> (Retry loop)
    """
    # 初始化 Agents
    planner = PlannerAgent(cfg)
    retriever = RetrieverAgent(cfg)
    reasoner = ReasonerAgent(cfg)
    reviewer = ReviewerAgent(cfg)

    console.print(Panel(user_query, title="User Query", style="bold blue"))

    # ================= 1. Planner Agent =================
    with console.status("[bold cyan]Planner is analyzing the query...[/bold cyan]"):
        plan = planner.plan(user_query)
    
    console.print(f"[yellow]Plan Reasoning:[/yellow] {plan.reasoning}")
    console.print(f"[yellow]Search Queries:[/yellow] {plan.search_queries}")
    console.print(f"[yellow]Need Visuals:[/yellow] {plan.need_visual_understanding}")

    # ================= 2. Retriever Agent =================
    # 根据计划执行多次检索并聚合上下文
    aggregated_context = ""
    
    with console.status("[bold magenta]Retriever is searching evidence...[/bold magenta]"):
        # 简单策略：遍历所有生成的查询词
        # 实际生产中可以并行检索，或者让 Planner 只生成一个最复杂的查询
        for query in plan.search_queries:
            console.print(f"   -> Searching: '{query}'")
            result = retriever.run(query)
            aggregated_context += f"\n--- Search Result for '{query}' ---\n{result}\n"

    # ================= 3. Reasoning & Review Loop =================
    max_retries = 2
    current_attempt = 0
    feedback = ""
    
    final_answer: ReasonerOutput = None

    while current_attempt <= max_retries:
        console.rule(f"[bold]Reasoning Attempt {current_attempt + 1}[/bold]")
        
        # --- Reasoner ---
        # 如果有反馈，拼接到 prompt 中
        effective_query = user_query
        if feedback:
            effective_query += f"\n\n(PREVIOUS FEEDBACK: {feedback}. Please improve the answer.)"

        with console.status("[bold green]Reasoner is thinking...[/bold green]"):
            # 注意：vl_results 传空列表，因为 VectorDB 已经把图片理解成了文本放在 context 里了
            # 如果需要实时的图片重分析，可以在这里逻辑处理
            draft_output = reasoner.run(
                query=effective_query, 
                retriever_result=aggregated_context, 
                vl_results=[] 
            )

        console.print(Panel(Markdown(draft_output.draft_answer), title="Draft Answer"))

        # --- Reviewer ---
        with console.status("[bold red]Reviewer is evaluating...[/bold red]"):
            review = reviewer.review(user_query, draft_output)

        console.print(f"[bold]Decision:[/bold] {review.decision.value} (Score: {review.confidence_score})")
        
        if review.decision == AgentDecision.ACCEPT:
            final_answer = draft_output
            console.print("[bold green]Answer Accepted![/bold green]")
            break
        else:
            console.print(f"[bold orange3]Critique:[/bold orange3] {review.critique}")
            feedback = review.critique
            
            # 如果 Reviewer 建议重新检索 (feedback_for_retriever 不为空)
            if review.feedback_for_retriever:
                console.print(f"[magenta]Executing Supplemental Search:[/magenta] {review.feedback_for_retriever}")
                new_evidence = retriever.run(review.feedback_for_retriever)
                aggregated_context += f"\n--- Supplemental Evidence ---\n{new_evidence}\n"
            
            current_attempt += 1

    # ================= 4. Final Output =================
    console.rule("[bold green]FINAL RESPONSE[/bold green]")
    if final_answer:
        console.print(Markdown(final_answer.draft_answer))
        if final_answer.citations:
            console.print("\n[bold]Sources:[/bold]")
            for cit in final_answer.citations:
                console.print(f"- {cit}")
    else:
        console.print("[bold red]Failed to generate a satisfactory answer after retries.[/bold red]")
        # 即使失败也打印最后一次的结果
        if draft_output:
            console.print(Markdown(draft_output.draft_answer))

def main():
    parser = argparse.ArgumentParser(description="SciMuse: Multi-Agent Scientific Document Analysis")
    
    subparsers = parser.add_subparsers(dest="mode", help="Run mode", required=True)
    
    # Ingest Mode
    ingest_parser = subparsers.add_parser("ingest", help="Parse PDF and ingest into Vector DB")
    ingest_parser.add_argument("path", type=str, help="Path to the PDF file")
    
    # Chat Mode
    chat_parser = subparsers.add_parser("chat", help="Ask questions about the ingested papers")
    chat_parser.add_argument("query", type=str, help="The question to ask")
    
    args = parser.parse_args()
    
    cfg = load_config()

    if args.mode == "ingest":
        ingest_document(cfg, args.path)
    elif args.mode == "chat":
        run_chat_pipeline(cfg, args.query)

if __name__ == "__main__":
    main()