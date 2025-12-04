import json
import re
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
from smolagents import LiteLLMModel

# 引入 Schema 用于验证
from schema import ReasonerOutput, EvidenceItem

class ReasonerAgent:
    def __init__(self, config: DictConfig):
        self.config = config
        
        # 1. 配置 LiteLLM 模型
        raw_model_id = config.models.reasoning.model_id
        
        # 简单的处理 model_id 前缀 (LiteLLM 规范)
        if not raw_model_id.startswith("openai/") and "deepseek" in raw_model_id.lower():
             # 如果是 siliconflow 或 deepseek 直连，通常在 litellm 中作为 openai/ 兼容接口调用
            model_id_for_litellm = f"openai/{raw_model_id}"
        else:
            model_id_for_litellm = raw_model_id

        self.model = LiteLLMModel(
            model_id=model_id_for_litellm,
            api_base=config.api.base_url,
            api_key=config.api.api_key,
            temperature=config.models.reasoning.temperature,
            max_tokens=config.models.reasoning.max_tokens
        )

    def _format_context(self, retrieved_text: str, vl_insights: List[Dict[str, str]]) -> str:
        """
        将检索到的文本和视觉模型的洞察合并为格式化的上下文 Prompt
        """
        context_str = "### 1. Textual Evidence (from Papers):\n"
        context_str += retrieved_text if retrieved_text else "No text evidence found."
        
        context_str += "\n\n### 2. Visual Evidence (from Figures):\n"
        if vl_insights:
            for idx, item in enumerate(vl_insights):
                desc = item.get("description", "N/A")
                insight = item.get("insights", "N/A")
                context_str += f"Figure Analysis {idx+1}:\n- Description: {desc}\n- Key Insight: {insight}\n\n"
        else:
            context_str += "No visual evidence provided."
            
        return context_str

    def _clean_json_output(self, raw_output: str) -> str:
        """
        清洗 LLM 输出，提取 JSON 部分
        """
        # 移除 markdown 代码块标记
        raw_output = raw_output.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.startswith("```"):
            raw_output = raw_output[3:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
        return raw_output.strip()

    def run(self, query: str, retriever_result: str, vl_results: List[Any]) -> ReasonerOutput:
        """
        执行推理任务
        Args:
            query: 用户的原始问题
            retriever_result: RetrieverAgent 返回的原始字符串文本
            vl_results: VL Agent 返回的 VLOutput 对象列表 (或字典列表)
        """
        print(f"[*] [Reasoner Agent] Synthesizing answer for: {query}")

        # 转换 VL 对象为字典 (如果是 VLOutput 对象)
        vl_data = []
        for item in vl_results:
            if hasattr(item, 'model_dump'):
                vl_data.append(item.model_dump())
            elif hasattr(item, '__dict__'):
                vl_data.append(item.__dict__)
            else:
                vl_data.append(item)

        # 1. 构建 Prompt
        context = self._format_context(retriever_result, vl_data)
        
        system_prompt = """You are an expert scientific researcher and academic writer. 
        Your task is to answer the User's Query based STRICTLY on the provided Evidence (Textual and Visual).

        ### INSTRUCTIONS:
        1. **Synthesis**: Combine insights from both text and figures. If figures contradict text, note the discrepancy.
        2. **Citations**: You MUST cite your sources. 
           - For text, use [Source ID] or [Page X] if available in the text evidence.
           - For figures, use [Fig. X].
        3. **Reasoning Trace**: Briefly explain your logic chain before giving the final answer.
        4. **Honesty**: If the evidence is insufficient to answer the question, state that clearly. Do not hallucinate information not present in the context.

        ### OUTPUT FORMAT:
        You must output a valid JSON object with EXACTLY the following structure (no markdown, just JSON):
        {
            "reasoning_trace": "Step-by-step logic used to derive the answer...",
            "draft_answer": "The comprehensive answer to the user query...",
            "citations": ["Fig.1", "Page 2", "Ref [3]"]
        }
        """

        user_message = f"""
        USER QUERY: {query}
        
        === CONTEXT START ===
        {context}
        === CONTEXT END ===
        
        Please provide the JSON response now.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # 2. 调用模型
        # Smolagents 的 LiteLLMModel 主要是给 CodeAgent 用的，但也可以直接用它的底层逻辑，
        # 或者直接作为 callable 调用 (取决于版本)。
        # 这里我们使用 LiteLLMModel 的标准调用方式模拟对话。
        
        try:
            # LiteLLMModel.__call__ 通常接受 messages 列表
            response = self.model(messages)
            
            # smolagents 的 model 返回通常是 ToolMessage 或简单的 Message 对象
            # 如果是直接返回字符串则直接使用
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # 3. 解析与验证
            cleaned_json = self._clean_json_output(content)
            
            try:
                data = json.loads(cleaned_json)
                
                # 验证并构建 Pydantic 对象
                output = ReasonerOutput(
                    reasoning_trace=data.get("reasoning_trace", "No trace provided."),
                    draft_answer=data.get("draft_answer", "No answer provided."),
                    citations=data.get("citations", [])
                )
                return output

            except json.JSONDecodeError:
                print(f"[!] [Reasoner Agent] JSON Parsing failed. Raw output:\n{content}")
                # 降级处理：将整个内容作为 draft_answer
                return ReasonerOutput(
                    reasoning_trace="JSON Parse Error",
                    draft_answer=content,
                    citations=[]
                )

        except Exception as e:
            print(f"[!] [Reasoner Agent] Error during inference: {str(e)}")
            return ReasonerOutput(
                reasoning_trace="System Error",
                draft_answer=f"An error occurred while generating the answer: {str(e)}",
                citations=[]
            )