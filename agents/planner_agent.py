import json
from typing import List
from omegaconf import DictConfig
from smolagents import LiteLLMModel

# 引入 Schema 用于验证
from schema import PlannerOutput

class PlannerAgent:
    def __init__(self, config: DictConfig):
        self.config = config
        
        # 使用配置中的 Planner 或 Reasoning 模型
        # Planner 需要较好的逻辑拆解能力
        raw_model_id = config.models.reasoning.model_id
        if not raw_model_id.startswith("openai/") and "deepseek" in raw_model_id.lower():
            model_id_for_litellm = f"openai/{raw_model_id}"
        else:
            model_id_for_litellm = raw_model_id

        self.model = LiteLLMModel(
            model_id=model_id_for_litellm,
            api_base=config.api.base_url,
            api_key=config.api.api_key,
            temperature=0.1,  # 规划需要确定性
            max_tokens=1024
        )

    def _clean_json_output(self, raw_output: str) -> str:
        """清洗 LLM 输出的 JSON 字符串"""
        raw_output = raw_output.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.startswith("```"):
            raw_output = raw_output[3:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
        return raw_output.strip()

    def plan(self, user_query: str) -> PlannerOutput:
        """
        分析用户问题，生成检索计划
        """
        print(f"[*] [Planner Agent] Analyzing query: {user_query}")

        system_prompt = """You are a Strategic Planner Agent for a scientific document analysis system.
        Your goal is to break down the User's Query into specific search actions for a Vector Database Retriever.

        ### TASKS:
        1. **Analyze**: Understand the core intent of the user's question.
        2. **Keywords**: Generate a list of specific, semantically rich search queries to find relevant text or figures in a scientific paper. Avoid generic words like "paper" or "article".
        3. **Visual Check**: Determine if the question implies looking at charts, graphs, figures, or visual results (e.g., "compare plots", "show trends", "Figure 3").

        ### OUTPUT FORMAT:
        You MUST output a valid JSON object strictly matching this schema:
        {
            "reasoning": "Brief explanation of why these queries were chosen...",
            "search_queries": ["keyword 1", "phrase 2", "specific term 3"],
            "need_visual_understanding": true/false
        }
        """

        user_message = f"""
        USER QUERY: {user_query}
        
        Please generate the search plan in JSON format.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            response = self.model(messages)
            
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            cleaned_json = self._clean_json_output(content)
            
            try:
                data = json.loads(cleaned_json)
                
                # 验证并构建 Pydantic 对象
                output = PlannerOutput(
                    reasoning=data.get("reasoning", "No reasoning provided."),
                    search_queries=data.get("search_queries", [user_query]), # 保底使用原问题
                    need_visual_understanding=data.get("need_visual_understanding", False)
                )
                return output

            except json.JSONDecodeError:
                print(f"[!] [Planner Agent] JSON Parse Error. Raw: {content}")
                # 降级策略：直接把原问题作为搜索词
                return PlannerOutput(
                    reasoning="JSON parsing failed, using original query.",
                    search_queries=[user_query],
                    need_visual_understanding=False
                )

        except Exception as e:
            print(f"[!] [Planner Agent] System Error: {str(e)}")
            return PlannerOutput(
                reasoning=f"System error: {str(e)}",
                search_queries=[user_query],
                need_visual_understanding=False
            )