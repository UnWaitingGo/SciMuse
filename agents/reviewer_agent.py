import json
from omegaconf import DictConfig
from smolagents import LiteLLMModel
from schema import ReviewerOutput, ReasonerOutput, AgentDecision

class ReviewerAgent:
    def __init__(self, config: DictConfig):
        self.config = config
        
        # 使用配置中的 Reviewer 或 Reasoning 模型
        # 通常 Reviewer 可以用和 Reasoner 同样的模型，或者更强的模型
        raw_model_id = config.models.reasoning.model_id
        if not raw_model_id.startswith("openai/") and "deepseek" in raw_model_id.lower():
            model_id_for_litellm = f"openai/{raw_model_id}"
        else:
            model_id_for_litellm = raw_model_id

        self.model = LiteLLMModel(
            model_id=model_id_for_litellm,
            api_base=config.api.base_url,
            api_key=config.api.api_key,
            temperature=0.1,  # 审查需要冷静、客观
            max_tokens=2048
        )

    def _clean_json_output(self, raw_output: str) -> str:
        """清洗 JSON 输出"""
        raw_output = raw_output.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.startswith("```"):
            raw_output = raw_output[3:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
        return raw_output.strip()

    def review(self, query: str, reasoner_output: ReasonerOutput) -> ReviewerOutput:
        """
        审查 Reasoner 的输出
        """
        print(f"[*] [Reviewer Agent] Reviewing draft answer...")

        system_prompt = """You are a Quality Assurance (QA) Agent for a scientific RAG system.
        Your goal is to evaluate the 'Draft Answer' provided by a Reasoner Agent based on the 'User Query'.
        
        ### EVALUATION CRITERIA:
        1. **Relevance**: Does the answer directly address the user's specific question?
        2. **Evidence**: Does the answer contain citations (e.g., [Page 4], [Fig.3])? 
        3. **Consistency**: Is the logic sound?
        4. **Visuals**: If the question asks about visual aspects (trends, plots), are figures cited?

        ### DECISION LOGIC:
        - Score < 0.8: If the answer is vague, misses the core question, lacks citations, or hallucinates.
        - Score >= 0.8: If the answer is accurate, well-cited, and clear.

        ### OUTPUT FORMAT:
        You MUST output valid JSON with this structure:
        {
            "confidence_score": 0.9, (float between 0.0 and 1.0)
            "decision": "ACCEPT",    (or "REJECT")
            "critique": "The answer is good but...", (String explanation)
            "feedback_for_retriever": "" (If REJECT, provide a new search query or specific instruction to find missing info. If ACCEPT, leave empty.)
        }
        """

        user_message = f"""
        USER QUERY: {query}
        
        === DRAFT ANSWER TO REVIEW ===
        {reasoner_output.draft_answer}
        
        === PROVIDED CITATIONS ===
        {reasoner_output.citations}
        
        Please evaluate this answer now.
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
            data = json.loads(cleaned_json)

            return ReviewerOutput(
                confidence_score=float(data.get("confidence_score", 0.0)),
                decision=AgentDecision(data.get("decision", "REJECT")),
                critique=data.get("critique", "No critique provided."),
                feedback_for_retriever=data.get("feedback_for_retriever", None)
            )

        except Exception as e:
            print(f"[!] [Reviewer Agent] Error: {str(e)}")
            # 出错时保守处理，拒绝并重试
            return ReviewerOutput(
                confidence_score=0.0,
                decision=AgentDecision.REJECT,
                critique=f"System error during review: {str(e)}",
                feedback_for_retriever="System error, please retry aggregation."
            )