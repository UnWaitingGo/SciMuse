import os
from smolagents import CodeAgent, LiteLLMModel
from omegaconf import DictConfig

# 引入定义好的工具
from tools.agent_tools import retriever_tool

class RetrieverAgent:
    def __init__(self, config: DictConfig):
        self.config = config
        
        # 处理 LiteLLM 模型 ID
        raw_model_id = config.models.reasoning.model_id
        if not raw_model_id.startswith("openai/"):
            model_id_for_litellm = f"openai/{raw_model_id}"
        else:
            model_id_for_litellm = raw_model_id

        # 1. 定义模型
        self.model = LiteLLMModel(
            model_id=model_id_for_litellm, 
            api_base=config.api.base_url,
            api_key=config.api.api_key,
            temperature=0.1, 
            max_tokens=4096
        )

        # 2. 定义行为指令 (更加严格，防止死循环)
        self.instructions = """
        【角色设定】
        你是一个专业的科学文献检索专家。你的任务是从向量数据库中检索具体的证据（文本和图表）来回答用户的问题。
        
        【任务】
        根据用户的问题，使用 `retriever_tool` 在知识库中搜索最相关的文本片段或图表描述。

        【规则】
        1. 必须调用 `retriever_tool` 获取信息。
        2. 工具会直接返回文本和图片的详细描述（如果存在），你不需要额外进行图片分析。
        3. 拿到工具返回的内容后，直接输出作为回答，不要进行过度的额外发挥，你的任务是提供证据。
        4. 承认失败 : 如果找不到相关内容，直接告诉用户“数据库中未找到相关信息”，不要捏造。
        """

        # 3. 初始化 Agent
        self.agent = CodeAgent(
            tools=[retriever_tool], 
            model=self.model,
            name="Retriever_agent",
            description="负责从向量数据库中检索文本和可视化证据。",
            add_base_tools=True,
            max_steps=6  
        )

    def run(self, query: str):
        print(f"[*] [Retriever Agent] Start searching for: {query}")
        
        # 提示词技巧：强制让 CodeAgent 把工具的输出打印或返回
        task = f"""
        {self.instructions}
        
        USER QUERY: {query}
        
        Please write python code to search the database and print the raw results found.
        """
        
        try:
            # CodeAgent 最终会返回它最后一步的输出
            result = self.agent.run(task)
            return result
        except Exception as e:
            return f"Retriever failed: {str(e)}"