import os
import sys
import time
import base64
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from openai import OpenAI


load_dotenv()

# 添加项目根目录到 sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from schema import VLOutput

class VLAgent:
    def __init__(self):
        self.config_path = BASE_DIR / "config.yaml"
        self._client = None
        self._cfg = None
        
        self._load_config()
        self._init_client()

    def _load_config(self):
        if not self.config_path.exists():
            print(f"[Warning] Config file not found at {self.config_path}")
            self._cfg = None
        else:
            # 加载配置
            self._cfg = OmegaConf.load(self.config_path)

    def _init_client(self):
        """初始化 OpenAI 客户端"""
        if self._cfg:
            api_key = self._cfg.api.api_key
            
            # 1. 尝试从 config 中获取 base_url
            # 使用 .get() 避免 'Missing key' 报错
            # 注意：OmegaConf 的 DictConfig 需要用 get 方法安全访问不存在的键
            vl_cfg = self._cfg.models.get("vl", {})
            base_url = vl_cfg.get("base_url")

            # 2. 如果 config 里没有配置 base_url，尝试从环境变量直接读取作为 Fallback
            if not base_url:
                # 尝试常见的环境变量名称
                base_url = os.getenv("BASE_URL") or os.getenv("VL_BASE_URL") or os.getenv("SILICONFLOW_BASE_URL")
                if base_url:
                    print(f"[*] [VL Agent] Config key missing, using env var: {base_url}")
            
            if not base_url:
                 # 最后的保底，或者报错
                 # base_url = "https://api.siliconflow.cn/v1" 
                 print("[!] [VL Agent] Warning: 'base_url' not found in config or env. API calls may fail.")
            
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
        else:
            print("[Error] Config not loaded, cannot init VL client.")

    def _encode_image(self, image_path: str) -> str:
        """辅助函数：将图片转换为 Base64 编码"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_image(self, image_path: str, query: str) -> VLOutput:
        """
        Uses a Vision-Language Model to analyze a scientific figure.
        """
        if not os.path.exists(image_path):
            return VLOutput(description="Error: Image file not found.", insights="Cannot analyze missing image.")

        print(f"[*] [VL Agent] Analyzing image: {Path(image_path).name}")

        if not self._client or not self._cfg:
            return VLOutput(description="Error: API Client not initialized", insights="Check config.")

        try:
            base64_image = self._encode_image(image_path)
        except Exception as e:
            return VLOutput(description="Error reading image file.", insights=str(e))

        system_prompt = """You are a scientific image analysis assistant. 
        Analyze the provided image and answer the user's query.
        You MUST output your response in valid JSON format with exactly two keys:
        {
            "description": "A detailed visual description...",
            "insights": "Key takeaways or answer to the query..."
        }
        Do not output markdown code blocks, just the raw JSON string.
        """

        model_id = self._cfg.models.vl.model_id
        max_tokens = self._cfg.models.vl.max_tokens
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user", 
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                {"type": "text", "text": query}
                            ]
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                
                raw_content = response.choices[0].message.content.strip()
                
                if raw_content.startswith("```json"):
                    raw_content = raw_content[7:]
                if raw_content.endswith("```"):
                    raw_content = raw_content[:-3]
                
                data = json.loads(raw_content)
                
                return VLOutput(
                    description=data.get("description", "No description provided."),
                    insights=data.get("insights", "No insights provided.")
                )

            except json.JSONDecodeError:
                print(f"[!] [VL Agent] JSON Parse Error (Attempt {attempt+1})")
                if attempt == max_retries - 1:
                    return VLOutput(description=raw_content, insights="Failed to parse structured insights.")
            except Exception as e:
                print(f"[!] [VL Agent] API Error (Attempt {attempt+1}): {str(e)}")
                time.sleep(2)
                if attempt == max_retries - 1:
                    return VLOutput(description="Error calling VL model.", insights=str(e))
        
        return VLOutput(description="Unknown Error.", insights="Max retries exceeded.")