import sys
import os
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv  

# 解决 Windows 控制台可能的中文乱码
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# ================= 路径设置 =================
# 获取项目根目录 (假设当前文件在 test/ 目录下，向上两级是根目录)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 将根目录加入 Python 路径，这样才能 import agents 和 tools
sys.path.append(str(PROJECT_ROOT))

# ================= 导入模块 =================
try:
    from agents.retriever_agent import RetrieverAgent
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"当前 sys.path: {sys.path}")
    exit(1)

def test_retriever_agent():
    print(f"[*] 项目根目录: {PROJECT_ROOT}")

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"[*] 已加载环境变量文件: {env_path}")
    else:
        print(f"[!] 警告: 未找到 .env 文件于 {env_path}，如果环境变量未在系统设置，可能会报错。")

    # 2. 加载 config.yaml
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        print(f"[!] 错误: 找不到配置文件 {config_path}")
        return

    print(f"[*] 加载配置: {config_path}")
    cfg = OmegaConf.load(config_path)

    # 3. 初始化 Agent
    print("[*] 初始化 RetrieverAgent...")
    try:
        retriever = RetrieverAgent(cfg)
    except Exception as e:
        print(f"[!] 初始化失败: {e}")
        # 打印详细堆栈以便调试
        import traceback
        traceback.print_exc()
        return

    # 4. 构造测试查询
    # 这个查询旨在触发 Agent 调用 retriever_tool
    query = "What is presented in this figure? Please explain detailedly in chinese."
    
    print("-" * 50)
    print(f"[*] 开始测试查询: {query}")
    print("-" * 50)

    # 5. 运行
    try:
        response = retriever.run(query)
        print("-" * 50)
        print("[Retriever Agent] 最终回答:")
        print(response)
        print("-" * 50)
    except Exception as e:
        print(f"[!] 运行期间发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retriever_agent()