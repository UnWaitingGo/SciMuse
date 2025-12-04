SciMuse:Scientific Collaborative Intelligence for Multi-agent Understanding & Synthesis Engine（用于多智能体理解与合成的科学协作智能引擎）

项目结构
SciMuse/
├── agents/                 # Agent实现
│   ├── planner.py         # Planner Agent
│   ├── retriever.py       # Retriever Agent
│   ├── caption.py         # Caption Agent
│   ├── reasoner.py        # Reasoner Agent
│   └── reviewer.py        # Reviewer Agent
├── models/                 # 模型文件
│   ├── qwen-embed/
│   ├── qwen-vl/
│   └── qwen-coder/
├── tools/                  # 工具函数
│   ├── pdf_parser.py      # PDF解析
│   └── vector_db.py       # 向量数据库
├── data/                   # 数据目录
│   ├── pdfs/              # 原始PDF
│   └── processed/         # 处理后数据
├── experiments/            # 实验结果
├── docs/                   # 文档
├── requirements.txt        # 依赖列表
├── config.yaml            # 配置文件
└── main.py                # 主程序入口

使用指南
1. 前置准备
确保你已经安装了 rich 库用于美化终端输出（除了你 requirements.txt 中已有的依赖）：
pip install requirements.txt
确保根目录下有 .env 文件配置了相关 API Key：
# .env
OPENAI_API_KEY=sk-xxxx
MINERU_TOKEN=xxxx
2. 运行入库 (Ingestion)
将 PDF 文件解析并存入 ChromaDB：
python main.py ingest data/pdfs/demo2.pdf

流程：PDF Parser 提取 -> VL Agent 分析图片 -> Embedding -> 存入 vector_db/chroma_db。

3. 运行问答 (Chat)
针对已入库的内容提问：
python main.py chat "Explain what is it shown in Figure 1 in chinese"