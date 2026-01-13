import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

# API 配置，不配置也能使用普通搜索功能
# 你需要在根目录创建一个 .env 文件，写上 OPENAI_API_KEY="你的密钥"，并检查下面的模型配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") #
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3.2-exp"

# 数据路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# 清洗参数
CLEAN_MIN_DEGREE = 2
CLEAN_MAX_TOKEN_LEN = 120

# 排序参数
CITATION_WEIGHT_ALPHA = 0.2

def build_openai_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
