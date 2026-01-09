import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

# API 配置，不配置也能使用普通搜索功能
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3.2"

# AI 开关（若为 false，则不调用 LLM，直接使用原始查询）
USE_AI = os.getenv("USE_AI", "true").lower() == "true"

# 数据/清洗参数
DATA_DIR = "./data/"
CLEAN_MIN_DEGREE = 2
CLEAN_MAX_TOKEN_LEN = 120

def build_openai_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
