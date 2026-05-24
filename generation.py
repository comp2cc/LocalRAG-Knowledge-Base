import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        )
        self.model = "deepseek-chat"
    
    def generate(self, query, contexts, history=None):
        """
        query: 当前用户问题
        contexts: 检索到的文本片段列表
        history: 历史对话列表，每个元素为 (user_msg, assistant_msg)
        """
        system_prompt = """你是一个基于知识库的智能助手。请严格根据【参考内容】和【历史对话】回答用户问题。
如果参考内容中没有相关信息，请明确说“根据现有资料无法回答该问题”，不要编造信息。
回答要简洁、准确，并用markdown格式列出要点。最后附上引用来源。"""
        
        # 构建历史上下文字符串
        history_str = ""
        if history:
            history_str = "\n".join([f"用户：{u}\n助手：{a}" for u, a in history[-6:]])  # 最近3轮
        
        context_str = "\n\n---\n".join([f"[片段{i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
        
        user_prompt = f"""【参考内容】
{context_str}

【历史对话】
{history_str if history_str else "无"}

【当前问题】
{query}

请回答："""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content