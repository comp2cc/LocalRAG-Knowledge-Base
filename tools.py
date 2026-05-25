import sqlite3
import re
import os
from typing import Dict, Any

# 工具1：敏感词检测
def sensitive_word_check(query: str, word_list_path: str = "data/sensitive_words.txt") -> Dict[str, Any]:
    """
    从文件加载敏感词库，检测 query 中是否包含敏感词。
    返回: {"has_sensitive": bool, "found_words": list}
    """
    if not os.path.exists(word_list_path):
        # 若文件不存在，返回空结果（相当于无敏感词）
        return {"has_sensitive": False, "found_words": []}
    
    with open(word_list_path, "r", encoding="utf-8") as f:
        sensitive_words = [line.strip() for line in f if line.strip()]
    
    found = [word for word in sensitive_words if word in query]
    return {"has_sensitive": len(found) > 0, "found_words": found}

# 工具2：本地分类模型判断（这里使用模拟实现，你可替换为真实模型）
def local_classifier_check(query: str, model=None) -> Dict[str, Any]:
    """
    调用微调后的模型进行违规判断。
    返回: {"prediction": "安全" 或 "违规", "confidence": float}
    """
    # TODO: 替换为实际模型调用（例如加载你微调好的 LoRA 模型）
    # 示例：简单的关键词规则
    dangerous_keywords = ["炸", "杀", "暴力"]
    if any(kw in query for kw in dangerous_keywords):
        return {"prediction": "违规", "confidence": 0.85}
    else:
        return {"prediction": "安全", "confidence": 0.90}

# 工具3：黑名单查询（基于 SQLite）
def blacklist_query(user_id: str, db_path: str = "data/blacklist.db") -> Dict[str, Any]:
    """
    查询该用户历史违规次数。
    返回: {"violation_count": int, "is_blacklisted": bool}
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # 创建表（如果不存在）
    c.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            user_id TEXT PRIMARY KEY,
            count INTEGER DEFAULT 0,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    
    c.execute("SELECT count FROM violations WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    count = row[0] if row else 0
    conn.close()
    
    is_blacklisted = count >= 3  # 3次以上视为黑名单
    return {"violation_count": count, "is_blacklisted": is_blacklisted}

# 辅助函数：记录违规（当 Agent 决策为违规时，可调用此函数更新黑名单）
def record_violation(user_id: str, db_path: str = "data/blacklist.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT INTO violations (user_id, count) VALUES (?, 1)
        ON CONFLICT(user_id) DO UPDATE SET count = count + 1, last_update = CURRENT_TIMESTAMP
    ''', (user_id,))
    conn.commit()
    conn.close()