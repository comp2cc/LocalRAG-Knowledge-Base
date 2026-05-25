from typing import TypedDict, Dict, Any, Optional

class AgentState(TypedDict):
    query: str                     # 用户输入
    user_id: str                   # 用户标识（用于黑名单）
    step: str                      # 当前步骤（保留）
    tools_output: Dict[str, Any]   # 存储各工具的输出
    final_decision: str            # 最终决策（"安全" 或 "违规：原因"）
    reasoning: str                 # 推理过程描述（可选）