from langgraph.graph import StateGraph, END
from .state import AgentState
from .tools import sensitive_word_check, local_classifier_check, blacklist_query, record_violation

# 定义各个节点（每个节点接收 state，返回更新后的 state）
def check_sensitive(state: AgentState) -> AgentState:
    result = sensitive_word_check(state["query"])
    state["tools_output"]["sensitive"] = result
    return state

def check_classifier(state: AgentState) -> AgentState:
    result = local_classifier_check(state["query"])
    state["tools_output"]["classifier"] = result
    return state

def check_blacklist(state: AgentState) -> AgentState:
    result = blacklist_query(state["user_id"])
    state["tools_output"]["blacklist"] = result
    return state

def make_decision(state: AgentState) -> AgentState:
    """
    综合三个工具的输出，做出最终决策。
    决策逻辑：
        - 若敏感词命中 → 违规
        - 若分类器预测违规且置信度 > 0.7 → 违规
        - 若黑名单中违规次数 ≥ 3 → 违规
        - 否则安全
    """
    sensitive = state["tools_output"].get("sensitive", {})
    classifier = state["tools_output"].get("classifier", {})
    blacklist = state["tools_output"].get("blacklist", {})
    
    if sensitive.get("has_sensitive"):
        reason = f"命中敏感词: {', '.join(sensitive['found_words'])}"
        state["final_decision"] = f"违规：{reason}"
        state["reasoning"] = reason
        # 记录违规行为
        record_violation(state["user_id"])
        return state
    
    if classifier.get("prediction") == "违规" and classifier.get("confidence", 0) > 0.7:
        reason = f"分类模型判断违规，置信度 {classifier['confidence']:.2f}"
        state["final_decision"] = f"违规：{reason}"
        state["reasoning"] = reason
        record_violation(state["user_id"])
        return state
    
    if blacklist.get("is_blacklisted"):
        reason = f"历史违规次数 {blacklist['violation_count']} 次，达到黑名单阈值"
        state["final_decision"] = f"违规：{reason}"
        state["reasoning"] = reason
        return state
    
    state["final_decision"] = "安全"
    state["reasoning"] = "所有检测均通过"
    return state

# 构建 LangGraph 工作流
def build_safety_agent():
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("sensitive", check_sensitive)
    workflow.add_node("classifier", check_classifier)
    workflow.add_node("blacklist", check_blacklist)
    workflow.add_node("decision", make_decision)
    
    # 设置边（按顺序执行）
    workflow.set_entry_point("sensitive")
    workflow.add_edge("sensitive", "classifier")
    workflow.add_edge("classifier", "blacklist")
    workflow.add_edge("blacklist", "decision")
    workflow.add_edge("decision", END)
    
    return workflow.compile()

# 对外接口：运行 Agent，返回 (final_decision, reasoning)
def run_agent(query: str, user_id: str) -> tuple:
    app = build_safety_agent()
    initial_state: AgentState = {
        "query": query,
        "user_id": user_id,
        "step": "",
        "tools_output": {},
        "final_decision": "",
        "reasoning": ""
    }
    final_state = app.invoke(initial_state)
    return final_state["final_decision"], final_state["reasoning"]