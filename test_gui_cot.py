#!/usr/bin/env python3
"""
测试 GUI 中的 CoT 响应解析功能
"""

from app.gui_chat import ChatWindow
import json

def test_parse_cot_response():
    """测试 CoT 响应解析"""
    # 模拟一个任意的 GUI 实例（不启动 Tkinter）
    # 通过直接测试 _parse_cot_response 方法
    
    # 创建一个最小化的 GUI 实例
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    try:
        chat = ChatWindow()
        
        # 测试1：off 模式 - 原样返回
        chat.cot_mode = "off"
        answer, steps = chat._parse_cot_response("普通回答内容")
        assert answer == "普通回答内容", f"off 模式失败：{answer}"
        assert steps == [], f"off 模式应该返回空步骤：{steps}"
        print("✅ Test 1 passed: off 模式正确返回原文")
        
        # 测试2：brief 模式 - 解析 JSON
        chat.cot_mode = "brief"
        cot_json = {
            "answer": "最终答案是42",
            "brief_steps": [
                "第一步：理解问题",
                "第二步：分析条件",
                "第三步：得出结论"
            ]
        }
        answer, steps = chat._parse_cot_response(json.dumps(cot_json))
        assert answer == "最终答案是42", f"解析答案失败：{answer}"
        assert len(steps) == 3, f"步骤个数应该是3，得到{len(steps)}"
        assert steps[0] == "第一步：理解问题", f"第一步失败：{steps[0]}"
        print("✅ Test 2 passed: brief 模式正确解析 JSON")
        
        # 测试3：brief 模式 - JSON 格式但步骤为空
        chat.cot_mode = "brief"
        cot_json_no_steps = {
            "answer": "只有答案，没有步骤"
        }
        answer, steps = chat._parse_cot_response(json.dumps(cot_json_no_steps))
        assert answer == "只有答案，没有步骤", f"答案解析失败：{answer}"
        assert steps == [], f"没有步骤时应返回空：{steps}"
        print("✅ Test 3 passed: brief 模式在缺少步骤时正确处理")
        
        # 测试4：brief 模式 - 无效 JSON，回退到原文
        chat.cot_mode = "brief"
        answer, steps = chat._parse_cot_response("这不是JSON的文本内容")
        assert answer == "这不是JSON的文本内容", f"回退失败：{answer}"
        assert steps == [], f"无效JSON应返回空步骤：{steps}"
        print("✅ Test 4 passed: brief 模式在无效JSON时正确回退")
        
        print("\n🎉 所有 CoT 解析测试通过！")
        
    finally:
        root.destroy()

if __name__ == "__main__":
    test_parse_cot_response()
