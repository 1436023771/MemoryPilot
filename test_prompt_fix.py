#!/usr/bin/env python3
"""
验证改进后的 prompt 是否能正确地从对话中只提取用户信息，
而不是提取助手提供的知识内容。
"""

from app.write_memory import extract_candidate_facts_from_dialogue
from langchain_core.messages import HumanMessage, AIMessage

def test_ignore_assistant_knowledge():
    """测试：当助手提供知识检索结果时，应该忽略这些内容"""
    
    messages = [
        HumanMessage(content="我叫小王，我想学习Python编程"),
        AIMessage(content="很好，小王。我为你查找了关于Python的信息。Python是一种高级编程语言，由Guido van Rossum在1989年发明。它以易读性著称，广泛应用于数据科学、Web开发、自动化等领域。"),
        HumanMessage(content="谢谢，我也很喜欢算法"),
        AIMessage(content="很好！我再给你搜索了一些资源。算法是计算机科学的基础，包括排序、搜索、动态规划等多个领域。"),
    ]
    
    facts = extract_candidate_facts_from_dialogue(messages)
    print("=" * 60)
    print("测试：忽略助手提供的知识")
    print("=" * 60)
    print(f"\n输入对话信息：")
    print(f"  - 用户说的话：'我叫小王，我想学习Python编程'")
    print(f"  - 用户说的话：'谢谢，我也很喜欢算法'")
    print(f"  - 助手的回复1：包含Python的百科知识（应被忽略）")
    print(f"  - 助手的回复2：包含算法的知识（应被忽略）")
    print(f"\n提取结果：")
    for fact in facts:
        print(f"  - key={fact.key}, value={fact.value}")
    
    # 验证：应该提取用户信息，不提取"Guido van Rossum"或"1989"等知识
    extracted_values = {fact.value for fact in facts}
    
    assert "小王" in extracted_values, "应该提取用户的姓名"
    assert "Python" in extracted_values or any("Python" in str(f) for f in facts), "可以提取用户喜欢的技术"
    assert "算法" in extracted_values, "应该提取用户的兴趣爱好"
    assert "Guido van Rossum" not in extracted_values, "❌ 问题：提取了Python的发明者信息（这是知识，不是用户信息）"
    assert "1989" not in extracted_values, "❌ 问题：提取了年份信息（这是知识，不是用户信息）"
    assert "高级编程语言" not in extracted_values, "❌ 问题：提取了通用定义（这是知识，不是用户信息）"
    
    print(f"\n✅ 验证通过！新的 prompt 成功过滤掉了助手提供的知识内容。")
    print(f"   只提取了用户本人的信息：{extracted_values}")

def test_extract_user_preferences():
    """测试：正确提取用户个人信息和偏好"""
    
    messages = [
        HumanMessage(content="我叫李华，我是一个软件工程师，我喜欢函数式编程"),
        AIMessage(content="很高兴认识你。函数式编程是一种编程范式，强调函数的使用和不可变数据..."),
        HumanMessage(content="是的，我讨厌代码重复"),
        AIMessage(content="代码重复（Code Duplication）是一个常见的问题..."),
    ]
    
    facts = extract_candidate_facts_from_dialogue(messages)
    print("\n" + "=" * 60)
    print("测试：正确提取用户个人信息")
    print("=" * 60)
    print(f"\n输入对话信息：")
    print(f"  - 用户说的话：'我叫李华，我是一个软件工程师，我喜欢函数式编程'")
    print(f"  - 用户说的话：'是的，我讨厌代码重复'")
    print(f"\n提取结果：")
    for fact in facts:
        print(f"  - key={fact.key}, value={fact.value}")
    
    extracted_values = {fact.value for fact in facts}
    
    assert "李华" in extracted_values, "应该提取用户的姓名"
    assert "软件工程师" in extracted_values or any("软件" in str(v) for v in extracted_values), "应该提取用户的职位或身份"
    assert "函数式编程" in extracted_values, "应该提取用户的偏好"
    assert "代码重复" in extracted_values or "讨厌代码重复" in extracted_values or "代码重复" in str(facts), "应该提取用户的厌恶之处"
    
    # 验证没有提取助手提供的定义
    assert not any("范式" in str(v) and "编程" in str(v) for v in extracted_values if "用户喜欢" not in str(v)), "不应该提取范式的定义"
    
    print(f"\n✅ 验证通过！新的 prompt 正确提取了用户的个人信息。")

if __name__ == "__main__":
    try:
        test_ignore_assistant_knowledge()
        test_extract_user_preferences()
        print("\n" + "=" * 60)
        print("🎉 所有验证成功！改进后的 prompt 工作正常。")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ 验证失败：{e}")
        import sys
        sys.exit(1)
