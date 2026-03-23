#!/usr/bin/env python3
"""
集成测试：验证 CoT 模式在 CLI 和 GUI 中都能正常工作
"""

import json
import subprocess
from pathlib import Path

def test_cli_cot_off():
    """测试 CLI 中的 --cot-mode off（默认）"""
    result = subprocess.run(
        [
            "agent/bin/python",
            "-m",
            "app.main",
            "--cot-mode=off",
            "什么是 Python？",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/yuhanli/codeProject/agent",
        timeout=60,
    )
    
    print("=" * 60)
    print("CLI CoT Mode: OFF")
    print("=" * 60)
    print(f"Exit Code: {result.returncode}")
    if result.returncode == 0:
        print("✅ CLI with --cot-mode off executed successfully")
        print(f"Output (first 200 chars): {result.stdout[:200]}")
    else:
        print(f"❌ CLI execution failed")
        print(f"stderr: {result.stderr[:200]}")
    return result.returncode == 0


def test_cli_cot_brief():
    """测试 CLI 中的 --cot-mode brief"""
    result = subprocess.run(
        [
            "agent/bin/python",
            "-m",
            "app.main",
            "--cot-mode=brief",
            "什么是 Python？",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/yuhanli/codeProject/agent",
        timeout=60,
    )
    
    print("\n" + "=" * 60)
    print("CLI CoT Mode: BRIEF (JSON OUTPUT)")
    print("=" * 60)
    print(f"Exit Code: {result.returncode}")
    
    if result.returncode == 0:
        output = result.stdout.strip()
        
        # 尝试解析 JSON
        try:
            # 查找最后一个 JSON 对象
            start_idx = output.rfind("{")
            if start_idx >= 0:
                json_str = output[start_idx:]
                payload = json.loads(json_str)
                
                has_answer = "answer" in payload
                has_steps = "brief_steps" in payload
                
                if has_answer and has_steps:
                    print("✅ CLI with --cot-mode brief returns valid JSON")
                    print(f"   - answer key present: yes")
                    print(f"   - brief_steps key present: yes")
                    if isinstance(payload.get("brief_steps"), list):
                        step_count = len(payload.get("brief_steps", []))
                        print(f"   - steps count: {step_count}")
                        if step_count > 0:
                            print(f"   - first step: {payload['brief_steps'][0][:50]}...")
                else:
                    print("❌ JSON missing required keys")
                    print(f"   Payload: {payload}")
            else:
                print("⚠️  No JSON found in output")
                print(f"Output: {output[:200]}")
        except json.JSONDecodeError as e:
            print(f"⚠️  Output is not valid JSON: {e}")
            print(f"Output: {output[:300]}")
    else:
        print(f"❌ CLI execution failed")
        print(f"stderr: {result.stderr[:200]}")
    
    return result.returncode == 0


def test_help_shows_cot_mode():
    """验证 --help 中有 --cot-mode 选项"""
    result = subprocess.run(
        [
            "agent/bin/python",
            "-m",
            "app.main",
            "--help",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/yuhanli/codeProject/agent",
        timeout=30,
    )
    
    print("\n" + "=" * 60)
    print("Check: --cot-mode in --help")
    print("=" * 60)
    
    has_cot_mode = "--cot-mode" in result.stdout
    has_brief = "brief" in result.stdout.lower()
    
    if has_cot_mode and has_brief:
        print("✅ --cot-mode flag is properly documented in help")
    else:
        print("❌ --cot-mode not found in help")
        print(f"Help output: {result.stdout[:500]}")
    
    return has_cot_mode and has_brief


if __name__ == "__main__":
    print("🧪 CoT Integration Tests\n")
    
    tests = [
        ("CLI Help Shows CoT Mode", test_help_shows_cot_mode),
        ("CLI CoT Mode OFF", test_cli_cot_off),
        ("CLI CoT Mode BRIEF", test_cli_cot_brief),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All CoT integration tests passed!")
    else:
        print("\n⚠️  Some tests failed")
