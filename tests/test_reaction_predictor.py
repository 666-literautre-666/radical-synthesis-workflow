"""
测试 reaction_predictor 模块

运行:
    python3 -m pytest tests/test_reaction_predictor.py -v
    或
    python3 tests/test_reaction_predictor.py
"""

import sys
from pathlib import Path

# 把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.reaction_predictor import (
    analyze_substrate,
    suggest_reaction_routes,
    predict_conditions,
    _estimate_bde,
    _assess_synthesizability,
)


def test_analyze_substrate_basic():
    """测试：溴化苄底物分析"""
    result = analyze_substrate("BrCc1ccccc1")
    assert result["formula"] == "C7H7Br"
    assert result["molecular_weight"] > 170
    assert result["n_reactive_sites"] >= 2  # C-Br + benzylic C-H


def test_analyze_substrate_invalid():
    """测试：非法 SMILES 的处理"""
    result = analyze_substrate("invalid_smiles")
    assert "error" in result


def test_estimate_bde_benzyl():
    """测试：苄位 BDE 估算，甲苯 benzylic C-H ~89 kcal/mol"""
    result = _estimate_bde("Cc1ccccc1")
    assert result["estimated_bde_kcal"] == 89
    assert "benzylic" in result["label"]


def test_estimate_bde_allylic():
    """测试：烯丙位 BDE 估算，丙烯 allylic C-H ~87 kcal/mol"""
    result = _estimate_bde("C=CC")
    # 丙烯 CH3-CH=CH2，allylic CH3 应该是 87
    assert result["estimated_bde_kcal"] == 87
    assert "allylic" in result["label"]


def test_estimate_bde_aldehyde():
    """测试：醛基 BDE 估算，乙醛 ~87 kcal/mol"""
    result = _estimate_bde("CC=O")
    assert result["estimated_bde_kcal"] == 87


def test_predict_conditions_smarts_match():
    """测试：predict_conditions 有 SMARTS 匹配"""
    result = predict_conditions("BrCc1ccccc1")
    assert result["smiles"] == "BrCc1ccccc1"
    assert len(result["smarts_matches"]) >= 2  # C-Br + benzylic C-H
    assert result["decision"]["worth_synthesizing"] is True


def test_assess_synthesizability_positive():
    """测试：合成可行性打分 — 应该推荐合成"""
    substrate = {"formula": "C7H7Br"}
    smarts_hits = [
        {"name": "C-Br", "reaction_family": "atom_transfer"},
        {"name": "benzylic_CH", "reaction_family": "HAT"},
    ]
    similar = [{"yield_percent": 85}]
    bde_info = {"estimated_bde_kcal": 88}

    result = _assess_synthesizability(substrate, smarts_hits, similar, bde_info)
    assert result["worth_synthesizing"] is True
    assert result["score"] >= 50


if __name__ == "__main__":
    # 手动运行所有测试
    tests = [
        ("test_analyze_substrate_basic", test_analyze_substrate_basic),
        ("test_analyze_substrate_invalid", test_analyze_substrate_invalid),
        ("test_estimate_bde_benzyl", test_estimate_bde_benzyl),
        ("test_estimate_bde_allylic", test_estimate_bde_allylic),
        ("test_estimate_bde_aldehyde", test_estimate_bde_aldehyde),
        ("test_predict_conditions_smarts_match", test_predict_conditions_smarts_match),
        ("test_assess_synthesizability_positive", test_assess_synthesizability_positive),
    ]

    passed = 0
    for name, test_func in tests:
        try:
            test_func()
            print(f"  PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    print(f"\n{passed}/{len(tests)} passed")
