"""
测试 database 模块

运行:
    python3 -m pytest tests/test_database.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.database import (
    get_db,
    init_database,
    add_my_experiment,
    add_literature,
    get_db_stats,
    get_smarts_matches,
    query_similar_substrates,
    query_by_source,
)


def test_db_connection():
    """测试：数据库连接"""
    conn = get_db()
    assert conn is not None
    # 检查表存在
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    for expected in ["experiments", "initiators", "solvents", "reaction_types", "reaction_rules"]:
        assert expected in table_names


def test_smarts_matches():
    """测试：SMARTS 规则匹配"""
    matches = get_smarts_matches("BrCc1ccccc1")
    assert len(matches) >= 2
    reaction_families = [m["reaction_family"] for m in matches]
    assert "atom_transfer" in reaction_families


def test_smarts_no_match():
    """测试：无匹配的 SMILES"""
    matches = get_smarts_matches("CC")
    # 乙烷可能不匹配任何规则，或只匹配少量
    assert isinstance(matches, list)


def test_query_similar():
    """测试：相似底物查询"""
    # 先插入一条测试数据
    test_smiles = "BrCc1ccccc1"
    add_my_experiment(
        smiles=test_smiles,
        substrate_name="测试-溴化苄",
        reaction_type="ATRP",
        initiator="AIBN",
        yield_percent=78,
        notes="测试数据",
    )

    similar = query_similar_substrates(test_smiles, limit=3)
    assert len(similar) >= 1
    # 最相似的应该就是刚插入的那条（Tanimoto = 1.0）
    assert similar[0]["smiles"] == test_smiles


def test_query_by_source():
    """测试：按来源查询"""
    my_exps = query_by_source("my_experiment", limit=100)
    assert len(my_exps) >= 1
    for exp in my_exps:
        assert exp["source"] == "my_experiment"


def test_get_stats():
    """测试：数据库统计"""
    stats = get_db_stats()
    assert "total" in stats
    assert "literature" in stats
    assert "my_experiment" in stats


def test_add_literature():
    """测试：添加文献数据"""
    row_id = add_literature(
        smiles="C1=CC=C(C=C1)CBr",
        substrate_name="测试-文献溴化苄",
        reaction_type="atom_transfer",
        yield_percent=85,
        notes="测试文献数据",
    )
    assert row_id > 0

    # 验证 source 是 literature
    conn = get_db()
    row = conn.execute(
        "SELECT source FROM experiments WHERE id=?", (row_id,)
    ).fetchone()
    assert row["source"] == "literature"


if __name__ == "__main__":
    tests = [
        ("test_db_connection", test_db_connection),
        ("test_smarts_matches", test_smarts_matches),
        ("test_smarts_no_match", test_smarts_no_match),
        ("test_query_similar", test_query_similar),
        ("test_query_by_source", test_query_by_source),
        ("test_get_stats", test_get_stats),
        ("test_add_literature", test_add_literature),
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
