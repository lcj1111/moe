# 作用：验证选择题和数值题答案抽取与评分。
from clients.quality_eval import normalize_number, parse_choice, parse_number, score


def test_parse_choice_prefers_explicit_final():
    assert parse_choice("Reason A and B. FINAL: I") == "I"
    assert parse_choice("C") == "C"


def test_numeric_parsing_and_scoring():
    assert parse_number("work... FINAL: $1,234.50") == normalize_number("1234.50")
    row = {"score_type": "numeric", "answer": "18"}
    assert score(row, "FINAL: 18")[0] is True


def test_deferred_code_is_not_locally_executed():
    assert score({"score_type": "deferred_code"}, "arbitrary code") == (None, None)
