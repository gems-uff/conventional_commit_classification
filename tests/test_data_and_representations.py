from src.representations import build_representation


def test_representations():
    row = {"processed_message": "hello", "diff": "diff --git ...", "git_show": "commit abc\nhello"}
    assert build_representation(row, "message_only") == "hello"
    assert build_representation(row, "diff_only") == "diff --git ..."
    assert build_representation(row, "message_and_diff") == "hello\n\ndiff --git ..."
    assert build_representation(row, "git_show") == "commit abc\nhello"
