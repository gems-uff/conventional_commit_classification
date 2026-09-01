from src.parsing.strict import INVALID_OUTPUT, PARSER_VERSION, parse_strict, parse_hqcm
assert PARSER_VERSION == "4.1.1"
labels11 = {"fix","feat","build","ci","docs","style","refactor","perf","test","chore","others"}
assert parse_strict("fix", task_labels=labels11).prediction == "fix"
assert parse_strict("others", task_labels=labels11).prediction == "others"
assert parse_strict("randomlabel", task_labels=labels11).prediction == INVALID_OUTPUT
assert parse_hqcm("cicd", task_labels={"fix","feat","build","ci","docs","style","refactor","test"}).prediction == "ci"
assert parse_strict("others", task_labels={"fix","feat"}).prediction == INVALID_OUTPUT
print("Parser policy tests OK.")
