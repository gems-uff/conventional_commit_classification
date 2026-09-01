from src.consensus import resolve_first_to_two
from src.parsing.strict import INVALID_OUTPUT
labels = {"fix", "feat", "build", "ci", "docs", "style", "refactor", "perf", "test", "chore", "others"}
base = [10,20,30]; extra = [40,50,60,70,80]
r = resolve_first_to_two({10:"fix",20:"fix",30:"feat"}, initial_seeds=base, adaptive_seeds=extra, task_labels=labels)
assert r.complete and r.classification == "fix" and r.decisive_seed == 30
r = resolve_first_to_two({10:"fix",20:"feat",30:"docs"}, initial_seeds=base, adaptive_seeds=extra, task_labels=labels)
assert not r.complete and r.missing_seed == 40
r = resolve_first_to_two({10:INVALID_OUTPUT,20:INVALID_OUTPUT,30:"fix"}, initial_seeds=base, adaptive_seeds=extra, task_labels=labels)
assert r.complete and r.classification == INVALID_OUTPUT
print("Consensus tests OK.")
