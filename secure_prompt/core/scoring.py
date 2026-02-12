PATTERN_WEIGHTS = {
    "override": 1.0,
    "system": 0.9,
    "roleplay": 0.8,
    "freedom": 0.7
}
BUNCH_WEIGHTS = {
    "main": 3.0,
    "reverse": 0.8,
}
ML_JAIL_SCORE = 1.157
VECTOR_JAIL_SCORE = 3.736
PIPELINE_POLICY = (0, 1, 40, 23, 17, 1, 40, 23)

def score_regex(result) -> int:
    if not result.is_jailbreak:
        return 0
    score = 0
    for rule in result.rules:
        if rule[0] == "MULTIPLE_DANGEROUS_WORDS":
            score += 5 * len(rule[1].split(","))
        else:
            c1, c2 = rule[0].split("_")
            score += (PATTERN_WEIGHTS[c1] * 8) + (BUNCH_WEIGHTS.get(c2, 1) * 15)
    return int(score)


def score_ml(result) -> int:
    if result.score < ML_JAIL_SCORE:
        return -int(1 / result.score)
    else:
        return int(result.score * 35)


def score_vector(result) -> int:
    return int((result.score - VECTOR_JAIL_SCORE) * 20)
