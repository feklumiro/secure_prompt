def score_regex(result) -> int:
    if not result.is_jailbreak:
        return 0
    score = 0
    for matched in result.rules:
        if matched[0] == "system_prompt_extraction":
            score += 90
        elif matched[0] == "override_rules":
            score += 80
        elif matched[0] == "role_manipulation":
            score += 70
        elif matched[0] == "freedom_farming":
            score += 70
        else:
            score += 50
    return score


def score_ml(result) -> int:
    if result.score < 1:
        return -int(1 / result.score * 8)
    else:
        return int(result.score * 15)
