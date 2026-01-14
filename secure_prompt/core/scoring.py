def score_regex(result) -> int:
    if not result.is_jailbreak:
        return 0

    if result.rule == "system_prompt_extraction":
        return 90

    if result.rule == "override_rules":
        return 80

    if result.rule == "role_manipulation":
        return 70

    if result.rule == "freedom_farming":
        return 70

    return 50
