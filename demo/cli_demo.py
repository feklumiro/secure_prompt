from time import perf_counter


if __name__ == "__main__":
    from secure_prompt.core.decision import DecisionCore
    hybrid = DecisionCore(use_vector=True)
    while True:
        text = input(">>> ")
        a = perf_counter()
        decision = hybrid.decide([text])[0]
        b = perf_counter()
        print(f"Verdict: {decision.verdict}, probability: {decision.probability}, work time: {b-a}")
        a = b
