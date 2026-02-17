from secure_prompt.core.decision import DecisionCore


if __name__ == "__main__":
    hybrid = DecisionCore()
    while True:
        text = input(">>> ")
        decision = hybrid.decide([text])[0]
        print(decision.verdict, decision.score)
