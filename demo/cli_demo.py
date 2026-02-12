from secure_prompt.core.decision import HybridGuard


if __name__ == "__main__":
    hybrid = HybridGuard()
    while True:
        text = input(">>> ")
        decision = hybrid.decide(text)
        print(decision)
