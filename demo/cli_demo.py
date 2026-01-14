from secure_prompt.core.decision import decide


if __name__ == "__main__":
    while True:
        text = input(">>> ")
        decision = decide(text)
        print(decision)
