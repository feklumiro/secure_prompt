from secure_prompt.core.decision import DecisionCore
from time import perf_counter

if __name__ == "__main__":
    a = perf_counter()
    hybrid = DecisionCore()
    print(perf_counter() - a)
    while True:
        text = input(">>> ")
        a = perf_counter()
        decision = hybrid.decide([text])[0]
        print(perf_counter() - a)
        print(decision.verdict, decision.probability, decision.score)
