def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Delen door nul is niet mogelijk")
    return a / b
# werkt het??#
#kk ding werkt eindelijk vm
#testt

def main():
    print("=== Rekenmachine ===")
    print("Operaties: + - * /")
    print("Typ 'stop' om af te sluiten\n")

    while True:
        invoer = input("Berekening (bijv. 5 + 3): ").strip()
        if invoer.lower() == "stop":
            break

        for op in ["+", "-", "*", "/"]:
            if op in invoer:
                delen = invoer.split(op)
                if len(delen) == 2:
                    try:
                        a = float(delen[0].strip())
                        b = float(delen[1].strip())
                        if op == "+":
                            print(f"= {add(a, b)}")
                        elif op == "-":
                            print(f"= {subtract(a, b)}")
                        elif op == "*":
                            print(f"= {multiply(a, b)}")
                        elif op == "/":
                            print(f"= {divide(a, b)}")
                    except ValueError as e:
                        print(f"Fout: {e}")
                    break
        else:
            print("Ongeldige invoer")

if __name__ == "__main__":
    main()
