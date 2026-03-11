from code import add, subtract, multiply, divide

def test_optellen():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_aftrekken():
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5

def test_vermenigvuldigen():
    assert multiply(3, 4) == 12
    assert multiply(-2, 5) == -10

def test_delen():
    assert divide(10, 2) == 5
    assert divide(7, 2) == 3.5

def test_delen_door_nul():
    try:
        divide(5, 0)
        assert False, "Geen fout gegooid"
    except ValueError:
        pass

if __name__ == "__main__":
    test_optellen()
    test_aftrekken()
    test_vermenigvuldigen()
    test_delen()
    test_delen_door_nul()
    print("Alle tests geslaagd!")
