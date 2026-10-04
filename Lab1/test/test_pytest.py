import pytest
from src import calculator

def test_fun1():
    assert calculator.fun1(2, 3) == 5
    assert calculator.fun1(5,0) == 5
    assert calculator.fun1 (-1, 1) == 0
    assert calculator.fun1 (-1, -1) == -2


def test_fun2():
    assert calculator.fun2(2, 3) == -1
    assert calculator.fun2(5,0) == 5
    assert calculator.fun2 (-1, 1) == -2
    assert calculator.fun2 (-1, -1) == 0

def test_fun3():
    assert calculator.fun3(2, 3) == 6
    assert calculator.fun3(5,0) == 0
    assert calculator.fun3 (-1, 1) == -1
    
    assert calculator.fun3 (-1, -1) == 1

def test_fun4():
    assert calculator.fun4(2, 3, 5) == 10
    assert calculator.fun4(5,0, -1) == 4
    assert calculator.fun4 (-1, -1, -1) == -3
    
    assert calculator.fun4 (-1, -1, 100) == 98

def test_fun5():
    assert calculator.fun5(6, 3) == 2
    assert calculator.fun5(5, 2) == 2.5
    with pytest.raises(ValueError):
        calculator.fun5(5, 0)

def test_fun6():
    assert calculator.fun6(2, 3) == 8
    assert calculator.fun6(5, 0) == 1
    assert calculator.fun6(2, -1) == 0.5
    with pytest.raises(ValueError):
        calculator.fun6(0, -1)

def test_fun7():
    assert calculator.fun7([1, 2, 3]) == 2
    assert calculator.fun7([5]) == 5
    assert calculator.fun7([-1, 1, 2.5]) == pytest.approx(0.8333, rel=1e-3)
    with pytest.raises(ValueError):
        calculator.fun7([])
    with pytest.raises(ValueError):
        calculator.fun7([1, "2", 3])

@pytest.mark.parametrize("x, y, expected", [
    (10, 2, 5),
    (-9, 3, -3),
    (7, 2, 3.5),
    (0, 4, 0),
])
def test_fun5_parametrized(x, y, expected):
    assert calculator.fun5(x, y) == expected

@pytest.mark.parametrize("func, args", [
    (calculator.fun1, ("2", 3)),
    (calculator.fun2, (2, None)),
    (calculator.fun3, ([1], 3)),
    (calculator.fun4, (1, 2, "3")),
    (calculator.fun5, ("6", 3)),
    (calculator.fun6, (2, "3")),
])
def test_invalid_inputs(func, args):
    with pytest.raises(ValueError):
        func(*args)
    
