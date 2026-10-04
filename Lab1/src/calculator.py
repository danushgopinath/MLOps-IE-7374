def fun1(x, y):
    """
    Adds two numbers together.
    Args:
        x (int/float): First number.
        y (int/float): Second number.
    Returns:
        int/float: Sum of x and y.
        Raises:
        ValueError: If x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    
    return x + y

def fun2(x, y):
    """
    Subtracts two numbers.
    Args:
        x (int/float): First number.
        y (int/float): Second number.
    Returns:
        int/float: Difference of x and y.
        Raises:
        ValueError: If x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return x - y

def fun3(x, y):
    """
    Multiplies two numbers together.
    Args:
        x (int/float): First number.
        y (int/float): Second number.
    Returns:
        int/float: Product of x and y.
        Raises:
        ValueError: If either x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return x * y

def fun4(x,y,z):
    """
    Adds three numbers together.
    Args:
        x (int/float): First number.
        y (int/float): Second number.
        z (int/float): Third number.
    Returns:
        int/float: Sum of x, y and z.
        Raises:
        ValueError: If x, y or z is not a number.
    """
    if not all(isinstance(n, (int, float)) for n in (x, y, z)):
        raise ValueError("All inputs must be numbers.")
    total_sum = x + y + z
    return total_sum

def fun5(x, y):
    """
    Divides x by y.
    Args:
        x (int/float): Numerator.
        y (int/float): Denominator.
    Returns:
        float: Quotient of x and y.
        Raises:
        ValueError: If x or y is not a number, or if y is zero.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y

def fun6(x, y):
    """
    Raises x to the power of y.
    Args:
        x (int/float): Base.
        y (int/float): Exponent.
    Returns:
        int/float: x raised to the power of y.
        Raises:
        ValueError: If x or y is not a number, or if x is zero and y is negative.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    if x == 0 and y < 0:
        raise ValueError("Zero cannot be raised to a negative power.")
    return x ** y

def fun7(numbers):
    """
    Calculates the average of a list of numbers.
    Args:
        numbers (list): List of int/float values.
    Returns:
        float: Average of the numbers.
        Raises:
        ValueError: If the list is empty or contains a non-number.
    """
    if not numbers:
        raise ValueError("List must not be empty.")
    if not all(isinstance(n, (int, float)) for n in numbers):
        raise ValueError("All list items must be numbers.")
    return sum(numbers) / len(numbers)


# f1_op = fun1(2,3)
# f2_op = fun2(2,3)
# f3_op = fun3(2,3)
# f4_op = fun4(f1_op,f2_op,f3_op)

