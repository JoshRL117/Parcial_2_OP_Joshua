import numpy as np
from scipy.optimize import minimize

# Define the Himmelblau function and its gradient
def himmelblau(p):
    x, y = p
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def gradient_himmelblau(p):
    x, y = p
    df_dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    df_dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.array([df_dx, df_dy])

# Evaluate the gradient at the given point
specific_x = -6.18887472e+13
specific_y = -3.98088300e+12
grad_at_point = gradient_himmelblau([specific_x, specific_y])

print(f"Gradient at ({specific_x}, {specific_y}): {grad_at_point}")

# Use numerical optimization to find a local minimum near the given point
result = minimize(himmelblau, [specific_x, specific_y], method='BFGS', jac=gradient_himmelblau)

print("Local minimum found by numerical optimization:")
print(result)

# Check if the gradient is close to zero
if np.allclose(grad_at_point, [0, 0], atol=1e-5):
    print(f"The point ({specific_x}, {specific_y}) is a critical point.")
else:
    print(f"The point ({specific_x}, {specific_y}) is not a critical point.")
