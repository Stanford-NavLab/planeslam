import autograd.numpy as np
import pymanopt

from pymanopt.manifolds import SpecialOrthogonalGroup

manifold = SpecialOrthogonalGroup(n=3)

a = np.array([1,2,3])[:,None]
b = np.array([1,2,-3])[:,None]

@pymanopt.function.autograd(manifold)
def cost(R):
    return np.linalg.norm(R @ a - b)**2

problem = pymanopt.Problem(manifold, cost)

optimizer = pymanopt.optimizers.SteepestDescent()
result = optimizer.run(problem)