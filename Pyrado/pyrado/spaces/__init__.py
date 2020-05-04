from pyrado import inf
from pyrado.spaces.box import BoxSpace

# Output space for value functions (unbounded scalar)
ValueFunctionSpace = BoxSpace(-inf, inf, labels=['value'])
