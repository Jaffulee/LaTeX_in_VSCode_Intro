import sympy as sp
from modules.sympy_eq_operator import BilateralEq

# symbols (assume a ≠ 0)
a = sp.symbols('a', real=True, nonzero=True)
b, c, x = sp.symbols('b c x', real=True)

def show(tag, eq: BilateralEq, also_simplify=True):
    print(f"{tag}: $$ {eq.latex()} $$\n")
    if also_simplify:
        print(f"{tag} (simplified): $$ {eq.simplify().latex()} $$\n")

# Start: a x^2 + b x + c = 0
eq0 = BilateralEq(a*x**2 + b*x + c, 0)
show("Start", eq0)

# 1) Divide by a
eq1 = (eq0 / a)
show("Divide by a", eq1)

# 2) Move constant to RHS (subtract c/a)
eq2 = (eq1 - c/a)
show("Move constant to RHS", eq2)

# 3) Complete the square: add (b/2a)^2 to both sides
comp = (b/(2*a))**2
eq3 = eq2 + comp
show(r"Add $(\frac{b}{2a})^2$ to both sides", eq3)

# 4) Recognize the perfect square on LHS, keep RHS as-is (unsimplified)
eq4 = BilateralEq((x + b/(2*a))**2, eq3.rhs)
show("Perfect square form (unsimplified RHS)", eq4)

# Optionally, simplify only the RHS to show the standard discriminant form
eq4s = BilateralEq(eq4.lhs, sp.simplify(eq4.rhs))
show("Perfect square form (simplified RHS)", eq4s)

# 5) Take square roots → two branches (±). Build them explicitly.
D = b**2 - 4*a*c
eq5_plus  = BilateralEq(x + b/(2*a),  sp.sqrt(D)/(2*a))
eq5_minus = BilateralEq(x + b/(2*a), -sp.sqrt(D)/(2*a))
show("Square root (plus branch)",  eq5_plus)
show("Square root (minus branch)", eq5_minus)

# 6) Solve for x by subtracting b/(2a) from both sides
eq6_plus  = (eq5_plus  - b/(2*a)).simplify()
eq6_minus = (eq5_minus - b/(2*a)).simplify()
show("Solve for x (plus)",  eq6_plus)
show("Solve for x (minus)", eq6_minus)

# Final ± formula (pretty print as a single LaTeX line)
quad_latex = r"x = \frac{-\,b \pm \sqrt{b^{2}-4ac}}{2a}"
print(f"Quadratic formula: $$ {quad_latex} $$")
