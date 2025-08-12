import sympy as sp
from typing import Any, Callable, Optional

class BilateralEq:
    """Apply Python ops to both sides of an equation without simplifying."""
    def __init__(self, lhs, rhs=0):
        self.lhs = sp.sympify(lhs)
        self.rhs = sp.sympify(rhs)

    # ---- helpers ----
    def _new(self, lhs, rhs):
        return self.__class__(lhs, rhs)

    def equality(self):
        return sp.Eq(self.lhs, self.rhs, evaluate=False)

    def latex(self):
        return sp.latex(self.equality())

    def __repr__(self):
        return f"Eq({self.lhs}, {self.rhs})"

    # ---- add/sub ----
    def __add__(self, other):
        o = sp.sympify(other)
        return self._new(sp.Add(self.lhs, o, evaluate=False),
                         sp.Add(self.rhs, o, evaluate=False))

    def __radd__(self, other):
        o = sp.sympify(other)
        return self._new(sp.Add(o, self.lhs, evaluate=False),
                         sp.Add(o, self.rhs, evaluate=False))

    def __sub__(self, other):
        o = sp.sympify(other)
        return self._new(sp.Add(self.lhs, sp.Mul(-1, o, evaluate=False), evaluate=False),
                         sp.Add(self.rhs, sp.Mul(-1, o, evaluate=False), evaluate=False))

    def __rsub__(self, other):
        o = sp.sympify(other)
        return self._new(sp.Add(o, sp.Mul(-1, self.lhs, evaluate=False), evaluate=False),
                         sp.Add(o, sp.Mul(-1, self.rhs, evaluate=False), evaluate=False))

    # ---- mul/div ----
    def __mul__(self, other):
        o = sp.sympify(other)
        return self._new(sp.Mul(self.lhs, o, evaluate=False),
                         sp.Mul(self.rhs, o, evaluate=False))

    def __rmul__(self, other):
        o = sp.sympify(other)
        return self._new(sp.Mul(o, self.lhs, evaluate=False),
                         sp.Mul(o, self.rhs, evaluate=False))

    def __truediv__(self, other):
        o = sp.sympify(other)
        inv = sp.Pow(o, -1, evaluate=False)
        return self._new(sp.Mul(self.lhs, inv, evaluate=False),
                         sp.Mul(self.rhs, inv, evaluate=False))

    def __rtruediv__(self, other):
        # yields Eq(other/lhs, other/rhs)
        o = sp.sympify(other)
        return self._new(sp.Mul(o, sp.Pow(self.lhs, -1, evaluate=False), evaluate=False),
                         sp.Mul(o, sp.Pow(self.rhs, -1, evaluate=False), evaluate=False))

    # ---- pow, unary ----
    def __pow__(self, p):
        p = sp.sympify(p)
        return self._new(sp.Pow(self.lhs, p, evaluate=False),
                         sp.Pow(self.rhs, p, evaluate=False))

    def __rpow__(self, base):
        base = sp.sympify(base)
        return self._new(sp.Pow(base, self.lhs, evaluate=False),
                         sp.Pow(base, self.rhs, evaluate=False))

    def __neg__(self):
        return self._new(sp.Mul(-1, self.lhs, evaluate=False),
                         sp.Mul(-1, self.rhs, evaluate=False))

    def __pos__(self):
        return self._new(self.lhs, self.rhs)

    # ---- utilities ----
    def subs(self, *args, **kwargs):
        return self._new(self.lhs.subs(*args, **kwargs),
                         self.rhs.subs(*args, **kwargs))

    def mapboth(self, f):
        """Apply a callable to both sides (no simplify)."""
        return self._new(f(self.lhs), f(self.rhs))
    
    def simplify(
        self,
        ratio: Any = None,
        measure: Optional[Callable[[sp.Expr], Any]] = None,
        rational: Optional[bool] = None,
        inverse: bool = False,
        doit: bool = False,
        **kwargs: Any,
    ) -> "BilateralEq":
        """
        Apply sympy.simplify to both sides and return a new BilateralEq.
        Only forwards options that are not None to avoid SymPy min/measure issues.
        """
        # Build kwargs for sympy.simplify, skipping None entries
        simp_kw = dict(kwargs)
        if ratio is not None:    simp_kw["ratio"] = ratio
        if measure is not None:  simp_kw["measure"] = measure  # must be a callable
        if rational is not None: simp_kw["rational"] = rational
        if inverse:              simp_kw["inverse"] = True
        if doit:                 simp_kw["doit"] = True

        def _simp(e: sp.Expr) -> sp.Expr:
            return sp.simplify(e, **simp_kw)

        return self._new(_simp(self.lhs), _simp(self.rhs))
    
    def expand(
        self,
        *,
        deep: Optional[bool] = None,
        modulus: Optional[int] = None,
        power_exp: Optional[bool] = None,
        power_base: Optional[bool] = None,
        mul: Optional[bool] = None,
        log: Optional[bool] = None,
        multinomial: Optional[bool] = None,
        basic: Optional[bool] = None,
        force: Optional[bool] = None,
        func: Optional[bool] = None,
        trig: Optional[bool] = None,
        complex_: Optional[bool] = None,  # maps to 'complex' in SymPy
        frac: Optional[bool] = None,      # for expand_fraction
        radical: Optional[bool] = None,   # for expand_radical
        **kwargs: Any,
    ) -> "BilateralEq":
        """
        Apply sympy.expand to both sides and return a new BilateralEq.

        Only forwards flags that are not None.
        Flags mirror SymPy's expand options (version-dependent).
        """
        exp_kw = dict(kwargs)
        if deep is not None:         exp_kw["deep"] = deep
        if modulus is not None:      exp_kw["modulus"] = modulus
        if power_exp is not None:    exp_kw["power_exp"] = power_exp
        if power_base is not None:   exp_kw["power_base"] = power_base
        if mul is not None:          exp_kw["mul"] = mul
        if log is not None:          exp_kw["log"] = log
        if multinomial is not None:  exp_kw["multinomial"] = multinomial
        if basic is not None:        exp_kw["basic"] = basic
        if force is not None:        exp_kw["force"] = force
        if func is not None:         exp_kw["func"] = func
        if trig is not None:         exp_kw["trig"] = trig
        if complex_ is not None:     exp_kw["complex"] = complex_
        if frac is not None:         exp_kw["frac"] = frac
        if radical is not None:      exp_kw["radical"] = radical

        return self._new(
            sp.expand(self.lhs, **exp_kw),
            sp.expand(self.rhs, **exp_kw),
        )


if __name__ =='__main__':
    x, a, b, c = sp.symbols('x a b c', nonzero=True)
    eq0 = BilateralEq(a*x**2 + b*x + c, 0)
    print(eq0)
    eq1 = eq0 / a    
    print(eq1)             # divide both sides by a
    eq2 = eq1 - c/a   
    print(eq2)            # subtract c/a from both sides
    eq3 = eq2 + (b/(2*a))**2      # add (b/2a)^2 to both sides
    print(eq3.equality())         # raw SymPy Eq (no simplification)
    print(eq3.latex())            # LaTeX string

    # If/when you want a simplified snapshot:
    eq3_simpl = eq3.expand().simplify()
    print(eq3_simpl.latex())
