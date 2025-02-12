#!/usr/bin/env python
# coding: utf-8

# In[1]:


# toc
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import odeint

plt.style.use('../maroon_py.mplstyle')

# If there was not a limitation on the source, then the best transformer modulus would be $T=492,104\,\frac{1}{m^3}$, resulting in a speed of $73.8\,rad/s$. However, this occurs at a flow rate of $Q_{max}=0.00015$, which is not within the operating conditions of the source. If you wanted to see the results without this limitation, then you can uncomment the line `Q_max_power = Q_max  # m^3/s` in the code cell 5.
# 
# You should also recognize that in order to get that dot to move to the left, you need to keep increasing the flow rate $Q$. Looking at the graph, you can see that the output shaft speed decreases with increasing modulus, so we can definitely say that this is the maximum speed.
# 
# # Problem 2
# ## Given
# 
# ![Figure 3](fig3.png)
# 
# For the figure shown above assume the armature has a moment-speed curve of $M_a(\omega_1)=-315\omega_1+2000$ when the motor is on and acts as a damper when the motor is off with $M_a(\omega_1)=-14.4\omega_1$. In both cases the armature speed is in $rad/s$ and the moment is in $lbf\cdot in$.
# 
# Assume that all shafts are made of steel ($G=1.15\cdot10^7\,psi$) and that the diameter of the bevel gears is 2 inches. Model each shaft as a rotational spring $\left(k=\frac{\pi r^4G}{2L}\right)$. Model the resistance felt by each of the paddles (inertias 7 and 8 shown at the bottom of the industrial mixer) with rotational dampers of a coefficient of $B=2880\,lbf\cdot in\cdot s$. The number of teeth of each gear (20T, 12T, and 36T) is given in the figure.
# 
# ## Find
# Obtain the following:
# 
# a. A detailed free body diagram including your choice of reference coordinates.
# b. A complete set of differential equations in state variable form. Notice that there are 8 inertias and 4 springs, but not all inertias are independent.
# c. Solve the state variable equations using a 4th order Runge-Kutta method for a time period of 20 seconds. Assume that all initial states are zero and that the motor armature is on for the first 10 seconds and off for the last 10 seconds. It is recommended that a very small time step is used. Try $\Delta t=0.0005s$.
# d. Verify using energy conservation and include a discussion on the results with the following points:
#     - When the motor is turned on and at steady-state ($\omega_1$ is constant), does it operate at its maximum power?
#     - Also, consider the mechanical failure modes that might occur with the plotted speed results.
# 
# ## Solution
# ### Part A
# 
# ![Figure 4](fig4.png)
# 
# The free body diagram drawn above has no combined mass, resulting in the lengthiest method to solve this problem.
# 
# ### Part B

# In[17]:


# Defining symbols
I0, I1, I2, I3, I4, I5, I6, I7 = sp.symbols('I0 I1 I2 I3 I4 I5 I6 I7')
k1, k2 = sp.symbols('k1 k2')
B = sp.Symbol('B')
F1, F2, F3 = sp.symbols('F1 F2 F3')
r1, r2, r3, r4, r5 = sp.symbols('r1 r2 r3 r4 r5')

# Defining functions
t = sp.Symbol('t')
th0, th1, th2, th3, th4, th5, th6, th7 = [sp.Function(fr'\theta_{i}')(t) for i in range(0, 8)]
# Ma = sp.Function('M_a')(th0.diff())
Ma = sp.Symbol('M_a')

# Defining the equations of motion
eq1 = sp.Eq(I0*th0.diff(t, 2), k1*(th1 - th0) + Ma)
eq2 = sp.Eq(I1*th1.diff(t, 2), k1*(th0 - th1) - F1*r1)
eq3 = sp.Eq(I2*th2.diff(t, 2), k1*(th3 - th2) + F1*r2)
eq4 = sp.Eq(I3*th3.diff(t, 2), k1*(th2 - th3) - F2*r3 - F3*r3)
eq5 = sp.Eq(I4*th4.diff(t, 2), k2*(th6 - th4) + F2*r4)
eq6 = sp.Eq(I5*th5.diff(t, 2), k2*(th7 - th5) + F3*r5)
eq7 = sp.Eq(I6*th6.diff(t, 2), k2*(th4 - th6) - B*th6.diff())
eq8 = sp.Eq(I7*th7.diff(t, 2), k2*(th5 - th7) - B*th7.diff())
eqs = [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]
print(*eqs)

# We essentially have 11 unknowns (the $\theta$'s and reaction forces $F_1$, $F_2$, and $F_3$) and 8 equations. We can use the velocity ratios to add the additionally 3 equations.

# In[18]:


eq9 = sp.Eq(th2.diff(t, 2), th1.diff(t, 2)*r1/r2)
eq10 = sp.Eq(th4.diff(t, 2), th3.diff(t, 2)*r3/r4)
eq11 = sp.Eq(th5.diff(t, 2), th3.diff(t, 2)*r3/r5)
eqs = eqs + [eq9, eq10, eq11]
print(eq9, eq10, eq11)

# In[19]:


# Solving the equations
eqs = eqs + [
    sp.Eq(r1, r2),
    sp.Eq(r3, sp.Rational(1, 3)*r4),
    sp.Eq(r3, sp.Rational(1, 3)*r5)
]
sol = sp.solve(
    eqs,
    (th0.diff(t, 2), th1.diff(t, 2), th2.diff(t, 2), th3.diff(t, 2), th4.diff(t, 2), th5.diff(t, 2), th6.diff(t, 2),
     th7.diff(t, 2), F1, F2, F3, r1, r2, r3, r4, r5),
    dict=True
)[0]

subs = [
    (th1, th2),
    (th4, sp.Rational(1, 3)*th3),
    (th5, sp.Rational(1, 3)*th3)
]

for key, value in sol.items():
    if key not in [F1, F2, F3, r1, r2, r3, r4, r5]:
        print(sp.Eq(key, value.subs(subs).simplify()))

# In the above equations, we see that there are a couple of redundancies. Since we coupled $\theta_4$ and $\theta_5$ with $\theta_3$, we can ignore the equations for $\theta_4$ and $\theta_5$. Similarly, we can only use $\theta_2$ since I decided to represent $\theta_1$ in terms of $\theta_2$. Notice that the equations for $\theta_6$ and $\theta_7$ can be proven to be the same, but if there was a different gear ratio, then these equations would be different, so I will choose to solve these equations independently for the possibility of considering a different ratio. This results in the following state variable equations:

# In[20]:


# Grab only the equations we want
eq1 = sp.Eq(th0.diff(t, 2), sol[th0.diff(t, 2)].subs(subs).simplify())
eq2 = sp.Eq(th2.diff(t, 2), sol[th2.diff(t, 2)].subs(subs).simplify())
eq3 = sp.Eq(th3.diff(t, 2), sol[th3.diff(t, 2)].subs(subs).simplify())
eq4 = sp.Eq(th6.diff(t, 2), sol[th6.diff(t, 2)].subs(subs).simplify())
eq5 = sp.Eq(th7.diff(t, 2), sol[th7.diff(t, 2)].subs(subs).simplify())
eq_motion = [eq1, eq2, eq3, eq4, eq5]

# Define new state variables
th8, th9, th10, th11, th12 = [sp.Function(fr'\theta_{i}')(t) for i in range(8, 13)]

eq6 = sp.Eq(th0.diff(), th8)
eq7 = sp.Eq(th2.diff(), th9)
eq8 = sp.Eq(th3.diff(), th10)
eq9 = sp.Eq(th6.diff(), th11)
eq10 = sp.Eq(th7.diff(), th12)
state_eqs = [eq6, eq7, eq8, eq9, eq10]

# Make the substitutions of the state variables
state_subs = [
    (eq.lhs, eq.rhs) for eq in state_eqs
]
eq_motion = [eq.subs(state_subs) for eq in eq_motion]

state_sol = sp.solve(
    eq_motion + state_eqs,
    (th0.diff(), th2.diff(), th3.diff(), th6.diff(), th7.diff(), th8.diff(), th9.diff(), th10.diff(), th11.diff(),
     th12.diff()),
    dict=True
)[0]

# sympy is giving me an undesired order in the solution. I'll force it to print this order
sol_order = [th0.diff(), th2.diff(), th3.diff(), th6.diff(), th7.diff(), th8.diff(), th9.diff(), th10.diff(),
             th11.diff(), th12.diff()]

final_eq = []
for key in sol_order:
    eq = sp.Eq(key, state_sol[key].simplify())
    final_eq.append(eq)
    print(eq)

# You can easily convert this to the matrix form with `sympy`'s `linear_eq_to_matrix` function.

# In[21]:


# Converting to matrix form
x = [th0, th2, th3, th6, th7, th8, th9, th10, th11, th12]
rhs_values = [eq.rhs for eq in final_eq]
A, b = sp.linear_eq_to_matrix(rhs_values, x)
mat_eq = sp.Eq(sp.Matrix(sol_order), sp.Add(sp.MatMul(A, sp.Matrix(x)), -b))
mat_eq

# In[22]:


mat_eq.doit()

# ### Part C
# I will now bring this to the numerical world with a python function. There is a way to automate this and actually use the expressions given by `sympy`, but for clarity, I will not do this.

# In[23]:


# Defining constants
# Interias in lbf*s^2*in
I0 = 0.2
I1, I2 = 0.05, 0.05
I3 = 0.1
I4, I5 = 0.5, 0.5
I6, I7 = 0.8, 0.8

# k1 and k2 lengths and diameters in inches
L1, D1 = 36, 0.75
L2, D2 = 40, 0.75

G = 1.15e7  # psi
k1 = (np.pi*(D1/2)**4*G)/(2*L1)  # lbf*in
k2 = (np.pi*(D2/2)**4*G)/(2*L2)  # lbf*in

B = 2880  # lbf*in*s


# Defining the input function
def Ma(w0, t_):
    return -315*w0 + 2000 if t_ < 10 else -14.4*w0


# Defining the state variable function
def state_vars(thetas, t_):
    x0, x2, x3, x6, x7, x8, x9, x10, x11, x12 = thetas
    return [
        x8,
        x9,
        x10,
        x11,
        x12,
        (Ma(x8, t_) - k1*x0 + k1*x2)/I0,
        k1*(x0 - 2*x2 + x3)/(I1 + I2),
        (9*k1*x2 - 9*k1*x3 - 2*k2*x3 + 3*k2*x6 + 3*k2*x7)/(9*I3 + I4 + I5),
        (k2/3*x3 - k2*x6 - B*x11)/I6,
        (k2/3*x3 - k2*x7 - B*x12)/I7
    ]


t_array = np.linspace(0, 20, 40_001)
sol = odeint(state_vars, [0]*10, t_array)

# for th, w in [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]:
for th, w in [(0, 5)]:
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    theta_label = sp.latex(state_eqs[th].lhs.args[0])
    omega_label = sp.latex(state_eqs[th].lhs)

    ax.plot(t_array, sol[:, th], label=f'${theta_label}$')
    ax2.plot(t_array, sol[:, w], label=f'${omega_label}$', color='black')

    ax2.grid(False)
    ax.legend()
    ax2.legend(loc='lower right')

    ax.set_xlabel('$t$ $(s)$')
    ax.set_ylabel(f'${theta_label}$ $(rad)$')
    ax2.set_ylabel(f'${omega_label}$ $(rad/s)$')

plt.show()
