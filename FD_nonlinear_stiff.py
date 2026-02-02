import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ======================================================
# Page configuration
# ======================================================

st.set_page_config(
    page_title="Finite Difference Method: Nonlinear & Stiff BVP",
    layout="wide"
)

# ======================================================
# Title and problem description
# ======================================================

st.title("Finite Difference Method for a Nonlinear Boundary Value Problem")

st.markdown(
    "This application compares backward, central, and forward finite difference "
    "schemes applied to a **nonlinear and potentially stiff boundary value problem**."
)

st.latex(r"""
x'(t)=f(x(t))=-\lambda x(t)+x(t)^{\alpha},
\qquad x(0)=x_0,\; x(T)=x_T
""")

st.info(
    "The power term introduces nonlinearity, while large values of λ "
    "may lead to stiffness. The example highlights qualitative differences "
    "between one–sided and symmetric discretizations."
)

# ======================================================
# Sidebar: parameters
# ======================================================

with st.sidebar:
    st.header("Model parameters")

    lam = st.number_input(
        "λ (stiffness parameter)", value=7.0, step=0.5, format="%.2f"
    )

    alpha = st.number_input(
        "α (degree of nonlinearity)", value=3.0, min_value=2.0, step=0.5, format="%.1f"
    )

    T = st.number_input(
        "Terminal time T", value=1.0, step=0.1, format="%.2f"
    )

    x0 = st.number_input(
        "Boundary value x(0)", value=0.5, step=0.1, format="%.2f"
    )

    xT = st.number_input(
        "Boundary value x(T)", value=0.0, step=0.1, format="%.2f"
    )

    st.markdown("---")
    st.header("Grid resolution")

    N = st.slider(
        "Number of grid points N",
        min_value=50,
        max_value=500,
        value=200,
        step=25
    )

# ======================================================
# Grid
# ======================================================

h = T / N
t = np.linspace(0, T, N + 1)

# ======================================================
# Nonlinear ODE
# ======================================================

def f(x):
    return -lam * x + x**alpha

# ======================================================
# Backward difference (implicit)
# ======================================================

x_back = np.zeros(N + 1)
x_back[0] = x0
x_back[-1] = xT

for i in range(1, N):
    xi = x_back[i - 1]
    for _ in range(15):  # fixed-point iteration
        xi = (x_back[i - 1] + h * xi**alpha) / (1 + h * lam)
    x_back[i] = xi

# ======================================================
# Central difference (symmetric)
# ======================================================

x_cent = np.zeros(N + 1)
x_cent[0] = x0
x_cent[-1] = xT
x_cent[1] = x0 * np.exp(-lam * h)  # consistent seed

for i in range(1, N):
    x_cent[i + 1] = x_cent[i - 1] + 2 * h * f(x_cent[i])

# ======================================================
# Forward difference (explicit)
# ======================================================

x_forw = np.zeros(N + 1)
x_forw[0] = x0

for i in range(N):
    x_forw[i + 1] = x_forw[i] + h * f(x_forw[i])

# ======================================================
# Plot
# ======================================================

st.subheader("Comparison of finite difference schemes")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

axes[0].plot(t, x_back, color="green", lw=2)
axes[0].set_title("Backward difference")
axes[0].grid(True)

axes[1].plot(t, x_cent, color="orange", lw=2)
axes[1].set_title("Central difference")
axes[1].grid(True)

axes[2].plot(t, x_forw, color="red", lw=2)
axes[2].set_title("Forward difference")
axes[2].grid(True)

for ax in axes:
    ax.set_xlabel("t")

axes[0].set_ylabel("x(t)")

plt.tight_layout()
st.pyplot(fig)

# ======================================================
# Interpretation
# ======================================================

st.info(
    "In nonlinear and potentially stiff problems, symmetric finite difference schemes "
    "may generate spurious oscillatory modes even when the numerical solution remains bounded. "
    "Implicit one–sided discretizations better preserve the qualitative behavior "
    "of the continuous system."
)
