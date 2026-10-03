---
template: algorithm.html
---

# Algorithm

## Standard form

PDHCG solves the convex quadratic conic program

$$
\begin{aligned}
\min_x\quad &\tfrac12x^\top(Q+R^\top D R)x+c^\top x\\
\text{s.t.}\quad &\ell_c\le Ax\le u_c,\\
&Fx+g\in\mathcal K_a,\\
&\ell_v\le x\le u_v,\\
&x_J\in\mathcal K_v.
\end{aligned}
$$

The last condition applies to each variable-cone block $J$. Here $Q$ is the
sparse quadratic component, and $R^\top D R$ is an optional structured low-rank
component. The full Hessian $H=Q+R^\top D R$ is positive semidefinite; $D$
defaults to the identity when omitted.

The vectors $\ell_c,u_c$ and $\ell_v,u_v$ specify constraint and variable bounds,
which may be infinite. The cones $\mathcal K_a$ and $\mathcal K_v$ may contain
SOC, rotated SOC, exponential, power, and positive-semidefinite blocks.

## Primal–dual form

Define the objective and the variable and row-constraint sets:

$$
\begin{aligned}
f(x)&=\tfrac12x^\top Hx+c^\top x,\\
\mathcal X&=\{x:\ell_v\le x\le u_v,\;
 x_J\in\mathcal K_v\},\\
\mathcal B&=[\ell_c,u_c].
\end{aligned}
$$

Let $y$ be the dual variable for $Ax\in\mathcal B$, and let
$z\in\mathcal K_a^*$ be the dual variable for $Fx+g\in\mathcal K_a$.
The support function of $\mathcal B$ is

$$
p_{\mathcal B}(s)=\sup_{w\in\mathcal B}\langle s,w\rangle,
$$

and the dual cone is
$\mathcal K_a^*=\{z:\langle z,w\rangle\ge0\text{ for all }w\in\mathcal K_a\}$.
The problem has the saddle-point form

$$
\min_{x\in\mathcal X}\;
\max_{y\in\mathbb R^m,\;z\in\mathcal K_a^*}\;
\mathcal L(x,y,z),
$$

where

$$
\begin{aligned}
\mathcal L(x,y,z)
&=f(x)-\langle y,Ax\rangle\\
&\quad-\langle z,Fx+g\rangle-p_{\mathcal B}(-y).
\end{aligned}
$$

Maximization over $y$ enforces the row bounds, and maximization over $z$
enforces the affine cone constraints. Variable bounds and variable cones
remain in the primal domain $\mathcal X$.

## Primal–dual updates

The base PDHG iteration updates $x$ first, then updates $y$ and $z$ using the
extrapolated primal point. Choose $\tau,\sigma>0$ satisfying

$$
\tau\sigma\left\|\begin{bmatrix}A\\F\end{bmatrix}\right\|_2^2<1.
$$

### Primal update

The dual variables shift the input to the quadratic proximal subproblem:

$$
\begin{aligned}
v^k&=x^k+\tau(A^\top y^k+F^\top z^k),\\
x^{k+1}&\approx\underset{x\in\mathcal X}{\arg\min}\;
 \left\{f(x)+\frac{\|x-v^k\|^2}{2\tau}\right\}.
\end{aligned}
$$

This subproblem retains the quadratic objective and the variable constraints.
Its Hessian $H+\tau^{-1}I$ is positive definite, so the exact proximal point
is unique.

#### M-norm formulation

Completing the square gives

$$
\begin{aligned}
M&=I+\tau H,\\
r^k&=M^{-1}(v^k-\tau c).
\end{aligned}
$$

The exact primal point is the weighted projection $\Pi_{\mathcal X}^M(r^k)$,
where

$$
\begin{gathered}
\Pi_{\mathcal X}^M(r)
=\underset{x\in\mathcal X}{\arg\min}\;\tfrac12\|x-r\|_M^2,\\
\|w\|_M^2=w^\top Mw.
\end{gathered}
$$

Thus the quadratic curvature enters the projection through $M$. This is the
weighted-projection viewpoint of Section 4.1 of the
[Conic QP paper](https://arxiv.org/html/2608.09159v1). The representation does
not require forming $M^{-1}$: its evaluation depends on the structure of $H$.

#### Diagonal quadratic objectives

If the full Hessian is $H=\operatorname{diag}(h)$, then

$$
M_{ii}=1+\tau h_i,\qquad
r_i^k=\frac{v_i^k-\tau c_i}{1+\tau h_i}.
$$

For $R=0$, this is the diagonal-$Q$ case. On box-constrained blocks, the
weighted projection reduces to clipping:

$$
x_i^{k+1}=\min\{u_{v,i},\max\{\ell_{v,i},r_i^k\}\}.
$$

On variable-cone blocks, use the corresponding diagonal-metric cone projection
$\Pi_{\mathcal K_v}^{M_J}(r_J^k)$. The metric is essential for coupled cone
coordinates; Appendix C of the paper gives the weighted projection formulas.
The implementation uses these specialized projections where supported;
PSD variable blocks currently use the inner solver.

#### Inner iterations

For a general sparse or low-rank $H$, the default `inner` mode approximates the
same proximal point using projected iterations. Starting from $u^0=x^k$, a
basic inner step is

$$
\begin{aligned}
g^j&=Hu^j+c+\tau^{-1}(u^j-v^k),\\
u^{j+1}&=\Pi_{\mathcal X}(u^j-\alpha_jg^j).
\end{aligned}
$$

PDHCG uses safeguarded Barzilai–Borwein step lengths and stops when the inner
accuracy target is reached or the inner iteration limit is met. The final
inner iterate supplies $x^{k+1}$. Products are evaluated as
$Hu=Qu+R^\top(D(Ru))$, preserving the sparse and low-rank representation.
See [inner solver parameters](python/parameters.md#inner-solver-parameters)
for accuracy and iteration controls.

#### Linearized update

The `linearized` mode changes the proximal metric. Choose a curvature bound
$L\geq\lambda_{\max}(H)$ and add the term

$$
\begin{gathered}
\tfrac12\|x-x^k\|_S^2,\\
S=LI-H\succeq0
\end{gathered}
$$

to the original primal subproblem. The resulting projection metric is

$$
M_S=I+\tau(H+S)=(1+\tau L)I.
$$

The quadratic terms combine into a scalar Hessian, so completing the square
now gives one Euclidean projection:

$$
\begin{aligned}
\alpha&=(L+\tau^{-1})^{-1},\\
d^k&=Hx^k+c-A^\top y^k-F^\top z^k,\\
x^{k+1}&=\Pi_{\mathcal X}(x^k-\alpha d^k).
\end{aligned}
$$

This update solves the modified proximal subproblem defined by the chosen
metric. The implementation uses a safeguarded spectral estimate for $L$.
[`NonDiagonalQuadraticMode`](python/parameters.md#algorithm-parameters)
selects `inner` or `linearized` for non-diagonal quadratic objectives.

After the primal update, form the extrapolated point for both dual updates:

$$
\bar x^{k+1}=2x^{k+1}-x^k.
$$

### Dual updates

For the bounded linear rows, let $s^{k+1}=A\bar x^{k+1}$. Then

$$
\begin{aligned}
y^{k+1}&=y^k-\sigma s^{k+1}\\
&\quad+\sigma\Pi_{\mathcal B}
 \!\left(s^{k+1}-y^k/\sigma\right).
\end{aligned}
$$

Here $\Pi_{\mathcal B}$ clips each coordinate to $[\ell_c,u_c]$.
For the affine cone constraints,

$$
z^{k+1}=\Pi_{\mathcal K_a^*}
 \!\left(z^k-\sigma(F\bar x^{k+1}+g)\right).
$$

The cone projection can also be evaluated using the primal cone:

$$
\Pi_{\mathcal K_a^*}(w)=w+\Pi_{\mathcal K_a}(-w).
$$

Thus each base iteration combines a quadratic proximal solve, products with
$A,A^\top,F,F^\top$, and projections onto the row bounds and cone blocks.

## References

1. Hongpei Li, Yicheng Huang, Huikang Liu, Dongdong Ge, and Yinyu Ye.
    [*GPU-Accelerated Conic Quadratic Programming with Local Linear Convergence under Strict Complementarity*](https://arxiv.org/abs/2608.09159), 2026.

2. Yicheng Huang, Wanyu Zhang, Hongpei Li, Dongdong Ge, Huikang Liu, and Yinyu Ye.
    [*A Restarted Primal-Dual Hybrid Conjugate Gradient Method for Large-Scale Quadratic Programming*](https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0983),
    *INFORMS Journal on Computing*, 2025.

For the recommended PDHCG citation and copyable BibTeX, see [Citation](citation.md).
