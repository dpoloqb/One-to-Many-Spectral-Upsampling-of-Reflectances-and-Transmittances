# One-to-Many Spectral Upsampling of Reflectances and Transmittances

Implementation of the algorithm from *L. Belcour, P. Barla, G. Guennebaud — "One-to-Many Spectral Upsampling of Reflectances and Transmittances"* (Eurographics Symposium on Rendering, 2023).

## Problem

**Spectral upsampling** — converting RGB colors to spectral curves — is essential for physically-based spectral rendering. Existing methods provide only a **one-to-one** mapping: one RGB triplet → one single spectrum. This prevents reproducing effects such as:

- **Vathochromism** — color change of a material depending on optical depth (a generalization of the Usambara effect observed in tourmaline gems).
- **Metamerism** — different perceived colors of the same material under different illuminants.

The alternative **Metameric Blacks** approach requires tracking hard constraints for discretized spectra in a high-dimensional space, is not adapted for non-linear effects, and for some target colors the set of valid metamers may be empty.

## Solution

The algorithm builds a **one-to-many** mapping: for a given RGB color, an entire **equivalence class** of spectra is generated. The user can then pick a spectrum with desired properties (e.g., a specific color at greater optical depth).

---

## Algorithm

### 1. Partition of Unity (PU) Construction

The method is based on a set of $K$ basis functions $B_k : U \to \mathbb{R}$, $k \in [0, K-1]$, implemented as non-uniform B-splines of degree 2 over the visible wavelength interval $U = [385\text{nm}, 700\text{nm}]$.

The basis satisfies the **partition of unity** property:

$$\sum_k B_k(x) = 1, \quad \forall x \in U$$

Key consequence: if all weights $w_k \in [0, 1]$, then the reconstructed spectrum $f(x) = \sum_k w_k B_k(x)$ also lies in $[0, 1]$, automatically guaranteeing physical plausibility (energy conservation) for reflectance and transmittance spectra.

### 2. Geometric Interpretation in Chromaticity Space

Each basis function, when integrated with the CIE sensitivity functions $\mathbf{s}(\lambda) = [\bar{x}(\lambda), \bar{y}(\lambda), \bar{z}(\lambda)]^\top$, yields an XYZ color:

$$\mathbf{B}_k = \int B_k(\lambda)\,\mathbf{s}(\lambda)\,d\lambda$$

A spectrum $f(\lambda) = \sum_k w_k B_k(\lambda)$ produces the XYZ color $\mathbf{F} = \sum_k w_k \mathbf{B}_k$.

Converting to chromaticity $\mathbf{c} = [c_x, c_y]^\top$:

$$\mathbf{c} = \sum_k a_k \mathbf{b}_k$$

where $\mathbf{b}_k = \frac{[B_{k,X},\; B_{k,Y}]^\top}{|B_k|}$ are **basis chromaticities** (vertices of a polygon — the *basis gamut* in chromaticity space), and

$$a_k = \frac{w_k |B_k|}{\sum_l w_l |B_l|}$$

are **homogeneous barycentric coordinates**. For $K > 3$, these are generalized barycentric coordinates and are **not unique** — this is what enables the one-to-many mapping.

### 3. Finding the Equivalence Class

#### 3.1. Achieving Target Chromaticity

The target chromaticity $\mathbf{c}$ is expressed via the system:

$$\begin{bmatrix} 1 & 1 & \cdots & 1 \\ b_{0,x} & b_{1,x} & \cdots & b_{K-1,x} \\ b_{0,y} & b_{1,y} & \cdots & b_{K-1,y} \end{bmatrix} \begin{bmatrix} a_0 \\ \vdots \\ a_{K-1} \end{bmatrix} = \begin{bmatrix} 1 \\ c_x \\ c_y \end{bmatrix}$$

**Step 1.** A triangle formed by three basis vertices that contains $\mathbf{c}$ is selected. Standard barycentric coordinates $\mathbf{a}_T = [a_0, a_1, a_2]^\top$ are computed with respect to this triangle; the remaining coordinates $\mathbf{a}_F = [a_3, \dots, a_{K-1}]^\top$ are set to zero.

**Step 2.** The remaining $K - 3$ coordinates $\mathbf{a}_F$ represent **degrees of freedom**. Each element $a_{3+n}$ is randomly sampled in the interval $[0, a_{3+n}^{\max}]$, where the upper bound is computed iteratively:

$$a_{3+n}^{\max} = \min_{i \in \{0,1,2\}} \frac{a_i + H(m_{in}) - 1 - \sum_{l=0}^{n-1} m_{il}\, a_{3+l}}{m_{in}}$$

Here $M = T^{-1}F$ is a $3 \times (K-3)$ matrix, and $H(m)$ is the Heaviside function that accounts for the sign of matrix elements.

For each vector $\mathbf{a}_F$, the offset $\Delta\mathbf{a} = M \mathbf{a}_F$ is computed, and the full generalized barycentric coordinate vector $\mathbf{a} = [(\mathbf{a}_T - \Delta\mathbf{a})^\top, \mathbf{a}_F^\top]^\top$ is guaranteed to reproduce the target chromaticity.

#### 3.2. Achieving Target Luminance

From the barycentric coordinate vector $\mathbf{a}$, the basis coefficients $\mathbf{w}$ are recovered:

$$\mathbf{w}(w_0) = \begin{bmatrix} 1 \\ \frac{a_1 |B_0|}{a_0 |B_1|} \\ \vdots \\ \frac{a_{K-1} |B_0|}{a_0 |B_{K-1}|} \end{bmatrix} w_0 = L\, w_0, \quad w_0 \in \left(0,\; w_0^{\max}\right]$$

where $w_0^{\max} = \min\!\left\{1,\; \frac{a_0|B_1|}{a_1|B_0|},\; \dots\right\}$ ensures $\mathbf{w} \in [0, 1]$.

The value of $w_0$ that achieves the target luminance $F_Y$:

$$w_0^{\star} = \frac{F_Y}{L^\top \mathbf{B}_y}$$

If $w_0^{\star} \leq w_0^{\max}$, the luminance is achieved. Otherwise the spectrum is rescaled:

$$\mathcal{W}(\mathbf{w}) = \frac{\mathbf{w}(w_0^{\max})}{\max\!\left(f^{\max},\; \frac{\mathbf{w}(w_0^{\max})^\top \mathbf{B}_Y}{F_Y}\right)}$$

where $f^{\max} = \max_\lambda f(\lambda)$.

**A priori feasibility check:** a linear programming problem is solved — maximizing $\overline{\mathbf{w}}^\top \mathbf{B}_y$ subject to $0 \leq \overline{\mathbf{w}} \leq 1$ and $A\,\overline{\mathbf{w}} = 0$, where $A = [T\; F]\,\text{diag}(|\mathbf{B}|)^\top - |\mathbf{B}|^\top \begin{bmatrix} 1 \\ \mathbf{c} \end{bmatrix}$.

### 4. Basis Optimization

#### 4.1. B-spline Knot Warping

To expand the *basis gamut* (so that it covers the sRGB gamut), a knot warping function is applied:

$$C_{s,p}(x) = \begin{cases} \frac{x^c}{p^{c-1}} & \text{if } x \in [0, p] \\ 1 - \frac{(1-x)^c}{(1-p)^{c-1}} & \text{otherwise} \end{cases}, \quad c = \frac{2}{1+s} - 1$$

Parameters $(s, p) \in [0,1]^2$ control the strength and position of warping. Warped knots: $\kappa_k = U_0 + C_{s,p}(u_k)(U_1 - U_0)$, where $u_k$ is a uniform sequence on $[0, 1]$.

Boundary knots are offset by 100 nm outside $U$ to ensure smooth spectral decay beyond the visible range.

#### 4.2. Expressivity–Smoothness Trade-off

Two metrics are used to select optimal parameters:

- **Excess area** $\mathcal{A}$ — the signed area between the basis gamut and the sRGB gamut (normalized). Larger values indicate a more expressive basis.
- **Smoothness** $\mathcal{S} = \min_k \text{FWHM}_k$ — the minimum full width at half maximum across all basis functions. Larger values produce smoother spectra.

For a given $K$, all combinations of $(s, p)$ are searched. The one that **maximizes** $\mathcal{A}$ subject to $\mathcal{S} > 20\text{nm}$ is selected.

### 5. Applications

#### Vathochromism

Spectra from the equivalence class are interpreted as unit-depth transmittance $T_1(\lambda)$. The Beer-Lambert-Bouguer law at depth $d$: $T_d(\lambda) = T_1(\lambda)^d$. Different spectra from the class yield different colors as $d$ increases, reproducing the Usambara effect.

#### Metamerism

Basis functions are premultiplied by the illuminant spectrum $I(\lambda)$: $B_k^I(\lambda) = B_k(\lambda) I(\lambda)$. Different illuminants (D65, F2) produce different gamuts, enabling the generation of spectra that appear identical under one illuminant but differ under another.

---

## Usage

```bash
python method.py
```

The program takes the number of basis functions $K$, finds optimal warping parameters $(s, p)$, and for given chromaticity and luminance values reconstructs the equivalence class of spectra.

## Examples

### Successful spectrum reconstruction ($K = 7$)

Target chromaticity $(x = 0.38,\; y = 0.45)$ lies inside the basis gamut. The algorithm successfully generates 218 distinct spectra, all reproducing the same color.

![Successful example](example_success.png)

Top plot — 7 basis functions (Partition of Unity) with warped knots. Bottom left — chromaticity space: the orange polygon (basis gamut) contains the target point (black dot). Bottom right — the equivalence class of spectra.

### Failed spectrum reconstruction ($K = 5$)

The target chromaticity lies outside the basis gamut. No set of weights $w_k \in [0, 1]$ can reproduce this color — the algorithm correctly reports failure.

![Failed example](example_fail.png)

With a small $K$, the basis gamut is too narrow and does not fully cover the sRGB gamut. The solution is to increase $K$ and/or apply knot warping.

## References

- [Paper (arXiv)](https://arxiv.org/abs/2306.11464)
- Belcour L., Barla P., Guennebaud G. *One-to-Many Spectral Upsampling of Reflectances and Transmittances.* Computer Graphics Forum, Volume 42, Number 4, 2023.
