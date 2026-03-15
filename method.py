import numpy as np
import csv
import os
from scipy.interpolate import BSpline
from scipy.optimize import linprog
from itertools import combinations
import matplotlib.pyplot as plt


def load_cmf_from_csv(filepath):
    wavelengths = []
    x_vals = []
    y_vals = []
    z_vals = []

    with open(filepath, 'r') as f:
        sample = f.read(2048)
        f.seek(0)

        if '\t' in sample:
            delimiter = '\t'
        elif ';' in sample:
            delimiter = ';'
        else:
            delimiter = ','

        reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                w = float(row[0])
            except ValueError:
                continue

            wavelengths.append(w)
            x_vals.append(float(row[1]))
            y_vals.append(float(row[2]))
            z_vals.append(float(row[3]))

    if len(wavelengths) == 0:
        raise ValueError(f"Failed to read from {filepath}")

    return (np.array(wavelengths),
            np.array(x_vals),
            np.array(y_vals),
            np.array(z_vals))


class SpectralUpsampler:

    def __init__(self, cmf_file, K=7, degree=2, warp_s=0.66, warp_p=0.39,
                 lambda_min=385.0, lambda_max=700.0, n_wavelengths=256,
                 boundary_offset=100.0):
        self.K = K
        self.degree = degree
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.wavelengths = np.linspace(lambda_min, lambda_max, n_wavelengths)

        self.knots = self._build_knots(warp_s, warp_p, boundary_offset)
        self.basis = self._evaluate_basis()

        cmf_wl, cmf_x, cmf_y, cmf_z = load_cmf_from_csv(cmf_file)
        self.cmf = np.zeros((3, n_wavelengths))
        self.cmf[0] = np.interp(self.wavelengths, cmf_wl, cmf_x)
        self.cmf[1] = np.interp(self.wavelengths, cmf_wl, cmf_y)
        self.cmf[2] = np.interp(self.wavelengths, cmf_wl, cmf_z)
        self.cmf /= np.trapezoid(self.cmf[1], self.wavelengths)
        
        self.B_XYZ = np.zeros((K, 3))
        for k in range(K):
            for c in range(3):
                self.B_XYZ[k, c] = np.trapezoid(
                    self.basis[k] * self.cmf[c], self.wavelengths)

        self.B_sum = np.sum(self.B_XYZ, axis=1)
        self.b_chrom = self.B_XYZ[:, :2] / self.B_sum[:, np.newaxis]

    def _warp(self, x, s, p):
        if s <= 1e-12:
            return x.copy()
        c = 2.0 / (1.0 + s) - 1.0
        result = np.empty_like(x, dtype=float)
        for i in range(len(x)):
            xi = x[i]
            if xi <= p:
                result[i] = xi**c / p**(c-1) if p > 1e-15 else 0.0
            else:
                result[i] = 1.0 - (1.0-xi)**c / (1.0-p)**(c-1) \
                    if (1.0-p) > 1e-15 else 1.0
        return result

    def _build_knots(self, s, p, offset):
        K, d = self.K, self.degree
        n_unique = K - d + 1
        u = np.linspace(0, 1, n_unique)
        warped = self._warp(u, s, p)
        kappa = self.lambda_min + warped * (self.lambda_max - self.lambda_min)
        kappa[0] = self.lambda_min - offset
        kappa[-1] = self.lambda_max + offset
        knot_vector = np.concatenate([
            np.full(d+1, kappa[0]),
            kappa[1:-1],
            np.full(d+1, kappa[-1])
        ])
        return knot_vector

    def _evaluate_basis(self):
        basis = np.zeros((self.K, len(self.wavelengths)))
        for k in range(self.K):
            coeffs = np.zeros(self.K)
            coeffs[k] = 1.0
            spl = BSpline(self.knots, coeffs, self.degree, extrapolate=False)
            vals = spl(self.wavelengths)
            basis[k] = np.nan_to_num(vals, nan=0.0)
        return basis

    def _point_in_triangle(self, p, a, b, c):
        v0, v1, v2 = b - a, c - a, p - a
        d00 = v0 @ v0
        d01 = v0 @ v1
        d11 = v1 @ v1
        d20 = v2 @ v0
        d21 = v2 @ v1
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-15:
            return None
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        bary = np.array([u, v, w])
        if np.all(bary >= -1e-10):
            return np.maximum(bary, 0.0)
        return None

    def find_enclosing_triangle(self, cx, cy):
        target = np.array([cx, cy])
        for tri in combinations(range(self.K), 3):
            i, j, k = tri
            bary = self._point_in_triangle(
                target, self.b_chrom[i], self.b_chrom[j], self.b_chrom[k])
            if bary is not None:
                return list(tri), bary
        return None, None

    def sample_equivalence_class(self, cx, cy, FY, n_samples=100, seed=None):
        rng = np.random.RandomState(seed)

        tri_idx, a_T = self.find_enclosing_triangle(cx, cy)
        if tri_idx is None:
            print(f"ERROR: color ({cx},{cy}) is outside basis gamut!")
            return []

        free_idx = [i for i in range(self.K) if i not in tri_idx]
        order = tri_idx + free_idx

        b_ord = self.b_chrom[order]
        B_sum_ord = self.B_sum[order]
        B_Y_ord = self.B_XYZ[order, 1]
        basis_ord = self.basis[order]

        T = np.array([
            [1.0, 1.0, 1.0],
            [b_ord[0,0], b_ord[1,0], b_ord[2,0]],
            [b_ord[0,1], b_ord[1,1], b_ord[2,1]]
        ])

        n_free = self.K - 3
        F = np.ones((3, n_free))
        F[1, :] = b_ord[3:, 0]
        F[2, :] = b_ord[3:, 1]

        M = np.linalg.solve(T, F)
        results = []

        for _ in range(n_samples):
            a_F = np.zeros(n_free)

            for n in range(n_free):
                bounds_list = []
                for i in range(3):
                    m_in = M[i, n]
                    if abs(m_in) < 1e-15:
                        continue
                    contrib = sum(M[i, l] * a_F[l] for l in range(n))
                    H = 1.0 if m_in > 0 else 0.0
                    bound = (a_T[i] + H - 1.0 - contrib) / m_in
                    bounds_list.append(bound)

                if not bounds_list:
                    continue
                a_max = min(min(bounds_list), 1.0)
                if a_max > 1e-15:
                    a_F[n] = rng.uniform(0, a_max)

            delta_a = M @ a_F
            a_full = np.zeros(self.K)
            a_full[:3] = a_T - delta_a
            a_full[3:] = a_F

            if np.any(a_full < -1e-10):
                continue
            a_full = np.maximum(a_full, 0.0)

            pivot = np.argmax(a_full * (B_sum_ord > 1e-15))
            if a_full[pivot] < 1e-15:
                continue

            L = np.zeros(self.K)
            for k in range(self.K):
                if a_full[k] > 1e-15 and B_sum_ord[k] > 1e-15:
                    L[k] = (a_full[k] * B_sum_ord[pivot]) / \
                           (a_full[pivot] * B_sum_ord[k])

            w_max = 1.0
            for k in range(self.K):
                if L[k] > 1e-15:
                    w_max = min(w_max, 1.0 / L[k])

            LtBy = L @ B_Y_ord
            if LtBy < 1e-15:
                continue

            w_star = FY / LtBy
            achieves = w_star <= w_max + 1e-10
            w_pivot = min(w_star, w_max) if achieves else w_max
            w_ordered = L * w_pivot
            spectrum = basis_ord.T @ w_ordered
            spectrum = np.clip(spectrum, 0, 1)
            if not achieves:
                f_max = np.max(spectrum)
                achieved_FY = w_ordered @ B_Y_ord
                if f_max > 0 and achieved_FY > 0:
                    scale = min(1.0 / f_max, FY / achieved_FY)
                    if scale > 1.0:
                        spectrum_scaled = spectrum * scale
                        if np.max(spectrum_scaled) <= 1.0 + 1e-10:
                            spectrum = np.clip(spectrum_scaled, 0, 1)
                            w_ordered = w_ordered * scale
                            achieves = abs(w_ordered @ B_Y_ord - FY) / max(FY, 1e-10) < 0.01
            w_orig = np.zeros(self.K)
            for new_k in range(self.K):
                w_orig[order[new_k]] = w_ordered[new_k]

            results.append({
                'weights': w_orig,
                'spectrum': spectrum,
                'FY_achieved': w_ordered @ B_Y_ord,
                'achieves_target': achieves,
            })

        return results

    def spectrum_to_xy(self, spectrum):
        XYZ = np.array([np.trapezoid(spectrum * self.cmf[c], self.wavelengths)
                        for c in range(3)])
        s = np.sum(XYZ)
        return XYZ[:2] / s if s > 1e-15 else np.array([1/3, 1/3])


def polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i, 0] * points[j, 1]
        area -= points[j, 0] * points[i, 1]
    return abs(area) / 2.0


def optimize_warp_params(cmf_file, K, degree=2, lambda_min=385.0, lambda_max=700.0,
                         n_wavelengths=256, min_fwhm=20.0, n_grid=50):
    wavelengths = np.linspace(lambda_min, lambda_max, n_wavelengths)
    srgb_xy = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])

    best_s, best_p = 0.0, 0.5
    best_area = -np.inf

    for si in range(n_grid):
        for pi in range(n_grid):
            s = si / (n_grid - 1)
            p = 0.1 + 0.8 * pi / (n_grid - 1)

            try:
                up = SpectralUpsampler(
                    cmf_file=cmf_file, K=K, degree=degree,
                    warp_s=s, warp_p=p)
            except Exception:
                continue

            min_fwhm_val = np.inf
            for k in range(K):
                bk = up.basis[k]
                peak = np.max(bk)
                if peak < 1e-10:
                    continue
                half = peak / 2.0
                above = wavelengths[bk >= half]
                if len(above) >= 2:
                    fwhm = above[-1] - above[0]
                    min_fwhm_val = min(min_fwhm_val, fwhm)

            if min_fwhm_val < min_fwhm:
                continue

            area_score = 0.0
            for v in srgb_xy:
                tri, bary = up.find_enclosing_triangle(v[0], v[1])
                if tri is not None:
                    area_score += 1.0

            area_score += polygon_area(up.b_chrom)

            if area_score > best_area:
                best_area = area_score
                best_s, best_p = s, p

    return best_s, best_p


if __name__ == "__main__":

    CMF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "xyz_matching_fun.csv")

    def plot_horseshoe(ax):
        from scipy.interpolate import CubicSpline

        locus_x = np.array([
            0.1741, 0.1740, 0.1738, 0.1736, 0.1733, 0.1730, 0.1726, 0.1721,
            0.1714, 0.1703, 0.1689, 0.1669, 0.1644, 0.1611, 0.1566, 0.1510,
            0.1440, 0.1355, 0.1241, 0.1096, 0.0913, 0.0687, 0.0454, 0.0235,
            0.0082, 0.0039, 0.0139, 0.0389, 0.0743, 0.1142, 0.1547, 0.1929,
            0.2296, 0.2658, 0.3016, 0.3373, 0.3731, 0.4087, 0.4441, 0.4788,
            0.5125, 0.5448, 0.5752, 0.6029, 0.6270, 0.6482, 0.6658, 0.6801,
            0.6915, 0.7006, 0.7079, 0.7140, 0.7190, 0.7230, 0.7260, 0.7283,
            0.7300, 0.7311, 0.7320, 0.7327, 0.7334, 0.7340, 0.7344, 0.7346,
            0.7347
        ])
        locus_y = np.array([
            0.0050, 0.0050, 0.0049, 0.0049, 0.0048, 0.0048, 0.0048, 0.0048,
            0.0051, 0.0058, 0.0069, 0.0086, 0.0109, 0.0138, 0.0177, 0.0227,
            0.0297, 0.0399, 0.0578, 0.0868, 0.1327, 0.2007, 0.2950, 0.4127,
            0.5384, 0.6548, 0.7502, 0.8120, 0.8338, 0.8262, 0.8059, 0.7816,
            0.7543, 0.7243, 0.6923, 0.6589, 0.6245, 0.5896, 0.5547, 0.5202,
            0.4866, 0.4544, 0.4242, 0.3965, 0.3725, 0.3514, 0.3340, 0.3197,
            0.3083, 0.2993, 0.2920, 0.2859, 0.2809, 0.2770, 0.2740, 0.2717,
            0.2700, 0.2689, 0.2680, 0.2673, 0.2666, 0.2660, 0.2656, 0.2654,
            0.2653
        ])

        t = np.zeros(len(locus_x))
        for i in range(1, len(t)):
            dx = locus_x[i] - locus_x[i-1]
            dy = locus_y[i] - locus_y[i-1]
            t[i] = t[i-1] + np.sqrt(dx*dx + dy*dy)

        cs_x = CubicSpline(t, locus_x)
        cs_y = CubicSpline(t, locus_y)

        t_fine = np.linspace(t[0], t[-1], 2000)
        smooth_x = cs_x(t_fine)
        smooth_y = cs_y(t_fine)

        ax.plot(smooth_x, smooth_y, '-', color='black', linewidth=1, label='Spectral locus')
        ax.plot([smooth_x[-1], smooth_x[0]], [smooth_y[-1], smooth_y[0]],
                '--', color='black', linewidth=1)

    K = int(input("Enter the number of basis functions K (Recommended 4-11): "))
    if K < 4:
        print("K must be >= 4")
        exit()

    print(f"Optimizing (s, p) for K={K}...")
    s_opt, p_opt = optimize_warp_params(cmf_file=CMF_FILE, K=K)
    print(f"Found: s = {s_opt:.2f}, p = {p_opt:.2f}")

    print("Creating upsampler...")
    up = SpectralUpsampler(
        cmf_file=CMF_FILE,
        K=K,
        warp_s=s_opt,
        warp_p=p_opt
    )

    print(f"\nBasis: {up.K} functions")
    print("Gamut vertices (xy):")
    for k in range(up.K):
        print(f"  b_{k} = ({up.b_chrom[k,0]:.4f}, {up.b_chrom[k,1]:.4f})")

    cx, cy, FY = 0.38, 0.45, 0.46
    print(f"\nTarget color: x = {cx}, y = {cy}, Y = {FY}")

    tri = up.find_enclosing_triangle(cx, cy)
    in_gamut = tri[0] is not None

    if in_gamut:
        print(f"Enclosing triangle: bases {tri[0]}")
        print("\nGenerating spectra...")
        results = up.sample_equivalence_class(cx, cy, FY, n_samples=500, seed=42)
        good = [r for r in results if r['achieves_target']]
        print(f"Total: {len(results)}, achieved target: {len(good)}")

        if len(good) < 2:
            print("Too few spectra, using all")
            good = results[:20]

        print("\nChromaticity check:")
        for i, r in enumerate(good[:5]):
            xy = up.spectrum_to_xy(r['spectrum'])
            print(f"  #{i}: x={xy[0]:.4f}, y={xy[1]:.4f}, Y={r['FY_achieved']:.4f}")
    else:
        print("COLOR OUTSIDE BASIS GAMUT!")
        good = []
        results = []

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2], hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    for k in range(up.K):
        ax1.plot(up.wavelengths, up.basis[k], label=f'B_{k}')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('B_k(λ)')
    ax1.set_title('Basis functions (Partition of Unity)')
    ax1.legend(fontsize=10, ncol=up.K, loc='upper right')
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1, 0])
    plot_horseshoe(ax2)
    ax2.plot(up.b_chrom[:,0], up.b_chrom[:,1], 'o-', color='orange', label='Basis gamut')
    srgb = np.array([[0.64,0.33],[0.30,0.60],[0.15,0.06],[0.64,0.33]])
    ax2.plot(srgb[:,0], srgb[:,1], 'b-', label='sRGB')
    if in_gamut:
        ax2.plot(cx, cy, 'ko', markersize=6, label='Target')
    else:
        ax2.plot(cx, cy, 'ro', markersize=6, label='Target (outside)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Chromaticity space')
    ax2.legend(fontsize=8)
    ax2.set_aspect('equal')
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 1])
    if good:
        for r in good[:15]:
            ax3.plot(up.wavelengths, r['spectrum'], alpha=0.5, color='steelblue')
        max_val = max(r['spectrum'].max() for r in good[:15])
        ax3.set_ylim(0, max_val * 1.05)
        ax3.set_title(f'{len(good)} spectra, same chromaticity (x={cx}, y={cy})')
    else:
        ax3.text(0.5, 0.5, 'No spectra found\nColor outside basis gamut',
                 transform=ax3.transAxes,
                 ha='center', va='center', fontsize=14, color='red')
        ax3.set_title('Spectra — not found')
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Reflectance / Transmittance')
    ax3.grid(True)

    plt.savefig('result.png', dpi=150)
    print("\nSaved: result.png")
    plt.show()
