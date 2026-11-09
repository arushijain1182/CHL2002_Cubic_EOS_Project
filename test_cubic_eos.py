"""Validation against Smith/Van Ness Ch. 3 worked examples.  Run: python -m pytest -q  (or python test_cubic_eos.py)"""
import math
from cubic_eos import CubicEOS, Species, EOS_MODELS, virial_pitzer_Z


def rel(a, b):
    return abs(a - b) / abs(b)


def test_example_3_9_four_eos():
    """n-butane, 350 K, P = 9.4573 bar: textbook table of V^v and V^l."""
    sp = Species.lookup("n-butane")
    T, P = 350.0, 9.4573
    expected = {  # eos: (Vv, Vl)
        "vdW": (2667, 191.0), "RK": (2555, 133.3), "SRK": (2520, 127.8), "PR": (2486, 112.6)}
    for name, (vv, vl) in expected.items():
        V = CubicEOS(sp, name).volume_roots(T, P)
        assert len(V) == 3
        assert rel(V[-1], vv) < 0.005, (name, V[-1], vv)
        assert rel(V[0], vl) < 0.005, (name, V[0], vl)


def test_example_3_9_RK_intermediate_numbers():
    sp = Species.lookup("n-butane")
    m = CubicEOS(sp, "RK")
    beta, q = m.beta_q(350.0, 9.4573)
    assert abs(beta - 0.026214) < 1e-5
    assert abs(q - 6.6048) < 5e-3
    Z = m.Z_roots(350.0, 9.4573)
    assert abs(Z[-1] - 0.8305) < 5e-4 and abs(Z[0] - 0.04331) < 5e-5


def test_example_3_11_methane_RK_pressure():
    """500 mol CH4 in 0.06 m^3 at 50 C -> V = 120 cm3/mol; RK gives 198.3 bar."""
    sp = Species.lookup("methane")
    m = CubicEOS(sp, "RK")
    assert abs(m.b - 29.85) < 0.05
    assert abs(m.a(323.15) * 1e-12 - 1.793e-6) < 1e-8   # bar m^6/mol^2
    assert abs(m.pressure(323.15, 120.0) - 198.3) < 0.4


def test_saturation_butane_PR():
    sp = Species.lookup("n-butane")
    m = CubicEOS(sp, "PR")
    Ps, Vl, Vv = m.saturation(350.0)
    assert rel(Ps, 9.4573) < 0.03           # exp. 9.4573 bar
    assert rel(Vl, 115.0) < 0.05            # exp. 115.0
    assert rel(Vv, 2482) < 0.03


def test_equal_fugacity_at_saturation():
    sp = Species.lookup("ammonia")
    for e in EOS_MODELS:
        m = CubicEOS(sp, e)
        Ps, Vl, Vv = m.saturation(310.0)
        zl, zv = Vl * Ps / (m.R * 310), Vv * Ps / (m.R * 310)
        assert abs(m.ln_phi(310, Ps, zl) - m.ln_phi(310, Ps, zv)) < 1e-9


def test_phase_selection():
    m = CubicEOS(Species.lookup("n-butane"), "PR")
    assert m.state(350, 5.0).phase == "vapour"
    assert m.state(350, 20.0).phase == "liquid"
    assert m.state(510, 25.0).phase.startswith("gas")


def test_ideal_gas_limit():
    m = CubicEOS(Species.lookup("nitrogen"), "SRK")
    s = m.state(300.0, 1e-3)
    assert abs(s.Z - 1.0) < 1e-5 and abs(s.phi - 1.0) < 1e-5


def test_critical_Z():
    """RK/SRK predict Zc = 1/3 and vdW 3/8 at (Tc, Pc)."""
    sp = Species.lookup("methane")
    for name, zc in (("vdW", 3 / 8), ("RK", 1 / 3), ("SRK", 1 / 3), ("PR", 0.30740)):
        z = CubicEOS(sp, name).Z_roots(sp.Tc, sp.Pc)
        assert abs(max(z) - zc) < 2.5e-2  # near-triple root: ill-conditioned wrt rounded Omega, Psi, (name, z)


def test_inverse_consistency():
    sp = Species.lookup("carbon dioxide")
    m = CubicEOS(sp, "PR")
    T, V = 350.0, 300.0
    P = m.pressure(T, V)
    assert abs(m.temperature(P, V) - T) < 1e-6
    # Z-root route and direct route agree
    s = m.state(T, P)
    assert rel(s.V, V) < 1e-8 or any(rel(v, V) < 1e-8 for v in m.volume_roots(T, P))


def test_butane_510K_25bar_vs_virial():
    """Example 3.10: virial/generalized Z ~0.879 and 3-param ~0.873; cubic EOS should be close."""
    sp = Species.lookup("n-butane")
    assert abs(virial_pitzer_Z(sp, 510, 25) - 0.879) < 0.002
    for e in ("RK", "SRK", "PR"):
        assert abs(CubicEOS(sp, e).state(510, 25).Z - 0.873) < 0.02


def test_numpy_scalar_inputs():
    import numpy as np
    m = CubicEOS(Species.lookup("n-butane"), "PR")
    a = m.saturation(np.float64(350.0))
    b = m.saturation(350.0)
    assert abs(a[0] - b[0]) < 1e-12


# ----------------------------------------------------------------------------
# Enthalpy / entropy
# ----------------------------------------------------------------------------
from cubic_eos import R_J, ig_delta_H, ig_delta_S


def _lnphi_root(m, T, P, which):
    z = m.Z_roots(T, P)
    Z = z[-1] if which == "vap" else z[0]
    return m.ln_phi(T, P, Z)


def test_HR_from_fugacity_derivative():
    """Gibbs-Helmholtz: H^R/RT = -T d(ln phi)/dT at const P.  Checks Eq. 13.75 independently."""
    for name in ("vdW", "RK", "SRK", "PR"):
        m = CubicEOS(Species.lookup("n-butane"), name)
        for T, P, which in ((450.0, 20.0, "vap"), (350.0, 3.0, "vap"), (300.0, 40.0, "liq")):
            h = 1e-3 * T
            dlnphi = (_lnphi_root(m, T + h, P, which) - _lnphi_root(m, T - h, P, which)) / (2 * h)
            HR_num = -R_J * T * T * dlnphi
            Z = m.Z_roots(T, P)[-1 if which == "vap" else 0]
            HR, SR = m.residual(T, P, Z)
            assert abs(HR - HR_num) < 2e-5 * abs(HR) + 1e-3, (name, T, P, HR, HR_num)
            # G^R = H^R - T S^R = RT ln(phi)
            GR = R_J * T * m.ln_phi(T, P, Z)
            assert abs((HR - T * SR) - GR) < 1e-8 * abs(GR) + 1e-6


def test_residuals_vanish_at_low_pressure():
    m = CubicEOS(Species.lookup("methane"), "PR")
    s = m.state(300.0, 1e-4)
    assert abs(s.H_res) < 1e-2 and abs(s.S_res) < 1e-4


def test_clapeyron_consistency():
    """dH_vap = T (Vv - Vl) dPsat/dT must hold exactly for the EOS (J = 0.1 bar cm3)."""
    for name in ("RK", "SRK", "PR"):
        m = CubicEOS(Species.lookup("ammonia"), name)
        T = 310.0
        v = m.vaporization(T)
        h = 0.05
        dPdT = (m.saturation(T + h)[0] - m.saturation(T - h)[0]) / (2 * h)
        dH = T * (v["V_vap"] - v["V_liq"]) * dPdT * 0.1
        assert abs(dH - v["dH_vap"]) < 2e-4 * dH, (name, dH, v["dH_vap"])
        assert abs(v["dS_vap"] - v["dH_vap"] / T) < 1e-6 * v["dS_vap"]


def test_delta_across_phase_change_equals_latent_heat():
    m = CubicEOS(Species.lookup("n-butane"), "PR")
    T = 350.0
    Ps = m.saturation(T)[0]
    d = m.delta(T, Ps, T, Ps, "liquid", "vapor")
    v = m.vaporization(T)
    assert abs(d["dH"] - v["dH_vap"]) < 1e-6 * v["dH_vap"]
    assert abs(d["dS"] - v["dS_vap"]) < 1e-6 * v["dS_vap"]


def test_ideal_gas_cp_integrals_numeric():
    cp = Species.lookup("methane").cp
    A, B, C, D = cp
    f = lambda T: A + B * T + C * T * T + D / T ** 2
    T0, T1, n = 300.0, 700.0, 4000
    h = (T1 - T0) / n
    simpson = lambda g: h / 3 * (g(T0) + g(T1) + sum((4 if i % 2 else 2) * g(T0 + i * h) for i in range(1, n)))
    assert abs(ig_delta_H(cp, T0, T1) - R_J * simpson(f)) < 1e-3
    assert abs(ig_delta_S(cp, T0, T1, 1.0, 1.0) - R_J * simpson(lambda T: f(T) / T)) < 1e-6
    assert abs(ig_delta_S(cp, 300, 300, 1.0, 10.0) + R_J * math.log(10)) < 1e-12


def test_book_data_spot_checks():
    nb = Species.lookup("n-butane")
    assert (nb.Tc, nb.Pc, nb.omega) == (425.1, 37.96, 0.200)
    assert Species.lookup("n-octane").cp[0] == 4.108          # Table C.1
    assert abs(Species.lookup("acetylene").cp[3] + 1.299e5) < 1       # D term
    assert Species.lookup("carbon-dioxide").Tc == 304.2 and Species.lookup("Carbon Dioxide").Tc == 304.2
    assert Species.lookup("helium").Tc == 5.2 and Species.lookup("CO2").Pc == 73.83


def test_reference_state_and_missing_cp():
    m = CubicEOS(Species.lookup("nitrogen"), "SRK")
    s = m.state(298.15, 1e-6)
    assert abs(s.H) < 1e-3
    m2 = CubicEOS(Species.lookup("methyl chloride"), "PR")     # no Table C.1 row
    s2 = m2.state(400.0, 5.0)
    assert s2.H is None and s2.H_res is not None


def test_residuals_vs_lee_kesler_tables_in_book():
    """(HR)/RTc and (SR)/R from Tables D.5-D.12 (omega = 0.012): HR0 + w HR1, SR0 + w SR1.
    Lee/Kesler is a different correlation, so only close (not exact) agreement is expected."""
    lk = [(1.2, 0.2, -0.1490, -0.0859), (1.3, 0.6, -0.4007, -0.2188), (1.5, 1.0, -0.5056, -0.2474),
          (2.0, 1.0, -0.2750, -0.1114), (1.5, 2.0, -1.0459, -0.5220), (3.0, 1.0, -0.1073, -0.0411)]
    sp = Species("methane-like", 190.6, 45.99, 0.012, 16.043)
    for name, tolH, tolS in (("SRK", 0.06, 0.03), ("PR", 0.12, 0.04)):
        m = CubicEOS(sp, name)
        for tr, pr, hr, sr in lk:
            T, P = tr * sp.Tc, pr * sp.Pc
            HR, SR = m.residual(T, P, m.Z_roots(T, P)[-1])
            assert abs(HR / (R_J * sp.Tc) - hr) < tolH, (name, tr, pr, HR / (R_J * sp.Tc), hr)
            assert abs(SR / R_J - sr) < tolS, (name, tr, pr, SR / R_J, sr)


def test_energy_balance_isenthalpic_throttle_sanity():
    """Gas expansion at const H cools methane (Joule-Thomson) - sign check of H(T,P)."""
    m = CubicEOS(Species.lookup("methane"), "PR")
    H1 = m.state(300.0, 100.0).H
    # find T2 at 10 bar with the same enthalpy (bisection)
    lo, hi = 250.0, 300.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if m.state(mid, 10.0).H > H1:
            hi = mid
        else:
            lo = mid
    T2 = 0.5 * (lo + hi)
    assert 240.0 < T2 < 275.0      # JT cooling of tens of K for 100 -> 10 bar


def test_u_w_lecture_notation():
    got = {n: (round(e.u, 9), round(e.w, 9)) for n, e in EOS_MODELS.items()}
    assert got == {"vdW": (0, 0), "RK": (1, 0), "SRK": (1, 0), "PR": (2, -1)}


if __name__ == "__main__":
    import sys
    fails = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            try:
                f(); print("PASS", k)
            except AssertionError as e:
                fails += 1; print("FAIL", k, e)
    sys.exit(fails)
