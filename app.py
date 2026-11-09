"""
Cubic EOS Calculator - web interface (Streamlit)

Run locally:   streamlit run app.py
Needs:         cubic_eos.py in the same folder (all calculations and the substance
               database from Tables B.1 / C.1 of the textbook live there).

Flow: 1) choose the substance  2) choose the equation of state
      3) choose what is known (T,P | T,V | P,V) and enter the values  ->  results + graph.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import cubic_eos as ce
from cubic_eos import CubicEOS, Species, EOS_MODELS, R_GAS

st.set_page_config(page_title="Cubic EOS Calculator", page_icon="🧪", layout="wide")

EOS_LABELS = {
    "vdW": "van der Waals (vdW)",
    "RK": "Redlich/Kwong (RK)",
    "SRK": "Soave/Redlich/Kwong (SRK)",
    "PR": "Peng/Robinson (PR)",
}
COMMON = ["methane", "ethane", "propane", "n-butane", "carbon dioxide", "nitrogen", "oxygen",
          "hydrogen", "ammonia", "water", "ethylene", "benzene", "methanol"]
CUSTOM = "Custom fluid (enter Tc, Pc, ω)"
T_UNITS = {"K": "K", "°C": "C", "°F": "F"}
P_UNITS = ["bar", "kPa", "MPa", "atm", "psi"]
V_UNITS = ["cm3/mol", "L/mol", "m3/kmol", "m3/mol"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def pretty(name: str) -> str:
    return name.title().replace("N-", "n-").replace("Iso", "iso")


def species_options():
    allnames = ce.list_species()
    rest = [n for n in allnames if n not in COMMON]
    return [n for n in COMMON if n in allnames] + rest


def phase_label(m, T, P, Z, roots):
    s = m.state(T, P)
    if abs(Z - s.Z) < 1e-6 * max(1.0, Z):
        return s.phase
    if len(roots) == 3 and abs(Z - roots[1]) < 1e-9 * max(1.0, Z):
        return "unstable (inside the EOS loop)"
    return ("liquid-like" if abs(Z - roots[0]) < 1e-12 else "vapour-like") + " (metastable)"


def solve_case(m, mode, T, P, V):
    """Return a dict with all results. Internal units: K, bar, cm3/mol."""
    sp = m.sp
    notes = []
    if mode == "TP":
        s = m.state(T, P)
        Z, roots, phase, V = s.Z, s.Z_roots, s.phase, s.V
        notes += s.notes
    else:
        if mode == "TV":
            P = m.pressure(T, V)
        else:
            T = m.temperature(P, V)
        if P <= 0:
            raise ValueError("The EOS gives zero or negative pressure at this state "
                             "(too cold / too dense for a liquid under this EOS).")
        roots = m.Z_roots(T, P)
        Z0 = P * V / (R_GAS * T)
        Z = min(roots, key=lambda z: abs(z - Z0))
        phase = phase_label(m, T, P, Z, roots)
    Psat = Vl = Vv = None
    if T < sp.Tc:
        Psat, Vl, Vv = m.saturation(T)
        if mode != "TP" and Vl < V < Vv:
            notes.append(f"This (T, V) lies inside the two-phase region. A real fluid there is a "
                         f"liquid-vapour mixture at P = Psat = {Psat:.4g} bar; the pressure from the "
                         f"EOS is a non-physical value on the van der Waals loop.")
    HR, SR = m.residual(T, P, Z)
    H, S = m.total_HS(T, P, Z)
    return dict(T=T, P=P, V=V, Z=Z, phase=phase, roots=roots, HR=HR, SR=SR, H=H, S=S,
                phi=float(np.exp(m.ln_phi(T, P, Z))), Psat=Psat, Vl=Vl, Vv=Vv, notes=notes,
                rho=1000.0 / V, rho_m=(sp.M / V * 1000.0 if sp.M else None))


@st.cache_data(show_spinner=False)
def dome(Tc, Pc, omega, eos):
    m = CubicEOS(Species("x", Tc, Pc, omega), eos)
    Vl, Vv, Ps = [], [], []
    for Tr in np.linspace(0.5, 0.9995, 70):
        try:
            p, vl, vv = m.saturation(float(Tr) * Tc)
            Ps.append(p); Vl.append(vl); Vv.append(vv)
        except Exception:
            pass
    return np.array(Vl), np.array(Vv), np.array(Ps)


def z_curve(m, T, Pgrid):
    Ps = m.saturation(T)[0] if T < m.sp.Tc else None
    out = []
    for P in Pgrid:
        z = m.Z_roots(T, float(P))
        out.append(z[0] if (Ps is not None and P > Ps) else z[-1])
    return np.array(out)


def make_figure(m, r):
    sp, T, P, V, Z = m.sp, r["T"], r["P"], r["V"], r["Z"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    # ---- P-V diagram
    vmax = max(60 * R_GAS * sp.Tc / sp.Pc, 3 * V)
    Vg, Pg = m.isotherm(T, 1.02 * m.b, vmax, 500)
    a1.plot(Vg, Pg, color="#1f77b4", lw=2, label=f"{m.eos.name} isotherm, T = {T:.1f} K")
    a1.plot(Vg, R_GAS * T / Vg, "k--", lw=1.2, label="ideal gas")
    if T < sp.Tc:
        Vl, Vv, Ps = dome(sp.Tc, sp.Pc, sp.omega, m.eos.name)
        a1.plot(Vl, Ps, color="gray", lw=1.5); a1.plot(Vv, Ps, color="gray", lw=1.5, label="saturation dome")
    a1.plot([V], [P], "o", color="#d62728", ms=9, zorder=5, label="your state")
    a1.set_xscale("log"); a1.set_xlim(1.0 * m.b, vmax)
    a1.set_ylim(0, max(2.5 * sp.Pc, 1.5 * P))
    a1.set_xlabel("V  [cm³/mol]"); a1.set_ylabel("P  [bar]")
    a1.set_title(f"{pretty(sp.name)}: P–V isotherm"); a1.grid(alpha=0.3, which="both"); a1.legend(fontsize=8)
    # ---- Z-P diagram
    Pmax = max(1.6 * P, 0.5 * sp.Pc)
    Pgr = np.linspace(Pmax / 200, Pmax, 200)
    a2.plot(Pgr, z_curve(m, T, Pgr), color="#1f77b4", lw=2, label=f"{m.eos.name}")
    a2.axhline(1.0, color="k", ls="--", lw=1.2, label="ideal gas (Z = 1)")
    a2.plot([P], [Z], "o", color="#d62728", ms=9, zorder=5, label="your state")
    a2.set_xlabel("P  [bar]"); a2.set_ylabel("Z = PV/RT")
    a2.set_title(f"Compressibility factor at T = {T:.1f} K"); a2.grid(alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# page header / theory
# ----------------------------------------------------------------------------
st.title("Cubic Equation-of-State Calculator")
st.caption("PVT behaviour and enthalpy / entropy of real fluids using the van der Waals, Redlich/Kwong, "
           "Soave/Redlich/Kwong and Peng/Robinson equations.  Substance data: Tables B.1 and C.1 of "
           "Smith, Van Ness, Abbott & Swihart.")

# ----------------------------------------------------------------------------
# sidebar: inputs
# ----------------------------------------------------------------------------
sb = st.sidebar
sb.header("1 · Substance")
choice = sb.selectbox("Choose the substance", species_options() + [CUSTOM],
                      format_func=lambda n: n if n == CUSTOM else pretty(n))
custom_cp = None
if choice == CUSTOM:
    cname = sb.text_input("Name", "my fluid")
    c1, c2 = sb.columns(2)
    Tc_in = c1.number_input("Tc [K]", 1.0, 5000.0, 400.0)
    Pc_in = c2.number_input("Pc [bar]", 0.1, 5000.0, 50.0)
    om_in = sb.number_input("Acentric factor ω", -0.5, 2.0, 0.2, format="%.3f")
    sp0 = Species(cname, Tc_in, Pc_in, om_in, None, None)
else:
    sp0 = Species.lookup(choice)
if sp0.cp is None:
    with sb.expander("Optional: ideal-gas Cp constants (for total H, S)"):
        st.caption("Cp/R = A + B·T + C·T² + D/T²  (T in K).  Without these only the residual "
                   "H^R and S^R are shown.")
        cA, cB = st.columns(2); cC, cD = st.columns(2)
        A_ = cA.number_input("A", value=0.0, format="%.4f"); B_ = cB.number_input("B", value=0.0, format="%.3e")
        C_ = cC.number_input("C", value=0.0, format="%.3e"); D_ = cD.number_input("D", value=0.0, format="%.3e")
        if A_ != 0.0:
            sp0 = Species(sp0.name, sp0.Tc, sp0.Pc, sp0.omega, sp0.M, (A_, B_, C_, D_))
sp = sp0
sb.caption(f"Tc = {sp.Tc} K · Pc = {sp.Pc} bar · ω = {sp.omega}" + (f" · M = {sp.M} g/mol" if sp.M else ""))

sb.header("2 · Equation of state")
eos_name = sb.selectbox("Choose the equation", list(EOS_MODELS), index=3, format_func=lambda k: EOS_LABELS[k])
compare_all = sb.checkbox("Also compare all four equations", value=True)

sb.header("3 · What do you know?")
mode_label = sb.radio("Known quantities", ["T and P  →  find V, Z, H, S", "T and V  →  find P",
                                           "P and V  →  find T"], label_visibility="collapsed")
mode = {"T and P": "TP", "T and V": "TV", "P and V": "PV"}[mode_label.split("  →")[0]]

with sb.form("inputs"):
    st.subheader("4 · Enter values")
    T_val = P_val = V_val = None
    if mode in ("TP", "TV"):
        c1, c2 = st.columns([2, 1])
        T_val = c1.number_input("Temperature", value=350.0, format="%.4f")
        T_unit = T_UNITS[c2.selectbox("unit ", list(T_UNITS), key="tu")]
    if mode in ("TP", "PV"):
        c1, c2 = st.columns([2, 1])
        P_val = c1.number_input("Pressure", value=10.0, min_value=1e-9, format="%.5g")
        P_unit = c2.selectbox("unit  ", P_UNITS, key="pu")
    if mode in ("TV", "PV"):
        c1, c2 = st.columns([2, 1])
        V_val = c1.number_input("Molar volume", value=500.0, min_value=1e-9, format="%.5g")
        V_unit = c2.selectbox("unit   ", V_UNITS, key="vu")
    with st.expander("Reference state for H and S"):
        T0 = st.number_input("T₀ [K]", value=298.15)
        P0 = st.number_input("P₀ [bar]", value=1.0)
        st.caption("H = S = 0 for the ideal gas at (T₀, P₀). Only differences between states matter.")
    go = st.form_submit_button("Calculate", type="primary", width="stretch")

# ----------------------------------------------------------------------------
# main page
# ----------------------------------------------------------------------------
with st.expander("About the equations of state", expanded=not go):
    st.markdown("A cubic EOS relates **P, V and T** of a real fluid. All four equations taught in class are "
                "special cases of one general form, written in two equivalent ways:")
    st.markdown("**As discussed in class** (u, w form):")
    st.latex(r"P=\frac{RT}{V-b}-\frac{a(T)}{V^{2}+u\,b\,V+w\,b^{2}}")
    st.markdown("**As in the textbook** (σ, ε form, Eq. 3.41):")
    st.latex(r"P=\frac{RT}{V-b}-\frac{a(T)}{(V+\epsilon b)(V+\sigma b)}\qquad u=\sigma+\epsilon,\; w=\sigma\epsilon")
    st.latex(r"b=\Omega\frac{RT_c}{P_c},\qquad a(T)=\Psi\,\alpha(T_r,\omega)\,\frac{R^{2}T_c^{2}}{P_c}")
    st.table([
        {"EOS": "van der Waals", "u": 0, "w": 0, "σ": "0", "ε": "0", "Ω": "1/8", "Ψ": "27/64", "a(T)": "constant"},
        {"EOS": "Redlich/Kwong", "u": 1, "w": 0, "σ": "1", "ε": "0", "Ω": "0.08664", "Ψ": "0.42748",
         "a(T)": "function of T  (α = Tr^-1/2)"},
        {"EOS": "Soave/Redlich/Kwong", "u": 1, "w": 0, "σ": "1", "ε": "0", "Ω": "0.08664", "Ψ": "0.42748",
         "a(T)": "function of T and ω"},
        {"EOS": "Peng/Robinson", "u": 2, "w": -1, "σ": "1+√2", "ε": "1−√2", "Ω": "0.07780", "Ψ": "0.45724",
         "a(T)": "function of T and ω"},
    ])
    st.markdown("Solving these by hand means finding the roots of a cubic and, below the critical temperature, "
                "iterating on phase equilibrium. This tool does it for you. The enthalpy and entropy follow from "
                "the **residual properties** H^R and S^R (textbook Eqs. 13.75–13.76) plus the ideal-gas change.")

if not go:
    st.info("👈 Choose a substance, an equation, what you know, and enter the values in the sidebar, then press **Calculate**.")
    st.stop()

# ---- convert inputs to internal units
try:
    T = ce.T_to_K(T_val, T_unit) if T_val is not None else None
    P = ce.P_to_bar(P_val, P_unit) if P_val is not None else None
    V = ce.V_to_cm3(V_val, V_unit) if V_val is not None else None
    if T is not None and T <= 0:
        raise ValueError("Temperature must be above absolute zero.")
    m = CubicEOS(sp, eos_name)
    m.T_ref, m.P_ref = T0, P0
    if V is not None and V <= m.b:
        raise ValueError(f"Molar volume must exceed the EOS co-volume b = {m.b:.2f} cm³/mol.")
    r = solve_case(m, mode, T, P, V)
except (ValueError, RuntimeError) as e:
    st.error(str(e))
    st.stop()

# output units follow the input units where given
Tu = next((k for k, v in T_UNITS.items() if v == T_unit), "K") if T_val is not None else "K"
Pu = P_unit if P_val is not None else st.session_state.get("pu", "bar")
Vu = V_unit if V_val is not None else "cm3/mol"
fT = lambda x: f"{ce.T_from_K(x, T_UNITS[Tu]):.6g} {Tu}"
fP = lambda x: f"{ce.P_from_bar(x, Pu):.6g} {Pu}"
fV = lambda x: f"{ce.V_from_cm3(x, Vu):.6g} {Vu}"

st.subheader(f"Results — {pretty(sp.name)} · {EOS_LABELS[eos_name]}")
if mode == "TP":
    st.success(f"**V = {fV(r['V'])}**    ·    Z = {r['Z']:.5f}    ·    phase: {r['phase']}")
elif mode == "TV":
    st.success(f"**P = {fP(r['P'])}**    ·    Z = {r['Z']:.5f}    ·    phase: {r['phase']}")
else:
    alt = f"  ({r['T']:.3f} K)" if Tu != "K" else ""
    st.success(f"**T = {fT(r['T'])}**{alt}    ·    Z = {r['Z']:.5f}    ·    phase: {r['phase']}")
for n in r["notes"]:
    st.warning(n)

c = st.columns(5)
c[0].metric("T", fT(r["T"])); c[1].metric("P", fP(r["P"])); c[2].metric("V", fV(r["V"]))
c[3].metric("Z", f"{r['Z']:.5f}"); c[4].metric("Density", f"{r['rho']:.4g} mol/L")
c = st.columns(5)
c[0].metric("H  (J/mol)", f"{r['H']:.6g}" if r["H"] is not None else "n/a")
c[1].metric("S  (J/mol·K)", f"{r['S']:.5g}" if r["S"] is not None else "n/a")
c[2].metric("Hᴿ (J/mol)", f"{r['HR']:.6g}")
c[3].metric("Sᴿ (J/mol·K)", f"{r['SR']:.5g}")
c[4].metric("Fugacity coeff. φ", f"{r['phi']:.5f}")
if r["H"] is None:
    st.caption("Total H and S need ideal-gas Cp data (Table C.1); this substance has none, so only the residual "
               "properties Hᴿ and Sᴿ are shown. You can enter Cp constants in the sidebar.")
else:
    st.caption(f"H and S are relative to the ideal gas at T₀ = {T0:g} K, P₀ = {P0:g} bar (H = S = 0 there).")
if r["Psat"] is not None:
    st.caption(f"At this temperature (below Tc): Psat = {fP(r['Psat'])}; saturated liquid V = {fV(r['Vl'])}, "
               f"saturated vapour V = {fV(r['Vv'])}.")

# ---- graph
st.subheader("Graph")
fig = make_figure(m, r)
st.pyplot(fig)
plt.close(fig)

# ---- comparison table
if compare_all:
    st.subheader("All four equations at the same inputs")
    rows = []
    for k in EOS_MODELS:
        mk = CubicEOS(sp, k); mk.T_ref, mk.P_ref = T0, P0
        try:
            rk = solve_case(mk, mode, T, P, V)
            rows.append({"EOS": k, "T": fT(rk["T"]), "P": fP(rk["P"]), "V": fV(rk["V"]), "Z": round(rk["Z"], 5),
                         "phase": rk["phase"], "H (J/mol)": None if rk["H"] is None else round(rk["H"], 1),
                         "S (J/mol·K)": None if rk["S"] is None else round(rk["S"], 3),
                         "Hᴿ (J/mol)": round(rk["HR"], 1), "Sᴿ (J/mol·K)": round(rk["SR"], 3)})
        except Exception as e:  # keep the table going if one EOS fails
            rows.append({"EOS": k, "T": "–", "P": "–", "V": "–", "Z": "–", "phase": f"error: {e}"})
    st.dataframe(rows, hide_index=True, width="stretch")

# ---- deviation from ideal gas (bottom line)
if mode == "TP":
    ideal_txt = f"V = {fV(R_GAS * r['T'] / r['P'])}"
    dev = (R_GAS * r["T"] / r["P"] - r["V"]) / r["V"] * 100
    what = "molar volume"
elif mode == "TV":
    ideal_txt = f"P = {fP(R_GAS * r['T'] / r['V'])}"
    dev = (R_GAS * r["T"] / r["V"] - r["P"]) / r["P"] * 100
    what = "pressure"
else:
    ideal_txt = f"T = {fT(r['P'] * r['V'] / R_GAS)}"
    dev = (r["P"] * r["V"] / R_GAS - r["T"]) / r["T"] * 100
    what = "temperature"
st.info(f"**Deviation from ideal gas:** the ideal-gas law (PV = RT) would give {ideal_txt}, which is "
        f"**{abs(dev):.2f} % {'higher' if dev > 0 else 'lower'}** than the {eos_name} value for the {what} "
        f"(Z = {r['Z']:.4f} instead of 1). A real-fluid equation of state is needed whenever Z departs from 1, "
        f"which happens at high pressure or low temperature.")
