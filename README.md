# cubic_eos: PVT of real fluids from cubic equations of state

Single-file Python program, `cubic_eos.py` (numpy; matplotlib only for plots). All species data (89 compounds, book Tables B.1 and C.1) are inside the file. Implements the generic cubic of
Smith/Van Ness Ch. 3 (Eqs. 3.41-3.51, Table 3.1) for **vdW, RK, SRK, PR**.

    P = RT/(V-b) - a(T)/((V+eps*b)(V+sigma*b))
    b = Omega*R*Tc/Pc      a(T) = Psi*alpha(Tr;w)*R^2*Tc^2/Pc
    beta = Omega*Pr/Tr     q = Psi*alpha/(Omega*Tr)

Units: T in K, P in bar, V in cm3/mol internally. The CLI converts K/C/F, bar/kPa/MPa/Pa/atm/psi
and cm3/mol, L/mol, m3/mol, m3/kmol.

## Equations of state included (lecture notation)
P = RT/(V - b) - a(T) / (V^2 + u b V + w b^2),  b = Omega R Tc/Pc,  a(T) = Psi alpha(Tr, omega) R^2 Tc^2/Pc

| EOS | u | w | a(T) |
|---|---|---|---|
| van der Waals | 0 | 0 | constant |
| Redlich/Kwong | 1 | 0 | function of T (alpha = Tr^-1/2) |
| Soave/Redlich/Kwong | 1 | 0 | function of T and omega |
| Peng/Robinson | 2 | -1 | function of T and omega |

(u = sigma + eps, w = sigma*eps; the book's sigma, eps are used internally. `python cubic_eos.py eos-table` prints all constants.)

## Website
    pip install -r requirements.txt
    streamlit run app.py
Choose the substance first, then the EOS, then enter two of T, P, V (with units). The page shows
P, T, V, Z, phase, H and S, a P-V and Z-P graph, and the deviation from the ideal gas.
To host it free: push the repo to GitHub and deploy `app.py` on Streamlit Community Cloud.

## Data source
Tc, Pc, omega, molar mass: Table B.1 of the textbook; ideal-gas Cp constants: Table C.1. Both are stored in `cubic_eos.py`.
Substances are chosen by name (e.g. `methane`, `n-butane`, `carbon dioxide`); `python cubic_eos.py list-species` lists them.

## Repository contents
* `cubic_eos.py` - the program (library + command line + data)
* `app.py` - Streamlit website
* `requirements.txt` - streamlit, numpy, matplotlib
* `test_cubic_eos.py` - validation tests (`python test_cubic_eos.py`, 21 tests)
* `report/Report_Cubic_EOS.pdf` (and `.docx`) - project report

## Thermodynamic properties (H and S)
Residual properties follow Smith/Van Ness Eqs. 13.74-13.76 for the generic cubic:

    G^R/RT = Z - 1 - ln(Z - beta) - q I
    H^R/RT = Z - 1 + [dln(alpha)/dln(Tr) - 1] q I
    S^R/R  = ln(Z - beta) + [dln(alpha)/dln(Tr)] q I
    I = ln[(Z + sigma*beta)/(Z + eps*beta)] / (sigma - eps)   (vdW: I = beta/Z)

dln(alpha)/dln(Tr) is analytic: 0 (vdW), -1/2 (RK), -kappa*sqrt(Tr)/(1 + kappa*(1 - sqrt(Tr))) (SRK, PR).
Total properties add the ideal-gas change from Cp/R = A + B T + C T^2 + D T^-2 (Table C.1):

    H = H^ig(T) - H^ig(T0) + H^R              S = S^ig(T,P) - S^ig(T0,P0) + S^R

Reference state (H = S = 0): ideal gas at T0 = 298.15 K, P0 = 1 bar (change with --T0/--P0). Only
differences between states are physically meaningful. Species lacking a Table C.1 row (about 40 of
the 89) return residual properties only. Supply --cp A B C D to get total H and S.
Units: J/mol and J/(mol K).

## Method
* Z(T,P): the cubic from Eq. 3.48 is solved for all roots with Z > beta (V > b).
* Phase: for T < Tc, Psat(T) comes from equal fugacity of liquid and vapour roots, bracketed by
  the spinodal pressures. P < Psat gives the largest root (vapour); P > Psat gives the
  smallest root (liquid). For T > Tc the single fluid root is returned.
* Fugacity coefficient: ln phi = Z - 1 - ln(Z-beta) - q*I.
* P(T,V) is direct. T(P,V) is found by bisection.

## CLI
    python cubic_eos.py list-species
    python cubic_eos.py eos-table
    python cubic_eos.py z methane --T 50 --T-unit C --P 200 --eos PR --all-eos
    python cubic_eos.py p methane --T 323.15 --V 120 --eos RK          # 198.3 bar (Ex. 3.11)
    python cubic_eos.py t carbon-dioxide --P 80 --V 250 --eos PR
    python cubic_eos.py sat n-butane --T 350 --all-eos                  # Psat, V_liq, V_vap
    python cubic_eos.py props n-butane --T 350 --P 5 --eos PR          # Z, V, phi, H^R, S^R, H, S
    python cubic_eos.py delta ammonia --T1 300 --P1 5 --T2 400 --P2 20 --eos SRK   # dH, dS
    python cubic_eos.py delta n-butane --T1 350 --P1 9.47 --T2 350 --P2 9.47 --phase1 liquid --phase2 vapor
    python cubic_eos.py hvap n-butane --T 350 --all-eos                 # latent heat, dS_vap, Psat
    python cubic_eos.py compare n-butane --T 510 --P 25                 # all models + ideal + virial
    python cubic_eos.py isotherm ammonia --T 338.15 --Pmax 30 --eos PR
    python cubic_eos.py plot n-butane --eos PR                          # PV isotherms + dome
    python cubic_eos.py plot methane --kind z --T 323.15 --Pmax 300
    # custom fluid:
    python cubic_eos.py z myfluid --Tc 400 --Pc 50 --omega 0.2 --T 350 --P 10

## Python API
    from cubic_eos import CubicEOS, Species
    m = CubicEOS(Species.lookup("n-butane"), "SRK")
    s = m.state(350, 9.4573)       # StateResult: phase, Z, V, rho, phi, Psat, ...
    m.volume_roots(350, 9.4573)    # all roots [liquid, (unstable), vapour] cm3/mol
    m.pressure(323.15, 120.0)
    m.saturation(350)              # (Psat, V_liq, V_vap)
    s.H_res, s.S_res, s.H, s.S     # J/mol, J/(mol K)
    m.residual(T, P, Z)            # (H^R, S^R) for a chosen root
    m.delta(300, 5, 400, 20)       # dict: dH, dS, state1, state2
    m.vaporization(350)            # dH_vap, dS_vap, Psat, residuals of both phases

## Validation (python test_cubic_eos.py)
* Ex. 3.9, n-butane at 350 K and 9.4573 bar: vdW/RK/SRK/PR volumes match the book's table
  (2667/191.0, 2555/133.3, 2520/127.8, 2486/112.6 cm3/mol) to within 0.5%.
* Ex. 3.11 RK pressure for methane: 198.3 bar.
* Ex. 3.10 Z for n-butane at 510 K and 25 bar: virial 0.879; cubic EOS within 0.02 of 0.873.
* Equal fugacity at saturation, ideal-gas limit, critical Z, and T/P/V round-trip consistency.

* H and S: Gibbs-Helmholtz check (H^R/RT = -T dln(phi)/dT at constant P) for all four EOS; Clapeyron
  check (dH_vap = T dV dPsat/dT) for RK/SRK/PR; ideal-gas Cp integrals against numerical quadrature;
  dH across a saturated liquid to vapor step equals dH_vap; residuals against the book's Lee/Kesler
  Tables D.5-D.12 (SRK within 0.06 RTc and 0.03 R; PR within 0.12 RTc and 0.04 R on gas states).

## Limitations
* Species data are copied from Tables B.1 and C.1 of the book. Override any value with
  --Tc/--Pc/--omega/--cp. Sulfuric acid has no omega in Table B.1.
* Ideal-gas Cp polynomials are valid only up to the Tmax listed in Table C.1 (stored as the 5th entry of each `_C1` row).
* vdW/RK/SRK liquid volumes are poor (see Ex. 3.9). Use PR, or the Rackett/Lydersen
  correlations of Sec. 3.8, for liquids.
* Quantum gases (H2, He, Ne) need the effective critical constants of Eqs. 3.54-3.56.
* Highly polar/associating fluids (water, alcohols) give larger errors with 1-parameter-omega EOS.
