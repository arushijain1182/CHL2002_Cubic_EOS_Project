# PVT Behaviour of Real Gases (Using Cubic EOS)

**CHL2002: Chemical Engineering Thermodynamics**, Prof. S. K. Gupta
<br> Arushi Jain (2025CH11574) and Pinaki Narula (2025CH71879), Tutorial group 2

Predicts the PVT behaviour and the enthalpy (H) and entropy (S) of real gases using the vdW, RK, SRK and Peng/Robinson cubic equations of state. Substance data (Tc, Pc, ω) are from Table B.1 of the textbook and ideal-gas Cp constants from Table C.1.

## Files

| File | What it contains |
|---|---|
| `cubic_eos.py` | The program: substance data, the four EOS, solver, H and S, plots and command line |
| `app.py` | Streamlit website built on `cubic_eos.py` |
| `requirements.txt` | Packages needed (streamlit, numpy, matplotlib) |
| `test_cubic_eos.py` | 21 checks against the textbook examples |
| Report PDF | Project report: theory, method and validation |

## How to use the code

Install once: `pip install -r requirements.txt`

Give a substance, an EOS (`vdW`, `RK`, `SRK` or `PR`) and any two of T, P, V:

```
python cubic_eos.py props n-butane --T 350 --P 5 --eos PR      # all properties at T and P: phase, Z, V, H, S
python cubic_eos.py p methane --T 323.15 --V 120 --eos RK      # P from T and V
python cubic_eos.py t ammonia --P 23.82 --V 1021.2 --eos PR    # T from P and V
python cubic_eos.py sat n-butane --T 350                       # saturation pressure
python cubic_eos.py compare n-butane --T 510 --P 25            # all four EOS vs ideal gas
python cubic_eos.py list-species                               # available substances
```

Units are K, bar and cm³/mol by default; other units can be given (see `python cubic_eos.py -h`). For a substance not in the list, add `--Tc --Pc --omega`. Run the checks with `python test_cubic_eos.py`.

## How to use the website

```
streamlit run app.py
```

1. Choose the **substance**.
2. Choose the **EOS**.
3. Enter two of **T, P, V** (with units) and run.

It shows P, T, V, Z, phase, H and S, a P–V and Z–P graph against the ideal gas, and the deviation from the ideal gas.

## Report

The report PDF in this repository covers the equations (class and textbook forms), the method, the data used and the validation against the textbook examples.
