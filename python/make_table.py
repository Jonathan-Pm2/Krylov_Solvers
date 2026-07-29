"""Emit a booktabs LaTeX table from the benchmark CSVs (reproducible, no manual
transcription). Run: python code/python/make_table.py
Writes code/results/table_drazin.tex, \\input-ed by the paper section.
"""
import csv
import os

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def load(lang):
    with open(os.path.join(RES, f"drazin_{lang}.csv")) as f:
        return list(csv.DictReader(f))


def fmt(x):
    """Clean LaTeX scientific notation, e.g. 2.63\\times10^{-9}."""
    s = f"{float(x):.2e}"
    mant, exp = s.split("e")
    return rf"{mant}\times10^{{{int(exp)}}}"


rows_jl = {(r["n"], r["index"], r["method"]): r for r in load("julia")}
rows_py = {(r["n"], r["index"], r["method"]): r for r in load("python")}

keys = sorted({(int(r["n"]), int(r["index"])) for r in load("julia")})

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\small",
    r"\caption{Classical GMRES vs.\ DGMRES on non-symmetric singular systems with"
    r" Drazin index $k$. \emph{d-err} is the relative error against the true Drazin"
    r" solution $A^{D}b$; \emph{res} is the ordinary relative residual"
    r" $\|Ax-b\|/\|b\|$. GMRES minimises \emph{res} yet never reaches $A^{D}b$;"
    r" DGMRES drives \emph{d-err} below the $10^{-10}$ tolerance. Iteration cap 400.}",
    r"\label{tab:drazin}",
    r"\begin{tabular}{cc l cc cc}",
    r"\toprule",
    r" & & & \multicolumn{2}{c}{Julia} & \multicolumn{2}{c}{Python} \\",
    r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"$n$ & $k$ & method & iters & d-err & iters & d-err \\",
    r"\midrule",
]

for (n, k) in keys:
    for method in ("GMRES", "DGMRES"):
        jl = rows_jl[(str(n), str(k), method)]
        py = rows_py[(str(n), str(k), method)]
        mlab = method if method == "GMRES" else r"\textbf{DGMRES}"
        line = (f"{n} & {k} & {mlab} & {jl['iters']} & ${fmt(jl['drazin_error'])}$ "
                f"& {py['iters']} & ${fmt(py['drazin_error'])}$ \\\\")
        lines.append(line)
    lines.append(r"\addlinespace[2pt]")

lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

out = os.path.join(RES, "table_drazin.tex")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", out)
