"""Emit results/table_restart.tex from restart_{julia,python}.csv.
Run: python code/python/make_restart_table.py
"""
import csv
import os

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def load(lang):
    with open(os.path.join(RES, f"restart_{lang}.csv")) as f:
        return list(csv.DictReader(f))


def fmt(x):
    mant, exp = f"{float(x):.1e}".split("e")
    return rf"{mant}\times10^{{{int(exp)}}}"


jl = {(r["n"], r["index"]): r for r in load("julia")}
py = {(r["n"], r["index"]): r for r in load("python")}
keys = sorted({(int(r["n"]), int(r["index"])) for r in load("julia")})

lines = [
    r"\begin{table}[htbp]\centering\small",
    r"\caption{Restarted DGMRES(20) at scale. Three restart cycles ($60$ total"
    r" iterations, a fixed $20$-column basis) recover $A^{D}b$ up to $n=50{,}001$"
    r" in both languages; the iteration count is independent of $n$ (well-"
    r"conditioned core) and the Julia/Python behavior is identical.}",
    r"\label{tab:restart}",
    r"\begin{tabular}{r r r r r r}",
    r"\toprule",
    r" & & & & \multicolumn{2}{c}{$\|x-A^{D}b\|/\|A^{D}b\|$}\\",
    r"\cmidrule(lr){5-6}",
    r"$n$ & $k$ & cycles & total iters & Julia & Python\\",
    r"\midrule",
]
for (n, k) in keys:
    j, p = jl[(str(n), str(k))], py[(str(n), str(k))]
    lines.append(rf"{n} & {k} & {j['cycles']} & {j['total_iters']} "
                 rf"& ${fmt(j['drazin_error'])}$ & ${fmt(p['drazin_error'])}$ \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

with open(os.path.join(RES, "table_restart.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote results/table_restart.tex")
