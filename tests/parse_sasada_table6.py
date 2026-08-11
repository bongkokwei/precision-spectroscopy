"""Parse Table VI of Sasada & Yamada, Appl. Opt. 29, 3535 (1990).

Table VI lists calculated wavenumbers for the (2 0^0 0)-(0 0^0 0) band of
H12C14N and H13C14N -- the band commercial cells label 2*nu3.

The PDF text layer is OCR'd and scatters spaces and commas inside the
numbers ("6395.9 1 1 0", "6474 .5257", "6414.70,34"), so tokens are scrubbed
before parsing and every row is then checked against the physics.
"""

import re
import subprocess

PDF = "/sessions/festive-jolly-allen/mnt/uploads/ao-29-24-3535.pdf"

txt = subprocess.run(
    ["pdftotext", "-layout", "-f", "8", "-l", "8", PDF, "-"],
    capture_output=True, text=True,
).stdout


def cell_value(cell: str) -> float | None:
    """Recover one wavenumber from an OCR-mangled table cell.

    Every entry is dddd.dddd followed by an accuracy tag, but the OCR splits
    the digits with stray spaces, drops the decimal point, doubles it, or
    renders it as a comma. Dropping the tag and keeping the digits handles
    all of those the same way -- the 8-digit count is the check that it
    worked.
    """
    digits = re.sub(r"\D", "", re.sub(r"\(.*", "", cell))
    return float(f"{digits[:4]}.{digits[4:]}") if len(digits) == 8 else None


rows = {}
for line in txt.splitlines():
    m = re.match(r"\s{3,}(\d{1,2})\s{3,}(.*)", line)   # leading J column
    if not m:
        continue
    J = int(m.group(1))
    cells = re.split(r"\s{3,}", m.group(2))
    vals = [v for v in map(cell_value, cells) if v is not None]
    if J == 0 and len(vals) == 2:                      # J=0 has no P branch
        rows[0] = {"R12": vals[0], "R13": vals[1]}
    elif len(vals) == 4:
        rows[J] = dict(zip(("P12", "R12", "P13", "R13"), vals))

print(f"parsed {len(rows)} J rows: {min(rows)}..{max(rows)}")
missing = set(range(41)) - set(rows)
assert not missing, f"missing J rows: {sorted(missing)}"

nm = lambda wn: 1e7 / wn

# --- check 1: H13C14N must reproduce the NIST-certified wavelengths --------
# All 54 SRM 2519a lines, read back out of the config written earlier.
import yaml
NIST = yaml.safe_load(
    open("/sessions/festive-jolly-allen/mnt/precision-spectroscopy/configs/default.yaml")
)["gascell"]["species"]["H13C14N"]["lines_nm"]

dev = {}
for label, ref in NIST.items():
    br, J = label[0], int(label[1:])
    dev[label] = (nm(rows[J]["R13" if br == "R" else "P13"]) - ref) * 1000

worst = max(map(abs, dev.values()))
rms = (sum(d * d for d in dev.values()) / len(dev)) ** 0.5
print(f"  compared {len(dev)} lines: RMS {rms:.2f} pm, worst {worst:.2f} pm")
for label in ("R26", "R10", "R0", "P10", "P27"):
    print(f"    {label:4s} {dev[label]:+6.2f} pm")
# Sasada's values are extrapolated from fitted molecular constants, so the
# deviation grows with |J| while staying inside the paper's own error budget.
assert worst < 1.5, f"H13C14N disagrees with NIST by {worst:.2f} pm -- suspect a parse error"
print(f"check 1 OK\n")

# --- check 2: each branch is smooth in J ----------------------------------
# A rovibrational branch is a low-order polynomial in J, so third differences
# are tiny; a single mangled digit shows up as a large spike.
for key in ("P12", "R12", "P13", "R13"):
    Js = sorted(J for J in rows if key in rows[J])
    v = [rows[J][key] for J in Js]
    d3 = [v[i + 3] - 3 * v[i + 2] + 3 * v[i + 1] - v[i] for i in range(len(v) - 3)]
    bad = [Js[i] for i, d in enumerate(d3) if abs(d) > 0.01]
    print(f"  {key}: max |3rd diff| = {max(map(abs, d3)):.5f} cm-1 {'BAD ' + str(bad) if bad else 'ok'}")
    assert not bad, f"{key} mis-parsed near J={bad}"
print("check 2 OK: all four branches smooth\n")

# --- emit the YAML fragment for H12C14N -----------------------------------
lines = {f"R{J}": nm(r["R12"]) for J, r in rows.items() if "R12" in r}
lines |= {f"P{J}": nm(r["P12"]) for J, r in rows.items() if "P12" in r}
lo, hi = min(lines.values()), max(lines.values())
print(f"H12C14N: {len(lines)} lines spanning {lo:.4f}-{hi:.4f} nm")

keys = sorted(lines, key=lambda k: (k[0], int(k[1:])))
chunks = [keys[i:i + 4] for i in range(0, len(keys), 4)]
body = ",\n         ".join(
    ", ".join(f"{k + ':':5s} {lines[k]:.5f}" for k in c) for c in chunks
)
print("\n--- paste into default.yaml ---\n      lines_nm:\n        {" + body + "}")
