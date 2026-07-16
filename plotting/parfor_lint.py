"""Static check for the parfor rule that just bit us:
'Assigning to the for-loop variable X is not supported in parfor-loops'.

Inside a parfor body, a name used as a `for X = ...` loop variable ANYWHERE
cannot also receive a plain assignment `X = ...` elsewhere in that body.
"""
import re, pathlib, sys

path = sys.argv[1] if len(sys.argv) > 1 else "scour_MATLAB/A00_Run.m"
src = pathlib.Path(path).read_text(encoding="utf-8").splitlines()

def code_of(line):
    """strip comments, but not '%' inside a quoted string"""
    out, inq, i = [], False, 0
    while i < len(line):
        c = line[i]
        if c == "'" and not (inq and i+1 < len(line) and line[i+1] == "'"):
            inq = not inq
        if c == '%' and not inq:
            break
        out.append(c); i += 1
    return "".join(out)

start = next(i for i, l in enumerate(src) if re.match(r"\s*parfor\s", code_of(l)))
depth, end = 0, None
for i in range(start, len(src)):
    c = code_of(src[i])
    depth += len(re.findall(r"(?:^|;|\s)(?:parfor|for|if|while|switch|function)\b", c))
    # count only STATEMENT 'end' (not the index keyword in P_(end+1,:) )
    depth -= len(re.findall(r"(?:^|;)\s*end\b(?!\s*[+\-),])", c))
    if i > start and depth <= 0:
        end = i
        break
body = src[start:(end or len(src))]
print(f"parfor body: lines {start+1}..{(end or len(src))}  ({len(body)} lines)")

loopvars, assigns = {}, {}
for n, line in enumerate(body, start+1):
    c = code_of(line)
    m = re.match(r"\s*for\s+([A-Za-z_]\w*)\s*=", c)
    if m:
        loopvars.setdefault(m.group(1), []).append(n)
        continue
    for m in re.finditer(r"(?:^|;)\s*([A-Za-z_]\w*)\s*=(?!=)", c):
        assigns.setdefault(m.group(1), []).append(n)

bad = False
for v, lns in sorted(loopvars.items()):
    clash = assigns.get(v)
    if clash:
        bad = True
        print(f"  [FAIL] '{v}': loop var at {lns} BUT plain-assigned at {clash}")
if not bad:
    print("  [OK] no loop variable receives a plain assignment")
print(f"  loop vars seen: {', '.join(sorted(loopvars))}")
