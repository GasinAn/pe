from pathlib import Path
ps = [p for p in Path("./Config_INIs_GR").iterdir()]
for p in ps:
    with p.open('r') as f:
        inis = f.readlines()
    inis.sort()
    with p.open('w') as f:
        for ini in inis:
            if ini != '\n':
                f.write(ini)
