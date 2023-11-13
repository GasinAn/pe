from pathlib import Path
ps = [p for p in Path("./Config_INIs_GR").iterdir()]
for p in ps:
    with p.open('r') as f:
        inis = f.readlines()
    for i in range(len(inis)):
        if inis[i] != '\n':
            inis[i] = inis[i].replace("'", "").replace('"', '')
            inis[i] = inis[i].replace(' ', '')
            inis[i] = inis[i].replace('=', ' = ')
            inis[i] = inis[i].replace(',}', '}').replace(',]', ']')
            inis[i] = inis[i].replace(":", ": ").replace(',', ', ')
            if inis[i][:6] == "time =":
                inis[i] = inis[i].replace(': ', ':')
    with p.open('w') as f:
        for ini in inis:
            if ini != '\n':
                f.write(ini)
