import glob
from pathlib import Path
import pandas as pd
from scipy.io import arff

src = glob.glob(r"D:\asd_project\asd_project\autism+screening+adult\*.arff")[0]
data, meta = arff.loadarff(src)
df = pd.DataFrame(data)
for c in df.select_dtypes(object):
    df[c] = df[c].str.decode("utf-8")
out = Path("data/questionnaire/autism_screening.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("rows:", len(df), "| columns:", len(df.columns))
print(list(df.columns))
print("saved:", out.resolve())