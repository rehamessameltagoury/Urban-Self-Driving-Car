import pandas as pd
third = pd.read_csv("test.csv")
first = pd.read_csv("driving_log.csv")
second = pd.read_csv("files_path.csv")

merged = first.merge(second)
merged = merged.merge(third)
merged.to_csv("comdataandpix.csv", index=False)
print("DONE")	 

