
# load csv
df = pd.read_csv("../data/data_eng.csv", index_col=0)

# filter nan data
df[["Upside_Review", "Downside_Review"]] = df[["Upside_Review", "Downside_Review"]].fillna("")

# rebuild the nan empty, because the data seems wrong. for up and down side
df["up_empty"] = df["Upside_Review"].str.strip() == ""
df["down_empty"] = df["Downside_Review"].str.strip() == ""

# text: upside + downside, lowercase, newlines and repeated spaces collapsed
df["text"] = (df["Upside_Review"] + " " + df["Downside_Review"]).str.lower()

# Splitting csv file and sentiment
strata = df["source"].astype(str) + "_" + df["Sentiment"]
train_df, test_df = train_test_split(
    df, test_size=0.25, stratify=strata, random_state=42
)
y_train = train_df["source"]   # 0 = human, 1 = AI
y_test = test_df["source"]
X_train = train_df[["text", "Hotel Name", "City Name"]]
X_test = test_df[["text", "Hotel Name", "City Name"]]

