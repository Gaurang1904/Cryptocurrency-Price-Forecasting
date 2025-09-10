import pandas as pd

# Load your scaled dataset
df = pd.read_csv("BNB_FE.csv")

# Rename 'timestamp' to 'ds' and 'Close' to 'y'
df = df.rename(columns={
    "timestamp": "ds",
    "Close": "y"
})

# Add a static 'unique_id' column (e.g., "BTC")
df["unique_id"] = "ETH"

# Reorder columns: unique_id, ds, y, then everything else
ordered_cols = ["unique_id", "ds", "y"] + [col for col in df.columns if col not in ["unique_id", "ds", "y"]]
df = df[ordered_cols]

#save the formatted file
df.to_csv("BNB_Final_ready.csv", index=False)
print("✅ Data formatted for TFT and saved to BNB_Final_ready.csv")   
