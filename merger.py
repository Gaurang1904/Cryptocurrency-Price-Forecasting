import pandas as pd

# Load ETH OHLCV data
ohlcv_df = pd.read_csv("ETH_USDT_1d.csv")
ohlcv_df['date'] = pd.to_datetime(ohlcv_df['timestamp']).dt.date

# Load ETH on-chain data
onchain_df = pd.read_csv("dune_eth_daily.csv")
onchain_df['date'] = pd.to_datetime(onchain_df['date']).dt.date

# Merge both on date
merged_df = pd.merge(ohlcv_df, onchain_df, on='date', how='inner')

# Drop date column from final output (not from input files)
final_df = merged_df.drop(columns=['date'])

# Save or preview
final_df.to_csv("eth_merged_final.csv", index=False)
print(final_df.head())
