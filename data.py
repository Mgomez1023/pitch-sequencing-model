from pybaseball import statcast, cache
cache.enable()
df = statcast(start_dt="2026-06-01", end_dt="2026-06-30")
df.to_parquet("data/statcast_june.parquet")