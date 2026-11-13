from pybaseball import statcast, cache
cache.enable()
df = statcast(start_dt="2026-09-01", end_dt="2026-09-30")
df.to_parquet("data/statcast_september.parquet")