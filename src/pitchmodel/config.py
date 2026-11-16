from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

RAW_COLUMNS = ["game_date", "game_pk", "game_type", "at_bat_number", "pitch_number",
               "pitcher", "batter", "pitch_type", "description",
               "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "plate_x", "plate_z",
               "balls", "strikes", "stand", "p_throws",
               "release_pos_x", "release_pos_y", "release_pos_z",
               "vx0", "vy0", "vz0", "ax", "ay", "az"]

WHIFF_DESCRIPTIONS = ["swinging_strike", "swinging_strike_blocked", "missed_bunt", "foul_tip"]
SWING_DESCRIPTIONS = ["swinging_strike", "swinging_strike_blocked", "missed_bunt",
                      "foul", "foul_tip", "hit_into_play", "foul_bunt", "bunt_foul_tip"]

FEATURES = ["pitch_type", "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
            "plate_x", "plate_z", "balls", "strikes", "stand", "p_throws"]   # Model A (the one we ship)
CATEGORICAL_FEATURES = ["pitch_type", "stand", "p_throws"]

SEQUENCE_FEATURES = ["prev_pitch_type", "prev_speed", "speed_change", "same_type_as_prev",
                     "plate_z_change", "plate_x_change", "prev_whiff"]          # Model B (experimental)

FASTBALLS = ["FF", "SI", "FC"]
OFFSPEED = ["SL", "ST", "CU", "KC", "CH", "FS"]

TRAIN_CUTOFF = "2026-09-01"
RANDOM_STATE = 42