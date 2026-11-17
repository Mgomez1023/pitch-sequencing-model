from typing import Literal
from pydantic import BaseModel, Field

class PitchRequest(BaseModel):
    pitch_type: Literal["FF", "SI", "SL", "CH", "ST", "FC", "CU",
                        "FS", "KC", "SV", "EP", "FA", "FO", "KN",
                        "CS"]
    release_speed: float = Field(ge=28.3, le=107.5)
    release_spin_rate: float = Field(ge=1.0, le=4000.0)   
    pfx_x: float = Field(ge=-3.0, le=3.0)     
    pfx_z: float = Field(ge=-3.0, le=3.0)         
    plate_x: float = Field(ge=-6.5, le=7.2)       
    plate_z: float = Field(ge=-5.2, le=12.0)      
    balls: int = Field(ge=0, le=3)
    strikes: int = Field(ge=0, le=2)
    stand: Literal["L", "R"]
    p_throws: Literal["L", "R"]

class PredictionResponse(BaseModel):
    whiff_probability: float