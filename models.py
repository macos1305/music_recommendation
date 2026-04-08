from openenv_core.env_server.types import Action, Observation
from pydantic import Field
from typing import List



class MusicRecommendationAction(Action):
    """Action: Recommend a track ID"""

    track_id: int = Field(..., description="Track ID to recommend")


class MusicRecommendationObservation(Observation):
    """Observation: Current user session state"""

    history: List[int] = Field(
        ..., description="Last 5 recommended track IDs"
    )

    session_length: int = Field(
        ..., description="Current session step count"
    )

    reward: float = Field(
        default=0.0, description="Reward received from last action"
    )

    done: bool = Field(
        default=False, description="Whether session has ended"
    )