from typing import Dict, List
import random

from models import (
    MusicRecommendationAction,
    MusicRecommendationObservation,
)


class MusicRecommendationEnv:
    def __init__(self):
        self.max_history = 5
        self.max_session = 20

        full_catalog = self._build_catalog()
        sampled_items = random.sample(list(full_catalog.items()), 40)
        self.catalog = dict(sampled_items)

        self.history: List[int] = []
        self.session_length: int = 0
        self.user_preference: Dict = {}

        

    # ----------------------------
    # RESET
    # ----------------------------
    def reset(self) -> MusicRecommendationObservation:
        self.history = [-1] * self.max_history
        self.session_length = 0
        self.user_preference = self._sample_user()

        return MusicRecommendationObservation(
            history=self.history,
            session_length=self.session_length,
            reward=0.0,
            done=False,
        )

    # ----------------------------
    # STEP
    # ----------------------------
    def step(
        self, action: MusicRecommendationAction
    ) -> MusicRecommendationObservation:

        track_id = action.track_id
        reward = 0.0
        done = False

        # ----------------------------
        # HARD PENALTIES
        # ----------------------------
        if track_id not in self.catalog:
            reward = -5.0

        elif self.history[-1] == track_id:
            reward = -5.0

        else:
            track = self.catalog[track_id]
            pref = self.user_preference

        track = self.catalog[track_id]
        pref = self.user_preference

        if track["genre"] == pref["genre"]:
            reward = 2
        elif track["language"] == pref["language"]:
            reward = 1
        else:
            reward = -1
        # ----------------------------
        # UPDATE STATE
        # ----------------------------
        self.history.pop(0)
        self.history.append(track_id)

        self.session_length += 1

        # ----------------------------
        # TERMINATION
        # ----------------------------
        if self.session_length >= self.max_session:
            done = True

        # ----------------------------
        # RETURN OBSERVATION
        # ----------------------------
        return MusicRecommendationObservation(
            history=self.history,
            session_length=self.session_length,
            reward=reward,
            done=done,
        )

    # ----------------------------
    # STATE (OPTIONAL)
    # ----------------------------
    def state(self) -> MusicRecommendationObservation:
        return MusicRecommendationObservation(
            history=self.history,
            session_length=self.session_length,
            reward=0.0,
            done=False,
        )

    # ----------------------------
    # CATALOG
    # ----------------------------
    def _build_catalog(self) -> Dict[int, Dict]:
        genres = ["romantic", "mass","sad", "party","devotional"]
        languages = ["telugu", "hindi","english"]
        artists = ["karthik", "arjith singh","taylor swift"]
        album_artists = ["devi sri prasad","ar rahman","hans zimmer"]
        years = [2015,2016,2017,2018,2019, 2020, 2021]

        catalog = {}
        track_id = 0

        for g in genres:
            for l in languages:
                for a in artists:
                    for aa in album_artists:
                        for y in years:
                            catalog[track_id] = {
                                "genre": g,
                                "language": l,
                                "artist": a,
                                "album_artist": aa,
                                "year": y,
                            }
                            track_id += 1

        return catalog

    # ----------------------------
    # USER SIMULATION
    # ----------------------------
    def _sample_user(self) -> Dict:
        # Pick a random track as preference profile
        return random.choice(list(self.catalog.values())).copy()