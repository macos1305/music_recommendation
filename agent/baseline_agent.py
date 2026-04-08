import random


class BaselineAgent:
    def __init__(self, catalog):
        self.catalog = catalog
        self.last_good_track = None

    def predict(self, state):
        history = state.history
        avoid = history[-1]

        # If we found a good track → exploit similar ones
        if self.last_good_track is not None:
            base = self.catalog[self.last_good_track]

            candidates = []
            for tid, t in self.catalog.items():
                if tid == avoid:
                    continue

                score = 0

                if t["genre"] == base["genre"]:
                    score += 2
                if t["language"] == base["language"]:
                    score += 1

                if score >= 2:
                    candidates.append(tid)

            if candidates:
                return random.choice(candidates)

        # Otherwise explore
        choices = list(self.catalog.keys())
        if avoid in choices:
            choices.remove(avoid)

        return random.choice(choices)

    def update(self, action, reward):
        # If reward is relatively good → remember this track
        if reward > -0.2:
            self.last_good_track = action