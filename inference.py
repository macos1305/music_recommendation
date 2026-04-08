import os
import torch
import random
from typing import List

from openai import OpenAI

from server.music_recommendation_environment import MusicRecommendationEnv
from models import MusicRecommendationAction
from agent.dqn_agent import DQN


# ----------------------------
# CONFIG (MANDATORY)
# ----------------------------
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASK_NAME = "music-recommendation"
BENCHMARK = "openenv-music"
MAX_STEPS = 20


# ----------------------------
# LOGGING
# ----------------------------
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success, steps, score, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ----------------------------
# STATE ENCODING
# ----------------------------
def encode(state, catalog):
    genres = list(set([t["genre"] for t in catalog.values()]))
    genre_map = {g: i for i, g in enumerate(genres)}

    genre_counts = [0] * len(genres)

    for track_id in state.history:
        if track_id == -1:
            continue
        track = catalog[track_id]
        genre_counts[genre_map[track["genre"]]] += 1

    genre_counts = [x / 5 for x in genre_counts]
    return genre_counts + [state.session_length / 20]


# ----------------------------
# LLM HINT (LIGHTWEIGHT)
# ----------------------------
def get_llm_genre_hint(obs):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Given a list of track IDs, suggest a music genre (one word).",
                },
                {
                    "role": "user",
                    "content": f"History: {obs.history}",
                },
            ],
            max_tokens=5,
            temperature=0.2,
        )

        return (response.choices[0].message.content or "").strip().lower()

    except Exception:
        return None


# ----------------------------
# MAIN
# ----------------------------
def main():
    env = MusicRecommendationEnv()

    action_list = list(env.catalog.keys())
    action_size = len(action_list)

    sample_state = encode(env.reset(), env.catalog)
    state_size = len(sample_state)

    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    rewards = []
    steps_taken = 0

    log_start(TASK_NAME, BENCHMARK, "DQN+LLM")

    try:
        obs = env.reset()
        state = encode(obs, env.catalog)

        for step in range(1, MAX_STEPS + 1):

            state_tensor = torch.FloatTensor(state)
            q_values = model(state_tensor)

            sorted_indices = torch.argsort(q_values, descending=True)

            llm_genre = get_llm_genre_hint(obs)

            action_id = None

            # 🔥 DQN FIRST, LLM ONLY HELPS
            for idx in sorted_indices:
                candidate = action_list[idx.item()]

                # avoid repeating same track
                if candidate == obs.history[-1]:
                    continue

                action_id = candidate  # default DQN choice

                # if LLM agrees → accept immediately
                if llm_genre:
                    track = env.catalog[candidate]
                    if llm_genre in track["genre"]:
                        break

            # fallback
            if action_id is None:
                action_id = random.choice(action_list)

            # small exploration (prevents loops)
            if random.random() < 0.05:
                action_id = random.choice(action_list)

            action = MusicRecommendationAction(track_id=action_id)
            result = env.step(action)

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step, str(action_id), reward, done)

            state = encode(result, env.catalog)
            obs = result

            if done:
                break

        # ----------------------------
        # SCORE
        # ----------------------------
        max_possible = MAX_STEPS * 2
        score = sum(rewards) / max_possible
        score = max(0.0, min(1.0, score))

        success = score >= 0.7

    except Exception as e:
        log_step(steps_taken, "error", 0.0, True, str(e))
        score = 0.0
        success = False

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()