import torch

from server.music_recommendation_environment import MusicRecommendationEnv
from models import MusicRecommendationAction
from agent.dqn_agent import DQNAgent


# ----------------------------
# STATE ENCODING (GENRE ONLY)
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

    # normalize
    genre_counts = [x / 5 for x in genre_counts]

    return genre_counts + [state.session_length / 20]


# ----------------------------
# INIT ENV
# ----------------------------
env = MusicRecommendationEnv()

# Action mapping (important!)
action_list = list(env.catalog.keys())
action_size = len(action_list)

# Determine state size dynamically
sample_state = encode(env.reset(), env.catalog)
state_size = len(sample_state)

print("State size:", state_size)
print("Action size:", action_size)


# ----------------------------
# INIT AGENT
# ----------------------------
agent = DQNAgent(state_size, action_size)

EPISODES = 300


# ----------------------------
# TRAINING
# ----------------------------
rewards = []
best_reward = float("-inf")

for e in range(EPISODES):
    obs = env.reset()
    state = encode(obs, env.catalog)

    total_reward = 0

    for step in range(20):

        # DQN → action index
        action_index = agent.act(state)

        # map index → actual track_id
        action_id = action_list[action_index]

        action = MusicRecommendationAction(track_id=action_id)

        # environment step
        next_obs = env.step(action)
        next_state = encode(next_obs, env.catalog)

        reward = next_obs.reward/2

        # store experience
        agent.remember(state, action_index, reward, next_state, next_obs.done)

        # train every step
        agent.replay(batch_size=32)

        state = next_state
        total_reward += reward

        if next_obs.done:
            break

    # ----------------------------
    # LOGGING
    # ----------------------------
    rewards.append(total_reward)

    if len(rewards) >= 10:
        avg_reward = sum(rewards[-10:]) / 10
        print(f"Episode {e} | Reward {total_reward} | Avg {avg_reward:.2f}")
    else:
        print(f"Episode {e} | Reward {total_reward}")

    # ----------------------------
    # SAVE BEST MODEL
    # ----------------------------
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(agent.model.state_dict(), "best_model.pth")