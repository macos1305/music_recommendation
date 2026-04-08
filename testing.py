from server.music_recommendation_environment import MusicRecommendationEnv
from models import MusicRecommendationAction
from agent.baseline_agent import BaselineAgent

env = MusicRecommendationEnv()
agent = BaselineAgent(env.catalog)

obs = env.reset()

total_reward = 0

for step in range(20):
    action_id = agent.predict(obs)
    action = MusicRecommendationAction(track_id=action_id)

    obs = env.step(action)

    agent.update(action_id, obs.reward)

    print(f"Step {step} | Action {action_id} | Reward {obs.reward}")

    total_reward += obs.reward

    if obs.done:
        break

print("Total Reward:", total_reward)
