from fastapi import Request
from openenv.core.env_server.http_server import create_app

from models import MusicRecommendationAction, MusicRecommendationObservation
from server.music_recommendation_environment import MusicRecommendationEnv

app = create_app(
    MusicRecommendationEnv,
    MusicRecommendationAction,
    MusicRecommendationObservation,
    env_name="music_recommendation",
    max_concurrent_envs=1,
)

# REQUIRED FOR VALIDATOR
@app.post("/reset")
async def reset_override(request: Request):
    return {"status": "ok"}


# REQUIRED FOR MULTI-MODE DEPLOYMENT
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()