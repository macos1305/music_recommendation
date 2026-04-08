---
title: Music Recommendation RL
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

# Reinforcement Learning Music Recommendation System

## Overview
This project implements a Reinforcement Learning (DQN) based music recommendation system using the OpenEnv framework.

The agent learns to recommend tracks that maximize long-term user engagement based on simulated user preferences.

---

## Approach

### Environment
- Simulated music streaming platform
- Tracks defined by:
  - Genre
  - Language
  - Artist
  - Album Artist
  - Year

### State
- Last 5 tracks played
- Session length

### Action
- Recommend next track (track_id)

### Reward
- +2 → Genre match  
- +1 → Language match  
- -1 → Mismatch  
- -5 → Invalid or repeated track  

---

## Model

- Deep Q-Network (DQN)
- Target Network for stability
- Experience Replay
- Epsilon-greedy exploration

---

## Results

- Learns user preferences over time
- Achieves high engagement score (>0.8)
- Stable and deterministic inference

---

## How to Run

### Local
```bash
python inference.py