# train.py
import random
from env import NQueensEnv
from agent import SimpleAgent, save_agent_state

def train_random_episodes(n=8, episodes=1000, save_path="models/agent.json"):
    env = NQueensEnv(n=n)
    agent = SimpleAgent(mode="random")
    best = None
    for ep in range(episodes):
        env.reset()
        done = False
        steps = 0
        while not done:
            action = agent.act(env)
            if action is None:
                break
            obs, done, info = env.step(action)
            steps += 1
            # si mouvement invalide, on arrête cet épisode
            if "error" in info:
                break
        if done:
            best = env.state
            print(f"Episode {ep}: solved! solution={env.state}")
            break
    if best:
        save_agent_state(save_path, {"n": n, "solution": best})
    else:
        print("No solution found during random training.")
    return best

if __name__ == "__main__":
    train_random_episodes(n=8, episodes=2000)
