# -*- coding: utf-8 -*-

from bettermdptools.algorithms.rl import RL
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.test_env import TestEnv
import numpy as np
import itertools
from joblib import Parallel, delayed

class GridSearch:
    @staticmethod
    def q_learning_grid_search(env, gamma, epsilon_decay, iters, verbose=True):
        results = []
        for i in itertools.product(gamma, epsilon_decay, iters):
            if verbose:
                print("running q_learning with gamma:", i[0],  "epsilon decay:", i[1],  " iterations:", i[2])
            Q, V, pi, Q_track, pi_track = RL(env).q_learning(gamma=i[0], epsilon_decay_ratio=i[1], n_episodes=i[2])
            episode_rewards = TestEnv.test_env(env=env, n_iters=100, pi=pi)
            avg_reward = np.mean(episode_rewards)
            results.append({
                "gamma": i[0],
                "epsilon_decay": i[1],
                "iters": i[2],
                "average_reward": avg_reward
            })
            if verbose:
                print("Avg. episode reward: ", avg_reward)
                print("###################")
        return results
    
    @staticmethod
    def q_learning_grid_search_2(env, gamma, init_epsilon, min_epsilon, epsilon_decay_ratio, init_alpha, min_alpha, alpha_decay_ratio, iters, verbose=True, n_jobs=6):
        # Setup the parameter grid using itertools.product
        param_grid = list(itertools.product(gamma, init_epsilon, min_epsilon, epsilon_decay_ratio, init_alpha, min_alpha, alpha_decay_ratio, iters))

        # Execute in parallel
        results = Parallel(n_jobs=n_jobs)(delayed(GridSearch.run_single_q_learning_instance)(
            env, *params) for params in param_grid)

        if verbose:
            for result in results:
                print(f"Running Q-learning with {result}")
                print("Avg. episode reward: ", result["average_reward"])
                print("###################")

        return results
    @staticmethod
    def run_single_q_learning_instance(env, gamma, eps_init, eps_min, eps_decay, alpha_init, alpha_min, alpha_decay, num_iters):
        agent = RL(env)
        Q, V, pi, Q_track, pi_track = agent.q_learning(
            gamma=gamma,
            init_epsilon=eps_init,
            min_epsilon=eps_min,
            epsilon_decay_ratio=eps_decay,
            init_alpha=alpha_init,
            min_alpha=alpha_min,
            alpha_decay_ratio=alpha_decay,
            n_episodes=num_iters
        )
        episode_rewards = TestEnv.test_env(env=env, n_iters=100, pi=pi)
        avg_reward = np.mean(episode_rewards)
        return {
            "gamma": gamma,
            "init_epsilon": eps_init,
            "min_epsilon": eps_min,
            "epsilon_decay_ratio": eps_decay,
            "init_alpha": alpha_init,
            "min_alpha": alpha_min,
            "alpha_decay_ratio": alpha_decay,
            "iters": num_iters,
            "average_reward": avg_reward
        }

    @staticmethod
    def sarsa_grid_search(env, gamma, epsilon_decay, iters, verbose=True):
        results = []
        for i in itertools.product(gamma, epsilon_decay, iters):
            if verbose:
                print("running sarsa with gamma:", i[0],  "epsilon decay:", i[1],  " iterations:", i[2])
            Q, V, pi, Q_track, pi_track = RL(env).sarsa(gamma=i[0], epsilon_decay_ratio=i[1], n_episodes=i[2])
            episode_rewards = TestEnv.test_env(env=env, n_iters=100, pi=pi)
            avg_reward = np.mean(episode_rewards)
            results.append({
                "gamma": i[0],
                "epsilon_decay": i[1],
                "iters": i[2],
                "average_reward": avg_reward
            })
            if verbose:
                print("Avg. episode reward: ", avg_reward)
                print("###################")
        return results

    @staticmethod
    def pi_grid_search(env, gamma, n_iters, theta, verbose=True):
        results = []        
        for i in itertools.product(gamma, n_iters, theta):
            if verbose:
                print("running PI with gamma:", i[0],  " n_iters:", i[1], " theta:", i[2])
            V, V_track, pi = Planner(env.P).policy_iteration(gamma=i[0], n_iters=i[1], theta=i[2])
            episode_rewards = TestEnv.test_env(env=env, n_iters=100, pi=pi)
            avg_reward = np.mean(episode_rewards)
            results.append({
                "gamma": i[0],
                "n_iters": i[1],
                "theta": i[2],
                "average_reward": avg_reward
            })
            if verbose:
                print("Avg. episode reward: ", avg_reward)
                print("###################")
        return results

    @staticmethod
    def vi_grid_search(env, gamma, n_iters, theta, verbose=True):
        results = []
        for i in itertools.product(gamma, n_iters, theta):
            if verbose:
                print("running VI with gamma:", i[0],  " n_iters:", i[1], " theta:", i[2])
            V, V_track, pi = Planner(env.P).value_iteration(gamma=i[0], n_iters=i[1], theta=i[2])
            episode_rewards = TestEnv.test_env(env=env, n_iters=100, pi=pi)
            avg_reward = np.mean(episode_rewards)
            results.append({
                "gamma": i[0],
                "n_iters": i[1],
                "theta": i[2],
                "average_reward": avg_reward
            })
            if verbose:
                print("Avg. episode reward: ", avg_reward)
                print("###################")
        
        return results
            