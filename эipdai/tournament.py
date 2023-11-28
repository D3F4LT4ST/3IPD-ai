import random
import numpy as np
from .environment import IPDGame
from .strategies import Strategy
from typing import List, Dict

def schedule_games_subset(n_players: int) -> List[List[int]]:

    players = np.random.permutation(n_players)

    n_basic_games = n_players // 2

    schedule = []
    for i in range(n_basic_games):
        schedule.append([players[i * 2], players[i * 2 + 1]])

    if n_players % 2 == 1:
        r = random.randint(0, n_players - 2)
        if r == players[-1]:
            r += 1
        schedule.append([players[-1], r])

    return schedule

def run_tournament(
        environment: IPDGame,
        players: Strategy, 
        n_runs: int
    ) -> Dict[str, int]:
    
    results = []

    for player in players:

        tally = 0
        n_games = 0

        for run in range(n_runs):
            for roster in schedule_games_subset(len(players)):

                round_players = [player] + [players[i] for i in roster]
                for player in round_players:
                    player.reset()

                observations, infos = environment.reset()
                rewards = {player_id: None for player_id in environment.possible_agents}

                while True:
                    actions = {
                        agent_id : round_players[i].play(observations[agent_id], rewards[agent_id]) 
                        for i, agent_id in enumerate(environment.possible_agents)
                    }

                    observations, rewards, terminations, trunctations, infos = environment.step(actions)

                    tally += rewards[environment.possible_agents[0]]

                    if any(terminations.values()): break

                n_games += 1
        
        results.append(tally / n_games)

    return {players[i].name : results[i] for i in np.argsort(results)}