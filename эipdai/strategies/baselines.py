import random
import numpy as np
from enum import Enum
from typing import List
from .base import Strategy
from ..common import Actions, PAYOFF_MATRIX

class Naive(Strategy):
    '''
    Always cooperates.
    '''
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        return Actions.C
    

class Defector(Strategy):
    '''
    Always defects.
    '''
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        return Actions.D
    

class Random(Strategy):
    '''
    Random strategy.
    '''
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        return random.randint(Actions.C, Actions.D)


class SoftT4T(Strategy):
    '''
    Defects only if both of the opponents defected in the last move.
    Taken from: https://www.classes.cs.uchicago.edu/archive/1998/fall/CS105/Project/node6.html
    '''
    def play(
        self, 
        observation=np.array,
        reward=None
    ) -> Actions:
        
        start = (self._rounds_played == 0)
        self._rounds_played += 1

        if start: return Actions.C

        if observation[1][-1] == Actions.D and observation[2][-1] == Actions.D: return Actions.D

        return Actions.C
    

class ToughT4T(Strategy):
    '''
    Defects if either of the opponents defected in the last move.
    Taken from: https://www.classes.cs.uchicago.edu/archive/1998/fall/CS105/Project/node6.html
    '''
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:

        start = (self._rounds_played == 0)
        self._rounds_played += 1

        if start: return Actions.C

        if observation[1][-1] == Actions.D or observation[2][-1] == Actions.D: return Actions.D

        return Actions.C
    

class FairT4T(Strategy):

    class Defector(Enum):
        BOTH = 0
        OPP1 = 1
        OPP2 = 2

    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:

        start = (self._rounds_played == 0)
        self._rounds_played += 1

        if start: 
            return Actions.C

        if self._defector == None:
            if observation[1][-1] == Actions.D and observation[2][-1] == Actions.D: 
                self._defector = self.Defector.BOTH
            elif observation[1][-1] == Actions.D:
                self._defector = self.Defector.OPP1
            elif observation[2][-1] == Actions.D:
                self._defector = self.Defector.OPP2

        if self._defector == self.Defector.BOTH:
            if observation[1][-1] == Actions.D or observation[2][-1] == Actions.D:
                return Actions.D
            else:
                self._defector = None
                return Actions.C
        elif self._defector == self.Defector.OPP1:
            if observation[1][-1] == Actions.D:
                return Actions.D
            else:
                self._defector = None
                return Actions.C
        elif self._defector == self.Defector.OPP2:
            if observation[2][-1] == Actions.D:
                return Actions.D
            else:
                self._defector = None
                return Actions.C

        return Actions.C
    
    def reset(self):
        super().reset()
        self._defector = None
    

class DecayingT4T(FairT4T):

    def __init__(self, name=None) -> None:
        super().__init__(name)
        self._rounds_of_decay = 200
        self._end_coop_prob = 0.5

    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        action = super().play(observation)

        if action == Actions.C:
            if random.random() > max(
                1 - self._end_coop_prob / self._rounds_of_decay * self._rounds_played, 
                self._end_coop_prob
            ):
                action = Actions.D
        
        return action
    

class GradualT4T(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/titfortat.py 
    '''
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        start = (self._rounds_played == 0)
        self._rounds_played += 1

        if start: return Actions.C

        if self._calming:
            self._calming = False
            return Actions.C

        if self._punishing:
            if self._punishment_count < self._punishment_limit:
                self._punishment_count += 1
                return Actions.D
            else:
                self._calming = True
                self._punishing = False
                self._punishment_count = 0
                return Actions.C
            
        if observation[1][-1] == Actions.D or observation[2][-1] == Actions.D:
            self._punishing = True
            self._punishment_count += 1
            self._punishment_limit += 1
            return Actions.D
        
        return Actions.C
    
    def reset(self):
        super().reset()
        self._calming = False
        self._punishing = False
        self._punishment_count = 0
        self._punishment_limit = 0


class SoftT42T(Strategy):

    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:

        if self._rounds_played == 0 or self._rounds_played == 1:
            action = Actions.C
        elif (
            observation[1][-1] == Actions.D and 
            observation[1][-2] == Actions.D and 
            observation[2][-1] == Actions.D and 
            observation[2][-2] == Actions.D
        ): 
            action = Actions.D
        else:
            action = Actions.C

        self._rounds_played += 1

        return action


class ToughT42T(Strategy):

    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:

        if self._rounds_played == 0 or self._rounds_played == 1:
            action = Actions.C
        elif (
            (observation[1][-1] == Actions.D and observation[1][-2] == Actions.D) or 
            (observation[2][-1] == Actions.D and observation[2][-2] == Actions.D)
        ): 
            action = Actions.D
        else:
            action = Actions.C

        self._rounds_played += 1

        return action


class AnotherT42T(Strategy):

    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        
        start = (self._rounds_played == 0)
        self._rounds_played += 1

        if start: return Actions.C

        if observation[1][-1] == Actions.D:
            self._opp1_defection_count += 1
        if observation[2][-1] == Actions.D:
            self._opp2_defection_count += 1

        if self._opp1_defection_count >= 2 and self._opp2_defection_count >= 2:
            self._opp1_defection_count = self._opp2_defection_count = 0
            return Actions.D
        
        return Actions.C
    
    def reset(self):
        super().reset()
        self._opp1_defection_count = self._opp2_defection_count = 0
    
    
class SoftGrudger(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/grudger.py
    '''
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        
        start = (self._rounds_played == 0)
        self._rounds_played += 1

        if start: return Actions.C

        if (observation[1][-1] == Actions.D and observation[2][-1] == Actions.D) and not self._triggered:
            self._triggered = True

        if self._triggered: return Actions.D

        return Actions.C
    
    def reset(self):
        super().reset()
        self._triggered = False


class ToughGrudger(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/grudger.py
    '''
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        
        start = (self._rounds_played == 0)
        self._rounds_played += 1

        if start: return Actions.C

        if (observation[1][-1] == Actions.D or observation[2][-1] == Actions.D) and not self._triggered:
            self._triggered = True

        if self._triggered: return Actions.D

        return Actions.C

    def reset(self):
        super().reset()
        self._triggered = False


class Grofman(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/axelrod_first.py FirstByGrofman
    '''
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:

        if self._rounds_played == 0 or observation[0][-1] == observation[1][-1] == observation[2][-1]:
            action = Actions.C
        else:
            action = int(random.random() > 2/7)

        self._rounds_played += 1

        return action


class Joss(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/axelrod_first.py FirstByJoss
    '''
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:

        start = (self._rounds_played == 0)
        self._rounds_played += 1

        if start:
            action = Actions.C
        elif (observation[1][-1] == Actions.D or observation[2][-1] == Actions.D):
            action = Actions.D  
        else: 
            action = int(random.random() < 0.1)

        return action
    

class Davis(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/axelrod_first.py FirstByDavis
    '''
    def __init__(self, name=None) -> None:
        super().__init__(name)
        self._rounds_to_cooperate = 10

    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
    
        if (
            self._rounds_played > self._rounds_to_cooperate and
            (observation[1][-1] == 1 or observation[2][-1] == 1) and not self._triggered
        ):
            self._triggered = True
        
        if self._triggered: 
            action = Actions.D
        else:
            action = Actions.C
    
        self._rounds_played += 1

        return action

    def reset(self):
        super().reset()
        self._triggered = False


class AverageCopier(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/averagecopier.py
    '''
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        
        if self._rounds_played == 0: 
            action = Actions.C
        else:
            opp1_p_coop = (self._rounds_played - sum(observation[1,:])) / self._rounds_played
            opp2_p_coop = (self._rounds_played - sum(observation[2,:])) / self._rounds_played

            action = int(random.random() < (opp1_p_coop + opp2_p_coop) / 2)
    
        self._rounds_played += 1

        return action


class Proposer(Strategy):

    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        
        if self._rounds_played & (self._rounds_played - 1) == 0:
            action = Actions.C
        elif observation[1][-1] == Actions.D or observation[2][-1] == Actions.D: 
            action = Actions.D
        else:
            action = Actions.C

        self._rounds_played += 1

        return action
    

class Stalker(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/stalker.py
    '''
    def __init__(self, name=None) -> None:
        super().__init__(name)
        self._good_score = PAYOFF_MATRIX[0][0][0]
        self._bad_score = PAYOFF_MATRIX[1][1][1]
    
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
    
        if self._rounds_played == 0:
            action = Actions.C
        else:
            self._score += PAYOFF_MATRIX[observation[0][-1]][observation[1][-1]][observation[2][-1]]
            avg_score = self._score / self._rounds_played

            if avg_score > self._good_score:
                action = Actions.D
            elif avg_score > (self._good_score + self._bad_score) / 2:
                action = Actions.C
            elif avg_score > self._bad_score:
                action = Actions.D
            else:
                action = random.randint(Actions.C, Actions.D)
        
        self._rounds_played += 1

        return action
    
    def reset(self):
        super().reset()
        self._score = 0


class BetterAndBetter(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/better_and_better.py
    '''
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        action = int(random.random() < self._rounds_played / 1000)

        self._rounds_played += 1

        return action


class Shubik(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/axelrod_first.py FirstByShubik
    '''
    def retaliate(self):
        self._retaliation_remaining -= 1
        if self._retaliation_remaining == 0: self._retaliating = False
        return Actions.D
    
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        start = (self._rounds_played == 0)
        self._rounds_played += 1

        if start: return Actions.C

        if self._retaliating:
            return self.retaliate()
        
        if observation[0][-1] == Actions.C and (observation[1][-1] == Actions.D or observation[2][-1] == Actions.D):
            self._retaliating = True
            self._retaliation_length += 1
            self._retaliation_remaining = self._retaliation_length
            return self.retaliate()
        
        return Actions.C
    
    def reset(self):
        super().reset()
        self._retaliating = False
        self._retaliation_length = 0
        self._retaliation_remaining = 0


class SoftTullock(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/axelrod_first.py FirstByTullock
    '''
    def __init__(self, name=None) -> None:
        super().__init__(name)
        self._rounds_to_coop = 11
    
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:
        
        if self._rounds_played < self._rounds_to_coop:
            action = Actions.C
        else:
            opp1_p_coop_10 = (self._rounds_to_coop - 1 - sum(observation[1, -self._rounds_to_coop + 1:])) / (self._rounds_to_coop - 1)
            opp2_p_coop_10 = (self._rounds_to_coop - 1 - sum(observation[2, -self._rounds_to_coop + 1:])) / (self._rounds_to_coop - 1)
            p_coop = max(0, (opp1_p_coop_10 + opp2_p_coop_10) / 2 - 0.1)

            action = int(random.random() > p_coop)

        self._rounds_played += 1

        return action


class ToughTullock(Strategy):
    '''
    Adapted from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/axelrod_first.py FirstByTullock
    '''
    def __init__(self, name=None) -> None:
        super().__init__(name)
        self._rounds_to_coop = 11
    
    def play(
        self, 
        observation: np.array,
        reward: int=None
    ) -> Actions:

        if self._rounds_played < self._rounds_to_coop: 
            action = Actions.C
        else:
            opp1_p_coop_10 = (self._rounds_to_coop - 1 - sum(observation[1, -self._rounds_to_coop + 1:])) / (self._rounds_to_coop - 1)
            opp2_p_coop_10 = (self._rounds_to_coop - 1 - sum(observation[2, -self._rounds_to_coop + 1:])) / (self._rounds_to_coop - 1)
            p_coop = max(0, min(opp1_p_coop_10, opp2_p_coop_10) - 0.1)

            action = int(random.random() > p_coop)
        
        self._rounds_played += 1

        return action