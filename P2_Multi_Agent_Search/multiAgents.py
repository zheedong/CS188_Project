# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print(legalMoves)
        # print(scores)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if not newFood.asList():
            return 100

        score = 1 / len(newFood.asList())

        ghostDistance = manhattanDistance(newPos, newGhostStates[0].getPosition())
        foodDistance =  manhattanDistance(newPos, newFood.asList()[0])

        if sum(newScaredTimes) == 0:
            for i in range(1, len(newGhostStates)):
                newGhostPos = newGhostStates[i].getPosition()
                curGhostDistance =  manhattanDistance(newPos, newGhostPos)
                if newScaredTimes[i] < 3:
                    ghostDistance = min(ghostDistance, curGhostDistance)
            if ghostDistance < 5:
                score -= 10000 
                score -= 1 / ghostDistance
            elif not ghostDistance == 0:
                score -= 0.0001 / ghostDistance

        for j in newFood.asList()[1:]:
            curFoodDistance =  manhattanDistance(newPos, j)
            foodDistance = min(curFoodDistance, foodDistance)
        if not foodDistance == 0:
            if not ghostDistance < 3:
                score += 0.0001 / foodDistance

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        pacman_legal_moves = gameState.getLegalActions(self.PACMAN_INDEX)
        value = -self.INF
        for pacman_move in pacman_legal_moves:
            cur_value = self._get_value(gameState.generateSuccessor(self.PACMAN_INDEX, pacman_move), 1, self.INIT_DEPTH)
            if value < cur_value:
                value = cur_value
                best_move = pacman_move
        return best_move

    INF = 100000000000
    PACMAN_INDEX = 0
    INIT_DEPTH = 0

    def _get_value(self, gameState, agent_index, current_search_depth):
        if self._check_is_terminal_state(gameState, current_search_depth):
            return self.evaluationFunction(gameState)
        elif agent_index == self.PACMAN_INDEX:
            return self._get_max_value(gameState, current_search_depth) 
        else:
            return self._get_min_value(gameState, agent_index, current_search_depth)

    def _check_is_terminal_state(self, gameState, current_search_depth):
        if current_search_depth == self.depth or gameState.isWin() or gameState.isLose():
            return True
        else:
            return False

    def _get_successor_of_state(self, gameState, agent_index):
        agent_legal_moves = gameState.getLegalActions(agent_index)
        state_successors = []
        for agent_move in agent_legal_moves:
            state_successors.append(gameState.generateSuccessor(agent_index, agent_move))
        return state_successors

    def _get_max_value(self, gameState, search_depth):
        value = -self.INF
        pacman_successors = self._get_successor_of_state(gameState, self.PACMAN_INDEX)
        for successor in pacman_successors:
            value = max(value, self._get_value(successor, 1, search_depth))         # 1 is the index of the ghost
        return value

    def _get_min_value(self, gameState, agent_index, search_depth):
        value = self.INF
        ghost_num = gameState.getNumAgents() - 1
        ghost_successors = self._get_successor_of_state(gameState, agent_index)
        if agent_index == ghost_num:
            for successor in ghost_successors:
                value = min(value, self._get_value(successor, self.PACMAN_INDEX, search_depth + 1))     # min layer finished
        else:
            for successor in ghost_successors:
                value = min(value, self._get_value(successor, agent_index + 1, search_depth))
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacman_legal_moves = gameState.getLegalActions(self.PACMAN_INDEX)
        value = -self.INF
        alpha = -self.INF
        beta = self.INF
        for pacman_move in pacman_legal_moves:
            successor = gameState.generateSuccessor(self.PACMAN_INDEX, pacman_move)
            cur_value = self._get_value(successor, 1, self.INIT_DEPTH, alpha, beta)
            if value < cur_value:
                value = cur_value
                best_move = pacman_move
            if value > beta:
                break
            alpha = max(alpha, value)
        return best_move

    INF = 100000000000
    PACMAN_INDEX = 0
    INIT_DEPTH = 0

    def _get_value(self, gameState, agent_index, current_search_depth, alpha, beta):
        if self._check_is_terminal_state(gameState, current_search_depth):
            return self.evaluationFunction(gameState)
        elif agent_index == self.PACMAN_INDEX:
            return self._get_max_value(gameState, current_search_depth, alpha, beta) 
        else:
            return self._get_min_value(gameState, agent_index, current_search_depth, alpha, beta)

    def _check_is_terminal_state(self, gameState, current_search_depth):
        if current_search_depth == self.depth or gameState.isWin() or gameState.isLose():
            return True
        else:
            return False

    def _get_max_value(self, gameState, search_depth, alpha, beta):
        value = -self.INF
        pacman_legal_moves = gameState.getLegalActions(self.PACMAN_INDEX)
        for pacman_move in pacman_legal_moves:
            successor = gameState.generateSuccessor(self.PACMAN_INDEX, pacman_move)
            value = max(value, self._get_value(successor, 1, search_depth, alpha, beta))         # 1 is the index of the ghost
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value

    def _get_min_value(self, gameState, agent_index, search_depth, alpha, beta):
        value = self.INF
        ghost_num = gameState.getNumAgents() - 1
        ghost_legal_moves = gameState.getLegalActions(agent_index)
        for ghost_move in ghost_legal_moves:
            successor = gameState.generateSuccessor(agent_index, ghost_move)
            if agent_index == ghost_num:
                value = min(value, self._get_value(successor, self.PACMAN_INDEX, search_depth + 1, alpha, beta))     # min layer finished
            else:
                value = min(value, self._get_value(successor, agent_index + 1, search_depth, alpha, beta))
            if value < alpha:
                return value
            beta = min(beta, value)
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        pacman_legal_moves = gameState.getLegalActions(self.PACMAN_INDEX)
        value = -self.INF
        for pacman_move in pacman_legal_moves:
            successor = gameState.generateSuccessor(self.PACMAN_INDEX, pacman_move)
            cur_value = self._get_value(successor, 1, self.INIT_DEPTH)
            if value < cur_value:
                value = cur_value
                best_move = pacman_move
        return best_move

    INF = 100000000000
    PACMAN_INDEX = 0
    INIT_DEPTH = 0

    def _get_value(self, gameState, agent_index, current_search_depth):
        if self._check_is_terminal_state(gameState, current_search_depth):
            return self.evaluationFunction(gameState)
        elif agent_index == self.PACMAN_INDEX:
            return self._get_max_value(gameState, current_search_depth) 
        else:
            return self._get_exp_value(gameState, agent_index, current_search_depth)

    def _check_is_terminal_state(self, gameState, current_search_depth):
        if current_search_depth == self.depth or gameState.isWin() or gameState.isLose():
            return True
        else:
            return False

    def _get_max_value(self, gameState, search_depth):
        value = -self.INF
        pacman_legal_moves = gameState.getLegalActions(self.PACMAN_INDEX)
        for pacman_move in pacman_legal_moves:
            successor = gameState.generateSuccessor(self.PACMAN_INDEX, pacman_move)
            value = max(value, self._get_value(successor, 1, search_depth,))         # 1 is the index of the ghost
        return value

    def _get_exp_value(self, gameState, agent_index, search_depth):
        value = self.INF
        ghost_num = gameState.getNumAgents() - 1
        ghost_legal_moves = gameState.getLegalActions(agent_index)
        for ghost_move in ghost_legal_moves:
            successor = gameState.generateSuccessor(agent_index, ghost_move)
            prob = 1 / len(ghost_legal_moves)               # Assume uniform choose 
            if agent_index == ghost_num:
                value += prob * self._get_value(successor, self.PACMAN_INDEX, search_depth + 1)     # min layer finished
            else:
                value += prob * self._get_value(successor, agent_index + 1, search_depth)
        return value       

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    '''
    pacman_position = currentGameState.getPacmanPosition()
    num_food = currentGameState.getNumFood()
    # score = currentGameState.getScore()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghost_state.scaredTimer for ghost_state in ghost_states]
    food_list = currentGameState.getFood().asList()

    try:
        return currentGameState.getScore() - 10 * sorted([manhattanDistance(food, pacman_position) for food in food_list])[0]
    except:
        return currentGameState.getScore()

    eps = 0.000000001
    score = 10 / (num_food + eps)
    pacman_food_man_distance = sorted([manhattanDistance(food, pacman_position) for food in food_list])
    half = pacman_food_man_distance[:num_food//2 + 1]
    score += 1 / (sum(half) + eps)
    #
    for i in range(0,len(ghost_states)):
        ghost_state = ghost_states[i]
        pacman_ghost_distance = manhattanDistance(pacman_position, ghost_state.getPosition())
        scared_time = ghost_state.scaredTimer
        if pacman_ghost_distance < 2:
            score -= 10000000
    #
    if num_food == 0:
        score = -100

    # print(score)
    # return score
    '''

    ##############
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    scaredGhost = [ghostState.scaredTimer for ghostState in ghosts]
    
    foodList = food.asList()
    foodDistance = [0]
    
    for f in foodList:
        foodDistance.append(manhattanDistance(position, f))

    ghostLocation = []
    for ghost in ghosts:
        ghostLocation.append(ghost.getPosition())
    
    ghostDistance = [0]
    
    for l in ghostLocation:
        ghostDistance.append(manhattanDistance(position, l))

    totalPowerPellets = len(currentGameState.getCapsules())

    score = 0
    eatenFood = len(food.asList(False))           
    totalTimesScared = sum(scaredGhost)
    totalGhostDistances = sum(ghostDistance)
    foodDistances = 0
    
    if sum(foodDistance) > 0:
        foodDistances = 1.0 / sum(foodDistance)
        
    score += currentGameState.getScore()  + foodDistances + eatenFood

    if totalTimesScared > 0:    
        score +=   totalTimesScared + (-1 * totalPowerPellets) + (-1 * totalGhostDistances)
    else :
        score +=  totalGhostDistances + totalPowerPellets
    
    return score
# Abbreviation
better = betterEvaluationFunction
