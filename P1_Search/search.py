# search.py
# ---------
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
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    visitedState = []   # Array of tuple [(5,5), (5,4), ...]
    result = []         # Array of Direction [s, w, s, ...]

    stateStack = util.Stack()
    stateStack.push((problem.getStartState(), [], 0))   # Store (state waiting for search, route, distance from start)

    while ~stateStack.isEmpty():
      stackPop = stateStack.pop()
      currentState = stackPop[0]      # tuple of int (0, 0)
      currentRoute = stackPop[1]      # array of Directions [s, w, s...]
      currentDistance = stackPop[2]   # int

      if problem.isGoalState(currentState):   # In case of Current is Goal
        result = currentRoute   # Current Route becomes result  
        break
      else:
        if currentState in visitedState:    # In case of Already Visited
          continue                          # Just Pass That!

        visitedState.append(currentState)   # Now we visit current state

        successors = problem.getSuccessors(currentState)   # Array of tuple [((5, 4), 'West', 1), ((), '', 0 ), ...]

        for idx in range(0, len(successors)):   # Loop For all successor
          successorState = successors[idx][0]
          successorDirection = successors[idx][1]
          successorDistance = successors[idx][2]

          stateStack.push((successorState , currentRoute + [successorDirection], currentDistance + successorDistance))
    pass
    return result
    # util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visitedState = []   # Array of tuple [(5,5), (5,4), ...]
    result = []         # Array of Direction [s, w, s, ...]

    stateQueue = util.Queue()
    stateQueue.push((problem.getStartState(), [], 0))   # Store (state waiting for search, route, distance from start)

    while ~stateQueue.isEmpty():
      queuePop = stateQueue.pop()
      currentState = queuePop[0]      # tuple of int (0, 0)
      currentRoute = queuePop[1]      # array of Directions [s, w, s...]
      currentDistance = queuePop[2]   # int

      if problem.isGoalState(currentState):   # In case of Current is Goal
        result = currentRoute   # Current Route becomes result  
        break
      else:
        if currentState in visitedState:    # In case of Already Visited
          continue                          # Just Pass That!

        visitedState.append(currentState)   # Now we visit current state

        successors = problem.getSuccessors(currentState)   # Array of tuple [((5, 4), 'West', 1), ((), '', 0 ), ...]

        for idx in range(0, len(successors)):   # Loop For all successor
          successorState = successors[idx][0]
          successorDirection = successors[idx][1]
          successorDistance = successors[idx][2]

          stateQueue.push((successorState , currentRoute + [successorDirection], currentDistance + successorDistance))
    pass
    return result
    # util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visitedState = []   # Array of tuple [(5,5), (5,4), ...]
    result = []         # Array of Direction [s, w, s, ...]

    stateQueue = util.PriorityQueue()
    stateQueue.push((problem.getStartState(), [], 0), 0)   # Store (state waiting for search, route, distance from start)

    while ~stateQueue.isEmpty():
      queuePop = stateQueue.pop()
      currentState = queuePop[0]      # tuple of int (0, 0)
      currentRoute = queuePop[1]      # array of Directions [s, w, s...]
      currentDistance = queuePop[2]   # int

      if problem.isGoalState(currentState):   # In case of Current is Goal
        result = currentRoute   # Current Route becomes result  
        break
      else:
        if currentState in visitedState:    # In case of Already Visited
          continue                          # Just Pass That!

        visitedState.append(currentState)   # Now we visit current state

        successors = problem.getSuccessors(currentState)   # Array of tuple [((5, 4), 'West', 1), ((), '', 0 ), ...]

        for idx in range(0, len(successors)):   # Loop For all successor
          successorState = successors[idx][0]
          successorDirection = successors[idx][1]
          successorDistance = successors[idx][2]

          stateQueue.push((successorState , currentRoute + [successorDirection], currentDistance + successorDistance), currentDistance + successorDistance)
    pass
    return result
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visitedState = []   # Array of tuple [(5,5), (5,4), ...]
    result = []         # Array of Direction [s, w, s, ...]

    startState = problem.getStartState()
    stateQueue = util.PriorityQueue()
    stateQueue.push((startState, [], 0), heuristic(startState, problem))   # Store (state waiting for search, route, distance from start)

    while ~stateQueue.isEmpty():
      queuePop = stateQueue.pop()
      currentState = queuePop[0]      # tuple of int (0, 0)
      currentRoute = queuePop[1]      # array of Directions [s, w, s...]
      currentDistance = queuePop[2]   # int

      if problem.isGoalState(currentState):   # In case of Current is Goal
        result = currentRoute   # Current Route becomes result  
        break
      else:
        if currentState in visitedState:    # In case of Already Visited
          continue                          # Just Pass That!

        visitedState.append(currentState)   # Now we visit current state

        successors = problem.getSuccessors(currentState)   # Array of tuple [((5, 4), 'West', 1), ((), '', 0 ), ...]

        for idx in range(0, len(successors)):   # Loop For all successor
          successorState = successors[idx][0]
          successorDirection = successors[idx][1]
          successorDistance = successors[idx][2]

          successorHeuristic = heuristic(successorState, problem)
          stateQueue.push((successorState , currentRoute + [successorDirection], currentDistance + successorDistance), currentDistance + successorDistance + successorHeuristic)
    pass
    return result
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
