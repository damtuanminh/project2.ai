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
        minFoodist = float("inf")
        for food in newFood.asList():
            minFoodist = min(minFoodist, manhattanDistance(newPos, food))

        # avoid ghost if too close
        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')
        # reciprocal
        return successorGameState.getScore() + 1.0/minFoodist
#        return successorGameState.getScore()

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

        def minimax(state, depth, agentIndex):
            if agentIndex == state.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return minimax(state, depth + 1, 0)
            else:
                moves = state.getLegalActions(agentIndex)
                if len(moves) == 0:
                    return self.evaluationFunction(state)
                next = (minimax(state.generateSuccessor(agentIndex, move), depth, agentIndex + 1) for move in moves)
                if agentIndex == 0:
                    return max(next)
                else:
                    return min(next)

        result = max(gameState.getLegalActions(0), key=lambda x: minimax(gameState.generateSuccessor(0, x), 1, 1))
        return result

    # def minimax(self, agent, depth, gameState):
    #     if gameState.isLose() or gameState.isWin() or depth == self.depth:
    #         return self.evaluationFunction(gameState)
    #     if agent == 0:  # maximize for pacman
    #         return max(self.minimax(1, depth, gameState.generateSuccessor(agent, action)) for action in
    #                    getLegalActionsNoStop(0, gameState))
    #     else:  # minimize for ghosts
    #         nextAgent = agent + 1  # get the next agent
    #         if gameState.getNumAgents() == nextAgent:
    #             nextAgent = 0
    #         if nextAgent == 0:  # increase depth every time all agents have moved
    #             depth += 1
    #         return min(self.minimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
    #                    getLegalActionsNoStop(agent, gameState))


    #
    #     return self.maxval(gameState, 0, 0)[0]
    #
    # def minimax(self, gameState, agentIndex, depth):
    #     if depth is self.depth * gameState.getNumAgents() \
    #             or gameState.isLose() or gameState.isWin():
    #         return self.evaluationFunction(gameState)
    #     if agentIndex is 0:
    #         return self.maxval(gameState, agentIndex, depth)[1]
    #     else:
    #         return self.minval(gameState, agentIndex, depth)[1]
    #
    # def maxval(self, gameState, agentIndex, depth):
    #     bestAction = ("max",-float("inf"))
    #     for action in gameState.getLegalActions(agentIndex):
    #         succAction = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),
    #                                   (depth + 1)%gameState.getNumAgents(),depth+1))
    #         bestAction = max(bestAction,succAction,key=lambda x:x[1])
    #     return bestAction
    #
    # def minval(self, gameState, agentIndex, depth):
    #     bestAction = ("min",float("inf"))
    #     for action in gameState.getLegalActions(agentIndex):
    #         succAction = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),
    #                                   (depth + 1)%gameState.getNumAgents(),depth+1))
    #         bestAction = min(bestAction,succAction,key=lambda x:x[1])
    #     return bestAction
#        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def minValue(gameState, agentID, depth, a, b):

            actionList = gameState.getLegalActions(agentID)  # Get the actions of the ghost
            if len(actionList) == 0:
                return (self.evaluationFunction(gameState), None)

            value = float("inf")
            bestAction = None

            for action in actionList:
                if (agentID == gameState.getNumAgents() - 1):
                    succ = maxValue(gameState.generateSuccessor(agentID, action), depth + 1, a, b)[0]
                else:
                    succ = minValue(gameState.generateSuccessor(agentID, action), agentID + 1, depth, a, b)[0]

                if (succ < value):
                    value, bestAction = succ, action

                if (value < a):
                    return (value, bestAction)

                b = min(b, value)

            return (value, bestAction)

        def maxValue(gameState, depth, a, b):
            actionList = gameState.getLegalActions(0)  # Get actions of pacman
            if len(actionList) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            value = -(float("inf"))
            bestAction = None
            for action in actionList:
                succ = minValue(gameState.generateSuccessor(0, action), 1, depth, a, b)[0]
                if (value < succ):
                    value, bestAction = succ, action

                if (value > b):
                    return (value, bestAction)

                a = max(a, value)

            return (value, bestAction)

        alpha = -(float("inf"))
        beta = float("inf")
        return maxValue(gameState, 0, alpha, beta)[1]


    #     return self.maxval(gameState, 0, 0, -float("inf"), float("inf"))[0]
    #
    # def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
    #     if depth is self.depth * gameState.getNumAgents() \
    #             or gameState.isLose() or gameState.isWin():
    #         return self.evaluationFunction(gameState)
    #     if agentIndex is 0:
    #         return self.maxval(gameState, agentIndex, depth, alpha, beta)[1]
    #     else:
    #         return self.minval(gameState, agentIndex, depth, alpha, beta)[1]
    #
    # def maxval(self, gameState, agentIndex, depth, alpha, beta):
    #     bestAction = ("max",-float("inf"))
    #     for action in gameState.getLegalActions(agentIndex):
    #         succAction = (action,self.alphabeta(gameState.generateSuccessor(agentIndex,action),
    #                                   (depth + 1)%gameState.getNumAgents(),depth+1, alpha, beta))
    #         bestAction = max(bestAction,succAction, key=lambda x:x[1])
    #
    #         # Prunning
    #         if bestAction[1] > beta: return bestAction
    #         else: alpha = max(alpha,bestAction[1])
    #
    #     return bestAction
    #
    # def minval(self, gameState, agentIndex, depth, alpha, beta):
    #     bestAction = ("min",float("inf"))
    #     for action in gameState.getLegalActions(agentIndex):
    #         succAction = (action,self.alphabeta(gameState.generateSuccessor(agentIndex,action),
    #                                   (depth + 1)%gameState.getNumAgents(),depth+1, alpha, beta))
    #         bestAction = min(bestAction,succAction, key=lambda x:x[1])
    #
    #         # Prunning
    #         if bestAction[1] < alpha: return bestAction
    #         else: beta = min(beta, bestAction[1])
    #
    #     return bestAction
    #     util.raiseNotDefined()

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

        def expectimax_search(state, agentIndex, depth):
            # if in min layer and last ghost
            if agentIndex == state.getNumAgents():
                # if reached max depth, evaluate state
                if depth == self.depth:
                    return self.evaluationFunction(state)
                # otherwise start new max layer with bigger depth
                else:
                    return expectimax_search(state, 0, depth + 1)
            # if not min layer and last ghost
            else:
                moves = state.getLegalActions(agentIndex)
                # if nothing can be done, evaluate the state
                if len(moves) == 0:
                    return self.evaluationFunction(state)
                # get all the minimax values for the next layer with each node being a possible state after a move
                next = (expectimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

                # if max layer, return max of layer below
                if agentIndex == 0:
                    return max(next)
                # if min layer, return expectimax values
                else:
                    l = list(next)
                    return sum(l) / len(l)

        # select the action with the greatest minimax value
        result = max(gameState.getLegalActions(0),
                     key=lambda x: expectimax_search(gameState.generateSuccessor(0, x), 1, 1))

        return result

        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    if newCapsules:
        closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
    else:
        closestCapsule = 0

    if closestCapsule:
        closest_capsule = -20 / closestCapsule
    else:
        closest_capsule = 200

    if closestGhost:
        ghost_distance = -2 / closestGhost
    else:
        ghost_distance = -500

    foodList = newFood.asList()
    if foodList:
        closestFood = min([manhattanDistance(newPos, food) for food in foodList])
    else:
        closestFood = 0

    return -1.5 * closestFood + ghost_distance - 20 * len(foodList) + closest_capsule


# Abbreviation
better = betterEvaluationFunction
