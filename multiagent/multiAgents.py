# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        #raw_input()
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
        #Eval(s) = w1f1(s) + w2f2(s) + ... + wnfn(s)
        #in general, go toward food
        #if newScaredTimes > 1 or 0, go eat the dots
        #   else, stay away from ghosts
        #action away from ghost if < 3 away
        print "successorGameState:", successorGameState
        print "newPos:", newPos
        print "newFoodCount:", newFood.count(True)
        print "newGhostStates:", newGhostStates
        print "newScaredTimes:", newScaredTimes
        print "ghostPositions:", successorGameState.getGhostPositions()
        #return successorGameState.getScore()
        #return successor with max(eval(s))

        foodCount = -0.1*newFood.count(True)
        foodDistance = []
        foodList = newFood.asList(True)
        ghostDistance = []
        if foodList:
            for f in foodList:
                foodDistance.append(((newPos[0]-f[0])**2+(newPos[1]-f[1])**2)**0.5)
        else:
            foodDistance.append(1)
        for g in successorGameState.getGhostPositions():
            ghostDistance.append(((newPos[0]-g[0])**2+(newPos[1]-g[1])**2)**0.5)
        evalFunc = 1/(min(foodDistance)+1) - 1/(min(ghostDistance)+1) + newScaredTimes[0] + successorGameState.getScore() - foodCount
        print "(",action,",",evalFunc,")"
        return evalFunc

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
        """
        "*** YOUR CODE HERE ***"
        #agentList = []
        #if not agentList:
        #    agentList.append(0)
        #    a = 1
        #    while not len(agentList) == numAgents:
        #        agentList.append(a)
        #        a += 1
        #print "agentList:", agentList
        #print gameState.getLegalActions(0)
        #v = []
        #for action in gameState.getLegalActions(0):
        #    v.append(gameState.generateSuccessor(0, action))
        #print v
        #for agentIndex in agentList:
        #    value(gameState, agentIndex)
        #    depth -= 1
        #def value(state):
        #    if depth == 0 or gameState.isWin() or gameState.isLose():
        #        return self.evaluationFunction(gameState)
        #    elif not depth == 0:
        #        return minValue(state)
        numAgents = gameState.getNumAgents()
        print "numAgents:", numAgents
        depth = self.depth
        print "depth:", depth
        def minimizer(state, agentIndex, depth):
            inf = float("infinity")
            v = inf
            bestAction = None
            if depth == 0 or state.isWin() or state.isLose():
                print "ok"
                evalFunc = self.evaluationFunction(state)
                print "evalFunc:", evalFunc
                return (evalFunc,0)
            else:
                agentActions = state.getLegalActions(agentIndex)
                print "minagentActions:", agentActions
                if agentIndex == numAgents-1:
                    depth -= 1
                    for action in agentActions:
                        print "minaction:", action
                        print "depth:", depth
                        actionValue = maximizer(state.generateSuccessor(agentIndex,action),depth)[0]
                        print "minmaxactionValue:", actionValue
                        if actionValue < v:
                            v = actionValue
                            print "minv:", v
                            bestAction = action
                else:
                    for action in agentActions:
                        print "min2action:", action
                        actionValue = minimizer(state.generateSuccessor(agentIndex,action),agentIndex+1,depth)[0]
                        print "minminactionValue:", actionValue
                        if actionValue < v:
                            v = actionValue
                            bestAction = action
                print "min(v,bestAction):", (v,bestAction)
                return (v,bestAction)
        def maximizer(state, depth):
            negInf = float("-infinity")
            v = negInf
            bestAction = None
            if depth == 0 or state.isWin() or state.isLose():
                print "ok"
                evalFunc = self.evaluationFunction(state)
                print "evalFunc:", evalFunc
                return (evalFunc,0)
            else:
                agentActions = state.getLegalActions(0)
                print "maxagentActions:", agentActions
                for action in agentActions:
                    print "maxaction:", action
                    actionValue = minimizer(state.generateSuccessor(0,action),1,depth)[0]
                    print "maxactionValue:", actionValue
                    if actionValue > v:
                        v = actionValue
                        print "maxv:", v
                        bestAction = action
                print "max(v,bestAction):", (v,bestAction)
                return (v,bestAction)
        pacAction = maximizer(gameState, depth)[1]
        print "Pacman's action:", pacAction
        return pacAction

            #if action is > than previous, save

        #def maxValue(state):
        #    negInf = float("-infinity")
        #    v = [negInf]
        #    pacActions = gameState.getLegalActions(0)
        #    pacSuccessors = []
        #    for action in pacActions:
        #        pacSuccessors.append((action, gameState.generateSuccessor(0, action)))
        #    for act,suc in pacSuccessors:
        #        minV = minValue(suc)
        #        v.append(minV)
        #        print "v:", v
        #    depth -= 1
        #    return actDict[max(v)]
        #def minValue(state):
        #    inf = float("infinity")
        #    v = [inf]
        #    value = []
        #    if len(agentList) > 2
        #        for agentIndex in agentList:
        #            if not agentIndex == 0:
        #                for action in gameState.getLegalActions(agentIndex):
        #                    s = gameState.generateSuccessor(agentIndex, action)
        #                    if depth == 0 or gameState.isWin() or gameState.isLose():
        #                        v.append(self.evaluationFunction(gameState))
        #        value.append(min(v))
        #    if not depth == 0:
        #        maxValue()
        #    return min(value)

        #return maxValue(gameState)

        #    legalActions = gameState.getLegalActions(agentIndex)
        #    for action in legalActions:
        #        successorState = gameState.generateSuccessor(agentIndex, action)
        #value(gameState)
        #   if not gameState == *terminalState*:
        #       if *maxAgent* agentIndex==0:
        #           inf = float("-infinity")
        #           v = [inf]
        #           for s in successorState:
        #               v.append(*value(s)*)
        #           return max(v)
        #       if *minAgent* agentIndex>0:
        #           negInf = float("infinity")
        #           v = [negInf]
        #           for s in successorState:
        #               v.append(*value(s)*)
        #           return min(v)
        #   else:
        #       return *utility* (part of gameState? Evalfunc of gameState?)


        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        print "numAgents:", numAgents
        depth = self.depth
        print "depth:", depth
        inf = float("infinity")
        negInf = float("-infinity")
        a = negInf
        b = inf
        def minimizer(state, agentIndex, depth, a, b):
            inf = float("infinity")
            v = inf
            bestAction = None
            if depth == 0 or state.isWin() or state.isLose():
                print "ok"
                evalFunc = self.evaluationFunction(state)
                print "evalFunc:", evalFunc
                return (evalFunc,0)
            else:
                agentActions = state.getLegalActions(agentIndex)
                print "minagentActions:", agentActions
                if agentIndex == numAgents-1:
                    depth -= 1
                    for action in agentActions:
                        print "minaction:", action
                        print "depth:", depth
                        actionValue = maximizer(state.generateSuccessor(agentIndex,action),depth,a,b)[0]
                        print "minmaxactionValue:", actionValue
                        if actionValue < v:
                            v = actionValue
                            print "minv:", v
                            bestAction = action
                            print bestAction
                        if v < a:
                            print "min2v:", v
                            return (v,bestAction)
                        else:
                            b = min(b,v)
                            print b
                else:
                    for action in agentActions:
                        print "min2action:", action
                        actionValue = minimizer(state.generateSuccessor(agentIndex,action),agentIndex+1,depth,a,b)[0]
                        print "minminactionValue:", actionValue
                        if actionValue < v:
                            v = actionValue
                            bestAction = action
                        if v < a:
                            print "min3v:", v
                            return (v,bestAction)
                        else:
                            b = min(b,v)
                            print b
                print "min(v,bestAction):", (v,bestAction)
                return (v,bestAction)

        def maximizer(state, depth, a, b):
            negInf = float("-infinity")
            v = negInf
            bestAction = None
            if depth == 0 or state.isWin() or state.isLose():
                print "ok"
                evalFunc = self.evaluationFunction(state)
                print "evalFunc:", evalFunc
                return (evalFunc,0)
            else:
                agentActions = state.getLegalActions(0)
                print "maxagentActions:", agentActions
                for action in agentActions:
                    print "maxaction:", action
                    actionValue = minimizer(state.generateSuccessor(0,action),1,depth,a,b)[0]
                    print "maxactionValue:", actionValue
                    if actionValue > v:
                        v = actionValue
                        print "maxv:", v
                        bestAction = action
                    if v > b:
                        return (v,bestAction)
                    else:
                        a = max(a,v)
                print "max(v,bestAction):", (v,bestAction)
                return (v,bestAction)
        pacAction = maximizer(gameState, depth, a, b)[1]
        print "Pacman's action:", pacAction
        return pacAction

        util.raiseNotDefined()

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
        numAgents = gameState.getNumAgents()
        print "numAgents:", numAgents
        depth = self.depth
        print "depth:", depth
        def expectimizer(state, agentIndex, depth):
            inf = float("infinity")
            v = inf
            bestAction = None
            successorList = []
            if depth == 0 or state.isWin() or state.isLose():
                evalFunc = self.evaluationFunction(state)
                print "evalFunc:", evalFunc
                return evalFunc
            else:
                agentActions = state.getLegalActions(agentIndex)
                print "minagentActions:", agentActions
                print "len:", len(agentActions)
                p = float(1.0/len(agentActions))
                if agentIndex == numAgents-1:
                    depth -= 1
                    for action in agentActions:
                        print "minaction:", action
                        print "depth:", depth
                        actionValue = maximizer(state.generateSuccessor(agentIndex,action),depth)[0]
                        print "minmaxactionValue:", actionValue
                        pValue = p*actionValue
                        print "pValue:", pValue
                        successorList.append(pValue)
                    v = float(sum(successorList)/len(agentActions))
                    print "minmaxv:", v
                    return v
                else:
                    for action in agentActions:
                        print "min2action:", action
                        actionValue = expectimizer(state.generateSuccessor(agentIndex,action),agentIndex+1,depth)
                        print "minminactionValue:", actionValue
                        pValue = p*actionValue
                        print "pValue:", pValue
                        successorList.append(pValue)
                    v = float(sum(successorList)/len(agentActions))
                    print "minminv:", v
                    return v
        def maximizer(state, depth):
            negInf = float("-infinity")
            v = negInf
            bestAction = None
            if depth == 0 or state.isWin() or state.isLose():
                evalFunc = self.evaluationFunction(state)
                print "evalFunc:", evalFunc
                return (evalFunc,0)
            else:
                agentActions = state.getLegalActions(0)
                print "maxagentActions:", agentActions
                for action in agentActions:
                    print "maxaction:", action
                    actionValue = expectimizer(state.generateSuccessor(0,action),1,depth)
                    print "maxactionValue:", actionValue
                    if actionValue > v:
                        v = float(actionValue)
                        print "maxv:", v
                        bestAction = action
                print "max(v,bestAction):", (v,bestAction)
                return (v,bestAction)
        pacAction = maximizer(gameState, depth)[1]
        print "Pacman's action:", pacAction
        return pacAction
        
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacActions = currentGameState.getLegalActions(0)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodCount = -0.1*newFood.count(True)
    foodDistance = []
    foodList = newFood.asList(True)
    ghostDistance = []
    if foodList:
        for f in foodList:
            foodDistance.append(((newPos[0]-f[0])**2+(newPos[1]-f[1])**2)**0.5)
    else:
        foodDistance.append(1)
    for g in currentGameState.getGhostPositions():
        ghostDistance.append(((newPos[0]-g[0])**2+(newPos[1]-g[1])**2)**0.5)
    evalFunc = 1/(min(foodDistance)+1) - 1/(min(ghostDistance)+1) + newScaredTimes[0] + currentGameState.getScore() - foodCount
    return evalFunc

#isWin()/isLose()
#if scared, go toward ghosts
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

