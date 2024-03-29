# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state,action) in self.qvalues:
            return self.qvalues[(state,action)]
        else:
            return 0.0


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        v = float("-infinity")
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        else:
            #print "legalActions:", legalActions
            for action in legalActions:
             #   print "action:", action
                qVal = self.getQValue(state,action)
              #  print "qVal:", qVal
                if qVal > v:
                    v = qVal
               #     print "new v:", v
            return v

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        v = float("-infinity")
        qAct = None
        if not self.getLegalActions(state):
            return None
        else:
            for action in self.getLegalActions(state):
                qVal = self.getQValue(state,action)
                if qVal > v:
                    v = qVal
                    qAct = action
                elif qVal == v:
                    qAct = random.choice([qAct,action])
            return qAct

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if legalActions:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.qvalues[(state,action)] = (1-self.alpha)*self.qvalues[(state,action)]+self.alpha*(reward + self.discount*self.getValue(nextState))
        #if self.getLegalActions(nextState):
            #v = float("-infinity")
            #for nextAction in self.getLegalActions(nextState):
                #print "nextAction:", nextAction
                #qVal = self.qvalues[(nextState,nextAction)]
                #print "qVal:", qVal
                #if qVal > v:
                    #v = qVal
                    #print "v:", v
        #print "prev qvalue:", self.qvalues[(state,action)]
        #print "nextState qvalue:", self.getValue(nextState)
        #print "reward:", reward
        #newQValue = (1-self.alpha)*self.qvalues[(state,action)]+self.alpha*int(reward+self.discount*self.getValue(nextState))
        #print "newQValue:", newQValue
        #self.qvalues[(state,action)] = newQValue
        #print "self.qvalues[(state,action)]:", self.qvalues[(state,action)]
        #Q(s,a) = (1-self.alpha)Q(s,a) + (self.alpha)[R(s,a,s')+ self.discount*max_a'Q(s',a')]
        #tmp1 += prob*(self.mdp.getReward(state,action,nextS) + self.discount*tmp[nextS])

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
        
    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        qval = 0
        features = self.featExtractor.getFeatures(state,action)
        for feature in features:
            qval += features[feature]*self.weights[feature]
        return qval


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state,action)
        difference = (reward + self.discount*self.getValue(nextState)) - self.getQValue(state,action) #self.getValue since it returns Max Qvalue
        for feature in features:
            self.weights[feature] = self.weights[feature] + self.alpha*difference*features[feature]



    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
