# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        #print "states:", states
        tmp = util.Counter()
        #print "iterations:", self.iterations
        for i in range(self.iterations):
            #print "iteration number:", i
            tmp = self.values.copy()
            for state in states:
                v = float("-infinity")
                #print "state:", state
                actions = self.mdp.getPossibleActions(state)
                #print "actions:", actions
                for action in actions:
                    tmp1=0
                    #print "action:", action
                    #print self.mdp.getTransitionStatesAndProbs(state, action)
                    for (nextS,prob) in self.mdp.getTransitionStatesAndProbs(state,action):
                        #if not self.mdp.isTerminal(nextS):
                        #print "(nextS,prob):", (nextS,prob)
                        #print "tmp[nextS]:", tmp[nextS]
                        #print "prob:", prob
                        #print "reward:", self.mdp.getReward(state,action,nextS)
                        #print "discount:", self.discount
                        #print "prev tmp1:", tmp1
                        tmp1 += prob*(self.mdp.getReward(state,action,nextS) + self.discount*tmp[nextS])
                        #print "tmp1:", tmp1
                    if tmp1 > v:
                        v = tmp1
                  #  tmp2[state] = v
                  #  print "value:", tmp2[state]
                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                else:
                    self.values[state] = v
                    #print "v:", v
        print "end values:", self.values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        #print "getValue:", self.values[state]
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qval = 0
        for (nextS,prob) in self.mdp.getTransitionStatesAndProbs(state, action):
             qval += prob*(self.mdp.getReward(state,action,nextS) + self.discount*(self.values[nextS]))
        return qval
       

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #states = self.mdp.getStates()
        #for state in states:
        if self.mdp.isTerminal(state):
            return None    
        else:
            v = float("-infinity")
            policy = None
            for action in self.mdp.getPossibleActions(state):
                qVal = self.getQValue(state, action)
                if  qVal > v:
                    v = qVal
                    policy = action
            #print "policy:", policy
            return policy
                      
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
