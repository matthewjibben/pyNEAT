import numpy as np

nodeInnovationNum = 0

class Connection:
    def __init__(self, input, out, w):
        self.input = input #expects Node type.
        self.output = out #expects Node type.
        self.weight = w
        self.isEnabled = True #on or off (disabled or enabled)

        self.isRecurrent = False

        # print("connection constructor")

    def updateWeight(self, stdev=1):
        self.weight = np.random.normal(scale=stdev)

class Node:
    def __init__(self, iNum=None):
        # if the ID is given set it to that. Otherwise use the global innovation number
        if iNum == None:
            global nodeInnovationNum
            self.id = nodeInnovationNum
            nodeInnovationNum += 1
        else:
            self.id = iNum


        self.value = 0
        self.prevValue = 0
        self.layer = None # expects a layer class where the node is contained

    # checks if a node can be evaluated by checking the connections
    # if it can, it evaluates the node
    def evaluate(self, connections):
        # this function should calculate the value of the specific node
        # this can be done by finding the sum of all connections to the node, and using a sigmoid or tanh

        # if we use self.value it may overwrite the value needed by a recurrent connection to itself
        newValue = 0

        # first, add the values for all recurrent connections
        # if we dont do this first, important recurrent connection values may be overwritten
        for connection in connections:
            if connection.output is self and connection.isRecurrent:
                # the default value is that recurrent connections will add 0
                # once a node has been evaluated once, it can then add value
                newValue += connection.input.prevValue * connection.weight
                #print("recurrent connections eval========================", connection.input.value * connection.weight)


        # get all of the normal connections to the current node
        backConnections = [connection for connection in connections
                           if connection.output is self and not connection.isRecurrent]

        # check if all backconnections have been evaluated. If so, evaluate
        # todo can this be optimized to only get the input values once?
        #if False not in [connection.input.isEvaluated for connection in backConnections]:
        # set the value of the node
        value = sum([connection.input.value * connection.weight
                     for connection in backConnections])
        newValue += value
        self.value = newValue # todo should this be tanh, relu, or sigmoid? (or return value)
