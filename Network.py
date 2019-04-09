from Node import *
from random import random, sample, randint
import numpy as np
import scipy.special

def softmax(values):
    # print("Values: ", values)
    # implementation of the softmax function
    e_x = np.exp(values - np.max(values))
    v = e_x / e_x.sum()
    # v = scipy.special.softmax(values)
    return  [float(i) for i in v]

#---------------------------------------------

class Layer:
    def __init__(self, prevLayer, nextLayer, numNodes):
        self.nodes = []     # list of all nodes in the layer
        self.connections = []   # list of all connections from within the layer
        self.numNodes = 0
        self.numConnections = 0

        # the next and previous layers
        self.next = nextLayer
        self.prev = prevLayer
        for i in range(numNodes):
            self.addNode()

    def addNode(self):
        temp = Node()  # type 0 for input
        self.nodes.append(temp)
        self.numNodes +=1
        temp.layer = self


#---------------------------------------------

class Network:
    def __init__(self, inputNum, outputNum):
        self.nodes = [] #list of all nodes in the genome.
        self.connections = [] #list of all connections(edges) in the genome.

        # for safekeeping, We will have separate lists to include the input and output nodes
        # They will also be stored in the self.nodes list

        self.numNodes = 0
        self.numConnections = 0

        # set up input and output like the head and tail of a linked list
        self.inputLayer = None
        self.outputLayer = None

        self.fitness = 0
        self.adjustedFitness = 0
        # this function initializes the first nodes and connections
        self.__setInitialNodes(inputNum, outputNum)

    # def reset(self):
    #     # reset all recurrent values so that the network does not perform poorly if played a second time?
    #     return
    #==========================================================================
    # private internal functions
    '''
    Check if the input node is before the output node
    If the input node is in a layer before the output, the connection will be recurrent
    '''
    def __checkRecurrent(self, inNode, outNode):
        temp = inNode.layer.next
        # loop through all layers after the input node's layer
        # if the output nodes layer is found, it is a normal connection
        # otherwise it is recurrent
        while temp != None:
            if temp == outNode.layer:
                return False
            temp = temp.next
        return True


    """creates the initial nodes with the total number of input and output nodes
            All nodes in the input layer will be connected to the output layer"""
    def __setInitialNodes(self, inputNum, outputNum):
        # set up the input and output nodes
        self.inputLayer = Layer(None,None,inputNum)
        self.outputLayer = Layer(self.inputLayer,None,outputNum)
        self.inputLayer.next = self.outputLayer
        self.numNodes += inputNum + outputNum
        # update self lists
        self.nodes += self.inputLayer.nodes + self.outputLayer.nodes
        # loop through all initial nodes and add connections between them
        # todo is it best to start with fully connected, or mutate connections as necessary?
        for inp in self.inputLayer.nodes:
            for out in self.outputLayer.nodes:
                self.addConnection(inp,out)

    # ==========================================================================

    # ==========================================================================

    # Network building functions
    def addConnection(self, input, output, weight=None, stdev=1):
        # TODO is a stdev of 1 the best value?
        # todo we need to somehow confirm that the current connection does not already exist
        #    (this can probably be done in other functions, only run this function if we already are sure)

        # add a connection between two nodes, with either a set weight or a random one
        if weight is None:
            # use a normal distribution to create a weight. This will almost always run
            connection = Connection(input, output, np.random.normal(scale=stdev))
        else:
            connection = Connection(input, output, weight)

        self.numConnections+=1
        # check if the new connection is recurrent, if so set the necessary boolean
        connection.isRecurrent = self.__checkRecurrent(input, output)
        #self.connections.append(connection)
        input.layer.connections.append(connection)

        # add the connection to the total list of connections
        self.connections.append(connection)


    def addNode(self, layer, iNum=None):
        # create the temporary node, then add it to all necessary locations in the network and its layer
        temp = Node(iNum)
        temp.layer = layer
        self.nodes.append(temp)
        self.numNodes += 1
        layer.numNodes += 1
        layer.nodes.append(temp)

    '''
    Adds a hidden layer after the given prevLayer. This layer will have n nodes
    '''
    def addLayer(self,prevLayer,numNodes=0):
        # the previous layer cannot be the output layer
        assert prevLayer != self.outputLayer

        # create the new layer
        newLayer = Layer(prevLayer,prevLayer.next,numNodes)
        # reconnect the other layers
        prevLayer.next.prev = newLayer
        prevLayer.next = newLayer
        # update the lists in the network
        self.numNodes += numNodes
        self.nodes += newLayer.nodes

    # ==========================================================================
    # ==========================================================================

    '''
    evaluate the entire network and return the output vector with softmax
    '''
    # todo we should allow sigmoid for games that allow pressing multiple buttons (any value above some threshold could be set as true)
    def evaluate(self, inputVector):
        # the input vector should have the same size as the input layer
        assert self.inputLayer.numNodes == len(inputVector)
        # set the values for the input layer
        for i, node in enumerate(self.inputLayer.nodes):
            node.value = inputVector[i]
            node.isEvaluated = True
        # print("input layer: ", [node.value for node in self.inputLayer.nodes])
        # loop through each layer and evaluate them one at a time
        current = self.inputLayer.next
        currentConnections = [c for c in self.connections if c.isEnabled]
        while(current!=None):
            # loop through every node, evaluating it if possible
            # we can use the node.evaluate() function because it checks if the node can be evaluated
            for node in current.nodes:
                node.evaluate(currentConnections)
            current = current.next

        # set the prevValue used for recurrent connections
        for node in self.nodes:
            node.prevValue = node.value

        outputVector = softmax([outNode.value for outNode in self.outputLayer.nodes])

        return outputVector

    def getAllConnections(self):
        allConnections = []
        currentLayer = self.inputLayer
        while(currentLayer != None):
            allConnections += currentLayer.connections
            currentLayer = currentLayer.next
        return allConnections

    def getAllNodes(self):
        allNodes = []
        currentLayer = self.inputLayer
        while (currentLayer != None):
            allNodes += currentLayer.nodes
            currentLayer = currentLayer.next
        return allNodes

    def print(self):
        # print details on the network
        print("Network:")
        print("{} Input nodes: ".format(len(self.inputLayer.nodes)))
        for node in self.inputLayer.nodes:
            print("node id: {}".format(node.id))

        print("{} Input layer connections: ".format(len(self.inputLayer.connections)))
        for connection in self.inputLayer.connections:
            print("connection ({0}) {1} -> {2}".format(connection.weight, connection.input.id,
                                                              connection.output.id))

        # print all hidden layers
        hiddenLayer = self.inputLayer.next
        i = 0
        while(hiddenLayer!=self.outputLayer):
            print("\nHidden layer {}: ".format(i))
            print("{} nodes: ".format(len(hiddenLayer.nodes)))
            for node in hiddenLayer.nodes:
                print("node id: {}".format(node.id))
            print("{} connections in this hidden layer: ".format(len(hiddenLayer.connections)))
            for connection in hiddenLayer.connections:
                print("connection ({0}) {1} -> {2}".format(connection.weight, connection.input.id,
                                                                  connection.output.id))
            i += 1
            hiddenLayer = hiddenLayer.next
        #print("{} hidden layer(s)".format(i))


        print("\n{} Output nodes: ".format(len(self.outputLayer.nodes)))
        for node in self.outputLayer.nodes:
            print("node id: {}".format(node.id))

        print("{} output layer connections: ".format(len(self.outputLayer.connections)))
        for connection in self.outputLayer.connections:
            print("connection ({0}) {1} -> {2}".format(connection.weight, connection.input.id,
                                                              connection.output.id))
        print("\n")

    # ==========================================================================

    # ==========================================================================
    # check if a connection exists between two nodes, including disabled connections
    def doesConnectionExist(self, input, output):
        for connection in self.connections:
            if connection.input == input and connection.output == output:
                return True
        return False

    def mutateConnections(self):
        # mutate update random connection weight
        # how many weights should we update? I choose a random number between 1 and 5% of all existing connections
        numMutations = randint(1, max(1, int(0.05 * len(self.connections))))
        # choose a random weight(s) and reassign them
        randIndices = sample(range(len(self.connections)), numMutations)
        # print("mutated connection ids: ", [self.connections[i].id for i in randIndices])
        for i in randIndices:
            self.connections[i].updateWeight()

    def mutateAddConnection(self):
        # choose a random node
        inputNode = self.nodes[randint(0, len(self.nodes)-1)]
        # find a random node that it is not already connected to, and add the connection
        unorderedNodes = sample(self.nodes, len(self.nodes))
        for outputNode in unorderedNodes:
            # do not make connections to the input layer, this results in an error if done
            if outputNode.layer != self.inputLayer:
                if not self.doesConnectionExist(inputNode, outputNode):
                    # if this code is reached, a new connection is possible with the given input and output nodes
                    # print("Added connection {0} -> {1}".format(inputNode.id, outputNode.id))
                    self.addConnection(inputNode, outputNode)
                    # add the connection and exit function
                    return
        # if the node is already connected to all other nodes, exit function
        return

    def mutateRemoveConnection(self):
        # choose a random connection and either remove or re-add it (isEnabled = !isEnabled)
        connectionIndex = randint(0, len(self.connections)-1)
        self.connections[connectionIndex].isEnabled = not self.connections[connectionIndex].isEnabled
        # print("connection: {0} isEnabled set to {1}".format(connectionIndex, self.connections[connectionIndex].isEnabled))

    def mutateAddNode(self):
        # choose a random non-recurrent connection and set its isEnabled to be disabled
        nonRecurrents = [connection for connection in self.connections if not connection.isRecurrent and connection.isEnabled]
        if (len(nonRecurrents) == 0):
            print("Error mutateAddNode: There are no nonRecurrents in the network")
            return
        mutConnection = nonRecurrents[randint(0, len(nonRecurrents)-1)]
        mutConnection.isEnabled = False

        # check if there is an existing layer between the two nodes
        # if there is no layer between the two, add a new layer after the input
        if mutConnection.input.layer.next is mutConnection.output.layer:
            self.addLayer(mutConnection.input.layer)

        # if there is at least one layer between the two nodes, add the third node in the first layer after input
        self.addNode(mutConnection.input.layer.next)
        centerNode = self.nodes[-1]
        # then add connections to the new node
        self.addConnection(mutConnection.input, centerNode, weight=1)
        self.addConnection(centerNode, mutConnection.output, weight=mutConnection.weight)





    def mutate(self, p_mutate_weights=0.98, p_add_connection=0.09, p_remove_connection=0.05, p_add_node=0.01):
        # mutate update random connection weight(s)
        if random() < p_mutate_weights:
            self.mutateConnections()
        # mutate add connection
        if random() < p_add_connection:
            self.mutateAddConnection()
        # mutate remove connection
        if random() < p_remove_connection:
            self.mutateRemoveConnection()
        # mutate add node
        if random() < p_add_node:
            self.mutateAddNode()

        # re sort the connection list and node list for future use
        self.connections = sorted(self.connections, key=lambda connection: (connection.input.id, connection.output.id))
        return

    def setAdjustedFitness(self, speciesSize):
        # technically we are supposed to divide by the number of networks with distance below delta_t
        # however I will assume that all networks below distance delta_t are already in the species
        self.adjustedFitness = self.fitness / speciesSize