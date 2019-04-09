import numpy as np
import copy
import operator
import multiprocessing as mp
from random import choice
from GameObjects import *
from Network import *

# runs a network using the polecart game and assign a fitness score to it
def train(network):
    """game main loop."""
    playing = True
    # input will be the 4 polecart variables, output will be move left,right or no movement
    high = 0
    highNetwork = 0


    scores = []
    for i in range(15):
        playing = True
        polecart = Polecart()       # generate new random polecart

        # reset timer
        time = 0
        while (playing):
            time += 1
            inputVector = [polecart.x, polecart.xVel, polecart.theta, polecart.thetaVel]
            output = network.evaluate(inputVector)
            decision = np.argmax(output) - 1
            polecart.run(decision)

            # Check for gameover status
            playing = not polecart.hasLost()

            # if the game has run too long, we can quit
            if time >= 10000:
                # print("player won!")
                playing = False


        scores.append(time)


    network.fitness = np.average(scores)
    print("average score: ", network.fitness)

import gym

def trainAcrobot(network):

    # network must have a shape: Network(6, 3)
    env = gym.make('Acrobot-v1')
    scores = []
    for i_episode in range(10):
        observation = env.reset()
        fitness = 0
        for t in range(500):
            # env.render()
            output = network.evaluate(observation)
            action = np.argmax(output)
            observation, reward, done, info = env.step(action)
            fitness += reward
            if done:
                # print("done!")
                break
        scores.append(fitness)

    env.close()

    network.fitness = np.average(scores)
    print("average score: ", network.fitness)



def binaryNodeSearch(nodeList, id):
    # the given node list must be sorted!
    # this is a binary search to find the index of a node in a node list
    # if the item is not found, return -1
    first = 0
    last = len(nodeList)-1
    index = -1
    while (first <= last) and (index == -1):
        mid = (first + last) // 2
        # the node has been found
        if id == nodeList[mid].id:
            index = mid
        else:
            # if the connection is lower than the middle:
            if id < nodeList[mid].id:
                last = mid -1
            else:
                first = mid + 1
    return index

def binaryGenomeSearch(connectionList, connection):
    # the given connectionList must be sorted!
    # this is a binary search for the index of a connection in a genome
    # if the item is not found, return -1
    first = 0
    last = len(connectionList)-1
    index = -1
    while (first <= last) and (index == -1):
        mid = (first + last) // 2
        if (connection.input.id, connection.output.id) == (connectionList[mid].input.id, connectionList[mid].output.id):
            index = mid
        else:
            # if the connection is lower than the middle:
            if (connection.input.id, connection.output.id) < \
                    (connectionList[mid].input.id, connectionList[mid].output.id):
                last = mid -1
            else:
                first = mid + 1
    return index

def getMismatchedGenes(genome1, genome2):
    # returns the genes in the first network that do not exist in the second, including both excess and disjoint
    # assumes that both genomes have been sorted
    mismatchedGenes = []
    for gene in genome1:
        index = binaryGenomeSearch(genome2, gene)
        if index == -1:
            mismatchedGenes.append(gene)
    return mismatchedGenes

def getSharedGenes(genome1, genome2):
    # assumes that the given genomes are sorted
    # returns a list of tuples that includes the shared genes between both given networks
    # each tuple contains (connection1, connection2) for each shared connection
    # connection1 comes from genome1, and connection2 from genome2
    sharedGenes = []
    for gene in genome1:
        index = binaryGenomeSearch(genome2, gene)
        if index != -1:
            sharedGenes.append((gene, genome2[index]))
    return sharedGenes


def crossover(parent1, parent2):
    # start the child being completely empty
    child = Network(0, 0)

    # set up the input and output layers
    child.inputLayer = Layer(None, None, 0)
    child.outputLayer = Layer(child.inputLayer, None, 0)
    child.inputLayer.next = child.outputLayer

    # print("adding input nodes")
    # add the nodes that are in the input/output layers
    for i in range(parent1.inputLayer.numNodes):
        child.addNode(child.inputLayer, iNum=i)
    for i in range(parent1.outputLayer.numNodes):
        child.addNode(child.outputLayer, iNum=i+child.inputLayer.numNodes)

    # chose the main parent as the one with the higher fitness
    parentChoice = [parent1, parent2]
    mainParent = max(parentChoice, key=lambda parent: parent.fitness)
    secondParent = min(parentChoice, key=lambda parent: parent.fitness)

    # loop through all the hidden layers, adding the nodes in each layer to the child
    currentLayer = mainParent.inputLayer.next
    currentChildLayer = child.inputLayer
    while currentLayer != mainParent.outputLayer:
        # add a new layer to be in the same location as the layer in the parent
        child.addLayer(currentChildLayer)
        currentChildLayer = currentChildLayer.next

        # add all nodes in the main parent to the child
        for node in currentLayer.nodes:
            child.addNode(currentChildLayer, node.id)

        currentLayer = currentLayer.next


    mismatchedGenes = getMismatchedGenes(mainParent.connections, secondParent.connections)

    # loop through all mismatched genes and add to the child directly
    for connection in mismatchedGenes:
        inputID = connection.input.id
        outputID = connection.output.id
        # the child currently has all nodes of the parent, so we can find the nodes in the child to use for the connection
        inputNode = child.nodes[binaryNodeSearch(child.nodes, inputID)]
        outputNode = child.nodes[binaryNodeSearch(child.nodes, outputID)]
        child.addConnection(inputNode, outputNode, weight=connection.weight)
        child.connections[-1].isEnabled = connection.isEnabled

    # loop through all matching genes and choose one at random to add to the child
    sharedGenes = getSharedGenes(mainParent.connections, secondParent.connections)

    # loop through all shared connections and add to the child
    for connectionSet in sharedGenes:
        connection = choice(list(connectionSet))
        inputID = connection.input.id
        outputID = connection.output.id
        # the child currently has all nodes of the parent, so we can find the nodes in the child to use for the connection
        inputNode = child.nodes[binaryNodeSearch(child.nodes, inputID)]
        outputNode = child.nodes[binaryNodeSearch(child.nodes, outputID)]
        child.addConnection(inputNode, outputNode, weight=connection.weight)
        child.connections[-1].isEnabled = connection.isEnabled

    child.mutate()
    return child


def distance(network1, network2):
    # get sorted lists of genes. these will need to be used and sorted for the binary search functions
    # genome1 = sorted(network1.connections, key=lambda connection: (connection.input.id, connection.output.id))
    # genome2 = sorted(network2.connections, key=lambda connection: (connection.input.id, connection.output.id))
    # because network connections are sorted in mutation(), we can always assume network connections are sorted
    genome1 = network1.connections
    genome2 = network2.connections
    # ==============================================================
    # get the number of genes that are mismatched (excess+disjoint)
    g1MismatchGenes = getMismatchedGenes(genome1, genome2)
    g2MismatchGenes = getMismatchedGenes(genome2, genome1)
    numMismatchedGenes = len(g1MismatchGenes) + len(g2MismatchGenes)

    # ==============================================================

    # get the shared genes and find the average difference in weights
    shared = getSharedGenes(genome1, genome2)
    weightDifferences = []
    for i in shared:
        weightDifferences.append(abs(i[0].weight-i[1].weight))

    # ==============================================================

    # value N is set to the number of genes in the larger genome
    N = max(len(genome1), len(genome2))

    # ==============================================================

    # in the event of np.average getting nan, dont use it
    if len(weightDifferences) == 0:
        distance = numMismatchedGenes/N
    else:
        npaverage = np.average(weightDifferences)
        distance = numMismatchedGenes/N + npaverage

    return distance



class Generation:
    def __init__(self, inputLayerSize, outputLayerSize, size=100):
        self.size = size
        self.speciesList = []
        self.generationNumber = 0

        # create an initial network to populate the first generation (using asexual reproduction)
        startNetwork = Network(inputLayerSize, outputLayerSize)
        startNetwork2 = Network(inputLayerSize, outputLayerSize)
        self.speciesList.append(Species())
        self.speciesList[0].build(startNetwork, startNetwork2, size)

    def run(self):
        # train each network in the species
        for id, species in enumerate(self.speciesList):
            print("species {} size {} =============================".format(id, len(species.networkList)))
            for network in species.networkList:
                trainAcrobot(network)

    def getChampion(self):
        bestNets = []
        for species in self.speciesList:
            topNet = species.getTopNetworks(1)
            if len(topNet)>0:
                bestNets.append(topNet[0])
        bestNets = sorted(bestNets, key=lambda network: network.fitness, reverse=True)
        return bestNets[0]


    def setAdjustedFitnessess(self):
        # set the adjusted fitness for every network in the generation
        for species in self.speciesList:
            for network in species.networkList:
                network.setAdjustedFitness(len(species.networkList))

    def getAverageAdjustedFitness(self):
        # first set all the adjusted fitnesses. holding this in memory gives some improvement on efficiency
        self.setAdjustedFitnessess()
        # get all adjusted fitnesses in the generation and return the average
        adjustedFitnesses = [network.adjustedFitness for species in self.speciesList for network in species.networkList]

        return np.average(adjustedFitnesses)

    def __getExpectedOffspring(self):
        averageFitness = self.getAverageAdjustedFitness()
        expectedOffspring = []
        # loop through all species and get their expected offspring
        for index, species in enumerate(self.speciesList):
            nextPopulation = species.getNewSpeciesSize(averageFitness)
            expectedOffspring.append(nextPopulation)

        # use the floor of the expected species size and save all of the skimmed decimals
        # this will later be distributed to the rest of the generation
        skim = int(round(sum([i - int(i) for i in expectedOffspring])))
        expectedOffspring = [int(i) for i in expectedOffspring]

        allExpectedOffspring = sum(expectedOffspring)
        # to fix any lost floating point precision, add to the skim if necessary
        if (allExpectedOffspring + skim) != self.size:
            skim = self.size - allExpectedOffspring

        # spread the skimmed offspring to the best species
        bestSpecies = expectedOffspring.index(max(expectedOffspring))
        expectedOffspring[bestSpecies] += skim
        return expectedOffspring


    def deadSpeciesExists(self):
        # loop through the species list and find any dead species. If there are any, return True
        deadSpeciesList = [len(species.networkList) <= 0 for species in self.speciesList]
        return True in deadSpeciesList


    def removeDeadSpecies(self):
        # is there a dead species in the generation
        # remove dead species
        while(self.deadSpeciesExists()):
            for species in self.speciesList:
                if len(species.networkList) <= 0:
                    self.speciesList.remove(species)


    def build(self):
        newSpeciesList = []
        # for each species:
        expectedOffspring = self.__getExpectedOffspring()
        for index, species in enumerate(self.speciesList):
            # if len(species.networkList) <= 0:
            #     continue
            # Use N parents from the species to generate J children and store in an array
            # If there is only one, use asexual reproduction
            parents = species.getTopNetworks(2)
            # find how many children the species is allowed to create
            newSize = int(expectedOffspring[index])

            # crossover to create as many children as allowed
            # if the species is only one network, crossover with itself
            if len(parents) == 1:
                species.networkList = []
                species.build(parents[0], parents[0], newSize)
            else:
                species.networkList = []
                species.build(parents[0], parents[1], newSize)

        # loop through each species and locate outcasts
        outcasts = []
        for species in self.speciesList:
            for index, network in enumerate(species.networkList):
                if distance(network, species.representative) > 0.35:        # todo how can we choose a better delta_t?
                    outcast = species.networkList.pop(index)
                    outcasts.append(outcast)

        # try to find a new home for the outcast.
        for index, network in enumerate(outcasts):
            for species in self.speciesList:
                if distance(network, species.representative) <= 0.35:
                    species.networkList.append(outcasts.pop(index))
                    break

        # If there is no compatible home, make a new species
        for index, network in enumerate(outcasts):

            newSpecies = Species()
            newSpecies.representative = network
            newSpecies.networkList.append(network)
            newSpeciesList.append(newSpecies)
        # speciesSizes= [len(species.networkList) for species in newSpeciesList]
        # print("1 species created, size : ", len(newSpecies.networkList))
        self.speciesList += newSpeciesList

        self.removeDeadSpecies()



class Species:
    def __init__(self):
        # self.size = size
        self.networkList = []
        self.representative = None

    def build(self, parent1, parent2, size):
        self.representative = parent1
        if size == 0:
            # if the size is 0, set the representative and do not add any networks
            return

        self.networkList.append(parent1)
        size -= 1
        if size == 0:
            # we cannot add any more networks, return
            return

        for i in range(0, size):
            self.networkList.append(crossover(parent1, parent2))

    def getTopNetworks(self, num):
        # print("networklist size: ", len(self.networkList))
        # sort the networks based on their fitness and return a list of the top N networks
        self.networkList = sorted(self.networkList, key=lambda network: network.fitness, reverse=True)
        return self.networkList[:num]


    def getNewSpeciesSize(self, averageAdjustedFitnesses):
        # nPrime = sum(all adjusted fitnesses in the species) / generation average adjusted fitness
        speciesAdjustedFitnesses = [network.adjustedFitness for network in self.networkList]
        nPrime = sum(speciesAdjustedFitnesses) / averageAdjustedFitnesses

        return nPrime

