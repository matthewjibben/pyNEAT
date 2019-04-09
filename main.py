import random
import copy
import pickle
import numpy as np

# np.random.seed(1)
# random.seed(1)


from Game import PoleBalancing
from GameObjects import Polecart
from time import sleep
from Network import *
from Generation import *

def printConnections(network):
    for connection in network.connections:
        print("{} -> {} weight: {}".format(connection.input.id, connection.output.id, connection.weight))







# gen = Generation(4, 3, size=100)
# gen.run()



def genSize(generation):
    count = 0
    for species in generation.speciesList:
        for network in species.networkList:
            count += 1
    return count



# if __name__ == '__main__':
#     gen = Generation(4, 3)
#     highscore = 0
#     # try at most 20 generations
#     for i in range(20):
#         print("==========================starting new generation: ", i, genSize(gen))
#         gen.run()
#
#         bestNet = gen.getChampion()
#         print("Champion fitness: ", bestNet.fitness)
#
#         if bestNet.fitness >= 10000:
#             with open(r"bestNet4-8.obj", "wb") as out:
#                 pickle.dump(bestNet, out)
#             print("perfect score!!!!")
#             break
#
#         gen.build()

if __name__ == '__main__':
    gen = Generation(6, 3)
    highscore = -4000000
    # try at most 20 generations
    for i in range(200):
        print("==========================starting new generation: ", i, genSize(gen))
        gen.run()

        bestNet = gen.getChampion()
        print("Champion fitness: ", bestNet.fitness)

        if bestNet.fitness > highscore:
            print("new champion: ", bestNet.fitness)
            highscore = bestNet.fitness
            with open(r"Acrobot3.obj", "wb") as out:
                pickle.dump(bestNet, out)

        gen.build()


# with open(r"bestNet4-8.obj", "rb") as inp:
#     network = pickle.load(inp)
#
# g = PoleBalancing()
# g.startAI(network)


# with open(r"Acrobot2.obj", "rb") as inp:
#     network = pickle.load(inp)
#
# trainAcrobot(network)

# g = PoleBalancing()
# g.startAI(network)


# for species in gen.speciesList:
#     parent1, parent2 = species.getTopNetworks(2)
#     species.build(parent1, parent2)



# l = []
# original = Network(3, 4)
# for _ in range(100000):
#     n1 = copy.deepcopy(original)
#     n2 = copy.deepcopy(original)
#     [n1.mutate() for i in range(5)]
#     [n2.mutate() for j in range(5)]
#     n1.fitness = 1000
#     n2.fitness = 4
#     child = crossover(n1, n2)
#     l.append(distance(n1, child))

    # if distance(n1, child) == 2.0:
    #     print("n1: ========================")
    #     printConnections(n1)
    #     print("child: ========================")
    #     printConnections(child)
    #     print("========================")
    # print(distance(n1, child))

# matching = [i for i in l if i > 0.85]
# print("percent above: ", len(matching))

# print(l)
# print("average ", np.average(l))
# print(sorted(l))
# print(sorted(l, reverse=True))

# # print(distance(n1, n2))
#
# n1.connections = sorted(n1.connections, key=lambda connection: (connection.input.id, connection.output.id))
#

# printConnections(n1)
# print("===================================")
# printConnections(n2)
# print("===================================")

# child.print()
# printConnections(child)
# child.print()

# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")
# child.mutateAddNode()



# print(n1.evaluate([1,2,3]))


# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")
# print(child.evaluate([1,2,3]))



# n1.print()

# print("final distance:", distance(n1, n2))



# # train(network)
# with open(r"best-Net4-1.obj", "rb") as inp:
#     network2 = pickle.load(inp)

# train(network2)



# gen = Generation(4,3)
# gen.run()
# gen.setAdjustedFitnessess()
#
# print([network.fitness for species in gen.speciesList for network in species.networkList])
#
# print(gen.getAverageAdjustedFitness())
#
#
# print(gen.speciesList[0].getNewSpeciesSize(gen.getAverageAdjustedFitness()))
# print(gen.speciesList[1].getNewSpeciesSize(gen.getAverageAdjustedFitness()))

'''
gen = Generation(4, 3)
highscore = 0
# try at most 20 generations
for i in range(20):
    print("==========================starting new generation: ", i)
    gen.run()
    bestNet = gen.getChampion()
    print("Champion fitness: ", bestNet.fitness)

    if bestNet.fitness >= 10000:
        with open(r"bestNet4-5.obj", "wb") as out:
            pickle.dump(bestNet, out)
        print("perfect score!!!!")
        break

    for species in gen.speciesList:
        parent1, parent2 = species.getBestNetworks()
        # todo get the number of networks that will be generated (N'_j)
        species.build(parent1, parent2)
'''


'''
startNetwork = Network(4, 3)
generation=buildGeneration(startNetwork, startNetwork)


bestScore = 0
bestNetwork = None
genID = 0
while(True):
    # train each network and assign fitness scores
    for network in generation:
        train(network)

    generation = sorted(generation, key=lambda net: net.fitness, reverse=True)

    # if generation[0].fitness > bestScore:
    #     bestNetwork = generation[0]
    #     with open(r"bestNet4-1.obj", "wb") as out:
    #         pickle.dump(bestNetwork, out)



    print("===================================")

    print("generation ID: ", genID)
    genID += 1
    print("This generation's best network has score: ", generation[0].fitness)
    print("Best ever score: ", bestNetwork.fitness)

    print("===================================")




    generation=buildGeneration(generation[0], generation[1])


'''










#g = PoleBalancing()
#g.start()

#polecart = Polecart()

#while(polecart.theta < 3.14/2):
#    sleep(0.1)
#    print("====================================")
#    print("Current time:", polecart.currentTime)
#    print("Pole angle:", polecart.theta*180/3.14159)
#    print("Pole Angular Vel:", polecart.thetaVel)
#    print("Pole Angular Acc:",polecart.thetaAcc)
#    print("Pole Pos:",polecart.x)
#    print("====================================")
#    polecart.run(0)
#    break #TODO: Remove before pushing.


#g.startAI()

# test = Network(3, 3)
# test2 = copy.deepcopy(test)

# test hidden layer and recurrent connections
# test.addLayer(test.inputLayer, 2)
#
# # add connection between the input and the hidden layer
# test.addConnection(test.inputLayer.nodes[0], test.inputLayer.next.nodes[0])
# test.addConnection(test.inputLayer.next.nodes[0], test.outputLayer.nodes[0])
# #
# # # add recurrent connections from output layer to the hidden layer
# test.addConnection(test.outputLayer.nodes[0], test.inputLayer.next.nodes[0])
# test.addConnection(test.outputLayer.nodes[1], test.inputLayer.next.nodes[0])
# #
# # # add connection between the input and the hidden layer
# test.addConnection(test.inputLayer.nodes[0], test.inputLayer.next.nodes[1])
# test.addConnection(test.inputLayer.next.nodes[1], test.outputLayer.nodes[2])


# [test.mutate() for i in range(5000)]
# test.print()
# test.mutateAddNode()
# test.print()

# print("===============================")
# test2.mutateAddNode()
# test2.print()



# print(set(test.getAllConnections()) == set(test.connections))
# watch that the recurrent connection values converge to a certain value. It's working!
# print(test.evaluate([1,2,3]))
# print(test.evaluate([1,2,3]))
# print(test.evaluate([1,2,3]))
#
# print(test.evaluate([1,2,3]))
# print(test.evaluate([1,2,3]))
#
# print(test.evaluate([1,2,3]))
# print(test.evaluate([1,2,3]))
#
# print(test.evaluate([1,2,3]))
# print(test.evaluate([1,2,3]))
#
# print(test.evaluate([1,2,3]))
# print(test.evaluate([1,2,3]))
# print(test.evaluate([1,2,3]))
#
# print(test.evaluate([1,2,3]))

#test.mutate()
#test.print()


#print("\n\n\n\n\n\n\n\n\n\n\n\n")
