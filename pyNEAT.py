# globalInnoNum = 0 not needed here? moved to Node.py

class NEAT:
    def __init__(self, inputDim, outputDim):
        print("Input Vector Length: {}\nOutput Vector Length: {}".format(inputDim, outputDim))
        self.inputSize = inputDim
        self.outputSize = outputDim

    """Outputs genome in a format that can be read back in. 
    """
    def save(self):
        print("Save a network struct")

    """Rebuild the networks from file (output of save function)
    """
    def load(self, i):
        print("Load in the network from a save file.")

    """    
    """
    def getOutput(self, input, network = None):
        print("get output values")
        #TODO:: Make sure the vector saved in input matches the length of self.inputSize.
        #TODO:: if the user inputs a network get the output for that network.
        #TODO:: Decide if this is the best place for this function.

    """After initializing the network. train the network with given input"""
    def train(self, input, numItter):
        print("train all of the networks in the system.")

    """calculate and return the distance between two input networks.
        Distance represents how genetically similar the networks are. 
    """
    def getNetworkDistance(self, net1, net2):
        print("Get distance between two networks genetically")

#-----------------------------------------------------------------
