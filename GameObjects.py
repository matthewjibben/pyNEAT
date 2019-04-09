from math import sin, cos
import random

class Polecart:
    # state variables
    theta = 0.00
    thetaVel = 0.0
    thetaAcc = 0.0

    x = 0.0 #pole position.
    y = 0.0
    xVel = 0.0
    xAcc = 0.0



    gravity = -9.81

    massCart = 1.0
    massPole = 0.1
    totalMass = massCart+massPole

    poleLen = 0.5       # this is half of the poles actual length (measured from bottom to center)
    force = 10          # this will either be +- 10 newtons

    #cart variables:
    cartW, cartH = 40, 40

    #pole positions.
    poleX1, poleY1 = x , y   #bottom point attatched to the cart.
    poleX2, poleY2 = poleX1, poleY1 - poleLen

    trackLimit = 2.4    # the cart is only allowed to travel +- 2.4 meters from the center
    poleFail = 0.523599 # the pole can go 30 degrees from level before the test is failed

    currentTime = 0.00  # this will mostly be used for analytics
    tau = 0.0045          # this will be the time jump each tick

    def __init__(self, randomize=True):
        if(randomize):
            self.theta = random.uniform(0.05, 0.25) * random.choice([-1, 1])
            self.thetaVel = random.uniform(0.05, 0.15) * random.choice([-1, 1])
            # print(self.theta, self.thetaVel)
        else:
            self.theta = 0.01
            self.thetaVel = 0.03

    #calculate the new position of the pole's points.
    def _calculatePolePos(self):
        #update the attatched pole position to be equal to the x and y of the main cart body.
        self.poleX1, self.poleY1 = self.x, self.y

        #convert polar coordinates to cartesian coordinates.
        self.poleX2 = self.poleX1 + (self.poleLen * sin(self.theta))
        self.poleY2 = self.poleY1 - (self.poleLen * cos(self.theta))

    # the complete run function
    # takes the action and updates game variables as necessary
    def run(self, action):
        if action == 0:
            self.force = 0
        elif action > 0:
            self.force = 10
        else:
            self.force = -10
        # do things
        # calculate the angular acceleration
        self.thetaAcc = (-self.gravity*sin(self.theta) + cos(self.theta)*
                         ((-self.force-self.massPole*self.poleLen*(self.thetaVel**2)*sin(self.theta))/self.totalMass))
        self.thetaAcc /= self.poleLen*((4/3)-(self.massPole*cos(self.theta)**2)/self.totalMass)

        # calculate the cart acceleration
        self.xAcc = (self.force+self.massPole*self.poleLen*
                     ((self.thetaVel**2)*sin(self.theta)-self.thetaAcc*cos(self.theta))) / self.totalMass

        # update the 4 variables using the time tau
        # should tau be a set 0.02 seconds?
        self.x += self.tau * self.xVel
        self.xVel += self.tau * self.xAcc
        self.theta += self.tau * self.thetaVel
        self.thetaVel += self.tau * self.thetaAcc

        # update the current time and pole positions.
        #self.currentTime += self.tau
        self._calculatePolePos()

    def hasLost(self):
        result = False
        if abs(self.theta) > self.poleFail:
            #print("theta greater than limit", self.theta, ">", self.poleFail)
            result = True
        elif abs(self.x) > self.trackLimit:
            #print("x greater than limit", self.x, ">", self.trackLimit)
            result = True

        #print("result = ", result)
        return result
