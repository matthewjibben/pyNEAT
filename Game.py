import pygame
from GameObjects import Polecart

from Network import *
import pickle

white = (255,255,255)
red = (255,0,0)
green = (0,255,0)
blue = (0,0, 255)

class PoleBalancing:
    def __init__(self):
        stats = 0
        #initialize the pygame.
        pygame.init()
        pygame.display.set_caption("Pole balance")

        #create a window to draw on.
        self.width, self.height = 600, 600
        self.myScreen = pygame.display.set_mode((self.width, self.height))  # tuple for width height.

        #initialize fonts.
        self.fonts = {"header": pygame.font.SysFont("Arial", 30), "sub": pygame.font.SysFont("Arial", 15)}
        self.staticTexts = [self.fonts["header"].render("Fitness:", True, white),
                           self.fonts["header"].render("Time Elapsed:", True, white)
                           ]

        #Initialize variables for game time.
        self.clock = pygame.time.Clock()
        self.FPS = 120
        self.timeElapsed = 0
        self.tau = 0.2

        self.balancedTime = 0
        self.fitness = 0

        #Set loosing conditions.
        self. trackLimit = 2.4 * 100  # the cart is only allowed to travel +- 2.4 meters from the center
        self. poleFail = 0.523599  # the pole can go 30 degrees from level before the test is failed

        #create a polecart variable.
        self.polecart = Polecart()

    def _printPoleCartVars(self):
        print("====================================")
        print("Current time:", self.polecart.currentTime)
        print("Pole angle:", self.polecart.theta * 180 / 3.14159)
        print("Pole Angular Vel:", self.polecart.thetaVel)
        print("Pole Angular Acc:", self.polecart.thetaAcc)
        print("Cart Pos:", self.polecart.x)
        print("Pole connected point(x: {}, y: {}".format(self.polecart.poleX1, self.polecart.poleY1))
        print("Pole free endpoint(x: {}, y: {}".format(self.polecart.poleX2, self.polecart.poleY2))
        print("====================================")

        print(self.tau)

    def _drawSprites(self, dText):
        ###Fonts
        # draw the static font boxes.
        for i in range(len(self.staticTexts)):
            text = self.staticTexts[i]
            position = (15, i * 30 + 10)
            self.myScreen.blit(text,position)

        for i in range(len(dText)):
            text = self.fonts["sub"].render(dText[i], True, white)
            position = (200, 20 + i * 30)
            self.myScreen.blit(text, position)

        ###Game objects
        #transform the cart position to game coordinates so it can be displayed.
        cx, cy = self._transformPoint(self.polecart.x, self.polecart.y)

        #transform the pole endpoint connected to the cart to game coordinates.
        px1, py1 = self._transformPoint(self.polecart.poleX1, self.polecart.poleY1)
        px1 += self.polecart.cartW/2 #transform the attatched pole endpoint relative to the middle of the cart.

        #transform the pole's free endpoint to game coordinates.
        px2, py2= self._transformPoint(self.polecart.poleX2, self.polecart.poleY2)
        px2 += self.polecart.cartW / 2  # transform the free pole endpoint relative to the middle of the cart.

        pygame.draw.line(self.myScreen, white, [0, self.height/2], [self.width, self.height/2]) #track
        pygame.draw.rect(self.myScreen, green, (cx, cy, self.polecart.cartW, self.polecart.cartH)) #cart
        pygame.draw.line(self.myScreen, (255, 255, 0), [px1, py1], [px2, py2], 5)  # pole
        pygame.display.update()

    def _updateTime(self, delay):
        self.tau = self.clock.tick(self.FPS)/delay
        self.timeElapsed += self.tau
        #print("Time Elapsed: {}\n game time:{}\n\n".format(self.tau, self.timeElapsed))


    def _transformPoint(self,x,y):
        #convert meters to milimeters
        x *= 100
        y *= 100

        x += self.width/2 - self.polecart.cartW/2
        y += self.height/2 - self.polecart.cartH
        return x, y

    def _getGameStatus(self, x):
        #Determine if the player has lost the game.
        #game loosing conditions.
            #trackLimit = 2.4    the cart is only allowed to travel +- 2.4 meters from the center
            #poleFail = 0.523599 the pole can go 30 degrees from level before the test is failed
        result = True
        #check for failure.
        if( x < -1*self.trackLimit
            or x  > self.trackLimit
            or abs(self.polecart.theta) > self.poleFail):
            result = False

        #check for win.
        if(self.balancedTime >= 1):
            #add a bonus to the fitness score for perfectly balancing the pole.
            self.fitness += 1000 - self.timeElapsed
            print("win achieved")
            result = False
        return result

    def _updateFitness(self, w):
        if(self.polecart.theta == 0):
            self.balancedTime += self.tau
            self.fitness += w * 2
        else:
            self.balancedTime = 0
            self.fitness = self.fitness + w * (abs(self.polecart.theta)  ** -1 )

    def start(self):
        """game main loop."""
        playing = True
        while (playing):
            #calculate the amount of mselapsed and update the game time.
            self._updateTime(2000) # 2000 = delay.

            #for human player.
            #TODO: create a menu system for different play types human vs. top neural network.
            #check for an event(I/O)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    playing = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                #implement forces.
                self.polecart.run(-1)
                #self._printPoleCartVars()

            if keys[pygame.K_RIGHT]:
                self.polecart.run(1)
                #self._printPoleCartVars()

            else:
                self.polecart.run(0)
                #self._printPoleCartVars()

            """Draw"""
            # fill the background back in so that it doesn't keep track of old drawings.
            self.myScreen.fill((0, 0, 0))
            self._drawSprites(["{0:.3f}".format(self.fitness), "{0:.3f}".format(self.timeElapsed)])

            pygame.display.update()

            """Check for gameover status"""
            self._updateFitness(1)
            playing = self._getGameStatus(self.polecart.x)


        self.myScreen.fill((0, 0, 0))

        texts = ["Game Over: Thanks for playing :)", "Total time alive: {}".format(self.timeElapsed), "Fitness: {0:.3f}".format(self.fitness)]
        for i in range(len(texts)):
            text = texts[i]
            text = self.fonts["header"].render(text, True, white)
            position = (self.width/2 - text.get_rect().width/2, (self.height/2 - 100) + i * 50)
            self.myScreen.blit(text, position)

        pygame.display.update()

        #TODO: When not so tired figure out menu system for human player vs neural network.
        #TODO: return a value to main so it can be stored in a save file if the fitness score is the best the game
        #todo:              has ever recorded.
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()


    def startAI(self, network):
        """game main loop."""
        playing = True
        # input will be the 4 polecart variables, output will be move left,right or no movement
        high = 0
        highNetwork = 0
        for i in range(1):
            # with open(r"3-3-19bestNet.obj", "rb") as inp:
            #     network = pickle.load(inp)
            # print("network ", i)
            # network = Genome(4, 3)
            scores = []
            for i in range(500):
                # reset timer
                self.timeElapsed = 0

                playing=True
                self.polecart = Polecart()
                time=0
                while (playing):
                    time+=1
                    inputVector = [self.polecart.x, self.polecart.xVel, self.polecart.theta, self.polecart.thetaVel]
                    output = network.evaluate(inputVector)
                    decision = np.argmax(output) - 1
                    self.polecart.run(decision)
                    #calculate the amount of mselapsed and update the game time.
                    # self._updateTime(2000) # 2000 = delay.


                    #check for an event(I/O)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            playing = False


                    """Draw"""
                    # fill the background back in so that it doesn't keep track of old drawings.
                    self.myScreen.fill((0, 0, 0))
                    self._drawSprites(["{0:.3f}".format(self.fitness), "{0:.3f}".format(self.timeElapsed)])

                    pygame.display.update()

                    """Check for gameover status"""
                    self._updateFitness(1)
                    #playing = self._getGameStatus(self.polecart.x)
                    playing = not self.polecart.hasLost()
                    if (not playing):
                        print("Player has died!")
                    if time > 5000:
                        print("player has won!")
                        playing = False
                scores.append(time)
                self.fitness = 0
                # print("one down")
            # if np.average(scores) > high:
            #     high = np.average(scores)
            #     print("highscore: ", high)
            #     highNetwork = network
                # with open(r"bestNet.obj", "wb") as out:
                #     pickle.dump(highNetwork, out)








        self.myScreen.fill((0, 0, 0))

        texts = ["Game Over: Thanks for playing :)", "Total time alive: {}".format(self.timeElapsed), "Fitness: {0:.3f}".format(self.fitness)]
        for i in range(len(texts)):
            text = texts[i]
            text = self.fonts["header"].render(text, True, white)
            position = (self.width/2 - text.get_rect().width/2, (self.height/2 - 100) + i * 50)
            self.myScreen.blit(text, position)

        pygame.display.update()

        #TODO: When not so tired figure out menu system for human player vs neural network.
        #TODO: return a value to main so it can be stored in a save file if the fitness score is the best the game
        #todo:              has ever recorded.
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()