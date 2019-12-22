import cv2
import matplotlib.pyplot as plt 
import sys
import numpy as np

class SoundPulse:
    def __init__(self, power, position, trajectory, currentStep = None):
        self.power = power
        self.position = position
        self.direction = trajectory
        if currentStep == None:
            self.usesTrajectory = False
        else:
            self.usesTrajectory = True
            self.currentStep = currentStep
    direction = (0,1)
    usesTrajectory = False
    power = 1
    phase = 1
    position = (0,0)
    currentStep = 0

def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])
def _mul(a, c):
    return (a[0] * c, a[1] * c)
def appendPulse(dictP, pulse):
    if pulse.position in dictP.keys():
        (dictP[pulse.position]).append(pulse)
    else:
        dictP[pulse.position] = [pulse]
    return dictP

class Cell:
    def __init__(self, density):
        self.density = density
    density = 0

def calcCoeffients(n1, n2):
    n1 = float(n1)
    n2 = float(n2)
    if n1 == n2:
        return (1, 0, False)
    r = (n1 - n2) / (n1 + n2)
    R = r**2
    T = 1 - R
    print("Reflection: {}".format(R))
    return (T, R, r < 0)

class Environment:
    def __init__(self, image, transceiverLocation, operationalAngle):
        self.cells = np.array([Cell(x) for x in np.nditer(image)]).reshape(image.shape)
        self.pulses = dict()
        self.transceiverLocation = transceiverLocation
        self.angle = operationalAngle
    def getCell(self, pos):
        return self.cells[pos]
    def createPulse(self, position, direction, currentPosition = None):
        if currentPosition == None:
            newPulse = SoundPulse(1.0, position, direction)
        else:
            newPulse = SoundPulse(1.0, position, trajectory, currentPosition)
        print("Created pulse {}".format(position))
        if position in self.pulses.keys():
            (self.pulses[position]).append(newPulse)
        else:
            self.pulses[position] = [newPulse]
    def getPulseList(self, pos):
        return self.pulses[pos]
    def readFromTransceiver(self):
        if not (self.transceiverLocation in self.pulses.keys()):
            return 0.0
        pulses = self.pulses[self.transceiverLocation]
        sum = 0.0
        for pulse in pulses:
            sum += pulse.power * pulse.phase
        return sum
    def log(self):
        print(self.pulses)
        print("Next step")
    def runFor(self, time, verbose = True):
        if time < 1:
            return
        for t in range(1, time):
            self.step()
            if verbose:
                self.log()
    def step(self):
        localPulses = dict()
        for pulseList in self.pulses.values():
            for pulse in pulseList:
                # move pulse
                if pulse.usesTrajectory:
                    nextMove = pulse.trajectory.pop(pulse.currentStep)
                    pulse.currentStep += 1
                else:
                    nextMove = (pulse.position[0] + pulse.direction[0], pulse.position[1] + pulse.direction[1])
                # check if pulse is out of bonds
                # if isn't, do nothing
                pulseRow, pulseCow = nextMove
                # print("Next move: {}".format(nextMove))
                # print(self.cells.shape)
                row, cow = self.cells.shape
                is_in_bonds = (pulseRow in range(0, row)) and (pulseCow in range(0, cow))
                if not is_in_bonds:
                    print("out of bonds!")
                    continue
                # calculate transmission coefficient
                n1 = self.cells[pulse.position].density
                n2 = self.cells[nextMove].density
                T, R, reversePhase = calcCoeffients(n1, n2)
                pulse.power = pulse.power * T
                if R != 0 and pulse.power > 1e-5:
                    print("Old pulse has power {}".format(T))
                    # create new pulse
                    if pulse.usesTrajectory:
                        currentStep = pulse.currentStep - 1
                        reflectedPulse = SoundPulse(pulse.power * R, pulse.trajectory[currentStep], pulse.trajectory, currentStep)
                    else:
                        reflectedDirection = _mul(pulse.direction, (-1))
                        reflectedPosition = _add(pulse.position, reflectedDirection)
                        reflectedPulse = SoundPulse(pulse.power * R, reflectedPosition, reflectedDirection)
                    if reversePhase:
                        reflectedPulse.phase = -1
                    print("Created new pulse via reflection {} - power {}".format(reflectedPulse.position, reflectedPulse.power))
                    localPulses = appendPulse(localPulses, reflectedPulse)
                pulse.position = nextMove
                if pulse.power > 0:
                    localPulses = appendPulse(localPulses, pulse)
        self.pulses = localPulses
        self.time += 1
        self.history.append(self.readFromTransceiver())
    time = 0
    history = []
    cells = []
    pulses = dict()
    angle = 0
    tranceiverLocation = (0,0)

def load(path):
    image = cv2.imread(path)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if len(sys.argv) < 2:
    print("usage: usg <image_path>")
    exit(0);
path = sys.argv[1];
image = load(path)
row, cow = image.shape
mid = int(cow/2)
tranPos = (0, mid)
np.set_printoptions(threshold=sys.maxsize)
print(image)
print("Transceiver location: {}".format(tranPos))
print("Image size: {}".format(image.shape))
env = Environment(image, tranPos, 0)
env.createPulse(tranPos, (1,0))
env.createPulse(tranPos, (1,0))
env.log()
env.runFor(10000, False)
result = env.history
print(env.history)
cv2.imshow("Image", image)
plt.plot(range(0,len(result)),result)
plt.show()
