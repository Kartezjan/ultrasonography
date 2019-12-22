import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
from ast import literal_eval as make_tuple
import operator

def scanl(f, base, l):
    for x in l:
        base = f(base, x)
        yield base

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
    isReflection = False

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

def pairwise(t):
    # Add guard
    t.append(None)
    result = zip(t[::2], t[1::2])
    return result

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
    def __init__(self, image, transceiverLocation, operationalAngle, recursiveReflections = False):
        self.cells = np.array([Cell(x) for x in np.nditer(image)]).reshape(image.shape)
        self.pulses = dict()
        self.transceiverLocation = transceiverLocation
        self.angle = operationalAngle
        self. reflectedPulsesCanReflect = recursiveReflections
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
                pulse_threshold = pulse.power > 1e-5
                can_reflect = not pulse.isReflection or self.reflectedPulsesCanReflect
                if R != 0 and pulse_threshold and can_reflect:
                    # create new pulse
                    if pulse.usesTrajectory:
                        currentStep = pulse.currentStep
                        reflectedTrajectory = pulse.trajectory
                        # Reverse trajectory of passing pulse to get reflected pulse's trajectory.
                        # Simple as that.
                        reflectedTrajectory.reverse()
                        reflectedPulse = SoundPulse(pulse.power * R, pulse.trajectory[pulse.currentStep - 1], reflectedTrajectory, currentStep)
                    else:
                        reflectedDirection = _mul(pulse.direction, (-1))
                        reflectedPosition = _add(pulse.position, reflectedDirection)
                        reflectedPulse = SoundPulse(pulse.power * R, reflectedPosition, reflectedDirection)
                    if reversePhase:
                        reflectedPulse.phase = -1
                    reflectedPulse.isReflection = True
                    print("Created new pulse via reflection {} - power {}".format(reflectedPulse.position, reflectedPulse.power))
                    localPulses = appendPulse(localPulses, reflectedPulse)
                pulse.power = pulse.power * T
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
    reflectedPulsesCanReflect = False

def load(path):
    image = cv2.imread(path)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def makeImage(transceiverHistory):
    transceiverHistory = [x * (-1) for x in transceiverHistory]
    strip = list(scanl(operator.add, 0, transceiverHistory))
    minv = min(strip)
    maxv = max(strip)
    stripWidth = 200;
    adjusted = [[int(((x - minv) / (maxv - minv))*255.0)]*stripWidth for x in strip]
    image = np.array(adjusted, dtype=np.uint8).reshape((len(strip), stripWidth))
    cv2.imshow("Image created from transceiver", image)
    plt.plot(range(0,len(strip)), strip)
    plt.show()

def print_help():
    print("usage: usg <image_path> [<flag> <flag_value>]")
    print("available flags:")
    print("\t-recursiveReflections Bool (allow reflections to reflect further)")
    print("\t-time Int (set simulation time)")

def boolArg(arg, val):
    val = val.upper()
    if val == "TRUE":
        return True
    if val == "FALSE":
        return False
    print("Invalid value '{}' for -{} flag!".format(val, arg))
    print("\tFlag type: Bool")
    exit(2)

def intArg(arg, val):
    try:
        val = int(val)
        return val
    except:
        print("Invalid value '{}' for -{} flag!".format(val, arg))
        print("\tFlag type: Int")
        exit(2)

if len(sys.argv) < 2:
    print_help()
    exit(1)
path = sys.argv[1];
allowReflections = False
time = 3000
image = load(path)
row, cow = image.shape
mid = int(cow/2)
tranPos = (0, mid)
if len(sys.argv) >= 2:
    args = sys.argv[2:]
    arg_pairs = list(pairwise(args))
    for arg, val in arg_pairs:
        if(arg[0] != '-'):
            print_help()
            exit(1)
        arg = arg[1:]
        if arg == "recursiveReflections":
            allowReflections = boolArg(arg, val)
        elif arg == "time":
            time = intArg(arg, val)
        else:
            print("Invalid flag -{}".format(arg))
            print_help()
            exit(1)
print("Transceiver location: {}".format(tranPos))
print("Image size: {}".format(image.shape))
print(allowReflections)
env = Environment(image, tranPos, 0, allowReflections)
env.createPulse(tranPos, (1,0))
env.createPulse(tranPos, (1,0))
env.runFor(time, False)
result = env.history
cv2.imshow("Image", image)
plt.plot(range(0,len(result)),result)
plt.show()
makeImage(result)
