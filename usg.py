import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
from ast import literal_eval as make_tuple
from math import sqrt, ceil, floor
import operator
from beam_maker import makeBeams, calculatePoint


def scanl(f, base, l):
    for x in l:
        base = f(base, x)
        yield base

class SoundPulse:
    def __init__(self, power, position, trajectory, currentStep = None):
        self.power = power
        self.position = position
        if currentStep == None:
            self.direction = trajectory
            self.usesTrajectory = False
        else:
            self.usesTrajectory = True
            self.trajectory = trajectory
            self.currentStep = currentStep
        self.angle = 0
    direction = (0,1)
    usesTrajectory = False
    power = 1
    phase = 1
    position = (0,0)
    currentStep = 0
    isReflection = False
    angle = 0

def translate(ll, t):
    return [_add(e,t) for e in ll]
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
        self.reflectedPulsesCanReflect = recursiveReflections
        self.pulseIds = list()
    def getCell(self, pos):
        return self.cells[pos]
    def createPulseWithTrajectory(self, position, direction, currentPosition, angle):
            newPulse = SoundPulse(1.0, position, direction, currentPosition)
            newPulse.angle = angle
            if position in self.pulses.keys():
                (self.pulses[position]).append(newPulse)
            else:
                self.pulses[position] = [newPulse]
            self.history[angle] = []
    def createPulse(self, position, direction, currentPosition = None):
        if currentPosition == None:
            newPulse = SoundPulse(1.0, position, direction)
        else:
            newPulse = SoundPulse(1.0, position, direction, currentPosition)
        if position in self.pulses.keys():
            (self.pulses[position]).append(newPulse)
        else:
            self.pulses[position] = [newPulse]
    def getPulseList(self, pos):
        return self.pulses[pos]
    def readFromTransceiver(self):
        if not (self.transceiverLocation in self.pulses.keys()):
            for angle in self.history.keys():
                (self.history[angle]).append(0.0)
            return
        pulsesInTransceiver = self.pulses[self.transceiverLocation]
        sums = {key: 0.0 for key in self.history.keys()}
        for pulse in pulsesInTransceiver:
            sums[pulse.angle] += pulse.power * pulse.phase
        print(sums)
        for (key, val) in sums.items():
            (self.history[key]).append(val)
    def log(self):
        for pulseList in self.pulses.values():
            for pulse in pulseList:
                nextPos = pulse.trajectory[pulse.currentStep]
                print("Pulse: pos: {} power: {} reflected: {}, next pos: {} angle: {}".format(pulse.position, pulse.power, pulse.isReflection, nextPos, pulse.angle))
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
                    nextMove = pulse.trajectory[pulse.currentStep]
                else:
                    nextMove = (pulse.position[0] + pulse.direction[0], pulse.position[1] + pulse.direction[1])
                # check if pulse is out of bonds
                # if isn't, do nothing
                pulseRow, pulseCow = nextMove
                # print("Next move: {}".format(nextMove))
                # print(self.cells.shape)
                row, cow = self.cells.shape
                is_in_bonds = (pulseRow in range(0, row)) and (pulseCow in range(0, cow))
                if not is_in_bonds or nextMove == self.tranceiverLocation:
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
                        currentStep = len(pulse.trajectory) - pulse.currentStep
                        reflectedTrajectory = pulse.trajectory.copy()
                        # Reverse trajectory of passing pulse to get reflected pulse's trajectory.
                        # Simple as that.
                        reflectedTrajectory.reverse()
                        # Make sure it returns to the transceiver
                        reflectedTrajectory.append(self.tranceiverLocation)
                        print(pulse.trajectory[pulse.currentStep-1])
                        reflectedPulse = SoundPulse(pulse.power * R, pulse.trajectory[pulse.currentStep - 1], reflectedTrajectory, currentStep)
                    else:
                        reflectedDirection = _mul(pulse.direction, (-1))
                        reflectedPosition = _add(pulse.position, reflectedDirection)
                        reflectedPulse = SoundPulse(pulse.power * R, reflectedPosition, reflectedDirection)
                    if reversePhase:
                        reflectedPulse.phase = -1
                    reflectedPulse.isReflection = True
                    reflectedPulse.angle = pulse.angle
                    print("Created new pulse via reflection {} - power {}".format(reflectedPulse.position, reflectedPulse.power))
                    print("Old pulse position: {}".format(pulse.position))
                    localPulses = appendPulse(localPulses, reflectedPulse)
                pulse.currentStep += 1
                pulse.power = pulse.power * T
                pulse.position = nextMove
                if pulse.power > 0:
                    localPulses = appendPulse(localPulses, pulse)
        self.pulses = localPulses
        self.time += 1
        self.readFromTransceiver()
    time = 0
    history = dict()
    cells = []
    pulses = dict()
    angle = 0
    tranceiverLocation = (0,0)
    reflectedPulsesCanReflect = False

def load(path):
    image = cv2.imread(path)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def normalizeImage(image):
    minv = np.amin(image)
    maxv = np.amax(image)
    print("MIN {} MAX {}".format(minv, maxv))
    if maxv - minv == 0:
        print("Division by 0 in normalizeImage function.")
        exit(-1)
    adjusted = [int(((x - minv) / (maxv - minv))*255.0) for x in np.nditer(image)]
    return np.array(adjusted, dtype=np.uint8).reshape(image.shape)

def makeImage(transceiverHistory):
    transceiverHistory = [x * (-1) for x in transceiverHistory]
    strip = list(scanl(operator.add, 0, transceiverHistory))
    minv = min(strip)
    maxv = max(strip)
    stripWidth = 200;
    adjusted = [[int(((x - minv) / (maxv - minv))*255.0)]*stripWidth for x in strip]
    image = np.array(adjusted, dtype=np.uint8).reshape((len(strip), stripWidth))
    cv2.imshow("Image created from transceiver", image)
    cv2.imwrite('result.png', image)
    plt.plot(range(0,len(strip)), strip)
    plt.show()

def makeImage2D(transceiverHistory, time):
    mid = (int(time/2), int(time/2))
    r = time*2 # Inscibed circle
    image = np.zeros((r, r), dtype=np.float).reshape((r,r))
    for angle, angleHistory in transceiverHistory.items():
        integrated = list(scanl(operator.add, 0.0, angleHistory))
        for i in range(0, len(integrated) - 1):
            point = calculatePoint(i, angle)
            # Translate to center.
            point = _add(point, mid)
            image[point] += integrated[i]
    normalized = normalizeImage(image)
    cv2.imwrite('result.png', normalized)

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
time = 1000
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
angleRange = (0, 360)
step = 0.2
angles = [x for x in range(angleRange[0], angleRange[1], step)]
beams = makeBeams(image.shape, angleRange, step)
beams = [translate(beam, tranPos) for beam in beams]
for beam in beams:
    env.createPulseWithTrajectory(tranPos, beam, 1, angles.pop(0))
env.runFor(time, True)
result = env.history
cv2.imshow("Image", image)
#plt.plot(range(0,len(result)),result)
#plt.show()
makeImage2D(result, time)
