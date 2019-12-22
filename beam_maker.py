import math

def calculatePoint(r, angle):
    angle = angle*math.pi/180
    Re = round(r*(math.cos(angle)))
    Im = round(r*(math.sin(angle)))
    return (Re, Im)

def makeBeams(boundaries, angle_range, delta_angle):
    row, col = boundaries
    start_angle, end_angle = angle_range
    diagonal = math.ceil(math.sqrt((col**2)+(row**2)))
    all_beams = list()
    for current_angle in range(start_angle, end_angle, delta_angle):
        one_beam = [calculatePoint(i, current_angle) for i in range(0, int(diagonal))]
        all_beams.append(one_beam)
    return all_beams
