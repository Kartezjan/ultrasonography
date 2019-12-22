import math

def makeBeams(boundaries, angle_range, delta_angle):
    row, col = boundaries
    start_angle, end_angle = angle_range
    diagonal = math.ceil(math.sqrt((col**2)+(row**2)))
    for current_angle in range(start_angle,end_angle, delta_angle):
        one_beam = []
        current_angle = current_angle*math.pi/180
        for i in range(0,int(diagonal)):
            float(current_angle)
            Re = round(i*(math.cos(current_angle)))
            Im = round(i*(math.sin(current_angle)))
            one_beam.append((Re,Im))
        all_beams.append(one_beam)
    return all_beams
