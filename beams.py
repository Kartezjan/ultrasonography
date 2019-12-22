import math

class beams(col, row, start_angle, end_angle, delta_angle):
    def __init__(self):
        for current_angle in range(start_angle,end_angle, delta_angle):
            r = 0
            con = True
            one_beam = []
            while con==True:
                print("creating beams trajectory...")
                current_angle = current_angle*math.pi/180
                Re = int(r*(cos(current_angle)))
                Im = int(r*(sin(current_angle)))
                if (Re > row and Im > col):
                    con = False
                else:
                    one_beam.append([Re,Im])
            all_beams.append(one_beam)               
    all_beams = []

           
'''def beam(start_angle, end_angle, delta_angle ,r):
    current_angle = start_angle + delta_angle
    Re = math.cos(current_angle)
    Im = math.sin(current_angle)
    result = []
    for i in range(0:r):
        result.append([int(r*Re), int(r*Im)])
    return result'''
