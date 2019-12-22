import math

class beams:
    def __init__(self, col, row, start_angle, end_angle, delta_angle):
        print("[+] creating beams trajectory...")
        diagonal = math.ceil(math.sqrt((col**2)+(row**2)))
        for current_angle in range(start_angle,end_angle, delta_angle):
            r = 0
            con = True
            one_beam = []
            for i in range(0,int(diagonal)):
                float(current_angle)
                current_angle = current_angle*math.pi/180
                Re = math.ceil(r*(math.cos(current_angle)))
                Im = math.ceil(r*(math.sin(current_angle)))
                one_beam.append([Re,Im])
                r+=1
            self.all_beams.append(one_beam)               
    all_beams = []