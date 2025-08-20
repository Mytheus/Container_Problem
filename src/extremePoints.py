#Extreme points
class ExtremePoints:
    def __init__(self, container_dim):
        self.container = container_dim
        self.points = [(0,0,0)]
        self.boxes = []  # lista de tuplas: (point, box, id)

    def fits(self, point, box):
        x, y, z = point
        L, W, H = box
        if x + L > self.container[0] or y + W > self.container[1] or z + H > self.container[2]:
            return False
        for (px, py, pz), (bL, bW, bH, _) in self.boxes:
            if (x < px + bL and x + L > px and
                y < py + bW and y + W > py and
                z < pz + bH and z + H > pz):
                return False
        return True

    def update_points(self, point, box):
        x, y, z = point
        L, W, H = box
        new_points = [
            (x + L, y, z),
            (x, y + W, z),
            (x, y, z + H)
        ]
        for px, py, pz in new_points:
            if px <= self.container[0] and py <= self.container[1] and pz <= self.container[2]:
                self.points.append((px, py, pz))
        if point in self.points:
            self.points.remove(point)
        # ordenar pontos: primeiro mais baixo (z), depois y, depois x
        self.points.sort(key=lambda p: (p[2], p[1], p[0]))

    def place_box(self, box, box_id=None):
        for point in self.points:
            if self.fits(point, box):
                self.boxes.append((point, (box[0], box[1], box[2], box_id)))
                self.update_points(point, box)
                return True
        return False

    def utilization(self):
        return sum(b[1][0]*b[1][1]*b[1][2] for b in self.boxes)
