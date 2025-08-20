#Wall building
class WallBuilding:
    def __init__(self, container_dim):
        self.container = container_dim
        self.layers = []

    def build_layer(self, boxes):
        layer = []
        width_used = 0
        for box in boxes:
            if width_used + box[0] <= self.container[0]:
                layer.append(box)
                width_used += box[0]
        self.layers.append(layer)