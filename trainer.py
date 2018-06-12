class RecurrentGANTrainer:
    def __init__(self, output_dir, data_loader, imsize):
        self.output_dir = output_dir
        self.data_loader = data_loader
        self.imsize = imsize

    def train(self):
        print self.output_dir
        print self.imsize

    def evaluate(self):
        pass
