import torch
import torch.nn as nn

class PCHModel(nn.Module):
    def __init__(self, args, num_class, dim1):
        super(PCHModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dim1, dim1),
            nn.BatchNorm1d(dim1),
            nn.ReLU(inplace=False),
        )

        self.projector = nn.Sequential(
            nn.Linear(dim1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=False),

            nn.Linear(1024, args.nbit),
            nn.Tanh()
        )


    def forward(self, source, target):
        source_f1 = self.encoder(source)
        source_h = self.projector(source_f1)
        # source_clf = self.classifier(source_f1)

        target_f1 = self.encoder(target)
        target_h = self.projector(target_f1)
        # target_clf = self.classifier(target_f1)

        return source_f1, source_h, target_f1, target_h

    def predict(self, x):
        x = self.encoder(x)
        h = self.projector(x)
        return h