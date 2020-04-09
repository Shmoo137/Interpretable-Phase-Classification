import torch.nn as nn

class CNN1D_tiny_ED(nn.Module):

    # W_out = (W_input - filter_size + 2*padding) / stride + 1
    def __init__(self, num_classes=2):
        super(CNN1D_tiny_ED, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 5, kernel_size=15, stride=1, padding=1),
            # W_out = (924 - 15 + 2*1) / 1 + 1 = 912
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            # W_out = (912 - 4 + 2*0) / 4 + 1 = 228
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(5, 8, kernel_size=5, stride=1, padding=0),
            # W_out = (228 - 5 + 2*0) / 1 + 1 = 224
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            # W_out = (224 - 4 + 2*0) / 4 + 1 = 56
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv1d(8, 10, kernel_size=2, stride=1, padding=1),
            # W_out = (56 - 3 + 2*1) / 1 + 1 = 56
            nn.AvgPool1d(kernel_size=4, stride=4, padding=0),
            # W_out = (56 - 4 + 2*0) / 4 + 1 = 14
            nn.ReLU())
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(14 * 10, num_classes) # output: 2 classes

    def forward(self, x):
        #print("Begin at: ", x.shape)
        x = self.layer1(x)
        #print("1st layer, 5x228: ", x.shape)
        x = self.layer2(x)
        #print("2nd layer, 8x56: ", x.shape)
        x = self.layer3(x)
        #print("3rd layer, 10x14: ", x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)

        return x