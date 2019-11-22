import torch.nn as nn
import torch
import torch.utils.data as Data
import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_data(csv_path):
    train_data = pd.read_csv(csv_path, sep=',', header=None)
    data = train_data.values.astype(np.float)[:, 0:128]
    labels = train_data.values.astype(np.int)[:, 128:]

    total_data = np.hstack((data, labels))

    np.random.shuffle(total_data)

    total_size = len(total_data)
    train_size = int(0.8 * total_size)
    # test_size = total_size - train_size

    train = total_data[0:train_size, :-1]
    test = total_data[train_size:, :-1]
    train_label = total_data[0:train_size, -1].reshape(-1, 1)
    test_label = total_data[train_size:, -1].reshape(-1, 1)
    return data, labels, train, test, train_label, test_label


# 网络类
class DNNNet(nn.Module):
    def __init__(self):
        super(DNNNet, self).__init__()
        self.fc = nn.Sequential(  # 添加神经元以及激活函数
            nn.Linear(128, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 2)
        )
        self.mse = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(params=self.parameters(), lr=0.001)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

    def train(self, x, label):
        out = self.forward(x)
        loss = self.mse(out, label)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def test(self, test_):
        return self.fc(test_)


if __name__ == '__main__':
    data, labels, train, test, train_label, test_label = get_data('G:\\share\\training_data\\func.csv')
    net = DNNNet()
    net.cuda()
    train_dataset = Data.TensorDataset(torch.from_numpy(train).float(), torch.from_numpy(train_label).long())
    BATCH_SIZE = 1
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(100):
        for step, (x, y) in enumerate(train_loader):
            y = torch.reshape(y, [BATCH_SIZE])
            net.train(x.cuda(), y.cuda())
            if epoch % 20 == 0:
                print('Epoch: ', epoch, '| Step: ', step, '| batch y: ', y.numpy())

    out = net.test(torch.from_numpy(data).cuda().float())

    prediction = torch.max(out, 1)[1].cuda()  # 1返回index  0返回原值
    pred_y = prediction.data.cpu().numpy()
    test_y = labels.reshape(1, -1)
    target_y = torch.from_numpy(test_y).long().data.cpu().numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    print("accuracy=", accuracy)
