import torch
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import torch.nn as nn
from torch.utils import data
import numpy as np
import torchvision.transforms as transforms

# 确保在运行前创建了 archive1 文件夹，并且里面有 process_training.h5 和 process_testing.h5

def h5_to_tensor(h5_file_path):
    h5_tensor={}
    with h5py.File(h5_file_path,'r') as h5:
        keys=list(h5.keys())
        for key in keys:
            data = h5[key][:]
            if key == 'labels':
                # Labels should be of type Long (int64) for CrossEntropyLoss
                h5_tensor[key]=torch.from_numpy(data).long() # Modified: .long() for labels
            else:
                # Other data (images, keypoints) should be float32
                h5_tensor[key]=torch.from_numpy(data).float() #
    return h5_tensor

# 加载数据
print("正在加载H5数据...")
training_data=h5_to_tensor("archive1/process_training.h5")
testing_data=h5_to_tensor("archive1/process_testing.h5")
print("H5数据加载完成。")

'''img_train = training_data['images'][:]'''
keypoints_train=training_data['keypoints'][:]
labels_train = training_data['labels'][:]

'''img_test = testing_data['images'][:]'''
keypoints_test=testing_data['keypoints'][:]
labels_test = testing_data['labels'][:]

# 动态获取类别数量
num_classes = len(labels_train.unique())
print(f"检测到的手势类别数量: {num_classes}")

# 自定义数据集类
class customDataset(data.Dataset):
    #def __init__(self,images, keypoints, labels, use_keypoints=True, transform=None):
    def __init__(self,keypoints, labels, use_keypoints=True, transform=None):
        super().__init__()
        '''self.images = images'''
        self.keypoints = keypoints
        self.labels = labels
        self.use_keypoints = use_keypoints
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = self.labels[index]
        
        if self.use_keypoints:
            data_input = self.keypoints[index]
        '''else:
            img = self.images[index].numpy()
            img = transforms.ToPILImage()(img.transpose(2, 0, 1))
            if self.transform:
                data_input = self.transform(img)
            else:
                data_input = transforms.ToTensor()(img)'''

        return data_input, label

# 关键点手势识别器模型
class KeypointGestureRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(KeypointGestureRecognizer, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(21 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

# 评估指标累加器
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 准确率计算函数
def accuracy(y_hat, y):
    if y_hat.shape[1] > 1:
        return (y_hat.argmax(axis=1) == y.type(y_hat.dtype)).sum().type(y.dtype).item()
    else:
        return (y_hat.type(y.dtype) == y.type(y_hat.dtype)).sum().type(y.dtype).item()

# 训练函数
def train(net, train_iter, test_iter, loss, num_epochs, trainer, device):
    net.to(device)
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for X, y in tqdm(train_iter, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)"):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.sum().backward()
            trainer.step()
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        
        train_loss = metric[0]/metric[2]
        train_acc = metric[1]/metric[2]

        test_acc = evaluate_accuracy(net, test_iter, device)
        
        print(f'Epoch {epoch + 1}, '
              f'Train Loss: {train_loss:.3f}, '
              f'Train Acc: {train_acc:.3f}, '
              f'Test Acc: {test_acc:.3f}')

# 评估准确率函数
def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 预测并可视化结果
'''def predict_and_visualize(net, test_iter, img_original_test, num_display=5, device="cpu", use_keypoints=True):
    net.to(device)
    net.eval()
    
    def get_gesture_label_name(label_idx):
        # 可以根据实际的标签值映射到对应的手势名称
        return f"{label_idx}"

    with torch.no_grad():
        for i, (X, y_true) in enumerate(test_iter):
            if i<=2:#第三批
                continue
            X_device = X.to(device)
            y_true_device = y_true.to(device)

            y_hat = net(X_device)
            y_pred = y_hat.argmax(axis=1)
            
            display_count = min(num_display, X.shape[0])
            
            fig, axes = plt.subplots(1, display_count, figsize=(3 * display_count, 4))
            if display_count == 1:
                axes = [axes]

            for j in range(display_count):
                true_label_idx = y_true[j].item()
                pred_label_idx = y_pred[j].item()
                
                # 获取原始图像数据，并将其转换为 uint8 类型以避免 imshow 警告
                # img_original_test 已经是通过 h5_to_tensor 加载的 float 类型
                # 假设它仍然在 0-255 范围内，将其强制转换为 uint8
                img_to_display = img_original_test[i * test_iter.batch_size + j].numpy().astype(np.uint8)

                axes[j].imshow(img_to_display)
                axes[j].set_title(f"real: {get_gesture_label_name(true_label_idx)}\npredict: {get_gesture_label_name(pred_label_idx)}")
                axes[j].axis('off')
            
            plt.tight_layout()
            plt.show()
            break # 只展示第一个批次的 num_display 个样本
'''

if __name__ == "__main__":
    batch_size = 32
    total_epochs =50
    learning_rate = 0.001

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备: {device}")

    # 数据集和数据加载器
    #trainData = customDataset(img_train, keypoints_train, labels_train, use_keypoints=True)
    trainData = customDataset(keypoints_train, labels_train, use_keypoints=True)
    train_iter = data.DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=0)
    
    #testData = customDataset(img_test, keypoints_test, labels_test, use_keypoints=True)
    testData = customDataset(keypoints_test, labels_test, use_keypoints=True)
    test_iter = data.DataLoader(testData, batch_size=batch_size, shuffle=False, num_workers=0)

    net = KeypointGestureRecognizer(6)
    #net.load_state_dict(torch.load("keypoint_gesture_recognizer.pth", map_location="cpu"))  
    #net.eval()
    #net.to(torch.device("cpu"))  
    
    error = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    print("\n开始训练模型...")
    train(net, train_iter, test_iter, error, total_epochs, optimizer, device)

    torch.save(net.state_dict(), 'keypoint_gesture_recognizer.pth')

    #print("\n正在进行预测和可视化...")
    #predict_and_visualize(net, test_iter, img_test, num_display=5, device=device, use_keypoints=True)