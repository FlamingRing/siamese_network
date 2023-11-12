# reference https://qiita.com/saliton/items/5aa6ead4de4d66e8adf5

import argparse
# STEP 1
import torch

# GPU設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# STEP 2
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# STEP 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image 
import os

import torch.nn as nn
import copy
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Siamese Network用MNISTデータセットクラス
class SiameseDataset(Dataset):
    def __init__(self, img_dir):
        df = pd.read_csv("characters2.csv")
        df = df[["UTF-8", "字"]]
        self.pair_index = []                                          # Siamese Network用画像ペアインデックス配列
        self.transform = transforms.Compose([transforms.Resize((64, 64)),
                                                    transforms.Grayscale(), transforms.ToTensor()])
        self.img_dir = img_dir
        self.img_name_list = os.listdir(self.img_dir)
        
        labels = [df.index[df["UTF-8"]==img_name.split(".")[0].split("_")[0]].tolist()[0] for img_name in self.img_name_list]                 # 入力されたデータセットからラベル情報のみ抽出
        # positive_count = 0                                            # Positiveペアのカウント
        # negative_count = 0                                            # Negativeペアのカウント
        
        # self.length = len(self.img_name_list)                               # 入力されたデータセットサイズと同じとする
        # random_index = np.arange(self.length)
        # while positive_count + negative_count < self.length:
        #   np.random.shuffle(random_index)                             # インデックス配列をランダムに並び替え
        #   for i in np.arange(self.length):
        #     if labels[i] == labels[random_index[i]]:                  # 画像ペアのラベルが等しい場合（＝Positive）
        #         if positive_count < self.length / 2:
        #             self.pair_index.append([i, random_index[i], 1])       # 要素の構成：[<画像1のインデックス>, <画像2のインデックス>, <Positive/Negativeフラグ>]
        #             positive_count += 1
        #         else:
        #             continue
        #     else:                                                     # 画像ペアのラベルが異なる場合（＝Negative）
        #         if negative_count < self.length / 2:
        #             self.pair_index.append([i, random_index[i], 0])       # 要素の構成：[<画像1のインデックス>, <画像2のインデックス>, <Positive/Negativeフラグ>]
        #             negative_count += 1
        #         else:
        #             continue

        # rewrite
        self.length = len(self.img_name_list)*2
        for idx, img_name in enumerate(self.img_name_list):
            utf8code, font_idx = img_name.split(".")[0].split("_")
            random_font_indice = np.arange(7) # 7 kinds of font are used
            np.random.shuffle(random_font_indice)
            for random_font_index in random_font_indice:
                if random_font_index == int(font_idx): # same image
                    continue
                else: # same character
                    self.pair_index.append([idx, self.img_name_list.index(utf8code+f"_{random_font_index}.png"), 1])
                    break
            random_indice = np.arange(len(self.img_name_list))
            np.random.shuffle(random_indice)
            for random_index in random_indice:
                if labels[idx] == labels[random_index]: # same character
                    continue
                else: # different character
                    self.pair_index.append([idx, random_index, 0])
                    break

        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.transform(Image.open(os.path.join(self.img_dir, self.img_name_list[self.pair_index[index][0]]))), \
            self.transform(Image.open(os.path.join(self.img_dir, self.img_name_list[self.pair_index[index][1]]))), torch.tensor(self.pair_index[index][2])
    
class TestDataset(Dataset):
    def __init__(self, img_dir):
        df = pd.read_csv("characters2.csv")
        df = df[["UTF-8", "字"]]
        self.transform = transforms.Compose([transforms.Resize((64, 64)),
                                                    transforms.Grayscale(), transforms.ToTensor()])
        self.img_dir = img_dir
        self.img_name_list = os.listdir(self.img_dir)
        self.length = len(self.img_name_list)                               # 入力されたデータセットサイズと同じとする
        self.labels = [df.index[df["UTF-8"]==img_name.split(".")[0].split("_")[0]].tolist()[0] for img_name in self.img_name_list]                
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.transform(Image.open(os.path.join(self.img_dir, self.img_name_list[index]))), torch.tensor(self.labels[index])

# STEP 4

# import torch.nn as nn

# Siamse Networkモデルクラス
class SiameseMnistModel(nn.Module):
    def __init__(self):
        super(SiameseMnistModel, self).__init__()
        self.flatten = nn.Flatten()

        self.encoder = nn.Sequential(
            nn.Linear(64*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64)
        )
        # self.encoder = nn.Sequential(
        #     nn.Linear(28*28, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 32)
        # )
    
    def forward_once(self, x):
        x = self.flatten(x)
        z = self.encoder(x)
        return z
  
    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2
    

class SiameseNet12(SiameseMnistModel):
    def __init__(self):
        super(SiameseNet12, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(64*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.Sigmoid()
        )
    

# STEP 5

# 損失関数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, y):
        difference = z1 - z2
        distance_squared = torch.sum(torch.pow(difference, 2), 1)
        distance = torch.sqrt(distance_squared)       #平均：0.813，最大：1.663，最小：0.023，中央値：0.492
        negative_distance = self.margin - distance
        negative_distance = torch.clamp(negative_distance, min=0.0)
        loss = (y * distance_squared + (1 - y) * torch.pow(negative_distance, 2)) / 2.0
        loss = torch.sum(loss) / z1.size()[0]
        return loss
    
def get_distance(z1, z2):
    difference = z1 - z2
    distance_squared = torch.sum(torch.pow(difference, 2), 0)
    distance = torch.sqrt(distance_squared)
    return distance
    
def train(model_type):
    # Siamese Network学習用Dataset，DataLoaderの作成
    batch_size = 1024
    train_dataset = SiameseDataset("images")
    train_loader = DataLoader(
        train_dataset,                                                    # データセット
        batch_size=batch_size,                                            # バッチサイズ
        shuffle=True                                                      # データセットからランダムに取り出す
    )

    # ペア画像の確認（画像表示）
    # X1, X2, y = iter(train_loader).next()                                 # １バッチ分だけデータを取り出す
    X1, X2, y = next(iter(train_loader))
    fig = plt.figure(tight_layout=True, figsize=(8, 16))
    rows = 4                                                              # 表示行数，バッチサイズよりも小さな値で 
    for i in range(rows):
        print(f"y[{i}]={y[i]}")
        ax = fig.add_subplot(rows, 2, i*2+1)
        ax.imshow(X1[i][0].numpy(), cmap='gray')                          # X1[i].shape = (1, 28, 28)，X1[i][0].shape = (28, 28)
        ax = fig.add_subplot(rows, 2, i*2+2)
        ax.imshow(X2[i][0].numpy(), cmap='gray')                          # X2[i].shape = (1, 28, 28)，X2[i][0].shape = (28, 28)
    # STEP 6
    import torch.optim as optim
    from torchsummary import summary

    # モデルのインスタンス化
    if model_type == "net1":
        model = SiameseMnistModel().to(device)                # GPUを使用するには「.to(device)」が必要
    elif model_type == "net2":
        model = SiameseNet12().to(device)
    else:
        raise RuntimeError("model not correctly defined")
    print(model.parameters)
    summary(model, input_size=[(1, 64*64), (1, 64*64)])   # 入力が２つあるので（ペア画像だから）input_sizeはリストで複数指定する

    # 最適化関数の定義
    # optimizer = optim.SGD(model.parameters(), lr=0.001)    # パラメータ探索アルゴリズム=確率的勾配降下法(SGD), 学習率lr=0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # 損失関数のインスタンス化
    criterion = ContrastiveLoss()                         # 引数として「margin=○○」が指定できる。デフォルト値は「margin=1.0」


    # STEP 7
    # import copy
    # import matplotlib.pyplot as plt
    
    # モデル学習
    total_epochs = 100                                                       # 学習回数
    losses = []                                                       # 表示用損失値配列

    model.train()                                                     # 学習モード
    for epoch in range(total_epochs): 
        print(f"epoch={epoch+1}")
        nan_count = 0
        normal_count = 0

        for X1, X2, y in train_loader:                                  # 学習用DataLoader
            # モデルによる特徴ベクトル算出
            output1, output2 = model(X1.to(device), X2.to(device))

            # 損失関数の計算
            loss = criterion(output1, output2, y.to(device))

            # nan対策（lossにnanが含まれていれば１回前のモデルに戻す）
            if torch.isnan(loss):
                model = prev_model
                #   optimizer = optim.SGD(model.parameters(), lr=0.001)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
                optimizer.load_state_dict(prev_optimizer.state_dict())
                nan_count += 1
                continue
            else:
                prev_model = copy.deepcopy(model)
                prev_optimizer = copy.deepcopy(optimizer)
                normal_count += 1

            # 表示用lossデータの記録
            losses.append(loss.item())

            # 勾配を初期化
            optimizer.zero_grad()
            
            # 損失関数の値から勾配を求め誤差逆伝播による学習実行
            loss.backward()
            
            # 学習結果に基づきパラメータを更新
            optimizer.step()

    print(f"nan/normal: {nan_count}/{normal_count}")
    plt.plot(losses)                                                  # loss値の推移を表示

    # STEP 8
    # モデル評価
    # テスト用DataLoaderの作成（学習用は画像ペアが必要なので後ほど）
    test_dataset = TestDataset("images")
    test_loader = DataLoader(
        test_dataset,                             # データセット
        batch_size=1,                             # バッチサイズは１なので１画像毎にモデルへ入力
        shuffle=True                              # データセットからランダムに取り出す
    )

    model.eval()                                                      # 評価モード
    with torch.no_grad():
        z_test = []
        y_test = []       
        for X, y in test_loader:                                      # テスト用DataLoader     
            output = model.forward_once(X.to(device))
            z_test.append(output)           # テストデータをモデルに通して出力ベクトルを得る
            y_test.append(y)   
        z_test = torch.cat(z_test, dim=0)                             # 多次元torch.tensor要素のリストをtorch.tensor化
        y_test = torch.tensor(y_test)                                 # スカラ要素(int)リストをtorch.tensor化


    # STEP 9
    # from sklearn.manifold import TSNE
    
    def plot_tsne(x, y, colormap=plt.cm.Paired):
        plt.figure(figsize=(13, 10))
        plt.clf()
        tsne = TSNE()
        x_embedded = tsne.fit_transform(x)
        # try:
        #     x_embedded = tsne.fit_transform(x)
        # except ValueError:
        #    print("ValueError detected")
        #    print(x)
        #    print(np.where(np.isnan(x)))
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y, cmap='jet')
        plt.colorbar()
        # plt.show()

    # t-SNEによるベクトル分布表示
    z_test_np = z_test.to('cpu').detach().numpy().copy()                    # t-SNEはdeviceとしてCPUのみに対応（GPUはダメ）
    y_test_np = y_test.to('cpu').detach().numpy().copy()
    plot_tsne(z_test_np, y_test_np)


    # STEP 10
    # from sklearn.cluster import KMeans

    # k-meansによるクラスタリング
    kmeans = KMeans(n_clusters=3107, n_init=10)                                             # クラスタ数は3107で指定します
    kmeans.fit(z_test_np)

    # ラベルの付け替え
    counts = {}
    for cluster_label, test_label in zip(kmeans.labels_, y_test_np):           # cluster_label：クラスタラベルすなわちその画像の予測の結果，test_label：テスト画像ラベル
        if cluster_label not in counts.keys():
            counts[cluster_label] = [0] * 3107                                       # 「<cluster_label>:[0,0,0,0,0,0,0,0,0,0]」をcountsに追加（初期化）
        counts[cluster_label][test_label] += 1             #　各クラスターで各種類（ラベル、真実）のサンプルの数を記録

    mapping = {}      # cluster index -> class index
    for cluster_label in range(3107):
        mapping[cluster_label] = counts[cluster_label].index(max(counts[cluster_label])) # クラスターリングされたクラスターで、一番多いラベルのインデックス
        #   if cluster_label in counts.keys():
        #     mapping[cluster_label] = counts[cluster_label].index(max(counts[cluster_label])) # クラスターリングされたクラスターで、一番多いラベルのインデックス
        #   else:
        #     pass

    # 正解率の計算
    mapped_cluster_label = np.array([mapping[cluster_label] for cluster_label in kmeans.labels_])
    accuracy = sum(mapped_cluster_label == y_test_np) / len(y_test_np)
    print(accuracy)
    torch.save(model.state_dict(), 'checkpoints/net2.pth')

def trained_model_initialization(model_type):
    if model_type == "net1":
        model = SiameseMnistModel().to(device)                # GPUを使用するには「.to(device)」が必要
    elif model_type == "net2":
        model = SiameseNet12().to(device)
    else:
        raise RuntimeError("model not correctly defined")
    model.load_state_dict(torch.load("checkpoints/net2.pth"))
    model.eval()
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                                    transforms.Grayscale(), transforms.ToTensor()])
    return model, transform

def exam(utf8code1, utf8code2):
    model, transform = trained_model_initialization()
    img1_tensor = transform(Image.open(f"images/{utf8code1}_0.png"))
    img2_tensor = transform(Image.open(f"images/{utf8code2}_0.png"))
    img3_tensor = transform(Image.open(f"images/{utf8code1}_1.png"))
    output1 = model.forward_once(img1_tensor.unsqueeze(0).to(device)).squeeze(0)
    output2 = model.forward_once(img2_tensor.unsqueeze(0).to(device)).squeeze(0)
    output3 = model.forward_once(img3_tensor.unsqueeze(0).to(device)).squeeze(0)
    print(f"distance between img1 and img2 is {get_distance(output1, output2)}")
    print(f"distance between img1 and img3 is {get_distance(output1, output3)}")

def generate_labels(file_name, model_type):
    model, transform = trained_model_initialization(model_type)
    df = pd.read_csv("characters2.csv")
    output_file = open(file_name, mode="w")
    for idx in range(len(df.index)):
        img_tensor_list = []
        utf8code = df["UTF-8"].iloc[idx]
        for font_idx in range(7): # 7 kinds of font
            img_tensor_list.append(transform(Image.open(f"images/{utf8code}_{font_idx}.png")))
        img_tensor = torch.sum(torch.cat(img_tensor_list, dim=0), dim=0, keepdim=True)/7
        output = model.forward_once(img_tensor.unsqueeze(0).to(device)).squeeze(0)
        output_file.write(" ".join([str(num) for num in output.tolist()]))
        if idx < len(df.index) - 1:
            output_file.write("\n")
    output_file.close()


parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="train", type=str)
parser.add_argument("--utf8code", default="", nargs='+')
parser.add_argument("--model_type", default="net2", type=str)
parser.add_argument("--file_name", default="label_embedding2.txt", type=str)

args = parser.parse_args()
if args.mode == "train":
    train(args.model_type)
elif args.mode == "exam":
    exam(args.utf8code[0], args.utf8code[1])
elif args.mode == "generate_labels":
    generate_labels(args.file_name, args.model_type)

