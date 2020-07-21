from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Parameter
from sklearn.cluster import KMeans
from deep_learning_methods.utils.win_data_fun import robot_dataset
from deep_learning_methods.utils.dp_funs import *

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=200,
                      out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,
                      out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,
                      out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,
                      out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048,
                      out_features=10),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=10,
                      out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048,
                      out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,
                      out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,
                      out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,
                      out_features=200),
        )

    def Encoder(self, x):
        return self.encoder(x)

    def forward(self, x):
        out = self.encoder(x)
        x_h = self.decoder(out)

        return x_h

class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters=3, hidden=3, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                self.hidden,
                dtype=torch.float
            ).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()  # soft assignment using t-distribution
        return t_dist


class DEC(nn.Module):
    def __init__(self, n_clusters=3, autoencoder=None, hidden=3, cluster_centers=None, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder.Encoder(x)
        return self.clusteringlayer(x)


def add_noise(mat):
    noise = torch.randn(mat.size()) * 0.2
    noisy_img = mat + noise
    return noisy_img


def pretrain(**kwargs):
    datapth = kwargs['data_path']
    Ypth = kwargs['Y_path']
    mpth = kwargs['mean_path']
    stdpth = kwargs['std_path']

    model = kwargs['model']
    model = model.cuda()
    num_epochs = kwargs['num_epochs']

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_data = robot_dataset(data_path=datapth, Y_path=Ypth, mean_path=mpth, std_path=stdpth)
    train_dataloader = DataLoader(dataset=train_data, batch_size=50, shuffle=True, num_workers=8, pin_memory=True)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, samples in tqdm(enumerate(train_dataloader, start=0)):
            inputs, labels = samples['data'], samples['label']
            inputs = inputs.reshape(inputs.shape[0], -1)
            noisy_inputs = add_noise(inputs).cuda()
            inputs = inputs.cuda()
            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:  # print every 2000 mini-batches
                print("{} epoch, {} objects, loss: {}".format(epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0


def get_test_acc(test_dataloader, mod, gtruth):
    y_h = []
    y_all = []
    for data in tqdm(test_dataloader):
        inputs, y = data['data'], data['label']
        inputs = inputs.reshape(inputs.shape[0], -1)
        inputs = inputs.cuda()
        outputs = mod(inputs)
        out = outputs.argmax(1)
        y_h.append(out.cpu())
        y_all.append(y)
    y_h = torch.cat(y_h).numpy()
    y_all = torch.cat(y_all).numpy()
    acc_test = km_get_acc(y_all, y_h, 3)
    np.save("y_h.npy", y_h)

    return acc_test


def training(**kwargs):
    datapth = kwargs['data_path']
    Ypth = kwargs['Y_path']
    mpth = kwargs['mean_path']
    stdpth = kwargs['std_path']

    testpth = kwargs['test_X_path']
    test_Ypth = kwargs['test_Y_path']

    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    model = model.cuda()
    train_data = robot_dataset(data_path=datapth, Y_path=Ypth, mean_path=mpth, std_path=stdpth)
    train_dataloader = DataLoader(dataset=train_data, batch_size=100, shuffle=False,num_workers=8, pin_memory=True)

    test_data = robot_dataset(data_path=testpth, Y_path=test_Ypth, mean_path=mpth, std_path=stdpth)
    test_dataloader = DataLoader(dataset=test_data, batch_size=200, shuffle=False, num_workers=8, pin_memory=True)

    features = []
    labels = []
    for data in tqdm(train_dataloader):
        x, y = data['data'], data['label']
        x = x.reshape(x.shape[0], -1)
        x = x.cuda()
        y_h = model.autoencoder.Encoder(x)
        y_np = y_h.detach().cpu()
        features.append(y_np)
        labels.append(y)
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    km = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=100000,
                n_clusters=3, n_init=20, n_jobs=-1, precompute_distances='auto',
                random_state=None, verbose=0)
    km.fit(features)
    cluster_centers = km.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)

    train_Y = np.load(Ypth)
    test_Y = np.load(test_Ypth)
    print("test_Y", test_Y.shape)

    rst_train = km.predict(features)
    train_acc = km_get_acc(train_Y, rst_train, 3)
    print('Initial Accuracy: {}'.format(train_acc))

    loss_function = nn.KLDivLoss(size_average=False)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.99)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5)
    print("training: ")

    for epoch in range(num_epochs):
        out_all = []
        for data in tqdm(train_dataloader):
            inputs, _ = data['data'], data['label']
            inputs = inputs.reshape(inputs.shape[0], -1)
            inputs = inputs.cuda()
            outputs = model(inputs)
            targets = model.target_distribution(outputs).detach()
            out = outputs.argmax(1)
            loss = loss_function(outputs.log(), targets) / outputs.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            out_all.append(out.cpu())
        out_all = torch.cat(out_all).numpy()

        print(labels)
        print(out_all)
        acc_train = km_get_acc(train_Y, out_all, 3)
        acc_test = get_test_acc(test_dataloader, model, test_Y)
        print('Epochs: [{}/{}] train_acc:{}, test_acc:{}'.format(epoch+1, num_epochs, acc_train, acc_test))

    torch.save(model.state_dict(), 'dec_parm.pkl')
    print("training finished")
