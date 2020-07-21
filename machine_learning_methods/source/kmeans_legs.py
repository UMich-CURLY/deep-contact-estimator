from machine_learning_methods.utils.ml_funs import*
from sklearn.cluster import KMeans

PATH = input("path: ")
# PATH = '/home/yicheng/Documents/imu project/contact_estimate'

train = np.load("{}/machine_learning_methods/data/train_data_0.npy".format(PATH))
test = np.load("{}/machine_learning_methods/data/test_data_0.npy".format(PATH))
train_Y = np.load("{}/machine_learning_methods/data/train_Y.npy".format(PATH))
test_Y = np.load("{}/machine_learning_methods/data/test_Y.npy".format(PATH))

data_mean = np.mean(train, 0).reshape(1, -1)
data_std = np.std(train, 0).reshape(1, -1)

train_processed = train.copy()
test_processed = test.copy()

train_processed = (train_processed-data_mean)/data_std
test_processed = (test_processed-data_mean)/data_std

train_Y = combine_Y(train_Y)
test_Y = combine_Y(test_Y)
print("training: ")
model = KMeans(n_clusters=3, tol=0.00001, n_init=10, init='random', n_jobs=-1)
model.fit(train_processed)
rst_test = model.predict(test_processed)

test_acc = km_get_acc(test_Y.astype(int), rst_test, 3)
print("test_acc: ", test_acc)

rst_train = model.predict(train_processed)
train_acc = km_get_acc(train_Y.astype(int), rst_train, 3)
print("train_acc: ", train_acc)

