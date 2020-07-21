from machine_learning_methods.utils.ml_funs import*
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier

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

leg = input("which leg: ")  # left represents the left leg, right represents the right leg
if leg == 'left':
    train_processed = np.delete(train_processed, [7, 8, 9, 10, 11, 12, 13], axis=1)
    test_processed = np.delete(test_processed, [7, 8, 9, 10, 11, 12, 13], axis=1)
if leg == 'right':
    train_processed = np.delete(train_processed, [0, 1, 2, 3, 4, 5, 6], axis=1)
    test_processed = np.delete(test_processed, [0, 1, 2, 3, 4, 5, 6], axis=1)


print("training")

method = input("which method: ")  # SVM represents SVM, LG represents Logistic Regression, MLP represents MLP

if method =='SVM':
    model = svm.SVC(C=2, kernel='rbf', gamma=0.005, tol=1e-10, decision_function_shape='ovr', max_iter=10000,
                    cache_size=10000)
    model.fit(train_processed, train_Y[:, 1])

if method =='LG':
    model = linear_model.LogisticRegression(C=1, max_iter=1000)
    model.fit(train_processed, train_Y[:, 1])

if method == "MLP":
    model = MLPClassifier(solver='sgd', alpha=1e-8, hidden_layer_sizes=(400, 400), max_iter=1000)
    model.fit(train_processed, train_Y[:, 1])


print("training done")

rst_test = model.predict(test_processed)
print(rst_test)
test_acc = get_acc(test_Y[:, 1], rst_test)

print("test_acc: ", test_acc)

rst_train = model.predict(train_processed)
train_acc = get_acc(train_Y[:, 1], rst_train)

print("train_acc: ", train_acc)

