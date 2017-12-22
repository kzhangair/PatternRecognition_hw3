from scipy import io
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def Main(hidden_layer_sizes=(85,85), max_iter=3500, activation="relu", alpha=0.0001):
    fo = io.loadmat("trainingData.mat")
    trainingData = fo['trainingData']
    fo = io.loadmat("testingData.mat")
    testingData = fo['testingData']
    X = trainingData[:, :-1]
    pca = PCA(n_components=15)
    pca.fit(X)
    X_pca = pca.transform(X)
    std_scaler = preprocessing.StandardScaler()
    std_scaler.fit(X_pca)
    X_scaled = std_scaler.transform(X_pca)
    #X_scaled = preprocessing.scale(X_pca)
    Y = trainingData[:, -1]
    
    testX = testingData[:, :-1]
    testX_pca = pca.transform(testX)
    testX_scaled = std_scaler.transform(testX_pca)
    testY = testingData[:, -1]
    
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, \
                        max_iter=max_iter, \
                        activation=activation, \
                        alpha=alpha)
    clf.fit(X_scaled, Y)
    predictions = clf.predict(testX_scaled)
    print("confusion_matrix:")
    print(confusion_matrix(testY, predictions))
    print("classification_report:")
    print(classification_report(testY, predictions))
    print("accuracy score:")
    print(accuracy_score(testY, predictions))
    
