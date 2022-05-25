import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from feature import feature

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=20)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels,fontsize=20)
    plt.yticks(xlocations, labels,fontsize=20)
    plt.ylabel('True label',fontsize=20)
    plt.xlabel('Predicted label',fontsize=20)

if __name__ == '__main__':
    datapath = r'../data/4ML/result'
    txtpath = r'../data/4ML/result/train.txt'
    imgs, label, label2, yuzhi, tidu, bianyuan, pingjuntidu = feature(datapath, txtpath)

    XX_train = []
    for i in range(len(imgs)):
        image = cv2.imread((datapath + '/' + imgs[i]), cv2.IMREAD_COLOR)
        img = cv2.resize(image, (256, 256),
                        interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([img], [0, 1], None,
                            [256, 256], [0.0, 255.0, 0.0, 255.0])
        XX_train.append(((hist / 255).flatten()))


    data={"tidu" : tidu,
    "bianyuan" : bianyuan,
    "yuzhi" : yuzhi,
    "pingjuntidu" :pingjuntidu,
    "tuxiang" : XX_train,
    "label" : label2,
    "label_cla" : label
    }
    data=DataFrame(data)
    cols = ["tidu","bianyuan","yuzhi","pingjuntidu"]
    box_df = data['tuxiang'].apply(lambda x: pd.Series(x))
    box_df.columns = ['box_{}'.format(i) for i in range(65536)]
    y = data['label'].astype('float')
    y_cla = data['label_cla'].astype('int')
    X = pd.concat([data[cols], box_df], axis=1)
    x1 = data[cols]
    x2 = box_df
    std = StandardScaler()
    mm = MinMaxScaler()
    X = std.fit_transform(X)
    x1 = std.fit_transform(x1)
    x2 = std.fit_transform(x2)
    X_train, X_test, y_train, y_test = train_test_split(x1, y,test_size=0.2, random_state=1)
    XX_train, XX_test, yy_train, yy_test = train_test_split(x2, y_cla,test_size=0.2, random_state=1)
    print(len(X_train), len(X_test), len(y_train), len(y_test))



    clf = svm.SVC(kernel="rbf").fit(XX_train, yy_train.astype('int'))

    clf = BernoulliNB().fit(XX_train, yy_train.astype('int'))

    clf = KNeighborsClassifier(n_neighbors=11).fit(XX_train, yy_train.astype('int'))
    rfc = RandomForestClassifier(random_state=0,n_estimators = 31,max_depth = 10).fit(XX_train, yy_train.astype('int'))


    predictions_labels = rfc.predict(XX_test)
    print(u'预测结果:')
    print(predictions_labels)
    print(u'算法评价:')
    print(classification_report(yy_test, predictions_labels))
    labels = ['DI', 'TRI', 'VG', 'LG', 'RAN']
    y_true = yy_test
    y_pred = predictions_labels
    tick_marks = np.array(range(len(labels))) + 0.5

    sns.set(color_codes=True)
    sns.set_style("white")
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    plt.figure(figsize=(12, 8), dpi=100)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=14, va='center', ha='center')
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plot_confusion_matrix(cm_normalized, title='Naive Bayes', cmap = plt.cm.Blues)
    plt.savefig('nb_.png')
    plt.show()