import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from feature import feature
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import seaborn as sns

def adj_r_squared(x_test, y_test, y_predict):
    SS_R = sum((y_test - y_predict) ** 2)
    SS_T = sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - (float(SS_R)) / SS_T
    adj_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)
    return adj_r_squared


def regression_method(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    result_ = model.predict(x_train)
    num_regress = len(result)
    MSE = mean_squared_error(result, y_test)
    MSE_ = mean_squared_error(result_, y_train)
    RSS = MSE * num_regress
    MAE = mean_absolute_error(result, y_test)
    MAE_ = mean_absolute_error(result_, y_train)
    R2 = r2_score(y_test, result)
    ARS = adj_r_squared(x_test, y_test, result)
    ARS_ = adj_r_squared(x_train, y_train, result_)
    PR = pearsonr(y_test, result)
    PR_ = pearsonr(y_train, result_)

    print(f'MSE={MSE}')
    print(f'MSE_train={MSE_}')
    print(f'MAE={MAE}')
    print(f'MAE_train={MAE_}')
    print(f'ARS={ARS}')
    print(f'ARS_train={ARS_}')
    print(f'PR={PR}')
    print(f'PR_train={PR_}')
    plt.figure()
    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('R^2: %f' % MSE)
    plt.legend()
    plt.show()
    return result


def scatter_plot(TureValues, PredictValues):
    xxx = [-0.5, 175]
    yyy = [-0.5, 175]
    plt.figure()
    plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
    plt.scatter(TureValues, PredictValues, s=20, c='r', edgecolors='k', marker='o', alpha=0.8)  # 绘制散点图，横轴是真实值，竖轴是预测值
    plt.title('ScatterPlot')
    plt.show()

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False

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

model_LinearRegression = linear_model.LinearRegression()
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
model_Lasso = Lasso()
model_Ridge = Ridge()
model_NB = BayesianRidge()
model_LRR = LogisticRegression()
model_SVR = svm.SVR(kernel= "poly"
                    ,C = 1
                    ,gamma = 1
                    ,degree = 3
                    ,coef0 = 1
                   )
model_SVR1 = svm.SVR(kernel= "rbf"
                    ,C = 10
                    ,gamma = 1
                    #,degree = 3
                    #,coef0 = 1
                   )
model_SVR_line = svm.SVR(kernel= "linear"
                   )
model_KNeighborsRegressor = neighbors.KNeighborsRegressor(n_neighbors = 54)
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=5, max_depth = 3)#这里使用20个决策树
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=100)#这里使用50个决策树
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
model_BaggingRegressor = BaggingRegressor()
model_ExtraTreeRegressor = ExtraTreeRegressor()
NNmodel = MLPRegressor([10,6],learning_rate_init= 0.001,activation='relu',solver='adam', alpha=0.0001,max_iter=30000)



#X_train = PF(degree=2).fit_transform(X_train)
#X_test = PF(degree=2).fit_transform(X_test)
y_pred = regression_method(model_Lasso, X_train, X_test, y_train, y_test)
scatter_plot(y_test,y_pred)

y_test = np.array(y_test)
y_test[0]
_a = []
_b = []
_c = []
_d = []
_e = []

for i in range(len(y_test)):
    if y_test[i] == 23.3:
        _a.append(y_pred[i])
    elif y_test[i] == 49.3:
        _b.append(y_pred[i])
    elif y_test[i] == 51.2:
        _c.append(y_pred[i])
    elif y_test[i] == 149.4:
        _d.append(y_pred[i])
    elif y_test[i] == 172.3:
        _e.append(y_pred[i])
my_data = [_a[:8], _b[:8], _c[:8], _d[:8], _e[:8]]
_a = np.array(_a)
_b = np.array(_b)
_c = np.array(_c)
_d = np.array(_d)
_e = np.array(_e)
all__data = [_a, _b, _c, _d, _e]

sns.set(color_codes=True)
#sns.set_style("dark")
fig, axes = plt.subplots(figsize=(6, 5), dpi = 100)
parts = axes.violinplot(
        my_data, showmeans=False, showmedians=False,
        showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)
quartile1, medians, quartile3 = np.percentile(my_data, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(my_data, quartile1, quartile3)])
whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
inds = np.arange(1, len(medians) + 1)
axes.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
axes.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
axes.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

axes.set_title('Lasso',fontsize=20,weight='bold')

axes.yaxis.grid(True)
axes.set_xticks([y + 1 for y in range(len(my_data))], )
axes.set_xlabel('type of polymer', fontsize=18
                ,weight='bold'
                )
axes.set_ylabel('width of Tg(℃)', fontsize=17
                ,weight='bold'
                )

plt.setp(axes, xticks=[y + 1 for y in range(len(my_data))],
         xticklabels=['RAN', 'DI', 'TRI', 'LG', 'VG'],
         #lablesize=18
         #,weight='bold'
        )

plt.savefig("lasso.png",bbox_inches='tight',pad_inches=0.0)
plt.show()