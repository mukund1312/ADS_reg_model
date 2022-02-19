# importing all the libraies
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.sparse.sputils import matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt
from pandas.core.reshape.reshape import stack_multiple

# creating the data set
Data_set = pd.read_csv("Clasification\Social_Network_Ads.csv")
x = Data_set.iloc[:, :-1].values
y = Data_set.iloc[:, -1].values

# creating the train and the test set
x_test, x_train, y_test, y_train = train_test_split(
    x, y, train_size=.25, random_state=0)


# feature scale
scal = StandardScaler()
x_train = scal.fit_transform(x_train)
x_test = scal.fit_transform(x_test)


# training the model
Clasi = LogisticRegression(random_state=0)
Clasi.fit(x_train, y_train)

# predict
# print(Clasi.predict(scal.transform([[30, 87000]])))

# predict vs real
y_pred = Clasi.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


# lets now make the matrix
cm = confusion_matrix(y_test, y_pred)
print(f' this is matrix below shows the number of prediction made \n{cm}')

# now lets find the accuracy of the model we have build now
ac = accuracy_score(y_test, y_pred)
print(f'this is the accuracy of the model out of 1 given bleow {ac}')


# now lets visulize the train model prediction

X_set, y_set = scal.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
mlt.contourf(X1, X2, Clasi.predict(scal.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
mlt.xlim(X1.min(), X1.max())
mlt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    mlt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
mlt.title('Logistic Regression (Training set)')
mlt.xlabel('Age')
mlt.ylabel('Estimated Salary')
mlt.legend()
# mlt.show()


# now lets visulize the test model prediction
X_set, y_set = scal.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
mlt.contourf(X1, X2, Clasi.predict(scal.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
mlt.xlim(X1.min(), X1.max())
mlt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    mlt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
mlt.title('Logistic Regression (Training set)')
mlt.xlabel('Age')
mlt.ylabel('Estimated Salary')
mlt.legend()
# mlt.show()
