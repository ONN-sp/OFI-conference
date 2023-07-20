from sklearn import svm

# 定义三个用于分类的点和real标签
X = [[2, 0], [1, 1], [2, 3]]
Y = [0, 0, 1]
# 定义分类器
clf = svm.SVC(kernel = 'linear')
# 训练分类器
clf.fit(X, Y)
print(clf)
print(clf.predict([[2, 2]]))
