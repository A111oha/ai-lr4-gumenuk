import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

def plot_learning_curve(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)  # Виправлено на test_size
    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training error")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=2, label="Validation error")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.legend()


#лінійна регресія
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
lin_reg = LinearRegression()
plot_learning_curve(lin_reg, X, y)
plt.title("Лінійна регресія")

#поліноміальна регресія (ступінь 2)
plt.subplot(1, 3, 2)
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly_2 = poly_features_2.fit_transform(X)
poly_reg_2 = LinearRegression()
plot_learning_curve(poly_reg_2, X_poly_2, y)
plt.title("Поліноміальна регресія (ступінь 2)")


plt.tight_layout()
plt.show()
