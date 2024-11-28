import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#генерація випадкових даних
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

#лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

#поліноміальна регресія (ступінь 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)


plt.figure(figsize=(10, 5))
#лінійна регресія
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='green', label='Data')
plt.plot(X, y_lin_pred, color='blue', linewidth=2, label='Linear Fit')
plt.title('Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

#поліноміальна регресія (ступінь 2)
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='green', label='Data')
plt.plot(X, y_poly_pred, color='red', linewidth=2, label='Polynomial Fit (degree 2)')
plt.title('Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()

#коефіцієнти
print("Результати моделювання:\n")
print("Лінійна регресія:")
print(f"  Коефіцієнт нахилу (slope): {lin_reg.coef_[0][0]:.4f}")
print(f"  Вільний член (intercept): {lin_reg.intercept_[0]:.4f}\n")

print("Поліноміальна регресія (ступінь 2):")
print(f"  Коефіцієнт при X: {poly_reg.coef_[0][0]:.4f}")
print(f"  Коефіцієнт при X^2: {poly_reg.coef_[0][1]:.4f}")
print(f"  Вільний член (intercept): {poly_reg.intercept_[0]:.4f}")
