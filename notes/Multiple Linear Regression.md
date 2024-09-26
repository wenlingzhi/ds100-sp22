# Multiple Linear Regression

## 介绍

### 1. **预测向量（Prediction Vector，\(\mathbf{y}\)）**：
   - 这是因变量（目标变量）组成的列向量，表示每个观测值的实际结果。
   - 如果有 \(n\) 个数据点，预测向量的形式是：
     $$
     \mathbf{y} =
     \begin{pmatrix}
     y_1 \\
     y_2 \\
     \vdots \\
     y_n
     \end{pmatrix}
     $$

   - 这里 \(y_i\) 表示第 \(i\) 个观测点的实际值。

### 2. **设计矩阵（Design Matrix，\(\mathbf{X}\)）**：
   - 设计矩阵是自变量的数据表示，包含了所有自变量的值。每一行对应一个观测数据点，每一列对应一个自变量。
   - 如果有 \(n\) 个观测值和 \(p\) 个自变量，设计矩阵的形式是：
     $$
     \mathbf{X} =
     \begin{pmatrix}
     1 & X_{11} & X_{12} & \dots & X_{1p} \\
     1 & X_{21} & X_{22} & \dots & X_{2p} \\
     \vdots & \vdots & \vdots & \ddots & \vdots \\
     1 & X_{n1} & X_{n2} & \dots & X_{np}
     \end{pmatrix}
     $$

   - 其中第一列是全为 1 的常数项，代表截距，后面的列代表每个自变量的值。

### 3. **参数向量（Parameter Vector，\(\boldsymbol{\theta}\) 或 \(\boldsymbol{\beta}\)）**：
   - 这是模型的回归系数（参数），包括截距和各个自变量的系数。
   - 参数向量的形式是：
     $$
     \boldsymbol{\theta} =
     \begin{pmatrix}
     b_0 \\
     b_1 \\
     b_2 \\
     \vdots \\
     b_p
     \end{pmatrix}
     $$

   - 其中 \(b_0\) 是截距，\(b_1, b_2, \dots, b_p\) 是各个自变量的系数。

### 4. **线性模型表示**：
   使用线性代数的方式，可以将多元线性回归表示为：
$$
\mathbf{y} = \mathbf{X} \boldsymbol{\theta} + \epsilon
$$
   其中：
   - $\mathbf{y}$是目标向量（观测值）。
   - $\mathbf{X}$是设计矩阵。
   - $\boldsymbol{\theta}$ 是参数向量。
   - $\epsilon$ 是误差项。

### 5. **最小二乘法的解**：
   通过最小化损失函数（通常是均方误差），可以通过以下公式计算回归系数 $\boldsymbol{\theta}$：
$$
   \boldsymbol{\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
   
$$


   - $\mathbf{X}^T$是设计矩阵的转置。
   - $\mathbf{X}^T \mathbf{X})^{-1}$ 是 \(\mathbf{X}^T \mathbf{X}\) 的逆矩阵。

### 6. **预测**：
   训练好模型之后，可以通过计算预测值：
$$
   \hat{\mathbf{y}} = \mathbf{X} \boldsymbol{\theta}
$$
   $\hat{\mathbf{y}}$ 是预测值向量，表示每个输入数据点通过回归模型得出的预测结果。

##  L2 vector norm

L2范数（L2 vector norm），也称为欧几里得范数（Euclidean norm），是向量长度的一种度量方式。它表示从原点到向量终点的欧几里得距离，即向量的每个分量的平方和再开方。

### 计算公式：
给定一个向量 $\mathbf{v} = [v_1, v_2, \dots, v_n]$，其L2范数定义为：
$$
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}
$$


### 解释：
- **几何意义**：L2范数衡量的是向量从原点到其位置的直线距离。可以把它理解为我们平常理解的“长度”或“距离”。
- **性质**：L2范数总是非负的，并且只有在向量的所有元素都是零时，L2范数为零。

### 示例：
对于二维向量 $\mathbf{v} = [3, 4]$，其L2范数为：
$$
\|\mathbf{v}\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$
这个值表示向量从原点到点 \((3, 4)\) 的距离是5。

L2范数在许多机器学习算法中也非常重要，例如在正则化方法（如L2正则化或Ridge回归）中，用来控制模型的复杂度。如果你对L2范数的应用有更多问题，随时问我！

## 误差公式

评估简单线性回归模型的常用公式包括**均方误差（MSE）**和**R²（决定系数）**。

### 1. **均方误差（MSE）**:
MSE 用于衡量模型预测值与实际值之间的误差。计算公式为：
$$

MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- $ y_i $ 是实际值。
- $\hat{y}_i$  是预测值。
- $n$  是样本数量。

MSE 越小，模型的预测效果越好。

### 2. **R²（决定系数）**:
R² 用于衡量模型解释自变量和因变量之间关系的程度，取值范围在 0 和 1 之间。计算公式为：
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

-  $\bar{y}$  是实际值的平均值。

R² 越接近 1，说明模型对数据的拟合效果越好。