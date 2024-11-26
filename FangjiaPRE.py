import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# 1. 数据预处理
data = pd.read_csv('USA_Housing.csv')
X = data.drop('Price', axis=1).values
#df = df.drop(['A'], axis=1)  # 删除列名为'A'的列
#这里的.values是用来将去掉price后的新数据赋值给x的作用
y = data['Price'].values
#自然而然y的数据就是price了
# 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#这行代码使用 scikit-learn 的 train_test_split 函数将特征矩阵 X 和目标变量 y 划分为训练集和测试集。
'''
在机器学习中,我们经常需要将数据划分为训练集和测试集,以便对模型进行训练和评估。通常情况下,我们会使用随机抽样的方式来进行这个划分。
问题在于,如果每次运行代码时都使用随机抽样,那么得到的训练集和测试集可能会不同。这可能会影响模型的性能评估,因为模型在不同的测试集上的表现可能会有差异。
为了解决这个问题,我们可以设置一个随机种子(random state)。随机种子是一个数字,它决定了随机数生成器的初始状态。如果我们每次都使用相同的随机种子,
那么就可以保证每次运行代码时得到的训练集和测试集划分是一致的。
'''
# 转换为 PyTorch 张量,转化以后可以加速运算也可以优化内存
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # 确保 y_train 是 (batch_size, 1) 的形状，因为原始的y是一维的
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # 确保 y_train 是 (batch_size, 1) 的形状
# 2. 模型建立（这里定义了一个线性回归的类）
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)#（input，output）
    def forward(self, x):
        return self.linear(x)
model = LinearRegression(input_size=X_train.shape[1])
#X_train.shape[1] 就是访问 X_train 的第2个维度,因为第一个维度是样本的个数
'''
class LinearRegression(nn.Module):
这行代码定义了一个名为 LinearRegression 的 PyTorch 模型类。
该类继承自 nn.Module，这是 PyTorch 中所有神经网络模型的基类。
def __init__(self, input_size):
这是 LinearRegression 类的构造函数。
input_size 参数表示输入特征的维度。
super().__init__()
这行代码调用了父类 nn.Module 的构造函数。
这是在定义自定义模型类时的标准做法。
self.linear = nn.Linear(input_size, 5)
在构造函数中,我们创建了一个 nn.Linear 层。input_size 指定输入特征的维度。(这里需要结合自己的数据进行修改)
这个线性层将作为模型的核心部分。
def forward(self, x):
这个方法定义了模型的前向传播过程。
当给定输入 x 时,该方法将通过 self.linear 层进行计算,并返回输出。
model = LinearRegression(input_size=X_train.shape[5])
最后,我们创建了一个 LinearRegression 模型实例。
input_size=X_train.shape[5] 表示输入特征的维度,这里取自训练集特征矩阵 X_train 的列数。
'''
# 3. 模型训练
criterion = nn.MSELoss()#选择一个自己想要的损失函数，可以结合数据进行更改
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
'''
这行代码创建了一个随机梯度下降(Stochastic Gradient Descent, SGD)优化器对象。
model.parameters() 告诉优化器需要优化的模型参数,即线性层的权重和偏置。
lr=0.01 设置了学习率为 0.01。学习率是优化过程中的一个关键超参数,它控制着每次参数更新的幅度
'''
loss_values=[]
num_epochs = 1000#训练次数
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X_train)#这里的意思就是我们将x的训练数据输入到实例化的model中，就会得到一个预测值
    loss = criterion(y_pred, y_train)#然后我们计算预测值和真实值之间的损失
    loss_values.append(loss.detach().item())
    '''
    在 PyTorch 中,当我们计算损失函数时,PyTorch 会自动构建一个计算图,用于跟踪计算过程中涉及的变量和操作。
    loss.detach() 的作用是从这个计算图中分离出 loss 张量,创建一个新的张量
    .item() 方法用于从张量中提取单个标量值。
    '''
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''
optimizer.zero_grad()
在每次迭代中,我们需要先将优化器的梯度清零。
这是因为 PyTorch 会在反向传播过程中累积梯度,而不是每次都清零。
如果不清零,新的梯度会与上一次的梯度累积在一起,这可能会导致优化过程出现问题。
loss.backward()
这行代码会触发模型的反向传播过程。
它会计算当前损失函数关于模型参数的梯度,并将这些梯度存储在每个参数的 .grad 属性中。
optimizer.step()
这行代码会让优化器使用刚刚计算得到的梯度来更新模型参数。
具体来说,优化器会根据所选择的优化算法(在这里是 SGD)和学习率,来调整模型参数的值,以减小损失函数
'''
# 4.模型评估（主要看在测试集上的效果）
with torch.no_grad():#在我们进行测试的时候就不用再更新梯度值了
    y_pred = model(X_test)#将X_test输入到已经训练了1000次后的模型当中去
    mse = criterion(y_pred, y_test)
    r2 = 1 - mse / torch.var(y_test)
    print(f'R-squared: {r2:.2f}')
    print(f'Mean Squared Error: {mse:.2f}')
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()