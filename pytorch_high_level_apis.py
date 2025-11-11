"""
PyTorch 的多种使用方式对比
从手动循环到高层API
"""

print("="*80)
print("PyTorch 使用方式对比")
print("="*80)
print()

print("方式1: 纯PyTorch - 手动循环（最灵活，代码最长）")
print("-"*80)
print("""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 需要手动写训练循环
for epoch in range(100):
    # 训练
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            val_loss = criterion(y_pred, y_batch)

优点: 完全控制、最灵活
缺点: 代码长、容易出错、需要手动处理很多细节
""")

print("\n" + "="*80)
print("方式2: PyTorch Lightning ⭐ 推荐 - 类似Keras的高层API")
print("-"*80)
print("""
import pytorch_lightning as pl

class LSTMModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2)
        self.fc = nn.Linear(128, 48)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# 一行代码训练！类似 Keras 的 model.fit()
model = LSTMModel()
trainer = pl.Trainer(max_epochs=100, accelerator='gpu')
trainer.fit(model, train_loader, val_loader)

优点: 代码简洁、自动处理GPU、分布式、进度条等
缺点: 需要安装额外的库
类似: Keras/TensorFlow 的 model.fit()
""")

print("\n" + "="*80)
print("方式3: Ignite - 高层训练引擎")
print("-"*80)
print("""
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, MeanSquaredError

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

# 创建训练器和评估器
trainer = create_supervised_trainer(model, optimizer, criterion)
evaluator = create_supervised_evaluator(model, metrics={'mse': MeanSquaredError()})

# 简单的训练
for epoch in range(100):
    trainer.run(train_loader)
    evaluator.run(val_loader)

优点: 灵活性和易用性平衡
缺点: 比Lightning稍复杂
""")

print("\n" + "="*80)
print("方式4: fastai - 最高层API（类似Keras）")
print("-"*80)
print("""
from fastai.tabular.all import *

# 创建DataLoaders
dls = DataLoaders(train_dl, val_dl)

# 创建学习器（Learner）
learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=mae)

# 一行训练！
learn.fit_one_cycle(100, 1e-3)

优点: 极其简单、开箱即用、自动很多优化
缺点: 抽象层次高、自定义较难
最类似: Keras 的易用性
""")

print("\n" + "="*80)
print("方式5: Hugging Face Trainer - 专门用于NLP/Transformers")
print("-"*80)
print("""
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    per_device_train_batch_size=128,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 一行训练
trainer.train()

优点: 专门优化用于Transformers、功能强大
缺点: 主要用于NLP任务
""")

print("\n" + "="*80)
print("总结对比")
print("="*80)
print()
print(f"{'方式':<20} {'代码量':<10} {'灵活性':<10} {'易用性':<10} {'适合场景':<25}")
print("-"*80)
print(f"{'纯PyTorch':<20} {'很多':<10} {'最高':<10} {'较低':<10} {'研究、自定义算法':<25}")
print(f"{'PyTorch Lightning':<20} {'少':<10} {'高':<10} {'高':<10} {'生产、通用深度学习':<25}")
print(f"{'Ignite':<20} {'中等':<10} {'高':<10} {'中等':<10} {'需要灵活性的项目':<25}")
print(f"{'fastai':<20} {'最少':<10} {'中等':<10} {'最高':<10} {'快速原型、教学':<25}")
print(f"{'HF Trainer':<20} {'少':<10} {'中等':<10} {'高':<10} {'NLP、Transformers':<25}")
print(f"{'Keras/TensorFlow':<20} {'最少':<10} {'中等':<10} {'最高':<10} {'通用、生产部署':<25}")
print()

print("="*80)
print("如果用 PyTorch Lightning 写我们的LSTM代码")
print("="*80)
print("""
import pytorch_lightning as pl
import torch
import torch.nn as nn

class LSTMPowerPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 128, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 48)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # 取最后一个时间步
        x = self.dropout2(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = nn.functional.mse_loss(y_pred, y)
        mae = torch.mean(torch.abs(y_pred - y))
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae', mae, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = nn.functional.mse_loss(y_pred, y)
        mae = torch.mean(torch.abs(y_pred - y))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# 使用（类似Keras）
model = LSTMPowerPredictor()

trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    callbacks=[
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        pl.callbacks.ModelCheckpoint(monitor='val_loss'),
    ],
    enable_progress_bar=True,  # tqdm进度条
)

# 一行训练！
trainer.fit(model, train_loader, val_loader)

# 和Keras的 model.fit() 一样简单！
""")

print("\n" + "="*80)
print("结论")
print("="*80)
print("""
PyTorch 不是只能手动循环！

有多种高层API可选：
1. PyTorch Lightning ⭐ - 最推荐，类似Keras，生产级
2. fastai - 最简单，适合快速原型
3. Ignite - 灵活但比Lightning稍复杂
4. HF Trainer - 专门用于NLP

我们项目用 Keras/TensorFlow 是因为：
- model.fit() 非常成熟稳定
- 文档完善、社区大
- 部署方便（TF Serving、TF Lite）
- 对于标准任务足够简单

如果用PyTorch，推荐用PyTorch Lightning，
代码复杂度和Keras差不多！
""")
