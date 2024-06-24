from dataset import CustomDataModule
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn


class Basemodel(nn.Module):
    def __init__(self, num_classes=2):
        super(Basemodel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ViolenceClassifier(pl.LightningModule):
    def __init__(self, lr=3e-4):
        super(ViolenceClassifier, self).__init__()
        self.model = Basemodel()  # 在这里初始化模型
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        # 加载模型
        model = cls()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # 计算正确率
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)

        # 记录正确率
        self.log('test_accuracy', accuracy, prog_bar=True)
        return loss

    def classify(self,input_tensor):
        gpu_id = [0]
        lr = 3e-4
        batch_size = 128
        log_name = "resnet18_pretrain_test"
        print("{} gpu: {}, batch size: {}, lr: {}".format(log_name, gpu_id, batch_size, lr))
        # 创建数据模块
        data_module = CustomDataModule(batch_size=128)
        # 设置数据模块
        data_module.setup()
        # 创建模型
        model = ViolenceClassifier(lr=3e-4)
        # 实例化 ModelCheckpoint 回调以保存最佳模型
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename=log_name + '-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )
        # 实例化 TensorBoardLogger 用于记录日志
        logger = TensorBoardLogger("train_logs", name=log_name)
        # 实例化训练器
        trainer = Trainer(
            max_epochs=100,
            accelerator='gpu',
            logger=logger,
            callbacks=[checkpoint_callback]
        )
        # 加载数据
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        trainer.fit(model, train_loader)
        model.eval()  # 将模型设置为评估模式
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.tolist()

    def classify1(self, input_tensor):
        ckpt_path = "train_logs/resnet18_pretrain_test/version_3/checkpoints/resnet18_pretrain_test-epoch=69-val_loss=0.00.ckpt"

        # 从检查点加载模型
        model = ViolenceClassifier.load_from_checkpoint(ckpt_path)

        # 确保模型处于评估模式
        model.eval()

        # 如果有 GPU 可用，将模型和输入张量移动到 GPU
        if torch.cuda.is_available():
            model = model.cuda()
            input_tensor = input_tensor.cuda()

        # 禁用梯度计算进行推理
        with torch.no_grad():
            # 获取预测结果
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.tolist()