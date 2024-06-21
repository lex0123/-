from classifier import ViolenceClassifier
from dataset import CustomDataModule

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    data_module = CustomDataModule(batch_size=128)
    data_module.setup()
    classifier=ViolenceClassifier()
    model1=classifier.train_model()
    print(classifier.classify(model1, data_module.test_dataset.convert_to_tensor()))

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
