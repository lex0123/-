from classifier import ViolenceClassifier
from dataset import CustomDataModule

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    data_module = CustomDataModule(batch_size=128)
    data_module.setup()
    classifier=ViolenceClassifier()

    print(classifier.classify1(data_module.test_dataset.convert_to_tensor()))
    print(classifier.classify(data_module.test_dataset.convert_to_tensor()))