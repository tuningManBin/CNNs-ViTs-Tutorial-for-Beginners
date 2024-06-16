# Comparison-of-Neural-Networks-in-Classification-Tasks
This repository will test the classification accuracy of different neural networks trained in the same manner on the CIFAR-10 and MNIST datasets.
论文中模型为了达到SOTA水平存在许多训练tricks, 为了快速评估一类模型是否适合您的任务, 我们定义了一种标准训练函数：相同的优化器、相同的学习率策略、相同的数据增强、相同的batch_size、不使用预训练......ALL FACTORS IS THE SAME.
你可以用这个训练函数训练您的模型，测试您的架构与主流模型在准确率上的差异，以便观察您的模型是否存在达到SOTA的可能。
更多常见模型正在施工中！
