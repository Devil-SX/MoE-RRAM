import torch
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # 定义你的模型结构

    def forward(self, x):
        # 定义模型的前向传播逻辑
        return x

# 创建模型实例
model = YourModel()

# 用于保存中间变量 tensor 的字典
intermediates = {}

# 定义钩子函数，在每个层的 forward 方法执行时被调用
def hook_fn(module, input, output):
    intermediates[module] = output

# 注册钩子函数到每个层
for name, layer in model.named_modules():
    layer.register_forward_hook(hook_fn)

# 将输入传递给模型进行推理
input_tensor = torch.randn(1, 3, 224, 224)
output_tensor = model(input_tensor)

# 保存中间变量 tensor 到文件
torch.save(intermediates, 'intermediates.pth')