# 新建convert_format.py转换脚本
import torch

# 加载原始检测器权重
detector_weights = torch.load(r"D:\Downloading\detector-base.pt", map_location='cpu')

# # 提取关键参数
# new_state_dict = {
#     "model_state_dict": detector_weights["model_state_dict"],
#     # 补充其他必要字段（根据训练代码需求）
#     "args": detector_weights.get("args", {"large": False})  # 确保模型规模正确
# }
model_state_dict = detector_weights["model_state_dict"]
# 保存为训练兼容格式
torch.save(model_state_dict, "converted_roberta_model.pth")
