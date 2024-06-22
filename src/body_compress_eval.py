import torch
import torch.nn.utils.prune as prune
from torchsummary import summary
from src.model import bodypose_model

def prune_model(model, pruning_factor=0.2):
    parameters_to_prune = (
        (model.model0, 'weight'),
        (model.model1_1, 'weight'),
        (model.model1_2, 'weight'),
        (model.model2_1, 'weight'),
        (model.model2_2, 'weight'),
        # 为其他所有层重复此操作...
    )
    for layer, param in parameters_to_prune:
        prune.l1_unstructured(layer, name=param, amount=pruning_factor)
    for layer, param in parameters_to_prune:
        prune.remove(layer, param)

def print_model_summary(model, input_size):
    if torch.cuda.is_available():
        summary(model.cuda(), input_size)
    else:
        summary(model, input_size)

if __name__ == "__main__":
    # 初始化和加载模型
    model_path = '../model/body_pose_model.pth'  # 修改为实际路径
    model = bodypose_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 打印模型summary
    print("Original Model Summary:")
    print_model_summary(model, (3, 368, 368))

    # 进行模型剪枝
    prune_model(model)

    # 再次打印模型summary以查看剪枝后的效果
    print("\nPruned Model Summary:")
    print_model_summary(model, (3, 368, 368))
