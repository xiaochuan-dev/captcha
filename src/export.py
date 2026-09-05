import torch
import onnx
from .model import CaptchaCNNTransformer
from .dataset.const import num_classes

def export_to_onnx(model_path='./best.pth', output_path='model.onnx'):

    model = CaptchaCNNTransformer(
        img_h=32,
        img_w=128,
        dim=256,
        depth=6,
        heads=4,
        num_classes=num_classes,
        channels=1,
        dropout=0.2,
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 32, 128)  # (batch, channels, height, width)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        dynamo=False,
        opset_version=18,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

    # 检查
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f'模型已导出为: {output_path}')
    print('ONNX 模型检查通过')

if __name__ == '__main__':
    export_to_onnx()