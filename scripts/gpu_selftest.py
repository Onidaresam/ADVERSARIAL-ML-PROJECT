import torch, torch_directml
import onnxruntime as ort

print("==== PyTorch DirectML ====")
dml = torch_directml.device()
x = torch.randn(1024, 1024, device=dml)
y = torch.mm(x, x.T)
print("PyTorch DirectML OK:", y.shape)

print("\n==== ONNX DirectML ====")
so = ort.SessionOptions()
sess = ort.InferenceSession("C:\\Dev\\gpu\\models\\demo.onnx", providers=["DmlExecutionProvider"])
print("Providers:", sess.get_providers())
print("GPU OK")
