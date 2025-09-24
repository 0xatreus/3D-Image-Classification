import torch

def load_model(model_class, checkpoint_path):
    model = model_class(num_classes=10)  # the correct target output size

    # Load the pretrained weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # Remove mismatched classifier head (fc3) from state_dict
    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc3.')}

    # Load all compatible layers
    model.load_state_dict(filtered_dict, strict=False)

    # fc3 layer will remain randomly initialized with correct size
    model.eval()
    return model

def predict(model, point_cloud):
    with torch.no_grad():
        pc_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)
        output = model(pc_tensor)
        pred = output.argmax(dim=1).item()
    return pred
