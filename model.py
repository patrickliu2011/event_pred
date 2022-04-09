import numpy as np
import torch

class Model(torch.nn.Module):
    def __init__(self, encoder, input_shape=(1,3,32,32), proj_features=256):
        super().__init__()
        self.encoder = encoder
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.zeros(input_shape).to(device=device).float()
        output_shape = self.encoder(dummy_input).shape
        
        in_features = np.prod(output_shape[1:])
        self.start_proj = torch.nn.Sequential(
            torch.nn.Linear(in_features, proj_features),
            torch.nn.ReLU()
        )
        self.end_proj = torch.nn.Sequential(
            torch.nn.Linear(in_features, proj_features),
            torch.nn.ReLU()
        )
        self.sample_proj = torch.nn.Sequential(
            torch.nn.Linear(in_features, proj_features),
            torch.nn.ReLU()
        )
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(3 * proj_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        
    def forward(self, img_start, img_end, img_sample):
        """
        Input:
            x (FloatTensor) - Batch of input images of shape (B, length, C, H, W)
        Output:
            FloatTensor of shape (B, num_classes)
        """
        B = img_start.shape[0]
        images = torch.cat([img_start, img_end, img_sample], dim=0)
        x_start, x_end, x_sample = self.encoder(images).flatten().reshape((3, B, -1))
        x_start = self.start_proj(x_start)
        x_end = self.end_proj(x_end)
        x_sample = self.sample_proj(x_sample)
        x = torch.cat([x_start, x_end, x_sample], dim=-1)
        x = self.clf(x)
        x = torch.sigmoid(x)
        return x
    
if __name__ == "__main__":
    import torchvision
    from torchinfo import summary
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoder = torchvision.models.resnet18(pretrained=False).to(device=device)
    encoder.layer4 = torch.nn.Identity()
    encoder.avgpool = torch.nn.Identity()
    encoder.fc = torch.nn.Identity()
    model = Model(encoder)
    summary(model, [(4, 3, 32, 32), (4, 3, 32, 32), (4, 3, 32, 32)])