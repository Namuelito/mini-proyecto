import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import cv2

class herramientas(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = ["Alicate", "Destornillador", "Huincha", "Llave"]
        self.root_dir = root_dir
        self.image_path = os.listdir(root_dir)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()]) 
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img = cv2.imread(f'{self.root_dir}/{self.image_path[idx]}')
        img = cv2.resize(img, (400, 400))
        label = self.classes.index(self.image_path[idx][0:self.image_path[idx].find("_")])
        return(self.transform(img), label)