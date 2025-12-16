# clip_processor.py (arquivo separado)
import json
import torch
import clip
from PIL import Image
from pathlib import Path
import pickle
import os

class PillCLIPProcessor:
    def __init__(self, dataset_path="dataset_small"):
        self.dataset_path = Path(dataset_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Cache para features
        self.features_cache = self.dataset_path / "features_cache.pkl"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        metadata_file = self.dataset_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        # Converter para lista uniforme
        if isinstance(data, dict):
            items = []
            for id, content in data.items():
                if isinstance(content, dict):
                    content['id'] = id
                items.append(content)
            return items
        return data
    
    def _find_image_path(self, item):
        """Tenta encontrar o caminho da imagem para um item"""
        # Verifica se já tem caminho explícito
        if isinstance(item, dict):
            if 'image_path' in item and os.path.exists(item['image_path']):
                return item['image_path']
            
            # Procura por campos que possam conter caminhos de imagem
            for value in item.values():
                if isinstance(value, str) and any(ext in value.lower() for ext in ['.jpg', '.png', '.jpeg']):
                    possible_path = self.dataset_path / value
                    if possible_path.exists():
                        return str(possible_path)
        
        # Procura por imagens na pasta
        image_dir = self.dataset_path / "images"
        if image_dir.exists():
            # Tenta encontrar por ID ou nome
            if isinstance(item, dict) and 'id' in item:
                for ext in ['.jpg', '.png', '.jpeg']:
                    possible_path = image_dir / f"{item['id']}{ext}"
                    if possible_path.exists():
                        return str(possible_path)
        
        return None
    
    def process_dataset(self):
        """Processa todas as imagens do dataset e extrai features"""
        if self.features_cache.exists():
            with open(self.features_cache, 'rb') as f:
                return pickle.load(f)
        
        image_features = []
        valid_items = []
        
        for item in self.metadata:
            image_path = self._find_image_path(item)
            
            if image_path and os.path.exists(image_path):
                try:
                    # Carregar e processar imagem
                    image = Image.open(image_path)
                    image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        features = self.model.encode_image(image_input)
                    
                    image_features.append(features.cpu().numpy())
                    valid_items.append({
                        'item': item,
                        'image_path': image_path
                    })
                    
                except Exception as e:
                    print(f"Erro ao processar {image_path}: {e}")
        
        # Salvar cache
        result = {
            'features': np.vstack(image_features),
            'items': valid_items
        }
        
        with open(self.features_cache, 'wb') as f:
            pickle.dump(result, f)
        
        return result