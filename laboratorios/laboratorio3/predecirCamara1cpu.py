import cv2
import torch
from torchvision import transforms
from PIL import Image
import timm
import torch.nn as nn
import matplotlib.pyplot as plt  # Importar matplotlib para graficar

# Definir las clases manualmente
classes = ['guayaba', 'llanten', 'maracuya', 'molle', 'naranja']

class ModelCustom(nn.Module):
    def __init__(self, n_outputs=5, pretrained=False, freeze=False):
        super().__init__()

        # Descargar el modelo Swin Transformer
        swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)

        # Extraer todas las capas excepto la última
        self.features = nn.Sequential(*list(swin_transformer.children())[:-1])

        # Inicializar un tensor de ejemplo para obtener las dimensiones de salida
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.features(dummy_input)
            self.num_features = dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)

        # Añadir una nueva capa de clasificación
        self.fc = nn.Linear(self.num_features, n_outputs)

        if freeze:
            # Congelar todos los parámetros del modelo Swin Transformer
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)  # Obtener características
        x = x.flatten(start_dim=1)  # Aplanar todas las dimensiones excepto la del batch
        x = self.fc(x)  # Pasar por la capa de clasificación
        return x

    def unfreeze(self):
        # Descongelar todos los parámetros del modelo Swin Transformer
        for param in self.features.parameters():
            param.requires_grad = True

# Función para preprocesar la imagen capturada
def preprocess_image(image, image_size=224):
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

# Función para predecir la clase de una imagen
def predict_image(image_tensor, model, classes, device):
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# Función para capturar una imagen y hacer la predicción
def capture_and_predict(model, classes, device):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la imagen")
        return
    
    # Convertir el frame a una imagen PIL
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Mostrar la imagen capturada
    plt.imshow(pil_image)
    plt.title("Imagen Capturada")
    plt.axis('off')  # Ocultar los ejes
    plt.show()
    
    # Preprocesar la imagen
    image_tensor = preprocess_image(pil_image)
    
    # Predecir la clase
    predicted_class = predict_image(image_tensor, model, classes, device)
    
    print(f"Predicted Class: {predicted_class}")
    
    # Liberar la cámara
    cap.release()

# Crear una instancia del modelo
model = ModelCustom(n_outputs=len(classes), pretrained=True, freeze=True)

# Forzar el uso de la CPU
device = torch.device('cpu')
model.to(device)

# Cargar el estado del modelo guardado
model.load_state_dict(torch.load('model_customTransferFineTuning.pth', map_location=device))

# Menú principal
def main_menu():
    while True:
        print("===== Menú Principal =====")
        print("1. Capturar y predecir")
        print("2. Salir")
        
        choice = input("Selecciona una opción: ")
        
        if choice == '1':
            capture_and_predict(model, classes, device)
        elif choice in ['2', '0']:
            print("Saliendo...")
            break
        else:
            print("Opción no válida, intenta de nuevo.")

if __name__ == "__main__":
    main_menu()

