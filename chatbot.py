import torch
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from base_modelo import tokenize, stem, bag_of_words, NeuralNet

# Cargar el archivo de intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Cargar los datos y el modelo entrenado
FILE = "model.pth"
data = torch.load(FILE)

# Extraer los parámetros del diccionario
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Crear una nueva instancia del modelo y cargar su estado
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Chatbot"
last_response = {}  # Variable para rastrear las respuestas anteriores

# Función para generar curva de consumo
def generar_curva_consumo(suministro):
    # Generar una curva de consumo aleatoria para un suministro específico
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 24, 100)  # Simula 24 horas del día
    y = np.random.rand(100) * 100  # Datos aleatorios de consumo
    plt.plot(x, y, label=f'Curva de Consumo - Suministro {suministro}')
    plt.xlabel('Horas del Día')
    plt.ylabel('Consumo (kWh)')
    plt.title(f'Curva de Consumo Diario - Suministro {suministro}')
    plt.legend()
    plt.grid(True)
    
    # Guardar la curva como imagen PNG
    file_name = f'curva_consumo_{suministro}.png'
    plt.savefig(file_name)
    plt.close()  # Cerrar el gráfico después de guardarlo
    return file_name

# Función para obtener respuesta del chatbot
def get_response(msg):
    global last_response
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float()

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Si la confianza es alta
    if prob.item() > 0.75:
        if tag in last_response and last_response[tag]:
            return f"Ya te proporcioné una respuesta sobre {tag}. ¿Necesitas más ayuda?"

        for intent in intents["intents"]:
            if tag == intent["tag"]:
                if tag == "curva_consumo":
                    suministro = input("Por favor, ingresa tu número de suministro: ")
                    file_name = generar_curva_consumo(suministro)
                    last_response[tag] = True
                    return f"Aquí tienes la curva de consumo para el suministro {suministro}. Imagen guardada como {file_name}."

                elif tag == "reducir_planilla":
                    last_response[tag] = True
                    return random.choice(intent["responses"])

                elif tag == "reportar_corte":
                    last_response[tag] = True
                    # Obtener la fecha y hora actual
                    now = datetime.now()
                    fecha_hora = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Formato: 2024-09-12 11:43:44.264
                    # Crear el mensaje con la fecha y hora
                    mensaje = f"Corte reportado en Ubicación del usuario el {fecha_hora}. Se agendará una visita."
                    # Guardar el mensaje en el archivo
                    with open('reportes_cortes.txt', 'a') as file:
                        file.write(f"{mensaje}\n")
                    return random.choice(intent["responses"])

    return "Lo siento, no entiendo lo que estás diciendo."

# Bucle principal del chatbot
print(f"{bot_name} para empresa eléctrica. Escribe 'salir' para terminar.")
while True:
    sentence = input("Tú: ")
    if sentence == "salir":
        break

    resp = get_response(sentence)
    print(resp)
