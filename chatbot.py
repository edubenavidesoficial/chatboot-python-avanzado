import torch
import json
import random
import matplotlib.pyplot as plt
import datetime
from base_modelo import tokenize, stem, bag_of_words, NeuralNet

def load_model():
    # Cargar el modelo entrenado desde el archivo 'model.pth'
    input_size = len(all_words)
    hidden_size = 8
    output_size = len(tags)
    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model

def predict_class(sentence):
    # Predecir la clase (intento) para una oración dada
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float()
    output = model(X)
    _, predicted = torch.max(output, dim=0)
    tag = tags[predicted.item()]
    return tag

def generar_curva_consumo():
    # Generar una curva de consumo aleatoria y guardarla como imagen PNG
    consumo = [random.randint(5, 30) for _ in range(7)]  # Genera 7 valores aleatorios entre 5 y 30
    plt.figure(figsize=(10, 5))
    plt.plot(consumo, marker='o', linestyle='-', color='b')
    plt.title("Curva de Consumo")
    plt.xlabel("Día")
    plt.ylabel("Consumo (kWh)")
    plt.grid(True)
    plt.savefig('curva_consumo.png')
    plt.close()
    return "Tu curva de consumo ha sido generada. Revisa el archivo 'curva_consumo.png'."

def reportar_corte(ubicacion):
    # Registrar un reporte de corte en el archivo 'reportes_cortes.txt'
    fecha = datetime.datetime.now()
    reporte = f"Corte reportado en {ubicacion} el {fecha}. Se agendará una visita."
    with open("reportes_cortes.txt", "a") as file:
        file.write(reporte + "\n")
    return reporte

def obtener_curvas_aleatorias(suministro):
    # Obtener curvas de consumo aleatorias basadas en el suministro dado
    curvas = {
        "343223": [1, 2, 3, 4, 5],
        "85456": [2, 4, 6, 8]
    }
    return curvas.get(suministro, [])

def chatbot_response(msg):
    # Generar una respuesta del chatbot basada en el mensaje del usuario
    tag = predict_class(msg)
    
    for intent in intents['intents']:
        if tag == intent['tag']:
            response = random.choice(intent['responses'])
            
            if tag == "curva_consumo":
                response += "\n" + generar_curva_consumo()
            elif tag == "reportar_corte":
                response += "\n" + reportar_corte("Ubicación del usuario")
            elif tag == "reducir_planilla":
                response += "\n" + "Algunas formas de reducir tu consumo incluyen: \n- Usar electrodomésticos eficientes.\n- Apagar luces que no uses.\n- Aprovechar la luz natural."
            elif tag == "curvas_aleatorias":
                suministro = msg.split()[-1]  # Extraer el suministro de la última palabra en el mensaje
                curvas = obtener_curvas_aleatorias(suministro)
                response += "\nCurvas para el suministro: " + ", ".join(map(str, curvas))
            
            return response

    return "No entendí tu mensaje."

def chat():
    # Función principal para iniciar el chat con el usuario
    print("Chatbot para empresa eléctrica. Escribe 'salir' para terminar.")
    while True:
        msg = input("Tú: ")
        if msg.lower() == "salir":
            break
        
        response = chatbot_response(msg)
        print("Bot:", response)

if __name__ == "__main__":
    # Cargar intents y preprocesar datos
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
    all_words = []
    tags = []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Cargar el modelo y iniciar el chat
    model = load_model()
    chat()