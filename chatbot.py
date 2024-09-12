import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import datetime


with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)


import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

model = NeuralNet(input_size, hidden_size, output_size)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    outputs = model(torch.from_numpy(X_train).float())
    loss = criterion(outputs, torch.from_numpy(y_train).long())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')



def predict_class(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float()
    output = model(X)
    _, predicted = torch.max(output, dim=0)
    tag = tags[predicted.item()]
    return tag

def generar_curva_consumo(consumo):
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
    fecha = datetime.datetime.now()
    reporte = f"Corte reportado en {ubicacion} el {fecha}. Se agendará una visita."
    with open("reportes_cortes.txt", "a") as file:
        file.write(reporte + "\n")
    return reporte

def chatbot_response(msg):
    tag = predict_class(msg)
    
    for intent in intents['intents']:
        if tag == intent['tag']:
            response = random.choice(intent['responses'])
            
            if tag == "curva_consumo":
                consumo = [10, 20, 25, 25, 20, 15, 5]  # Simulación de datos de consumo
                response += "\n" + generar_curva_consumo(consumo)
            elif tag == "reportar_corte":
                response += "\n" + reportar_corte("Ubicación del usuario")
            
            return response

    return "No entendí tu mensaje."

def chat():
    print("Chatbot para empresa eléctrica. Escribe 'salir' para terminar.")
    while True:
        msg = input("Tú: ")
        if msg.lower() == "salir":
            break
        
        response = chatbot_response(msg)
        print("Bot:", response)

if __name__ == "__main__":
    chat()





