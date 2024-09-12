# Chatbot Python Avanzado

Este es un proyecto de chatbot avanzado en Python que utiliza PyTorch para la clasificación basada en aprendizaje automático. El chatbot está diseñado para manejar diversas intenciones, como generar curvas de consumo y reportar cortes de energía.

## Descripción del Proyecto

Este chatbot está diseñado para interactuar con los usuarios y realizar tareas específicas basadas en sus solicitudes. Está entrenado para reconocer y responder a diferentes intenciones utilizando un modelo de aprendizaje automático.

## Estructura del Proyecto

Resumen:

base_modelo.py: Este archivo entrena y guarda el modelo del chatbot.
chatbot.py: Es el encargado de hablar con el usuario, usando el modelo entrenado para responder preguntas.
Opciones adicionales:

Curvas aleatorias: El método generar_curva_consumo() ahora crea datos de consumo de manera aleatoria cada vez que lo llamas.
Curvas por suministro: Puedes obtener curvas específicas para diferentes suministros usando obtener_curvas_aleatorias().

## Instalación

1. **Clona el repositorio:**

   ```bash
   git clone https://github.com/edubenavidesoficial/chatboot-python-avanzado.git
