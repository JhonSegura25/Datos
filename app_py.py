{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNFYEhC+W4ZD2ovL+Q9UOk/",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JhonSegura25/Datos/blob/main/app_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eliKugxnwyV9"
      },
      "outputs": [],
      "source": [
        "import streamlit as st\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.models import load_model\n",
        "from tensorflow.keras.preprocessing import image\n",
        "import numpy as np\n",
        "\n",
        "class ImageClassifier:\n",
        "    def __init__(self, model_path):\n",
        "        self.model = load_model(model_path)\n",
        "        self.class_names = ['Cura', 'Falla']\n",
        "\n",
        "    def preprocess_image(self, img_path):\n",
        "        img = image.load_img(img_path, target_size=(224, 224))\n",
        "        img_array = image.img_to_array(img)\n",
        "        img_array = np.expand_dims(img_array, axis=0)\n",
        "        img_array /= 255.0\n",
        "        return img_array\n",
        "\n",
        "    def predict_image(self, img_path):\n",
        "        img_array = self.preprocess_image(img_path)\n",
        "        prediction = self.model.predict(img_array)\n",
        "        class_index = int(round(prediction[0, 0]))\n",
        "        return self.class_names[class_index]\n",
        "\n",
        "def main():\n",
        "    st.title(\"Image Classifier App\")\n",
        "    uploaded_file = st.file_uploader(\"Choose an image...\", type=\"jpg\")\n",
        "\n",
        "    if uploaded_file is not None:\n",
        "        st.image(uploaded_file, caption=\"Uploaded Image.\", use_column_width=True)\n",
        "        st.write(\"\")\n",
        "        st.write(\"Classifying...\")\n",
        "\n",
        "        # Instanciar el clasificador de imágenes\n",
        "        model_path = 'model.pkl'  # Asegúrate de que el modelo esté en el mismo directorio\n",
        "        classifier = ImageClassifier(model_path)\n",
        "\n",
        "        # Realizar la predicción\n",
        "        prediction = classifier.predict_image(uploaded_file)\n",
        "\n",
        "        st.write(f\"Prediction: {prediction}\")\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()"
      ]
    }
  ]
}