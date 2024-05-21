import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import get_method_here
from torchsummary import summary
import pandas as pd


from imgbeddings import imgbeddings


def prova_embedding():
    ibed = imgbeddings()

    image_folders = ['coco', 'foodphoto', 'biggan_256', 'biggan_512', 'dalle_2', 'glide', 'stylegan3_t_ffhqu_256x256']
    folder_colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0, 1.0, 1.0), (1.0, 1.0, 0.0),
                     (1.0, 0.0, 1.0), (0.0, 0.0, 0.0)]
    embeddings = []
    labels = []
    for folder_index, image_folder in enumerate(image_folders):
        image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]

        for image_path in image_files:
            # Controlla se il file è un'immagine .jpg o .png
            if image_path.endswith(('.jpg', '.jpeg', '.png')):
                # Carica l'immagine
                image = Image.open(image_path).convert('RGB')

                e = ibed.to_embeddings(image)
                e[0][0:5]
                embeddings.append(e.flatten())
                # Appiattisci l'array in un vettore
                labels.append(folder_index)

    # Applica il t-SNE allo spazio di embedding
    tsne = TSNE(n_components=2, random_state=42)
    embedded_space_tsne = tsne.fit_transform(np.array(embeddings))

    # Visualizza i risultati
    plt.figure(figsize=(10, 8))
    for folder_index, color in enumerate(folder_colors):
        mask = np.array(labels) == folder_index
        folder_name = image_folders[folder_index]
        plt.scatter(embedded_space_tsne[mask, 0], embedded_space_tsne[mask, 1], c=[color], label=f'{folder_name}')

    plt.legend()
    plt.show()


def feature_map(model, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        #transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    processed_image = preprocess(image).unsqueeze(0)
    processed_image = processed_image.to(device)

    with torch.no_grad():
        model.eval()
        model.to(device)
        output, y = model(processed_image)

    return output.squeeze().cpu().numpy(), y




def main():


    weights_path = "./weights"
    model_name = 'Grag2021_latent'
    model_path = os.path.join(weights_path, model_name + '/model_epoch_best.pth')
    arch = 'res50stride1'

    model = get_method_here.def_model(arch, model_path, localize=False)


    # Dizionario per tracciare i colori assegnati a ciascuna cartella
    image_folders = ['coco', 'glide',  'biggan_512',]
    folder_colors = ['red', 'blue', 'yellow', ]

    embeddings = []

    for folder_index, image_folder in enumerate(image_folders):
        image_files = [os.path.join(image_folder, file)
                       for file in os.listdir(image_folder)]

        for image_path in image_files:
            if image_path.endswith(('.jpg', '.jpeg', '.png')):
                e, y = feature_map(model, image_path)
                embeddings.append({
                    'embedding': e.flatten(),
                    'real': y,
                    'label': folder_index,
                    'folder': image_folder,
                })


    grafico_svm(embeddings)



def grafico_tsne(embeddings):
    image_folders = ['coco', 'glide', 'biggan_512', ]
    folder_colors = ['red', 'blue', 'yellow', ]

    df = pd.DataFrame(embeddings)

    # Applica il t-SNE allo spazio di embedding
    tsne = TSNE(n_components=2, random_state=42)
    grafico_embedding = tsne.fit_transform(np.vstack(df['embedding']))



    df['tsne-2d-one'] = grafico_embedding[:, 0]
    df['tsne-2d-two'] = grafico_embedding[:, 1]

    # Visualizza i risultati
    plt.figure(figsize=(10, 8))
    for folder_index, color in enumerate(folder_colors):
        folder_name = image_folders[folder_index]
        subset = df[df['label'] == folder_index]
        plt.scatter(
            subset['tsne-2d-one'],
            subset['tsne-2d-two'],
            c=color,
            label=folder_name,
        )

    plt.legend()
    plt.show()




def grafico_svm(embeddings):
    image_folders = ['coco', 'glide', 'biggan_512', ]
    folder_colors = ['red', 'blue', 'yellow', ]

    df = pd.DataFrame(embeddings)
    X = np.array([item['embedding'] for item in embeddings])
    y = np.array([item['label'] for item in embeddings])
    colors = np.array([folder_colors[item['label']] for item in embeddings])

    # Riduzione della dimensionalità
    pca = PCA(n_components=50)  # Riduzione della dimensionalità con PCA
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    X_tsne = tsne.fit_transform(X_pca)

    # Addestramento del modello SVM
    svm = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    svm.fit(X, y)

    # Visualizzazione
    plt.figure(figsize=(10, 10))
    for i, color in enumerate(folder_colors):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], c=color, label=image_folders[i])

    plt.legend()
    plt.title('t-SNE visualization of embeddings with SVM classification')
    plt.show()




def massimi_minimi_tensore(embeddings):
    #for emb in embeddings:
    #    minimo = min(emb['embedding'])
    #    massimo = max(emb['embedding'])
    #    print(f"Real: {emb['real']}, Folder: {emb['folder']}, minimo: {minimo}, massimo: {massimo}")
    folder_colors1 = ['red', 'blue', 'yellow', ]

    for entry in embeddings:
        minimo = min(entry['embedding'])
        massimo = max(entry['embedding'])
        cartella = entry['label']
        if cartella == 0:
            colore = folder_colors1[0]
        elif cartella == 1:
            colore = folder_colors1[1]
        elif cartella == 2:
            colore = folder_colors1[2]
        plt.scatter(minimo, massimo, color=colore, label=entry['folder'])


    # Aggiungi etichette agli assi
    plt.xlabel('minimo')
    plt.ylabel('massimo')

    # Aggiungi una legenda

    # Mostra il grafico
    plt.show()



def prova_tensor():
    weights_path = "./weights"
    model_name = 'Grag2021_latent'
    model_path = os.path.join(weights_path, model_name + '/model_epoch_best.pth')
    arch = 'res50stride1'

    model = get_method_here.def_model(arch, model_path, localize=False)


    image_path = './dalle_2/DALLE 2022-10-24 00.12.45 - A couple of cats laying next to each other.png'

    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    processed_image = preprocess(image).unsqueeze(0)
    # la cnn prende in input un tensore rgb e un immagine 224 x 224
    print("processed image" + str(processed_image))

    # Estrai l'embedding utilizzando il modello
    with torch.no_grad():
        model.eval()
        output = model(processed_image)
        # il modello mi dà in output un tensore in scala di grigi e un immagine 28x28

    print(output.shape)
    print(output)
    #visualizza_tensore_as_immagine(output)



#def visualizza_tensore_as_immagine(tensore):
    #tensore = np.ndarray((1, 28, 28, 1))  # This is your tensor
    #print(tensore.shape)
    #print(tensore)
    #out = np.squeeze(tensore)
    #plt.imshow(out)
    #plt.show()







if __name__ == "__main__":
    main()
