import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from sentence_transformers import SentenceTransformer
import torchvision
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib

def image2embedding(image_model, image_file: str) -> torch.Tensor:
    tsr_img = torchvision.io.read_image(image_file)
    tsr_img = tsr_img.unsqueeze(0).float()
    embedding = image_model(tsr_img)
    return embedding.squeeze(0)


def text2embedding(model, text: str) -> torch.Tensor:
    embedding = model.encode(text, convert_to_tensor=True)
    return embedding


def build_model():
    image_model = models.mobilenet_v3_small(pretrained=True).eval()
    text_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    text_model.max_seq_length = 128
    return image_model, text_model


def load_data():
    df = pd.read_csv("/Users/moses/Downloads/shopee-product-matching/train.csv")
    print("****" * 20)
    image_model, text_model = build_model()
    print("load model finished")

    emb_list = []
    ids = []
    image_list = []
    title_list = []
    for i, row in df.iterrows():
        image_path = "/Users/moses/Downloads/shopee-product-matching/train_images/" + row['image']
        title = str(row['title'])
        ids.append(row['posting_id'])
        image_list.append(row['image'])
        title_list.append(title)
        img_emb = image2embedding(image_model, image_path)

        image_embeddings = img_emb.detach().numpy()
        txt_emb = text2embedding(text_model, title)

        text_embeddings = txt_emb
        comb_emb = np.concatenate((text_embeddings, image_embeddings), axis=0)
        norm = np.linalg.norm(comb_emb)
        comb_emb_norm = comb_emb / norm
        emb_list.append(comb_emb_norm)
    return emb_list, ids, image_list, title_list


def image_text_similarity(data):
    kneigh = NearestNeighbors(n_neighbors=5, leaf_size=5000, algorithm="kd_tree")
    kneigh.fit(data)
    return kneigh


def get_neighbors(kneigh, query_emb):
    pos_id_list = []
    neigh_dist, neigh_ind = kneigh.kneighbors(X=query_emb.reshape(1, -1), n_neighbors=10, return_distance=True)
    for ind in neigh_ind:
        print(str(ind))
        for i in ind:
            pos_id_list.append(str(i))
    return pos_id_list


def similary_detach():
    emb_list, ids, image_list, title_list = load_data()
    model = image_text_similarity(emb_list)
    joblib.dump(model, "/Users/moses/Desktop/data/model/img_txt.bin")
    print("finished traing ")
    index = 0
    postIdList = []
    matchList = []
    for val in ids:
        query_emb = emb_list[index]
        pos_id_list = get_neighbors(model, query_emb)
        postIdList.append(val)
        matchList.append(" ".join(pos_id_list))
        index += 1
        if index == 100:
            break

    for i, item in enumerate(postIdList):
        print("title=", title_list[i])
        print("key=", ids[i])
        imagePath = "/Users/moses/Downloads/shopee-product-matching/train_images/" + image_list[i]
        pli_image =Image.open(imagePath)
        plt.figure()
        plt.imshow(pli_image)
        plt.show()
        matching_indexs = matchList[i].split(" ")
        print("***"*20)
        for ind in matching_indexs:
            print(title_list[int(ind)])
            print(ids[int(ind)])
            imgPath = "/Users/moses/Downloads/shopee-product-matching/train_images/" + image_list[int(ind)]
            pli_img1 = Image.open(imgPath,'r')
            plt.figure()
            plt.imshow(pli_img1)
            plt.show()
        if i == 10:
            break

if __name__ == '__main__':
    # image_text_similarity()
    similary_detach()
