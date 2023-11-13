import os
import numpy as np
import pandas as pd
from PIL import Image

# step 1(finished)
# image_names = os.listdir("images")
# df = pd.read_csv("characters2.csv")
# avg_values = np.zeros((3107, 64, 64))
# matrix = np.zeros((3107, 3107))
# for character_idx in range(3107):
#     utf8code = df.iloc[character_idx]["UTF-8"]
#     avg_value = np.zeros((64, 64), dtype=np.float32)
#     for img_idx in range(7):
#         img = Image.open(os.path.join("images", f"{utf8code}_{img_idx}.png"), mode="r").convert("L")
#         np_img = np.asarray(img)
#         avg_value = avg_value + np_img.astype(np.float32)
#     avg_value = avg_value/7
#     avg_values[character_idx] = avg_value

#     matrix[character_idx][character_idx] = 1
#     for previous_char_idx in range(character_idx):
#         matrix[character_idx][previous_char_idx] = 1 / np.sqrt(np.sum(
#             np.power(avg_values[character_idx] - avg_values[previous_char_idx], 2)))
#     print(f"character idx {character_idx} finished")
# output = pd.DataFrame(matrix, index=df["UTF-8"], columns=df["UTF-8"])
# output.to_csv("distance_matrix.csv")

# step 2
df = pd.read_csv("characters2.csv")
matrix_df = pd.read_csv("distance_matrix.csv", index_col=0)
matrix = matrix_df.to_numpy()
matrix = matrix + np.transpose(matrix) - np.identity(3107)
matrix = matrix * 1000
matrix = np.clip(matrix, 0, 1)
if np.any(matrix > 1):
    raise RuntimeError("similarity greater than 1")
matrix_df = pd.DataFrame(matrix, index=df["字"], columns=df["字"])
matrix_df.to_csv("distance_matrix_kanji.csv")

