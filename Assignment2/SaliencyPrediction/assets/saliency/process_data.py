import pandas as pd
import os
import re
import json
dir = os.path.dirname(os.path.abspath(__file__))
# get images' name

image_path = os.path.join(dir, "image")
text_path = os.path.join(dir, "text.csv")
whole_text_path = os.path.join(dir, "wholeText.csv")

text_df = pd.read_csv(text_path)
whole_text_df = pd.read_csv(whole_text_path)
# print(text_df)
""" 
图片名字后缀：pure_0 整体_1 非显著_2 显著_3 whole就是原名字
_0:pure 纯图片 实验条件不含文字
_1:整体(all) 对应论文里面type4
_2:非显著 对应论文里面type3
_3:显著 对应论文里面type2
无：whole 对应论文里面type1, 好像没找到text ? # TODO

the images in image_path has:
- image_number.png
- 

"""

# design reg exprs, to extract image numbers, image numbers's example is: 000000000071.png, 000000000071_1.png,
# 1031973097.png, we just extract the number before "_" or ".png" into a set
image_number_reg = re.compile(r"(\d+)(?:_|\.)")
image_numbers = set()
for image_name in os.listdir(image_path):
    image_number = image_number_reg.match(image_name).group(1)
    image_numbers.add(image_number)
# print(len(image_numbers))

# then for each image number in image_numbers, we find the corresponding text in text_df. We use a dict to store
# the result
image_text_dict = {}
for image_number in image_numbers:
    # print(image_number)
    # find pure, maybe not exists
    image_text_dict[image_number] = {}
    pure = image_number + "_0.png"
    if pure in os.listdir(image_path):
        image_text_dict[image_number]["pure"] = pure
    # find whole, that is, image_number_1.png; maybe not exists
    whole = image_number + ".png"
    if whole in os.listdir(image_path):
        image_text_dict[image_number]["whole"] = {}
        image_text_dict[image_number]["whole"]["img_path"] = whole
        image_text_dict[image_number]["whole"]["text"] = whole_text_df[(whole_text_df["image"] == int(image_number))]["text"].values[0]

    all = image_number + "_1.png"
    if all in os.listdir(image_path):
        image_text_dict[image_number]["all"] = {}
        image_text_dict[image_number]["all"]["img_path"] = all
        image_text_dict[image_number]["all"]["text"] = text_df[(text_df["image"] == int(image_number)) & (text_df["描述种类"] == "整体")]["text"].values[0]
    # find non_salient, that is, image_number_2.png; maybe not exists
    non_salient = image_number + "_2.png"
    if non_salient in os.listdir(image_path):
        image_text_dict[image_number]["non_salient"] = {}
        image_text_dict[image_number]["non_salient"]["img_path"] = non_salient
        image_text_dict[image_number]["non_salient"]["text"] = text_df[(text_df["image"] == int(image_number)) & (text_df["描述种类"] == "非显著")]["text"].values[0]
    # find salient, that is, image_number_3.png; maybe not exists
    salient = image_number + "_3.png"
    if salient in os.listdir(image_path):
        image_text_dict[image_number]["salient"] = {}
        image_text_dict[image_number]["salient"]["img_path"] = salient
        image_text_dict[image_number]["salient"]["text"] = text_df[(text_df["image"] == int(image_number)) & (text_df["描述种类"] == "显著")]["text"].values[0]


# store as json file
json.dump(image_text_dict, open(os.path.join(dir, "image_text_dict.json"), "w"), indent=4, ensure_ascii=False)


# split image_text_dict into four different types, store as seperate csv files, use pd dataframe
all_df = pd.DataFrame(columns=["image_number", "img_path", "text"])
whole_df = pd.DataFrame(columns=["image_number","img_path", "text"])
salient_df = pd.DataFrame(columns=["image_number", "img_path", "text"])
non_salient_df = pd.DataFrame(columns=["image_number", "img_path", "text"])
pure_df = pd.DataFrame(columns=["image_number", "img_path"])

for image_number in image_text_dict.keys():
    if "all" in image_text_dict[image_number].keys():
        all_df.loc[len(all_df)] = [image_number, image_text_dict[image_number]["all"]["img_path"],
                                       image_text_dict[image_number]["all"]["text"]]
    if "salient" in image_text_dict[image_number].keys():
        salient_df.loc[len(salient_df)] = [image_number, image_text_dict[image_number]["salient"]["img_path"],
                                       image_text_dict[image_number]["salient"]["text"]]
    if "non_salient" in image_text_dict[image_number].keys():
        non_salient_df.loc[len(non_salient_df)] = [image_number, image_text_dict[image_number]["non_salient"]["img_path"],
                                       image_text_dict[image_number]["non_salient"]["text"]]
    if "pure" in image_text_dict[image_number].keys():
        pure_df.loc[len(pure_df)] = [image_number, image_text_dict[image_number]["pure"]]
    if "whole" in image_text_dict[image_number].keys():
        whole_df.loc[len(whole_df)] = [image_number, image_text_dict[image_number]["whole"]["img_path"],image_text_dict[image_number]["whole"]["text"] ]

# store as csv files
all_df.to_csv(os.path.join(dir, "all.csv"), index=False)
whole_df.to_csv(os.path.join(dir, "whole.csv"), index=False)
salient_df.to_csv(os.path.join(dir, "salient.csv"), index=False)
non_salient_df.to_csv(os.path.join(dir, "non_salient.csv"), index=False)
pure_df.to_csv(os.path.join(dir, "pure.csv"), index=False)

