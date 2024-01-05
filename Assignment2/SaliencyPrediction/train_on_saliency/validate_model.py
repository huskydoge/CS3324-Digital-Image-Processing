import sys
sys.path.append('../')
import ImageReward as RM
from tqdm import tqdm
from custom_dataset import *
opts.reshape = True

def save_map_to_png(map, path):
    # Convert the tensor to a NumPy array if it's not already
    if not isinstance(map, np.ndarray):
        map = map.detach().cpu().numpy()

    map = np.squeeze(map)

    # Normalize the array to be in the range [0, 255] if it's not already
    # This step is optional and depends on the data range in your tensor
    map = (255 * (map - np.min(map)) / np.ptp(map)).astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(map)

    # Save the image
    image.save(path, format='PNG')

def convert_2_grey(img):
    img = img.convert('L')
    return img

if __name__ == "__main__":
    model = RM.load_saliency(name="../checkpoint/ImageReward/ImageReward.pt", med_config="./config/med_config.json", reshape=opts.reshape)
    # model = load_model(model,
    #                    "/data/husky/ImageReward/train_on_AGIQA_3K/checkpoint/11092209_bsNone_fix=32_lr=0for_align/best_lr=5e-06.pt")

    # model = load_model(model, "/data/husky/ImageReward/train_on_saliency/checkpoint/2023-11-30_11-33-47_epochs=300_bs32_loss=metric_fixrate=0_lr=5e-06_cosine/best_lr=5e-06_loss=loss0_metric_epoch=100.pt")

    model = load_model(model, "/data/husky/ImageReward/train_on_saliency/checkpoint/2023-12-02_01-31-15_epochs=300_bs32_loss=saloss_fixrate=0.2_reshape=False_lr=5e-05_cosine/best_lr=5e-05_loss=loss=saloss_epoch=50.pt")
    dataset = SaliencyDataSet()
    model.eval() # set model to eval mode
    type = "all"
    train_loader, test_loader = dataset.get_loader(type)


    cc_scores = []
    auc_scores = []
    sauc_scores = []
    nss_scores = []
    pbar = tqdm(test_loader, total=len(train_loader))
    cnt = 0
    for batch in pbar:
        img_paths, map_paths, fix_map_paths, texts = batch
        for i in tqdm(range(len(img_paths)),total=len(img_paths)):
            img_path = img_paths[i]
            map_path = map_paths[i]
            fix_map_path = fix_map_paths[i]
            text = texts[i]
            img = Image.open(img_path)
            map_img = Image.open(map_path)
            fix_map_img = Image.open(fix_map_path).convert('L') # convert to grey
            map_tensor = transfer_map_to_tensor(map_img)
            fix_map_tensor = transfer_map_to_tensor(fix_map_img)
            d1 = model(text, img)
            d1 = d1.detach().cpu()
            save_map_to_png(d1, "./d1.png")
            save_map_to_png(map_tensor, "./gt.png")
            # convert d1 into img
            cc_score = CC(d1, map_tensor)
            sAUC_score = sAUC(d1, map_tensor)
            nss_score = NSS(d1, fix_map_tensor)
            auc_score = AUC(d1, map_tensor)

            cc_scores.append(cc_score)
            sauc_scores.append(sAUC_score)
            nss_scores.append(nss_score)
            auc_scores.append(auc_score)
            cnt += 1
            print("cnt = {} | cc: {}, auc: {}, sauc: {}, nss: {}".format(cnt, np.mean(cc_scores), np.mean(auc_scores), np.mean(sauc_scores), np.mean(nss_scores)))


