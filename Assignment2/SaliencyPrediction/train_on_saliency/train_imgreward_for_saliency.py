from PIL import Image
import torch
import sys
from tqdm import tqdm
import pickle

sys.path.append("../")
import ImageReward as RM
import torch.nn as nn
from utils import *
# set random seed for torch, numpy, pandas and python.random
# init_seed(100)
# init_seed(200)
# init_seed(42)
from ImageReward.TranSalNet.utils.loss_function import SaliencyLoss
from ImageReward.TranSalNet.utils.data_process import postprocess_img
from custom_dataset import *
from torchvision import transforms

# 正向传播时：开启自动求导的异常侦测
# torch.autograd.set_detect_anomaly(True)

# 反向传播时：在求导时开启侦测


dirname = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    writer = visualizer()
    load_model_flag = True
    if opts.model_name == "transalnet":
        load_model_flag = False
        print("transalnet don't need to load imagereward!")
    toPIL = transforms.ToPILImage()
    model = RM.load_to_train_saliency('../checkpoint/ImageReward/ImageReward.pt',
                                      download_root="../checkpoint/ImageReward/ImageReward.pt ",
                                      device=opts.device,
                                      med_config="./config/med_config.json", load=load_model_flag)
    # new model, epoch = 15, fix = 0.9, msew = 4
    if load_model_flag:
        model = load_model(model,"/data/husky/ImageReward/train_on_saliency/checkpoint/2023-12-02_19-52-27_epochs=15_bs32_loss=saloss_fixrate=0.9_reshape=False_lr=5e-05_cosine_oneforall_ccw=2_simw=1_kldivw=10_nssw=1_msew=4/best_lr=5e-05_loss=losstype=saloss_epoch=13_auc=0.8343859810942056_nss=2.0189130306243896_cc=0.7808075547218323_sauc=0.6672451510893598.pt")
    # print(model)
    dataset = SaliencyDataSet()
    type = opts.task
    if opts.model_name == "transalnet" or "blend":
        print("using saliency loss!")
        loss_fn = SaliencyLoss()
        # opts.loss = "saloss"
        print(opts.loss)

    dirname = os.path.join(dirname, type, make_path())

    train_loader, test_loader = dataset.get_loader(type)

    # train model
    mse_loss = nn.MSELoss('sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.adam_beta1, opts.adam_beta2),
                                 eps=opts.adam_eps)

    # lr decay
    # from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR
    # total_iters = opts.train_iters * opts.epochs
    # lr_decay_iters = opts.lr_decay_iters if opts.lr_decay_iters is not None else total_iters
    #
    # if opts.lr_decay_style == 'cosine':
    #     scheduler = CosineAnnealingLR(optimizer, T_max=lr_decay_iters, eta_min=opts.lr_decay_ratio * opts.lr)
    # elif opts.lr_decay_style == 'exponential':
    #     scheduler = ExponentialLR(optimizer, gamma=opts.lr_decay_ratio)



    epoch_loss = dict()
    best_test_score = 0
    for epoch in range(opts.epochs):
        epoch_loss[0] = []
        epoch_loss[1] = []
        epoch_auc = []
        epoch_nss = []
        epoch_cc = []
        epoch_sauc = []
        i = 0
        pbar = tqdm(train_loader, total=len(train_loader),
                    desc="epoch = {} | training on image {} of this batch | batch loss = {} | task = {}".format(epoch,
                                                                                                                i, 0,
                                                                                                                type))
        batch_cnt = 0
        for batch in pbar:
            loss0 = torch.zeros(1).requires_grad_(True).to(opts.device)
            # loss1 = torch.zeros(1).requires_grad_(True).float().to(opts.device)
            img_paths, map_paths, fix_map_paths, texts = batch
            batch_score = {'auc': [], 'nss': [], 'cc': [], 'sauc': []}
            i = 0
            for i in range(len(img_paths)):
                model.train()
                img_path = img_paths[i]
                map_path = map_paths[i]
                fix_map_path = fix_map_paths[i]
                img_name = str(img_path.split('/')[-1])

                text = texts[i]
                img = Image.open(img_path)
                map_img = Image.open(map_path)
                fix_map_img = Image.open(fix_map_path).convert('L')

                # if opts.reshape: # model has already reshape the output to original size
                #     map_tensor = transfer_map_to_tensor(map_img).to(opts.device)
                # else:
                #     # we should transform map to the same size as the output of model
                #     map_tensor = (transform_map(224)(map_img)).to(opts.device)
                #     fix_map_tensor = (transform_map(224)(fix_map_img)).to(opts.device)
                map_tensor = transform_for_sal(map_img, opts.device)
                fix_map_tensor = transform_for_sal(fix_map_img, opts.device)

                # d1, d2, d3, d4, d5, d6, d7 = model(text, img)
                salmap = model(text, img)
                salmap = salmap.squeeze(0)
                # salmap = postprocess_img(salmap, map_img)
                # to tensor
                # save tensor to img

                save_path = os.path.join(dirname,
                                         "train/{}/{}".format(img_name, epoch))
                os.makedirs(save_path, exist_ok=True)
                img = img.convert('RGB')
                img.save(os.path.join(save_path, 'origin.jpg'))
                toPIL(salmap).save(os.path.join(save_path, 'pred.jpg'))
                toPIL(map_tensor).save(os.path.join(save_path, 'map.jpg'))
                toPIL(fix_map_tensor).save(os.path.join(save_path, 'fix.jpg'))
                with open(os.path.join(save_path, 'text.txt'), 'w') as f:
                    f.write(text)
                img = transform_ori(img)
                writer.add_image("train/{}/pred".format(img_name), salmap, epoch)
                writer.add_image("train/{}/gt".format(img_name), map_tensor, epoch)
                writer.add_image("train/{}/fix".format(img_name), fix_map_tensor, epoch)
                writer.add_image("train/{}/origin".format(img_name), img, epoch)

                if opts.loss == "saloss":
                    loss = - opts.cc_w * loss_fn(salmap, map_tensor, loss_type='cc') \
                           - opts.sim_w * loss_fn(salmap, map_tensor, loss_type='sim') + \
                           opts.kldiv_w * loss_fn(salmap, map_tensor, loss_type='kldiv') - opts.nss_w * loss_fn(salmap,
                                                                                                                fix_map_tensor,
                                                                                                                loss_type='nss') + opts.mse_w * mse_loss(
                        salmap, map_tensor)
                elif opts.loss == 'mse':
                    # loss = mse_loss_fusion(d1, d2, d3, d4, d5, d6, d7, map_tensor)
                    loss = 100 * mse_loss(salmap, map_tensor)
                    # cc_score = CC(salmap, map_tensor)
                    # loss = loss + torch.clamp(1 / cc_score, 1, 10000) - 1
                    # print(loss[)
                else:
                    # loss = multi_metric_loss(salmap, map_tensor, fix_map_tensor)
                    loss = nn.CrossEntropyLoss()(salmap, map_tensor)
                    print(loss)
                loss0 = loss0 + loss
                # print(loss0)
                with torch.no_grad():
                    model.eval()
                    auc_score = AUC(salmap, map_tensor)
                    sauc_score = sAUC(salmap, map_tensor)
                    nss_score = NSS(salmap, fix_map_tensor).detach().cpu()
                    cc_score = CC(salmap, map_tensor).detach().cpu()
                    batch_score['auc'].append(auc_score)
                    batch_score['nss'].append(nss_score)
                    batch_score['cc'].append(cc_score)
                    batch_score['sauc'].append(sauc_score)

                model.train()
                pbar.set_description(
                    "epoch = {} | task = {} | training on image {} of this batch | batch loss = {} | step loss = {}".format(
                        epoch, type,
                        i,
                        loss0.item(),
                        loss.item()))
            print("finish this batch")
            writer.add_scalar("batch_score/auc_score ", np.mean(batch_score['auc']), batch_cnt + epoch * len(pbar))
            writer.add_scalar("batch_score/nss_score ", np.mean(batch_score['nss']), batch_cnt + epoch * len(pbar))
            writer.add_scalar("batch_score/cc_score ", np.mean(batch_score['cc']), batch_cnt + epoch * len(pbar))
            writer.add_scalar("batch_score/sauc_score ", np.mean(batch_score['sauc']), batch_cnt + epoch * len(pbar))

            # loss1 += loss[1]
            pbar.set_description(
                "epoch = {} | training on image {} of this batch | batch loss = {} | step loss = {}".format(epoch, i,
                                                                                                            loss0.item(),
                                                                                                            loss.item()))
            writer.add_scalar("batch_loss/loss0 ", loss0, batch_cnt + epoch * len(pbar))
            # writer.add_scalar("batch_loss/loss1 ", loss1, batch_cnt + epoch * len(pbar))
            epoch_loss[0].append(loss0.item())

            # epoch_loss[1].append(loss1.item())
            # input the print into a single txt file
            # with open("train_log.txt", "w") as f:
            #     for name, parms in model.named_parameters():
            #         f.write('-->name:' + str(name) + '\n')
            #         f.write('-->para:' + str(parms) + '\n')
            #         f.write('-->func:' + str(parms.grad_fn) + '\n')
            #         f.write('-->grad_requirs:' + str(parms.requires_grad) + '\n')
            #         f.write('-->grad_value:' + str(parms.grad) + '\n')
            #         f.write("===" + '\n')
            # print(loss0)
            with torch.autograd.detect_anomaly():
                loss0.backward()
            optimizer.step()

            # with open("train_log_after.txt", "w") as f:
            #     for name, parms in model.named_parameters():
            #         f.write('-->name:' + str(name) + '\n')
            #         f.write('-->para:' + str(parms) + '\n')
            #         f.write('-->func:' + str(parms.grad_fn) + '\n')
            #         f.write('-->grad_requirs:' + str(parms.requires_grad) + '\n')
            #         f.write('-->grad_value:' + str(parms.grad) + '\n')
            #         f.write("===" + '\n')
            batch_cnt += 1
            optimizer.zero_grad()
        # print("epoch loss = ", epoch_loss[epoch])
        writer.add_scalar("epoch_loss/loss0 ", sum(epoch_loss[0]) / len(epoch_loss[0]), epoch)
        # writer.add_scalar("epoch_loss/loss1 ", sum(epoch_loss[1]) / len(epoch_loss[1]), epoch)
        score = - np.inf
        if epoch % 1 == 0:
            # eval model on test set
            with torch.no_grad():
                model.eval()
                test_auc_scores = []
                test_nss_scores = []
                test_cc_scores = []
                test_sauc_scores = []
                test_cnt = 0
                total_img = 0
                for batch in test_loader:
                    img_paths, map_paths, fix_map_paths, texts = batch
                    i = 0
                    for i in range(len(img_paths)):
                        img_path = img_paths[i]
                        map_path = map_paths[i]
                        fix_map_path = fix_map_paths[i]
                        img_name = str(map_path.split('/')[-1])

                        text = texts[i]
                        img = Image.open(img_path)
                        map_img = Image.open(map_path)
                        fix_map_img = Image.open(fix_map_path).convert('L')

                        map_tensor = transform_for_sal(map_img, opts.device)

                        fix_map_tensor = transform_for_sal(fix_map_img, opts.device)

                        # d1, d2, d3, d4, d5, d6, d7 = model(text, img)
                        salmap = model(text, img)
                        salmap = salmap.squeeze(0)

                        auc_score = AUC(salmap, map_tensor)
                        sauc_score = sAUC(salmap, map_tensor)
                        nss_score = NSS(salmap, fix_map_tensor).detach().cpu()
                        cc_score = CC(salmap, map_tensor).detach().cpu()

                        test_auc_scores.append(auc_score)
                        test_nss_scores.append(nss_score)
                        test_cc_scores.append(cc_score)
                        test_sauc_scores.append(sauc_score)
                        img = img.convert('RGB')
                        save_path = os.path.join(dirname,
                                                 "test/{}/{}".format(img_name, epoch))
                        os.makedirs(save_path, exist_ok=True)
                        img.save(os.path.join(save_path, 'origin.jpg'))
                        toPIL(salmap).save(os.path.join(save_path, 'pred.jpg'))
                        toPIL(map_tensor).save(os.path.join(save_path, 'map.jpg'))
                        toPIL(fix_map_tensor).save(os.path.join(save_path, 'fix.jpg'))
                        with open(os.path.join(save_path, 'text.txt'), 'w') as f:
                            f.write(text)
                        img = transform_ori(img)
                        writer.add_image("test/{}/pred".format(img_name), salmap, epoch / 1)
                        writer.add_image("test/{}/origin".format(img_name), img, epoch / 1)
                        writer.add_image("test/{}/gt".format(img_name), map_tensor, epoch / 1)
                        writer.add_image("test/{}/fix".format(img_name), fix_map_tensor, epoch / 1)

                    test_cnt += 1

                writer.add_scalar("test_score/auc_score ", np.mean(test_auc_scores), epoch / 1)
                writer.add_scalar("test_score/nss_score ", np.mean(test_nss_scores), epoch / 1)
                writer.add_scalar("test_score/cc_score ", np.mean(test_cc_scores), epoch / 1)
                writer.add_scalar("test_score/sauc_score ", np.mean(test_sauc_scores), epoch / 1)

            score = np.mean(test_auc_scores) + np.mean(test_nss_scores) + np.mean(test_cc_scores) + np.mean(
                test_sauc_scores)
        if score >= best_test_score:
            best_test_score = score
            save_model(model, subfix="losstype={}_epoch={}_auc={}_nss={}_cc={}_sauc={}".format(opts.loss, epoch,
                                                                                               np.mean(test_auc_scores),
                                                                                               np.mean(test_nss_scores),
                                                                                               np.mean(test_cc_scores),
                                                                                               np.mean(
                                                                                                   test_sauc_scores)))
            print("new best model!")
        model.train()
    save_model(model, subfix="new_model")
    # pickle.dump(epoch_loss, open("/data/husky/ImageReward/train_on_AGIQA_3K/checkpoint/epoch_loss_for_align_epoch50.pkl", "wb"))
