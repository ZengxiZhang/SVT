import numpy as np
import torch
import os
import argparse
import time

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from dataset import Image_style
import utils
from network import ImageTransformNet
from vgg import Vgg16
import cv2
# Global Variables
IMAGE_SIZE = 256
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 2000
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0
L1_WEIGHT=1e0
TV_WEIGHT = 1e-7
net_path='models/1250.pkl'
def train(args):          
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" %torch.cuda.current_device())

    # visualization of training controlled by flag
    visualize = (args.visualize != None)
    if (visualize):
        img_transform_512 = transforms.Compose([
            transforms.Scale(512),                  # scale shortest side to image_size
            transforms.CenterCrop(512),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
        ])

        testImage_amber = utils.load_image("content_imgs/amber.jpg")
        testImage_amber = img_transform_512(testImage_amber)
        testImage_amber = Variable(testImage_amber.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

        testImage_dan = utils.load_image("content_imgs/dan.jpg")
        testImage_dan = img_transform_512(testImage_dan)
        testImage_dan = Variable(testImage_dan.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

        testImage_maine = utils.load_image("content_imgs/maine.jpg")
        testImage_maine = img_transform_512(testImage_maine)
        testImage_maine = Variable(testImage_maine.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

    # define network
    image_transformer = ImageTransformNet().type(dtype)
    if net_path is not None:
        image_transformer.load_state_dict(torch.load(net_path,map_location='cpu'))
    optimizer = Adam(image_transformer.parameters(), LEARNING_RATE) 

    loss_mse = torch.nn.MSELoss()
    loss_l1 = torch.nn.L1Loss()

    # load vgg network
    vgg = Vgg16().type(dtype)

    # get training dataset
    # dataset_transform = transforms.Compose([
    #     transforms.Resize(IMAGE_SIZE),           # scale shortest side to image_size
    #     transforms.CenterCrop(IMAGE_SIZE),      # crop center image_size out
    #     transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
    #     utils.normalize_tensor_transform()      # normalize with ImageNet values
    # ])
    # train_dataset = datasets.ImageFolder(args.dataset, dataset_transform)
    train_dataset = Image_style(ir_path=os.path.join(args.dataset,'IR'), vis_path=os.path.join(args.dataset,'VIS'))
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)

    # style image
    # style_transform = transforms.Compose([
    #     transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
    #     utils.normalize_tensor_transform()      # normalize with ImageNet values
    # ])
    # style = utils.load_image(args.style_image)
    # style = style_transform(style)
    # style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1)).type(dtype)
    # style_name = os.path.split(args.style_image)[-1].split('.')[0]

    # calculate gram matrices for style feature layer maps we care about
    # style_features = vgg(style)
    # style_gram = [utils.gram(fmap) for fmap in style_features]

    for e in range(EPOCHS):

        # track values for...
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        # aggregate_tv_loss = 0.0
        aggregate_l1_loss=0.0
        # train network
        image_transformer.train()
        for batch_num, (x, label) in enumerate(train_loader):
            # print(x.shape)
            # time.sleep(1000)
            img_batch_read = len(x)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()
            x=x.cuda()
            label=label.cuda()
            # input batch to transformer network
            x = Variable(x).type(dtype)
            y_hat = image_transformer(x)
            # get vgg features
            # y_c_features = vgg(torch.cat((x,x,x),1))
            y_hat_features = vgg(torch.cat((y_hat,y_hat,y_hat),1))
            
            # calculate style loss
            y_c_features = vgg(torch.cat((label,label,label),1))
            style_gram = [utils.gram(fmap) for fmap in y_c_features]
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                # print(img_batch_read)
                # print(y_hat_gram[j].shape)
                style_loss += loss_mse(y_hat_gram[j], style_gram[j])
                # style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = STYLE_WEIGHT*style_loss
            # aggregate_style_loss += style_loss.data[0]
            aggregate_style_loss += style_loss.data.item()

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            # aggregate_content_loss += content_loss.data[0]
            aggregate_content_loss += content_loss.data.item()
            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            # diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            # diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            # tv_loss = TV_WEIGHT*(diff_i + diff_j)
            l1_loss=L1_WEIGHT*loss_l1(y_hat, label)
            # aggregate_tv_loss += tv_loss.data[0]
            # aggregate_tv_loss += tv_loss.data.item()
            aggregate_l1_loss += l1_loss.data.item()

            # total loss
            total_loss = style_loss + content_loss + l1_loss #+ tv_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            # if ((batch_num + 1) % 100 == 0):
        # status = "{}  Epoch [{}/{}]    agg_style: {:.6f}  agg_content: {:.6f}  agg_tv: {:.6f}".format(
        #                 time.ctime(), e , EPOCHS,
        #                 aggregate_style_loss/len(train_loader), aggregate_content_loss/len(train_loader), aggregate_tv_loss/len(train_loader))
        status = "{}  Epoch [{}/{}]    agg_style: {:.6f}  agg_content: {:.6f}  agg_l1: {:.6f}".format(
                        time.ctime(), e , EPOCHS,
                        aggregate_style_loss/len(train_loader), aggregate_content_loss/len(train_loader), aggregate_l1_loss/len(train_loader))
        print(status)

        battle = np.concatenate((x[0,0].cpu().detach().numpy(), label[0,0].cpu().detach().numpy(), y_hat[0,0].cpu().detach().numpy()),1)*255
        cv2.imwrite('visual.png', battle)
        if e%10==0:
            filename = "models/" + str(e) + ".pkl"
            torch.save(image_transformer.state_dict(), filename)
    # save model
    image_transformer.eval()

    if use_cuda:
        image_transformer.cpu()

    if not os.path.exists("models"):
        os.makedirs("models")
    filename = "models/" + str(style_name) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
    torch.save(image_transformer.state_dict(), filename)
    
    if use_cuda:
        image_transformer.cuda()

def style_transfer(args):
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" %torch.cuda.current_device())

    # content image
    img_transform_512 = transforms.Compose([
            transforms.Scale(512),                  # scale shortest side to image_size
            transforms.CenterCrop(512),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])

    content = utils.load_image(args.source)
    content = img_transform_512(content)
    content = content.unsqueeze(0)
    content = Variable(content).type(dtype)

    # load style model
    style_model = ImageTransformNet().type(dtype)
    style_model.load_state_dict(torch.load(args.model_path))

    # process input image
    stylized = style_model(content).cpu()
    utils.save_image(args.output, stylized.data[0])


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    train_parser = argparse.ArgumentParser()
    # train_parser = subparsers.add_parser("train", help="train a model to do style transfer")
    train_parser.add_argument("--style-image", type=str, default='style_imgs/mosaic.jpg', help="path to a style image to train with")
    train_parser.add_argument("--dataset", type=str, default='/data/zhangzengxi/RoadScene-rgb/', help="path to a dataset")
    train_parser.add_argument("--gpu", type=int, default='2', help="ID of GPU to be used")
    train_parser.add_argument("--visualize", type=int, default=None, help="Set to 1 if you want to visualize training")

    # style_parser = subparsers.add_parser("transfer", help="do style transfer with a trained model")
    # train_parser.add_argument("--model-path", type=str, default=None, help="path to a pretrained model for a style image")
    # train_parser.add_argument("--source", type=str, default='Input', help="path to source image")
    # train_parser.add_argument("--output", type=str, default='output', help="file name for stylized output image")
    # train_parser.add_argument("--gpu", type=int, default=2, help="ID of GPU to be used")

    args = train_parser.parse_args()

    print("Training!")
    train(args)


if __name__ == '__main__':
    main()








