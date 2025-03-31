import argparse
import time
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from model import EfficientPose
from dataset import create_loaders
from losses import smooth_l1, focal, transformation_loss

LEARNING_RATE = 1e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = '/workspace/dataset-custom-v10-ladle-test-deleted'
BATCH_SIZE = 32
NUM_EPOCHS = 500
NUM_WORKERS = 2
PIN_MEMORY = True

loss_weights = [1.0, 1.0, 0.5]

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        images = data[0].to(device=DEVICE)
        Ks = torch.unsqueeze(data[1].to(device=DEVICE),dim=1)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(images,Ks)
            losses = []
            for i in range(3):
                losses.append((loss_fn[i](targets[i].to(DEVICE),predictions[i])))
            loss = losses[0]*loss_weights[0] + losses[1]*loss_weights[1] + losses[2]*loss_weights[2]
        
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy (validation_loader, model, loss_fn):
    loop = tqdm(validation_loader)
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        images = data[0].to(device=DEVICE)
        Ks = torch.unsqueeze(data[1].to(device=DEVICE),dim=1)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(images,Ks)
            losses = []
            for i in range(3):
                losses.append((loss_fn[i](targets[i].to(DEVICE),predictions[i])))
            loss = losses[0]*loss_weights[0] + losses[1]*loss_weights[1] + losses[2]*loss_weights[2]


        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main(args = None):
    """
    Train an EfficientPose model.
    Args:
        args: parseargs object containing configuration for the training procedure.
    """

    # parse arguments
    """ will be implemented later on """

    
    #creating the dataloader
    print("\nCreating the DataLoaders...")
    train_loader, validation_loader, model_3d_points = create_loaders(DATASET_DIR,BATCH_SIZE,num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print("\nBuilding the Model...")
    model = EfficientPose(0, num_classes = 1).to(DEVICE)
    print("Done!")
    #load pretrained weights
    print("\nLoading weights from h5 file...")
    model.load_h5('phi_0_linemod_best_ADD.h5')
    model = model.to(DEVICE)
    print("Done!")

    regression_loss = smooth_l1()
    classification_loss = focal()
    transformation = transformation_loss(model_3d_points_np = torch.tensor(model_3d_points).to(DEVICE))

    loss_fn = [classification_loss, regression_loss, transformation]

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range (NUM_EPOCHS):
        print ("\nStarting epoch number", epoch,"...")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
	
        # check accuracy
        print ('Validation ...')
        check_accuracy(validation_loader, model, loss_fn)

    input = torch.randn((1, 3, 512, 512)).to(DEVICE)
    k = torch.randn(1,1,6).to(DEVICE)

    print("\nRandom test...")
    pred = model(input,k)
    for i in pred:
        print (i.shape)


if __name__=='__main__':
    main()
