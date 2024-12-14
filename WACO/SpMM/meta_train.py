import os
import random 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
#import matplotlib
#import matplotlib.pyplot as plt 
import sys
from meta_model import ResNet14
from meta_model import MetaTransferModel
from Loader.superschedule_loader import SuperScheduleDataset
from Loader.sparsematrix_loader import SparseMatrixDataset, collate_fn
import MinkowskiEngine as ME



if __name__ == "__main__":
    f = open("trainlog.txt",'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    pretrain_dataset = SparseMatrixDataset('/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/filtered_train_small.txt')
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)


    pretrain_model = ResNet14(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    pretrain_model = pretrain_model.to(device)

    model_path = '/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/pretrained_resnet14.pth'
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        pretrain_model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print("No pre-trained model found. Starting training from scratch.")
   
    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(pretrain_model.parameters(), lr=1e-4)    
    
    print("Starting pretraining phase...")

for epoch in range(80): 
    pretrain_model.train()
    pretrain_loss = 0
    pretrain_loss_cnt = 0

    for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(pretrain_loader):
            
            torch.cuda.empty_cache()

            # Move shapes to device
            shapes = shapes.to(device)

            # Create a sparse tensor from the coordinates and features
            
            SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)

            # Forward pass to extract query features
            query_feature = pretrain_model.embed_sparse_matrix(SparseMatrix, shapes)

            # Calculate the reconstruction loss (or other appropriate loss)
            query_feature = query_feature.expand((SparseMatrix.features.shape[0], query_feature.shape[1]))

            # Calculate the target for the MarginRankingLoss
            # Using the upper triangular indices to get pairs of predictions and their true values
            iu = torch.triu_indices(query_feature.shape[0], query_feature.shape[0], 1)
            pred1, pred2 = query_feature[iu[0]], query_feature[iu[1]]
            true1, true2 = SparseMatrix.features[iu[0]], SparseMatrix.features[iu[1]]

            # Calculate the difference (the 'sign' for MarginRankingLoss)
            sign = (true1 - true2).sign()

            # Compute the MarginRankingLoss
            loss = criterion(pred1, pred2, sign)

            # Accumulate loss and backpropagate
            pretrain_loss += loss.item()
            pretrain_loss_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress for debugging
            if sparse_batchidx % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {sparse_batchidx}, Loss: {loss.item()}")
                print("\tQuery Feature (first 5):", query_feature.detach()[0, :5])
                print("\tSparse Matrix Feature (first 5):", SparseMatrix.features.detach()[0, :5])

    # Print epoch-level loss
    print(f"--- Pretrain Epoch {epoch}: Loss {pretrain_loss/pretrain_loss_cnt} ---")
    # Save the model state dict after pretraining
    torch.save(pretrain_model.state_dict(), '/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/pretrained_resnet14.pth')
 

meta_transfer_dataset = SparseMatrixDataset('/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/filtered_generalization.txt')  # New, smaller dataset for meta-training
meta_transfer_loader = torch.utils.data.DataLoader(meta_transfer_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)




pretrain_model = ResNet14(in_channels=1, out_channels=1, D=2)
pretrain_model.load_state_dict(torch.load('pretrained_resnet14.pth'))
pretrain_model = pretrain_model.to(device)

meta_model = MetaTransferModel(in_channels=1, D=2)
meta_model.shared_backbone.load_state_dict(pretrain_model.state_dict(), strict=False)
meta_model = meta_model.to(device)

meta_criterion = nn.MarginRankingLoss(margin=1)
meta_optimizer = Adam(meta_model.parameters(), lr=1e-4)

print("Starting meta-transfer training phase...")

for epoch in range(80):
    meta_model.train()
    meta_loss = 0
    meta_loss_cnt = 0

    for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(meta_transfer_loader):
        torch.cuda.empty_cache()

        shapes = shapes.to(device)
        SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)

        # Forward pass for meta-transfer learning
        shared_features = meta_model(SparseMatrix)

        # Compute meta-transfer loss
        shared_features = shared_features.expand((SparseMatrix.features.shape[0], shared_features.shape[1]))
        iu = torch.triu_indices(shared_features.shape[0], shared_features.shape[0], 1)
        pred1, pred2 = shared_features[iu[0]], shared_features[iu[1]]
        true1, true2 = SparseMatrix.features[iu[0]], SparseMatrix.features[iu[1]]
        sign = (true1 - true2).sign()

        loss = meta_criterion(pred1, pred2, sign)
        meta_loss += loss.item()
        meta_loss_cnt += 1

        meta_optimizer.zero_grad()
        loss.backward()
        meta_optimizer.step()

        if sparse_batchidx % 100 == 0:
            print(f"Meta Epoch: {epoch}, Batch: {sparse_batchidx}, Meta Loss: {loss.item()}")

    print(f"--- Meta-Transfer Epoch {epoch}: Loss {meta_loss/meta_loss_cnt} ---")
    torch.save(meta_model.state_dict(), 'meta_transfer_model.pth')
    print("Meta-transfer model saved successfully.")



# Validation phase for the meta model
validation_dataset = SparseMatrixDataset('/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/filtered_validation.txt')  
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

meta_model.eval()  

validation_loss = 0
validation_loss_cnt = 0

with torch.no_grad():  # Disable gradient calculation for validation
    for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(validation_loader):
        shapes = shapes.to(device)
        SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)

        # Forward pass for validation
        shared_features = meta_model(SparseMatrix)

        # Compute validation loss
        shared_features = shared_features.expand((SparseMatrix.features.shape[0], shared_features.shape[1]))
        iu = torch.triu_indices(shared_features.shape[0], shared_features.shape[0], 1)
        pred1, pred2 = shared_features[iu[0]], shared_features[iu[1]]
        true1, true2 = SparseMatrix.features[iu[0]], SparseMatrix.features[iu[1]]
        sign = (true1 - true2).sign()

        loss = meta_criterion(pred1, pred2, sign)
        validation_loss += loss.item()
        validation_loss_cnt += 1

        if sparse_batchidx % 100 == 0:
            print(f"Validation Batch: {sparse_batchidx}, Loss: {loss.item()}")

# Print validation loss
print(f"--- Validation Loss: {validation_loss/validation_loss_cnt} ---")



