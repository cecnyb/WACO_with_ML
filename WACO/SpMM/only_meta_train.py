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
import torch.nn.init as init
import torch


if __name__ == "__main__":
    f = open("trainlog.txt",'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    meta_transfer_dataset = SparseMatrixDataset('/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/filtered_train_small.txt')  # New, smaller dataset for meta-training
    meta_transfer_loader = torch.utils.data.DataLoader(meta_transfer_dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn)

    SparseMatrix_Dataset_Valid = SparseMatrixDataset('/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/filtered_validation_small.txt')
    valid_SparseMatrix = torch.utils.data.DataLoader(SparseMatrix_Dataset_Valid, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)


    pretrain_model = ResNet14(in_channels=1, out_channels=1, D=2)


    # Load the weights into the model
    state_dict = torch.load('/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SpMM/resnet.pth', 
                        map_location=torch.device('cpu'))

   
    # Get the current model's state dictionary
    model_state_dict = pretrain_model.state_dict()

    # Filter the state dictionary to include only matching keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}



    # Load the filtered state dictionary
    pretrain_model.load_state_dict(filtered_state_dict, strict=False)
    print("Loaded weights with mismatched layers ignored.")

    

    # Reinitialize mismatched layers
    init.xavier_uniform_(pretrain_model.jsplit.weight)
    init.xavier_uniform_(pretrain_model.ksplit.weight)
    init.xavier_uniform_(pretrain_model.parchunk.weight)
    init.xavier_uniform_(pretrain_model.schedule_embedding[0].weight)
    print("Reinitialized mismatched layers with Xavier initialization.")
    
   # pretrain_model.load_state_dict(torch.load('/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SpMM/resnet.pth'))
    pretrain_model = pretrain_model.to(device)

    meta_model = MetaTransferModel(in_channels=1, D=2)
    meta_model.shared_backbone.load_state_dict(pretrain_model.state_dict(), strict=False)
    meta_model = meta_model.to(device)

    meta_criterion = nn.MarginRankingLoss(margin=0.1)
    #meta_criterion = nn.MSELoss()
    meta_optimizer = Adam(meta_model.parameters(), lr=1e-4)

    print("Starting meta-transfer training phase...")

    
    for epoch in range(80):
        meta_model.train()
        meta_loss = 0
        meta_loss_cnt = 0

        for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(meta_transfer_loader):
            torch.cuda.empty_cache()
            
            torch.save(meta_model.state_dict(), "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/meta_model.pth")
           
            SuperSchedule_Dataset = SuperScheduleDataset(mtx_names[0]) # Get rid of runtime<1000
            train_SuperSchedule = torch.utils.data.DataLoader(SuperSchedule_Dataset, batch_size=32, shuffle=True, num_workers=0)
            
            SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)

            for schedule_batchidx, (schedule, runtime) in enumerate(train_SuperSchedule) :
                if (schedule.shape[0] < 2) : break
                schedule, runtime = schedule.to(device), runtime.to(device)
                meta_optimizer.zero_grad()
                query_feature = meta_model(SparseMatrix, shapes)
                query_feature = query_feature.expand((schedule.shape[0], query_feature.shape[1]))
                predict = meta_model.forward_after_query(query_feature, schedule)

                #HingeRankingLoss
                iu = torch.triu_indices(predict.shape[0],predict.shape[0],1)
                pred1, pred2 = predict[iu[0]], predict[iu[1]]
                true1, true2 = runtime[iu[0]], runtime[iu[1]]
                sign = (true1-true2).sign()
                loss = meta_criterion(pred1, pred2, sign)
                meta_loss += loss.item()
                meta_loss_cnt += 1

                loss.backward()
                meta_optimizer.step()

                if (sparse_batchidx % 100 == 0 and schedule_batchidx == 0) :
                    print("Epoch: ", epoch, ", MTX: ", mtx_names[0], " " , shapes, "(", sparse_batchidx, "), Schedule : ", schedule_batchidx, ", Loss: ", loss.item())
                    print("\tPredict : ", predict.detach()[:5,0])
                    print("\tGT      : ", runtime.detach()[:5,0])
                    print("\tQuery   : ", query_feature.detach()[0,:5])
                break
        
    #Validation
    meta_model.eval()
    with torch.no_grad() :
        valid_loss = 0
        valid_loss_cnt = 0
        for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(valid_SparseMatrix) :
            torch.cuda.empty_cache()
            SuperSchedule_Dataset = SuperScheduleDataset(mtx_names[0]) # Get rid of runtime<1000
            valid_SuperSchedule = torch.utils.data.DataLoader(SuperSchedule_Dataset, batch_size=32, shuffle=True, num_workers=0)
            shapes = shapes.to(device)
            
            SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)
            for schedule_batchidx, (schedule, runtime) in enumerate(valid_SuperSchedule) :
                if (schedule.shape[0] < 6) : break
                schedule, runtime = schedule.to(device), runtime.to(device)
                query_feature = meta_model(SparseMatrix, shapes)
                query_feature = query_feature.expand((schedule.shape[0], query_feature.shape[1]))
                predict = meta_model.forward_after_query(query_feature, schedule)

                #HingeRankingLoss
                iu = torch.triu_indices(predict.shape[0],predict.shape[0],1)
                pred1, pred2 = predict[iu[0]], predict[iu[1]]
                true1, true2 = runtime[iu[0]], runtime[iu[1]]
                sign = (true1-true2).sign()
                loss = meta_criterion(pred1, pred2, sign)
                valid_loss += loss.item()
                valid_loss_cnt += 1
            
                if (sparse_batchidx % 100 == 0 and schedule_batchidx == 0) :
                    print("ValidEpoch: ", epoch, ", MTX: ", mtx_names[0], " " , shapes, "(", sparse_batchidx, "), Schedule : ", schedule_batchidx, ", Loss: ", loss.item())
                    print("\tValidPredict : ", predict.detach()[:5,0])
                    print("\tValidGT      : ", runtime.detach()[:5,0])
                    print("\tValidQuery   : ", query_feature.detach()[0,:5])

                break
    
    print ("--- Epoch {} : Train {} Valid {} ---".format(epoch, meta_loss/meta_loss_cnt, valid_loss/valid_loss_cnt))
    f.write("--- Epoch {} : Train {} Valid {} ---\n".format(epoch, meta_loss/meta_loss_cnt, valid_loss/valid_loss_cnt))
    f.flush()

'''      
            print(SparseMatrix.features)
            # Forward pass for meta-transfer learning
            shared_features = meta_model(SparseMatrix, shapes)

            # Compute meta-transfer loss
            shared_features = shared_features.expand((SparseMatrix.features.shape[0], shared_features.shape[1]))
            iu = torch.triu_indices(shared_features.shape[0], shared_features.shape[0], 1)
            pred1, pred2 = shared_features[iu[0]], shared_features[iu[1]]
            true1, true2 = SparseMatrix.features[iu[0]], SparseMatrix.features[iu[1]]

           
            print("pred1: ", pred1)
            print("pred2: ", pred2)

            print("true1: ", true1)
            print("true2: ", true2)
      
            sign = (true1 - true2).sign()

            print("Sign values:", sign.unique())

            #loss = meta_criterion(pred1, pred2, sign)
            loss = meta_criterion(pred1, true1)
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
        validation_dataset = SparseMatrixDataset('/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/filtered_validation_small.txt')  
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

        meta_model.eval()  

        validation_loss = 0
        validation_loss_cnt = 0

        with torch.no_grad():  # Disable gradient calculation for validation
            for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(validation_loader):
                shapes = shapes.to(device)
                SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)

                # Forward pass for validation
                shared_features = meta_model(SparseMatrix, shapes)

                # Compute validation loss
                shared_features = shared_features.expand((SparseMatrix.features.shape[0], shared_features.shape[1]))
                iu = torch.triu_indices(shared_features.shape[0], shared_features.shape[0], 1)
                pred1, pred2 = shared_features[iu[0]], shared_features[iu[1]]
                true1, true2 = SparseMatrix.features[iu[0]], SparseMatrix.features[iu[1]]
                sign = (true1 - true2).sign()

                #loss = meta_criterion(pred1, pred2, sign)
                loss = meta_criterion(pred1, true1)
                validation_loss += loss.item()
                validation_loss_cnt += 1

                if sparse_batchidx % 100 == 0:
                    print(f"Validation Batch: {sparse_batchidx}, Loss: {loss.item()}")

        # Print validation loss
        print(f"--- Validation Loss: {validation_loss/validation_loss_cnt} ---")



'''