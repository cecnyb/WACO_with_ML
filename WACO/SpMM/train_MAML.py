import os
import torch
import torch.nn as nn
from torch.optim import Adam
import learn2learn as l2l
from model import ResNet14
from Loader.superschedule_loader import SuperScheduleDataset
from Loader.sparsematrix_loader import SparseMatrixDataset, collate_fn
import MinkowskiEngine as ME

if __name__ == "__main__":
    f = open("trainlog.txt", 'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained ResNet14
    net = ResNet14(in_channels=1, out_channels=1, D=2).to(device)  # 2D Tensor
    pretrain_path = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/pretrained_resnet14.pth"
    net.load_state_dict(torch.load(pretrain_path))
    print(f"Loaded pre-trained model from {pretrain_path}")

    # Wrap the model with MAML
    maml = l2l.algorithms.MAML(net, lr=1e-3)  # Inner-loop learning rate
    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(maml.parameters(), lr=1e-4)  # Outer-loop optimizer

    # Load datasets
    train_dataset = SparseMatrixDataset('/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SpMM/TrainingData/train_similated.txt')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    valid_dataset = SparseMatrixDataset('/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SpMM/TrainingData/simulated_val.txt')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # Meta-training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        maml.train()
        train_loss = 0
        train_loss_cnt = 0

        for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(train_loader):
            torch.cuda.empty_cache()

            # Create dataset for current task
            SuperSchedule_Dataset = SuperScheduleDataset(mtx_names[0])
            task_loader = torch.utils.data.DataLoader(SuperSchedule_Dataset, batch_size=32, shuffle=True, num_workers=0)
            shapes = shapes.to(device)

            SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)

            for schedule_batchidx, (schedule, runtime) in enumerate(task_loader):
                if schedule.shape[0] < 2:
                    break

                schedule, runtime = schedule.to(device), runtime.to(device)

                # Meta-training inner loop
                learner = maml.clone()  # Create task-specific model
                query_feature = learner.embed_sparse_matrix(SparseMatrix, shapes)
                query_feature = query_feature.expand((schedule.shape[0], query_feature.shape[1]))
                predict = learner.forward_after_query(query_feature, schedule)

                # Compute task-specific loss
                iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                pred1, pred2 = predict[iu[0]], predict[iu[1]]
                true1, true2 = runtime[iu[0]], runtime[iu[1]]
                sign = (true1 - true2).sign()
                loss = criterion(pred1, pred2, sign)

                # Inner loop optimization
                learner.adapt(loss)
                train_loss += loss.item()
                train_loss_cnt += 1

                # Break after one schedule for debugging
                break

            # Outer loop optimization
            optimizer.zero_grad()
            query_feature = maml.embed_sparse_matrix(SparseMatrix, shapes)
            query_feature = query_feature.expand((schedule.shape[0], query_feature.shape[1]))
            predict = maml.forward_after_query(query_feature, schedule)
            loss.backward()
            optimizer.step()

            # Log training progress
            if sparse_batchidx % 100 == 0:
                print(f"Epoch: {epoch}, MTX: {mtx_names[0]}, Loss: {loss.item()}")

        # Validation
        maml.eval()
        valid_loss = 0
        valid_loss_cnt = 0

        with torch.no_grad():
            for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(valid_loader):
                torch.cuda.empty_cache()
                SuperSchedule_Dataset = SuperScheduleDataset(mtx_names[0])
                task_loader = torch.utils.data.DataLoader(SuperSchedule_Dataset, batch_size=32, shuffle=True, num_workers=0)
                shapes = shapes.to(device)
                SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)

                for schedule_batchidx, (schedule, runtime) in enumerate(task_loader):
                    if schedule.shape[0] < 6:
                        break
                    schedule, runtime = schedule.to(device), runtime.to(device)

                    query_feature = maml.embed_sparse_matrix(SparseMatrix, shapes)
                    query_feature = query_feature.expand((schedule.shape[0], query_feature.shape[1]))
                    predict = maml.forward_after_query(query_feature, schedule)

                    iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                    pred1, pred2 = predict[iu[0]], predict[iu[1]]
                    true1, true2 = runtime[iu[0]], runtime[iu[1]]
                    sign = (true1 - true2).sign()
                    loss = criterion(pred1, pred2, sign)
                    valid_loss += loss.item()
                    valid_loss_cnt += 1

        print(f"Epoch {epoch}: Train Loss = {train_loss/train_loss_cnt}, Valid Loss = {valid_loss/valid_loss_cnt}")
        f.write(f"Epoch {epoch}: Train Loss = {train_loss/train_loss_cnt}, Valid Loss = {valid_loss/valid_loss_cnt}\n")
        f.flush()

        save_path = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/pretrained_resnet14_meta.pth"
        torch.save(maml.state_dict(), save_path)
        print(f"Model saved after epoch {epoch} to {save_path}")

    f.close()
