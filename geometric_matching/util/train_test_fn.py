from __future__ import print_function, division


def train(epoch, model, loss_fn, optimizer, dataloader, pair_generation_tnf, use_cuda=True, log_interval=50):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # batch['image'].shape: (batch_size, 3, 480, 640)
        # batch['theta'].shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # batch['theta'].shape-affine: (batch_size, 2, 3)
        # tnf_batch['source_image'].shape and tnf_batch['target_image'].shape: (batch_size, 3, 240, 240)
        # tnf_batch['theta_GT'].shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # tnf_batch['theta_GT'].shape-affine: (batch_size, 2, 3)
        tnf_batch = pair_generation_tnf(batch)
        # theta.shape: (batch_size, 18) for tps, (batch_size, 6) for affine
        theta = model(tnf_batch)
        loss = loss_fn(theta, tnf_batch['theta_GT'])
        # print(theta.shape)
        # print(tnf_batch['theta_GT'].shape)
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()[0]
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader), loss.data[0]))
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def test(model, loss_fn, dataloader, pair_generation_tnf, use_cuda=True):
    model.eval()
    test_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta, tnf_batch['theta_GT'])
        test_loss += loss.data.cpu().numpy()[0]

    test_loss /= len(dataloader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss