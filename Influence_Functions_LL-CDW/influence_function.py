import torch
from torch.autograd import Variable, grad

def grad_z(z, t, model, loss_criterion, n, λ=0):
    model.eval()
    # initialize
    z, t = Variable(z), Variable(t) # here were two flags: volatile=False. True would mean that autograd shouldn't follow this. Got disabled
    y = model(z)
    
    loss = loss_criterion(y, t)

    # We manually add L2 regularization
    l2_reg = 0.0
    for param in model.parameters():
        l2_reg += torch.norm(param)**2
    loss += 1/n * λ/2 * l2_reg

    return list(grad(loss, list(model.parameters()), create_graph=True))