"""Optimizer factory matching the assignment experiments."""
import torch.optim as optim


def get_optimizer(name, model, learning_rate=0.001):
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate)
    if name == "Nesterov-SGD":
        return optim.SGD(model.parameters(), lr=learning_rate,
                         momentum=0.9, nesterov=True)
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    if name == "Adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate)
    if name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    if name == "Nadam":
        return optim.NAdam(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")
