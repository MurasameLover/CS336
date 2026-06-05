import torch
def save_checkpoint(
    model,
    optimizer,
    iteration,
    output_path
):
    checkpoint = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "iteration" : iteration,
    }
    torch.save(checkpoint, output_path)

def load_checkpoint(
    input_path,
    model,
    optimizer
):
    checkpoint = torch.load(input_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]