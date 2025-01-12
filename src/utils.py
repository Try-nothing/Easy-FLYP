import os

import torch
import pickle
from tqdm import tqdm
import math

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import random
import importlib
from torch.optim.lr_scheduler import LambdaLR


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps, min_lr=0.0):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr + min_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1, )):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0,
                                                  keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier.cpu(), f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def fisher_save(fisher, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fisher = {k: v.cpu() for k, v in fisher.items()}
    with open(save_path, 'wb') as f:
        pickle.dump(fisher, f)


def fisher_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        fisher = pickle.load(f)
    if device is not None:
        fisher = {k: v.to(device) for k, v in fisher.items()}
    return fisher


def get_logits(inputs, classifier, classification_head):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
        classification_head = classification_head.to(inputs.device)
    feats = classifier(inputs)
    return classification_head(feats)


def get_feats(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    feats = classifier(inputs)
    # feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class CLIPDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initialize the CLIPDataset.

        Args:
            data_dir (str): Path to the directory containing the dataset.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data_dir = data_dir  # Directory where the dataset is stored
        self.transform = transform  # Transform to be applied to the images
        self.classes = os.listdir(data_dir)  # List of class names (subdirectories in data_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  # Dictionary mapping class names to indices
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}  # Dictionary mapping indices to class names
        self.samples = self._make_dataset()  # Generate the list of samples (image paths, labels, and class names)

    def _make_dataset(self):
        """
        Create a list of samples where each sample is a tuple of (image_path, label, class_name).

        Returns:
            list: A list of tuples containing image paths, labels, and class names.
        """
        samples = []
        for cls in self.classes:  # Iterate through each class
            cls_dir = os.path.join(self.data_dir, cls)  # Path to the class directory
            for img_name in os.listdir(cls_dir):  # Iterate through each image in the class directory
                img_path = os.path.join(cls_dir, img_name)  # Full path to the image
                samples.append((img_path, self.class_to_idx[cls]))  # Add the sample to the list (omit class name)
        return samples

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - list: Transformed images (image_0, image_1).
                - int: Label of the image.
        """
        img_path, label = self.samples[idx]  # Get the image path and label
        image = Image.open(img_path).convert('RGB')  # Open the image and convert it to RGB format

        if self.transform:
            # Apply the transform to the image twice (e.g., for contrastive learning)
            image_0 = self.transform(image)
            image_1 = self.transform(image)

        return (image_0, image_1), label  # Return a tuple of transformed images and label
    
    def get_class_name(self, label):
        """
        Retrieve the class name corresponding to a given label.

        Args:
            label (int): Label of the class.

        Returns:
            str: Class name corresponding to the label.
        """
        return self.idx_to_class[label]

def turn_label2template(labels, dataset, templates):
    """
    Convert a label to a template string.

    Args:
        label (int): Label to convert.

    Returns:
        str: Template string corresponding to the label.
    """
    class_names = [dataset.get_class_name(label.item()) for label in labels]
    template_list = []
    for class_name in class_names:
        template = random.sample(templates,1)[0](class_name)
        template_list.append(template)
    return template_list

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def import_function_from_path(module_path, func_name):
    """
    Import a function from a given module path.

    Args:
        module_path (str): Path to the module (can be relative or absolute).
        func_name (str): Name of the function to import.

    Returns:
        function: Imported function.

    Raises:
        ImportError: If the module or function cannot be imported.
    """
    # Convert relative path to absolute import path
    if module_path.startswith('./'):
        # Remove './' and convert to a proper Python module path
        module_path = module_path[2:].replace('/', '.').replace('.py', '')
    elif module_path.startswith('/'):
        # Handle absolute paths (if needed)
        module_path = module_path.lstrip('/').replace('/', '.').replace('.py', '')
    
    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}")
    
    # Get the function from the module
    try:
        func = getattr(module, func_name)
    except AttributeError:
        raise ImportError(f"Function '{func_name}' not found in module '{module_path}'")
    
    return func

def save_model(img_prob, text_prob, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'image_prob': img_prob.state_dict(),
        'text_prob': text_prob.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    torch.save(state, os.path.join(save_file, 'checkpoint'))
    del state

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a learning rate scheduler with a warmup phase followed by a cosine decay phase.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        num_warmup_steps: The number of steps for the warmup phase, during which the learning rate increases linearly.
        num_training_steps: The total number of training steps.
        num_cycles: The number of cosine cycles (default: 0.5, which corresponds to half a cosine cycle).
        last_epoch: The index of the last epoch (default: -1, meaning start from scratch).

    Returns:
        A LambdaLR object that updates the optimizer's learning rate based on the current step.
    """

    def lr_lambda(current_step):
        """
        Compute the learning rate multiplier for the current step.

        Args:
            current_step: The current training step.

        Returns:
            The learning rate multiplier (a value between 0 and 1).
        """
        # Warmup phase: Linearly increase the learning rate from 0 to 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase: Decrease the learning rate following a cosine function from 1 to 0
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    # Return a LambdaLR object that applies the lr_lambda function to adjust the learning rate
    return LambdaLR(optimizer, lr_lambda, last_epoch)