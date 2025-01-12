import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler
import PIL.Image

# Custom module imports
from models import CLIPEncoder, ClassificationHead, LinearProbModule
from engine import flyp_linearprob_train, flyp_linearprob_eval
from utils import (
    CLIPDataset,
    turn_label2template,
    import_function_from_path,
    save_model,
    get_cosine_schedule_with_warmup,
)
from clip.loss import ClipLoss

# Add parent directory to system path
sys.path.append('../')

def parse_arguments():
    """
    Parse command-line arguments for training a CLIP-based FLYP model.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Training script for a CLIP-based FLYP model.")

    # Data-related arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_dir",
        type=str,
        default="./cooked_datasets/data/StanfordCars",
        help="Path to the data directory (default: ./cooked_datasets/data/StanfordCars)."
    )
    data_group.add_argument(
        "--template_dir",
        type=str,
        default="./templates/stanfordCars.py",
        help="Path to the template directory (default: ./templates/stanfordCars.py)."
    )
    data_group.add_argument(
        "--template_type",
        type=str,
        default="open_ai_classes",
        choices=["final_classes", "open_ai_classes"],
        help="Type of class names to use (default: open_ai_classes)."
    )
    data_group.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)."
    )

    # Model-related arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        choices=["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],
        help="Type of model to use (default: ViT-B/32)."
    )
    model_group.add_argument(
        "--pretrained",
        type=str,
        default="laion400m_e31",
        help="Path to pretrained CLIP weights (default: laion400m_e31)."
    )
    model_group.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=True,
        help="Whether to freeze the encoder (default: True)."
    )
    model_group.add_argument(
        "--n_dims",
        type=int,
        default=512,
        help="Dimension of the feature vector (default: 512)."
    )

    # Training-related arguments
    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training (default: 128)."
    )
    training_group.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for training (default: 1000)."
    )
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer (default: 1e-4)."
    )
    training_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for optimizer (default: 0.05)."
    )
    training_group.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for optimizer (default: 0.9)."
    )
    training_group.add_argument(
        "--warmup",
        action="store_true",
        default=True,
        help="Whether to use learning rate warmup (default: False)."
    )
    training_group.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs (default: 10)."
    )
    training_group.add_argument(
        "--lr_decay_epochs",
        type=int,
        nargs="+",
        default=[70, 80, 90],
        help="Epochs at which to decay the learning rate (default: [700, 800, 900])."
    )
    training_group.add_argument(
        "--lr_decay_rate",
        type=float,
        default=0.1,
        help="Learning rate decay rate (default: 0.1)."
    )
    training_group.add_argument(
        "--cosine",
        action="store_true",
        default=True,
        help="Whether to use cosine learning rate schedule (default: False)."
    )
    training_group.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature parameter for contrastive loss (default: 0.07)."
    )
    training_group.add_argument(
        "--contrast_mode",
        type=str,
        default="all",
        choices=["all", "one"],
        help="Contrast mode for contrastive loss (default: all)."
    )
    training_group.add_argument(
        "--base_temperature",
        type=float,
        default=0.07,
        help="Base temperature for contrastive loss (default: 0.07)."
    )

    # Logging and evaluation arguments
    log_group = parser.add_argument_group("Logging and Evaluation")
    log_group.add_argument(
        "--save_dir",
        type=str,
        default="./flyp_linearprob/",
        help="Directory to save checkpoints and logs (default: ./flyp_linearprob/)."
    )
    log_group.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        help="Frequency of evaluation during training (default: 1)."
    )
    log_group.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="Frequency of printing training information (default: 10)."
    )

    # Miscellaneous arguments
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    misc_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (default: cuda if available, else cpu)."
    )

    # Other arguments
    other_group = parser.add_argument_group("Other")
    other_group.add_argument(
        "--template_path",
        type=str,
        default="cooked_datasets/templates/stanfordCars.py",
        help="Path to the template file (default: cooked_datasets/templates/stanfordCars.py)."
    )
    other_group.add_argument(
        "--n_classes",
        type=int,
        default=196,
        help="Number of classes in the dataset (default: 196)."
    )

    # Parse and return arguments
    opt = parser.parse_args()
    return opt


def set_seed(seed):
    """
    Set random seeds fpr reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def create_save_dir(args):
    """
    Create a directory for saving checkpoints and logs.
    """
    temp_dir = f"{args.model}_{args.pretrained}_{args.freeze_encoder}_{args.n_dims}_{args.warmup}_{args.cosine}"
    return os.path.join(args.save_dir, temp_dir)

def get_transforms():
    """
    Define data transformations for training and validation.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, interpolation=PIL.Image.BICUBIC, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4708033, 0.4602116, 0.4549391], std=[0.26664826, 0.26570636, 0.27058327])
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4708033, 0.4602116, 0.4549391], std=[0.26664826, 0.26570636, 0.27058327])
    ])
    return train_transform, valid_transform

def load_dataset(data_dir, train_transform, valid_transform):
    """
    Load training, validation, and test datasets.
    """
    train_dataset = CLIPDataset(os.path.join(data_dir, 'train'), transform=train_transform)
    valid_dataset = CLIPDataset(os.path.join(data_dir, 'val'), transform=valid_transform)
    test_dataset = CLIPDataset(os.path.join(data_dir, 'test'), transform=valid_transform)
    return train_dataset, valid_dataset, test_dataset

def get_scheduler(optimizer, args, steps_per_epoch):
    """
    Configure the learning rate scheduler.
    """
    if args.warmup and args.cosine:
        num_warmup_steps = int(args.warmup_epochs * steps_per_epoch)
        num_training_steps = args.epochs * steps_per_epoch
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.warmup:
        num_warmup_steps = int(args.warmup_epochs * steps_per_epoch)
        return lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, float(step) / num_warmup_steps))
    elif args.cosine:
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch, eta_min=args.learning_rate * 0.01)
    else:
        return lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)

if __name__ == "__main__":
    
    # get parameters
    args = parse_arguments()
    # set seed
    set_seed(args.seed)
    # defined save dir
    save_dir = create_save_dir(args)
    args.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    # load model
    # load CLIP encoder model
    clip_encoder = CLIPEncoder(args.model, args.pretrained, args.device, args.save_dir, keep_lang=True)
    # initialize linear probing module with the CLIP encoder
    model = LinearProbModule(clip_encoder, args.n_dims, freeze_encoder=args.freeze_encoder)
    model = model.to(args.device)

    # load dataset
    # set transform
    # define data transformations for training and validation/test datasets
    train_transform, valid_transform = get_transforms()

    # load datasets
    train_dataset, valid_dataset, test_dataset = load_dataset(args.data_dir, train_transform, valid_transform)

    # create data loaders for training, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize loss function, optimizer, and scheduler
    criterion = ClipLoss(local_loss=False, gather_with_grad=False, cache_labels=True, rank=0, world_size=1, use_horovod=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args, len(train_loader))

    # load the templates from the specified path
    templates = import_function_from_path(args.template_dir, 'templates')

    # training loop
    history = {'train_loss':[], 'valid_acc': [], 'valid_loss': []}
    best_acc = 0.0

    for epoch in range(args.epochs):
        # training 
        train_loss = flyp_linearprob_train(model, train_loader, optimizer, scheduler, criterion, train_dataset, templates, epoch, args)

        # validation
        if (epoch + 1) % args.eval_freq == 0:
            valid_acc, valid_loss = flyp_linearprob_eval(model, valid_loader, criterion, valid_dataset, templates, args)

             # save the validation accuracy for the current epoch
            history['valid_acc'].append(valid_acc)
            history['valid_loss'].append(valid_loss)
            # save the training loss for the current epoch
            history['train_loss'].append(train_loss)
            # save the model if the validation accuracy is the best so far
            pd.DataFrame(history).to_csv(os.path.join(args.save_dir, 'history.csv'), index=False)

            if valid_acc > best_acc:
                best_acc = valid_acc
                save_model(
                    model.image_linearprob,
                    model.text_linearprob,
                    optimizer,
                    args,
                    epoch,
                    args.save_dir
                )
        print(f'Epoch: {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Valid Acc: {valid_acc:.4f}')