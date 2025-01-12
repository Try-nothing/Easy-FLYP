import torch
import time
import torch.nn.functional as F
from utils import turn_label2template
from utils import AverageMeter
import sys
from sklearn.metrics import accuracy_score



def update(self, val, n=1):
    if not torch.isnan(val) and not torch.isinf(val):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def compute_loss(criterion, image_features, text_features, labels, args):
    text_features = text_features[labels]
    labels = F.one_hot(labels, num_classes=args.n_classes).float()
    return criterion(image_features, text_features, labels, labels)


def flyp_linearprob_train(model, dataloader, optimizer, scheduler, criterion, dataset, templates, epoch, args):
    """
    Train the model for one epoch using linear probing.

    Args:
        model: The model to be trained.
        dataloader: DataLoader providing the training data.
        optimizer: Optimizer used for updating model parameters.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        dataset: Dataset object containing class information.
        templates: List of templates for text prompts.
        mixer: Function for data mixing (e.g., mixup).
        epoch: Current epoch number.
        args: Command-line arguments or configuration object.

    Returns:
        Average loss over the epoch.
    """
    model.train() # set model to training mode

    # initialize metrics for tracking time and loss
    batch_time = AverageMeter() # time taken to process each batch
    data_time = AverageMeter() # time taken to load data
    losses = AverageMeter() # average loss per batch

    end_time = time.time() # start time for the first batch
    for idx, (images, labels) in enumerate(dataloader):
        data_time.update(time.time() - end_time) # update data loading time
    
        # convert labels to text templates for CLIP
        templates_0 = turn_label2template(labels, dataset, templates)
        templates_1 = turn_label2template(labels, dataset, templates)

        # split the augmented images and move them to the specified device
        images_0, images_1 = images[0], images[1]
        images_0, images_1 = images_0.to(args.device), images_1.to(args.device)

        # get batch size
        local_bsz = images_0.size(0)
        
        # concatenate labels
        labels = torch.cat([labels, labels], dim=0)

        # concatenate the augmented images and templates
        images = torch.cat([images_0, images_1], dim=0)
        template_list = templates_0 + templates_1 # combine templates for both sets of images
        # forward pass: compute image and text features
        image_features, text_features, logit_scale = model(images, template_list)
        # compute the contrastive loss
        contrast_loss = criterion(image_features, text_features, logit_scale)

        # backward pass and optimization
        optimizer.zero_grad() # clear gradients
        contrast_loss.backward() # compute gradients
        optimizer.step() # update model parameters
        scheduler.step() # update learning rate

        # update metrics
        losses.update(contrast_loss.item(), local_bsz)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # print progress at the specified frequency
        if (idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'LR {local_lr:.6f} ({lr:.6f})'.format(
                epoch, idx + 1, len(dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses, local_lr=scheduler.get_last_lr()[0], lr=args.learning_rate))
            sys.stdout.flush()

    return losses.avg


@torch.no_grad()
def flyp_linearprob_eval(model, dataloader, criterion, dataset, templates, args):
    """
    Evaluate the model on the validation or test set.

    Args:
        model: The model to be evaluated.
        dataloader: DataLoader providing the evaluation data.
        dataset: Dataset object containing class information.
        templates: List of templates for text prompts.
        args: Command-line arguments or configuration object.

    Returns:
        Average accuracy over the evaluation set.
    """
    model.eval()
    torch.set_grad_enabled(False)

    # Initialize metrics
    batch_time = AverageMeter()
    accuracies = AverageMeter()
    losses = AverageMeter()

    # Precompute templates for all classes
    all_classes = torch.arange(args.n_classes).to(args.device)
    templates_0 = turn_label2template(all_classes, dataset, templates)

    end_time = time.time()
    for idx, (images, labels) in enumerate(dataloader):
        images = images[0].to(args.device)
        labels = labels.to(args.device)

        # Forward pass
        image_features, text_features, logit_scale = model(images, templates_0)

        # Compute loss
        contrast_loss = criterion(image_features, text_features[labels], logit_scale)
        losses.update(contrast_loss.item(), images.size(0))

        # Compute accuracy
        logits = torch.matmul(image_features, text_features.T) / args.temperature
        acc = compute_accuracy(logits, labels)
        accuracies.update(acc, images.size(0))

        # Update batch time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # Log progress
        if (idx + 1) % args.print_freq == 0:
            print('Eval: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                             idx + 1, len(dataloader), 
                             batch_time=batch_time, 
                             loss=losses,
                             acc=accuracies))

    return accuracies.avg, losses.avg