import torch
import copy
import utils
import open_clip
import clip.clip as clip
from open_clip import tokenizer
import torch.nn as nn


class CLIPEncoder(torch.nn.Module):
    def __init__(self, model, pretrained, device, cache_dir, keep_lang=False):
        super().__init__()
        # check the model type and load the appropriate CLIP model and preprocessors
        if model == 'ViT-L-14':
            self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                model, pretrained=pretrained)
        elif model == 'ViT-B-16':
            print("****************Loading ViTB16 from openCLIP****************")
            self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                model, pretrained=pretrained)
        else:
            self.model, self.train_preprocess, self.val_preprocess = clip.load(
                model, device, jit=False)
        # Store the cache directory for potential use
        self.cache_dir = cache_dir

        self.tokenizer = tokenizer
        self.device = device

    def forward(self, images, text=None):
        assert self.model is not None
        # if text==None:
        #     return self.model.encode_image(images)
        # else:
        text = self.tokenizer.tokenize(text).to(self.device)
        return self.model(images, text)

    def save(self, filename):
        print(f'Saving clip encoder to {filename}')
        utils.torch_save(self, filename)
        # torch.save(self.model, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading image encoder from {filename}')
        if logger != None:
            logger.info(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    # `ClassificationHead` is a linear layer with input normalization and pre-trained weight support, 
    # designed to map feature vectors to the output space of classification tasks.
    def __init__(self, normalize, weights, biases=None, shape=[512, 1000]):
        if weights is not None:
            output_size, input_size = weights.shape
            super().__init__(input_size, output_size)
        else:
            super().__init__(shape[0], shape[1])
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        else:
            self.weight = torch.nn.Parameter(torch.eye(shape[0], shape[1]))

        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading classification head from {filename}')
        if logger != None:
            logger.info(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    # ImageClassifier is a module that combines an image encoder and a classification head 
    # to process images and map them to classification outputs
    def __init__(self,
                 image_encoder,
                 classification_head,
                 process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


class ImageClassifier_Norm(torch.nn.Module):
    # combines an image encoder and a classification head
    # normalizing the encoded features before mapping them to classification outputs
    def __init__(self,
                 image_encoder,
                 classification_head,
                 process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


class ImageEncoder(torch.nn.Module):
    def __init__(self, model, device, cache_dir, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            model, device, jit=False)

        self.cache_dir = cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)
    

class LinearProbModule(torch.nn.Module):
    def __init__(self, clip_encoder, features_dim, freeze_encoder:bool):
        """
        Initialize the LinearProbModule.

        Args:
            clip_encoder (nn.Module): The CLIP encoder model.
            features_dim (int): The dimension of the features output by the CLIP encoder.
            freeze_encoder (bool): Whether to freeze the CLIP encoder parameters.
        """
        super(LinearProbModule, self).__init__()  # Initialize the parent class
        self.clip_encoder = clip_encoder  # The CLIP encoder model

        self.features_dim = features_dim  # Dimension of the features
        self.freeze_encoder = freeze_encoder  # Whether to freeze the encoder

        # Freeze the CLIP encoder parameters if freeze_encoder is True
        if self.freeze_encoder:
            self.clip_encoder.eval()  # Set the encoder to evaluation mode
            for param in self.clip_encoder.parameters():
                param.requires_grad = False  # Disable gradient computation for encoder parameters
        else:
            self.clip_encoder.train()

        # Initialize linear classification heads for image and text features
        self.image_linearprob = ClassificationHead(normalize=True, weights=None, shape=[512, features_dim])
        self.text_linearprob = ClassificationHead(normalize=True, weights=None, shape=[512, features_dim])

        self.image_linearprob.train()
        self.text_linearprob.train()

    def forward(self, images, texts):
        """
        Forward pass of the LinearProbModule.

        Args:
            images (torch.Tensor): Input images.
            texts (torch.Tensor): Input texts.

        Returns:
            tuple: A tuple containing:
                - image_features (torch.Tensor): Transformed image features.
                - text_features (torch.Tensor): Transformed text features.
        """
        # Extract features from the CLIP encoder without computing gradients
        image_features, text_features, logit_scale = self.clip_encoder(images, texts)
        # Apply linear classification heads to the features
        if self.freeze_encoder:
            image_features = self.image_linearprob(image_features.detach())
            text_features = self.text_linearprob(text_features.detach())
        else:
            image_features = self.image_linearprob(image_features)
            text_features = self.text_linearprob(text_features)

        return image_features, text_features, logit_scale