# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.model import BaseModel
import timm
from os.path import isfile
import torch.nn as nn
from glob import glob
from os.path import join
import torch
from torchmetrics import Accuracy, ConfusionMatrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchsummary import summary


# def
def load_from_checkpoint(
    device,
    checkpoint_path,
    classes,
    model,
):
    device = device if device == 'cuda' and torch.cuda.is_available(
    ) else 'cpu'
    map_location = torch.device(device=device)
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    # change the number of output
    for key in checkpoint['state_dict'].keys():
        if 'classifier.bias' in key or 'classifier.weight' in key:
            if checkpoint['state_dict'][key].shape[0] != len(classes):
                temp = checkpoint['state_dict'][key]
                checkpoint['state_dict'][key] = torch.stack([temp.mean(0)] *
                                                            len(classes), 0)
    # change the weight of loss function
    if model.loss_function.weight is None:
        if 'loss_function.weight' in checkpoint['state_dict']:
            # delete loss_function.weight in the checkpoint
            del checkpoint['state_dict']['loss_function.weight']
    else:
        # override loss_function.weight with model.loss_function.weight
        checkpoint['state_dict'][
            'loss_function.weight'] = model.loss_function.weight
    model.load_state_dict(checkpoint['state_dict'])
    return model


def create_model(project_parameters):
    model = SupervisedModel(
        optimizers_config=project_parameters.optimizers_config,
        lr=project_parameters.lr,
        lr_schedulers_config=project_parameters.lr_schedulers_config,
        model_name=project_parameters.model_name,
        in_chans=project_parameters.in_chans,
        classes=project_parameters.classes,
        loss_function_name=project_parameters.loss_function_name,
        data_balance=project_parameters.data_balance,
        root=project_parameters.root)
    if project_parameters.checkpoint_path is not None:
        if isfile(project_parameters.checkpoint_path):
            model = load_from_checkpoint(
                device=project_parameters.device,
                checkpoint_path=project_parameters.checkpoint_path,
                classes=project_parameters.classes,
                model=model)
        else:
            assert False, 'please check the checkpoint_path argument.\nthe checkpoint_path value is {}.'.format(
                project_parameters.checkpoint_path)
    return model


# class
class SupervisedModel(BaseModel):
    def __init__(self, optimizers_config, lr, lr_schedulers_config, model_name,
                 in_chans, classes, loss_function_name, data_balance,
                 root) -> None:
        super().__init__(optimizers_config, lr, lr_schedulers_config)
        self.backbone_model = self.create_backbone_model(model_name=model_name,
                                                         in_chans=in_chans,
                                                         classes=classes)
        self.activation_function = nn.Sigmoid()
        self.loss_function = self.create_loss_function(
            loss_function_name=loss_function_name,
            data_balance=data_balance,
            root=root,
            classes=classes)
        self.accuracy_function = Accuracy()
        self.confusion_matrix_function = ConfusionMatrix(
            num_classes=len(classes))
        self.classes = classes
        self.stage_index = 0

    def create_backbone_model(self, model_name, in_chans, classes):
        if model_name in timm.list_models():
            backbone_model = timm.create_model(model_name=model_name,
                                               pretrained=True,
                                               in_chans=in_chans,
                                               num_classes=len(classes))
        elif isfile(model_name):
            class_name = self.import_class_from_file(filepath=model_name)
            backbone_model = class_name(in_chans=in_chans,
                                        num_classes=len(classes))
        else:
            assert False, 'please check the model_name argument.\nthe model_name value is {}.'.format(
                model_name)
        return backbone_model

    def get_files(self, filepath, extensions):
        files = []
        for v in extensions:
            files += glob(join(filepath, '*{}'.format(v)))
        return files

    def calculate_weight(self, root, classes):
        weight = {}
        for c in classes:
            files = self.get_files(filepath=join(root, 'train/{}'.format(c)),
                                   extensions=('.jpg', '.jpeg', '.png', '.ppm',
                                               '.bmp', '.pgm', '.tif', '.tiff',
                                               '.webp'))
            weight[c] = len(files)
        weight = {
            c: 1 - (weight[c] / sum(weight.values()))
            for c in weight.keys()
        }
        return weight

    def create_loss_function(self, loss_function_name, data_balance, root,
                             classes):
        loss_function = super().create_loss_function(loss_function_name)
        if data_balance:
            weight = self.calculate_weight(root=root, classes=classes)
            weight = torch.Tensor(list(weight.values()))
        else:
            weight = None
        loss_function.weight = weight
        return loss_function

    def forward(self, x):
        return self.activation_function(self.backbone_model(x))

    def shared_step(self, batch):
        x, y = batch
        y_hat = self.backbone_model(x)
        loss = self.loss_function(y_hat, y)
        accuracy = self.accuracy_function(self.activation_function(y_hat),
                                          y.argmax(-1))
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.shared_step(batch=batch)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('train_accuracy',
                 accuracy,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.shared_step(batch=batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone_model(x)
        loss = self.loss_function(y_hat, y)
        accuracy = self.accuracy_function(self.activation_function(y_hat),
                                          y.argmax(-1))
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        # NOTE: in the 1.4.9+ version of PyTorch Lightning,
        # if deterministic is set to True,
        # an error will occur while calculating the confusion matrix.
        # it can use torch.use_deterministic_algorithms(False) to solved.
        confusion_matrix_step = self.confusion_matrix_function(
            y_hat.argmax(-1), y.argmax(-1)).cpu().data.numpy()
        loss_step = loss.item()
        accuracy_step = accuracy.item()
        return {
            'confusion_matrix': confusion_matrix_step,
            'loss': loss_step,
            'accuracy': accuracy_step
        }

    def test_epoch_end(self, test_outs):
        stages = ['train', 'val', 'test']
        print('\ntest the {} dataset'.format(stages[self.stage_index]))
        print('the {} dataset confusion matrix:'.format(
            stages[self.stage_index]))
        confusion_matrix = np.sum([v['confusion_matrix'] for v in test_outs],
                                  0)
        loss = np.mean([v['loss'] for v in test_outs])
        accuracy = np.mean([v['accuracy'] for v in test_outs])
        # use pd.DataFrame to wrap the confusion matrix to display it to the CLI
        confusion_matrix = pd.DataFrame(data=confusion_matrix,
                                        columns=self.classes,
                                        index=self.classes).astype(int)
        print(confusion_matrix)
        plt.figure(figsize=[11.2, 6.3])
        plt.title('{}\nloss: {}\naccuracy: {}'.format(stages[self.stage_index],
                                                      loss, accuracy))
        figure = sns.heatmap(data=confusion_matrix,
                             cmap='Spectral',
                             annot=True,
                             fmt='g').get_figure()
        plt.yticks(rotation=0)
        plt.ylabel(ylabel='Actual class')
        plt.xlabel(xlabel='Predicted class')
        plt.close()
        self.logger.experiment.add_figure(
            '{} confusion matrix'.format(stages[self.stage_index]), figure,
            self.current_epoch)
        self.stage_index += 1


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    summary(model=model,
            input_size=(project_parameters.in_chans, 224, 224),
            device='cpu')

    # create input data
    x = torch.ones(project_parameters.batch_size, project_parameters.in_chans,
                   224, 224)

    # get model output
    y = model(x)

    # display the dimension of input and output
    print('the dimension of input: {}'.format(x.shape))
    print('the dimension of output: {}'.format(y.shape))
