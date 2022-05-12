import torch
import torch.nn as nn
from RainforestDataset import get_classes_list
import torchvision.models as models

class TwoNetworks(nn.Module):
    '''
    This class takes two pretrained networks,
    concatenates the high-level features before feeding these into
    a linear layer.

    functions: forward
    '''
    def __init__(self, pretrained_net1 = models.resnet18(pretrained=True), pretrained_net2 = models.resnet18(pretrained=True)):
        super(TwoNetworks, self).__init__()

        _, num_classes = get_classes_list()
        self.network = "TwoNetworks"
        # select all parts of the two pretrained networks, except for
        # the last linear layer.
        #self.fully_conv1 = [module for module in pretrained_net1.modules() if not isinstance(module, nn.Sequential)][:-1]
        #self.fully_conv2 = [module for module in pretrained_net2.modules() if not isinstance(module, nn.Sequential)][:-1]
        out_first_layer = pretrained_net2.conv1.out_channels
        pretrained_net2.conv1 = nn.Conv2d(1, out_first_layer, kernel_size=7, stride=2, padding=3, bias=False)
        #pretrained_net2.conv1 = nn.init.kaiming_normal_(pretrained_net2.conv1.weight)

        self.fully_conv1 = torch.nn.Sequential(*(list(pretrained_net1.children())[:-1]))
        self.fully_conv2 = torch.nn.Sequential(*(list(pretrained_net2.children())[:-1]))

        #print("Length layers 1: ", len(self.fully_conv1))
        #print(self.fully_conv1)
        # Change the first layer of pretrain_net2 to only take in 1 channel (infrared)

        #out_first_layer = pretrained_net1.conv1.out_channels
        #self.fully_conv2[0] = nn.Conv2d(1, out_first_layer, kernel_size=7, stride=2, padding=3, bias=False)
        #self.fully_conv2[0] = nn.init.kaiming_normal_(self.fully_conv2[0].weight)


        # create a linear layer that has in_channels equal to
        # the number of in_features from both networks summed together.
        in_last_layer = pretrained_net1.fc.in_features
        self.fc = nn.Linear(2*in_last_layer, num_classes)


    def forward(self, inputs1, inputs2):
        # TODO feed the inputs through the fully convolutional parts
        # of the two networks that you initialised above, and then
        # concatenate the features before the linear layer.
        # And return the result.

        #print("Inputs size:", inputs1.size(),inputs2.size())
        x1 = self.fully_conv1(inputs1)
        x2 = self.fully_conv2(inputs2)


        last_features = torch.cat((x1,x2),dim = 1).squeeze()
        #print(x1.size(), x2.size(), last_features.size())
        #print(self.fc)
        return self.fc(last_features)


class SingleNetwork(nn.Module):
    '''
    This class takes one pretrained network,
    the first conv layer can be modified to take an extra channel.

    functions: forward
    '''

    def __init__(self, pretrained_net = models.resnet18(pretrained=True), weight_init=None):
        super(SingleNetwork, self).__init__()

        _, num_classes = get_classes_list()

        if weight_init is not None:
            # TODO Here we want an additional channel in the weights tensor, specifically in the first
            # conv2d layer so that there are weights for the infrared channel in the input aswell.
            
            current_weight = pretrained_net.conv1.weight.clone()
            out_features = pretrained_net.conv1.out_channels
            pretrained_net.conv1 = nn.Conv2d(4, out_features, kernel_size=7, stride=2, padding=3, bias=False)

            with torch.no_grad():
                pretrained_net.conv1.weight[:, :3] = current_weight
                pretrained_net.conv1.weight[:, 3] = pretrained_net.conv1.weight[:, 0]
            
            if weight_init == "kaiminghe":
                with torch.no_grad():
                    pretrained_net.conv1.weight[:, 3] = nn.init.kaiming_normal_(pretrained_net.conv1.weight[:, 3])
            
            # TODO Create a new conv2d layer, and set the weights to be
            # what you created above. You will need to pass the weights to
            # torch.nn.Parameter() so that the weights are considered
            # a model parameter.
            # eg. first_conv_layer.weight = torch.nn.Parameter(your_new_weights)

        # TODO Overwrite the last linear layer.

        last_layer_inputs = pretrained_net.fc.in_features
        pretrained_net.fc = nn.Linear(last_layer_inputs, num_classes)

        self.net = pretrained_net


    def forward(self, inputs):
        return self.net(inputs)





