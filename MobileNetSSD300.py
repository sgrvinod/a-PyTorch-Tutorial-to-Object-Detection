from torch import nn, Tensor
import torchsummary
import torchvision
import torchvision.models.feature_extraction as feature_extraction

class MobileNetV2(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(MobileNetV2, self).__init__()
        mobilenet_v2 = torchvision.models.mobilenet_v2()#(weights='MobileNet_V2_WEIGHTS.DEFAULT')
        # self.graph_names = feature_extraction.get_graph_node_names(mobilenet_v2)[0]
        # return_nodes = {v: str(i) for i, v in enumerate(self.graph_names)}

        self.return_nodes = {
            'features.1.conv.0':  "features_150", #96
            'features.4.conv.0':  "features_75",  #144
            'features.5.conv.0':  "features_38",  #192
            'features.13.conv.0': "features_19",  #576
            'features.17.conv.0': "features_10"   #320 channel
        }
        
        self.feature_extractor = feature_extraction.create_feature_extractor(mobilenet_v2, return_nodes=self.return_nodes)

    def forward(self, input):
        features = self.feature_extractor(input)
        return features.values()

if __name__ == '__main__':
    mnet = MobileNetV2()
    x = mnet.forward(Tensor(1, 3, 300, 300))

    for name, shape in zip(mnet.return_nodes, [v.shape for v in x]):
        print(name, shape)
