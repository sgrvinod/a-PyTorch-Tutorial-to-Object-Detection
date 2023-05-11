from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.models.feature_extraction as feature_extraction
import numpy as np
from itertools import product as product

from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        efficientnet_b3 = torchvision.models.efficientnet_b3(weights='EfficientNet_B3_Weights.DEFAULT')
        for param in efficientnet_b3.parameters():
            param.requires_grad = False

        return_nodes = {
            'features.4.0.block.0': "features_38",
            'features.6.0.block.0': "features_19",
            'features.8':           "features_10",
        }
        
        self.feature_extractor = feature_extraction.create_feature_extractor(efficientnet_b3, return_nodes=return_nodes)

    def forward(self, input):
        features = self.feature_extractor(input)
        features_38 = features['features_38']
        features_19 = features['features_19']
        features_10 = features['features_10']

        return features_38, features_19, features_10

class AuxiliaryNetwork(nn.Module):
    def __init__(self):
        super(AuxiliaryNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(1536, 384, kernel_size=1, padding=0)
        self.conv1_2 = nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(768, 192, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(384, 192, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(192, 384, kernel_size=3, padding=0)
        
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
        
    def forward(self, features_10):
        # (N, 1535, 10, 10)
        out = self.relu(self.conv1_1(features_10))  # (N, 384, 5, 5)
        out = self.relu(self.conv1_2(out))  # (N, 768, 5, 5)
        features_5 = out  # (N, 768, 5, 5)
       
        out = self.relu(self.conv2_1(features_5))  # (N, 192, 3, 3)
        out = self.relu(self.conv2_2(out))  # (N, 384, 3, 3)
        features_3 = out  # (N, 384, 3, 3)
       
        out = self.relu(self.conv3_1(features_3))  # (N, 192, 1, 1)
        out = self.relu(self.conv3_2(out))  # (N, 384, 1, 1)
        features_1 = out  # (N, 384, 1, 1))

        return features_5, features_3, features_1

class PredictionNetwork(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.
    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionNetwork, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'features_38': 4,
                   'features_19': 6,
                   'features_10': 6,
                   'features_5': 6,
                   'features_3': 4,
                   'features_1': 4}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_38 = nn.Conv2d(288, n_boxes['features_38'] * 4, kernel_size=3, padding=1)
        self.loc_19 = nn.Conv2d(816, n_boxes['features_19'] * 4, kernel_size=3, padding=1)
        self.loc_10 = nn.Conv2d(1536, n_boxes['features_10'] * 4, kernel_size=3, padding=1)
        self.loc_5 = nn.Conv2d(768, n_boxes['features_5'] * 4, kernel_size=3, padding=1)
        self.loc_3 = nn.Conv2d(384, n_boxes['features_3'] * 4, kernel_size=3, padding=1)
        self.loc_1 = nn.Conv2d(384, n_boxes['features_1'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_38 = nn.Conv2d(288, n_boxes['features_38'] * n_classes, kernel_size=3, padding=1)
        self.cl_19 = nn.Conv2d(816, n_boxes['features_19'] * n_classes, kernel_size=3, padding=1)
        self.cl_10 = nn.Conv2d(1536, n_boxes['features_10'] * n_classes, kernel_size=3, padding=1)
        self.cl_5 = nn.Conv2d(768, n_boxes['features_5'] * n_classes, kernel_size=3, padding=1)
        self.cl_3 = nn.Conv2d(384, n_boxes['features_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_1 = nn.Conv2d(384, n_boxes['features_1'] * n_classes, kernel_size=3, padding=1)
                
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, features):
        batch_size = features[0].size(0)

        localization_convolutions = [
            self.loc_38,
            self.loc_19,
            self.loc_10,
            self.loc_5,
            self.loc_3,
            self.loc_1,
        ]

        class_convolutions = [
            self.cl_38,
            self.cl_19,
            self.cl_10,
            self.cl_5,
            self.cl_3,
            self.cl_1,
        ]

        localization_predictions = []
        class_predictions = []
        for l_conv, c_conv, feat in zip(localization_convolutions, class_convolutions, features):
            localization_prediction = l_conv(feat)
            localization_prediction = localization_prediction.permute(0, 2, 3, 1).contiguous()
            localization_prediction = localization_prediction.view(batch_size, -1, 4)
            localization_predictions.append(localization_prediction)

            class_prediction = c_conv(feat)
            class_prediction = class_prediction.permute(0, 2, 3, 1).contiguous()
            class_prediction = class_prediction.view(batch_size, -1, self.n_classes)
            class_predictions.append(class_prediction)

        localizations = torch.cat(localization_predictions, dim=1)
        class_scores = torch.cat(class_predictions, dim=1)
        return localizations, class_scores

class EfficientNetSSD300(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNetSSD300, self).__init__()

        self.n_classes = n_classes

        self.base = Base()
        self.aux = AuxiliaryNetwork()
        self.pred = PredictionNetwork(n_classes)

        # Since lower level features (features_38) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 288, 1, 1))  # there are 288 channels in features_38
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.anchor_boxes = self.get_anchor_boxes()

    def forward(self, batch):
        features_38, features_19, features_10 = self.base(batch)

        # Rescale features_38 after L2 norm
        norm = features_38.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        features_38 = features_38 / norm  # (N, 512, 38, 38)
        features_38 = features_38 * self.rescale_factors  # (N, 512, 38, 38)

        # Run auxiliary convolutions (higher level feature map generators)
        features_5, features_3, features_1 = self.aux(features_10) 

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, class_scores = self.pred([features_38, features_19, features_10, features_5, features_3, features_1])  

        # (N, 8732, 4), (N, 8732, n_classes)
        return locs, class_scores
    
    def get_anchor_boxes(self):
        resolutions = [38, 19, 10, 5, 3, 1]
        scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
        aspect_ratios = [
            [1, 2, 1/2],
            [1, 2, 1/2, 3, 1/3],
            [1, 2, 1/2, 3, 1/3],
            [1, 2, 1/2, 3, 1/3],
            [1, 2, 1/2],
            [1, 2, 1/2],
        ]
        
        anchor_boxes = []
        for k, (resolution, scale, ratios) in enumerate(zip(resolutions, scales, aspect_ratios)):
            for i in range(resolution):
                for j in range(resolution):
                    x = (i + 0.5) / resolution
                    y = (j + 0.5) / resolution
                    for ratio in ratios:
                        w = scale * np.sqrt(ratio)
                        h = scale / np.sqrt(ratio)
                        anchor_boxes.append([x, y, w, h])
                        if ratio == 1 : # add in an additional scaled up square box
                            extra_scale = 1.
                            if k + 1 < len(scales):
                                extra_scale = np.sqrt(scale * scales[k + 1])
                            anchor_boxes.append([x, y, extra_scale, extra_scale])

        anchor_boxes = torch.FloatTensor(anchor_boxes).to(device)  # (8732, 4)
        anchor_boxes.clamp_(0, 1)  # (8732, 4)
        return anchor_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        batch_size = predicted_locs.size(0)
        n_anchors = self.anchor_boxes.size(0)
        predicted_scores = nn.functional.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_anchors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.anchor_boxes))  # (8732, 4), these are fractional pt. coordinates

            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            for c in range(1, self.n_classes):
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)
                
                # Non-Maximum Suppression (NMS)

                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                for box in range(class_decoded_locs.size(0)):
                    if suppress[box] == 1:
                        continue
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    suppress[box] = 0

                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

class MultiBoxLoss(nn.Module):
    def __init__(self, anchor_boxes, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.anchor_boxes = anchor_boxes
        self.anchor_box_boundaries = cxcy_to_xy(self.anchor_boxes)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def get_IoUs(self, boxes):
        intersections = torch.FloatTensor(len(boxes), len(self.anchor_boxes))
        boxes = boxes/300
        for b in range(len(boxes)):
            for a in range(len(self.anchor_boxes)):
                intersections[b][a] = get_intersection_over_union(box, self.anchor_boxes[a])
        return intersections

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_anchors = self.anchor_boxes.size(0)
        n_classes = predicted_scores.size(2)

        assert n_anchors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_anchors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_anchors), dtype=torch.long).to(device)  # (N, 8732)
        
        # For each set of boxes belonging to an image
        for i in range(batch_size):
            scaled_boxes = torch.stack([box/300 for box in boxes[i]])
            n_objects = boxes[i].size(0)
            # overlap is a (objects, anchor_boxes) tensor containing the IoU of each anchor for each object
            # IoU = self.get_IoUs(boxes[i])
            IoU = find_jaccard_overlap(scaled_boxes, self.anchor_boxes)
            IoU_for_each_anchor, object_for_each_anchor = IoU.max(dim=0)  # (8732)

            _, anchor_for_each_object = IoU.max(dim=1)  # (N_o)

            object_for_each_anchor[anchor_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            IoU_for_each_anchor[anchor_for_each_object] = 1.

            label_for_each_anchor = labels[i][object_for_each_anchor]  # (8732)
            label_for_each_anchor[IoU_for_each_anchor < self.threshold] = 0  # (8732)

            true_classes[i] = label_for_each_anchor

            true_locs[i] = cxcy_to_gcxgcy(boxes[i][object_for_each_anchor], self.anchor_boxes)  # (8732, 4)
        positive_anchors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS
        loc_loss = self.smooth_l1(predicted_locs[positive_anchors], true_locs[positive_anchors])  # (), scalar

        # CONFIDENCE LOSS

        n_positives = positive_anchors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_anchors)  # (N, 8732)

        conf_loss_pos = conf_loss_all[positive_anchors]  # (sum(n_positives))

        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_anchors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_anchors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar
        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss
