import torch
import torch.nn as nn

# import densenet as backbone
# from utils import MarginLoss, CFE_refine, UFE_E2, UCFCFusion

import model.densenet as backbone
from model.utils import MarginLoss, CFE_refine, UFE_E2, UCFCFusion

# Task-Common Feature Extraction module
class CFE(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, Arcm=0.5, multi=3, factor=1):
        super(CFE, self).__init__()
        dim_mlp = 1024
        self.name = 'CFE'

        # Initialize DenseNet backbone with specified input channels
        self.E1 = backbone.maskdense121(in_channels=in_channels)

        # Initialize refinement module with specified input channels and factor
        self.refine = CFE_refine(in_channels=[64, 128, 256, 512, 1024, 1024], factor=factor)

        # Initialize margin loss for SFTFPPredict with specified parameters
        self.SFTFPPredict = MarginLoss(in_features=dim_mlp, out_features=2, s=64.0, m=Arcm)

        # Initialize ModuleLists for normal vs tumour classification and tumour classification
        self.S1Predict = nn.ModuleList([])
        self.S2Predict = nn.ModuleList([])

        # Populate the ModuleLists with MarginLoss modules
        for i in range(multi):
            self.S1Predict.append(MarginLoss(in_features=dim_mlp, out_features=num_classes + 1, s=64.0, m=Arcm))
            self.S2Predict.append(MarginLoss(in_features=dim_mlp // factor, out_features=num_classes, s=64.0, m=Arcm))

        # Adaptive average pooling layer for 3D inputs
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def GetS1Predict(self, features_all, label_list):
        size_img = label_list[0].size(0)
        out_cls_list_NG = []
        # Iterate over the first three classification tasks
        for ssl_index in range(0, 3):
            # Extract features for the current task
            features_cls = features_all[size_img * ssl_index:size_img + size_img * ssl_index, :]
            # Apply the S1Predict margin loss
            out_cls_NG = self.S1Predict[ssl_index](features_cls, label_list[ssl_index])
            out_cls_list_NG.append(out_cls_NG)
        return out_cls_list_NG

    # Binary classification: returns true class labels for loss computation
    def GetS2Predict(self, features_all, label_list):
        size_img = label_list[0].size(0)
        out_cls_labels = []
        out_cls_list = []
        # Iterate over the first three classification tasks
        for ssl_index in range(0, 3):
            # Extract features for the current task
            features_cls = features_all[size_img * ssl_index:size_img + size_img * ssl_index, :]
            # Select features where label indicates glioma (label > 0)
            index_glioma = label_list[ssl_index] > 0
            features_cls_glioma = features_cls[index_glioma, :]
            # Adjust labels by subtracting 1 for binary classification
            features_cls_labels = label_list[ssl_index][index_glioma] - 1
            # Apply the S2Predict margin loss
            out_cls = self.S2Predict[ssl_index](features_cls_glioma, features_cls_labels)
            out_cls_list.append(out_cls)
            out_cls_labels.append(features_cls_labels)
        return out_cls_list, out_cls_labels

    # Forward pass with three-class labels
    # Labels: 0 = no tumor, 1 & 2 = two classes for normal binary classification
    def forward(self, input, labels_NG, label_list):
        # Extract features using the DenseNet backbone
        features_t1, features_t2, features_t3, features_t4, features_t5 = self.E1(input)

        # Apply average pooling and flatten the features
        Fi = self.avgpool(features_t5)
        Fi = torch.flatten(Fi, 1)

        # Concatenate all normal vs tumour labels
        labels_NG_all = torch.cat(labels_NG, 0)

        # Predict normal vs tumour logits
        logits_NG = self.SFTFPPredict(Fi, labels_NG_all)

        # Refine features using the CFE_refine module
        Fs1, Fs2 = self.refine(features_t1, features_t2, features_t3, features_t4, features_t5)

        # Apply average pooling and flatten Fs1
        Fs1 = self.avgpool(Fs1)
        Fs1 = torch.flatten(Fs1, 1)

        # Apply average pooling and flatten Fs2
        Fs2 = self.avgpool(Fs2)
        Fs2 = torch.flatten(Fs2, 1)

        # Get predictions for normal vs tumour classification
        out_cls_list_NG = self.GetS1Predict(Fs1, label_list)

        # Get predictions for tumour classification
        out_cls_list, out_cls_labels = self.GetS2Predict(Fs2, label_list)

        return logits_NG, labels_NG_all, out_cls_list_NG, out_cls_list, out_cls_labels

    def predict(self, inputs, ssl_index):
        with torch.no_grad():
            # Extract features using the DenseNet backbone
            features_t1, features_t2, features_t3, features_t4, features_t5 = self.E1(inputs)

            # Refine features
            Fs1, Fs2 = self.refine(features_t1, features_t2, features_t3, features_t4, features_t5)

            # Apply average pooling and flatten Fs2
            Fs2 = self.avgpool(Fs2)
            Fs2 = torch.flatten(Fs2, 1)

            # Get prediction from the S2Predict module
            out = self.S2Predict[ssl_index].predict(Fs2)
            return out

    def get_Sharefeatures(self, inputs):
        with torch.no_grad():
            # Extract features using the DenseNet backbone
            features_t1, features_t2, features_t3, features_t4, features_t5 = self.E1(inputs)

            # Get shared features using the refine module
            Fc = self.refine.getFc(features_t1, features_t2, features_t3, features_t4, features_t5)
            return features_t1, Fc

# Task-Specific Unique Feature Extraction module
class UFE(nn.Module):
    def __init__(self, num_classes=3, Arcm=0.5, multi=3, factor=1):
        super(UFE, self).__init__()
        dim_mlp = 1024
        self.name = 'UFE'

        # Initialize UFE_E2 module with specified input channels and factor
        self.E2 = UFE_E2(in_channels=[3, 64, 128, 256, 512, 1024, 1024], factor=factor)

        # Initialize Classifier ModuleList with MarginLoss modules
        self.Classifer = nn.ModuleList([])
        for i in range(multi):
            self.Classifer.append(MarginLoss(in_features=dim_mlp, out_features=num_classes, s=64.0, m=Arcm))

        # Adaptive average pooling layer for 3D inputs
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    # Binary classification: returns true class labels for loss computation
    def class_glioma(self, features_all, label_list):
        size_img = label_list[0].size(0)
        out_cls_labels = []
        out_cls_list = []
        # Iterate over the first three classification tasks
        for ssl_index in range(0, 3):
            # Extract features for the current task
            features_cls = features_all[size_img * ssl_index:size_img + size_img * ssl_index, :]
            # Get the labels for the current task
            features_cls_labels = label_list[ssl_index]
            # Apply the classifier margin loss
            out_cls = self.Classifer[ssl_index](features_cls, features_cls_labels)
            out_cls_list.append(out_cls)
            out_cls_labels.append(features_cls_labels)
        return out_cls_list, out_cls_labels

    def forward(self, input, label_list, condition, Fl):
        # Learn unique features by removing shared features through non-linear projection
        Fuhc_prime = self.E2(input, Fl, condition)

        # Apply average pooling and flatten the features
        Fuhc_prime_avg = self.avgpool(Fuhc_prime)
        Fuhc_prime_avg = torch.flatten(Fuhc_prime_avg, 1)

        # Get classification outputs and labels for glioma classification
        out_cls_list, out_cls_labels = self.class_glioma(Fuhc_prime_avg, label_list)
        return Fuhc_prime, out_cls_list, out_cls_labels

    def get_Uniquefeatures(self, input, condition, Fl):
        # Learn unique features by removing shared features through non-linear projection
        Fuhc_prime = self.E2(input, Fl, condition)
        return Fuhc_prime

# Unique-Common Feature Collaborative Classification module
class UCFC(nn.Module):
    def __init__(self, num_classes=2, multi=3, factor=1):
        super(UCFC, self).__init__()
        dim_mlp = 1024

        # Define the final fusion block with UCFCFusion, BatchNorm, and ReLU
        self.block_final = nn.Sequential(
            UCFCFusion(channel=dim_mlp,
                       ochannel=dim_mlp // factor,
                       img_size=[5, 6, 5],
                       fusion_num=2),
            nn.BatchNorm3d(dim_mlp),
            nn.ReLU()
        )

        # Define the fully connected block with Conv3d, BatchNorm, and ReLU
        self.block_fc = nn.Sequential(
            nn.Conv3d(in_channels=dim_mlp * 2,
                      out_channels=dim_mlp,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(dim_mlp // factor),
            nn.ReLU()
        )

        # Initialize Classifier ModuleList with Linear layers
        self.Classifer = nn.ModuleList([])
        for i in range(multi):
            self.Classifer.append(nn.Linear(dim_mlp // factor, num_classes))

        # Adaptive average pooling layer for 3D inputs
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    # Binary classification: returns true class labels for loss computation
    def class_glioma(self, features_all, label_list):
        size_img = label_list[0].size(0)
        out_cls_labels = []
        out_cls_list = []
        # Iterate over the first three classification tasks
        for ssl_index in range(0, 3):
            # Extract features for the current task
            features_cls = features_all[size_img * ssl_index:size_img + size_img * ssl_index, :]
            # Get the labels for the current task
            features_cls_labels = label_list[ssl_index]
            # Apply the classifier linear layer
            out_cls = self.Classifer[ssl_index](features_cls)
            out_cls_list.append(out_cls)
            out_cls_labels.append(features_cls_labels)
        return out_cls_list, out_cls_labels

    def forward(self, Fuhc_prime, Fc, label_list):
        # Apply the fully connected block to Fc
        Fc = self.block_fc(Fc)

        # Concatenate unique and shared features
        featuresF = torch.cat((Fuhc_prime, Fc), 1)

        # Apply the final fusion block
        featuresF = self.block_final(featuresF)

        # Apply average pooling and flatten the features
        featuresF = self.avgpool(featuresF)
        featuresF = torch.flatten(featuresF, 1)

        # Get classification outputs and labels for glioma classification
        out_cls_list, out_cls_labels = self.class_glioma(featuresF, label_list)
        return out_cls_list, out_cls_labels

    def predict(self, Fuhc_prime, Fc, ssl_index):
        with torch.no_grad():
            # Apply the fully connected block to Fc
            Fc = self.block_fc(Fc)

            # Concatenate unique and shared features
            featuresF = torch.cat((Fuhc_prime, Fc), 1)

            # Apply the final fusion block
            featuresF = self.block_final(featuresF)

            # Apply average pooling and flatten the features
            featuresF = self.avgpool(featuresF)
            featuresF = torch.flatten(featuresF, 1)

            # Get prediction from the classifier linear layer
            out = self.Classifer[ssl_index](featuresF)
            return out

# Cooperative Multi-Task Learning Network
class CMTLNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, Arcm=0.5, multi=3, factor=1):
        super(CMTLNet, self).__init__()

        # Initialize shared and unique task modules along with UCFC fusion module
        self.ShareTask = CFE(in_channels=in_channels, num_classes=num_classes, Arcm=Arcm, multi=multi, factor=factor)
        self.UniqueTask = UFE(num_classes=num_classes, Arcm=Arcm, multi=multi, factor=factor)
        self.UCFC = UCFC(num_classes=num_classes, multi=multi, factor=factor)
        self.name = 'CMTLNet'

    def _set_requires_grad(self, nets, requires_grad):
        # Helper function to set the requires_grad attribute for networks
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, input, label_list, condition):
        # Freeze the ShareTask parameters
        self._set_requires_grad(self.ShareTask, False)

        with torch.no_grad():
            # Get shared features from the ShareTask
            Fl, Fc = self.ShareTask.get_Sharefeatures(input)

        # Get unique features and classification outputs from the UniqueTask
        Fuhc_prime, out_cls_list_UFE, out_cls_labels_UFE = self.UniqueTask(input, label_list, condition, Fl)

        # Get classification outputs from the UCFC fusion module
        out_cls_list_UCFC, out_cls_labels_UCFC = self.UCFC(Fuhc_prime, Fc, label_list)

        return out_cls_list_UFE, out_cls_labels_UFE, out_cls_list_UCFC, out_cls_labels_UCFC

    def predict(self, inputs, ssl_index):
        with torch.no_grad():
            # Create a condition tensor based on the ssl_index
            ssl_index_condition = \
            torch.tensor(ssl_index, dtype=torch.long, device=inputs.device).repeat(1, inputs.size(0))[0]

            # Get shared features from the ShareTask
            Fl, Fc = self.ShareTask.get_Sharefeatures(inputs)

            # Get unique features from the UniqueTask
            Fuhc_prime = self.UniqueTask.get_Uniquefeatures(inputs, ssl_index_condition, Fl)

            # Get prediction from the UCFC fusion module
            out = self.UCFC.predict(Fuhc_prime, Fc, ssl_index)
            return out


if __name__ == "__main__":
    # First stage: Train CFE module.
    # In this stage, the first batch simulates data without tumors, labeled as 0 (INT in Fig. 2 of the original paper).
    # The second and third batches simulate data with tumors, labeled as 1 and 2 respectively.
    # Label 1 corresponds to IDH WildType, LGG, or 1p/19g Intact.
    # Label 2 corresponds to IDH Mutated, HGG, or lp/19g Co-deleted.

    # Generate random input tensors for the 1p19q, IDH, and LHG tasks
    input_1p19q = torch.randn(3, 3, 160, 192, 160)
    input_IDH = torch.randn(3, 3, 160, 192, 160)
    input_LHG = torch.randn(3, 3, 160, 192, 160)

    # Define labels for each task
    label_1p19q = torch.tensor([0, 1, 2], dtype=torch.long)
    label_IDH = torch.tensor([0, 1, 2], dtype=torch.long)
    label_LHG = torch.tensor([0, 1, 2], dtype=torch.long)

    # Combine labels into a list
    label_list = [label_1p19q, label_IDH, label_LHG]

    # Concatenate inputs from different tasks along the batch dimension
    input = torch.cat((input_1p19q, input_IDH, input_LHG), 0)

    # Define normal vs tumor labels for each task
    labels_NG_1p19q = torch.tensor([0, 1, 1], dtype=torch.long)
    labels_NG_IDH = torch.tensor([0, 1, 1], dtype=torch.long)
    labels_NG_LHG = torch.tensor([0, 1, 1], dtype=torch.long)

    # Combine normal vs tumor labels into a list
    labels_NG_list = [labels_NG_1p19q, labels_NG_IDH, labels_NG_LHG]

    # Initialize the CFE model with specified parameters
    model = CFE(in_channels=3, num_classes=2, Arcm=0.5, multi=3)

    # Perform a forward pass through the CFE model
    (logits_NG, labels_NG_all,
     out_cls_list_NG,
     out_cls_list, out_cls_labels) = model(input, labels_NG_list, label_list)

    # Print the outputs from the CFE model
    print(logits_NG.size())
    print(labels_NG_all.size())
    print(out_cls_list_NG[0].size())
    print(out_cls_list[0].size())
    print(out_cls_labels[0].size())

    # -------------------------------------------------------------------
    # Second stage: Train CMTLNet.
    # In this stage, all inputs contain tumors (IT in Fig. 2 of the original paper).
    # The labels correspond to 0 and 1.
    # Label 0 corresponds to IDH WildType, LGG, or 1p/19g Intact.
    # Label 1 corresponds to IDH Mutated, HGG, or lp/19g Co-deleted.

    # Generate random input tensors for the 1p19q, IDH, and LHG tasks
    input_1p19q = torch.randn(2, 3, 160, 192, 160)
    input_IDH = torch.randn(2, 3, 160, 192, 160)
    input_LHG = torch.randn(2, 3, 160, 192, 160)

    # Define labels for each task
    label_1p19q = torch.tensor([0, 1], dtype=torch.long)
    label_IDH = torch.tensor([0, 1], dtype=torch.long)
    label_LHG = torch.tensor([0, 1], dtype=torch.long)

    # Combine labels into a list
    label_list = [label_1p19q, label_IDH, label_LHG]

    # Concatenate inputs from different tasks along the batch dimension
    input = torch.cat((input_1p19q, input_IDH, input_LHG), 0)

    # Define condition tensor indicating the task for each input
    # Condition 0 for 1p19q prediction task, 1 for IDH prediction task, and 2 for LHG prediction task
    condition = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=input.device)

    # Initialize the CMTLNet model with specified parameters
    model = CMTLNet(in_channels=3, num_classes=2, Arcm=0.5, multi=3)

    # Perform a forward pass through the CMTLNet model
    out_cls_list_UFE, out_cls_labels_UFE, out_cls_list_UCFC, out_cls_labels_UCFC = model(input, label_list, condition)

    # Print the lengths of the classification output lists and label lists
    print(out_cls_list_UFE[0].size())
    print(out_cls_labels_UFE[0].size())
    print(out_cls_list_UCFC[0].size())
    print(out_cls_labels_UCFC[0].size())

    # -------------------------------------------------------------------
    # Final model inference:
    # For the 1p19q prediction task, set ssl_index to 0.
    # For the IDH prediction task, set ssl_index to 1.
    # For the LHG prediction task, set ssl_index to 2.

    # Perform prediction using the CMTLNet model for the IDH prediction task (ssl_index=1)
    logits = model.predict(input, ssl_index=1)

    # Print the size of the prediction logits
    print(logits.size())