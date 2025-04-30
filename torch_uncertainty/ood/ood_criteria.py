from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import torch
from torch import Tensor, nn
import numpy as np
from torch_uncertainty.metrics import MutualInformation, VariationRatio
from torch_uncertainty.ood.nets import *
from tqdm import tqdm
from omegaconf import OmegaConf


class OODCriterionInputType(Enum):
    """Enum representing the type of input expected by the OOD (Out-of-Distribution) criteria.

    Attributes:
        LOGIT (int): Represents that the input is in the form of logits (pre-softmax values).
        PROB (int): Represents that the input is in the form of probabilities (post-softmax values).
        ESTIMATOR_PROB (int): Represents that the input is in the form of estimated probabilities
            from an ensemble or other probabilistic model.
    """

    LOGIT = 1
    PROB = 2
    ESTIMATOR_PROB = 3
    DATASET = 4


class TUOODCriterion(ABC, nn.Module):
    input_type: OODCriterionInputType
    ensemble_only = False

    def __init__(self) -> None:
        """Abstract base class for Out-of-Distribution (OOD) criteria.

        This class defines a common interface for implementing various OOD detection
        criteria. Subclasses must implement the `forward` method.

        Attributes:
            input_type (OODCriterionInputType): Type of input expected by the criterion.
            ensemble_only (bool): Whether the criterion requires ensemble outputs.
        """
        super().__init__()
        self.setup_flag = False
        self.hyperparam_search_done = False
    
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass for the OOD criterion.

        Args:
            inputs (Tensor): The input tensor representing model outputs.

        Returns:
            Tensor: OOD score computed according to the criterion.
        """


class MaxLogitCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.LOGIT

    def __init__(self) -> None:
        """OOD criterion based on the maximum logit value.

        This criterion computes the negative of the highest logit value across
        the output dimensions. Lower maximum logits indicate greater uncertainty.

        Attributes:
            input_type (OODCriterionInputType): Expected input type is logits.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the negative of the maximum logit value.

        Args:
            inputs (Tensor): Tensor of logits with shape (batch_size, num_classes).

        Returns:
            Tensor: Negative of the maximum logit value for each sample.
        """
        return -inputs.mean(dim=1).max(dim=-1).values


class EnergyCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.LOGIT

    def __init__(self) -> None:
        r"""OOD criterion based on the energy function.

        This criterion computes the negative log-sum-exp of the logits.
        Higher energy values indicate greater uncertainty.

        .. math::
            E(\mathbf{z}) = -\log\left(\sum_{i=1}^{C} \exp(z_i)\right)

        where :math:`\mathbf{z} = [z_1, z_2, \dots, z_C]` is the logit vector.

        Attributes:
            input_type (OODCriterionInputType): Expected input type is logits.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the negative energy score.

        Args:
            inputs (Tensor): Tensor of logits with shape (batch_size, num_classes).

        Returns:
            Tensor: Negative energy score for each sample.
        """
        return -inputs.mean(dim=1).logsumexp(dim=-1)


class MaxSoftmaxCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.PROB

    def __init__(self) -> None:
        r"""OOD criterion based on maximum softmax probability.

        This criterion computes the negative of the highest softmax probability.
        Lower maximum probabilities indicate greater uncertainty.

        .. math::
            \text{score} = -\max_{i}(p_i)

        where :math:`\mathbf{p} = [p_1, p_2, \dots, p_C]` is the probability vector.

        Attributes:
            input_type (OODCriterionInputType): Expected input type is probabilities.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the negative of the maximum softmax probability.

        Args:
            inputs (Tensor): Tensor of probabilities with shape (batch_size, num_classes).

        Returns:
            Tensor: Negative of the highest softmax probability for each sample.
        """
        return -inputs.max(-1)[0]


class EntropyCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.ESTIMATOR_PROB

    def __init__(self) -> None:
        r"""OOD criterion based on entropy.

        This criterion computes the mean entropy of the predicted probability distribution.
        Higher entropy values indicate greater uncertainty.

        .. math::
            H(\mathbf{p}) = -\sum_{i=1}^{C} p_i \log(p_i)

        where :math:`\mathbf{p} = [p_1, p_2, \dots, p_C]` is the probability vector.

        Attributes:
            input_type (OODCriterionInputType): Expected input type is estimated probabilities.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the entropy of the predicted probability distribution.

        Args:
            inputs (Tensor): Tensor of estimated probabilities with shape (batch_size, num_classes).

        Returns:
            Tensor: Mean entropy value for each sample.
        """
        return torch.special.entr(inputs).sum(dim=-1).mean(dim=1)


class MutualInformationCriterion(TUOODCriterion):
    ensemble_only = True
    input_type = OODCriterionInputType.ESTIMATOR_PROB

    def __init__(self) -> None:
        r"""OOD criterion based on mutual information.

        This criterion computes the mutual information between ensemble predictions.
        Higher mutual information values indicate lower uncertainty.

        Given ensemble predictions :math:`\{\mathbf{p}^{(k)}\}_{k=1}^{K}`, the mutual information is computed as:

        .. math::
            I(y, \theta) = H\Big(\frac{1}{K}\sum_{k=1}^{K} \mathbf{p}^{(k)}\Big) - \frac{1}{K}\sum_{k=1}^{K} H(\mathbf{p}^{(k)})

        Attributes:
            ensemble_only (bool): Requires ensemble predictions.
            input_type (OODCriterionInputType): Expected input type is estimated probabilities.
        """
        super().__init__()
        self.mi_metric = MutualInformation(reduction="none")

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute mutual information from ensemble predictions.

        Args:
            inputs (Tensor): Tensor of ensemble probabilities with shape
                (ensemble_size, batch_size, num_classes).

        Returns:
            Tensor: Mutual information for each sample.
        """
        return self.mi_metric(inputs)


class VariationRatioCriterion(TUOODCriterion):
    ensemble_only = True
    input_type = OODCriterionInputType.ESTIMATOR_PROB

    def __init__(self) -> None:
        r"""OOD criterion based on variation ratio.

        This criterion computes the variation ratio from ensemble predictions.
        Higher variation ratio values indicate greater uncertainty.

        Given ensemble predictions where :math:`n_{\text{mode}}` is the count of the most frequently
        predicted class among :math:`K` predictions, the variation ratio is computed as:

        .. math::
            \text{VR} = 1 - \frac{n_{\text{mode}}}{K}

        Attributes:
            ensemble_only (bool): Requires ensemble predictions.
            input_type (OODCriterionInputType): Expected input type is estimated probabilities.
        """
        super().__init__()
        self.vr_metric = VariationRatio(reduction="none", probabilistic=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute variation ratio from ensemble predictions.

        Args:
            inputs (Tensor): Tensor of ensemble probabilities with shape
                (ensemble_size, batch_size, num_classes).

        Returns:
            Tensor: Variation ratio for each sample.
        """
        return self.vr_metric(inputs.transpose(0, 1))



class ScaleCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET

    def __init__(self,config) -> None:
        """OOD criterion based on the maximum logit value.

        This criterion computes the negative of the highest logit value across
        the output dimensions. Lower maximum logits indicate greater uncertainty.

        Attributes:
            input_type (OODCriterionInputType): Expected input type is logits.
        """
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = config.postprocessor.postprocessor_sweep

    def forward(self, net: nn.Module, data: Any) -> Tensor:
        net = ScaleNet(net)
        output = net.forward_threshold(data, self.percentile)
        #_, pred = torch.max(output, dim=1)
        energyconf = torch.logsumexp(output.data, dim=1)
        return -energyconf
    
    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]

    def get_hyperparam(self):
        return self.percentile




class ASHCriterion(TUOODCriterion):

    input_type = OODCriterionInputType.DATASET
    def __init__(self,config) -> None:
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = config.postprocessor.postprocessor_sweep


    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        net = ASHNet(net)
        output = net.forward_threshold(data, self.percentile)
        #_, pred = torch.max(output, dim=1)
        energyconf = torch.logsumexp(output.data, dim=1)
        return -energyconf

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]

    def get_hyperparam(self):
        return self.percentile




class ReactCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET
    def __init__(self,config) -> None:
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module,id_loader,ood_loaders):
        if not self.setup_flag:
            activation_log = []
            net = ReactNet(net)
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader['val'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch[0].cuda().float()
                    _, feature = net(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.setup_flag = True
        else:
            pass

        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        net = ReactNet(net)
        output = net.forward_threshold(data, self.threshold)
        energyconf = torch.logsumexp(output.data, dim=1)
        return -energyconf

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]
        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile, self.threshold))

    def get_hyperparam(self):
        return self.percentile





from statsmodels.distributions.empirical_distribution import ECDF

class AdaScaleCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET
    def __init__(self,config) -> None:
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.k1 = self.args.k1
        self.k2 = self.args.k2
        self.lmbda = self.args.lmbda
        self.o = self.args.o
        self.args_dict = config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module,id_loader,ood_loaders):
        net = AdaScaleANet(net)
        if not self.setup_flag:
            feature_log = []
            feature_perturbed_log = []
            feature_shift_log = []
            net.eval()
            self.feature_dim = net.backbone.feature_size
            with torch.no_grad():
                for batch in tqdm(id_loader['val'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch[0].cuda().float()
                    with torch.enable_grad():
                        data.requires_grad = True
                        output, feature = net(data, return_feature=True)
                        labels = output.detach().argmax(dim=1)
                        net.zero_grad()
                        score = output[torch.arange(len(labels)), labels]
                        score.backward(torch.ones_like(labels))
                        grad = data.grad.data.detach()
                    feature_log.append(feature.data.cpu())
                    data_perturbed = self.perturb(data, grad)
                    _, feature_perturbed = net(data_perturbed,
                                               return_feature=True)
                    feature_shift = abs(feature - feature_perturbed)
                    feature_shift_log.append(feature_shift.data.cpu())
                    feature_perturbed_log.append(feature_perturbed.data.cpu())
            all_features = torch.cat(feature_log, axis=0)
            all_perturbed = torch.cat(feature_perturbed_log, axis=0)
            all_shifts = torch.cat(feature_shift_log, axis=0)

            total_samples = all_features.size(0)
            num_samples = self.args.num_samples if hasattr(
                self.args, 'num_samples') else total_samples
            indices = torch.randperm(total_samples)[:num_samples]

            self.feature_log = all_features[indices]
            self.feature_perturbed_log = all_perturbed[indices]
            self.feature_shift_log = all_shifts[indices]
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def get_percentile(self, feature, feature_perturbed, feature_shift):
        topk_indices = torch.topk(feature, dim=1, k=self.k1_)[1]
        topk_feature_perturbed = torch.gather(
            torch.relu(feature_perturbed), 1,
            topk_indices)  # correction term C_o
        topk_indices = torch.topk(feature, dim=1, k=self.k2_)[1]
        topk_feature_shift = torch.gather(feature_shift, 1, topk_indices)  # Q
        topk_norm = topk_feature_perturbed.sum(
            dim=1) + self.lmbda * topk_feature_shift.sum(dim=1)  # Q^{\prime}
        percent = 1 - self.ecdf(topk_norm.cpu())
        percentile = self.min_percentile + percent * (self.max_percentile -
                                                      self.min_percentile)
        return torch.from_numpy(percentile)

    @torch.no_grad()
    def forward(self, net: nn.Module, data):
        net = AdaScaleANet(net)
        with torch.enable_grad():
            data.requires_grad = True
            output, feature = net(data, return_feature=True)
            labels = output.detach().argmax(dim=1)
            net.zero_grad()
            score = output[torch.arange(len(labels)), labels]
            score.backward(torch.ones_like(labels))
            grad = data.grad.data.detach()
            data.requires_grad = False
        data_perturbed = self.perturb(data, grad)
        _, feature_perturbed = net(data_perturbed, return_feature=True)
        feature_shift = abs(feature - feature_perturbed)
        percentile = self.get_percentile(feature, feature_perturbed,
                                         feature_shift)
        output = net.forward_threshold(feature, percentile)
        conf = torch.logsumexp(output, dim=1)
        return -conf

    @torch.no_grad()
    def perturb(self, data, grad):
        batch_size, channels, height, width = data.shape
        n_pixels = int(channels * height * width * self.o)
        abs_grad = abs(grad).view(batch_size, channels * height * width)
        _, topk_indices = torch.topk(abs_grad, n_pixels, dim=1, largest=False)
        mask = torch.zeros_like(abs_grad, dtype=torch.uint8)
        mask.scatter_(1, topk_indices, 1)
        mask = mask.view(batch_size, channels, height, width)
        data_ood = data + grad.sign() * mask * 0.5
        return data_ood

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]
        self.min_percentile, self.max_percentile = self.percentile[
            0], self.percentile[1]
        self.k1 = hyperparam[1]
        self.k2 = hyperparam[2]
        self.lmbda = hyperparam[3]
        self.o = hyperparam[4]
        self.k1_ = int(self.feature_dim * self.k1 / 100)
        self.k2_ = int(self.feature_dim * self.k2 / 100)
        topk_indices = torch.topk(self.feature_log, k=self.k1_, dim=1)[1]
        topk_feature_perturbed = torch.gather(
            torch.relu(self.feature_perturbed_log), 1, topk_indices)
        topk_indices = torch.topk(self.feature_log, k=self.k2_, dim=1)[1]
        topk_feature_shift_log = torch.gather(self.feature_shift_log, 1,
                                              topk_indices)
        sum_log = topk_feature_perturbed.sum(
            dim=1) + self.lmbda * topk_feature_shift_log.sum(dim=1)
        self.ecdf = ECDF(sum_log)

    def get_hyperparam(self):
        return [self.percentile, self.k1, self.k2, self.lmbda, self.o]



from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance


class VIMCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET
    def __init__(self,config) -> None:
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.args_dict = config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim

    def setup(self, net: nn.Module,id_loader,ood_loaders):
        if not self.setup_flag:
            net.eval()

            with torch.no_grad():
                self.w, self.b = net.get_fc()
                print('Extracting id training feature')
                feature_id_train = []
                for batch in tqdm(id_loader['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch[0].cuda().float()
                    _, feature = net(data, return_feature=True)
                    feature_id_train.append(feature.cpu().numpy())
                feature_id_train = np.concatenate(feature_id_train, axis=0)
                logit_id_train = feature_id_train @ self.w.T + self.b

            self.u = -np.matmul(pinv(self.w), self.b)
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(feature_id_train - self.u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            self.NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)

            vlogit_id_train = norm(np.matmul(feature_id_train - self.u,
                                             self.NS),
                                   axis=-1)
            self.alpha = logit_id_train.max(
                axis=-1).mean() / vlogit_id_train.mean()
            print(f'{self.alpha=:.4f}')

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        _, feature_ood = net.forward(data, return_feature=True)
        feature_ood = feature_ood.cpu()
        logit_ood = feature_ood @ self.w.T + self.b
        energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood.numpy() - self.u, self.NS),
                          axis=-1) * self.alpha
        score_ood = -vlogit_ood + energy_ood
        return -torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
    



class ODINCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET
    def __init__(self,config) -> None:
        super().__init__()
        self.args = config.postprocessor.postprocessor_args

        self.temperature = 1
        self.noise = 0.0014
        try:
            self.input_std = [0.2470, 0.2435, 0.2616]   # // to chnage
        except KeyError:
            self.input_std = [0.5, 0.5, 0.5]
        self.args_dict = config.postprocessor.postprocessor_sweep

    def forward(self, net: nn.Module, data: Any):
        data.requires_grad = True
        output = net(data)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(output, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
        gradient[:, 1] = (gradient[:, 1]) / self.input_std[1]
        gradient[:, 2] = (gradient[:, 2]) / self.input_std[2]

        # Adding small perturbations to images
        tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)
        output = net(tempInputs)
        output = output / self.temperature

        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        conf, _ = nnOutput.max(dim=1)

        return -conf

    def set_hyperparam(self, hyperparam: list):
        self.temperature = hyperparam[0]
        self.noise = hyperparam[1]

    def get_hyperparam(self):
        return [self.temperature, self.noise]






import faiss

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

class KNNCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET
    def __init__(self,config) -> None:
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.activation_log = None
        self.args_dict = config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch[0].cuda().float()
                    _, feature = net(data, return_feature=True)
                    activation_log.append(
                        normalizer(feature.data.cpu().numpy()))

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.index = faiss.IndexFlatL2(feature.shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        _,feature = net(data, return_feature=True)
        feature_normed = normalizer(feature.data.cpu().numpy())
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        return -torch.from_numpy(kth_dist)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K




class GENCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET
    def __init__(self,config) -> None:
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.gamma = self.args.gamma
        self.M = self.args.M
        self.args_dict = config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf = self.generalized_entropy(score, self.gamma, self.M)
        return -conf

    def set_hyperparam(self, hyperparam: list):
        self.gamma = hyperparam[0]
        self.M = hyperparam[1]

    def get_hyperparam(self):
        return [self.gamma, self.M]

    def generalized_entropy(self, softmax_id_val, gamma=0.1, M=100):
        probs = softmax_id_val
        probs_sorted = torch.sort(probs, dim=1)[0][:, -M:]
        scores = torch.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma),
                           dim=1)

        return -scores






import faiss
from scipy.special import logsumexp
from copy import deepcopy

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


def knn_score(bankfeas, queryfeas, k=100, min=False):

    bankfeas = deepcopy(np.array(bankfeas))
    queryfeas = deepcopy(np.array(queryfeas))

    index = faiss.IndexFlatIP(bankfeas.shape[-1])
    index.add(bankfeas)
    D, _ = index.search(queryfeas, k)
    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))
    return scores


class NNGuideCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET
    def __init__(self,config) -> None:
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.alpha = self.args.alpha
        self.activation_log = None
        self.args_dict = config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()
            bank_feas = []
            bank_logits = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch[0].cuda().float()

                    logit, feature = net(data, return_feature=True)
                    bank_feas.append(normalizer(feature.data.cpu().numpy()))
                    bank_logits.append(logit.data.cpu().numpy())
                    if len(bank_feas
                           ) * id_loader_dict['train'].batch_size > int(
                               len(id_loader_dict['train'].dataset) *
                               self.alpha):
                        break

            bank_feas = np.concatenate(bank_feas, axis=0)
            bank_confs = logsumexp(np.concatenate(bank_logits, axis=0),
                                   axis=-1)
            self.bank_guide = bank_feas * bank_confs[:, None]

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        logit, feature = net(data, return_feature=True)
        feas_norm = normalizer(feature.data.cpu().numpy())
        energy = logsumexp(logit.data.cpu().numpy(), axis=-1)

        conf = knn_score(self.bank_guide, feas_norm, k=self.K)
        score = conf * energy

        return -torch.from_numpy(score)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]
        self.alpha = hyperparam[1]

    def get_hyperparam(self):
        return [self.K, self.alpha]








































def get_ood_criterion(ood_criterion):
    """Get an OOD criterion instance based on a string identifier or class type.

    Args:
        ood_criterion (str or type): A string identifier for a predefined OOD criterion
            or a subclass of `TUOODCriterion`.

    Returns:
        TUOODCriterion: An instance of the requested OOD criterion.

    Raises:
        ValueError: If the input string or class type is invalid.
    """
    if isinstance(ood_criterion, str):
        config = OmegaConf.load(f"/home/firas/Bureau/torch-uncertainty_ood/torch_uncertainty/ood/configs/{ood_criterion}.yml")
        if ood_criterion == "logit":
            return MaxLogitCriterion()
        if ood_criterion == "energy":
            return EnergyCriterion()
        if ood_criterion == "msp":
            return MaxSoftmaxCriterion()
        if ood_criterion == "entropy":
            return EntropyCriterion()
        if ood_criterion == "mutual_information":
            return MutualInformationCriterion()
        if ood_criterion == "variation_ratio":
            return VariationRatioCriterion()        
        if ood_criterion == "scale":
            return ScaleCriterion(config)
        if ood_criterion == "ash":
            return ASHCriterion(config)        
        if ood_criterion == "react":
            return ReactCriterion(config)        
        if ood_criterion == "adascale_a":
            return AdaScaleCriterion(config)        
        if ood_criterion == "vim":
            return VIMCriterion(config)        
        if ood_criterion == "odin":
            return ODINCriterion(config)        
        if ood_criterion == "knn":
            return KNNCriterion(config)        
        if ood_criterion == "gen":
            return GENCriterion(config)        
        if ood_criterion == "nnguide":
            return NNGuideCriterion(config)        
        if ood_criterion == "react":
            return ReactCriterion()
        raise ValueError(
            "The OOD criterion must be one of 'msp', 'logit', 'energy', 'entropy',"
            f" 'mutual_information' or 'variation_ratio'. Got {ood_criterion}."
        )
    if isinstance(ood_criterion, type) and issubclass(ood_criterion, TUOODCriterion):
        return ood_criterion()
    raise ValueError(
        f"The OOD criterion should be a string or a subclass of TUOODCriterion. Got {type(ood_criterion)}."
    )
