# Learning Whom to Learn From

Code and experiments for **Learning Whom to Learn From**, the paper that introduces **pFedVClusters**, submitted to CVPR 2027.

The methodology, codebase and experiments are motivated by [PromptFL](https://arxiv.org/abs/2208.11625), [FedPGP](https://arxiv.org/abs/2405.09771), [FedMGP](https://arxiv.org/abs/2511.00480), [FedOTP](https://arxiv.org/abs/2403.00041), [pFedMoAP](https://arxiv.org/abs/2410.10114) and [pFedMMA](https://arxiv.org/abs/2507.05394). See [Acknowledgements](#acknowledgements).

## Abstract

Personalized federated prompt learning adapts pretrained vision–language models to heterogeneous client data, but effective collaboration requires more than identifying similar clients: a peer update must benefit the receiving client. Existing approaches typically aggregate or mix prompts without explicitly assessing their local utility and are primarily evaluated under fixed client distributions. We propose pFedVClusters, a personalized federated visual prompt learning framework that separates client discovery from selective knowledge transfer. Each client learns visual prompt tokens for a frozen CLIP vision transformer and shares a compact signature combining prompt adaptation directions with visual-token statistics. The server discovers compatible client groups using density-based clustering, without requiring domain annotations or a predefined number of clusters. Within these groups, each client assesses candidate peer updates using compatibility and measured local utility, incorporating accepted updates as bounded residuals while retaining its personalized prompt. We evaluate pFedVClusters on nine image classification datasets covering label and feature shifts, and on CIFAR-10 and CIFAR-100 under temporal label shift. The framework achieves 39.45% accuracy on DomainNet and 40.56% on Office–Caltech10, outperforming the strongest baseline by 8.39 and 20.06 percentage points, respectively. It also leads on four of five few-shot datasets and achieves average accuracies of 80.62% and 64.60% on CIFAR-10 and CIFAR-100 under temporal label shift. These results demonstrate the effectiveness of combining visual client discovery with utility-guided peer transfer for personalized federated adaptation under heterogeneous and evolving data distributions.

## Datasets

Download all datasets into the `dataset/` folder.

| Dataset | Heterogeneity studied | How to get it |
|---|---|---|
| [Caltech101](https://data.caltech.edu/records/mzrjq-6wc02), [OxfordPets](https://www.robots.ox.ac.uk/~vgg/data/pets/), [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | Label shift (pathological) | Follow [dataset.md](pFedMoAP/dataset/dataset.md) |
| [CIFAR-10 / CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) | Label shift (Dirichlet) and temporal label shift | Downloaded automatically on the first run |
| [DomainNet](https://ai.bu.edu/M3SDA/) | Feature and label shift | [Download here](https://github.com/med-air/FedBN) |
| Office-Caltech10 | Feature and label shift | [Download here](https://github.com/med-air/FedBN) |

## Running experiments

All experiments are run from `federated_main.py`.

**Default experiment (Food101):**

```bash
python federated_main.py
```

**Other datasets:** change the dataset name inside `federated_main.py`, or pass it as a flag:

```bash
python federated_main.py --dataset <dataset_name>
```

## Acknowledgements

The methodology, codebase and experiments are motivated by the following works. We thank their authors for releasing their papers and code.

| Method | Venue | Paper | Code |
|---|---|---|---|
| PromptFL | IEEE TMC 2024 | [arXiv](https://arxiv.org/abs/2208.11625) | [GitHub](https://github.com/PEILab-Federated-Learning/PromptFL) |
| FedPGP | ICML 2024 | [arXiv](https://arxiv.org/abs/2405.09771) | [GitHub](https://github.com/TianyuCuiOvO/FedPGP) |
| FedMGP | NeurIPS 2025 | [arXiv](https://arxiv.org/abs/2511.00480) | [GitHub](https://github.com/weihao-bo/FedMGP) |
| FedOTP | CVPR 2024 | [arXiv](https://arxiv.org/abs/2403.00041) | [GitHub](https://github.com/HongxiaLee/FedOTP) |
| pFedMoAP | ICLR 2025 | [arXiv](https://arxiv.org/abs/2410.10114) | [GitHub](https://github.com/ljaiverson/pFedMoAP) |
| pFedMMA | ICLR 2026 | [arXiv](https://arxiv.org/abs/2507.05394) | [GitHub](https://github.com/sajjad-ucsb/pFedMMA) |

pFedVClusters is built on the pretrained [CLIP](https://github.com/openai/CLIP) model.
