# FELRec

PyTorch implementation of the FELRec paper.

### ABSTRACT

*Recommender systems suffer from the cold-start problem whenever a new user joins the platform or a new item is added to the catalog. To address item cold-start, we propose to replace the embedding layer in sequential recommenders with a dynamic storage that has no learnable weights and can keep an arbitrary number of representations. In this paper, we present `FELRec`, a large embedding network that refines the existing representations of users and items in a recursive manner, as new information becomes available. In contrast to similar approaches, our model represents new users and items without side information or time-consuming fine-tuning. During item cold-start, our method outperforms similar method by 29.50%-47.45%. Further, our proposed model generalizes well to previously unseen datasets.*

### EXPERIMENTS

You can run our experiments with the short commands below. For a full overview of the parameters, refer to the script files. 

Training scripts automatically download the original datasets. 

To train `FELRec-Q` (contrastive learning), run:

```shell
python -m trainer.FELRec \
  --job-dir jobs/felrec-q \
  --dataset data/ml-25m \
  --use-queue
```

You can train `FELRec-P` (similarity loss) if you remove the `use-queue` flag:

```shell
python -m trainer.FELRec \
  --job-dir jobs/felrec-p \
  --dataset data/ml-25m
```

To train `BPR-MF`, run:

```shell
python -m trainer.MF \
  --job-dir jobs/mf \
  --dataset data/ml-25m
```

To train `GRU4Rec`, run:

```shell
python -m trainer.GRU4Rec \
  --job-dir jobs/gru4rec \
  --dataset data/ml-25m
```

You can train `GRU4Rec-BPTT`, the BPTT version of `GRU4Rec`, if you specify the length of session:

```shell
python -m trainer.GRU4Rec \
  --job-dir jobs/gru4rec-bptt \
  --dataset data/ml-25m \
  --session 64
```

To train `SASRec`, run:

```shell
python -m trainer.SASRec \
  --job-dir jobs/sasrec \
  --dataset data/ml-25m
```

To train `JODIE`, run:

```shell
python -m trainer.JODIE \
  --job-dir jobs/jodie \
  --dataset data/ml-25m
```