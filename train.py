import warnings

import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

def train_one_batch(model, dataloader, device):
    """
    Check if model can teach.
    """
    print("Train one batch started")
    test_optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    pos_weight = torch.tensor([8.837], device=device)
    test_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    model.train()
    
    batch = next(iter(dataloader))
    
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
            
    for i in range(60):
        test_optimizer.zero_grad()

        outputs = model(**batch)
        
        loss = test_criterion(outputs["logits"], batch["label"])

        if i % 5 == 0:
            print("iteration ", i, "; loss: ", loss.item())

        loss.backward()
        test_optimizer.step()
        
    print('Train one batch finished')


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # sanity check: training of one batch
    if config.trainer.get("train_one_batch", False):
        train_one_batch(model, dataloader=dataloaders["train"], device=device)
        logger.info(".")
        sys.exit(0)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    
    config.lr_scheduler.step_size = 10 * len(dataloaders["train"]) # here replace
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
