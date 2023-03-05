import torch 
import yaml
import logging 
from typing import Union, Dict, Generator
from pathlib import Path 
import random 
import os 
import numpy as np 
import heapq
import time 
from functools import partial 
from model import build_model
from data import load_data, OurDataset
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

def load_config(path: Union[Path,str]="configs/transformer.yaml") -> Dict:
    if isinstance(path, str):
        path = Path(path)
    with path.open("r", encoding="utf-8") as yamlfile:
        cfg = yaml.safe_load(yamlfile)
    return cfg

def make_logger(model_dir):
    logger = logging.getLogger("")
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info("Hello! This is Tong Ye's Transformer!")

def set_seed(seed:int) -> None:
    """
    Set the random seed for modules torch, numpy and random.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

class TrainManager(object):
    """
    Manage the whole training loop and validation.
    """
    def __init__(self, model, cfg) -> None:
        # File system 
        train_cfg = cfg["training"]
        self.model_dir = Path(train_cfg["model_dir"])
        assert self.model_dir.is_dir(), "model_dir:{} not found!".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=(self.model_dir/"tensorboard_log").as_posix())

        # logger information
        self.logging_frequency = train_cfg.get("logging_frequence", 100)
        self.valid_frequence = train_cfg.get("validation_frequence", 1000)
        self.log_valid_samples = train_cfg.get("log_valid_samples", [0, 1, 2])

        # CPU and GPU
        use_cuda = train_cfg["use_cuda"] and torch.cuda.is_available()
        self.n_gpu = torch.cuda.device_count() if use_cuda else 0  
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logger.info("*"*20 + "{} GPUs are used.".format(self.n_gpu) + "*"*20)
        num_workers = train_cfg.get("num_workers", 0)
        if num_workers > 0:
            self.num_workers = min(os.cpu_count(), num_workers)
        
        # data & batch & epochs
        self.epochs = train_cfg.get("epochs", 100)
        self.shuffle = train_cfg.get("shuffle", True)
        self.max_updates = train_cfg.get("max_updates", np.inf)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.seed = train_cfg.get('random_seed', 980820)

        # learning rate and optimization & schedular
        self.learning_rate_min = train_cfg.get("learning_rate_min", 1.0e-8)
        self.clip_grad_function = self.build_gradient_clipper(train_cfg)
        self.optimizer = self.build_optimizer(train_cfg, parameters=self.model.parameters())
        self.scheduler, self.scheduler_step_at = self.build_scheduler(train_cfg, optimizer=self.optimizer)

        # early stop
        self.early_stop_metric = train_cfg.get("early_stop_metric", None)
        if self.early_stop_metric in ["ppl", "loss"]:
            self.minimize_metric = True # lower is better
        elif self.early_stop_metric in ["acc", "bleu"]:
            self.minimize_metric = False # higher is better

        # model 
        self.model = model
        if self.device.type == "cuda":
            self.model.to(self.device)
        
        self.load_model = train_cfg.get("load_model", None)
        if self.load_model is not None:
            self.init_from_checkpoint(self.load_model,
                reset_best_ckpt=train_cfg.get("reset_best_ckpt", False),
                reset_scheduler=train_cfg.get("reset_scheduler", False),
                reset_optimizer=train_cfg.get("reset_optimizer", False),
                reset_iter_state=train_cfg.get("reset_iter_state", False))
        
        # initialize training process statistics
        self.stats = self.TrainStatistics(
            steps=0, is_min_lr=False, is_max_update=False,
            total_tokens=0, best_ckpt_iter=0, minimize_metric=self.minimize_metric,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
        )

    def build_gradient_clipper(self, train_cfg:dict):
        """
        Define the function for gradient clipping as specified in configuration.
        Current Options:
            clip_grad_val: clip the gradient if they exceed value.
                torch.nn.utils.clip_grad_value_
            clip_grad_norm: clip the gradients if they norm exceeds this value
                torch.nn.utils.clip_grad_norm_
        """
        clip_grad_function = None 
        if "clip_grad_val" in train_cfg.keys():
            clip_grad_function = partial(torch.nn.utils.clip_grad_value_, clip_value=train_cfg["clip_grad_val"])
        elif "clip_grad_norm" in train_cfg.keys():
            clip_grad_function = partial(torch.nn.utils.clip_grad_norm_, max_norm=train_cfg["clip_grad_norm"])
        return clip_grad_function

    def build_optimizer(self, train_cfg:dict, parameters: Generator) -> torch.optim.Optimizer:
        """
        Create an optimizer for the given parameters as specified in config.
        """
        optimizer_name = train_cfg.get("optimizer", "adam").lower()
        kwargs = {"lr": train_cfg.get("learning_rate", 3e-4), "weight_decay":train_cfg.get("weight_decay", 0)}
        if optimizer_name == "adam":
            # kwargs: lr, weight_decay; betas
            optimizer = torch.optim.Adam(params=parameters, **kwargs)
        elif optimizer_name == "sgd":
            # kwargs: lr, momentum; dampening; weight_decay; nestrov
            optimizer = torch.optim.SGD(parameters, **kwargs)
        else:
            assert False, "Invalid optimizer."

        logger.info("%s(%s)", optimizer.__class__.__name__, ", ".join([f"{key}={value}" for key,value in kwargs.items()]))
        return optimizer
    
    def build_schedular(self, train_cfg:dict, optimizer: torch.optim.Optimizer):
        """
        Create a learning rate scheduler if specified in train config and determine
        when a scheduler step should be executed.
        return:
            - scheduler: scheduler object
            - schedulers_step_at: "validation", "epoch", "step", or "none"
        """
        scheduler, scheduler_step_at = None, None 
        scheduler_name = train_cfg.get("scheduling", None)
        assert scheduler_name in ["StepLR", "ExponentialLR", "ReduceLROnPlateau"], "Invalid scheduling."
        
        if scheduler_name == "ReduceLROnPlateau":
            mode = train_cfg.get("mode", "max")
            factor = train_cfg.get("factor", 0.5)
            patience = train_cfg.get("patience", 5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, 
                        patience=patience, threshold=0.0001, threshold_mode='abs', eps=1e-8)
            scheduler_step_at = "validation"
        elif scheduler_name == "StepLR":
            step_size = train_cfg.get("step_size", 5)
            gamma = train_cfg.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            scheduler_step_at = "epoch"  
        elif scheduler_name == "ExponentialLR":
            gamma  = train_cfg.get("gamma", 0.98)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            scheduler_step_at = "epoch"
        else:
            assert False, "Invalid scheduling."
        
        logger.info("Scheduler = %s", scheduler.__class__.__name__)
        return scheduler, scheduler_step_at
        
    def train_and_validate(self, train_data:OurDataset, valid_data:OurDataset) -> None:
        """
        Train the model and validate it from time to time on the validation set.
        """
        self.train_iter = make_data_iter(dataset=train_data, sampler_seed=self.seed, shuffle=self.shuffle, batch_type=self.batch_type,
                                            batch_size=self.batch_size, num_workers=self.num_workers)

        if self.train_iter_state is not None:
            self.train_iter.batch_sampler.sampler.generator.set_state(self.train_iter_state.cpu())
        
        # train and validate main loop
        logger.info("Train stats:\n"
                    "\tdevice: %s\n"
                    "\tn_gpu: %d\n"
                    "\tbatch_size: %d",
                    self.device.type, self.n_gpu, self.batch_size)
        try:
            for epoch_no in range(self.epochs):
                logger.info("Epoch %d", epoch_no + 1)
                    
                self.model.train()
                self.model.zero_grad()
                
                # Statistic for each epoch.
                start_time = time.time()
                start_tokens = self.stats.total_tokens
                epoch_loss = 0

                for batch_data in self.train_iter:
                    batch_data.move2cuda(self.device)
                    normalized_batch_loss = self.train_step(batch_data)
                    
                    # reset gradients
                    self.optimizer.zero_grad()

                    normalized_batch_loss.backward()

                    # clip gradients (in-place)
                    if self.clip_grad_fun is not None:
                        self.clip_grad_fun(parameters=self.model.parameters())
                    
                    # make gradient step
                    self.optimizer.step()

                    # accumulate loss
                    epoch_loss += normalized_batch_loss.item()

                    # increment token counter
                    self.stats.total_tokens += batch_data.ntokens
                    
                    # increment step counter
                    self.stats.steps += 1
                    if self.stats.steps >= self.max_updates:
                        self.stats.is_max_update == True
                    
                    # check current leraning rate(lr)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if current_lr < self.learning_rate_min:
                        self.stats.is_min_lr = True 

                    # log learning process and write tensorboard
                    if self.stats.steps % self.logging_freq == 0:
                        elapse_time = time.time() - start_time
                        elapse_token_num = self.stats.total_tokens - start_tokens

                        self.tb_writer.add_scalar(tag="Train/batch_loss", scalar_value=normalized_batch_loss, global_step=self.stats.steps)
                        self.tb_writer.add_scalar(tag="Train/learning_rate", scalar_value=current_lr, global_step=self.stats.steps)

                        logger.info("Epoch %3d, Step: %7d, Batch Loss: %12.6f, Lr: %.6f, Tokens per sec: %6.0f",
                        epoch_no + 1, self.stats.steps, normalized_batch_loss, self.optimizer.param_groups[0]["lr"],
                        elapse_token_num / elapse_time)

                        start_time = time.time()
                        start_tokens = self.stats.total_tokens
                    
                    # decay learning_rate(lr)
                    if self.scheduler_step_at == "step":
                        self.scheduler.step(self.stats.steps)

                logger.info("Epoch %3d: total training loss %.2f", epoch_no + 1, epoch_loss)

                if self.scheduler_step_at == "epoch":
                    self.scheduler.step()

                # validate on the entire dev dataset
                if (epoch_no + 1) % self.validation_freq == 0:
                    valid_duration_time = self.validate(valid_data)
                    logger.info("Validation time = {}s.".format(valid_duration_time))

                # check after a number of whole epoch.
                if self.stats.is_min_lr or self.stats.is_max_update:
                    log_string = (f"minimum learning rate {self.learning_rate_min}" if self.stats.is_min_lr else 
                                    f"maximun number of updates(steps) {self.max_updates}")
                    logger.info("Training end since %s was reached!", log_string)
                    break 
            else: # normal ended after training.
                logger.info("Training ended after %3d epoches!", epoch_no + 1)

            logger.info("Best Validation result (greedy) at step %8d: %6.2f %s.",
                        self.stats.best_ckpt_iter, self.stats.best_ckpt_score, self.early_stopping_metric)
                    
        except KeyboardInterrupt:
            self.save_model_checkpoint(False, float("nan"))

        self.tb_writer.close()
        return None 
    
    class TrainStatistics:
        def __init__(self, steps:int=0, is_min_lr:bool=False,
                        is_max_update:bool=False, total_tokens:int=0,
                        best_ckpt_iter:int=0, best_ckpt_score: float=np.inf,
                        minimize_metric: bool=True) -> None:
            self.steps = steps 
            self.is_min_lr = is_min_lr
            self.is_max_update = is_max_update
            self.total_tokens = total_tokens
            self.best_ckpt_iter = best_ckpt_iter
            self.best_ckpt_score = best_ckpt_score
            self.minimize_metric = minimize_metric
        
        def is_best(self, score) -> bool:
            if self.minimize_metric:
                is_best = score < self.best_ckpt_score
            else: 
                is_best = score > self.best_ckpt_score
            return is_best
        
        def is_better(self, score: float, heap_queue: list):
            assert len(heap_queue) > 0
            if self.minimize_metric:
                is_better = score < heapq.nlargest(1, heap_queue)[0][0]
            else:
                is_better = score > heapq.nsmallest(1, heap_queue)[0][0]
            return is_better

def train(cfg_file: str) -> None:
    """
    Main Training function.
    """
    cfg = load_config(Path(cfg_file))

    model_dir = None # FIXME
    make_logger(model_dir)

    set_seed(seed=int(cfg["training"].get("random_seed", 820)))

    # load data
    train_data, valid_data, test_data, vocab_info = load_data(data_cfg=cfg["data"])
    
    # build model
    model = build_model(model_cfg=cfg["model"], vocab_info=vocab_info)

    # training management
    trainer = TrainManager(model=model, cfg=cfg)

    # train model
    trainer.train_and_validata(train_data=train_data, valid_data=valid_data)

if __name__ == "__main__":
    cfg_file = "test.yaml"
    train(cfg_file)