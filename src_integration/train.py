import torch 
import yaml
import logging 
from typing import Union, Dict, Generator, List
from pathlib import Path 
import random 
import os 
import numpy as np 
import heapq
import math 
import shutil
import time 
from functools import partial 
from model import build_model, Model
from data import load_data, OurDataset, make_data_loader, OurData, PAD_ID
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

def load_config(path: Union[Path,str]="test.yaml") -> Dict:
    if isinstance(path, str):
        path = Path(path)
    with path.open("r", encoding="utf-8") as yamlfile:
        cfg = yaml.safe_load(yamlfile)
    return cfg

def make_logger(model_dir:Path):
    logger = logging.getLogger("")
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    
    fh = logging.FileHandler(model_dir/"test.log")
    fh.setLevel(level=logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

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

def make_model_dir(model_dir:Path, overwrite:bool=False) -> Path:
    model_dir = model_dir.absolute()
    if model_dir.is_dir():
        if not overwrite: # don't allow overwite the directory
            raise FileExistsError(f"<model_dir> {model_dir} exists and don't allow overwrite.")
        else:
            shutil.rmtree(model_dir)
    model_dir.mkdir()
    return model_dir

class TrainManager(object):
    """
    Manage the whole training loop and validation.
    """
    def __init__(self, model:Model, cfg:dict):
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
        logger.info("*"*20 + "{} num_workers are used.".format(self.num_workers) + "*"*20)
        
        # data & batch & epochs
        self.epochs = train_cfg.get("epochs", 100)
        self.shuffle = train_cfg.get("shuffle", True)
        self.max_updates = train_cfg.get("max_updates", np.inf)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.seed = train_cfg.get('random_seed', 980820)

        # model 
        self.model = model
        if self.device.type == "cuda":
            self.model.to(self.device)
        
        self.load_model = train_cfg.get("load_model", False)
        if self.load_model is True:
            self.init_from_checkpoint(self.load_model,
                reset_best_ckpt=train_cfg.get("reset_best_ckpt", False),
                reset_scheduler=train_cfg.get("reset_scheduler", False),
                reset_optimizer=train_cfg.get("reset_optimizer", False),
                reset_iter_state=train_cfg.get("reset_iter_state", False))

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

        # save / delete checkpoints
        self.num_ckpts_keep = train_cfg.get("num_ckpts_keep", 3)
        self.ckpt_queue = [] # heapq queue List[Tuple[float, Path]]
        
        # initialize training process statistics
        self.train_stats = self.TrainStatistics(
            steps=0, is_min_lr=False, is_max_update=False,
            total_tokens=0, best_ckpt_step=0, minimize_metric=self.minimize_metric,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
        )
        self.train_loader = None

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
        kwargs = {"lr": train_cfg.get("learning_rate", 3.0e-4), "weight_decay":train_cfg.get("weight_decay", 0),
                  "betas": train_cfg.get("adam_betas", (0.9, 0.999)), "eps":train_cfg.get("eps", 1e-8)}
        if optimizer_name == "adam":
            # kwargs: lr, weight_decay; betas
            optimizer = torch.optim.Adam(params=parameters, **kwargs)
        elif optimizer_name == "sgd":
            logger.warning("Use sgd optimizer, please double check.")
            # kwargs: lr, momentum; dampening; weight_decay; nestrov
            optimizer = torch.optim.SGD(parameters, **kwargs)
        else:
            assert False, "Invalid optimizer."

        logger.info("%s(%s)", optimizer.__class__.__name__, ", ".join([f"{key}={value}" for key,value in kwargs.items()]))
        return optimizer
    
    def build_scheduler(self, train_cfg:dict, optimizer: torch.optim.Optimizer):
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
        
    def train_and_validate(self, train_dataset:OurDataset, valid_dataset:OurDataset):
        """
        Train the model and validate it from time to time on the validation set.
        """
        self.train_loader = make_data_loader(dataset=train_dataset, sampler_seed=self.seed, shuffle=self.shuffle,
                                    batch_size=self.batch_size, num_workers=self.num_workers, mode="train")
        
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
                start_tokens = self.train_stats.total_tokens
                epoch_loss = 0

                for batch_data in self.train_loader:
                    batch_data.to(self.device)
                    normalized_batch_loss, ntokens = self.train_step(batch_data)
                    
                    # reset gradients
                    self.optimizer.zero_grad()

                    normalized_batch_loss.backward()

                    # clip gradients (in-place)
                    if self.clip_grad_function is not None:
                        self.clip_grad_function(parameters=self.model.parameters())
                    
                    # make gradient step
                    self.optimizer.step()

                    # accumulate loss
                    epoch_loss += normalized_batch_loss.item()

                    # increment token counter
                    self.train_stats.total_tokens += ntokens
                    
                    # increment step counter
                    self.train_stats.steps += 1
                    if self.train_stats.steps >= self.max_updates:
                        self.train_stats.is_max_update = True
                    
                    # check current leraning rate(lr)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if current_lr < self.learning_rate_min:
                        self.train_stats.is_min_lr = True 

                    # log learning process and write tensorboard
                    if self.train_stats.steps % self.logging_frequency == 0:
                        elapse_time = time.time() - start_time
                        elapse_token_num = self.train_stats.total_tokens - start_tokens

                        self.tb_writer.add_scalar(tag="Train/batch_loss", scalar_value=normalized_batch_loss, global_step=self.train_stats.steps)
                        self.tb_writer.add_scalar(tag="Train/learning_rate", scalar_value=current_lr, global_step=self.train_stats.steps)

                        logger.info("Epoch %3d, Step: %7d, Batch Loss: %12.6f, Lr: %.6f, Tokens per sec: %6.0f",
                        epoch_no + 1, self.train_stats.steps, normalized_batch_loss, self.optimizer.param_groups[0]["lr"],
                        elapse_token_num / elapse_time)

                        start_time = time.time()
                        start_tokens = self.train_stats.total_tokens
                    
                    # decay learning_rate(lr)
                    if self.scheduler_step_at == "step":
                        self.scheduler.step(self.train_stats.steps)

                logger.info("Epoch %3d: total training loss %.2f", epoch_no + 1, epoch_loss)

                if self.scheduler_step_at == "epoch":
                    self.scheduler.step()

                # validate on the entire dev dataset
                if (epoch_no + 1) % self.valid_frequence == 0:
                    valid_duration_time = self.validate(valid_dataset)
                    logger.info("Validation time = {}s.".format(valid_duration_time))

                # check after a number of whole epoch.
                if self.train_stats.is_min_lr or self.train_stats.is_max_update:
                    log_string = (f"minimum learning rate {self.learning_rate_min}" if self.train_stats.is_min_lr else 
                                    f"maximun number of updates(steps) {self.max_updates}")
                    logger.info("Training end since %s was reached!", log_string)
                    break 
            else: # normal ended after training.
                logger.info("Training ended after %3d epoches!", epoch_no + 1)

            logger.info("Best Validation result (greedy) at step %8d: %6.2f %s.",
                        self.train_stats.best_ckpt_step, self.train_stats.best_ckpt_score, self.early_stop_metric)
                    
        except KeyboardInterrupt:
            self.save_model_checkpoint(False, float("nan"))

        self.tb_writer.close()
        return None 

    def train_step(self, batch_data: OurData):
        code_tokens = batch_data.code_tokens
        ast_nodes = batch_data.ast_nodes
        text_tokens = batch_data.text_tokens
        ast_positions = batch_data.ast_positions
        ast_edges = batch_data.ast_edges
        node_batch = batch_data.batch
        # prt = batch_data.ptr
        src_mask = (code_tokens != PAD_ID).unsqueeze(1) # src_mask (batch, 1, code_token_length)
        # src_mask: normal is True; pad is False
        text_tokens_input = text_tokens[:, :-1]
        text_tokens_output = text_tokens[:, 1:]
        # FIXME why is text_tokens output to make the trget mask
        trg_mask = (text_tokens_input != PAD_ID).unsqueeze(1) # trg_mask (batch, 1, trg_length)
        # trg_mask: normal is True, pad is False
        ntokens = (text_tokens_output != PAD_ID).data.sum().item()


        # self.unittest_batch_data(code_tokens, ast_nodes, text_tokens, ast_positions, ast_edges, 
        #                                src_mask, text_tokens_input, text_tokens_output, trg_mask, ntokens)
        # assert false

        # get loss (run as during training with teacher forcing)
        batch_loss = self.model(return_type="loss", src_input_code_token=code_tokens,
                                src_input_ast_token=ast_nodes, src_input_ast_position=ast_positions,
                                node_batch=node_batch, edge_index=ast_edges,
                                trg_input=text_tokens_input, trg_truth=text_tokens_output,
                                src_mask=src_mask, trg_mask=trg_mask) 
        
        # logger.warning("batch_loss = {}".format(batch_loss))
        normalized_batch_loss = batch_loss / self.batch_size
        # normalized_batch_loss is the average-sentence level loss.
        # logger.warning("normalized_batch_loss = {}".format(normalized_batch_loss))
        # logger.warning("ntokens = {}".format(ntokens))
        # assert False
        return normalized_batch_loss, ntokens

    def unittest_batch_data(self, code_tokens, ast_nodes, text_tokens, ast_positions, 
                                  ast_edges, src_mask, text_tokens_input, text_tokens_output, trg_mask, ntokens):
        logger.warning("code token shape = {}".format(code_tokens.size()))
        logger.warning("code token = {}".format(code_tokens))
        logger.warning("ast_nodes shape = {}".format(ast_nodes.size()))
        logger.warning("ast_nodes = {}".format(ast_nodes))
        logger.warning("text token shape = {}".format(text_tokens.size()))
        logger.warning("text token = {}".format(text_tokens))
        logger.warning("ast_position shape = {}".format(ast_positions.size()))
        logger.warning("ast_position = {}".format(ast_positions))
        logger.warning("ast edges shape = {}".format(ast_edges.size()))
        logger.warning("ast_edges = {}".format(ast_edges))
        logger.warning("src_mask shape = {}".format(src_mask.size()))
        logger.warning("src_mask = {}".format(src_mask))
        logger.warning("text_token_input shape = {}".format(text_tokens_input.size()))
        logger.warning("text_token_input = {}".format(text_tokens_input))
        logger.warning("text_tokens_output shape = {}".format(text_tokens_output.size()))
        logger.warning("text_tokens_output = {}".format(text_tokens_output))
        logger.warning("trg_mask shape = {}".format(trg_mask.size()))
        logger.warning("trg_mask = {}".format(trg_mask))
        logger.warning("ntokens = {}".format(ntokens))
        

    def validate(self, valid_data: OurDataset):
        """
        Validate on the valid dataset.
        return the validate time.
        """
        validate_start_time = time.time()
        # FIXME
        # predict()
        validate_duration_time = time.time() - validate_start_time

        # write eval_metric and corresponding score to tensorboard
        for eval_metric, score in valid_scores.items():
            if eval_metric in ["loss", "ppl"]:
                self.tb_writer.add_scalar(tag=f"valid/{eval_metric}",scalar_value=score, global_step=self.train_stats.steps)
            else: # FIXME what the difference of if and else
                self.tb_writer.add_scalar(tag=f"Valid/{eval_metric}",scalar_value=score, global_step=self.train_stats.steps)
        
        # the most important metric
        ckpt_score = valid_scores[self.early_stop_metric]

        # set scheduler
        if self.scheduler_step_at == "validation":
            self.scheduler.step(metrics=ckpt_score)

        # update new best
        new_best = self.train_stats.is_best(ckpt_score)
        if new_best:
            self.train_stats.best_ckpt_score = ckpt_score
            self.train_stats.best_ckpt_step = self.train_stats.steps
            logger.info("Horray! New best validation score [%s]!", self.early_stop_metric)

        # save checkpoints
        is_better = self.train_stats.is_better(ckpt_score, self.ckpt_queue) if len(self.ckpt_queue) > 0 else True
        if is_better or self.num_ckpts_keep < 0:
            self.save_model_checkpoint(new_best, ckpt_score)
        
        # append to validation report
        self.add_validation_report(valid_scores=valid_scores, new_best=new_best)
        self.log_examples(valid_hypotheses, valid_references, data=valid_data)

        # store validation set outputs
        validate_output_path = Path(self.model_dir) / f"{self.train_stats.steps}.hyps"
        write_validattion_output_to_file(validate_output_path, valid_hypotheses)

        # store attention plot for selected valid sentences
        # TODO 

        return validate_duration_time

    
    def add_validation_report(self, valid_scores:dict, new_best:bool):
        """
        Append a one-line report to validation logging file.
        """
        current_lr = self.optimizer.param_groups[0]["lr"]
        valid_file = Path(self.model_dir) / "validation.log"
        with valid_file.open("a", encoding="utf-8") as fg:
            score_string = "\t".join([f"Steps: {self.train_stats.steps:7}"] + 
            [f"{eval_metric}: {score:.2f}" if eval_metric in ["bleu", "meteor", "rouge-l"] 
            else f"{eval_metric}: {score:.2f}" for eval_metric, score in valid_scores.items()] +
            [f"LR: {current_lr:.8f}", "*" if new_best else ""])
            fg.write(f"{score_string}\n") 
    
    def log_examples(self, hypotheses: List[str], references:List[str], dataset: OurDataset):
        """
        Log the first self.log_valid_senteces from given examples.
        hypotheses: decoded hypotheses (list of strings)
        references: decoded references (list of strings)
        """
        for id in self.log_valid_sentences:
            if id >= len(hypotheses):
                continue
            logger.info("Example #%d", id)

            # detokenized text:
            logger.info("\tSource: %s", dataset[id])
            logger.info("\tReference: %s", references[id])
            logger.info("\tHypothesis: %s", hypotheses[id])
    
    class TrainStatistics:
        def __init__(self, steps:int=0, is_min_lr:bool=False,
                        is_max_update:bool=False, total_tokens:int=0,
                        best_ckpt_step:int=0, best_ckpt_score: float=np.inf,
                        minimize_metric: bool=True) -> None:
            self.steps = steps 
            self.is_min_lr = is_min_lr
            self.is_max_update = is_max_update
            self.total_tokens = total_tokens
            self.best_ckpt_step = best_ckpt_step
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
    
    def save_model_checkpoint(self, new_best:bool, score:float) -> None:
        """
        Save model's current parameters and the training state to a checkpoint.
        The training state contains the total number of training steps, 
                                    the total number of training tokens, 
                                    the best checkpoint score and iteration so far, 
                                    and optimizer and scheduler states.
        new_best: for update best.ckpt
        score: Validation score which is used as key of heap queue.
        """        
        model_checkpoint_path = Path(self.model_dir) / f"{self.train_stats.steps}.ckpt"
        model_state_dict = self.model.state_dict()
        global_states = {
            "steps": self.train_stats.steps,
            "total_tokens": self.train_stats.total_tokens,
            "best_ckpt_score": self.train_stats.best_ckpt_score,
            "best_ckpt_step": self.train_stats.best_ckpt_step,
            "model_state": model_state_dict,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(global_states, model_checkpoint_path)

        def symlink_update(target:Path, link_name:Path):
            """
            Find the file that the symlink currently points to, sets it to
            the new target, and return the previous target if it exists.
            """
            if link_name.is_symlink():
                current_link = link_name.resolve()
                link_name.unlink()
                link_name.symlink_to(target)
                return current_link
            else:
                link_name.symlink_to(target)
                return None 

        # how to update queue and keep the best number of ckpt.
        symlink_target = Path(f"{self.train_stats.steps}.ckpt")
        last_path = Path(self.model_dir) / "lastest.ckpt"
        prev_path = symlink_update(symlink_target, last_path)
        best_path = Path(self.model_dir) / "best.ckpt"
        if new_best:
            prev_best = symlink_update(symlink_target, best_path)
            assert best_path.resolve().stem == str(self.train_stats.best_ckpt_step)
        
        def delete_ckpt(path:Path) -> None:
            try:
                logger.info("Delete %s", path)
                path.unlink()  # delete file
            except FileNotFoundError as error:
                logger.warning("Want to delete old checkpoint %s"
                               "but file does not exist. (%s)", path, error)
        
        # push and pop from the heap queue.
        to_delete = None 
        if not math.isnan(score) and self.num_ckpts_keep > 0:
            if len(self.ckpt_queue) < self.num_ckpts_keep: # don't need pop, only push
                heapq.heappush(self.ckpt_queue, (score, model_checkpoint_path))
            else: # ckpt_queue already full.
                if self.minimize_metric: # the smaller score, the better
                    heapq._heapify_max(self.ckpt_queue) # change a list to a max-heap
                    to_delete = heapq._heappop_max(self.ckpt_queue)
                    heapq.heappush(self.ckpt_queue, (score, model_checkpoint_path))
                else: # the higher score, the better
                    to_delete = heapq.heappushpop(self.ckpt_queue, (score, model_checkpoint_path))
            
            if to_delete is not None:
                assert to_delete[1] != model_checkpoint_path  # don't delete the last ckpt, double check
                if to_delete[1].stem != best_path.resolve().stem:
                    delete_ckpt(to_delete[1]) # don't delete the best ckpt
            
            assert len(self.ckpt_queue) <= self.num_ckpts_keep
            if prev_path is not None and prev_path.stem not in [c[1].stem for c in self.ckpt_queue]:
                delete_ckpt(prev_path)

    def init_from_checkpoint(self, path:Path, 
                             reset_best_ckpt:bool=False, reset_scheduler:bool=False,
                             reset_optimizer:bool=False, reset_iter_state:bool=False):
        """
        Initialize the training from a given checkpoint file.
        The checkpoint file contain not only model parameters, but also 
        scheduler and optimizer states.
        """
        logger.info("Loading model from %s", path)
        assert path.is_file(), f"model checkpoint {path} not found!"
        model_checkpoint = torch.load(path, map_location=self.device)
        logger.info("Load model from %s", path)

        # restore model parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            logger.warning("Reset Optimizer.")
        
        if not reset_scheduler:
            if model_checkpoint["scheduler_state"] is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(model_checkpoint["schedular_state"])
        else:
            logger.warning("Reset Scheduler.")

        if not reset_best_ckpt:
            self.train_stats.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.train_stats.best_ckpt_step = model_checkpoint["best_ckpt_step"]
        else:
            logger.warning("Reset tracking of the best checkpoints.")
        
        if not reset_iter_state:
            assert "train_iter_state" in model_checkpoint
            self.train_stats.steps = model_checkpoint["steps"]
            self.train_stats.total_tokens = model_checkpoint["total_tokens"]
        else:
            logger.info("Reset data_loader (random seed: {%d}).", self.seed)
        
        if self.device.type == "cuda":
            self.model.to(self.device)

def train(cfg_file: str) -> None:
    """
    Main Training function.
    """
    cfg = load_config(Path(cfg_file))

    model_dir = Path(cfg["training"]["model_dir"])
    overwrite = cfg["training"].get("overwrite", False)
    model_dir = make_model_dir(model_dir, overwrite) # model_dir: absolute path

    make_logger(model_dir)

    set_seed(seed=int(cfg["training"].get("random_seed", 820)))

    # load data
    train_dataset, valid_dataset, test_dataset, vocab_info = load_data(data_cfg=cfg["data"])
    
    # build model
    model = build_model(model_cfg=cfg["model"], vocab_info=vocab_info)

    # training management
    trainer = TrainManager(model=model, cfg=cfg)

    # train model
    trainer.train_and_validate(train_dataset=train_dataset, valid_dataset=valid_dataset)

if __name__ == "__main__":
    cfg_file = "test.yaml"
    train(cfg_file)