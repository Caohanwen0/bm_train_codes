import torch,os
os.environ["CUDA_VISIBLE_DEVICES"]=','.join(map(str,[6,7,]))
import bmtrain as bmp
from model_center.model import Roberta, RobertaConfig
from model_center.tokenizer import BertTokenizer
from model_center.dataset import MMapIndexedDataset, DistributedMMapIndexedDataset, DistributedDataLoader
from model_center.utils import print_inspect
from dataset import BertDataset
from torch.utils.tensorboard import SummaryWriter
#from transformers import 
import time
from arguments import get_args
import os

def get_model(args):
    config = RobertaConfig.from_json_file(args.model_config)
    assert isinstance(config, RobertaConfig)
    model = Roberta(config)
    if args.load != None:
        bmp.print_rank(f"Loading from checkpoint {args.load}...")
        bmp.load(model, args.load)
    else:
        bmp.print_rank("Training model from scratch...")
        bmp.init_parameters(model)
    print_inspect(model, "*")
    return model

def get_optimizer(args, model):
    optimizer = bmp.optim.AdamOffloadOptimizer(model.parameters(), 
                                                lr = 5e-4,
                                                betas = (0.9, 0.98),
                                                weight_decay=args.weight_decay, 
                                                scale=args.loss_scale)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == 'linear':
        lr_scheduler = bmp.lr_scheduler.Linear(optimizer, 
                                         start_lr = args.lr,
                                         warmup_iter = args.warmup_iters, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = args.start_step)
    else:
        lr_scheduler = bmp.lr_scheduler.Noam(optimizer, 
                                         start_lr = args.lr,
                                         warmup_iter = args.warmup_iters, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = args.start_step) 
    return lr_scheduler

def setup_model_and_optimizer(args):
    # get the model
    model = get_model(args)
    bmp.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmp.synchronize()
    # get the memory usage
    bmp.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmp.synchronize()
    return model, optimizer, lr_scheduler

def get_train_dataset(args):
    print(bmp.rank(), bmp.world_size())
    input_ids_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'input_ids', bmp.rank(), bmp.world_size())
    lm_pos_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'lm_pos', bmp.rank(), bmp.world_size())
    masked_labels_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'masked_labels', bmp.rank(), bmp.world_size())
    length_list_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'length_list', bmp.rank(), bmp.world_size())
    bert_dataset = BertDataset(input_ids_dataset, lm_pos_dataset, masked_labels_dataset, length_list_dataset)

    return bert_dataset

def get_valid_dataset(dataset_path):
    input_ids_dataset = MMapIndexedDataset(os.path.join(dataset_path,'valid', 'input_ids'))
    lm_pos_dataset = MMapIndexedDataset(os.path.join(dataset_path, 'valid','lm_pos'))
    masked_labels_dataset = MMapIndexedDataset(os.path.join(dataset_path, 'valid','masked_labels'))
    length_list_dataset = MMapIndexedDataset(os.path.join(dataset_path, 'valid','length_list'))
    bert_dataset = BertDataset(input_ids_dataset, lm_pos_dataset, masked_labels_dataset, length_list_dataset)

    return bert_dataset

def valid(model, twitter_dev_dataloader,reddit_dev_dataloader, ccnews_dev_dataloader, loss_func, step, writer):
    return
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        dataloader_list = [twitter_dev_dataloader,reddit_dev_dataloader, ccnews_dev_dataloader] 
        valid_data_name = ['twitter', 'reddit', 'ccnews']
        for dev_dataloader, dev_name in zip(dataloader_list, valid_data_name):
            for data in dev_dataloader:
                input_ids, attention_mask, labels = data
                input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
                logits = model(input_ids=input_ids, attention_mask=attention_mask, return_logits=True)
                loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
                global_loss = bmp.sum_loss(loss).item()
                valid_loss += global_loss
            if bmp.rank() == 0:
                writer.add_scalar(f"Loss/{dev_name}", valid_loss, step)
            print_rank(model, "*")
            bmp.print_rank(
                            "{} | Iter: {:6d} | valid {} loss: {:.4f}".format(
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                step,
                                dev_name,
                                valid_loss / len(dev_dataloader)
                            )
                        )
    model.train()

def batch_iter(args, dataset):
    st = args.start_step * args.batch_size
    # st = 0
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    while True:
        input_ids, attention_mask, labels = dataset[st]
        st += 1
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

        if len(input_ids_list) > args.batch_size:
            yield {
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
                "labels": torch.stack(labels_list)
            }
            input_ids_list = []
            attention_mask_list = []
            labels_list = []

def pretrain(args, model, optimizer, lr_scheduler, train_dataset, twitter_dev_dataloader,\
    reddit_dev_dataloader, ccnews_dev_dataloader):
    average_time = 0
    average_time_shift = 0.9
    loss_func = bmp.loss.FusedCrossEntropy(ignore_index=-100)

    start_step = args.start_step
    log_loss = 0
    os.makedirs(os.path.join(args.save, 'checkpoints'), exist_ok=True)
    if bmp.rank() == 0:
        writer = SummaryWriter(os.path.join(args.save, 'tensorborads'))
    else:
        writer = None
    valid(model, twitter_dev_dataloader,\
        reddit_dev_dataloader, ccnews_dev_dataloader, loss_func, start_step // args.gradient_accumulate, writer)
    for step, data in enumerate(batch_iter(args, train_dataset)):
        optimizer.zero_grad()
        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        labels = data['labels'].cuda()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, return_logits=True)
        loss = loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        global_loss = bmp.sum_loss(loss).item()
        log_loss += global_loss
        loss = loss / args.gradient_accumulate
        loss = optimizer.loss_scale(loss)
        loss.backward()
        if (start_step + step + 1) % args.gradient_accumulate == 0:
            grad_norm = bmp.optim.clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale, norm_type = 2)
            bmp.optim_step(optimizer, lr_scheduler)

        if (start_step + step + 1) % args.log_iters == 0:
            print_inspect(model, "*")
            bmp.print_rank(
                    "{} | Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f}".format(
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        step + 1 + start_step,
                        log_loss / args.log_iters,
                        lr_scheduler.current_lr,
                        int(optimizer.scale),
                        grad_norm
                    )
                )
            log_loss = 0

        if (start_step + step + 1) % args.valid_iters == 0:
            print_inspect(model, "*")
            valid(model, twitter_dev_dataloader, reddit_dev_dataloader, ccnews_dev_dataloader, loss_func, (start_step + step + 1) // args.gradient_accumulate, writer)

        if bmp.rank() == 0:
            writer.add_scalar("Loss/train", global_loss, (step + start_step) // args.gradient_accumulate)
        if args.save != None and (step + start_step + 1) % args.save_iters == 0:
            bmp.save(model, os.path.join(args.save, 'checkpoints', "checkpoint-%d.pt" % (step + start_step)))
            print_inspect(model, "*")
            bmp.print_rank(f"Saving checkpoint at {step + start_step} step.")

def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    bmp.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

def main():
    args = initialize()
    print(args)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    train_dataset = get_train_dataset(args)
    twitter_valid_dataset = get_valid_dataset(args.test_dataset_twitter)
    reddit_valid_dataset = get_valid_dataset(args.test_dataset_reddit) 
    ccnews_valid_dataset = get_valid_dataset(args.test_dataset_ccnews)
    twitter_dev_dataloader = DistributedDataLoader(twitter_valid_dataset, batch_size=args.batch_size, shuffle=False)
    reddit_dev_dataloader = DistributedDataLoader(reddit_valid_dataset, batch_size=args.batch_size, shuffle=False)
    ccnews_dev_dataloader = DistributedDataLoader(ccnews_valid_dataset, batch_size=args.batch_size, shuffle=False) 
    pretrain(args, model, optimizer, lr_scheduler, train_dataset, twitter_dev_dataloader,\
        reddit_dev_dataloader, ccnews_dev_dataloader)

if __name__ == '__main__':
    main()

