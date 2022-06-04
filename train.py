import torch
import bmtrain as bmp
from bmtrain import print_rank
from model_center.model import Bert, BertConfig, Roberta, RobertaConfig
from model_center.tokenizer import BertTokenizer, RobertaTokenizer
from model_center.dataset import MMapIndexedDataset, DistributedMMapIndexedDataset, DistributedDataLoader
from model_center.utils import print_inspect
from dataset import BertDataset
from torch.utils.tensorboard import SummaryWriter
#from transformers import 
import time
from arguments import get_args
import os
from tokenizers import Tokenizer, ByteLevelBPETokenizer
def get_tokenizer():
    tokenizer= ByteLevelBPETokenizer.from_pretrained(vocab = '/home/user/bm_train_codes/tokenizer/tokenizer_.json') 
    return tokenizer

def get_model(args):
    # model = Roberta.from_pretrained('roberta-base')
    config = RobertaConfig.from_pretrained("roberta-base") 
    model = Roberta(config)
    if args.load != None:
        print_rank(f"Loading from checkpoint {args.load}")
        bmp.load(model, args.load)
    else:
        print_rank("Training model from scratch...")
        bmp.init_parameters(model)
    print_inspect(model, "*")
    return model

def get_optimizer(args, model):
    optimizer = bmp.optim.AdamOffloadOptimizer(model.parameters(),
                                                lr = args.lr,
                                                betas = (0.9,0.98),
                                                eps = 1e-6,
                                                weight_decay=args.weight_decay, 
                                                scale=args.loss_scale,
                                               )
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    lr_scheduler = None
    if args.lr_decay_style == "noam":
        lr_scheduler = bmp.lr_scheduler.Noam(optimizer, 
                                         start_lr = args.lr,
                                         warmup_iter = args.warmup_iters, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = args.start_step)
    elif args.lr_decay_style == "linear":
        lr_scheduler = bmp.lr_scheduler.Linear(optimizer, 
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
    bmp.print_rank("Model memory\n", torch.cuda.memory_summary())
    bmp.synchronize()
    return model, optimizer, lr_scheduler

def get_dataset(args):
    print(bmp.rank(), bmp.world_size())
    input_ids_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'input_ids', bmp.rank(), bmp.world_size())
    lm_pos_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'lm_pos', bmp.rank(), bmp.world_size())
    masked_labels_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'masked_labels', bmp.rank(), bmp.world_size())
    length_list_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'length_list', bmp.rank(), bmp.world_size())
    bert_dataset = BertDataset(input_ids_dataset, lm_pos_dataset, masked_labels_dataset, length_list_dataset)

    return bert_dataset

def batch_iter(args, dataset):
    st = args.start_step * args.batch_size
    # st = 0
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    while True:
        input_ids, attention_mask, labels = dataset[st]
        st += 1
        # bmp.print_rank(f"##################\ninput_ids={input_ids},\nattention={attention_mask}, \nlabel={labels}")
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

        if len(input_ids_list) >= args.batch_size:
            yield {
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
                "labels": torch.stack(labels_list)
            }
            input_ids_list = []
            attention_mask_list = []
            labels_list = []

def pretrain(args, model, optimizer, lr_scheduler, dataset):
    average_time = 0
    average_time_shift = 0.9
    loss_func = bmp.loss.FusedCrossEntropy(ignore_index=-100)

    start_step = args.start_step
    log_loss = 0
    os.makedirs(os.path.join(args.save, 'checkpoints'), exist_ok=True)
    if bmp.rank() == 0:
        writer = SummaryWriter(os.path.join(args.save, 'tensorborads'))

    for step, data in enumerate(batch_iter(args, dataset)):
        optimizer.zero_grad()

        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        labels = data['labels'].cuda()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, return_logits = True, attention_3dim = True)
        # output = model(input_ids=input_ids, attention_mask=attention_mask)
        # logits = output.logits
        loss = loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        global_loss = bmp.sum_loss(loss).item()
        log_loss += global_loss
        loss = optimizer.loss_scale(loss)
        loss.backward()
        grad_norm = bmp.optim.clip_grad_norm(optimizer.param_groups, \
            max_norm = float('inf'), scale = optimizer.scale, norm_type = 2)
        bmp.optim_step(optimizer, lr_scheduler)

        if (start_step + step + 1) % args.log_iters == 0:
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

        if bmp.rank() == 0:
            writer.add_scalar("Loss/train", global_loss, step + start_step)
        if args.save != None and (step + start_step + 1) % args.save_iters == 0:
            bmp.print_rank(f"Saving checkpoint at {step + start_step} step.")
            bmp.save(model, os.path.join(args.save, 'checkpoints', "checkpoint-%d.pt" % (step + start_step)))

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
    dataset = get_dataset(args)
    pretrain(args, model, optimizer, lr_scheduler, dataset)


if __name__ == '__main__':
    main()
    # args = initialize()
    # config = RobertaConfig.from_pretrained('roberta-base')
    # tokenizer = get_tokenizer()
    # input_ids = MMapIndexedDataset('/home/caohanwen/bm_train_codes/masked_output/original_wwm/input_ids_0_38')
    # length_list = MMapIndexedDataset('/home/caohanwen/bm_train_codes/masked_output/original_wwm/length_list_0_38')
    # masked_labels = MMapIndexedDataset('/home/caohanwen/bm_train_codes/masked_output/original_wwm/masked_labels_0_38')
    # lm_pos = MMapIndexedDataset('/home/caohanwen/bm_train_codes/masked_output/original_wwm/lm_pos_0_38')
    # bert_dataset = BertDataset(input_ids, lm_pos, masked_labels, length_list)
    
    # for i in range(len(bert_dataset)):
    #     if len(length_list[i]) != 0:
    #         input_ids, attention_mask, labels = bert_dataset[i]
    #         bmp.print_rank(input_ids[0])
    #         bmp.print_rank(attention_mask.shape)
    #         break