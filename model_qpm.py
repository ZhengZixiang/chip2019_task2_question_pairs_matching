# -*— coding: utf-8 -*-
""" Finetuning the library models for chip2019 question pairs matching. """

import argparse
import glob
import logging
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule
from data_utils import (compute_metrics, convert_examples_to_features, QPMProcessor)

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model. """
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info('***** Running training *****')
    logger.info('   Num examples = %d', len(train_dataset))
    logger.info('   Num Epochs = %d', args.num_train_epochs)
    logger.info('   Instantaneous batch size per GPU = %d', args.per_gpu_train_batch_size)
    logger.info('   Total train batch size (w. parallel & accumulation) = %d',
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info('   Gradient Accumulation steps = %d', args.gradient_accumulation_steps)
    logger.info('   Total optimization steps = %d', t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch')
    set_seed(args)  # Added here for reproductibility

    max_val_acc = 0
    max_val_f1 = 0

    for _ in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        # for step, batch in enumerate(epoch_iterator):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch_transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_los(loss.optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        result = evaluate(args, model, tokenizer)
                        for key, value in result.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        if result['acc'] > max_val_acc:
                            max_val_acc = result['acc']
                        if result['f1'] > max_val_f1:
                            max_val_f1 = result['f1']
                            output_dir = os.path.join(args.output_dir, 'best_checkpoint')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, 'training_args.bin')
                            logger.info('Saving model checkpoint with f1 {:.4f}'.format(max_val_f1))
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss-logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                # if args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, 'training_args.bin')
                #     logger.info('Saving model checkpoint to %s', output_dir)

            # if args.max_steps > 0 and global_step > args.max_steps:
            #     epoch_iterator.close()
            #     break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=''):
    eval_output_dir = args.output_dir

    results = {}
    eval_dataset = load_and_cache_examples(args, tokenizer, set_type='dev')

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info('***** Running evaluation {} *****'.format(prefix))
    logger.info('   Num examples = %d', len(eval_dataset))
    logger.info('   Batch size = %d', args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, 'eval_results.txt')
    with open(output_eval_file, 'a') as writer:
        for key in sorted(result.keys()):
            logger.info('   %s = %s', key, str(result[key]))
            writer.write('%s = %s\n' % (key, str(result[key])))
        writer.write('='*20 + '\n')

    return results


def predict(args, model, tokenizer, index):
    test_dataset = load_and_cache_examples(args, tokenizer, set_type='test')

    args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size)

    # Eval!
    logger.info('***** Running prediction *****')
    logger.info('   Num examples = %d', len(test_dataset))
    logger.info('   Batch size = %d', args.test_batch_size)
    preds = None
    for batch in tqdm(test_dataloader, desc='Testing'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    with open(os.path.join(args.data_dir + str(index), 'result.csv'), 'w') as f:
        f.write('id,label\n')
        for i, pred in enumerate(preds):
            f.write('%d,%d\n' % (i, pred))


def load_and_cache_examples(args, tokenizer, set_type):
    processor = QPMProcessor()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        set_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)
    ))
    if os.path.exists(cached_features_file):
        logger.info('Loading features from cache file %s', cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info('Creating features from dataset file at %s', args.data_dir)
        label_list = processor.get_labels()
        category_list = processor.get_categories()
        examples = processor.get_examples(args.data_dir, set_type)
        features = convert_examples_to_features(examples, label_list, category_list, args.max_seq_length, tokenizer,
            cls_token_at_end=False,    # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0
            # cls_token_at_end=bool(args.model_type in ['xlnet']),    # xlnet has a cls token at the end
            # cls_token=tokenizer.cls_token,
            # cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            # sep_token=tokenizer.sep_token,
            # sep_token_extra=bool(args.model_type in ['roberta']),   # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            # pad_on_left=bool(args.model_type in ['xlnet']),         # pad on the left for xlnet
            # pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            # pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_ct_clf_input_ids = torch.tensor([f.category_clf_input_ids for f in features], dtype=torch.long)
    all_ct_clf_input_mask = torch.tensor([f.category_clf_input_mask for f in features], dtype=torch.long)
    all_ct_clf_segment_ids = torch.tensor([f.category_clf_segment_ids for f in features], dtype=torch.long)
    all_category_ids = torch.tensor([f.category_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                            all_ct_clf_input_ids, all_ct_clf_input_mask, all_ct_clf_segment_ids, all_category_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--data_dir', default=None, type=str, required=True,
                        help='The input data dir. Should contain the .csv files for the task.')
    parser.add_argument('--model_name_or_path', default=None, type=str, required=True,
                        help='Path to pretrained model or shortcut name selected in the list.')
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='The output directory where the model predictions and checkpoints will be written.')

    ## Other parameters
    parser.add_argument('--config_name', default='', type=str,
                        help='Pretrained config name or path if not the same as model_name.')
    parser.add_argument('--tokenizer_name', default='', type=str,
                        help='Pretrained tokenizer name or path if not the same as model_name.')
    parser.add_argument('--max_seq_length', default='128', type=int,
                        help='The maximum total input sequence length after tokenization. Sequences longer than this '
                             'will be truncated, sequences shorter will be padded.')
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to run eval on the dev set.')
    parser.add_argument('--do_predict', action='store_true',
                        help='Whether to run test on the test set.')
    parser.add_argument('--evaluate_during_training', action='store_true',
                        help='Rul evaluation during training at each logging step.')
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Set this flag if you are using an uncased model.')

    parser.add_argument('--per_gpu_train_batch_size', default=1, type=int,
                        help='Batch size per GPU/CPU for training.')
    parser.add_argument('--per_gpu_eval_batch_size', default=8, type=int,
                        help='Batch size per GPU/CPU for evaluation.')
    parser.add_argument('--per_gpu_test_batch_size', default=8, type=int,
                        help='Batch size per GPU/CPU for prediction.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument('--weight_decay', default=0.0, type=float,
                        help='Weight decay if we apply some.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float,
                        help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Max gradient norm.')
    parser.add_argument('--num_train_epochs', default=4.0, type=float,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--max_steps', default=-1, type=int,
                        help='If > 0: set total number of training steps to perform. Override num_train_epochs.')
    parser.add_argument('--warmup_steps', default=0, type=int,
                        help='Linear warmup over warmup_steps.')

    parser.add_argument('--logging_steps', type=int, default=50,
                        help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='Save checkpoint every X updates steps.')
    parser.add_argument('--eval_all_checkpoints', action='store_true',
                        help='Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Avoid using CUDA when available.')
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help='Overwrite the content of the output directory.')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite the cached training and evaluation sets.')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for initialization')

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()

    # Setup CUDA, GPU
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError('Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.')

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning('Process device: %s, n_gpu: %s, 16-bits training: %s',
                   device, args.n_gpu, args.fp16)

    # Set seed
    set_seed(args)
    # Prepare QPM task
    processor = QPMProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    model.to(args.device)

    logger.info('Trainning/evaluation parameters %s', args)
    parent_data_dir = args.data_dir
    parent_output_dir = args.output_dir

    # Trainning
    results_tmp = {}
    if args.do_train:
        # 10-Fold dataset for training.
        for i in range(0, 10):
            # Reload the pretrained model.
            model = model_class.from_pretrained(args.model_name_or_path,
                                                from_tf=bool('.ckpt' in args.model_name_or_path),
                                                config=config)
            model.to(args.device)

            args.data_dir = parent_data_dir + str(i)
            args.output_dir = parent_output_dir + str(i)

            train_dataset = load_and_cache_examples(args, tokenizer, set_type='train')
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
            # Create output directory if needed
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            model.to(args.device)

            # for reduce the usage of disk, evluate and find the best checkpoint every sub dataset.
            # args.data_dir = parent_data_dir + str(i)
            # args.output_dir = parent_output_dir + str(i)
            # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            # checkpoints = [args.output_dir]
            # if args.eval_all_checkpoints:
            #     checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            #     logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            # logger.info("Evaluate the following checkpoints: %s", checkpoints)
            # best_f1 = 0.0
            # for checkpoint in checkpoints:
            #     global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            #     model = model_class.from_pretrained(checkpoint)
            #     model.to(args.device)
            #     result = evaluate(args, model, tokenizer, prefix=global_step)
            #     if result['f1'] > best_f1:
            #         best_f1 = result['f1']
            #         # Save the best model checkpoint
            #         output_dir = os.path.join(args.output_dir, 'best_checkpoint_fold' + str(i))
            #         if not os.path.exists(output_dir):
            #             os.makedirs(output_dir)
            #         model_to_save = model.module if hasattr(model, 'module') else model
            #         model_to_save.save_pretrained(output_dir)
            #         torch.save(args, 'training_args.bin')
            #         logger.info('Saving model checkpoint to %s', output_dir)
            #
            #     result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            #     results_tmp.update(result)
            # checkpoints.remove(args.output_dir)
            # for checkpoint in checkpoints:
            #     shutil.rmtree(checkpoint)

    # Evaluation
    results = {}
    if args.do_eval:
        for i in range(10):
            args.data_dir = parent_data_dir + str(i)
            args.output_dir = parent_output_dir + str(i)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            best_f1 = 0.0
            for checkpoint in checkpoints:
                global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
                result = evaluate(args, model, tokenizer, prefix=global_step)
                if result['f1'] > best_f1:
                    best_f1 = result['f1']
                    # Save the best model checkpoint
                    output_dir = os.path.join(args.output_dir, 'best_checkpoint_fold' + str(i))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, 'training_args.bin')
                    logger.info('Saving model checkpoint to %s', output_dir)

                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                results.update(result)

    # Prediction
    if args.do_predict:
        for i in range(1):
            args.output_dir = parent_output_dir + str(i)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoint = args.output_dir + '/best_checkpoint_fold' + str(i)
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            predict(args, model, tokenizer, i)

        # For bagging.
        all = pd.read_csv('./data/sample_submission.csv')
        for i in range(10):
            df = pd.read_csv(args.data_dir + str(i) + '/result.csv')
            all['label'] += df['label']
        all['label'] = all['label'] // 6
        all.to_csv('./data/result.csv', index=False)


if __name__ == '__main__':
    main()
