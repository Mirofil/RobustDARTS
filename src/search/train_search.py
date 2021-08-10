# python src/search/train_search.py --seed 2 --cutout --report_freq_hessian 1 --space s2 --drop_path_prob 0.0 --weight_decay 0.0003

import os
import sys
import glob
import numpy as np
import torch
import json
import codecs
import pickle
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from copy import deepcopy
from numpy import linalg as LA
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
import nasbench301 as nb

sys.path.insert(0, '../RobustDARTS')

from src import utils
from src.spaces import spaces_dict
from src.search.model_search import Network
from src.search.architect import Architect
from src.search.analyze import Analyzer
from src.search.args import Helper
import wandb

from collections import defaultdict, namedtuple
from tqdm import tqdm

helper = Helper()
args = helper.config

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log_{}.txt'.format(args.task_id)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

schedule_of_params = []

def format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, search_loader_iter, inner_steps, args, epoch = 1000, loader_type="train-val"):
    base_inputs, base_targets = base_inputs.cuda(non_blocking=True), base_targets.cuda(non_blocking=True)
    arch_inputs, arch_targets = arch_inputs.cuda(non_blocking=True), arch_targets.cuda(non_blocking=True)
    if args.higher_method == "sotl":
        arch_inputs, arch_targets = None, None
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = [base_inputs], [base_targets], [arch_inputs], [arch_targets]
    for extra_step in range(inner_steps-1):
        if args.inner_steps_same_batch and (args.warm_start is None or epoch >= args.warm_start):
            all_base_inputs.append(base_inputs)
            all_base_targets.append(base_targets)
            all_arch_inputs.append(arch_inputs)
            all_arch_targets.append(arch_targets)
            continue # If using the same batch, we should not try to query the search_loader_iter for more samples
        try:
            if loader_type == "train-val" or loader_type == "train-train":
              (extra_base_inputs, extra_base_targets), (extra_arch_inputs, extra_arch_targets)= next(search_loader_iter)
            else:
              extra_base_inputs, extra_base_targets = next(search_loader_iter)
              extra_arch_inputs, extra_arch_targets = None, None
        except Exception as e:
            continue
        # extra_base_inputs, extra_arch_inputs = extra_base_inputs.cuda(non_blocking=True), extra_arch_inputs.cuda(non_blocking=True)
        # extra_base_targets, extra_arch_targets = extra_base_targets.cuda(non_blocking=True), extra_arch_targets.cuda(non_blocking=True)
        extra_base_inputs, extra_base_targets = extra_base_inputs.cuda(non_blocking=True), extra_base_targets.cuda(non_blocking=True)
        if extra_arch_inputs is not None and extra_arch_targets is not None:
          extra_arch_inputs, extra_arch_targets = extra_arch_inputs.cuda(non_blocking=True), extra_arch_targets.cuda(non_blocking=True)
        
        all_base_inputs.append(extra_base_inputs)
        all_base_targets.append(extra_base_targets)
        all_arch_inputs.append(extra_arch_inputs)
        all_arch_targets.append(extra_arch_targets)

    return all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def genotype_to_adjacency_list(genotype, steps=4):
  # Should pass in genotype.normal or genotype.reduce
  G = defaultdict(list)
  for nth_node, connections in enumerate(chunks(genotype, 2), start=2): # Darts always keeps two connections per node and first two nodes are fixed input
    for connection in connections:
      G[connection[1]].append(nth_node)
  # Add connections from all intermediate nodes to Output node
  for intermediate_node in [2,3,4,5]:
    G[intermediate_node].append(6)
  return G
    
def DFS(G,v,seen=None,path=None):
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths
  
def count_edges_along_path(genotype, path):
  count = 0
  for i in range(1, len(path)-1): #Leave out the first and last nodes
    idx_in_genotype = path[i]-2
    relevant_edges = genotype[idx_in_genotype*2:idx_in_genotype*2+2]
    for edge in relevant_edges:
      if edge[1] == path[i-1]:
        count += 1
  return count

def genotype_depth(genotype):
  # The shortest path can start in either of the two input nodes
  all_paths0 = DFS(genotype_to_adjacency_list(genotype), 0)
  all_paths1 = DFS(genotype_to_adjacency_list(genotype), 1)

  cand0 = max(len(p)-1 for p in all_paths0)
  cand1 = max(len(p)-1 for p in all_paths1)
  
  # max_paths0 = [p for p in all_paths0 if len(p) == cand0]
  # max_paths1 = [p for p in all_paths1 if len(p) == cand1]

  # path_depth0 = max([count_edges_along_path(genotype, p) for p in max_paths0])
  # path_depth1 = max([count_edges_along_path(genotype, p) for p in max_paths1])
  
  # return max(path_depth0, path_depth1)

  return max(cand0, cand1)

def genotype_width(genotype):
  width = 0
  for edge in genotype:
    if edge[1] in [0, 1]:
      width += 1/2
  return width
      

def count_ops(genotype):
  PRIMITIVES = [
      'none',
      'max_pool_3x3',
      'avg_pool_3x3',
      'skip_connect',
      'sep_conv_3x3',
      'sep_conv_5x5',
      'dil_conv_3x3',
      'dil_conv_5x5'
  ]
  genotype = str(genotype)
  counts = {op: genotype.count(op) for op in PRIMITIVES}
  return counts

def wandb_auth(fname: str = "nas_key.txt"):
  gdrive_path = "/content/drive/MyDrive/colab/wandb/nas_key.txt"
  if "WANDB_API_KEY" in os.environ:
      wandb_key = os.environ["WANDB_API_KEY"]
  elif os.path.exists(os.path.abspath("~" + os.sep + ".wandb" + os.sep + fname)):
      # This branch does not seem to work as expected on Paperspace - it gives '/storage/~/.wandb/nas_key.txt'
      print("Retrieving WANDB key from file")
      f = open("~" + os.sep + ".wandb" + os.sep + fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists("/root/.wandb/"+fname):
      print("Retrieving WANDB key from file")
      f = open("/root/.wandb/"+fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key

  elif os.path.exists(
      os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname
  ):
      print("Retrieving WANDB key from file")
      f = open(
          os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname,
          "r",
      )
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists(gdrive_path):
      print("Retrieving WANDB key from file")
      f = open(gdrive_path, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  wandb.login()
  
def load_nb301():
    version = '0.9'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_0_9_dir = os.path.join(current_dir, 'nb_models_0.9')
    model_paths_0_9 = {
        model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    models_1_0_dir = os.path.join(current_dir, 'nb_models_1.0')
    model_paths_1_0 = {
        model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

    # If the models are not available at the paths, automatically download
    # the models
    # Note: If you would like to provide your own model locations, comment this out
    if not all(os.path.exists(model) for model in model_paths.values()):
        nb.download_models(version=version, delete_zip=True,
                        download_dir=current_dir)

    # Load the performance surrogate model
    #NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
    #NOTE: Defaults to using the default model download path
    print("==> Loading performance surrogate model...")
    ensemble_dir_performance = model_paths['xgb']
    print(ensemble_dir_performance)
    performance_model = nb.load_ensemble(ensemble_dir_performance)
    
    return performance_model

def main(primitives):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logger = logging.getLogger()

  wandb_auth()
  run = wandb.init(project="NAS", group=f"Search_Cell_darts_orig", reinit=True)
  
  api = load_nb301()

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  model_init = Network(args.init_channels, args.n_classes, args.layers, criterion,
                       primitives, steps=args.nodes)
  model_init = model_init.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model_init))

  optimizer_init = torch.optim.SGD(
      model_init.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  architect_init = Architect(model_init, args)

  scheduler_init = CosineAnnealingLR(
        optimizer_init, float(args.epochs), eta_min=args.learning_rate_min)

  analyser_init = Analyzer(args, model_init)
  la_tracker = utils.EVLocalAvg(args.window, args.report_freq_hessian,
                                args.epochs)

  train_queue, valid_queue, train_transform, valid_transform = helper.get_train_val_loaders()
  if args.merge_train_val:
    valid_queue = train_queue

  errors_dict = {'train_acc': [], 'train_loss': [], 'valid_acc': [],
                 'valid_loss': []}

  #for epoch in range(args.epochs):
  def train_epochs(epochs_to_train, iteration,
                   args=args, model=model_init, optimizer=optimizer_init,
                   scheduler=scheduler_init,
                   train_queue=train_queue, valid_queue=valid_queue,
                   train_transform=train_transform,
                   valid_transform=valid_transform,
                   architect=architect_init, criterion=criterion,
                   primitives=primitives, analyser=analyser_init,
                   la_tracker=la_tracker,
                   errors_dict=errors_dict, start_epoch=-1):

    logging.info('STARTING ITERATION: %d', iteration)
    logging.info('EPOCHS TO TRAIN: %d', epochs_to_train - start_epoch - 1)

    la_tracker.stop_search = False

    if epochs_to_train - start_epoch - 1 <= 0:
        return model.genotype(), -1
    for epoch in range(start_epoch+1, epochs_to_train):
      # set the epoch to the right one
      #epoch += args.epochs - epochs_to_train

      scheduler.step(epoch)
      lr = scheduler.get_lr()[0]
      if args.drop_path_prob != 0:
        model.drop_path_prob = args.drop_path_prob * epoch / (args.epochs - 1)
        train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
        logging.info('epoch %d lr %e drop_prob %e cutout_prob %e', epoch, lr,
                      model.drop_path_prob,
                      train_transform.transforms[-1].cutout_prob)
      else:
        logging.info('epoch %d lr %e', epoch, lr)

      # training
      train_acc, train_obj, ev = train(epoch, primitives, train_queue,
                                   valid_queue, model, architect, criterion,
                                   optimizer, lr, analyser, la_tracker,
                                   iteration)
      logging.info('train_acc %f', train_acc)

      # validation
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)

      # update the errors dictionary
      errors_dict['train_acc'].append(100 - train_acc)
      errors_dict['train_loss'].append(train_obj)
      errors_dict['valid_acc'].append(100 - valid_acc)
      errors_dict['valid_loss'].append(valid_obj)

      genotype = model.genotype()
      
      genotype_perf = api.predict(config=genotype, representation='genotype', with_noise=False)
      ops_count = count_ops(genotype)
      width = {k:genotype_width(g) for k, g in [("normal", genotype.normal), ("reduce", genotype.reduce)]}
      depth = {k:genotype_depth(g) for k, g in [("normal", genotype.normal), ("reduce", genotype.reduce)]}

      logging.info('genotype = %s', genotype)

      print(F.softmax(model.alphas_normal, dim=-1))
      print(F.softmax(model.alphas_reduce, dim=-1))

      state = {'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'alphas_normal': model.alphas_normal.data,
               'alphas_reduce': model.alphas_reduce.data,
               'arch_optimizer': architect.optimizer.state_dict(),
               'lr': lr,
               'ev': la_tracker.ev,
               'ev_local_avg': la_tracker.ev_local_avg,
               'genotypes': la_tracker.genotypes,
               'la_epochs': la_tracker.la_epochs,
               'la_start_idx': la_tracker.la_start_idx,
               'la_end_idx': la_tracker.la_end_idx,
               #'scheduler': scheduler.state_dict(),
              }
      wandb_log = {"train_acc":train_acc, "train_loss": train_obj, "val_acc": valid_acc, "valid_loss":valid_obj,
                  "search.final.cifar10": genotype_perf, "genotype":str(genotype),
                  "epoch":epoch, "ops":ops_count, "eigval": la_tracker.ev, "width":width, "depth":depth, "ev": ev}
      wandb.log(wandb_log)
      utils.save_checkpoint(state, False, args.save, epoch, args.task_id)

      if not args.compute_hessian:
        ev = -1
      else:
        ev = la_tracker.ev[-1]
      params = {'iteration': iteration,
                'epoch': epoch,
                'wd': args.weight_decay,
                'ev': ev,
               }

      schedule_of_params.append(params)

      # limit the number of iterations based on the maximum regularization
      # value predefined by the user
      final_iteration = round(
          np.log(args.max_weight_decay) / np.log(args.weight_decay), 1) == 1.

      if la_tracker.stop_search and not final_iteration:
        if args.early_stop == 1:
          # set the following to the values they had at stop_epoch
          errors_dict['valid_acc'] = errors_dict['valid_acc'][:la_tracker.stop_epoch + 1]
          genotype = la_tracker.stop_genotype
          valid_acc = 100 - errors_dict['valid_acc'][la_tracker.stop_epoch]
          logging.info(
              'Decided to stop the search at epoch %d (Current epoch: %d)',
              la_tracker.stop_epoch, epoch
          )
          logging.info(
              'Validation accuracy at stop epoch: %f', valid_acc
          )
          logging.info(
              'Genotype at stop epoch: %s', genotype
          )
          break

        elif args.early_stop == 2:
          # simulate early stopping and continue search afterwards
          simulated_errors_dict = errors_dict['valid_acc'][:la_tracker.stop_epoch + 1]
          simulated_genotype = la_tracker.stop_genotype
          simulated_valid_acc = 100 - simulated_errors_dict[la_tracker.stop_epoch]
          logging.info(
              '(SIM) Decided to stop the search at epoch %d (Current epoch: %d)',
              la_tracker.stop_epoch, epoch
          )
          logging.info(
              '(SIM) Validation accuracy at stop epoch: %f', simulated_valid_acc
          )
          logging.info(
              '(SIM) Genotype at stop epoch: %s', simulated_genotype
          )

          with open(os.path.join(args.save,
                                 'arch_early_{}'.format(args.task_id)),
                    'w') as file:
            file.write(str(simulated_genotype))

          utils.write_yaml_results(args, 'early_'+args.results_file_arch,
                                   str(simulated_genotype))
          utils.write_yaml_results(args, 'early_stop_epochs',
                                   la_tracker.stop_epoch)

          args.early_stop = 0

        elif args.early_stop == 3:
          # adjust regularization
          simulated_errors_dict = errors_dict['valid_acc'][:la_tracker.stop_epoch + 1]
          simulated_genotype = la_tracker.stop_genotype
          simulated_valid_acc = 100 - simulated_errors_dict[la_tracker.stop_epoch]
          stop_epoch = la_tracker.stop_epoch
          start_again_epoch = stop_epoch - args.extra_rollback_epochs
          logging.info(
              '(ADA) Decided to increase regularization at epoch %d (Current epoch: %d)',
              stop_epoch, epoch
          )
          logging.info(
              '(ADA) Rolling back to epoch %d', start_again_epoch
          )
          logging.info(
              '(ADA) Restoring model parameters and continuing for %d epochs',
              epochs_to_train - start_again_epoch - 1
          )

          if iteration == 1:
              logging.info(
                  '(ADA) Saving the architecture at the early stop epoch and '
                  'continuing with the adaptive regularization strategy'
              )
              utils.write_yaml_results(args, 'early_'+args.results_file_arch,
                                       str(simulated_genotype))

          del model
          del architect
          del optimizer
          del scheduler
          del analyser

          model_new = Network(args.init_channels, args.n_classes, args.layers, criterion,
                          primitives, steps=args.nodes)
          model_new = model_new.cuda()

          optimizer_new = torch.optim.SGD(
              model_new.parameters(),
              args.learning_rate,
              momentum=args.momentum,
              weight_decay=args.weight_decay)

          architect_new = Architect(model_new, args)

          analyser_new = Analyzer(args, model_new)

          la_tracker = utils.EVLocalAvg(args.window, args.report_freq_hessian,
                                        args.epochs)

          lr = utils.load_checkpoint(model_new, optimizer_new, None,
                                     architect_new, args.save, la_tracker,
                                     start_again_epoch, args.task_id)

          args.weight_decay *= args.mul_factor
          for param_group in optimizer_new.param_groups:
              param_group['weight_decay'] = args.weight_decay

          scheduler_new = CosineAnnealingLR(
              optimizer_new, float(args.epochs),
              eta_min=args.learning_rate_min)


          logging.info(
              '(ADA) Validation accuracy at stop epoch: %f', simulated_valid_acc
          )
          logging.info(
              '(ADA) Genotype at stop epoch: %s', simulated_genotype
          )

          logging.info(
              '(ADA) Adjusting L2 regularization to the new value: %f',
              args.weight_decay
          )

          genotype, valid_acc = train_epochs(args.epochs,
                                             iteration + 1, model=model_new,
                                             optimizer=optimizer_new,
                                             architect=architect_new,
                                             scheduler=scheduler_new,
                                             analyser=analyser_new,
                                             start_epoch=start_again_epoch)
          args.early_stop = 0
          break

    return genotype, valid_acc

  # call train_epochs recursively
  genotype, valid_acc = train_epochs(args.epochs, 1)

  with codecs.open(os.path.join(args.save,
                                'errors_{}.json'.format(args.task_id)),
                   'w', encoding='utf-8') as file:
    json.dump(errors_dict, file, separators=(',', ':'))

  with open(os.path.join(args.save,
                         'arch_{}'.format(args.task_id)),
            'w') as file:
    file.write(str(genotype))

  utils.write_yaml_results(args, args.results_file_arch, str(genotype))
  utils.write_yaml_results(args, args.results_file_perf, 100-valid_acc)

  with open(os.path.join(args.save,
                         'schedule_{}.pickle'.format(args.task_id)),
            'ab') as file:
    pickle.dump(schedule_of_params, file, pickle.HIGHEST_PROTOCOL)


def train(epoch, primitives, train_queue, valid_queue, model, architect,
          criterion, optimizer, lr, analyser, local_avg_tracker, iteration=1):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    if architect is not None:
      # get a random minibatch from the search queue with replacement
      input_search, target_search = next(iter(valid_queue))
      input_search = Variable(input_search, requires_grad=False).cuda()
      target_search = Variable(target_search, requires_grad=False).cuda()

      architect.step(input, target, input_search, target_search, lr, optimizer,
                     unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      if args.debug:
        break

  if args.compute_hessian:
    if (epoch % args.report_freq_hessian == 0) or (epoch == (args.epochs - 1)):
      _data_loader = deepcopy(train_queue)
      input, target = next(iter(_data_loader))

      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda()

      # get gradient information
      #param_grads = [p.grad for p in model.parameters() if p.grad is not None]
      #param_grads = torch.cat([x.view(-1) for x in param_grads])
      #param_grads = param_grads.cpu().data.numpy()
      #grad_norm = np.linalg.norm(param_grads)

      #gradient_vector = torch.cat([x.view(-1) for x in gradient_vector]) 
      #grad_norm = LA.norm(gradient_vector.cpu())
      #logging.info('\nCurrent grad norm based on Train Dataset: %.4f',
      #             grad_norm)

      if not args.debug:
        H = analyser.compute_Hw(input, target, input_search, target_search,
                                lr, optimizer, False)
        g = analyser.compute_dw(input, target, input_search, target_search,
                                lr, optimizer, False)
        g = torch.cat([x.view(-1) for x in g])

        del _data_loader

        state = {'epoch': epoch,
                 'H': H.cpu().data.numpy().tolist(),
                 'g': g.cpu().data.numpy().tolist(),
                 #'g_train': float(grad_norm),
                 #'eig_train': eigenvalue,
                }

        with codecs.open(os.path.join(args.save,
                                      'derivatives_{}.json'.format(args.task_id)),
                                      'a', encoding='utf-8') as file:
          json.dump(state, file, separators=(',', ':'))
          file.write('\n')

        # early stopping
        print(f"Hessian: {H}")
        ev = max(LA.eigvals(H.cpu().data.numpy()))
      else:
        ev = 0.1
        if epoch >= 8 and iteration==1:
          ev = 2.0
      logging.info('CURRENT EV: %f', ev)
      local_avg_tracker.update(epoch, ev, model.genotype())

      if args.early_stop and epoch != (args.epochs - 1):
        local_avg_tracker.early_stop(epoch, args.factor, args.es_start_epoch,
                                     args.delta)

  return top1.avg, objs.avg, ev


def train_higher(epoch, primitives, train_queue, valid_queue, network, architect,
          criterion, w_optimizer, a_optimizer, lr, analyser, local_avg_tracker, iteration=1, logger=None, inner_steps=100, steps_per_epoch=None, warm_start=None, args=None):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  train_iter = iter(train_queue)
  valid_iter = iter(valid_queue)
  search_loader_iter = zip(train_iter, valid_iter)
  for data_step, ((base_inputs, base_targets), (arch_inputs, arch_targets)) in tqdm(enumerate(search_loader_iter), total = round(len(train_queue)/inner_steps)):
    if steps_per_epoch is not None and data_step >= steps_per_epoch:
      break
    network.train()
    n = base_inputs.size(0)

    base_inputs = base_inputs.cuda()
    base_targets = base_targets.cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda(non_blocking=True)
    
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, 
                                                                                              search_loader_iter, inner_steps=inner_steps, epoch=epoch, args=args)

    network.zero_grad()

    model_init = deepcopy(network.state_dict())
    w_optim_init = deepcopy(w_optimizer.state_dict())
    arch_grads = [torch.zeros_like(p) for p in network.arch_parameters()]

    for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
        if data_step in [0, 1] and inner_step < 3:
            print(f"Base targets in the inner loop at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}")
        logits = network(base_inputs)
        base_loss = criterion(logits, base_targets)
        base_loss.backward()
        w_optimizer.step()
        w_optimizer.zero_grad()
        if args.higher_method in ["val_multiple", "val"]:
            # if data_step < 2 and epoch < 1:
            #   print(f"Arch grads during unrolling from last step: {arch_grads}")
            logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)
            arch_loss.backward()
            with torch.no_grad():
                for g1, g2 in zip(arch_grads, network.arch_parameters()):
                    g1.add_(g2)
            
            network.zero_grad()
            a_optimizer.zero_grad()
            w_optimizer.zero_grad()
            # if data_step < 2 and epoch < 1:
            #   print(f"Arch grads during unrolling: {arch_grads}")
            
    if args.higher_method in ["val_multiple", "val"]:
      print(f"Arch grads after unrolling: {arch_grads}")
      with torch.no_grad():
        for g, p in zip(arch_grads, network.arch_parameters()):
          p.grad = g
          

    
    if warm_start is None or (warm_start is not None and epoch >= warm_start):
      a_optimizer.step()
      a_optimizer.zero_grad()
      
      w_optimizer.zero_grad()
      architect.optimizer.zero_grad()
      # Restore original model state before unrolling and put in the new arch parameters
      new_arch = deepcopy(network._arch_parameters)
      network.load_state_dict(model_init)
      network.alphas_normal.data = new_arch[0].data
      network.alphas_reduce.data = new_arch[1].data
      
      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
          if data_step in [0, 1] and inner_step < 3 and epoch % 5 == 0:
              logger.info(f"Doing weight training for real in higher_loop={args.higher_loop} at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}")
          logits = network(base_inputs)
          base_loss = criterion(logits, base_targets)
          network.zero_grad()
          base_loss.backward()
          w_optimizer.step()
          n = base_inputs.size(0)

          prec1, prec5 = utils.accuracy(logits, base_targets, topk=(1, 5))

          objs.update(base_loss.item(), n)
          top1.update(prec1.data, n)
          top5.update(prec5.data, n)

      if data_step % args.report_freq == 0:
          logging.info('train %03d %e %f %f', data_step, objs.avg, top1.avg, top5.avg)
      if 'debug' in args.save:
          break
    else:
      a_optimizer.zero_grad()
      w_optimizer.zero_grad()

  if args.compute_hessian:
    if (epoch % args.report_freq_hessian == 0) or (epoch == (args.epochs - 1)):
      _data_loader = deepcopy(train_queue)
      input, target = next(iter(_data_loader))

      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda()

      # get gradient information
      #param_grads = [p.grad for p in model.parameters() if p.grad is not None]
      #param_grads = torch.cat([x.view(-1) for x in param_grads])
      #param_grads = param_grads.cpu().data.numpy()
      #grad_norm = np.linalg.norm(param_grads)

      #gradient_vector = torch.cat([x.view(-1) for x in gradient_vector]) 
      #grad_norm = LA.norm(gradient_vector.cpu())
      #logging.info('\nCurrent grad norm based on Train Dataset: %.4f',
      #             grad_norm)

      if not args.debug:
        H = analyser.compute_Hw(input, target, input_search, target_search,
                                lr, w_optimizer, False)
        g = analyser.compute_dw(input, target, input_search, target_search,
                                lr, w_optimizer, False)
        g = torch.cat([x.view(-1) for x in g])

        del _data_loader

        state = {'epoch': epoch,
                'H': H.cpu().data.numpy().tolist(),
                'g': g.cpu().data.numpy().tolist(),
                #'g_train': float(grad_norm),
                #'eig_train': eigenvalue,
                }

        with codecs.open(os.path.join(args.save,
                                      'derivatives_{}.json'.format(args.task_id)),
                                      'a', encoding='utf-8') as file:
          json.dump(state, file, separators=(',', ':'))
          file.write('\n')

        # early stopping
        print(f"Hessian: {H}")
        ev = max(LA.eigvals(H.cpu().data.numpy()))
      else:
        ev = 0.1
        if epoch >= 8 and iteration==1:
          ev = 2.0
      logging.info('CURRENT EV: %f', ev)
      local_avg_tracker.update(epoch, ev, network.genotype())

      if args.early_stop and epoch != (args.epochs - 1):
        local_avg_tracker.early_stop(epoch, args.factor, args.es_start_epoch,
                                    args.delta)

  return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input).cuda()
      target = Variable(target).cuda()

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        if args.debug:
          break

  return top1.avg, objs.avg


if __name__ == '__main__':
  space = spaces_dict[args.space]
  main(space)

