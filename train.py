import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent


def parser_common(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--runtime', type=Path, required=True,
                        help='Complete detection project containing src/core and src/solver')
    parser.add_argument('--recipe', type=Path, required=True,
                        help='Runtime YAML: dataset paths, transforms, losses and training settings')
    parser.add_argument('--structure', type=Path,
                        default=ROOT / 'configs/dcr_wave_slim_structure.json')
    parser.add_argument('--device', default='cuda', help='cpu, cuda, or cuda:N')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/slim')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--eval-batch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def import_runtime(path):
    runtime = path.expanduser().resolve()
    if not (runtime / 'src/core/yaml_config.py').is_file():
        raise ValueError('--runtime must contain src/core/yaml_config.py; the structure package alone is insufficient.')
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(ROOT))
    import src
    if not Path(src.__file__).resolve().is_relative_to(runtime):
        raise ValueError('A different src package is already imported; start a fresh process.')
    return runtime


def setup_device(requested):
    import torch
    import torch.distributed as distributed
    device = torch.device(requested)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu for a CPU check.')
        if world_size > 1:
            device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
    if world_size > 1:
        distributed.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    return device


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args, device):
    from src.core import YAMLConfig, create, merge_config
    from src.zoo.rtdetr.rtdetr import DETR
    from dcr_wave import DCRWaveBackbone

    if not Path(sys.modules[DCRWaveBackbone.__module__].__file__).resolve().is_relative_to(ROOT / 'dcr_wave'):
        raise ValueError('The imported backbone is not the one beside this script.')
    structure = json.loads(args.structure.expanduser().resolve().read_text(encoding='utf-8'))
    cfg = YAMLConfig(str(args.recipe.expanduser().resolve()), resume='', tuning='',
                     output_dir=str(args.output.expanduser().resolve()))
    cfg.device = device
    if cfg.yaml_cfg.get('task') != 'detection':
        raise ValueError('Only image-based COCO detection recipes are supported.')
    backbone_config = copy.deepcopy(structure['backbone'])
    if backbone_config.get('infer_mode', False):
        raise ValueError('This wave implementation requires infer_mode=false, including during evaluation.')
    classes = cfg.yaml_cfg.get('num_classes')
    if classes != backbone_config['num_classes']:
        raise ValueError(f'Class count mismatch: recipe={classes}, structure={backbone_config["num_classes"]}.')
    decoder_config = {k: v for k, v in structure['decoder'].items() if k != 'implementation'}
    if decoder_config['feat_channels'] != [backbone_config['dims'][-1]]:
        raise ValueError('Decoder input channels do not match the final backbone stage.')
    cfg.yaml_cfg.setdefault('RTDETRTransformer', {}).update(decoder_config)
    # Keep encoder.* parameter names for the recipe's optimizer groups and checkpoints.
    cfg.yaml_cfg['model'] = 'DETR'
    cfg.yaml_cfg['DETR'] = {'encoder': 'DCRWaveBackbone', 'decoder': 'RTDETRTransformer'}
    cfg.yaml_cfg['DCRWaveBackbone'] = backbone_config
    for section in ['train_dataloader', 'val_dataloader']:
        loader = cfg.yaml_cfg.get(section)
        if loader is None:
            if section == 'train_dataloader':
                continue
            raise ValueError('The recipe must define val_dataloader.')
        if args.workers is not None:
            if args.workers < 0:
                raise ValueError('--workers must be nonnegative.')
            loader['num_workers'] = args.workers
        dataset = loader['dataset']
        for key in ['img_folder', 'ann_file']:
            if key in dataset:
                path = Path(dataset[key]).expanduser().resolve()
                if not path.exists():
                    raise ValueError(f'{section}.{key} not found: {path}')
                dataset[key] = str(path)
    for section, value in [('train_dataloader', getattr(args, 'batch_size', None)),
                           ('val_dataloader', args.eval_batch_size)]:
        if value is not None:
            if value < 1:
                raise ValueError('Batch sizes must be positive.')
            cfg.yaml_cfg[section]['batch_size'] = value
    if cfg.yaml_cfg['val_dataloader'].get('shuffle', False) or cfg.yaml_cfg['val_dataloader'].get('drop_last', False):
        raise ValueError('Validation requires shuffle=false and drop_last=false.')
    if getattr(args, 'epochs', None) is not None:
        if args.epochs < 1:
            raise ValueError('--epochs must be positive.')
        cfg.epoches = args.epochs
        cfg.yaml_cfg['epoches'] = args.epochs
    merge_config(cfg.yaml_cfg)
    decoder = create('RTDETRTransformer')
    cfg._model = DETR(encoder=DCRWaveBackbone(**backbone_config), decoder=decoder)
    return cfg, structure


def make_loader(loader, shuffle):
    import torch.distributed as distributed
    from torch.utils.data import DataLoader, DistributedSampler
    if not distributed.is_initialized():
        return loader
    sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
    return DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=sampler,
                      num_workers=loader.num_workers, pin_memory=loader.pin_memory,
                      collate_fn=loader.collate_fn, drop_last=loader.drop_last)


def atomic_save(state, path):
    import torch
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def check_batch(cfg, batch_size):
    import torch
    from torch.utils.data import DataLoader
    loader = cfg.train_dataloader
    sample_loader = DataLoader(loader.dataset, batch_size=batch_size, shuffle=False,
                               num_workers=0, collate_fn=loader.collate_fn)
    images, targets = next(iter(sample_loader))
    images = images.to(cfg.device)
    targets = [{k: v.to(cfg.device) for k, v in target.items()} for target in targets]
    model, criterion = cfg.model.to(cfg.device).train(), cfg.criterion.to(cfg.device).train()
    loss_values = criterion(model(images, targets), targets)
    loss = sum(loss_values.values())
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite loss in preflight.')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError('Missing or nonfinite gradients in preflight.')
    model.eval()
    with torch.no_grad():
        prediction = model(images)
        sizes = torch.stack([t['orig_size'] for t in targets])
        detections = cfg.postprocessor(prediction, sizes)
    print(json.dumps({'check': 'passed', 'input_shape': list(images.shape),
                      'pred_logits_shape': list(prediction['pred_logits'].shape),
                      'loss': float(loss.detach()), 'loss_terms': list(loss_values),
                      'detections_per_image': [len(d['scores']) for d in detections],
                      'optimizer_steps': 0, 'backbone': type(model.encoder).__name__}))


def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])


def fit(cfg, structure, args):
    import torch
    import torch.distributed as distributed
    from src.data import get_coco_api_from_dataset
    from src.misc import dist
    from src.solver.det_engine import train_one_epoch, evaluate

    if cfg.epoches < 1:
        raise ValueError('Set a positive epoches value in the recipe or use --epochs.')
    output = Path(cfg.output_dir)
    if not args.resume and any((output / name).exists() for name in ['last.pth', 'best.pth', 'log.txt']):
        raise ValueError('Output already contains a run. Use --resume or a new --output.')
    output.mkdir(parents=True, exist_ok=True)
    module = cfg.model.to(cfg.device)
    if distributed.is_initialized() and cfg.sync_bn:
        if cfg.device.type != 'cuda':
            raise ValueError('The recipe requests SyncBatchNorm; distributed CPU training is unsupported.')
        module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        cfg._model = module
    criterion = cfg.criterion.to(cfg.device)
    optimizer, scheduler = cfg.optimizer, cfg.lr_scheduler
    ema = cfg.ema.to(cfg.device) if cfg.use_ema else None
    if cfg.use_amp and cfg.device.type != 'cuda':
        raise ValueError('AMP in this runtime requires CUDA; use an FP32 recipe for CPU checks.')
    scaler = cfg.scaler if cfg.use_amp else None
    start_epoch, best_map = 0, -math.inf
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key in ['model', 'optimizer', 'lr_scheduler', 'last_epoch', 'best_map', 'structure', 'rng_states']:
            if key not in resume_state:
                raise ValueError(f'Resume requires a checkpoint produced by train.py; missing {key}.')
        if resume_state['structure'] != structure:
            raise ValueError('Checkpoint architecture differs from --structure.')
        module.load_state_dict(resume_state['model'], strict=True)
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['lr_scheduler'])
        if ema is not None:
            ema.load_state_dict(resume_state['ema'])
        if scaler is not None:
            scaler.load_state_dict(resume_state['scaler'])
        start_epoch = int(resume_state['last_epoch']) + 1
        best_map = float(resume_state['best_map'])
    model = module
    if distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            module, device_ids=[cfg.device.index] if cfg.device.type == 'cuda' else None,
            find_unused_parameters=bool(cfg.find_unused_parameters))
    train_loader = make_loader(cfg.train_dataloader, cfg.train_dataloader.shuffle)
    val_loader = make_loader(cfg.val_dataloader, False)
    base_ds = get_coco_api_from_dataset(val_loader.dataset)
    if resume_state:
        states = resume_state['rng_states']
        if len(states) != dist.get_world_size():
            raise ValueError('Exact resume requires the same number of workers/ranks.')
        restore_rng(states[dist.get_rank()])
    if dist.is_main_process():
        (output / 'resolved_config.json').write_text(json.dumps(cfg.yaml_cfg, indent=2, default=str))
        print(f'Total epochs: {cfg.epoches}; starting epoch: {start_epoch + 1}; output: {output}')
    for epoch in range(start_epoch, cfg.epoches):
        if distributed.is_initialized():
            train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, cfg.device, epoch,
                                      cfg.clip_max_norm, print_freq=cfg.log_step, ema=ema, scaler=scaler)
        scheduler.step()
        stats, _ = evaluate(ema.module if ema is not None else model, criterion,
                            cfg.postprocessor, val_loader, base_ds, cfg.device, output)
        metrics = stats['coco_eval_bbox']
        if not math.isfinite(metrics[0]) or metrics[0] < 0:
            raise ValueError('COCO evaluation did not produce a valid bbox AP.')
        improved = metrics[0] > best_map
        best_map = max(best_map, metrics[0])
        rng_states = dist.all_gather(rng_state())
        if dist.is_main_process():
            state = {'model': module.state_dict(), 'optimizer': optimizer.state_dict(),
                     'lr_scheduler': scheduler.state_dict(), 'last_epoch': epoch,
                     'best_map': best_map, 'structure': structure, 'rng_states': rng_states}
            if ema is not None:
                state['ema'] = ema.state_dict()
            if scaler is not None:
                state['scaler'] = scaler.state_dict()
            atomic_save(state, output / 'last.pth')
            if improved:
                atomic_save(state, output / 'best.pth')
            row = {'epoch': epoch + 1, 'train': train_stats, 'coco_eval_bbox': metrics,
                   'mAP50_95_percent': 100 * metrics[0], 'AP50_percent': 100 * metrics[1],
                   'AP75_percent': 100 * metrics[2], 'best_mAP_percent': 100 * best_map}
            with (output / 'log.txt').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
            with (output / 'metrics.log').open('a', encoding='utf-8') as handle:
                if handle.tell() == 0:
                    handle.write('epoch mAP50:95(%) AP50(%) AP75(%) best_mAP(%)\n')
                handle.write(f'{epoch + 1} {100*metrics[0]:.3f} {100*metrics[1]:.3f} '
                             f'{100*metrics[2]:.3f} {100*best_map:.3f}\n')


def main():
    parser = parser_common('Train DCR-Wave using this repository backbone and an external detection runtime.')
    parser.add_argument('--batch-size', type=int, default=None, help='Training batch per process; otherwise use recipe')
    parser.add_argument('--epochs', type=int, default=None, help='Total epochs, including resumed epochs')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--check', action='store_true', help='One real-data forward/backward check; no optimizer update')
    parser.add_argument('--check-batch-size', type=int, default=1)
    args = parser.parse_args()
    try:
        import_runtime(args.runtime)
        device = setup_device(args.device)
        seed_everything(args.seed)
        cfg, structure = build_config(args, device)
        if args.check:
            if int(os.environ.get('WORLD_SIZE', '1')) != 1 or args.check_batch_size < 1:
                raise ValueError('--check requires a single process and positive --check-batch-size.')
            check_batch(cfg, args.check_batch_size)
        else:
            fit(cfg, structure, args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    finally:
        if 'torch.distributed' in sys.modules:
            import torch.distributed as distributed
            if distributed.is_initialized():
                distributed.destroy_process_group()


if __name__ == '__main__':
    main()
