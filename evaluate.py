import json
import sys
from pathlib import Path

from train import parser_common, import_runtime, setup_device, seed_everything, build_config, make_loader


def main():
    parser = parser_common('Evaluate DCR-Wave with the runtime COCO bbox evaluator.')
    parser.set_defaults(output=Path(__file__).resolve().parent / 'output/evaluation')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--weights', choices=['auto', 'ema', 'model'], default='auto')
    parser.add_argument('--limit', type=int, default=None,
                        help='Evaluate only the first N images for debugging; not a full-dataset score')
    args = parser.parse_args()
    try:
        import_runtime(args.runtime)
        import torch
        from torch.utils.data import DataLoader, Subset
        from src.data import get_coco_api_from_dataset
        from src.misc import dist
        from src.solver.det_engine import evaluate

        device = setup_device(args.device)
        seed_everything(args.seed)
        cfg, structure = build_config(args, device)
        checkpoint_path = args.checkpoint.expanduser().resolve()
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'structure' in state and state['structure'] != structure:
            raise ValueError('Checkpoint architecture differs from --structure.')
        branch = args.weights
        if branch == 'auto':
            branch = 'ema' if 'ema' in state else 'model'
        if branch == 'ema':
            if 'ema' not in state or 'module' not in state['ema']:
                raise ValueError('EMA weights are absent from this checkpoint.')
            weights = state['ema']['module']
        else:
            if 'model' not in state:
                raise ValueError('Expected checkpoint["model"]; raw state_dict files are not supported.')
            weights = state['model']
        # Strict loading prevents a full-width model from being evaluated as slim.
        cfg.model.load_state_dict(weights, strict=True)
        model, criterion = cfg.model.to(device).eval(), cfg.criterion.to(device).eval()
        loader = cfg.val_dataloader
        dataset_size = len(loader.dataset)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError('--limit must be positive.')
            loader = DataLoader(Subset(loader.dataset, range(min(args.limit, dataset_size))),
                                batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers,
                                pin_memory=loader.pin_memory, collate_fn=loader.collate_fn, drop_last=False)
        evaluated_images = len(loader.dataset)
        base_ds = get_coco_api_from_dataset(loader.dataset)
        loader = make_loader(loader, False)
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        stats, evaluator = evaluate(model, criterion, cfg.postprocessor, loader, base_ds, device, output)
        if dist.is_main_process():
            ap = stats['coco_eval_bbox']
            result = {'checkpoint': str(checkpoint_path), 'weights': branch,
                      'evaluated_images': evaluated_images, 'dataset_images': dataset_size,
                      'partial_evaluation': evaluated_images < dataset_size,
                      'batch_size_per_process': loader.batch_size,
                      'world_size': dist.get_world_size(),
                      'mAP50_95_percent': 100*ap[0], 'AP50_percent': 100*ap[1],
                      'AP75_percent': 100*ap[2], 'coco_eval_bbox': ap}
            filename = 'metrics_subset.json' if result['partial_evaluation'] else 'metrics.json'
            (output / filename).write_text(json.dumps(result, indent=2), encoding='utf-8')
            torch.save(evaluator.coco_eval['bbox'].eval,
                       output / ('coco_eval_subset.pth' if result['partial_evaluation'] else 'coco_eval.pth'))
            print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    finally:
        if 'torch.distributed' in sys.modules:
            import torch.distributed as distributed
            if distributed.is_initialized():
                distributed.destroy_process_group()


if __name__ == '__main__':
    main()
