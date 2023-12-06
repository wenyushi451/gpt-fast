python scripts/download.py --repo_id $1 && python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$1 && python quantize.py --mode int8 --checkpoint_path checkpoints/$1/model.pth
