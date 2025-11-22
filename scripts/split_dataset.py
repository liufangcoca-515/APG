#!/usr/bin/env python3
"""
scripts/split_dataset.py

按基准（子目录名或文件基名）将 `image/` 和 `mask/` 划分为 Training / Test

用法示例:
  python scripts/split_dataset.py data/jpg/CC-Mask --test-ratio 0.2 --dry-run

支持两种输入结构：
 1) image/ 下有若干子目录（每个 case 为一个目录），此时以子目录为样本单位
 2) image/ 下是平铺的文件（每个文件为样本），此时以文件基名为样本单位

可选参数：--move（移动代替复制），--dry-run（仅打印操作不执行），--seed
"""
import argparse
import os
import random
import shutil
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('base', help='base folder, e.g. data/jpg/CC-Mask')
    p.add_argument('--test-ratio', type=float, default=0.2)
    p.add_argument('--move', action='store_true', help='move files instead of copy')
    p.add_argument('--dry-run', action='store_true', help='do not perform file operations')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def ensure_dir(path, dry_run=False):
    if dry_run:
        print(f"DRYRUN mkdir -p {path}")
    else:
        os.makedirs(path, exist_ok=True)


def copy_move(src, dst, move=False, dry_run=False):
    if dry_run:
        print(f"DRYRUN {'mv' if move else 'cp'} {src} -> {dst}")
        return
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    base = args.base
    img_dir = os.path.join(base, 'image')
    msk_dir = os.path.join(base, 'mask')

    if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
        print('Error: image or mask dir not found:', img_dir, msk_dir)
        sys.exit(1)

    # detect if image dir contains subdirectories
    entries = sorted(os.listdir(img_dir))
    subdirs = [d for d in entries if os.path.isdir(os.path.join(img_dir, d))]
    if subdirs:
        mode = 'dir'
        items = subdirs
    else:
        # flat files: use basenames
        files = [f for f in entries if os.path.isfile(os.path.join(img_dir, f))]
        if not files:
            print('No images found in', img_dir)
            sys.exit(1)
        items = [os.path.splitext(f)[0] for f in files]
        mode = 'file'

    random.seed(args.seed)
    random.shuffle(items)

    cut = int(len(items) * (1.0 - args.test_ratio))
    train = items[:cut]
    test = items[cut:]

    train_dir = os.path.join(base, 'Training')
    test_dir = os.path.join(base, 'Test')

    # create target dirs
    for d in [os.path.join(train_dir, 'image'), os.path.join(train_dir, 'mask'),
              os.path.join(test_dir, 'image'), os.path.join(test_dir, 'mask')]:
        ensure_dir(d, dry_run=args.dry_run)

    def place(list_items, target_dir):
        for it in list_items:
            if mode == 'dir':
                img_src_dir = os.path.join(img_dir, it)
                msk_src_dir = os.path.join(msk_dir, it)
                if not os.path.isdir(msk_src_dir):
                    print('Warning: mask dir missing for', it)
                    continue
                tgt_img_dir = os.path.join(target_dir, 'image', it)
                tgt_msk_dir = os.path.join(target_dir, 'mask', it)
                ensure_dir(tgt_img_dir, dry_run=args.dry_run)
                ensure_dir(tgt_msk_dir, dry_run=args.dry_run)
                for f in sorted(os.listdir(img_src_dir)):
                    s = os.path.join(img_src_dir, f)
                    if not os.path.exists(s):
                        continue
                    d0 = os.path.join(tgt_img_dir, f)
                    copy_move(s, d0, move=args.move, dry_run=args.dry_run)
                for f in sorted(os.listdir(msk_src_dir)):
                    s = os.path.join(msk_src_dir, f)
                    if not os.path.exists(s):
                        continue
                    d0 = os.path.join(tgt_msk_dir, f)
                    copy_move(s, d0, move=args.move, dry_run=args.dry_run)
            else:
                # flat file mode: find file by basename in IMG_DIR and MSK_DIR
                cand_img = next((x for x in os.listdir(img_dir) if os.path.splitext(x)[0] == it), None)
                if cand_img is None:
                    print('Warning: image for', it, 'not found')
                    continue
                cand_msk = next((x for x in os.listdir(msk_dir) if os.path.splitext(x)[0] == it), None)
                if cand_msk is None:
                    print('Warning: mask for', it, 'not found')
                    continue
                src_img = os.path.join(img_dir, cand_img)
                src_msk = os.path.join(msk_dir, cand_msk)
                dst_img = os.path.join(target_dir, 'image', os.path.basename(src_img))
                dst_msk = os.path.join(target_dir, 'mask', os.path.basename(src_msk))
                copy_move(src_img, dst_img, move=args.move, dry_run=args.dry_run)
                copy_move(src_msk, dst_msk, move=args.move, dry_run=args.dry_run)

    place(train, train_dir)
    place(test, test_dir)

    # report counts
    def count_files(root):
        total = 0
        for _, _, files in os.walk(root):
            total += len(files)
        return total

    print(f"Done. Training images: {count_files(os.path.join(train_dir,'image'))}  Test images: {count_files(os.path.join(test_dir,'image'))}")


if __name__ == '__main__':
    main()
