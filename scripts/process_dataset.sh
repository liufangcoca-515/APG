#!/usr/bin/env bash
set -euo pipefail

# process_dataset.sh
# 使用说明
# 用法:
#   ./scripts/process_dataset.sh /path/to/dataset_root [--test-ratio 0.2] [--move]
# 示例:
#   ./scripts/process_dataset.sh /home/liufang882/YOLO_SAM/APG/CC-Mask --test-ratio 0.2
#
# 处理流程：
# 1) 运行 `pre_CTMR.py`：将 NIfTI 原始数据预处理并导出为切片 NPY（可通过参数指定输入/输出路径）
# 2) 运行 `npy_to_jpg.py --process_structure`：把 NPY 图像转换为 JPG，同时复制 gts（标签）
# 3) 将 `imgs` 重命名为 `image`，`gts` 重命名为 `mask`（统一命名以便后续处理）
# 4) 去除文件名前的前导零（例如 001.jpg -> 1.jpg），避免文件名排序/匹配问题
# 5) 将数据按基准（文件基名或子目录名）随机划分为 `Training` 和 `Test`，每个目录包含 `image/` 与 `mask/`
#


## 第一个位置参数为数据集根目录 DATA_ROOT
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/dataset_root [--test-ratio FLOAT] [--move] [--dry-run]"
  exit 1
fi

DATA_ROOT="$1"
shift

# 默认配置（可通过命令行选项覆盖）
## TEST_RATIO: 测试集比例；MOVE_FILES: 是否移动（1）而不是复制（0）；DRY_RUN: 仅打印不执行
## PYTHON: 使用的 Python 解释器，默认为 python3 或环境变量 $PYTHON
# defaults
TEST_RATIO=0.2
MOVE_FILES=0
DRY_RUN=0
PYTHON=${PYTHON:-python3}

while (( "$#" )); do
  case "$1" in
    --test-ratio)
      TEST_RATIO="$2"; shift 2;;
    --move)
      MOVE_FILES=1; shift;;
    --dry-run)
      DRY_RUN=1; shift;;
    --python)
      PYTHON="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 /path/to/dataset_root [--test-ratio FLOAT] [--move] [--dry-run] [--python python3]"; exit 0;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "Dataset root: $DATA_ROOT"
echo "Test ratio: $TEST_RATIO"
echo "Move files: $MOVE_FILES"
echo "Dry run: $DRY_RUN"
echo "Python: $PYTHON"

# 推导输出目录：以 DATA_ROOT 的 basename 作为 dataset 名称
# NPY_BASE: pre_CTMR 输出的 NPY 路径；JPG_BASE: npy_to_jpg 输出路径
BASE_NAME=$(basename "$DATA_ROOT")
NPY_BASE="$PWD/data/npy/CCMASK_test/${BASE_NAME}"
JPG_BASE="$PWD/data/jpg/${BASE_NAME}"

## 步骤 1: 运行 pre_CTMR.py（将 nii -> npy 切片）
echo "\n==> Step 1: run pre_CTMR.py with paths for this dataset"
## pre_CTMR.py 会读取 --nii_path（体数据）和 --gt_path（标签），并把切片存到 --npy_path
if [ $DRY_RUN -eq 0 ]; then
  # 将 DATA_ROOT 下的 imageTr/ 和 label/ 作为默认输入子目录
  $PYTHON pre_CTMR.py --nii_path "$DATA_ROOT/imageTr" --gt_path "$DATA_ROOT/label" --npy_path "$NPY_BASE" --modality "" --anatomy "$BASE_NAME" || { echo "pre_CTMR.py failed"; exit 1; }
else
  echo "Dry run: would run: $PYTHON pre_CTMR.py --nii_path $DATA_ROOT/imageTr --gt_path $DATA_ROOT/label --npy_path $NPY_BASE --anatomy $BASE_NAME"
fi

## 步骤 2: 使用 npy_to_jpg.py 将 npy 图像转换为 jpg，并复制 gts 标签
echo "\n==> Step 2: run npy_to_jpg.py to convert npy -> jpg and copy gts"
mkdir -p "$JPG_BASE"
if [ $DRY_RUN -eq 0 ]; then
  $PYTHON npy_to_jpg.py --process_structure --npy_folder "$NPY_BASE" --output_folder "$JPG_BASE"
else
  echo "Dry run: would run: $PYTHON npy_to_jpg.py --process_structure --npy_folder $NPY_BASE --output_folder $JPG_BASE"
fi

## 步骤 3: 重命名目录以统一命名（imgs -> image, gts -> mask）
echo "\n==> Step 3: rename dirs imgs->image and gts->mask under $JPG_BASE"
if [ -d "$JPG_BASE/imgs" ]; then
  echo "Found $JPG_BASE/imgs"
  if [ $DRY_RUN -eq 0 ]; then
    mv "$JPG_BASE/imgs" "$JPG_BASE/image"
  else
    echo "Dry run: mv $JPG_BASE/imgs $JPG_BASE/image"
  fi
else
  echo "Warning: $JPG_BASE/imgs not found"
fi

if [ -d "$JPG_BASE/gts" ]; then
  echo "Found $JPG_BASE/gts"
  if [ $DRY_RUN -eq 0 ]; then
    mv "$JPG_BASE/gts" "$JPG_BASE/mask"
  else
    echo "Dry run: mv $JPG_BASE/gts $JPG_BASE/mask"
  fi
else
  echo "Warning: $JPG_BASE/gts not found"
fi

# Step 4: run a small Python snippet to remove leading zeros from filenames inside image and mask
echo "\n==> Step 4: remove leading zeros in filenames under $JPG_BASE"
REMOVE_PY=$(cat <<'PY'
import os
import sys

def remove_leading_zeros(filename):
    name, ext = os.path.splitext(filename)
    if name and name[0].isdigit():
        try:
            name2 = str(int(name))
        except Exception:
            name2 = name.lstrip('0') or name
        return name2 + ext
    return filename

root = sys.argv[1]
for dirpath, dirs, files in os.walk(root):
    for f in files:
        old = os.path.join(dirpath, f)
        newname = remove_leading_zeros(f)
        new = os.path.join(dirpath, newname)
        if old != new:
            try:
                os.rename(old, new)
                print(f"Renamed {old} -> {new}")
            except Exception as e:
                print(f"Failed rename {old} -> {new}: {e}")
PY
)

## 步骤 4: 去前导零（处理文件名，例如 001.jpg -> 1.jpg）
if [ $DRY_RUN -eq 0 ]; then
  echo "$REMOVE_PY" | $PYTHON - "$JPG_BASE"
else
  echo "Dry run: would run python snippet to remove leading zeros under $JPG_BASE"
fi

## 步骤 5: 使用独立的 Python 脚本进行划分（支持子目录或平铺文件结构）
echo "\n==> Step 5: split into Test and Training under $JPG_BASE using scripts/split_dataset.py"
SPLIT_CMD=("$PYTHON" "$(dirname "$0")/split_dataset.py" "$JPG_BASE" "--test-ratio" "$TEST_RATIO" "--seed" "42")
if [ $MOVE_FILES -eq 1 ]; then
  SPLIT_CMD+=("--move")
fi
if [ $DRY_RUN -eq 1 ]; then
  SPLIT_CMD+=("--dry-run")
fi

echo "Running: ${SPLIT_CMD[*]}"
if [ $DRY_RUN -eq 0 ]; then
  "${SPLIT_CMD[@]}" || { echo "split_dataset.py failed"; exit 1; }
else
  echo "Dry run: ${SPLIT_CMD[*]}"
fi

echo "\nDone. Created:"
ls -l "$JPG_BASE" || true

echo "\nSummary:"
echo "Training: $(find "$JPG_BASE/Training/image" -type f | wc -l) images"
echo "Test:     $(find "$JPG_BASE/Test/image" -type f | wc -l) images"

echo "Script finished."
