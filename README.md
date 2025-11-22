
本 README 说明如何在当前仓库中准备数据集、将 NIfTI/NPY 转为训练所需的 `image` / `mask` 目录结构，并如何启动训练与测试。

以下示例假定你在仓库根目录。

## 目录要点
- `pre_CTMR.py`：将原始 `.nii.gz` 图像与标签处理并保存为切片的 `.npy` 文件（之前实现含默认硬编码路径，已支持命令行参数）。
- `npy_to_jpg.py`：把切片 `.npy`（图像）转为 `.jpg`，并复制标签 `.npy`。支持 `--process_structure` 模式。
- `scripts/process_dataset.sh`：一键自动化流程（运行 `pre_CTMR.py`、`npy_to_jpg.py`、重命名、去前导零、按比例划分 Train/Test）。
- `format.py`：用于去掉文件名前导零（仓库内存在多个副本）；脚本内也集成了等效逻辑。
- `train_3d_apg.py`：训练主脚本（训练 3D APG 模型）。
- `test_model.py` / `test.py`：测试/验证相关脚本。

## 依赖
- 建议使用 `conda` 或 `venv` 创建干净环境。仓库内有 `environment.yml` / `requeriment.txt`，可据此安装依赖。

示例（conda）：

```bash
conda env create -f environment.yml -n apg-env
conda activate apg-env
# 或手动安装 pip 依赖
pip install -r requeriment.txt
```

注意：部分包（如 `SimpleITK`, `cc3d`, `scikit-image`, `tqdm`, `Pillow` 等）需要可用的 Python 环境。

## 数据预处理与分割

流程总览：
1) 准备源数据目录（包含 NIfTI 文件）。例如把数据放在：`./CC-Mask`，其中包含子目录 `imageTr/` 与 `label/`。
2) 运行 `pre_CTMR.py`：将 nii 体数据裁剪/过滤并导出切片为 `.npy`，输出默认在 `data/npy/CCMASK_test/<dataset_name>`，现在可用命令行参数自定义路径。
3) 运行 `npy_to_jpg.py --process_structure`：把 `imgs` 转为 jpg 并复制 `gts`，输出在 `data/jpg/<dataset_name>`。
4) 重命名 `imgs->image`、`gts->mask`，去除文件名前导零，最后按比例随机分为 `Training` 与 `Test`（每个包含 `image/` 与 `mask/`）。

我们已经提供自动化脚本 `scripts/process_dataset.sh` 实现上面所有步骤：

示例（dry-run，仅打印而不修改）:

```bash
# 给脚本可执行权限（若尚未）
chmod +x scripts/process_dataset.sh

# dry-run：只显示将做的操作
./scripts/process_dataset.sh ./CC-Mask --test-ratio 0.2 --dry-run
```

真实运行（会生成数据并做分割）:

```bash
./scripts/process_dataset.sh ./CC-Mask --test-ratio 0.2
```

脚本默认行为说明：
- 会把 `pre_CTMR.py` 的输出写入 `data/npy/CCMASK_test/<dataset_name>`（`dataset_name` 为 `CC-Mask` 的 basename）；
- 会把 `npy_to_jpg.py` 输出写入 `data/jpg/<dataset_name>`，并在该目录下创建 `image/` 与 `mask/`，最终创建 `Training/` 和 `Test/` 子目录；
- 可使用 `--move` 选项改为移动源文件（代替复制）；使用 `--python` 指定 Python 可执行路径。

注意事项：
- `pre_CTMR.py` 支持以下参数：`--nii_path`、`--gt_path`、`--npy_path`、`--modality`、`--anatomy`。若你直接调用该脚本，请传入正确路径。脚本 `scripts/process_dataset.sh` 已把路径传入 `pre_CTMR.py`。
- 若原始数据命名/结构与上述不同，请先调整路径或改写脚本参数。

## 输出目录示例（成功运行后）

```
data/jpg/<dataset>/
  ├─ image/            # jpg 图像文件或分子目录
  ├─ mask/             # 标签 npy 文件或其他格式
  ├─ Training/
  │   ├─ image/
  │   └─ mask/
  └─ Test/
      ├─ image/
      └─ mask/
```

训练脚本和示例命令
------------------

项目中用于训练的主要脚本为 `train_3d_apg.py`（3D 训练）。训练前务必确认：
- `cfg.py` 与 `conf/global_settings.py` 中的超参是否符合你的实验需求；
- `args.data_path` 指向你处理好的数据（例如 `data/jpg/<dataset>`），并根据 `func_3d.dataset.*` 代码要求设置 `mode`（Training / Test）；
- 是否使用 SAM 权重（检查 `sam2_train` 下的配置与 ckpt）。

一个典型的训练命令（在你本地环境中可能需要设置 `LD_LIBRARY_PATH` 或指定 GPU 环境）：

```bash
# 例：使用已激活 conda 环境并在 GPU 0 上训练
python train_3d_apg.py -data_path ./data/jpg/<dataset> -gpu_device 0 -image_size 1024 -exp_name my_experiment
```

```bash
python test_model.py -weights <path_to_weights> -sam_config sam2_hiera_s -data_path ./data/jpg/<dataset> -gpu_device 0
```

测试/评估
---------

使用 `test_model.py` 或 `test.py`。示例：

```bash
python test_model.py -weights logs/<exp_name>/Model/latest_epoch.pth -sam_config sam2_hiera_s -data_path ./data/jpg/<dataset> -gpu_device 0
```
