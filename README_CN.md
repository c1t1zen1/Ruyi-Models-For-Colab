# Ruyi-Models

欢迎使用 Ruyi-Models！

Ruyi 是一款图生视频模型，能够生成 **768 分辨率、每秒 24 帧总计 5 秒 120 帧的影视级视频**，支持**镜头控制**与**运动幅度控制**，使用 **RTX 3090** 或 **RTX 4090** 可**无精度损失**地生成 512 分辨率、120 帧（768分辨率、~72帧）的视频。

## 更新说明

2025年1月14日：[TeaCache](https://github.com/ali-vilab/TeaCache) 和 [Enhance-A-Video](https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video) 已作为 ComfyUI 的节点添加，提供更快的生成速度和更高的视频质量。只需在 Sampler 节点之前链接这些节点即可使用。

2025年1月6日: 为ComfyUI加入了FP8的支持，有三种fp8的模式，显存消耗：bf16 default > fp8 lite > fp8 strong > fp8 extreme。

2024年12月24日: 更新了transformer模型以解决3:4和4:5比例的视频会出现黑杠的问题。

## 安装方法

克隆本仓库并安装所需的依赖。

```shell
git clone https://github.com/IamCreateAI/Ruyi-Models
cd Ruyi-Models
pip install -r requirements.txt
```

### ComfyUI 的安装方法

#### 方法（1）：通过 ComfyUI Manager 安装

下载并安装 [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)。

```shell
cd ComfyUI/custom_nodes/
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# install requirements
pip install -r ComfyUI-Manager/requirements.txt
```

启动 ComfyUI 并打开 Manager。选择 Custom Nodes Manager，然后搜索 “Ruyi”。选择搜索结果中的 ComfyUI-Ruyi（如下方截图所示），点击 “Install” 按钮安装。

<div align=center>
  <img src="https://github.com/user-attachments/assets/10dda65f-13d5-4da8-9437-9c98b114536c"></img>
</div>

最后，搜索 “ComfyUI-VideoHelperSuite” 并安装。

#### 方法（2）：手动安装

- 下载并保存本仓库到 *ComfyUI/custom_nodes/Ruyi-Models* 路径。

```shell
# download the repo
cd ComfyUI/custom_nodes/
git clone https://github.com/IamCreateAI/Ruyi-Models.git

# install requirements
pip install -r Ruyi-Models/requirements.txt
```

- 安装依赖项 [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) 以显示视频输出（如已安装，请跳过本步骤）。

```shell
# download ComfyUI-VideoHelperSuite
cd ComfyUI/custom_nodes/
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# install requirements
pip install -r ComfyUI-VideoHelperSuite/requirements.txt
```

##### Windows 操作系统下的安装方法

在使用Windows操作系统时，[ComfyUI_windows_portable_nvidia](https://github.com/comfyanonymous/ComfyUI/releases) 是一种常见的发行版。当通过 `run_nvidia_gpu.bat` 启动时，会使用其中嵌入的 Python 解释器。因此，需要在这个内置的 Python 环境中安装运行环境。

例如，如果提取后的发行版目录是 _ComfyUI_windows_portable_，通常可使用以下命令下载仓库并安装运行时环境：

```shell
# download the repo
cd ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/IamCreateAI/Ruyi-Models.git

# install requirements using embedded Python interpreter
..\..\python_embeded\python.exe -m pip install -r Ruyi-Models\requirements.txt
```

## 下载模型（可选）

下载模型并将其保存到指定路径。为了直接运行 Ruyi 模型，建议将模型保存到 _Ruyi-Models/models_ 文件夹中。对于 ComfyUI 用户，路径应为 _ComfyUI/models/Ruyi_。

| 名称 | 类型 | 分辨率 | 最大帧数 | 每秒帧数 | 存储空间 | 下载地址 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ruyi-Mini-7B | 图生视频 | 512 & 768 | 120 | 24 | 17 GB | [🤗](https://huggingface.co/IamCreateAI/Ruyi-Mini-7B) |

例如，下载 Ruyi-Mini-7B 后，文件的路径结构应该为：

```
📦 Ruyi-Models/models/ or ComfyUI/models/Ruyi/
├── 📂 Ruyi-Mini-7B/
│   ├── 📂 transformers/
│   ├── 📂 vae/
│   └── 📂 ...
```

> 本仓库**支持自动下载模型功能**，但手动下载提供了更多的可控性。例如，可以下载模型到其他位置，再通过软链接等方式链接到 *ComfyUI/models/Ruyi* 路径。

## 使用方法

我们提供两种运行模型的方法。第一种是直接使用Python代码。

```shell
python3 predict_i2v.py
```

具体来说，该脚本将模型下载到 _Ruyi-Models/models_ 文件夹，并使用 _[assets](./assets)_ 文件夹中的图像作为视频推理的起始帧和结束帧。您可以修改脚本中的变量来替换输入图像，并设置视频长度和分辨率等参数。

对于显存超过 24GB 的用户，可以使用 predict_i2v_80g.py 来提高生成速度。对于显存较少的用户，提供了优化显存使用的参数，这些参数可以通过延长推理时间来生成更高分辨率和更长时长的视频。这些参数的影响可以在下面的显存优化选项小节找到。

或者，您可以使用我们 GitHub 仓库中的 ComfyUI 封装，ComfyUI 节点的详细信息在 _[comfyui/README_CN.md](comfyui/README_CN.md)_ 中描述。

## 效果展示

### 图生视频效果

<table>
    <tr>
        <td><video src="https://github.com/user-attachments/assets/4dedf40b-82f2-454c-9a67-5f4ed243f5ea"></video></td>
        <td><video src="https://github.com/user-attachments/assets/905fef17-8c5d-49b0-a49a-6ae7e212fa07"></video></td>
        <td><video src="https://github.com/user-attachments/assets/20daab12-b510-448a-9491-389d7bdbbf2e"></video></td>
        <td><video src="https://github.com/user-attachments/assets/f1bb0a91-d52a-4611-bac2-8fcf9658cac0"></video></td>
    </tr>
</table>

### 镜头控制

<table>
    <tr>
        <td align=center><img src="https://github.com/user-attachments/assets/8aedcea6-3b8e-4c8b-9fed-9ceca4d41954" height=200></img>input</td>
        <td align=center><video src="https://github.com/user-attachments/assets/d9d027d4-0d4f-45f5-9d46-49860b562c69"></video>left</td>
        <td align=center><video src="https://github.com/user-attachments/assets/7716a67b-1bb8-4d44-b128-346cbc35e4ee"></video>right</td>
    </tr>
    <tr>
        <td align=center><video src="https://github.com/user-attachments/assets/cc1f1928-cab7-4c4b-90af-928936102e66"></video>static</td>
        <td align=center><video src="https://github.com/user-attachments/assets/c742ea2c-503a-454f-a61a-10b539100cd9"></video>up</td>
        <td align=center><video src="https://github.com/user-attachments/assets/442839fa-cc53-4b75-b015-909e44c065e0"></video>down</td>
    </tr>
</table>

### 运动幅度控制

<table>
    <tr>
        <td align=center><video src="https://github.com/user-attachments/assets/0020bd54-0ff6-46ad-91ee-d9f0df013772"></video>motion 1</td>
        <td align=center><video src="https://github.com/user-attachments/assets/d1c26419-54e3-4b86-8ae3-98e12de3022e"></video>motion 2</td>
        <td align=center><video src="https://github.com/user-attachments/assets/535147a2-049a-4afc-8d2a-017bc778977e"></video>motion 3</td>
        <td align=center><video src="https://github.com/user-attachments/assets/bf893d53-2e11-406f-bb9a-2aacffcecd44"></video>motion 4</td>
    </tr>
</table>

## 显存优化选项

提供了 **`GPU_memory_mode` 和 `GPU_offload_steps`** 选项以**降低显存占用**，满足不同用户的需求。

通常来说，**使用更少的显存，需要更多的内存和更长的生成时间**。以下列出预期显存使用和生成时间的参考表格。请注意，下面报告的显存是 `max_memory_allocated()` 的返回值，而 nvidia-smi 的显存数值通常会高于报告的数值。因为 CUDA 会占用一些显存（通常在500 - 800 MiB之间），而 PyTorch 的缓存机制也会请求额外的显存。

此外，在社区的帮助下创建了一张详细的表格 [gpu_memory_appendix.md](assets/gpu_memory_appendix.md)，包含了更多分辨率和选项组合。欢迎社区成员帮助完善表格。

### A100 的显存占用与运行时间

- Resolution of 512

| 帧数 | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps | low_gpu_mode + 0 steps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 24 frames  | 16119MiB <br> _01:01s_ | 15535MiB <br> _01:07s_ | 15340MiB <br> _01:13s_ | 15210MiB <br> _01:20s_ | 14950MiB <br> _01:32s_ |  4216MiB <br> _05:14s_ |
| 48 frames  | 18398MiB <br> _01:53s_ | 17230MiB <br> _02:15s_ | 16840MiB <br> _02:29s_ | 16580MiB <br> _02:32s_ | 16060MiB <br> _02:54s_ |  4590MiB <br> _09:59s_ |
| 72 frames  | 20678MiB <br> _03:00s_ | 18925MiB <br> _03:31s_ | 18340MiB <br> _03:53s_ | 17951MiB <br> _03:57s_ | 17171MiB <br> _04:25s_ |  6870MiB <br> _14:42s_ |
| 96 frames  | 22958MiB <br> _04:11s_ | 20620MiB <br> _04:54s_ | 19841MiB <br> _05:10s_ | 19321MiB <br> _05:14s_ | 18281MiB <br> _05:47s_ |  9150MiB <br> _19:17s_ |
| 120 frames | 25238MiB <br> _05:42s_ | 22315MiB <br> _06:34s_ | 21341MiB <br> _06:59s_ | 20691MiB <br> _07:07s_ | 19392MiB <br> _07:41s_ | 11430MiB <br> _24:08s_ |

- Resolution of 768

| 帧数 | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps | low_gpu_mode + 0 steps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 24 frames  | 18971MiB <br> _02:06s_ | 17655MiB <br> _02:40s_ | 17217MiB <br> _02:39s_ | 16925MiB <br> _02:41s_ | 16339MiB <br> _03:13s_ |  5162MiB <br> _13:42s_ |
| 48 frames  | 24101MiB <br> _04:52s_ | 21469MiB <br> _05:44s_ | 20592MiB <br> _05:51s_ | 20008MiB <br> _06:00s_ | 18837MiB <br> _06:49s_ | 10292MiB <br> _20:58s_ |
| 72 frames  | 29230MiB <br> _08:24s_ | 25283MiB <br> _09:45s_ | 25283MiB <br> _09:45s_ | 23091MiB <br> _10:10s_ | 21335MiB <br> _11:10s_ | 15421MiB <br> _39:12s_ |
| 96 frames  | 34360MiB <br> _12:49s_ | 29097MiB <br> _14:41s_ | 27343MiB <br> _15:33s_ | 26174MiB <br> _15:44s_ | 23834MiB <br> _16:33s_ | 20550MiB <br> _43:47s_ |
| 120 frames | 39489MiB <br> _18:21s_ | 32911MiB <br> _20:39s_ | 30719MiB <br> _21:34s_ | 29257MiB <br> _21:48s_ | 26332MiB <br> _23:02s_ | 25679MiB <br> _63:01s_ |

### RTX 4090 的显存占用与运行时间

表格中以 `---` 显示的值表示触发了显存溢出（OOM），无法生成视频。

- Resolution of 512

| 帧数 | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps | low_gpu_mode + 0 steps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 24 frames  | 16366MiB <br> _01:18s_ | 15805MiB <br> _01:26s_ | 15607MiB <br> _01:37s_ | 15475MiB <br> _01:36s_ | 15211MiB <br> _01:39s_ |  4211MiB <br> _03:57s_ |
| 48 frames  | 18720MiB <br> _02:21s_ | 17532MiB <br> _02:49s_ | 17136MiB <br> _02:55s_ | 16872MiB <br> _02:58s_ | 16344MiB <br> _03:01s_ |  4666MiB <br> _05:01s_ |
| 72 frames  | 21036MiB <br> _03:41s_ | 19254MiB <br> _04:25s_ | 18660MiB <br> _04:34s_ | 18264MiB <br> _04:36s_ | 17472MiB <br> _04:51s_ |  6981MiB <br> _06:36s_ |
| 96 frames  | -----MiB <br> _--:--s_ | 20972MiB <br> _06:18s_ | 20180MiB <br> _06:24s_ | 19652MiB <br> _06:36s_ | 18596MiB <br> _06:56s_ |  9298MiB <br> _10:03s_ |
| 120 frames | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | 21704MiB <br> _08:50s_ | 21044MiB <br> _08:53s_ | 19724MiB <br> _09:08s_ | 11613MiB <br> _13:57s_ |

- Resolution of 768

| 帧数 | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps | low_gpu_mode + 0 steps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 24 frames  | 19223MiB <br> _02:38s_ | 17900MiB <br> _03:06s_ | 17448MiB <br> _03:18s_ | 17153MiB <br> _03:23s_ | 16624MiB <br> _03:34s_ |  5251MiB <br> _05:54s_ |
| 48 frames  | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | 20946MiB <br> _07:28s_ | 20352MiB <br> _07:35s_ | 19164MiB <br> _08:04s_ | 10457MiB <br> _10:55s_ |
| 72 frames  | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | 15671MiB <br> _18:52s_ |

## 许可证

我们将以宽松的 **Apache 2.0** 许可证发布该模型。

## 引用

```
@misc{createai2024ruyi,
      title={Ruyi-Mini-7B},
      author={CreateAI Team},
      year={2024},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished={\url{https://github.com/IamCreateAI/Ruyi-Models}}
}
```

## 欢迎建议反馈与协同优化

我们真诚欢迎大家积极提供宝贵的反馈和建议，希望能够共同努力优化我们的服务和产品。您的意见将帮助我们更好地理解用户需求，从而不断提升用户体验。感谢您对我们工作的支持和关注！欢迎加入我们的[Discord](https://discord.com/invite/nueQFQwwGw) 或者微信群（扫描下方二维码，添加 Ruyi 小助手，加入官方交流群）！

<img src="https://github.com/user-attachments/assets/cc5e25c6-34ab-4be1-a59b-7d5789264a9c" style="width:300px"></img>
