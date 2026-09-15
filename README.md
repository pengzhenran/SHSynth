# SHSynth — 球谐系数解算（综合）

输入球谐系数 + 网格或散点位置，**输出网格或散点值并出图**。

Windows 10/11（64 位）桌面程序 · 免费 · 完全离线运行 · 不需要装 Python

**[⬇ 下载最新版 v2.0](https://github.com/pengzhenran/SHSynth/releases/latest)**

![SHSynth 界面](docs/screenshot_gui.png)

## 它做什么

球谐综合（synthesis）这一个方向做完整：**系数 → 场**。

- 系数格式**适配 SHKit 的全部输出布局**（`triangle` / `gmfcsv` / `gfc` / `npy` / `npz`），`layout=auto` 自动识别
- 求值位置五选一：全球网格、任意范围网格、借用已有网格文件的格点、散点文件、球面 Fibonacci 散点
- 输出 `.nc` / `.grd` / `.npy` / `.csv` / `.txt`；把处理过的系数（截断、平滑后）再按 SHKit 布局导出一份
- 高斯平滑（逐阶 `Wₙ`）与物理量换算（geoid / 等效水高 EWH / 面密度 σ / 径向形变 u_r，**逐阶因子**）
- 绘图五种：四联报告图、经纬地图、逐阶谱、直方图、不出图；PNG / PDF / SVG
- 地图书**不需要 cartopy**：海岸线用随包的 Natural Earth 110m（公有领域），离线可用
- 完全离线：不联网，也不上传任何数据

![四联报告图](docs/screenshot_report.png)

## v2.0 新增

| 新增 | 一句话 |
| --- | --- |
| **时间轴** | 系数 / 场知道自己对应哪一天（文件名三规则 + gfc 头），缺测与重复**只报告不静默处理** |
| **序列读取** `series-read` | 一个装着 `GSM-*.gfc` 的目录 → 带日期的 `(L+1,L+1,N)` 序列 |
| **批量综合** `series-synth` | 整条序列一次算完；可疑历元 MAD 离群**只报告不剔除** |
| **FFT 经度快路径** | 整圈均匀经度上把逐点经度展开换成一次 FFT；不满足自动回退并说明理由 |
| **水平形变 u_N / u_E** | 球面梯度算子，矢量（北 / 东 / 大小 / 方位角）；极点用解析值 |
| **趋势 / 周年 / 区域平均** | `series-fit`、`series-basin` 在**系数域**做，与场域逐点拟合一致到 1e-15 |
| **动画 / GIF** | 界面新增『动画』页：播放 / 暂停 / 逐帧 / 调帧率 / 一键导出 GIF |
| **去均值三口径** | GRACE 惯例（2004–2010 平均）/ 全时段 / 自定义时段；界面下拉 + 命令行长选项 |

实测：1° 全球 203 个历元批量综合约 **0.30 s**（自动走 FFT 快路径，直接法约 10 s）。

## 安装

- 安装包约 **83.2 MB**，装完即用，**不需要 Python**（运行时、Qt、绘图库都在包内）
- 下载：[Releases](https://github.com/pengzhenran/SHSynth/releases/latest) 页面中的 `SHSynth_Setup_v2.0.exe`
- 国内下载较慢，也可以用夸克网盘：<https://pan.quark.cn/s/821f88fe5425>（二维码见文末）
- 双击安装程序按向导完成：默认**按用户安装**（不弹管理员）到 `%LOCALAPPDATA%\Programs\SHSynth`；
  开始菜单生成 `SHSynth 球谐系数解算` / `使用说明 (HTML)` / `作者信息` / `卸载`，可选桌面快捷方式
- 卸载通过「设置 → 应用」或开始菜单中的卸载项

SHA256（v2.0）：

```
AD0B33394AA8487F2DAB0A926A9337F75FFC322E18E10652B34F14C522BFD4
```

### 首次运行可能被 Windows 拦下

程序与安装包都没有代码签名。新装的 Windows 11 上会遇到下面两种提示之一：

| 提示 | 怎么办 |
| --- | --- |
| 「Windows 已保护你的电脑 / 未知发布者」 | 点「更多信息」→「仍要运行」 |
| 「智能应用控制已阻止可能不安全的应用」 | 这是 Windows 11 的智能应用控制，**没有单应用白名单**：设置 → 隐私和安全性 → Windows 安全中心 → 应用和浏览器控制 → 智能应用控制 → 关闭，然后重新双击程序（不必重装） |

换安装目录、卸载重装都不能绕过第二种提示——它只看数字签名，不看路径。关闭智能应用控制不影响 Windows 自带的病毒防护。

## 验证安装

```cmd
"%LOCALAPPDATA%\Programs\SHSynth\SHSynth.exe" --self-test
```

程序会跑 **52 项关键路径自检**（系数读写、勒让德递推、解析解与 scipy 对照、单位换算、出图、命令行），退出码 0 表示一切就绪。

装到 `Program Files` 时把路径换成 `"C:\Program Files\SHSynth\SHSynth.exe"`。

自带命令行入口：

```cmd
SHSynth.exe --cli synth --coeffs model.sh --global-grid 1 --out out.nc --figure out.png
SHSynth.exe --cli info  --coeffs model.sh --spectrum
SHSynth.exe --version
```

## 三分钟上手

### 命令行：全球 1° 网格 + 报告图

```bash
SHSynth.exe --cli synth --coeffs model.sh --global-grid 1 \
    --out out/field.nc --figure out/report.png --figure-kind report
```

### 命令行：高斯平滑 + 换算成等效水高

```bash
SHSynth.exe --cli synth --coeffs model.gfc \
    --global-grid 0.5 --gaussian-km 300 --target-unit ewh \
    --out out/ewh.nc --figure out/ewh.png
```

### 界面

1. **① 球谐系数** → 「浏览…」选系数文件（或直接把文件拖进窗口），点「读取系数信息」
2. **② 求值位置** → 全球网格 / 范围网格 / 借用网格文件 / 散点文件 / 球面散点
3. **③ 综合选项** → 截断阶数、高斯半径、输出物理量
4. **④ 输出结果 / ⑤ 绘图** → 填结果文件与图片路径（留空则只在界面显示）
5. 点「开始解算」。右侧页签：**地图 / 报告图 / 逐阶谱 / 数值分布 / 时间序列 / 水平形变 / 动画 / 逐历元诊断 / 日志**

**批量流程**：点「读序列」选一个装着 `GSM-*.gfc` 的目录 → 「历元范围」选整条序列 → 设好求值位置 → 点「综合整条序列」。

界面内按 **F1** 打开随包的 HTML 使用说明。

## 支持的系数与网格格式

| 布局 | 扩展名 | 判据 |
| --- | --- | --- |
| `triangle` | `.sh .txt .csv .dat .tsv`（+`.gz`） | `#` 头含 `ncoef_triangle`，行数 = `2*(L+1)(L+2)/2` |
| `gmfcsv` | `.csv .txt .dat .tsv` | 表头 `n,m,C,S`，每行一个系数；多时次写成 `C_t1,S_t1,…` |
| `gfc` | `.gfc`（+`.gz`） | 首列 `gfc` / `gfct`（ICGEM/GFZ），支持 `begin/end` 标记与形式误差列 |
| `npy` | `.npy` | `(2,L+1,L+1[,ntime])`，也接受 `(L+1,L+1[,ntime])` 与三角 `(2*NC[,ntime])` |
| `npz` | `.npz` | 键名 `C`/`S`/`meta`，也接受 `flat_cs` / `triangle` |

还读得动 MATLAB 稠密方阵文本、`-180..180` 或 `0..360` 的经度、中文表头、`# key = value` 头里的 `field_unit`。

「借用网格文件」是**按内容识别**的，扩展名不准也能认出来：netCDF（含 GMT 那种实际是 netCDF 的 `.grd`）、Surfer ASCII `DSAA`、Esri ASCII、numpy 堆叠、三列文本都能读；Surfer 二进制 `.grd` 与投影坐标网格会**明确报错并给出三条出路**，不会把平面坐标当经纬度画出一张位置全错的图。

## 使用注意

- 系数文件里的 `field_unit` 标签决定能换算成什么；标签是 `scalar` / 未声明时**要 EWH 会直接报错**，不替你猜
- `--global-grid` 的经度是 `0,1,…,359`，**不含 360**；范围网格可以显式给到 360
- 一个 `.gfc` 只装一个时次；多时次要写 `.gfc` 必须用 `--time K` 指定，否则报错而不是悄悄拼出一个坏文件
- 散点文本默认 10 位有效数字；需要更多位请用 `.npy` 或 `.nc`
- 做 EWH 异常必须**先处理 C00**（GRACE GSM 的 `C₀₀ ≡ 1`，直接换算会得到约 1.2e7 m）
- `.gif` 只用 Pillow（随包带上）；`.mp4` / `.webm` 需要额外的 `imageio` + ffmpeg，没装时会明确报错并建议改用 `.gif`
- **HDF5 在网络盘上可能打不开 netCDF**：程序内置兼容层，失败时自动改用本机临时文件中转，结果不受影响

## 与 SHKit 的关系

同课题组的 **SHKit** 做「散点/网格 → 系数」（分析）这个方向，SHSynth 做「系数 → 网格/散点」（综合）。两边**逐位一致**：同一套系数、同一批点，本软件的输出与 SHKit 的 `synthesis` 完全相同（实测最大差 `0.0`），所有系数文件**双向兼容**。

- SHKit：<https://github.com/pengzhenran/SHKit>

## 引用

本程序使用 4π 归一化连带勒让德函数、无 Condon–Shortley 相位的球谐约定，载荷勒夫数表来自：

1. Wang H., Xiang L., Jia L., Wu P., Steffen H., Wang Q., Chen L. (2012). Load Love numbers and Green's functions for elastic Earth models PREM, iasp91, ak135, and modified models with refined crustal structure from Crust 2.0. Computers & Geosciences, 49, 190–199. doi:10.1016/j.cageo.2012.06.022
2. Sun J. W., Wang L. S., Peng Z. R., Fu Z. Y., Chen C (2022). The sea level fingerprints of global terrestrial water storage changes detected by GRACE and GRACE-FO data. Pure and Applied Geophysics, 179(9), 3303–3317. doi:10.1007/s00024-022-03099-5

在论文或报告中使用了本程序的计算结果，请引用上述文献。

## 许可与第三方组件

本软件自身代码采用 **MIT 许可**。界面使用 **PySide6-Essentials**（LGPLv3，动态链接、未修改）与 **matplotlib**（BSD 风格）；**刻意不使用** GPL-only 的 Qt Charts / Qt Data Visualization，以保证闭源分发可行。海岸线为 Natural Earth 110m（公有领域），随包离线提供。

安装目录下 `_internal\licenses\` 提供完整的第三方组件声明与许可全文（`NOTICE.txt`、`LGPL-3.0.txt`、`GPL-3.0.txt`）。

## 反馈

作者：彭桢燃（中国地质大学（武汉））　邮箱：zhenran.peng@cug.edu.cn

使用中遇到问题、发现异常结果，或希望增加新功能，欢迎提交 [Issue](https://github.com/pengzhenran/SHSynth/issues) 或邮件反馈；反馈时附上程序「日志」页签的内容，便于定位。

## 关注与获取

| 课题组公众号「地球重力与人类生活（TVGG）」 | 夸克网盘（安装包，国内下载更快） |
| :---: | :---: |
| <img src="docs/qr-tvgg.jpg" width="200" alt="课题组公众号二维码"> | <img src="docs/qr-quark.png" width="200" alt="夸克网盘二维码"> |
| 扫码关注，获取工具与更新 | 扫码打开网盘分享（`SHSynth_Setup_v2.0.exe`） |
