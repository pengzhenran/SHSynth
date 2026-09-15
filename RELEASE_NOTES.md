SHSynth 球谐系数解算（综合）v2.0 —— Windows 10/11（64 位），免费，完全离线运行。

**资产**：`SHSynth_Setup_v2.0.exe`（约 83.2 MB）
**SHA256**：`AD0B33394AA8487F2DAB0A926A9337F75FFC322E18E10652B34F14C522BFD4`

### 本版新增

三条主线：**多时间批量处理、水平形变 u_N/u_E、FFT 经度快路径**。

- **时间轴**：系数 / 场知道自己对应哪一天（文件名三规则 + gfc 头），缺测与重复**只报告不静默处理**
- **序列读取** `series-read`：一个装着 `GSM-*.gfc` 的目录 → 带日期的 `(L+1,L+1,N)` 序列；混入 GAC/GAD 会报错
- **批量综合** `series-synth`：整条序列一次算完，附逐历元诊断表；可疑历元用 MAD 3σ 标出但**绝不自动剔除**
- **FFT 经度快路径**：整圈均匀经度上把逐点经度展开换成一次 FFT，不满足条件自动回退并说明理由
- **水平形变 u_N / u_E**：球面梯度算子，输出北 / 东 / 大小 / 方位角；极点用解析值，`magnitude` 在极点同样有效
- **趋势 / 周年 / 区域平均** `series-fit`、`series-basin`：在**系数域**做，与场域逐点拟合一致到 1e-15
- **去均值三口径**：GRACE 惯例（2004–2010 平均）/ 全时段 / 自定义时段
- **动画 / GIF**：界面新增『动画』页，播放 / 暂停 / 逐帧 / 调帧率 / 一键导出 GIF（GIF 只需 Pillow）
- **界面**：新增时间轴控件、历元范围、批量序列、去均值口径，以及时间序列 / 水平形变 / 动画 / 逐历元诊断四个新页签

实测：1° 全球 203 个历元批量综合约 **0.30 s**（自动走 FFT 快路径，直接法约 10 s，相对差 1e-14）；区域平均 `coeff` 口径比 `spatial` 快 **48×**。

### 本版修复

修掉两个此前静默失效、直接不可用的缺陷：命令行自检里与 SHKit 的跨软件比对路径插错（`import shkit` 必然失败），
以及冻结版 `--cli` 在 cmd 里会抛 `OSError: [Errno 22]` 并挂住。

### 兼容性

v1.0 的全部行为保持不变——`synth` / `info` / `convert` 的数值与界面约定一条没动，旧测试套件仍然全绿。

### 安装

1. 下载本页的 `SHSynth_Setup_v2.0.exe`，双击按向导安装。默认**按用户安装**（不弹管理员）到
   `%LOCALAPPDATA%\Programs\SHSynth`；也可以在向导里改到 `Program Files`
2. 开始菜单生成 `SHSynth 球谐系数解算` / `使用说明 (HTML)` / `作者信息` / `卸载`，可选桌面快捷方式
3. **不需要装 Python**：运行时、Qt、绘图库都在包内的 `_internal\` 里
4. 验证安装（退出码 0 即正常）：

```cmd
"%LOCALAPPDATA%\Programs\SHSynth\SHSynth.exe" --self-test
```

会跑 52 项关键路径自检（系数读写、勒让德递推、解析解与 scipy 对照、单位换算、出图、命令行）。

### 首次运行可能被 Windows 拦下

程序与安装包都没有代码签名：

- 「Windows 已保护你的电脑 / 未知发布者」→ 点「更多信息」→「仍要运行」
- 「智能应用控制已阻止可能不安全的应用」→ 这是 Windows 11 的智能应用控制，没有单应用白名单，只能关闭：设置 → 隐私和安全性 → Windows 安全中心 → 应用和浏览器控制 → 智能应用控制 → 关闭，然后重新双击程序（不必重装）

换安装目录、卸载重装都不能绕过第二种提示——它只看数字签名，不看路径。关闭智能应用控制不影响 Defender 的病毒防护。

### 功能

- 输入球谐系数 + 网格或散点位置，输出网格或散点值并出图
- 系数格式**适配 SHKit 的全部输出布局**（`triangle` / `gmfcsv` / `gfc` / `npy` / `npz`），自动识别，双向兼容
- 求值位置五选一：全球网格、任意范围网格、借用已有网格文件的格点、散点文件、球面 Fibonacci 散点
- 输出 `.nc` / `.grd` / `.npy` / `.csv` / `.txt`；处理过的系数可再按 SHKit 布局导出
- 高斯平滑（逐阶 `Wₙ`）与物理量换算（geoid / EWH / 面密度 σ / 径向形变 u_r，**逐阶因子**）
- 绘图五种：四联报告图、经纬地图、逐阶谱、直方图、不出图；PNG / PDF / SVG
- 地图书不需要 cartopy：海岸线用随包的 Natural Earth 110m，离线可用

### 已知限制

- `--global-grid` 的经度不含 360（`0,1,…,359`）；范围网格可以显式给到 360
- 一个 `.gfc` 只装一个时次；多时次写 `.gfc` 必须用 `--time K` 指定，否则报错
- 系数文件里的 `field_unit` 标签不明时，**要 EWH 会直接报错**，程序不替你猜物理量
- `.gif` 只需 Pillow；`.mp4` / `.webm` 需要额外的 `imageio` + ffmpeg，没装时明确报错并建议改用 `.gif`

### 引用

1. Wang H., Xiang L., Jia L., Wu P., Steffen H., Wang Q., Chen L. (2012). Load Love numbers and Green's functions for elastic Earth models PREM, iasp91, ak135, and modified models with refined crustal structure from Crust 2.0. Computers & Geosciences, 49, 190–199. doi:10.1016/j.cageo.2012.06.022
2. Sun J. W., Wang L. S., Peng Z. R., Fu Z. Y., Chen C (2022). The sea level fingerprints of global terrestrial water storage changes detected by GRACE and GRACE-FO data. Pure and Applied Geophysics, 179(9), 3303–3317. doi:10.1007/s00024-022-03099-5
