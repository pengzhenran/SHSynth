# 随包数据说明

| 文件 | 内容 | 来源 | 许可 |
| --- | --- | --- | --- |
| `love_numbers.npy` | 载荷勒夫数 `k′ₙ`(逐阶索引,`k′₀ = 0`) | PREM / Wang et al. (2012),与参考实现 `m2py` / `gridSHconvert` 的表逐值一致 | 公开科研数据 |
| `load_love_numbers.npz` | 完整 `h′` / `l′` / `k′` 表(键 `n,h,l,k`,另有 `model`/`source`) | `PREM-LLNs.dat`(Wang et al. 2012) | 公开科研数据 |
| `love_numbers_h.npy` / `_l.npy` / `_k.npy` | 上面三张表的单独 `.npy` 版本(方便直接 `np.load`) | 同上 | 同上 |
| `coastline_110m.npz` | Natural Earth 110m 海岸线折线(`lon`, `lat`,NaN 分隔) | Natural Earth(公有领域) | 公有领域 |

> **没有国界数据。** 早期版本曾带一份 `borders_110m.npz`,现在已删除:
> 国界数据自带主权划线方式,画在地图上容易引起政治争议;海岸线是自然地理要素,
> 没有这个问题。需要国界请自行叠加自己的数据。

要点:

* `k′₁` 用的是与 `pz_LLN.m` 完全相同的 **CE → CF** 改正
  `k′₁ = -(h′₁ + 2 l′₁)/3`,所以既有 EWH 结果不受影响;
* `h′₀ = 0` —— 这意味着**径向形变的 0 阶信息不可反推**,
  软件会把该阶置 0 并如实告知(见使用说明里「物理量与单位:什么时候要换算」一节);
* 海岸线是**离线**的:地图不需要 cartopy,也不会在第一次画图时去下载;
* 想换成你本机的其它地球模型(ak135 / iasp91 / PREM-hard …):

  ```python
  from shsynth.lovenumbers import load_lln, load_love_numbers
  lln = load_lln(r"D:\models\PREM-LLNs.dat")      # 列 n h l k
  ```

  或设置环境变量 `SHSYNTH_DATA_DIR` 指向你自己的数据目录。
  **注意**:自己传的 `k′` 表若比要算的阶数短,软件会**报错**而不是按 `k′ = 0`
  悄悄算(那等于不做载荷改正)。

数据来源与 SHKit 的 `data/` 目录一致(直接复制),以保证两个软件的
物理量换算**逐位相同**。
