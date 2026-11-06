"""生成 TopKPQDistance 架构图（Pillow）。产物在 docs/assets/。"""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

REG = "C:/Windows/Fonts/msyh.ttc"
BOLD = "C:/Windows/Fonts/msyhbd.ttc"
INK = "#1F2328"
MUTED = "#59636E"
LINE = "#8C959F"

_cache = {}


def F(sz, bold=False):
    key = (sz, bold)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(BOLD if bold else REG, sz)
    return _cache[key]


def rbox(d, box, fill, outline, r=14, w=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=w)


def ctext(d, cx, y, s, f, fill=INK):
    d.text((cx, y), s, font=f, fill=fill, anchor="ma")


def ltext(d, x, y, s, f, fill=INK):
    d.text((x, y), s, font=f, fill=fill)


def arrow(d, p0, p1, color=LINE, w=3, head=11):
    d.line([p0, p1], fill=color, width=w)
    (x0, y0), (x1, y1) = p0, p1
    if abs(x1 - x0) >= abs(y1 - y0):
        s = 1 if x1 > x0 else -1
        d.polygon([(x1, y1), (x1 - s * head, y1 - head * 0.55), (x1 - s * head, y1 + head * 0.55)], fill=color)
    else:
        s = 1 if y1 > y0 else -1
        d.polygon([(x1, y1), (x1 - head * 0.55, y1 - s * head), (x1 + head * 0.55, y1 - s * head)], fill=color)


def card(d, box, title, body, fill="#FFFFFF", outline="#D0D7DE", tcolor=INK, bcolor=MUTED, tsz=21, bsz=17):
    rbox(d, box, fill, outline, r=12, w=2)
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    lines = body if isinstance(body, list) else [body]
    total = tsz + 8 + len(lines) * (bsz + 7)
    y = y0 + ((y1 - y0) - total) / 2
    ctext(d, cx, y, title, F(tsz, True), tcolor)
    y += tsz + 8
    for ln in lines:
        ctext(d, cx, y, ln, F(bsz), bcolor)
        y += bsz + 7


# ---------------------------------------------------------------------------
# 图 1：端到端架构
# ---------------------------------------------------------------------------
W, H = 1660, 1010
img = Image.new("RGB", (W, H), "#FFFFFF")
d = ImageDraw.Draw(img)

ltext(d, 56, 26, "TopKPQDistance · 架构总览", F(38, True))
ltext(d, 56, 78, "host 只做 tiling 与下发；比较 / 建堆 / 堆排 / gather 全在 device kernel 内完成", F(19), MUTED)

# ---- HOST ----
rbox(d, (40, 118, 1620, 300), "#F6F8FA", "#D0D7DE")
ltext(d, 62, 132, "HOST   run_kernel(...)", F(23, True))
steps = [
    ("① 解析属性", ['order == "ASC" ? → desc', "k / group_size 直接用"]),
    ("② 选路径", ["cap = shape[0]（容量上界）", "cap ≤ 65536 ? 快路径"]),
    ("③ 申请 workspace", ["aclrtMalloc(B·kAligned·8B)", "只有通用路径需要"]),
    ("④ 下发 + 同步", ["kernel<<<B, 0, stream>>>", "aclrtSynchronizeStream + free"]),
]
bw, gap, x = 360, 27, 62
for i, (t, b) in enumerate(steps):
    card(d, (x, 172, x + bw, 258), t, b, fill="#FFFFFF")
    if i < 3:
        arrow(d, (x + bw + 3, 215), (x + bw + gap - 3, 215))
    x += bw + gap
ltext(d, 62, 268, "host 不读也不写任何业务数据：没有 aclrtMemcpy、没有 CPU 排序 —— 它不知道答案", F(16), "#B42318")

# ---- DEVICE ----
ltext(d, 56, 330, "DEVICE   两条路径（按 pq_distance 的容量 cap 选）", F(23, True))

# 快路径
rbox(d, (40, 366, 700, 636), "#FFFBEB", "#D4A72C")
ltext(d, 62, 380, "快路径    cap ≤ 65536", F(20, True))
card(d, (62, 414, 678, 500), "TopKPQDistanceSmall   ×1 block", ["单核一次做完：建堆 → 扫描 → 堆排 → gather"], fill="#FFFFFF", outline="#E3B341")
card(d, (62, 516, 678, 616), "为什么单独开一条", ["小 case 的耗时全是下发开销（两次 launch ≈ 10 µs）", "砍掉 workspace 与第二次下发：14 µs → 6 µs"], fill="#FFFFFF", outline="#E3B341", bsz=16)

# 通用路径
rbox(d, (740, 366, 1620, 636), "#F3F8FF", "#4A7DCO".replace("O", "0"))
ltext(d, 762, 380, "通用路径    cap > 65536", F(20, True))
card(d, (762, 414, 1000, 540), "Stage1  ×B blocks", ["每核扫 n/B 个元素", "本地 top-k"], fill="#FFFFFF", outline="#4A7DC0")
card(d, (1050, 414, 1230, 540), "workspace", ["B × min(k, n/B)", "(value, pos)"], fill="#FFFFFF", outline="#4A7DC0")
card(d, (1280, 414, 1598, 540), "Stage2  ×1 block", ["归并 B 份候选", "堆排 + gather 载荷"], fill="#FFFFFF", outline="#4A7DC0")
arrow(d, (1002, 477), (1046, 477))
arrow(d, (1232, 477), (1276, 477))
ltext(d, 762, 560, "块之间没有通信：Stage1 各写各的槽位，Stage2 等 Stage1 整个 kernel 结束再读", F(16), MUTED)
ltext(d, 762, 590, "（两次下发在同一个 stream 上，天然串行）", F(16), MUTED)

# 输出
arrow(d, (370, 638), (370, 672))
arrow(d, (1180, 638), (1180, 672))
rbox(d, (240, 674, 1420, 748), "#F6F8FA", "#8C959F")
ctext(d, 830, 686, "topk_distance[k]      topk_ivf[k]      topk_index[k]", F(22, True))
ctext(d, 830, 718, "全部由 device kernel 用 SetValue 直接写，host 不搬运", F(16), MUTED)

# ---- block 内部 ----
ltext(d, 56, 782, "单个 block 内部：tile 循环（GM → UB → 标量堆）", F(23, True))
y0, y1 = 818, 930
chain = [
    (62, 330, "GM", ["pq_distance"]),
    (352, 640, "DataCopy", ["2048 元素 / 8KB"]),
    (662, 900, "UB tile", ["片上，标量可读"]),
    (922, 1180, "标量比较", ["只跟堆顶比一次"]),
    (1202, 1500, "容量 m 的堆", ["m = min(k, 本核元素数)"]),
]
for i, (xa, xb, t, b) in enumerate(chain):
    card(d, (xa, y0, xb, y1), t, b, fill="#FFFFFF")
    if i < len(chain) - 1:
        arrow(d, (xb + 3, (y0 + y1) / 2), (chain[i + 1][0] - 3, (y0 + y1) / 2))

# 回环箭头
loop_y = 962
d.line([(1350, y1 + 3), (1350, loop_y)], fill=LINE, width=3)
d.line([(1350, loop_y), (496, loop_y)], fill=LINE, width=3)
arrow(d, (496, loop_y), (496, y1 + 3))
ctext(d, 920, 938, "取下一个 tile（扫到 hi 为止）", F(16), MUTED)

# 分支标注
ltext(d, 1204, 800, "更优 → 替换堆顶 + SiftDown", F(16), "#1A7F37")

img.save(os.path.join(OUT, "architecture.png"))
print("wrote architecture.png")

# ---------------------------------------------------------------------------
# 图 2：堆
# ---------------------------------------------------------------------------
W2, H2 = 1500, 820
img2 = Image.new("RGB", (W2, H2), "#FFFFFF")
d2 = ImageDraw.Draw(img2)

ltext(d2, 56, 26, "核心数据结构：根 = 当前第 k 名", F(36, True))
ltext(d2, 56, 76, "容量 k 的「最差在堆顶」堆 —— 滑动 top-k 只需要维护一个门槛", F(19), MUTED)

# 树
nodes = {
    "r": (400, 250, "第 k 名\n(堆里最差)"),
    "a": (230, 400, "比根好"),
    "b": (570, 400, "比根好"),
    "c": (150, 550, "更好"),
    "d": (320, 550, "更好"),
    "e": (490, 550, "更好"),
    "f": (640, 550, "更好"),
}
edges = [("r", "a"), ("r", "b"), ("a", "c"), ("a", "d"), ("b", "e"), ("b", "f")]
for p, q in edges:
    d2.line([nodes[p][:2], nodes[q][:2]], fill=LINE, width=3)
for key, (x, y, label) in nodes.items():
    rr = 52 if key != "r" else 62
    fill = "#FDE8E8" if key == "r" else "#EAF7EE"
    out = "#C4383B" if key == "r" else "#3E9F5A"
    d2.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill, outline=out, width=3)
    lines = label.split("\n")
    ty = y - (len(lines) * 22) / 2
    for ln in lines:
        ctext(d2, x, ty, ln, F(17, key == "r"), "#C4383B" if key == "r" else "#1F2328")
        ty += 22

ltext(d2, 56, 640, "不变量：父节点永远不比子节点“更好”", F(19, True))
ltext(d2, 56, 672, "所以根就是整个堆里最差的那个 = 当前第 k 名 = 后面所有元素的准入门槛", F(17), MUTED)

# 右侧规则
rx = 800
rbox(d2, (rx - 24, 120, W2 - 56, 760), "#F6F8FA", "#D0D7DE")
ltext(d2, rx, 140, "一次比较决定命运", F(22, True))

steps2 = [
    ("① 读一个元素 v", "v 与堆顶比：IsWorse(根, v)"),
    ("② 不比根好", "整条丢弃，O(1) —— 绝大多数元素走这里"),
    ("③ 比根好", "根 = v，然后 SiftDown 把根沉到该在的位置"),
    ("④ 扫完之后", "弹出根 → 放数组尾部 → 尾部元素补根 → SiftDown"),
    ("⑤ 重复 m-1 次", "数组就地变成有序序列（堆排），无需额外空间"),
]
y = 190
for t, b in steps2:
    ltext(d2, rx, y, t, F(19, True))
    ltext(d2, rx + 24, y + 30, b, F(17), MUTED)
    y += 96

ltext(d2, rx, 690, "平局怎么办：比较时带上原始索引（索引小者优先）", F(16), "#B42318")
ltext(d2, rx, 718, "= np.argsort(kind=\"stable\") 的语义", F(16), "#B42318")

img2.save(os.path.join(OUT, "heap.png"))
print("wrote heap.png")
