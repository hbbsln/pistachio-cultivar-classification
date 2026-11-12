import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys, os

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

C_DATA, C_PROC, C_MODEL, C_EVAL = '#e8f1fb', '#fdf2e3', '#e8f5e9', '#f3e8fb'


def box(ax, x, y, w, h, text, fc, fs=8.5, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec='#333333', lw=0.9)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=11, lw=1.0, color='#333333'))
    if text:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2, text, fontsize=7, ha='left', va='center', color='#444444')


# ---------------------------------------------------------------- Figure 1: workflow
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.4); ax.axis('off')
# row 1: data
box(ax, 0.2, 4.9, 2.4, 1.1, 'Pistachio Image Dataset\n2,148 RGB images, 600×600\nKirmizi 1,232 · Siirt 916', C_DATA, fs=8)
box(ax, 2.9, 4.9, 2.5, 1.1, 'Pre-processing\nresize to 224×224\nRGB / grayscale (3-channel)\nImageNet normalization', C_PROC, fs=8)
box(ax, 5.7, 4.9, 2.45, 1.1, 'Near-duplicate detection\ncosine ≥ 0.99 (MobileNetV2)\n115 pairs → 2,033 groups', C_PROC, fs=8)
box(ax, 8.45, 4.9, 2.35, 1.1, 'Group-aware stratified split\n70/15/15, 5 seeds\n+ 5×5 group CV', C_PROC, fs=8)
arrow(ax, 2.6, 5.45, 2.9, 5.45); arrow(ax, 5.4, 5.45, 5.7, 5.45); arrow(ax, 8.15, 5.45, 8.45, 5.45)
# row 2: model
box(ax, 0.2, 2.75, 2.4, 1.35, 'Frozen ImageNet CNN\n(feature extractor)\nResNet50 · EfficientNetB0 ·\nMobileNetV2', C_MODEL, fs=8)
box(ax, 3.1, 2.75, 2.5, 1.35, 'Per-block standardization\n(optional fusion:\nRGB+gray, multi-CNN,\nvalidation-based weights)', C_MODEL, fs=8)
box(ax, 6.1, 2.75, 1.9, 1.35, 'PCA\n95 % variance', C_MODEL, fs=8)
box(ax, 8.5, 2.75, 2.3, 1.35, 'Classical classifier\nSVM (RBF) · RF · KNN', C_MODEL, fs=8)
ax.plot([9.62, 9.62, 1.4], [4.9, 4.72, 4.72], color='#333333', lw=1.0)
arrow(ax, 1.4, 4.72, 1.4, 4.1)
arrow(ax, 2.6, 3.42, 3.1, 3.42); arrow(ax, 5.6, 3.42, 6.1, 3.42); arrow(ax, 8.0, 3.42, 8.5, 3.42)
# row 3: evaluation
box(ax, 0.2, 0.35, 2.4, 1.6, 'Performance\nAcc · balanced acc. · macro-F1\nsensitivity · specificity\nκ · MCC · ROC-AUC\n(mean ± SD, 95 % CI)', C_EVAL, fs=8)
box(ax, 2.9, 0.35, 2.4, 1.6, 'Statistics\ncorrected resampled t-test\n(Nadeau–Bengio) · McNemar\nHolm correction', C_EVAL, fs=8)
box(ax, 5.6, 0.35, 2.5, 1.6, 'Baselines & robustness\nHOG/LBP + SVM · fine-tuned\nCNNs · augmentation ·\nacquisition perturbations', C_EVAL, fs=8)
box(ax, 8.4, 0.35, 2.4, 1.6, 'Explainability\nocclusion (deployed pipeline)\nGrad-CAM (fine-tuned CNN)\nSHAP (PCA inputs)', C_EVAL, fs=8)
arrow(ax, 9.65, 2.75, 9.65, 1.95)
ax.text(5.5, 6.25, 'Data and leakage control', ha='center', fontsize=10, fontweight='bold')
ax.text(5.5, 4.38, 'Hybrid model (scaler, fusion weights, PCA and classifier fitted on training data only)', ha='center', fontsize=10, fontweight='bold')
ax.text(5.5, 2.2, 'Evaluation (test sets never used for selection or tuning)', ha='center', fontsize=10, fontweight='bold')
plt.savefig(os.path.join(OUT, 'fig01_workflow.png'), dpi=300, bbox_inches='tight'); plt.close()


# ---------------------------------------------------------------- Figure 2: building blocks
def column(ax, x, ys, labels, fcs, w=2.0, h=0.55):
    for y, lab, fc in zip(ys, labels, fcs):
        box(ax, x - w / 2, y - h / 2, w, h, lab, fc, fs=8)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, x, y1 - h / 2, x, y2 + h / 2)


fig, axes = plt.subplots(1, 3, figsize=(13, 6.3))
for ax in axes:
    ax.set_xlim(0, 4); ax.set_ylim(0, 7.2); ax.axis('off')

# (a) ResNet50 bottleneck residual block
ax = axes[0]
ys = [6.4, 5.3, 4.3, 3.3, 2.2, 1.1]
labs = ['Input x (C channels)', '1×1 conv, BN, ReLU\n(reduce to C/4)', '3×3 conv, BN, ReLU', '1×1 conv, BN\n(expand to C)',
        '⊕  F(x) + x', 'ReLU → output']
fcs = [C_DATA, C_MODEL, C_MODEL, C_MODEL, '#fff9c4', C_DATA]
column(ax, 1.6, ys, labs, fcs, w=2.3)
ax.annotate('', xy=(2.78, 2.2), xytext=(2.78, 6.4), arrowprops=dict(arrowstyle='-|>', lw=1.2, color='#c62828',
                                                                    connectionstyle='bar,fraction=-0.25'))
ax.text(3.55, 4.3, 'identity\nshortcut\n(skip)', color='#c62828', fontsize=8, ha='center')
ax.set_title('(a) ResNet50 bottleneck residual block', fontsize=10, fontweight='bold')

# (b) MobileNetV2 inverted residual block
ax = axes[1]
ys = [6.4, 5.3, 4.3, 3.3, 2.2, 1.1]
labs = ['Input (C channels, low-dim.)', '1×1 conv, BN, ReLU6\n(expansion, t·C, t = 6)', '3×3 depthwise conv,\nBN, ReLU6',
        '1×1 conv, BN\n(linear bottleneck, no ReLU)', '⊕  add input\n(stride 1, same C)', 'Output']
column(ax, 1.6, ys, labs, fcs, w=2.3)
ax.annotate('', xy=(2.78, 2.2), xytext=(2.78, 6.4), arrowprops=dict(arrowstyle='-|>', lw=1.2, color='#c62828',
                                                                    connectionstyle='bar,fraction=-0.25'))
ax.text(3.55, 4.3, 'residual\nconnection', color='#c62828', fontsize=8, ha='center')
ax.set_title('(b) MobileNetV2 inverted residual block', fontsize=10, fontweight='bold')

# (c) EfficientNet-B0 MBConv + compound scaling
ax = axes[2]
ys = [6.4, 5.45, 4.55, 3.65, 2.75, 1.85]
labs = ['Input', '1×1 conv, BN, SiLU (expansion)', '3×3 / 5×5 depthwise conv,\nBN, SiLU',
        'Squeeze-and-excitation\n(channel attention)', '1×1 conv, BN (projection)', '⊕  add input (stride 1)']
fcs3 = [C_DATA, C_MODEL, C_MODEL, '#fde7e9', C_MODEL, '#fff9c4']
column(ax, 1.6, ys, labs, fcs3, w=2.4, h=0.5)
ax.annotate('', xy=(2.83, 1.85), xytext=(2.83, 6.4), arrowprops=dict(arrowstyle='-|>', lw=1.2, color='#c62828',
                                                                     connectionstyle='bar,fraction=-0.22'))
box(ax, 0.15, 0.2, 3.7, 0.95, 'Compound scaling (Tan & Le, 2019): depth $d=\\alpha^{\\phi}$, width $w=\\beta^{\\phi}$,\nresolution $r=\\gamma^{\\phi}$, '
    'with $\\alpha\\beta^{2}\\gamma^{2}\\approx 2$; B0 is the baseline ($\\phi=0$)', '#eeeeee', fs=7.2)
ax.set_title('(c) EfficientNet-B0 MBConv block', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig02_cnn_blocks.png'), dpi=300, bbox_inches='tight'); plt.close()
print('ok')
