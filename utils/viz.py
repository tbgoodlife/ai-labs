"""공용 시각화 유틸 — 지면 25줄 상한을 넘는 그리기 코드를 모아 둔다.

지면·노트북에는 이 함수들의 호출부만 노출한다(실습_표준.md §1).
"""

import matplotlib.pyplot as plt
import numpy as np

# 회색조 4단계 (도해 스타일 킷과 동일)
GRAY = ["#ffffff", "#e6e6e6", "#bfbfbf", "#8c8c8c"]


def draw_maze_search(grid, visited, path, start, goal,
                     title="", ax=None):
    """미로와 탐색 결과(방문 칸·경로)를 함께 그린다.

    grid: 0=통로, 1=벽의 2차원 리스트
    visited: 방문한 (행, 열) 집합 / path: 경로 리스트
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    g = np.array(grid, dtype=float)
    img = np.ones(g.shape + (3,))
    img[g == 1] = 0.0                      # 벽 = 검정
    for r, c in visited:                   # 방문 = 회색
        img[r, c] = 0.75
    for r, c in path:                      # 경로 = 진회색
        img[r, c] = 0.35
    img[start] = 0.55
    img[goal] = 0.55
    ax.imshow(img, interpolation="nearest")
    ax.plot(start[1], start[0], "o", color="black", ms=6)
    ax.plot(goal[1], goal[0], "s", color="black", ms=6)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_heuristic_comparison(results, ax=None):
    """휴리스틱별 노드 확장 수 묶음 막대그래프.

    results: {휴리스틱 이름: {난이도 라벨: 확장 수}} 딕셔너리
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    names = list(results.keys())
    labels = list(results[names[0]].keys())
    x = np.arange(len(labels))
    width = 0.8 / len(names)
    hatches = ["", "//"]
    for i, name in enumerate(names):
        vals = [results[name][lb] for lb in labels]
        ax.bar(x + i * width, vals, width, label=name,
               color=GRAY[1 + i], edgecolor="black",
               hatch=hatches[i % 2], linewidth=0.8)
        for xi, v in zip(x + i * width, vals):
            ax.annotate(f"{v:,}", (xi, v), ha="center",
                        va="bottom", fontsize=8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("노드 확장 수")
    ax.legend()
    ax.grid(axis="x", visible=False)
    return ax


def show_digits(x, y, n=10):
    """손글씨 숫자 0~n-1의 표본 격자. (8·9장 공용)"""
    fig, axes = plt.subplots(1, n, figsize=(n * 0.9, 1.2))
    for d, ax in zip(range(n), axes):
        ax.imshow(x[y == d][0].reshape(28, 28),
                  cmap="gray_r")
        ax.set_title(str(d), fontsize=8)
        ax.axis("off")
    plt.show()


def plot_curves(hist, title="", ax=None):
    """Keras History의 훈련/검증 손실 곡선. (9장)"""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(hist.history["loss"], "-", color="black",
            lw=1.2, label="훈련 손실")
    ax.plot(hist.history["val_loss"], "--",
            color=GRAY[3], lw=1.2, label="검증 손실")
    ax.set_xlabel("에폭")
    ax.set_ylabel("손실")
    ax.set_title(title)
    ax.legend()
    plt.show()
    return ax


def plot_curves_pair(h1, h2, labels=("A", "B")):
    """두 실험의 훈련/검증 손실을 나란히 비교. (9장)"""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2),
                             sharey=True)
    for ax, h, lb in zip(axes, (h1, h2), labels):
        ax.plot(h.history["loss"], "-", color="black",
                lw=1.2, label="훈련")
        ax.plot(h.history["val_loss"], "--",
                color=GRAY[3], lw=1.2, label="검증")
        ax.set_xlabel("에폭")
        ax.set_title(lb)
        ax.legend()
    axes[0].set_ylabel("손실")
    plt.tight_layout()
    plt.show()
    return axes


def plot_poly_panels(x_tr, y_tr, x_te, y_te, fits):
    """다항 회귀 차수별 적합 곡선 패널. (6장 실습용)

    fits: {차수: (예측 함수, 훈련 오차, 테스트 오차)}
    """
    fig, axes = plt.subplots(1, len(fits),
                             figsize=(3.2 * len(fits), 3.2))
    xs = np.linspace(x_tr.min(), x_tr.max(), 300)
    for ax, (deg, (pred, e_tr, e_te)) in zip(
            np.atleast_1d(axes), fits.items()):
        ax.scatter(x_tr, y_tr, s=16, color=GRAY[3],
                   edgecolor="black", linewidth=.4,
                   label="훈련 데이터")
        ax.plot(xs, pred(xs), color="black", lw=1.6)
        ax.set_title(f"{deg}차 — 훈련 {e_tr:.2f} / "
                     f"테스트 {e_te:.2f}")
        ax.set_ylim(y_tr.min() - 1, y_tr.max() + 1)
    axes[0].legend(loc="upper left")
    plt.tight_layout()
    return axes


def plot_learning_curve(sizes, err_tr, err_va, ax=None):
    """학습 곡선 — 훈련·검증 오차 두 곡선의 간격 읽기. (6장)"""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(sizes, err_tr, "-", color="black", lw=1.2,
            marker="o", ms=4, label="훈련 오차")
    ax.plot(sizes, err_va, "--", color=GRAY[3], lw=1.2,
            marker="s", ms=4, label="검증 오차")
    ax.set_xlabel("훈련 데이터 수")
    ax.set_ylabel("오차(MSE)")
    ax.legend()
    return ax


def plot_kmeans_steps(X, k=3, seed=0, steps=(0, 1, 2, 9)):
    """k-평균 반복 과정 4컷(초기→할당→갱신→수렴). (8장)

    본문 [그림 8-x] 4컷 다이어그램과 동일 구도.
    """
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    snaps, labels = [], None
    for it in range(max(steps) + 1):
        d = np.linalg.norm(X[:, None] - centers[None],
                           axis=2)
        labels = d.argmin(axis=1)
        if it in steps:
            snaps.append((it, centers.copy(),
                          labels.copy()))
        centers = np.array([X[labels == j].mean(axis=0)
                            for j in range(k)])
    fig, axes = plt.subplots(1, len(snaps),
                             figsize=(3 * len(snaps), 3))
    markers = ["o", "s", "^"]
    for ax, (it, cen, lab) in zip(axes, snaps):
        for j in range(k):
            pts = X[lab == j]
            ax.scatter(pts[:, 0], pts[:, 1], s=14,
                       marker=markers[j % 3],
                       color=GRAY[1 + j % 3],
                       edgecolor="black", linewidth=.3)
        ax.scatter(cen[:, 0], cen[:, 1], s=140, marker="*",
                   color="black", zorder=3)
        ax.set_title(f"반복 {it}회" if it else "초기 중심")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return axes


def plot_qtable(Q, ax=None, title=""):
    """FrozenLake 4×4 Q-테이블 히트맵 + 탐욕 행동 화살표. (13장)"""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.6))
    V = Q.max(axis=1).reshape(4, 4)
    ax.imshow(V, cmap="gray", vmin=0, vmax=max(V.max(), 1e-9))
    arrows = "←↓→↑"
    for s in range(16):
        r, c = divmod(s, 4)
        if Q[s].max() > 0:
            ax.text(c, r, arrows[int(Q[s].argmax())],
                    ha="center", va="center", fontsize=13,
                    color="black" if V[r, c] < V.max() * .6
                    else "white")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def plot_success_curves(runs, window=50, ax=None,
                        labels=None):
    """에피소드 성공 기록(0/1)의 이동 평균 곡선. (13장)

    runs: {라벨: 성공 배열} 딕셔너리, 또는 성공 배열의
    튜플/리스트(이때 labels로 라벨 지정)
    """
    if not hasattr(runs, "items"):
        names = (labels if labels is not None
                 else [f"run {i+1}" for i in range(len(runs))])
        runs = dict(zip(names, runs))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    styles = ["-", "--", "-."]
    for (label, wins), st, g in zip(runs.items(), styles,
                                    [0, 3, 2]):
        wins = np.asarray(wins, dtype=float)
        ma = np.convolve(wins, np.ones(window) / window,
                         mode="valid")
        ax.plot(ma, st, color=GRAY[g] if g else "black",
                lw=1.2, label=label)
    ax.set_xlabel("에피소드")
    ax.set_ylabel(f"성공률(최근 {window}회 평균)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    return ax


def plot_decision_boundary(model, X, y, ax=None,
                           steps=300, title=""):
    """2차원 특징 공간의 분류기 결정 경계를 그린다. (7장 실습용)

    model: fit이 끝난 scikit-learn 분류기
    X: (n, 2) 특징 배열 / y: 정수 라벨 배열
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))
    x0 = np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5,
                     steps)
    x1 = np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5,
                     steps)
    xx, yy = np.meshgrid(x0, x1)
    grid = np.c_[xx.ravel(), yy.ravel()]
    if hasattr(model, "predict"):
        zz = model.predict(grid)
    else:                       # forward()만 있는 밑바닥 구현
        zz = (model.forward(grid) > 0.5).astype(int)
    zz = np.asarray(zz).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=.35, cmap="gray")
    markers = ["o", "s", "^", "D"]
    for k in np.unique(y):
        pts = X[y == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=18,
                   marker=markers[int(k) % 4],
                   color=GRAY[3], edgecolor="black",
                   linewidth=0.5, label=f"클래스 {k}")
    ax.set_title(title)
    return ax


def show_samples(x, y, classes, n=10):
    """클래스 0~n-1의 대표 이미지를 한 줄로 보여준다. (10장)

    x: 이미지 배열 / y: 정수 라벨 / classes: 클래스 이름 목록
    """
    y = np.asarray(y).ravel()
    x = np.asarray(x)
    fig, axes = plt.subplots(1, n, figsize=(n, 1.4))
    for d, ax in enumerate(axes):
        ax.imshow(x[y == d][0])
        ax.set_title(classes[d], fontsize=8)
        ax.axis("off")
    plt.show()
    return axes


def plot_qtable_heatmap(Q_before, Q_after,
                        titles=("Before", "After")):
    """Before/After 두 Q-테이블 히트맵을 나란히 그린다. (13장)

    plot_qtable()의 지면 호출명 래퍼 — 두 수첩을 한 그림에.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for Q, ax, t in zip((Q_before, Q_after), axes, titles):
        plot_qtable(Q, ax=ax, title=t)
    plt.tight_layout()
    plt.show()
    return axes


def plot_group_rates(rate, title=""):
    """집단별 비율 막대 그래프. (14장)

    rate: pandas Series — 인덱스가 집단 이름, 값이 비율
    """
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.bar([str(i) for i in rate.index],
           np.asarray(rate, dtype=float),
           color=[GRAY[3], "#404040"], width=0.6)
    for i, v in enumerate(np.asarray(rate, dtype=float)):
        ax.text(i, v + 0.008, f"{v:.1%}",
                ha="center", fontsize=9)
    ax.set_ylabel("비율")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return ax
