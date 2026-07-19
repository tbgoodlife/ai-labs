"""공정성 지표 공용 도구 (14장 실습).

지면 코드가 호출부만 싣는 계산 함수를 모아 둔다(실습_표준.md
공용 도구 규약). 지표 부호는 모두 "남성 − 여성" 차이다.
"""
from collections import namedtuple

import numpy as np

FairnessResult = namedtuple("FairnessResult",
                            ["dp", "eo", "threshold"])


def demographic_parity_diff(pred, sex):
    """인구통계학적 동등성 차이 = 집단별 양성 예측률의 차.

    pred: 0/1 예측 배열 / sex: "Male"/"Female" 배열(Series)
    """
    pred = np.asarray(pred)
    male = np.asarray(sex) == "Male"
    return pred[male].mean() - pred[~male].mean()


def _tpr(y_true, pred):
    """참 양성률(TPR) = 실제 양성 중 양성으로 맞힌 비율."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    pos = y_true == 1
    return pred[pos].mean()


def equal_opportunity_diff(y_true, pred, sex):
    """균등 기회 차이 = 집단별 TPR의 차."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    male = np.asarray(sex) == "Male"
    return (_tpr(y_true[male], pred[male])
            - _tpr(y_true[~male], pred[~male]))


def fairness_report(model, X, sex, y_true,
                    thresholds=(0.5, 0.5)):
    """두 공정성 지표를 한 번에 계산한다.

    thresholds: (남성 임계값, 여성 임계값)
    반환: FairnessResult(dp, eo, threshold=여성 임계값)
    """
    proba = model.predict_proba(X)[:, 1]
    male = np.asarray(sex) == "Male"
    t_m, t_f = thresholds
    pred = np.where(male, proba >= t_m, proba >= t_f)
    pred = pred.astype(int)
    return FairnessResult(
        dp=demographic_parity_diff(pred, sex),
        eo=equal_opportunity_diff(y_true, pred, sex),
        threshold=t_f)


def threshold_adjust(model, X, sex, y_true,
                     target="demographic_parity"):
    """여성 집단 임계값을 조정해 목표 지표 차이를 0에
    맞춘다(남성 임계값 0.5 고정).

    이산 데이터라 정확히 0이 아니라 |차이|가 최소가 되는
    임계값을 고른다(동률이면 차이가 0 이상인 쪽).
    반환: FairnessResult(dp, eo, threshold=선택된 여성 임계값)
    """
    if target != "demographic_parity":
        raise ValueError("지원 목표: demographic_parity")
    proba = model.predict_proba(X)[:, 1]
    male = np.asarray(sex) == "Male"
    male_ppr = (proba[male] >= 0.5).mean()
    best_t, best_dp = 0.5, np.inf
    for t in np.unique(proba[~male]):
        d = male_ppr - (proba[~male] >= t).mean()
        if (abs(d) < abs(best_dp)
                or (abs(d) == abs(best_dp) and d >= 0)):
            best_dp, best_t = d, t
    return fairness_report(model, X, sex, y_true,
                           thresholds=(0.5, best_t))