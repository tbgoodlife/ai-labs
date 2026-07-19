"""공용 데이터 로더 (실습_표준.md §3 공통 결정 사항).

8·9장 MNIST 통일 로딩 등 두 장 이상에서 쓰는 로더를 모아 둔다.
"""


def fix_ssl():
    """로컬 macOS에서 데이터셋 다운로드용 인증서 우회.
    (Colab에서는 불필요하지만 호출해도 무해하다)"""
    _fix_macos_ssl()


def _fix_macos_ssl():
    """macOS 로컬 Python의 인증서 누락으로 데이터셋
    다운로드가 실패하는 문제를 certifi로 우회한다."""
    import ssl
    try:
        import certifi
        ctx = ssl.create_default_context(
            cafile=certifi.where())
        ssl._create_default_https_context = lambda: ctx
    except ImportError:
        pass


def load_mnist(n=None, flatten=False):
    """MNIST를 keras.datasets로 로드한다 (8·9장 공용, 결정 1).

    n: 훈련 데이터 앞 n장만 사용(8장 부분 표본 = 5000)
    flatten: True면 (n, 784)로 펼침 (PCA·scikit-learn용)
    반환: (x_train, y_train), (x_test, y_test)
    """
    _fix_macos_ssl()
    from tensorflow import keras

    (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
    if n is not None:
        x_tr, y_tr = x_tr[:n], y_tr[:n]
    if flatten:
        x_tr = x_tr.reshape(len(x_tr), -1) / 255.0
        x_te = x_te.reshape(len(x_te), -1) / 255.0
    return (x_tr, y_tr), (x_te, y_te)


def load_adult():
    """UCI Adult(1994 미국 인구조사 소득) 로더 (14장 공용).

    1차: OpenML 다운로드(version=2). 실패 시 저장소 동봉
    사본 data/adult.csv로 대체한다(오프라인·차단 환경 대비).
    반환: DataFrame — 원본 컬럼 + income(">50K"/"<=50K" 문자열)
    """
    import pandas as pd

    try:
        _fix_macos_ssl()
        from sklearn.datasets import fetch_openml

        df = fetch_openml("adult", version=2,
                          as_frame=True).frame.copy()
        df["income"] = df["class"].astype(str)
    except Exception:
        from pathlib import Path

        csv = (Path(__file__).resolve().parent.parent
               / "data" / "adult.csv")
        df = pd.read_csv(csv)
    return df


def prepare_features(df, drop=("sex", "race")):
    """Adult 데이터의 학습용 행렬 준비 (14장 공용).

    수치형 특징 5종(age, education-num, hours-per-week,
    capital-gain, capital-loss)만 쓰고 표준화한다 — 민감
    속성(sex·race)은 drop 지정과 무관하게 특징에 들어가지
    않으며, drop 인자는 지면 코드의 의도 표기를 겸한다.
    반환: (X 표준화 특징, y 고소득 여부 0/1, 민감 속성 sex)
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    feats = [c for c in ("age", "education-num",
                         "hours-per-week", "capital-gain",
                         "capital-loss") if c not in drop]
    X = pd.DataFrame(StandardScaler().fit_transform(df[feats]),
                     columns=feats, index=df.index)
    y = (df["income"] == ">50K").astype(int)
    return X, y, df["sex"]
