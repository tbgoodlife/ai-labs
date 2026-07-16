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
