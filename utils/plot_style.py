"""도해 스타일 킷 적용 + 한글 폰트 설정.

책 본문 도해와 실습 출력의 시각 통일을 위한 모듈
(표준_장_템플릿.md §5.1, assets/figures/_style/ 동기화 사본).

사용법:
    from utils import plot_style
    plot_style.apply()
"""

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

STYLE_FILE = Path(__file__).parent / "textbook.mplstyle"

_KOREAN_FONTS = {"Noto Sans KR", "Apple SD Gothic Neo",
                 "NanumGothic"}


def _ensure_korean_font():
    """한글 폰트가 없으면(주로 Colab) 나눔 폰트를 설치한다."""
    installed = {f.name for f in fm.fontManager.ttflist}
    if installed & _KOREAN_FONTS:
        return
    if sys.platform.startswith("linux"):
        subprocess.run(
            ["apt-get", "-qq", "install", "-y", "fonts-nanum"],
            check=False, capture_output=True)
        nanum = ("/usr/share/fonts/truetype/nanum/"
                 "NanumGothic.ttf")
        if os.path.exists(nanum):
            fm.fontManager.addfont(nanum)


def apply():
    """스타일 킷과 한글 폰트를 적용한다."""
    _ensure_korean_font()
    plt.style.use(str(STYLE_FILE))
