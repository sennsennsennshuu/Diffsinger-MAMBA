"""Deploy cloud-exported DiffSinger artifacts into OpenUtau voicebank.

- Acoustic ONNX + sidecars  -> bank root
- Variance linguistic/dur   -> dsdur/  dspitch/  dsvariance/  (shared front-end)
- Pitch ONNX                -> dspitch/
- Variance ONNX             -> dsvariance/

Sidecars copied along with each model: phonemes.json, languages.json,
.opencpop.emb, dictionary-*.txt (only when the bank already had them).

Existing OpenUtau-only files (dsdict-*.yaml, character.txt, vocoder.yaml,
dsconfig.yaml) are NOT touched.
"""

import shutil
from pathlib import Path

ARTIFACTS = Path(r"I:\Chaos_extend_solo\DiffSinger-3-Chaos\Diffsinger-main-SSM\artifacts_cloud")
BANK = Path(r"D:\OpenUtau for diffsinger\Singers\SSM_test_opencpop")

ACO = ARTIFACTS / "aco_testssm2"
VAR = ARTIFACTS / "var_testssm2"


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_size = src.stat().st_size
    if dst.exists() and dst.stat().st_size == src_size and dst.read_bytes() == src.read_bytes():
        print(f"SKIP  {dst.relative_to(BANK)} (identical)")
        return
    shutil.copy2(src, dst)
    print(f"COPY  {dst.relative_to(BANK)}  ({src_size} B)")


def deploy_acoustic() -> None:
    files = [
        "aco_testssm2.onnx",
        "aco_testssm2.opencpop.emb",
        "aco_testssm2.phonemes.json",
        "aco_testssm2.languages.json",
        "dictionary-ja.txt",
        "dictionary-zh.txt",
    ]
    for name in files:
        copy(ACO / name, BANK / name)


def deploy_variance() -> None:
    common = [
        "var_testssm2.linguistic.onnx",
        "var_testssm2.dur.onnx",
        "var_testssm2.opencpop.emb",
        "var_testssm2.phonemes.json",
        "var_testssm2.languages.json",
    ]
    extras = {
        "dsdur": [],
        "dspitch": ["var_testssm2.pitch.onnx"],
        "dsvariance": ["var_testssm2.variance.onnx"],
    }
    for sub, extra in extras.items():
        for name in common + extra:
            copy(VAR / name, BANK / sub / name)


if __name__ == "__main__":
    deploy_acoustic()
    deploy_variance()
    print("done.")