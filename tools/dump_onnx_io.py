"""Print input/output meta of each voicebank ONNX so we can build correct
dummy feeds for profile_voicebank.py."""
from pathlib import Path
import onnxruntime as ort

BANK = Path(r"D:\OpenUtau for diffsinger\Singers\SSM_test_opencpop")

paths = {
    "acoustic": BANK / "aco_testssm2.onnx",
    "vocoder":  BANK / "dsvocoder" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx",
    "linguistic": BANK / "dspitch" / "var_testssm2.linguistic.onnx",
    "dur":      BANK / "dsdur"   / "var_testssm2.dur.onnx",
    "pitch":    BANK / "dspitch" / "var_testssm2.pitch.onnx",
    "variance": BANK / "dsvariance" / "var_testssm2.variance.onnx",
}

for name, path in paths.items():
    print(f"\n=== {name}: {path.name} ===")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print("INPUTS:")
    for i in sess.get_inputs():
        print(f"  {i.name:18s} shape={i.shape} dtype={i.type}")
    print("OUTPUTS:")
    for o in sess.get_outputs():
        print(f"  {o.name:18s} shape={o.shape} dtype={o.type}")