# Kế hoạch Dự án 14 Ngày: Hệ thống Phân loại Khuyết tật Bề mặt Thép Thời gian thực (NEU Surface Defect) Tối ưu hóa bằng ONNX

> **Đánh giá & Nâng cấp bởi Senior Computer Vision / Edge AI Engineer (10+ năm kinh nghiệm)**
> Kế hoạch này tích hợp **100% nền tảng lý thuyết chuyên sâu** (kiến trúc mạng gọn nhẹ, chỉ số phần cứng biên, đồ thị tính toán) vào quy trình xây dựng một **Dự án Portfolio chuẩn Production** trên Google Colab & Local Git Repo.

---

## 🔍 Đánh giá Chuyên gia & Các Điểm Nâng cấp Kỹ thuật (Expert Review)

Bản kế hoạch gốc của bạn có cấu trúc sư phạm và thực chiến **rất xuất sắc** — đặc biệt ở việc kết nối trực tiếp lý thuyết (Inverted Residual, Roofline Model, BatchNorm Folding) với thực nghiệm (Profiler, Netron, Numerical Validation) và chiến lược tiết kiệm GPU trên Colab.

Dưới đây là **5 điểm tinh chỉnh quan trọng** từ góc nhìn triển khai hệ thống thị giác máy tính công nghiệp (Industrial AOI - Automated Optical Inspection) được bổ sung vào kế hoạch để nâng tầm Portfolio từ mức *Academic/Fresher* lên mức *Mid/Senior Edge AI Engineer*:

1. **Cơ chế ONNX Export & BatchNorm Folding thực tế:**
   - Khi gọi `torch.onnx.export(..., training=torch.onnx.TrainingMode.EVAL)` (mặc định), PyTorch JIT pass (`_jit_pass_onnx_peephole`) **đã tự động gập (fold) `Conv + BatchNorm`** trước khi sinh file `.onnx`. Nếu mở Netron ngay lúc đó, bạn sẽ không thấy node `BatchNormalization` đâu!
   - **Giải pháp trong kế hoạch:** Ở Ngày 8–9, ta xuất cả 2 bản: `model_unfolded.onnx` (dùng `TrainingMode.PRESERVE` để giữ nguyên node `BatchNormalization` phục vụ trực quan hóa Netron và thực hành phẫu thuật đồ thị thủ công ở Ngày 10–11) và `model_raw.onnx` (bản `EVAL` chuẩn).
2. **Đặc thù dữ liệu ảnh xám công nghiệp (NEU Surface Defect):**
   - Ảnh gốc của NEU là ảnh grayscale kích thước `200x200` (300 ảnh/lớp x 6 lớp = 1.800 ảnh: *Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches*).
   - Khi đưa vào `MobileNetV2` (pretrained ImageNet 3 kênh RGB, `224x224`), cần chuẩn hóa pipeline: replicate 3 kênh (hoặc tùy chọn sửa lớp `conv1` sang `in_channels=1` để giảm 3x FLOPs tầng đầu), dùng **Stratified Split (70/15/15)** với `seed=42` cố định để đảm bảo phân bố đều 6 lớp lỗi, và áp dụng Augmentation chuẩn công nghiệp (Random Horizontal/Vertical Flip, nhẹ nhàng với Rotation/Brightness, tránh biến dạng hình học làm mất đặc trưng vết xước).
3. **Đo lường độ chính xác chuẩn Kiểm tra Ngoại quan (AOI Metrics):**
   - Trong nhà máy thép, **False Negative (bỏ lọt phế phẩm)** nguy hiểm hơn nhiều so với False Positive. Do đó, bên cạnh **Top-1 Accuracy (>98%)**, Ngày 7 sẽ xuất thêm **Confusion Matrix** và **Per-class Precision / Recall / F1-score** (đặc biệt lưu ý 2 lớp dễ nhầm lẫn là *Crazing* và *Rolled-in Scale*).
4. **Chuẩn hóa Benchmark Suy luận Biên (Edge Inference Benchmarking):**
   - Trên CPU của Colab (thường là 2 vCPU Intel Xeon), để đo Latency không bị nhiễu (noiseless), cần:
     - Khóa cứng luồng trong `onnxruntime.SessionOptions`: `intra_op_num_threads = 1` (đo single-core latency) và `intra_op_num_threads = 2` (multi-core), đặt `execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL`, bật `GraphOptimizationLevel.ORT_ENABLE_ALL`.
     - Đo cả **Single-stream Latency (`batch_size = 1`)** cho kịch bản camera kích hoạt theo xung (trigger mode) và **Batch Throughput (`batch_size = 8, 16`)** cho kịch bản quét liên tục.
5. **Tinh chỉnh Sai số Số học (Numerical Tolerance) & Mở rộng Quantization (Bonus):**
   - Sai số tuyệt đối `max(|Y_torch - Y_onnx|)` trên `FP32` qua ~53 tầng tích chập ở phần **logits** (trước Softmax) thường dao động ở mức `1e-5` đến `5e-5` tùy tập lệnh SIMD (AVX2/AVX-512) của CPU. Sử dụng `np.testing.assert_allclose(rtol=1e-3, atol=1e-5)` trên cả **Logits** và **Softmax Probabilities**, kết hợp kiểm tra **100% Top-1 Prediction Agreement**.
   - Bổ sung nhánh thực nghiệm nhỏ (tùy chọn ở Ngày 11): **ONNX Dynamic/Static INT8 Quantization** để thấy dung lượng mô hình giảm từ ~9 MB (`FP32`) xuống ~2.4 MB (`INT8`).

---

## Tổng quan Dự án Portfolio & Môi trường Thực thi

* **Bài toán thực tế:** Nhận diện 6 loại lỗi bề mặt thép trên dây chuyền cán thép (*NEU Surface Defect Dataset* — 1.800 ảnh, 6 lớp: `Cr`, `In`, `Pa`, `Ps`, `RS`, `Sc`).
* **Mô hình triển khai:** `MobileNetV2` làm mô hình chính, đối chiếu trực tiếp với `ResNet-18` (CNN truyền thống) và `MobileViT-S` (Hybrid CNN-Transformer) để chứng minh sự đánh đổi giữa **Accuracy vs. FLOPs vs. Memory Access Cost vs. Real Hardware Latency**.
* **Chiến lược sử dụng Runtime trên Google Colab:**
  * **Tuần 1 (Ngày 1–6) & Tuần 2 (Ngày 8–14):** Dùng **CPU Runtime** mặc định của Colab (miễn phí, ổn định, không lo hết quota) để lập trình kiến trúc, phân tích profiler, phẫu thuật đồ thị ONNX và đo CPU Latency thực tế.
  * **Riêng Ngày 7:** Bật **GPU T4** khoảng 15 phút để fine-tune `MobileNetV2` và lưu trọng số `mobilenetv2_neu.pth` cùng lịch sử huấn luyện lên Google Drive.

---

## Giai đoạn 1: Kiến trúc Gọn nhẹ, Chỉ số Biên & Huấn luyện Baseline (Tuần 1)

| Thời gian | Chủ đề Học thuật Chuyên sâu | Thực hành & Sản phẩm đầu ra (Deliverables) | Tiêu chí Nghiệm thu (DoD) |
| :--- | :--- | :--- | :--- |
| **Ngày 1 – 2** | **Toán tử Tích chập Tối ưu**<br>- Công thức tính Params & FLOPs/MACs: Standard Conv vs. **Depthwise Separable Conv** (Depthwise $K \times K$, `groups=C_in` + Pointwise $1 \times 1$).<br>- Tỷ lệ giảm chi phí tính toán: $\frac{1}{N} + \frac{1}{D_K^2}$.<br>- **Inverted Residual Block** (Narrow $\to$ Wide $\to$ Narrow với `expansion_factor = 6`) & **Linear Bottleneck** (lý do loại bỏ `ReLU6` ở lớp chiếu $1 \times 1$ cuối cùng để tránh mất mát thông tin trên đa tạp số chiều thấp). | - Tự cài đặt `DepthwiseSeparableConv` và `InvertedResidual` bằng PyTorch thuần.<br>- Viết hàm tự tính Params/MACs bằng công thức toán và kiểm chứng chéo với thư viện `fvcore` / `thop` trên tensor đầu vào $(1, 32, 112, 112)$. | - Sai lệch công thức tay vs `thop`/`fvcore` bằng `0`.<br>- Minh họa được sự sụt giảm rank/thông tin nếu dùng ReLU ở không gian chiều thấp (2D/3D). |
| **Ngày 3 – 4** | **Kiến trúc Lai (Hybrid Transformers)**<br>- Nút thắt độ phức tạp bậc hai $O(N^2)$ của Spatial Self-Attention trên feature map độ phân giải cao.<br>- Cơ chế của **MobileViT**: dùng Conv ở các tầng nông để trích xuất đặc trưng cục bộ, sau đó dùng khối `MobileViTBlock` (Unfold $\to$ Transformer trên các patch không chồng lấp $\to$ Fold $\to$ Fusion) để học ngữ cảnh toàn cục. | - Tải `mobilevit_s`, `mobilenetv2_100`, `resnet18` từ `timm` / `torchvision`.<br>- Viết script hook theo dõi kích thước tensor (`B, C, H, W`) qua từng stage.<br>- Thống kê tỷ lệ phân bổ Params và MACs giữa các lớp `Conv2d` và các khối `Attention / Linear`. | - Xuất bảng phân rã (breakdown) cấu trúc 3 mạng theo từng stage.<br>- Chỉ ra điểm nghẽn tính toán của `MobileViT-S` nằm ở thao tác Unfold/Fold và MatMul. |
| **Ngày 5 – 6** | **Chỉ số Phần cứng Biên (Edge Metrics)**<br>- Phân biệt **FLOPs/MACs** (độ phức tạp lý thuyết) với **Wall-clock Latency** (độ trễ thực) và **Throughput** (FPS).<br>- **Memory Access Cost (MAC)**, **Arithmetic Intensity** (FLOPs / Byte) và **Roofline Model** (*Compute-bound* vs. *Memory-bound*).<br>- Lý giải nghịch lý: Depthwise Conv có rất ít FLOPs nhưng tỷ lệ đọc/ghi bộ nhớ cao, dễ gây nghẽn băng thông RAM/Cache. | - Viết script benchmark bằng `torch.profiler` và `time.perf_counter()` cho `ResNet-18`, `MobileNetV2`, và `MobileViT-S` với input $(1, 3, 224, 224)$.<br>- Thu thập: Params (M), MACs (G), Peak Memory (MB), và CPU Latency (ms).<br>- Vẽ biểu đồ Scatter Plot: *MACs vs. Actual CPU Latency*. | - Hoàn thiện notebook `01_architecture_benchmark.ipynb`.<br>- Bảng số liệu chứng minh `MobileViT-S` dù ít Params hơn `ResNet-18` nhưng CPU Latency cao hơn đáng kể. |
| **Ngày 7** | **Cột mốc Portfolio 1: Train Baseline trên NEU Defect**<br>- Chiến lược Transfer Learning cho dữ liệu công nghiệp quy mô nhỏ (1.800 ảnh).<br>- Kỹ thuật xử lý dữ liệu: Stratified Train/Val/Test split (70%/15%/15% = 1260/270/270 ảnh), chuẩn hóa ImageNet stats, Data Augmentation bảo toàn hình thái lỗi.<br>- Đánh giá mô hình AOI: Confusion Matrix, Precision, Recall, F1-score. | - **(Bật GPU T4)** Tải bộ dữ liệu **NEU Surface Defect**, chia tập dữ liệu cố định với `seed=42`.<br>- Fine-tune `MobileNetV2` (10–15 epochs, `AdamW`, `CosineAnnealingLR`, ~2 phút).<br>- Xuất Confusion Matrix trên tập Test (270 ảnh).<br>- Lưu `mobilenetv2_neu.pth` và file `metrics_baseline.json` lên Google Drive. | - Test Accuracy đạt **>98.5%**.<br>- Không có lớp lỗi nào có Recall < 96%.<br>- Trọng số `.pth` (~9 MB) được lưu an toàn. |

---

## Giai đoạn 2: ONNX Graph Surgery, Kiểm định & Đóng gói GitHub (Tuần 2)

| Thời gian | Chủ đề Học thuật & Kỹ thuật | Thực hành & Sản phẩm đầu ra (Deliverables) | Tiêu chí Nghiệm thu (DoD) |
| :--- | :--- | :--- | :--- |
| **Ngày 8 – 9** | **Computational Graph & Định dạng ONNX**<br>- Cấu trúc Protobuf của file `.onnx`: `ModelProto`, `GraphProto`, `NodeProto` (operators), `ValueInfoProto` (shapes/types), `TensorProto` (initializers/weights).<br>- Cơ chế `torch.jit.trace` vs `torch.jit.script` khi xuất đồ thị tĩnh.<br>- Tác động của cờ `TrainingMode.PRESERVE` vs `TrainingMode.EVAL` lên cấu trúc node khi export. | - Viết script xuất `mobilenetv2_neu.pth` sang 2 phiên bản (`opset_version=17`, `dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}`):<br>  1. `model_unfolded.onnx` (`TrainingMode.PRESERVE`) để giữ nguyên node `BatchNormalization`.<br>  2. `model_raw.onnx` (`TrainingMode.EVAL`).<br>- Dùng thư viện `onnx` duyệt cây đồ thị, đếm số lượng từng loại toán tử (`Conv`, `BatchNormalization`, `Clip`, `Add`, `Gemm`).<br>- Mở **Netron**, chụp ảnh cụm `Conv -> BatchNormalization -> Clip` lưu thành `assets/graph_before_opt.png`. | - Script in ra bảng thống kê số lượng từng loại Node trước tối ưu.<br>- Có ảnh `graph_before_opt.png` rõ nét hiển thị chuỗi `Conv -> BN -> Clip`. |
| **Ngày 10 – 11** | **Phẫu thuật Đồ thị (Graph Surgery)**<br>- Toán học của **BatchNorm Folding**: Gộp tham số $(\gamma, \beta, \mu, \sigma^2, \epsilon)$ vào trọng số $(W, b)$ của lớp `Conv` liền trước:<br>  $$W_{\text{fold}} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot W, \quad b_{\text{fold}} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot (b - \mu) + \beta$$<br>- **Constant Folding**, Shape Inference và loại bỏ các node dư thừa (`Identity`, `Dropout`, `Pad` tĩnh). | - Tự viết hàm Python/NumPy thực hiện **Manual BN Folding** trên lớp `features.0` (`Conv2d + BatchNorm2d`) của `MobileNetV2` và kiểm chứng sai số so với chạy tách rời là `< 1e-6`.<br>- Dùng `onnxsim` (`onnx-simplifier`) và `onnxoptimizer` để tối ưu đồ thị thành `model_opt.onnx`.<br>- Đếm lại số lượng node sau tối ưu, mở Netron xác nhận toàn bộ node `BatchNormalization` đã được gập vào `Conv`, chụp ảnh `assets/graph_after_opt.png`. | - Hàm Manual BN Folding chạy pass unit test (`max_diff < 1e-6`).<br>- Đồ thị `model_opt.onnx` có số lượng `BatchNormalization` = `0`.<br>- Có ảnh `graph_after_opt.png` đối chiếu trực tiếp với ảnh trước tối ưu. |
| **Ngày 12 – 13** | **Numerical Validation & Production Benchmark**<br>- Nguồn gốc sai số dấu phẩy động (`IEEE 754 FP32`) khi thay đổi thứ tự phép toán và tập lệnh vector hóa giữa PyTorch (ATen/MKL) và ONNX Runtime (MLAS/XNNPACK).<br>- Phương pháp đo hiệu năng chuẩn Production: *Warm-up runs* (ổn định bộ nhớ đệm L1/L2/L3 & cấp phát arena), khóa luồng CPU (`intra_op_num_threads`), phân tích phân phối độ trễ (**Mean, Std, P50, P95, P99 Latency**). | - Viết `validate.py` chạy song song PyTorch và ONNX Runtime trên toàn bộ 270 ảnh tập Test của NEU Defect:<br>  - Đo `max(|Y_torch - Y_onnx|)` và `mean(|Y_torch - Y_onnx|)` trên cả Logits và Softmax.<br>  - Kiểm chứng 100% dự đoán lớp (Top-1) trùng khớp.<br>- Viết `benchmark.py` (50 vòng warm-up + 500 vòng đo thực tế, khóa `intra_op_num_threads=1` và `2`), xuất bảng so sánh PyTorch Eager vs. ONNX Runtime (CPU). | - Độ lệch tuyệt đối trung bình `< 1e-6`, cực đại `< 1e-5` (trên xác suất Softmax), Top-1 Accuracy giữ nguyên 100%.<br>- ONNX Runtime đạt tốc độ nhanh hơn PyTorch CPU từ **1.5x – 2.5x** ở `batch_size=1`. |
| **Ngày 14** | **Cột mốc Portfolio 2: Đóng gói Mã nguồn & Báo cáo Kỹ thuật**<br>- Tư duy thiết kế Modular Codebase chuẩn kỹ sư AI (CLI arguments với `argparse`, tách biệt config, logging rõ ràng, reproducible).<br>- Cách trình bày Technical README thu hút Tech Lead / Hiring Manager (Problem Statement $\to$ Architecture Trade-offs $\to$ Graph Surgery Proof $\to$ Benchmark Results). | - Tách và chuẩn hóa toàn bộ code từ Colab thành 4 module độc lập trong `src/`: `train.py`, `graph_surgery.py`, `validate.py`, `benchmark.py`.<br>- Hoàn thiện 2 Colab Notebooks trong `notebooks/` có thể chạy *Run All* một chạm.<br>- Viết `README.md` chuyên nghiệp kèm ảnh đối chiếu Netron (`graph_before_opt.png` vs `graph_after_opt.png`) và bảng kết quả thực nghiệm đầy đủ. | - Toàn bộ script trong `src/` chạy mượt qua CLI (`python src/benchmark.py --help`).<br>- Repo sạch sẽ, đầy đủ `requirements.txt` và `README.md` sẵn sàng đính kèm vào CV. |

---

## Cấu trúc Repository Bàn giao Cuối khóa

```text
industrial-defect-onnx-optimization/
├── assets/
│   ├── graph_before_opt.png             # Ảnh chụp Netron trước khi gập BatchNorm (Conv -> BN -> Clip)
│   ├── graph_after_opt.png              # Ảnh chụp Netron sau khi tối ưu đồ thị (BN đã được gập vào Conv)
│   └── confusion_matrix_neu.png         # Ma trận nhầm lẫn trên 6 lớp lỗi bề mặt thép NEU
├── notebooks/
│   ├── 01_architecture_benchmark.ipynb  # Notebook Tuần 1: Tự code InvertedResidual, FLOPs vs Latency (ResNet18/MobileNetV2/MobileViT)
│   └── 02_end_to_end_pipeline.ipynb     # Notebook Tuần 2: Train NEU -> Export ONNX -> Graph Surgery -> Validate -> Benchmark
├── plans/
│   └── 14_day_master_plan.md            # Bản kế hoạch tổng thể 14 ngày & tiêu chí nghiệm thu
├── src/
│   ├── dataset.py                       # DataLoader & Stratified Split (70/15/15) cho NEU Surface Defect
│   ├── train.py                         # Fine-tune MobileNetV2 trên NEU Surface Defect & xuất metrics
│   ├── graph_surgery.py                 # Manual BN Folding (NumPy demo), Export ONNX & Simplification
│   ├── validate.py                      # Kiểm định sai số số học (< 1e-5) & Top-1 agreement trên tập Test
│   └── benchmark.py                     # Đo Mean / P50 / P95 / P99 Latency & Throughput (PyTorch vs ONNX Runtime)
├── requirements.txt                     # Danh sách thư viện cố định phiên bản (torch, timm, onnx, onnxruntime, onnxsim, thop)
└── README.md                            # Báo cáo kỹ thuật, lý giải lý thuyết & bảng kết quả thực nghiệm
```

---

## Checklist Bắt đầu Ngay (Actionable Next Steps)

- [ ] **Bước 1:** Khởi tạo cấu trúc thư mục chuẩn (`assets/`, `notebooks/`, `src/`, `plans/`) và file `requirements.txt` ngay trong workspace.
- [ ] **Bước 2:** Bắt đầu **Ngày 1–2** với việc xây dựng module `DepthwiseSeparableConv` & `InvertedResidual` từ con số 0 và kiểm chứng công thức tính MACs/Params trong `notebooks/01_architecture_benchmark.ipynb`.
