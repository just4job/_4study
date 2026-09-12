# RUNBOOK — từ dữ liệu thô tới kết quả train

Trình tự thực thi đầy đủ, kèm output mong đợi ở từng bước để biết bước đó đã đúng
chưa trước khi đi tiếp. Chi tiết thiết kế nằm ở [README.md](README.md); file này
chỉ là thứ tự làm và cách kiểm tra.

Quy ước: mọi lệnh chạy trong `~/Downloads/_4study/CAFA6` với venv đã kích hoạt
(`source myenv/bin/activate`).

---

## Trạng thái hiện tại

| Artifact | Trạng thái |
|---|---|
| `valid_protein_ids.csv` (20.550) | Xong |
| `proteins_edges/` (20.550 contact map) | Xong |
| `human_{BP,MF,CC}_ACS.json` | Xong |
| `uniprot_ensembl_mapping.csv` | Xong |
| `protein_node2vec` (14.871) | Xong |
| `protein_node2onehot` (20.521) | Xong |
| `dict_sequence_feature` | **Giai đoạn A** |
| `split_*`, `label_vocab_*`, `ppi_graph_*`, `emb_*` | **Giai đoạn B** |
| `divided_data/`, `kaggle_data.zip` | **Giai đoạn B** |
| Train + eval | **Giai đoạn C** |

---

## Giai đoạn A — `dict_sequence_feature` trên Kaggle (cần GPU)

Bước duy nhất cần GPU. Hướng dẫn từng cell: [`kaggle_seq_feature.md`](kaggle_seq_feature.md).

**A1.** Đẩy code mới nhất lên (Kaggle sẽ clone từ đây):

```bash
git pull
```

**A2.** Đóng gói input:

```bash
mkdir -p ~/cafa6_seq_input
cp ~/Downloads/_4study/raw_data/seq.fasta proceed_data/valid_protein_ids.csv ~/cafa6_seq_input/
ls -lh ~/cafa6_seq_input
```

Chờ: `seq.fasta` ~83 MB, `valid_protein_ids.csv` ~143 KB.

Dùng `$HOME` chứ không phải `/tmp` — trình duyệt snap/flatpak có `/tmp` riêng nên
hộp chọn file của nó không thấy thư mục nằm trong `/tmp` thật.

**A3.** Upload `~/cafa6_seq_input` lên [Kaggle Datasets](https://www.kaggle.com/datasets)
→ **New Dataset** → tên `cafa6-seq-input`.

**A4.** Notebook mới → **Settings → Accelerator → GPU T4 x2** → **Add Input** chọn
dataset vừa tạo → chạy các cell trong [`kaggle_seq_feature.md`](kaggle_seq_feature.md).

Chờ: 30–60 phút. Cuối cùng in `✓ Đã encode: ~20.4xx protein`.

**A5.** Tải `dict_sequence_feature` từ tab **Output** về, chép vào `proceed_data/`:

```bash
cp ~/Downloads/dict_sequence_feature proceed_data/
python scripts/check_inputs.py --skip-ppi
```

Chờ: `dict_sequence_feature: 20.4xx protein (shape (1024,))` và dòng phủ ≥ 99%.
**Không đi tiếp nếu dòng này chưa OK** — thiếu nó thì cả nhánh sequence của model
là vector 0 mà không có lỗi nào được in ra.

---

## Giai đoạn B — Build dataset ở máy local

Bốn bước, chạy đúng thứ tự này. Mỗi bước ăn output của bước trước.

### B1. Chia train/valid/test + lọc vocab

```bash
python data_processing/split_protein_ids.py --force
```

Vài giây. Sinh `split_{bp,mf,cc}.json` và `label_vocab_{bp,mf,cc}.json`.

Chờ thấy số nhãn **giảm mạnh** so với số thô trong ACS:

| Nhánh | GO term thô | Sau lọc (ước tính) |
|---|---|---|
| BP | 13.871 | vài trăm (`--min-bp 250`) |
| MF | 5.136 | vài trăm (`--min-other 100`) |
| CC | 1.988 | vài trăm |

Không giảm → bộ lọc không chạy, dừng lại hỏi. Đây chính là bug `MF 5136 label` cũ
trong README mục 3.

Script cũng in `[WARN]` nếu có nhãn không có positive nào ở valid/test — ghi lại
con số đó, nó ảnh hưởng độ tin cậy của F1 theo nhãn.

### B2. Đồ thị PPI

```bash
python data_processing/4_build_ppi_graph.py
```

Vài phút. Sinh `ppi_graph_global`, `ppi_protein_index`, và `ppi_graph_train_{bp,mf,cc}`.

Chờ thấy:

```
Số chiều seq feature đọc từ dict_sequence_feature: 1024
Gắn seq feature cho ~17.9xx / 20.5xx node
bp: giữ N/M cạnh, ẩn ... node valid/test
```

`ppi_graph_train_{ns}` là lá chắn chống rò rỉ PPI: lúc train, GraphSAGE chỉ thấy
cạnh train–train (README mục 4.5). Bước này bị bỏ qua thì `train_Struct2GO2.py`
phải tự mask lúc runtime — chậm và tốn RAM hơn.

### B3. Ghép thành graph dataset

```bash
python data_processing/3_build_graph_dataset.py
```

Nhanh hơn tên gọi gợi ý — khoảng 1 phút cho cả 3 nhánh (contact map đã dựng sẵn
ở `proteins_edges/`, bước này chỉ ghép lại). Sinh cho mỗi nhánh:
`emb_graph_{ns}`, `emb_seq_feature_{ns}`, `emb_label_{ns}`, `emb_ppi_node_id_{ns}`,
`label_{ns}_network`.

Chờ thấy ở đầu log — **đọc kỹ 4 dòng này**:

```
✓ node2vec: 14,871 protein
✓ onehot  : 20,521 protein
✓ SeqVec  : 20,485 protein
✓ PPI index: 19,661 protein
```

Dấu `✗` ở dòng nào nghĩa là feature đó bị thay bằng **vector 0** cho toàn bộ
protein — script vẫn chạy tiếp bình thường, không báo lỗi.

Và:

```
Dùng label_vocab_{ns}.json đã lọc từ split_protein_ids.py: N GO label (trước lọc: M)
Đang xây label network (chỉ từ ... protein train)
```

Thấy `[WARN] Chưa có label_vocab...` hoặc `[WARN] Chưa có split_...` → B1 chưa
chạy, quay lại.

### B4. Chia thành dataset train/valid/test

```bash
python data_processing/divide_data.py --force
```

Nặng RAM — nó nạp trọn `emb_*` của một nhánh vào bộ nhớ rồi mới ghi. Sinh
`divided_data/{ns}_{train,valid,test}_dataset`; kích thước bằng khoảng 3 lần
`emb_*` của nhánh đó (`du -sh proceed_data` để ước lượng trước).

Chờ thấy `Dùng split_{ns}.json (seed=42)` — nếu thấy `[WARN] Chưa có split_{ns}.json
— random split tại chỗ` thì split đang bị chia lại, **sai**, dừng và quay về B1.

Máy hết RAM thì chạy từng nhánh:

```bash
python data_processing/divide_data.py --force --namespace mf
python data_processing/divide_data.py --force --namespace cc
python data_processing/divide_data.py --force --namespace bp
```

### B5. Kiểm tra toàn bộ trước khi đóng gói

```bash
python scripts/audit_data.py --deep
```

Bảy nhóm kiểm tra: artifact đủ chưa, split có rời nhau không, vocab lọc đúng chưa,
`ppi_graph_train_{ns}` thật sự không còn cạnh chạm valid/test, chiều nhãn khớp giữa
vocab/network/dataset, nhãn nào không có positive ở valid/test, và — chỉ với
`--deep` — `divided_data` có khớp `split_{ns}.json` và `ppi_node_id` có hợp lệ không.

**Có `FAIL` thì sửa trước khi pack** — phát hiện ở đây rẻ hơn nhiều so với sau khi
đã upload vài GB lên Kaggle. Kiểm tra #4 là quan trọng nhất: nó chứng minh lá chắn
chống rò rỉ PPI hoạt động thật.

### B6. Đóng gói

```bash
python pack_for_kaggle.py
```

Sinh `kaggle_data.zip`. Muốn tách nhánh cho nhẹ: `python pack_for_kaggle.py --branch mf`.

---

## Giai đoạn C — Train trên Kaggle

Hướng dẫn từng cell: [`kaggle_notebook.md`](kaggle_notebook.md).

**C1.** Upload `kaggle_data.zip` lên Kaggle Datasets → tên `cafa6-data`.

**C2.** Notebook mới, bật GPU, gắn dataset, chạy theo `kaggle_notebook.md`:
clone repo → cài dgl → `scripts/kaggle_link_data.py` → `scripts/audit_data.py` →
train + eval cả 3 nhánh.

**C3.** Đọc log theo mục 7 của `kaggle_notebook.md` — 4 dòng cần kiểm tra, trong đó
có `macro_f1` và F1 theo nhóm nhãn hiếm/trung bình/phổ biến.

**C4.** Ablation (tuỳ chọn): `scripts/run_fusion_ablation.py` so 4 kiểu fusion
(`ppi_concat`, `no_ppi_attn`, `ppi_attn`, `ppi_bi_attn`), và `--loss` so
`bce` / `bce_pos_weight` / `focal`.

---

## Khi có sự cố

| Triệu chứng | Nguyên nhân thường gặp |
|---|---|
| `[WARN] Chưa có split_{ns}.json` | Quên B1, hoặc chạy B1 không có `--force` nên nó skip |
| Số label MF vẫn 5.136 | `label_vocab_mf.json` bị ghi đè bởi bản không lọc — chạy lại B1 rồi B3 |
| `✗ Không tìm thấy ...` ở B3 | Feature đó thành vector 0. Dừng lại, đừng train |
| F-max ~0.002 | Trộn data cũ với mới — chiều nhãn lệch. `audit_data.py` kiểm tra #5 bắt được |
| Máy đơ / VS Code tắt | Hết RAM. Chạy từng nhánh, hoặc bọc `systemd-run --user --scope -p MemoryMax=8G` |
| `import dgl` lỗi | Xem mục "Hướng dẫn cài" trong `python scripts/check_env.py` |

Ba script kiểm tra, dùng ở 3 thời điểm khác nhau:

```bash
python scripts/check_env.py       # thư viện + GPU + đĩa       (trước khi chạy gì)
python scripts/check_inputs.py    # nội dung file đầu vào       (trước giai đoạn B)
python scripts/audit_data.py      # artifact cuối pipeline      (trước khi pack)
```
