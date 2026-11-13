# SDP-datasets

Repository gồm bảy tệp ZIP dữ liệu dự đoán lỗi phần mềm và mã nguồn **DGCL-DP**
trong thư mục [`dgcl-dp/`](dgcl-dp).

**Cách chạy trên Google Colab, Kaggle và Windows:**
xem [hướng dẫn tiếng Việt](dgcl-dp/README.md).

Chạy nhanh sau khi clone repository và cài `uv`:

```bash
cd dgcl-dp
uv python install 3.11.11
uv sync --frozen
uv run python -m dgcl_dp.reproduce experiment=demo
```

Demo dùng dữ liệu giả, không cần giải nén các ZIP. Kết quả nằm trong
`dgcl-dp/artifacts/demo/run/`. Các ZIP được giữ nguyên và cần kiểm chứng
nguồn/phiên bản trước khi dùng cho nghiên cứu chính.
