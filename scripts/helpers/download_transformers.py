from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MCG-NJU/videomae-base",
    local_dir="../../models/transformer/VideoMAE",
)