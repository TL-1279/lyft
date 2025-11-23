import zarr

# Đường dẫn tới file .zarr
file_path = "/Users/tl/Desktop/dataset.zarr"

# Mở file zarr
root = zarr.open(file_path, mode='r')

# Hàm đệ quy in cấu trúc zarr
def print_zarr_tree(node, prefix=""):
    if isinstance(node, zarr.core.Array):
        print(f"{prefix} -> array, shape={node.shape}, dtype={node.dtype}")
    elif isinstance(node, zarr.hierarchy.Group):
        for key, item in node.items():
            print_zarr_tree(item, prefix + "/" + key)

# In toàn bộ cấu trúc
print_zarr_tree(root)
