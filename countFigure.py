import os

def count_png_files_sorted(folder_path):
    png_counts = {}
    subfolders = os.listdir(folder_path)

    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            png_files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
            png_count = len(png_files)
            png_counts[subfolder] = png_count

    # 按数量从高到低排序
    sorted_png_counts = sorted(png_counts.items(), key=lambda x: x[1], reverse=True)

    for subfolder, count in sorted_png_counts:
        print(f"{subfolder} contains {count} .png files.")

    total_png_count = sum(png_counts.values())
    print(f"Total .png files: {total_png_count}")

# 替换 'Images' 为你的文件夹路径
folder_path = '/mnt/backup/home/xd/whx/NeuDFL/data/gtsrb/Processed_Train/Images'
count_png_files_sorted(folder_path)

# 00002 contains 1500 .png files.
# 00001 contains 1500 .png files.
# 00013 contains 1440 .png files.
# 00012 contains 1410 .png files.
# 00038 contains 1380 .png files.
# 00010 contains 1350 .png files.
# 00004 contains 1320 .png files.
# 00005 contains 1260 .png files.
# 00025 contains 1020 .png files.
# 00009 contains 990 .png files.
# 00007 contains 960 .png files.
# 00008 contains 960 .png files.
# 00003 contains 960 .png files.
# 00011 contains 900 .png files.
# 00018 contains 810 .png files.
# 00035 contains 810 .png files.