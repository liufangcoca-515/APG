import os

def remove_leading_zeros(filename):
    # 分割文件名和扩展名
    name, ext = os.path.splitext(filename)
    
    # 如果文件名以数字开头,去除前导零
    if name[0].isdigit():
        name = str(int(name))
    
    return name + ext

def process_directory(path):
    # 遍历目录
    for root, dirs, files in os.walk(path):
        for filename in files:
            old_path = os.path.join(root, filename)
            new_filename = remove_leading_zeros(filename)
            new_path = os.path.join(root, new_filename)
            
            # 如果文件名有变化则重命名
            if filename != new_filename:
                os.rename(old_path, new_path)

if __name__ == "__main__":
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    process_directory(current_dir)
