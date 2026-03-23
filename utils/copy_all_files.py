import os
import shutil



def copy_files_exclude(src_dir, dst_dir):
    """
    复制src_dir下的所有文件到dst_dir，保持目录结构，
    但排除名为'data'的目录和以.out结尾的文件。
    """
    
    for root, dirs, files in os.walk(src_dir, topdown=True):
        """
        root 是当前正在遍历的目录
        dirs 是当前正在遍历目录下的文件夹， 之后for循环会不断遍历该dirs里面的东西直到其为空
        files 是当前正在便利的目录下的文件名称
        """
        # 过滤掉不需要复制的目录
        dirs[:] = [d for d in dirs if d != 'data' and d != 'test_out' and d !='checkpoints'and d !='tmp'and d !='watch'and d !='env' and d !='backup' and d!='ckpt']
        # 获得相对路径
        relative_folder_path = os.path.relpath(root,src_dir)
        # 目标目录路径
        dst_folder_path = os.path.join(dst_dir,relative_folder_path)
        
        if not os.path.exists(dst_folder_path):
            os.makedirs(dst_folder_path)
            
        for file in files:
            if not file.endswith('.out'):
                # 复制文件
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(dst_folder_path, file)
                shutil.copy2(src_file_path, dst_file_path)

