import os

def mkdirs(folder, permission):
    if not os.path.exists(folder):
        try:
            original_umask = os.umask(0)# os.umask(0) ：修改文件模式，让进程有较大权限，保证进程有读写执行权限
            os.makedirs(folder,permission, exist_ok=True)
        finally:
            os.umask(original_umask)