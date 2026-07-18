import os
import importlib

# 获取当前 __init__.py 所在包的根目录
current_module_name = __name__
current_module_file = os.path.abspath(__file__)
current_module_dir = os.path.dirname(current_module_file)

# 存放导出模块名
__all__ = []


def _import_submodules(package_dir, package_name):
    for root, dirs, files in os.walk(package_dir):
        # 构建当前目录相对于包根目录的模块路径
        relative_path = os.path.relpath(root, current_module_dir)
        if relative_path == ".":
            module_prefix = package_name
        else:
            subpackage = relative_path.replace(os.sep, ".")
            module_prefix = f"{package_name}.{subpackage}"

        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                module_name = file[:-3]
                full_module_name = f"{module_prefix}.{module_name}"

                imported_module = importlib.import_module(full_module_name)
                globals()[module_name] = imported_module
                if module_name not in __all__:
                    __all__.append(module_name)


# 执行导入
_import_submodules(current_module_dir, current_module_name)
