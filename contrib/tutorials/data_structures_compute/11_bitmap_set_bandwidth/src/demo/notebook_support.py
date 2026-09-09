"""CANN CMake discovery shared by the experiment 11 notebook setup."""
from pathlib import Path
import os

def configure_asc_cmake():
    config_names = ('ASCConfig.cmake', 'asc-config.cmake', 'ASCCConfig.cmake', 'ascc-config.cmake')
    roots = []
    checked = []

    def add_root(value):
        if not value:
            return
        path = Path(value).expanduser()
        if path.name in config_names:
            path = path.parent
        if path not in roots:
            roots.append(path)

    for variable in ('ASC_DIR', 'ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME'):
        add_root(os.environ.get(variable))
    add_root('/usr/local/Ascend/ascend-toolkit/latest')
    for base in (Path('/usr/local/Ascend'), Path('/home/developer/Ascend'), Path('/opt/conda/Ascend')):
        if base.is_dir():
            for candidate in sorted(base.glob('cann-*'), reverse=True):
                add_root(candidate)

    suffixes = (
        Path('.'), Path('aarch64-linux/lib64/cmake'), Path('x86_64-linux/lib64/cmake'),
        Path('aarch64-linux/tikcpp/ascendc_kernel_cmake'),
        Path('x86_64-linux/tikcpp/ascendc_kernel_cmake'), Path('lib64/cmake'),
    )
    selected = None
    for root in roots:
        for suffix in suffixes:
            directory = root / suffix
            for name in config_names:
                config = directory / name
                checked.append(str(config))
                if config.is_file():
                    selected = (root, config.resolve())
                    break
            if selected:
                break
        if selected:
            break
    if selected is None:
        raise RuntimeError('未找到 ASC CMake 配置。已检查:\n' + '\n'.join(f'  - {path}' for path in checked))

    matched_root, config = selected
    asc_dir = config.parent
    cann_home = next((parent for parent in config.parents if parent.name.startswith('cann-')), None)
    cann_home = cann_home or matched_root.resolve()
    old_prefixes = [item for item in os.environ.get('CMAKE_PREFIX_PATH', '').split(os.pathsep) if item]
    prefix_entries = list(dict.fromkeys([str(asc_dir), str(cann_home), *old_prefixes]))
    os.environ['ASC_DIR'] = str(asc_dir)
    os.environ['ASCEND_HOME_PATH'] = str(cann_home)
    os.environ['ASCEND_TOOLKIT_HOME'] = str(cann_home)
    os.environ['CMAKE_PREFIX_PATH'] = os.pathsep.join(prefix_entries)
    print('CANN home :', cann_home)
    print('ASC config:', config)
    return [f'-DASC_DIR={asc_dir}', f'-DCMAKE_PREFIX_PATH={";".join(prefix_entries)}']
