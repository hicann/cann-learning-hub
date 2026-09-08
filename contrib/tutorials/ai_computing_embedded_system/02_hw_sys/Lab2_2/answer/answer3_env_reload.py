"""
思考题 3 参考答案：升级后为何必须重新打开终端才能使新环境变量生效？

答案说明：
    环境变量脚本（set_env.sh）修改的是 /etc/profile.d/ascend_env.sh 文件。
    /etc/profile.d/ 下的脚本只在终端启动时被 /etc/profile 加载执行一次。

    已经打开的终端不会重新加载 profile 文件，因此：
    - 旧终端中的环境变量仍指向旧版本 CANN 路径
    - 必须重新打开终端，新终端才会重新加载 profile，使新环境变量生效

    也可以在当前终端手动执行：source /etc/profile 来重新加载。
"""

import os

print("思考题 3 答案：")
print("升级后必须重新打开终端的原因：")
print("  1. 环境变量脚本修改的是 /etc/profile.d/ascend_env.sh")
print("  2. profile.d 下的脚本只在终端启动时加载一次")
print("  3. 已打开的终端不会重新加载 profile")
print("  4. 解决方案：重新打开终端，或手动执行 source /etc/profile")
print()
print("当前环境变量示例：")
for key in ['ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME', 'ASCEND_OPP_PATH']:
    print(f"  {key} = {os.environ.get(key, '未设置')}")
