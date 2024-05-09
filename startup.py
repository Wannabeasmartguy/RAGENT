import subprocess
import time
import os

# 确保在当前目录下运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))


try:
    # 启动test_server.py作为子进程
    # 这里放 server 的代码
    server_process = subprocess.Popen(
        [
            "uvicorn", "server:app", 
            "--host", "127.0.0.1",
            "--port", "8000"
        ]
    )

    # 等待服务器启动
    time.sleep(1)

    # 启动test_front.py作为子进程
    # 这里放前端的代码，在 RAGenT 里是 Streamlit 的代码
    front_process = subprocess.Popen(
        [
            "streamlit", "run", "RAGenT.py",
            "--server.address", "127.0.0.1",
            "--server.port", "5998"
        ]
    )

    # 等待前端启动
    time.sleep(1)


    # 此处可以添加代码以执行其他任务，例如打开浏览器等

    # 等待用户中断程序或其他退出条件
    print("Server and front-end are running. Press Ctrl+C to stop.")

    try:
        # 使用wait()方法等待进程结束，这里会阻塞直到手动中断
        server_process.wait()
        front_process.wait()
    except KeyboardInterrupt:
        # 捕捉到中断信号，退出等待
        pass

finally:
    # 无论发生什么，都会执行finally块中的代码，确保进程被终止
    print("Terminating the server and the front-end...")
    if server_process.poll() is None: # 如果server_process仍在运行
        server_process.terminate()  # 终止server_process
    if front_process.poll() is None:  # 如果front_process仍在运行
        front_process.terminate()  # 终止front_process
  
    # 等待进程确实被终止
    server_process.wait()
    front_process.wait()
    print("Server and front-end have been terminated.")