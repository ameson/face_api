import multiprocessing

# 工作进程数 - 使用最少的工作进程
workers = 2  # 增加到2个工作进程以分散负载

# 工作模式
worker_class = 'gthread'  # 使用线程模式以better处理I/O密集型操作

# 每个工作进程的线程数
threads = 4

# 超时设置
timeout = 300  # 增加超时时间到5分钟

# 限制最大请求数
max_requests = 50  # 降低单个worker处理的最大请求数
max_requests_jitter = 10

# 日志级别
loglevel = 'info'

# 预加载应用
preload_app = True

# 限制请求大小
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# 工作进程内存限制
worker_memory_limit = 750 * 1024 * 1024  # 增加到750MB per worker

# 优雅重启时间
graceful_timeout = 60  # 增加优雅重启时间

# 保持连接超时
keepalive = 2

# 添加worker回收策略
max_worker_lifetime = 3600  # 1小时后回收worker
worker_reload_mercy = 60  # 给worker 60秒时间完成当前请求
