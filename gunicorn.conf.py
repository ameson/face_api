import multiprocessing

# 工作进程数 - 使用最少的工作进程
workers = 1  # 减少到1个工作进程以节省内存

# 工作模式
worker_class = 'sync'  # 使用同步模式减少内存开销

# 超时设置
timeout = 120

# 限制最大请求数
max_requests = 100
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
worker_memory_limit = 450 * 1024 * 1024  # 450MB per worker

# 优雅重启时间
graceful_timeout = 30

# 保持连接超时
keepalive = 2
